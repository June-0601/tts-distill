#!/usr/bin/env python3
"""v2 质检 + 训练清单（WSL，.venv-synth）：
  1. Whisper large-v3-turbo 回听 work/zh_v2/wavs/<voice>/*.wav，沿用 qa.py 的 judge（错字/语速/时长）；
  2. 读 spk_filter_v2.py 写的 spk_<voice>.tsv：低于 max(0.70, 该音色中位数 - 0.06) 的判为音色跑偏；
  3. 保留的条目用 pinyin_ids.mjs --word-boundary（v2 前端，带中文词边界 #）算音素 ID，写
       work/zh_v2/metadata.csv      voice/uid.wav|voice|text|ids（8 音色合训）
       work/zh_v2/qa_<voice>.tsv    uid  识别结果  错误率  去留原因（缓存，重跑只识别新音频）
     并生成 corpus/zh_phonemes_v2.json（= zh_phonemes.json + "#":[116]）。

  cd /root/tts-distill && .venv-synth/bin/python qa_v2.py --work /mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qa import DEFAULT_MODEL, HERE, judge, units_for

# 说话人编号沿用旧 zh_multi 模型，网页端的音色选择不用改
SPEAKERS = ["zf_032", "zf_046", "zf_021", "zf_022", "zf_023", "zf_024", "zf_059", "zf_079"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    v2 = args.work / "zh_v2"

    from faster_whisper import WhisperModel, decode_audio
    model = WhisperModel(DEFAULT_MODEL["zh"], device=args.device, compute_type="float16")

    kept_rows = []
    for voice in SPEAKERS:
        wav_dir = v2 / "wavs" / voice
        plan = {u: (k, t) for u, k, t in
                (l.split("\t") for l in (v2 / f"plan_{voice}.tsv").read_text(encoding="utf-8").splitlines() if l)}
        if not wav_dir.is_dir():
            print(f"{voice}: 还没有音频，跳过", flush=True)
            continue
        spk = {}
        sp = v2 / f"spk_{voice}.tsv"
        if sp.exists():
            spk = {u: float(s) for u, s in (l.split("\t") for l in sp.read_text(encoding="utf-8").splitlines() if l)}
            spk = {u: s for u, s in spk.items() if np.isfinite(s)}  # 空音频算不出声纹，交给时长检查
        spk_thr = max(0.70, float(np.median(list(spk.values()))) - 0.06) if spk else 0.0

        cache_p = v2 / f"qa_{voice}.tsv"
        cache = {}
        if cache_p.exists():
            for l in cache_p.read_text(encoding="utf-8").splitlines():
                c = l.split("\t")
                if len(c) >= 2:
                    cache[c[0]] = c[1]
        stats = {"keep": 0}
        with cache_p.open("a", encoding="utf-8") as cf:
            for wav in sorted(wav_dir.glob("*.wav")):
                uid = wav.stem
                if uid not in plan:
                    continue
                kind, text = plan[uid]
                audio = decode_audio(str(wav), sampling_rate=16000)
                sec = len(audio) / 16000
                if uid not in cache:
                    is_en = kind in ("en_word", "en_sent")
                    segs, _ = model.transcribe(audio, language="en" if is_en else "zh", beam_size=5,
                                               initial_prompt=None if is_en else "以下是普通话的句子。",
                                               condition_on_previous_text=False, vad_filter=False)
                    cache[uid] = "".join(s.text for s in segs).strip().replace("\t", " ")
                    cf.write(f"{uid}\t{cache[uid]}\n")
                    cf.flush()
                hyp = cache[uid]
                ok, rate, why = judge("zh", kind, units_for("zh", kind, text), units_for("zh", kind, hyp), sec)
                # 短于 2.5 秒的音频声纹不可靠（单词、短释义）；英文句子与中文原声质心天然有距离（中位 0.4–0.6），
                # 这两类只靠识别与时长把关，否则会把英文能力整批筛掉
                if ok and spk and sec >= 2.5 and kind not in ("en_word", "en_sent") and spk.get(uid, 1.0) < spk_thr:
                    ok, why = False, f"音色跑偏（{spk[uid]:.2f} < {spk_thr:.2f}）"
                if ok:
                    stats["keep"] += 1
                    kept_rows.append((voice, uid, kind, text))
                else:
                    stats[why.split("（")[0]] = stats.get(why.split("（")[0], 0) + 1
        total = sum(stats.values())
        print(f"{voice}: {total} 条，保留 {stats['keep']}（{stats['keep'] / max(1, total):.0%}）；"
              + "，".join(f"{k} {n}" for k, n in stats.items() if k != "keep")
              + (f"；音色阈值 {spk_thr:.2f}" if spk else "；（还没有音色相似度）"), flush=True)

    # 带词边界的音素 ID（与网页 v2 前端同一份逻辑）
    tsv = v2 / "kept.tsv"
    tsv.write_text("".join(f"{v}/{u}\t{k}\t{t}\n" for v, u, k, t in kept_rows), encoding="utf-8")
    jl = v2 / "kept_ids.jsonl"
    subprocess.run(["node", str(HERE / "pinyin_ids.mjs"), "--word-boundary", str(tsv), str(jl)], check=True)
    ids = {}
    for l in jl.read_text(encoding="utf-8").splitlines():
        o = json.loads(l)
        if not o["bad"]:
            ids[o["id"]] = " ".join(map(str, o["ids"]))
    n = 0
    with (v2 / "metadata.csv").open("w", encoding="utf-8") as f:
        for v, u, _k, t in kept_rows:
            key = f"{v}/{u}"
            if key in ids:
                f.write(f"{key}.wav|{v}|{t.replace('|', ' ')}|{ids[key]}\n")
                n += 1
    pm = json.loads((HERE / "corpus" / "zh_phonemes.json").read_text(encoding="utf-8"))
    pm["#"] = [116]
    (HERE / "corpus" / "zh_phonemes_v2.json").write_text(json.dumps(pm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"metadata.csv：{n} 条（注音失败丢掉 {len(kept_rows) - n} 条）→ {v2 / 'metadata.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
