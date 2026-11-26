#!/usr/bin/env python3
"""试验评测（WSL，.venv-synth）：对 work/pilot/out/<engine>/<voice>/*.wav 回听打分。

指标（每个 engine×voice 汇总，越低越好的标 ↓）：
  cer↓        Whisper large-v3-turbo 识别后与原文的字错误率（去标点）
  bad_pause↓  每 100 字里、不在标点处的 ≥180ms 停顿次数（“断句奇怪”的主要来源）
  miss_pause↓ 逗号/句号等标点处没有停顿（<60ms）的比例（“一口气念到底”）
  cps         语速：每秒汉字数（参考：有声书 4–5）
输出 work/pilot/eval.tsv（逐段）与 work/pilot/summary.tsv。

  .venv-synth/bin/python pilot/eval_pilot.py /mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work/pilot
"""
from __future__ import annotations

import difflib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

PUNCT = set("，。！？；：、,.!?;:…—“”\"'（）()《》")
HARD = set("，。！？；：,.!?;:")


def norm(s: str) -> str:
    return "".join(c for c in s if c not in PUNCT and not c.isspace()).lower()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    sm = difflib.SequenceMatcher(None, r, h, autojunk=False)
    edits = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return edits / max(1, len(r))


def main() -> int:
    root = Path(sys.argv[1])
    from faster_whisper import WhisperModel
    asr = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")

    texts = {}
    for p in (Path(__file__).resolve().parent / "texts.tsv", root / "texts_book.tsv"):
        if p.exists():
            texts.update({l.split("\t")[0]: l.split("\t")[2] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()})
    rows, agg = [], defaultdict(lambda: defaultdict(list))

    for wav in sorted((root / "out").glob("*/*/*.wav")):
        engine, voice, pid = wav.parent.parent.name, wav.parent.name, wav.stem
        ref = texts.get(pid)
        if ref is None:
            continue
        from faster_whisper import decode_audio
        audio = decode_audio(str(wav), sampling_rate=16000)  # PyAV 解码+重采样+转单声道
        segs, _ = asr.transcribe(audio, language="zh", word_timestamps=True, beam_size=5,
                                 initial_prompt="以下是普通话的句子。", vad_filter=False,
                                 condition_on_previous_text=False)
        words = [w for s in segs for w in (s.words or [])]
        hyp = "".join(w.word for w in words)

        # 把识别出的每个字对齐到原文的字位置，再判断每个停顿是否落在标点处
        ref_chars = [(i, c) for i, c in enumerate(ref) if c not in PUNCT and not c.isspace()]
        hyp_chars, char_end, char_start = [], [], []
        for w in words:
            cs = [c for c in w.word if c not in PUNCT and not c.isspace()]
            for k, c in enumerate(cs):
                hyp_chars.append(c.lower())
                char_start.append(w.start + (w.end - w.start) * k / len(cs))
                char_end.append(w.start + (w.end - w.start) * (k + 1) / len(cs))
        sm = difflib.SequenceMatcher(None, [c.lower() for _, c in ref_chars], hyp_chars, autojunk=False)
        h2r = {}
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                h2r[b + k] = ref_chars[a + k][0]

        def punct_after(ref_pos: int) -> bool:
            j = ref_pos + 1
            while j < len(ref) and (ref[j] in PUNCT or ref[j].isspace()):
                if ref[j] in HARD or ref[j] in "…—":
                    return True
                j += 1
            return False

        bad, gaps_at = 0, {}
        for k in range(len(hyp_chars) - 1):
            gap = char_start[k + 1] - char_end[k]
            rp = h2r.get(k)
            if rp is None:
                continue
            if punct_after(rp):
                gaps_at[rp] = gap
            elif gap >= 0.18:
                bad += 1
        punct_slots = [i for _, (i, _c) in enumerate(ref_chars[:-1]) if punct_after(i)]
        seen = [p for p in punct_slots if p in gaps_at]
        miss = sum(1 for p in seen if gaps_at[p] < 0.06) / max(1, len(seen))
        dur = len(audio) / 16000
        nchar = len(norm(ref))
        r = dict(engine=engine, voice=voice, pid=pid, cer=cer(ref, hyp), bad_pause=bad * 100 / max(1, nchar),
                 miss_pause=miss, cps=nchar / max(0.1, dur), dur=dur, hyp=hyp)
        rows.append(r)
        for k in ("cer", "bad_pause", "miss_pause", "cps"):
            agg[(engine, voice)][k].append(r[k])
        print(f"{engine:9s} {voice} {pid} cer={r['cer']:.3f} bad={r['bad_pause']:.2f} miss={miss:.2f} cps={r['cps']:.2f}", flush=True)

    with (root / "eval.tsv").open("w", encoding="utf-8") as f:
        f.write("engine\tvoice\tpid\tcer\tbad_pause\tmiss_pause\tcps\tdur\thyp\n")
        for r in rows:
            f.write(f"{r['engine']}\t{r['voice']}\t{r['pid']}\t{r['cer']:.4f}\t{r['bad_pause']:.3f}\t"
                    f"{r['miss_pause']:.3f}\t{r['cps']:.3f}\t{r['dur']:.2f}\t{r['hyp']}\n")
    with (root / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("engine\tvoice\tn\tcer\tbad_pause\tmiss_pause\tcps\n")
        print("\n== 汇总（均值）")
        for (e, v), d in sorted(agg.items()):
            line = (f"{e}\t{v}\t{len(d['cer'])}\t{np.mean(d['cer']):.4f}\t{np.mean(d['bad_pause']):.3f}\t"
                    f"{np.mean(d['miss_pause']):.3f}\t{np.mean(d['cps']):.3f}")
            f.write(line + "\n")
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
