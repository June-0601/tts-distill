#!/usr/bin/env python3
"""对 synth_en_zh.py 新合成的中文/混排音频做 Whisper 回听质检，并与原纯英文回放音频合并写出最终 metadata.csv。

输出（分别为 work/en_us_zh_mix/ 与 work/en_uk_zh_mix/）：
  metadata.csv          合训清单：voice/uid.wav|voice|text|phoneme_ids（0..231 统一 ID）
  qa_<voice>.tsv        中文/混排音频 Whisper 回听详情
  wavs/<voice>/en_*.wav 符号链接复用原纯英文音频（零磁盘复制）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from qa import DEFAULT_MODEL, HERE, en_units, judge, levenshtein, mix_units, zh_units
from synth_en_zh import CORPUS_EN_ZH, EN_A_CONTRAST_SENTENCES, EN_ZH_GROUPS
from synth_multi import stratified_subset


def judge_zh_or_mix(kind: str, ref: list[str], hyp: list[str], sec: float) -> tuple[bool, float, str]:
    """针对纯中文与中英双向混排句子分别放宽合理的每单位时长边界。"""
    if not ref:
        return False, 1.0, "参考文本为空"
    err = levenshtein(ref, hyp)
    rate = err / len(ref)
    if kind in ("word", "gloss", "short") and len(ref) <= 4:
        if sec < 0.15:
            return False, rate, "时长过短"
        if sec > 0.8 + 1.0 * len(ref):
            return False, rate, "时长过长"
        if err > (0 if len(ref) == 1 else 1):
            return False, rate, "识别不一致"
        return True, rate, ""

    per = sec / len(ref)
    hi = 1.15 if kind in ("en_mix", "mix") else 0.65
    if per < 0.08 or per > hi:
        return False, rate, f"语速异常（{per:.2f} 秒/单位）"

    max_err = max(2, round(0.15 * len(ref))) if kind in ("en_mix", "mix") else max(1, round(0.08 * len(ref)))
    if err > max_err:
        return False, rate, "识别差异过多"
    return True, rate, ""


def units_for_en_zh(kind: str, text: str) -> list[str]:
    if kind in ("en_mix", "mix"):
        return mix_units(text)
    return zh_units(text)


def load_en_replay_rows(en_src_dir: Path, voice: str, target_n: int, seed: int) -> list[tuple[str, str, str]]:
    """从已完成质检的原纯英文 metadata_<voice>.csv 分层抽取保真回放集。"""
    en_rows_kind = {}
    for line in (HERE / "corpus" / "en.tsv").read_text(encoding="utf-8").splitlines():
        if line:
            c = line.split("\t")
            en_rows_kind[c[0]] = c[1]

    single_meta = en_src_dir / f"metadata_{voice}.csv"
    if not single_meta.exists():
        raise FileNotFoundError(f"缺少原英文质检清单：{single_meta}")

    pool: list[tuple[str, str, str]] = []
    for line in single_meta.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        wav_name, text = line.split("|", 1)
        uid = wav_name[:-4] if wav_name.endswith(".wav") else wav_name
        pool.append((uid, en_rows_kind.get(uid, "sent"), text))

    return stratified_subset(pool, min(target_n, len(pool)), seed=seed)


def link_wav(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    os.symlink(src.resolve(), dst)


def batch_g2p_via_piper(items: list[tuple[str, str]], espeak_voice: str) -> dict[str, tuple[list[int], list[str]]]:
    """调用 .venv-piper 中的 en_zh_g2p 批量生成精确的 0..231 音素 ID。"""
    piper_py = HERE / ".venv-piper" / "bin" / "python"
    if not piper_py.exists():
        piper_py = Path(sys.executable)

    with tempfile.TemporaryDirectory() as tmp:
        in_json = Path(tmp) / "in.json"
        out_json = Path(tmp) / "out.json"
        in_json.write_text(json.dumps({"voice": espeak_voice, "items": items}, ensure_ascii=False), encoding="utf-8")
        script = (
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from en_zh_g2p import texts_to_en_zh_ids, load_en_zh_phoneme_map; "
            "cfg = json.load(open(sys.argv[2], encoding='utf-8')); "
            "id_map = load_en_zh_phoneme_map(); "
            "res = texts_to_en_zh_ids([(x[0], x[1]) for x in cfg['items']], cfg['voice'], id_map); "
            "json.dump(res, open(sys.argv[3], 'w', encoding='utf-8'))"
        )
        subprocess.run([str(piper_py), "-c", script, str(HERE), str(in_json), str(out_json)], check=True)
        return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["us", "uk", "all"], default="all", help="处理口音组（默认 all）")
    ap.add_argument("--model", default=DEFAULT_MODEL["zh"], help="faster-whisper 模型（默认 large-v3-turbo）")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--work", type=Path, default=HERE / "work")
    args = ap.parse_args()

    import soundfile as sf
    from faster_whisper import WhisperModel

    rows = {}
    for line in CORPUS_EN_ZH.read_text(encoding="utf-8").splitlines():
        if line:
            cols = line.split("\t")
            rows[cols[0]] = cols
    for uid, txt in EN_A_CONTRAST_SENTENCES:
        rows[uid] = [uid, "a_contrast", txt]

    groups = ["us", "uk"] if args.group == "all" else [args.group]
    whisper_model = None

    for g_idx, group_name in enumerate(groups):
        cfg = EN_ZH_GROUPS[group_name]
        base_dir = args.work / cfg["work_name"]
        en_src_dir = args.work / cfg["en_source"]
        espeak_voice = "en-gb-x-rp" if cfg["british"] else "en-us"
        print(f"\n================ 质检与合并 {group_name} ({cfg['work_name']}) ================", flush=True)

        # (voice, wav_rel, text, is_zh_or_mix)
        per_voice_records: dict[str, list[tuple[str, str, str, bool]]] = {}

        for v_idx, (voice, _zh_target, en_target) in enumerate(cfg["voices"]):
            wav_dir = base_dir / "wavs" / voice
            wav_dir.mkdir(parents=True, exist_ok=True)
            qa_path = base_dir / f"qa_{voice}.tsv"
            cache = {}
            if qa_path.exists():
                for line in qa_path.read_text(encoding="utf-8").splitlines():
                    if line:
                        uid, hyp = line.split("\t")[:2]
                        cache[uid] = hyp

            todo = [uid for uid in rows if (wav_dir / f"{uid}.wav").exists()]
            need = [uid for uid in todo if uid not in cache]
            print(f"\n[{voice}] 中文/混排音频 {len(todo)} 条，待 Whisper 识别 {len(need)} 条", flush=True)

            if need:
                if whisper_model is None:
                    whisper_model = WhisperModel(args.model, device=args.device, compute_type="auto")
                for n, uid in enumerate(need, 1):
                    w_lang = "en" if rows[uid][1] == "a_contrast" else "zh"
                    segments, _ = whisper_model.transcribe(
                        str(wav_dir / f"{uid}.wav"),
                        language=w_lang,
                        beam_size=5,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                    )
                    cache[uid] = " ".join(s.text.strip() for s in segments).replace("\t", " ")
                    if n % 200 == 0 or n == len(need):
                        print(f"  [{voice}] 已识别 {n}/{len(need)}", flush=True)

            zh_kept, acont_kept, report, reasons = [], [], [], {}
            for uid in todo:
                _, kind, text = rows[uid]
                info = sf.info(str(wav_dir / f"{uid}.wav"))
                if kind == "a_contrast":
                    ref = en_units(text)
                    hyp = en_units(cache[uid])
                    ok, rate, why = judge("en", "sent", ref, hyp, info.duration)
                else:
                    ref = units_for_en_zh(kind, text)
                    hyp = units_for_en_zh(kind, cache[uid])
                    ok, rate, why = judge_zh_or_mix(kind, ref, hyp, info.duration)
                report.append(f"{uid}\t{cache[uid]}\t{rate:.3f}\t{'保留' if ok else why}")
                if ok:
                    if kind == "a_contrast":
                        acont_kept.append((f"{voice}_{uid}", f"{voice}/{uid}.wav", text, False))
                    else:
                        zh_kept.append((f"{voice}_{uid}", f"{voice}/{uid}.wav", text, True))
                else:
                    reasons[why] = reasons.get(why, 0) + 1

            qa_path.write_text("\n".join(report) + "\n", encoding="utf-8")
            print(f"[{voice}] 新合成保留：中文/混排 {len(zh_kept)} 条 + A/a 对比句 {len(acont_kept)} 条 / 共 {len(todo)} 条", flush=True)
            for why, n in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  丢弃 {n} 条：{why}", flush=True)

            # 复用原有纯英文 QA 音频
            en_replay = load_en_replay_rows(en_src_dir, voice, en_target, seed=args.seed + g_idx * 100 + v_idx)
            en_kept = []
            for uid, _kind, text in en_replay:
                src_wav = en_src_dir / "wavs" / voice / f"{uid}.wav"
                dst_wav = wav_dir / f"{uid}.wav"
                if src_wav.exists():
                    link_wav(src_wav, dst_wav)
                    en_kept.append((f"{voice}_{uid}", f"{voice}/{uid}.wav", text, False))

            print(f"[{voice}] 英文回放挂接 {len(en_kept)} 条（合计 {len(en_kept) + len(acont_kept) + len(zh_kept)} 条）", flush=True)
            per_voice_records[voice] = en_kept + acont_kept + zh_kept

        # 释放 Whisper 显存再批量算 G2P
        all_items: list[tuple[str, str, str, str, bool]] = []
        for voice, _zh, _en in cfg["voices"]:
            for key, wav_rel, text, is_zh in per_voice_records.get(voice, []):
                all_items.append((key, voice, wav_rel, text, is_zh))

        g2p_res = batch_g2p_via_piper([(k, txt) for k, _, _, txt, _ in all_items], espeak_voice)

        anchors: list[str] = []
        rest: list[str] = []
        seen_speakers: set[str] = set()
        counts_en: dict[str, int] = {}
        counts_zh: dict[str, int] = {}
        dropped_g2p = 0

        for key, voice, wav_rel, text, is_zh in all_items:
            ids, bad = g2p_res.get(key, ([], ["missing"]))
            if bad or not ids:
                dropped_g2p += 1
                continue
            line = f"{wav_rel}|{voice}|{text}|{' '.join(str(x) for x in ids)}"
            if voice not in seen_speakers and not is_zh:
                anchors.append(line)
                seen_speakers.add(voice)
            else:
                rest.append(line)
            if is_zh:
                counts_zh[voice] = counts_zh.get(voice, 0) + 1
            else:
                counts_en[voice] = counts_en.get(voice, 0) + 1

        # 保持首行严格按原始 speaker_id=0..N-1 顺序对齐，其余行稳定打乱以便验证集均匀覆盖中英文与各音色
        rng = random.Random(args.seed + g_idx)
        rng.shuffle(rest)
        final_lines = anchors + rest

        meta_path = base_dir / "metadata.csv"
        meta_path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
        total = len(final_lines)
        total_zh = sum(counts_zh.values())
        print(
            f"\n=== [{cfg['work_name']}] 合并完成：共 {total} 条（英文 {total - total_zh} / 中文混排 {total_zh}，"
            f"G2P 丢弃 {dropped_g2p} 条）→ {meta_path} ===",
            flush=True,
        )
        for sid, (voice, _z, _e) in enumerate(cfg["voices"]):
            n_en = counts_en.get(voice, 0)
            n_zh = counts_zh.get(voice, 0)
            print(f"  speaker_id={sid}: {voice:12s} EN={n_en:4d}  ZH/MIX={n_zh:4d}  合计={n_en + n_zh:4d}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
