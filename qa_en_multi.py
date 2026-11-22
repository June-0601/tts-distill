#!/usr/bin/env python3
"""对 synth_en_multi.py 合成的多音色英文音频做 Whisper 回听质检，生成多音色与单音色 metadata.csv。

输出（分别为 work/en_us_multi/ 与 work/en_uk_multi/）：
  metadata.csv           多说话人合训清单：voice/uid.wav|voice|text
  metadata_<voice>.csv   单音色独立训练清单：uid.wav|text
  qa_<voice>.tsv         每个音色的识别缓存与去留原因
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qa import DEFAULT_MODEL, HERE, judge, units_for
from qa_multi import parse_caps
from synth_en_multi import EN_GROUPS, stratified_subset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["us", "uk", "all"], default="all", help="质检口音组（默认 all）")
    ap.add_argument("--voices", help="只质检指定音色（逗号分隔；仅更新单音色文件，不覆盖组 metadata.csv）")
    ap.add_argument("--cap", help="合训 metadata.csv 中各音色按 kind 分层抽样上限（如 af_heart=2500,af_bella=1800）")
    ap.add_argument("--seed", type=int, default=20261113, help="合训分层抽样随机种子")
    ap.add_argument("--model", default=DEFAULT_MODEL["en"], help="faster-whisper 模型名（默认 small.en）")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    args = ap.parse_args()

    caps = parse_caps(args.cap)

    import soundfile as sf
    from faster_whisper import WhisperModel

    rows = {}
    for line in (HERE / "corpus" / "en.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        rows[cols[0]] = cols

    groups = ["us", "uk"] if args.group == "all" else [args.group]
    wanted = [v.strip() for v in args.voices.split(",") if v.strip()] if args.voices else None
    model = None

    for g_idx, group_name in enumerate(groups):
        cfg = EN_GROUPS[group_name]
        base_dir = args.work / cfg["work_name"]
        group_voices = [v for v, _ in cfg["voices"]]
        voices = [v for v in wanted if v in group_voices or ((group_name == "us" and v.startswith("a")) or (group_name == "uk" and v.startswith("b")))] if wanted else group_voices
        if not voices:
            continue

        print(f"\n================ 质检组 {group_name} ({cfg['work_name']}) ================", flush=True)
        all_multi_kept: list[str] = []
        multi_counts: dict[str, int] = {}

        for idx, voice in enumerate(voices):
            wav_dir = base_dir / "wavs" / voice
            if not wav_dir.exists():
                print(f"跳过 {voice}：{wav_dir} 不存在", flush=True)
                continue

            qa_path = base_dir / f"qa_{voice}.tsv"
            cache = {}
            if qa_path.exists():
                for line in qa_path.read_text(encoding="utf-8").splitlines():
                    if line:
                        uid, hyp = line.split("\t")[:2]
                        cache[uid] = hyp

            todo = [uid for uid in rows if (wav_dir / f"{uid}.wav").exists()]
            need = [uid for uid in todo if uid not in cache]
            print(f"\n[{voice}] 有音频 {len(todo)} 条，待 Whisper 识别 {len(need)} 条", flush=True)

            if need:
                if model is None:
                    model = WhisperModel(args.model, device=args.device, compute_type="auto")
                for n, uid in enumerate(need, 1):
                    segments, _ = model.transcribe(
                        str(wav_dir / f"{uid}.wav"),
                        language="en",
                        beam_size=5,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                    )
                    cache[uid] = " ".join(s.text.strip() for s in segments).replace("\t", " ")
                    if n % 200 == 0 or n == len(need):
                        print(f"  [{voice}] 已识别 {n}/{len(need)}", flush=True)

            single_kept, kept_rows, report, reasons = [], [], [], {}
            for uid in todo:
                cols = rows[uid]
                kind, text = cols[1], cols[2]
                info = sf.info(str(wav_dir / f"{uid}.wav"))
                ref = units_for("en", kind, text)
                hyp = units_for("en", kind, cache[uid])
                ok, rate, why = judge("en", kind, ref, hyp, info.duration)
                report.append(f"{uid}\t{cache[uid]}\t{rate:.3f}\t{'保留' if ok else why}")
                if ok:
                    single_kept.append(f"{uid}.wav|{text}")
                    kept_rows.append((uid, kind, text))
                else:
                    reasons[why] = reasons.get(why, 0) + 1

            qa_path.write_text("\n".join(report) + "\n", encoding="utf-8")
            single_meta = base_dir / f"metadata_{voice}.csv"
            single_meta.write_text("\n".join(single_kept) + "\n", encoding="utf-8")
            print(f"[{voice}] 保留 {len(single_kept)}/{len(todo)} 条 → {single_meta}", flush=True)
            for why, n in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  丢弃 {n} 条：{why}", flush=True)

            cap = caps.get(voice)
            selected = stratified_subset(kept_rows, cap, seed=args.seed + g_idx * 100 + idx) if cap is not None else kept_rows
            multi_counts[voice] = len(selected)
            for uid, _kind, text in selected:
                all_multi_kept.append(f"{voice}/{uid}.wav|{voice}|{text}")
            if cap is not None and len(selected) < len(kept_rows):
                print(f"  [{voice}] 合训按 kind 分层抽样：{len(kept_rows)} → {len(selected)} 条（cap={cap}）", flush=True)

        if wanted is not None:
            print(f"\n（传入了 --voices，已保留 {base_dir / 'metadata.csv'} 未覆盖）", flush=True)
            continue

        multi_meta = base_dir / "metadata.csv"
        multi_meta.write_text("\n".join(all_multi_kept) + "\n", encoding="utf-8")
        total_multi = len(all_multi_kept)
        print(f"\n=== [{cfg['work_name']}] 汇总：共保留 {total_multi} 条多说话人训练数据 → {multi_meta} ===", flush=True)
        for sid, (voice, count) in enumerate(multi_counts.items()):
            pct = 100.0 * count / total_multi if total_multi else 0.0
            print(f"  speaker_id={sid}: {voice:12s} {count:5d} 条 ({pct:5.1f}%)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
