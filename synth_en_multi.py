#!/usr/bin/env python3
"""用 Kokoro-82M 批量合成 8 个英文女声音色（方案 A：美音 6 合 1 + 英音 2 合 1），支持断点续跑。

分组与标准档抽样配比（固定随机种子，按 word/short/sent/long 分层抽样）：
  en_us_multi（lang_code='a'，对应 espeak en-us，共 15,000 条）：
    sid=0: af_heart    3000 条
    sid=1: af_bella    2400 条
    sid=2: af_aoede    2400 条
    sid=3: af_jessica  2400 条
    sid=4: af_kore     2400 条
    sid=5: af_sarah    2400 条
  en_uk_multi（lang_code='b'，对应 espeak en-gb-x-rp，共 8,000 条）：
    sid=0: bf_emma     4000 条
    sid=1: bf_isabella 4000 条

用法：
  python3 synth_en_multi.py                        # 合成全部 8 个音色（先 us 后 uk）
  python3 synth_en_multi.py --group us             # 只合成美音 6 音色
  python3 synth_en_multi.py --group uk             # 只合成英音 2 音色
  python3 synth_en_multi.py --voices af_heart,bf_emma
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from synth import (
    HERE,
    LANGS,
    SAMPLE_RATE,
    prompt_for_kokoro,
    read_corpus,
    setup_espeak,
    trim_edge_silence,
)
from synth_multi import stratified_subset

EN_GROUPS = {
    "us": {
        "lang_code": "a",
        "work_name": "en_us_multi",
        "espeak_voice": "en-us",
        "seed_base": 20261113,
        "voices": [
            ("af_heart", 3000),
            ("af_bella", 2400),
            ("af_aoede", 2400),
            ("af_jessica", 2400),
            ("af_kore", 2400),
            ("af_sarah", 2400),
        ],
    },
    "uk": {
        "lang_code": "b",
        "work_name": "en_uk_multi",
        "espeak_voice": "en-gb-x-rp",
        "seed_base": 20261200,
        "voices": [
            ("bf_emma", 4000),
            ("bf_isabella", 4000),
        ],
    },
}


def resolve_group_plans(group_arg: str, voices_arg: str | None) -> list[tuple[str, list[tuple[str, int]]]]:
    """按 --group 与 --voices 解析每组待处理音色与目标条数。"""
    groups = ["us", "uk"] if group_arg == "all" else [group_arg]
    wanted = {v.strip() for v in voices_arg.split(",") if v.strip()} if voices_arg else None
    resolved: list[tuple[str, list[tuple[str, int]]]] = []
    seen_wanted: set[str] = set()

    for g in groups:
        plan = list(EN_GROUPS[g]["voices"])
        if wanted is not None:
            plan = [(v, n) for v, n in plan if v in wanted]
            seen_wanted.update(v for v, _ in plan)
            for v in sorted(wanted - seen_wanted):
                if (g == "us" and v.startswith("a")) or (g == "uk" and v.startswith("b")):
                    plan.append((v, 2400))
                    seen_wanted.add(v)
        if plan:
            resolved.append((g, plan))
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["us", "uk", "all"], default="all", help="合成口音组（默认 all）")
    ap.add_argument("--voices", help="只跑指定音色（逗号分隔，如 af_heart,bf_emma）")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--limit", type=int, help="每个音色最多合成前 N 条（调试用）")
    ap.add_argument("--device", help="cuda / cpu，默认有显卡就用显卡")
    ap.add_argument("--no-espeak", action="store_true", help="关掉生词的 espeak 回退（只用于试跑）")
    args = ap.parse_args()

    group_plans = resolve_group_plans(args.group, args.voices)
    if not group_plans:
        print("没有匹配到待合成的音色", flush=True)
        return 1

    import torch
    from huggingface_hub import snapshot_download

    setup_espeak(args.no_espeak)
    from kokoro import KModel, KPipeline

    _, repo, revision, _ = LANGS["en"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    all_voices = [v for _, plan in group_plans for v, _ in plan]
    allow = ["*.json", "*.pth"] + [f"voices/{v}.pt" for v in all_voices]
    local = Path(snapshot_download(repo, revision=revision, allow_patterns=allow))
    model = KModel(repo_id=repo, config=str(local / "config.json"),
                   model=str(next(local.glob("*.pth")))).to(device).eval()

    all_rows = read_corpus("en")

    for group_name, plan in group_plans:
        cfg = EN_GROUPS[group_name]
        pipe = KPipeline(lang_code=cfg["lang_code"], repo_id=repo, model=model)
        base_dir = args.work / cfg["work_name"]
        base_dir.mkdir(parents=True, exist_ok=True)
        bad_uids: set[str] = set()
        plan_idx = {v: i for i, (v, _) in enumerate(cfg["voices"])}

        print(
            f"\n================ 组 {group_name} ({cfg['work_name']}, lang_code={cfg['lang_code']}, "
            f"espeak={cfg['espeak_voice']}) ================",
            flush=True,
        )

        for idx, (voice, target_n) in enumerate(plan):
            voice_pack = pipe.load_voice(str(local / "voices" / f"{voice}.pt"))
            seed = cfg["seed_base"] + plan_idx.get(voice, 100 + idx)
            subset = stratified_subset(all_rows, target_n, seed=seed)
            if args.limit:
                subset = subset[: args.limit]

            out_dir = base_dir / "wavs" / voice
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = base_dir / f"synth_{voice}.tsv"
            done = set()
            if log_path.exists():
                done = {line.split("\t")[0] for line in log_path.read_text(encoding="utf-8").splitlines() if line}

            todo = [
                r
                for r in subset
                if (r[0] not in done or not (out_dir / f"{r[0]}.wav").exists()) and r[0] not in bad_uids
            ]
            print(
                f"\n=== [{group_name} {idx + 1}/{len(plan)}] {voice} on {device}: "
                f"计划 {len(subset)} 条，已完成 {len(subset) - len(todo)} 条，待合成 {len(todo)} 条 → {out_dir} ===",
                flush=True,
            )
            if not todo:
                continue

            t0, audio_sec = time.time(), 0.0
            with log_path.open("a", encoding="utf-8") as log:
                for n, (uid, kind, text) in enumerate(todo, 1):
                    if uid in bad_uids:
                        continue
                    synth_text = prompt_for_kokoro(kind, text)
                    results = list(pipe(synth_text, voice=voice_pack, speed=1.0, split_pattern=None))
                    if len(results) != 1 or results[0].audio is None:
                        print(f"  跳过 {uid}：Kokoro 切成了 {len(results)} 段", flush=True)
                        bad_uids.add(uid)
                        continue
                    skipped = [
                        t.text
                        for t in (results[0].tokens or [])
                        if not t.phonemes and any(c.isalpha() for c in t.text)
                    ]
                    if skipped or "❓" in (results[0].phonemes or ""):
                        print(f"  跳过 {uid}：Kokoro 没读出 {skipped or '❓'}", flush=True)
                        bad_uids.add(uid)
                        continue
                    audio = results[0].audio.detach().cpu().numpy().astype(np.float32)
                    audio = trim_edge_silence(audio, SAMPLE_RATE)
                    sf.write(out_dir / f"{uid}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
                    sec = len(audio) / SAMPLE_RATE
                    audio_sec += sec
                    log.write(f"{uid}\t{kind}\t{sec:.2f}\t{results[0].phonemes}\n")
                    if n % 200 == 0 or n == len(todo):
                        log.flush()
                        el = time.time() - t0
                        eta = el / n * (len(todo) - n)
                        print(
                            f"  [{voice}] {n}/{len(todo)}  已合成 {audio_sec / 3600:.2f} h 音频  "
                            f"用时 {el / 60:.1f} 分  预计当前音色还要 {eta / 60:.1f} 分",
                            flush=True,
                        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
