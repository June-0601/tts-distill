#!/usr/bin/env python3
"""用 Kokoro-82M-v1.1-zh 批量合成 8 个中文女声音色（多说话人蒸馏数据集），支持断点续跑。

音色与分层抽样计划（固定随机种子，保持各词句类别比例一致）：
  sid=0: zf_032  全量 6500 条
  sid=1: zf_046  分层抽样 2500 条
  sid=2: zf_021  分层抽样 1000 条
  sid=3: zf_022  分层抽样 1000 条
  sid=4: zf_023  分层抽样 1000 条
  sid=5: zf_024  分层抽样 1000 条
  sid=6: zf_059  分层抽样 1000 条
  sid=7: zf_079  分层抽样 1000 条

用法：
  python3 synth_multi.py                        # 按计划合成全部 8 个音色（默认 --en-blend 0.15）
  python3 synth_multi.py --voices zf_032,zf_046 # 只合成指定音色
"""

from __future__ import annotations

import argparse
import collections
import random
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

VOICE_PLAN = [
    ("zf_032", 6500),
    ("zf_046", 3500),
    ("zf_021", 1000),
    ("zf_022", 1000),
    ("zf_023", 1000),
    ("zf_024", 1000),
    ("zf_059", 1000),
    ("zf_079", 3500),
]


def stratified_subset(rows: list[tuple[str, str, str]], target_n: int, seed: int) -> list[tuple[str, str, str]]:
    """按 kind（word/gloss/short/sent/long/mix/en_word/en_sent）等比例分层抽样。"""
    if target_n >= len(rows):
        return list(rows)

    by_kind: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for r in rows:
        by_kind[r[1]].append(r)

    rng = random.Random(seed)
    total = len(rows)
    selected_ids: set[str] = set()
    leftovers: list[tuple[str, str, str]] = []

    for kind, group in by_kind.items():
        grp = list(group)
        rng.shuffle(grp)
        k = int(round(len(grp) * target_n / total))
        k = max(1, min(len(grp), k))
        for item in grp[:k]:
            selected_ids.add(item[0])
        leftovers.extend(grp[k:])

    rng.shuffle(leftovers)
    while len(selected_ids) < target_n and leftovers:
        selected_ids.add(leftovers.pop()[0])

    # 保持在原 zh.tsv 中的顺序
    out = [r for r in rows if r[0] in selected_ids]
    return out[:target_n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en-blend", type=float, default=0.15,
                    help="混入英文女声 af_heart 的比例（默认 0.15）")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--voices", help="只跑指定音色（逗号分隔，如 zf_032,zf_046）")
    ap.add_argument("--limit", type=int, help="每个音色最多合成前 N 条（调试用）")
    ap.add_argument("--device", help="cuda / cpu，默认自动选 cuda")
    ap.add_argument("--no-espeak", action="store_true")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    setup_espeak(args.no_espeak)
    from kokoro import KModel, KPipeline
    from misaki import en as misaki_en
    from misaki import espeak as misaki_espeak

    code, repo, revision, _ = LANGS["zh"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    plan = VOICE_PLAN
    if args.voices:
        wanted = {v.strip() for v in args.voices.split(",") if v.strip()}
        plan = [(v, n) for v, n in VOICE_PLAN if v in wanted]
        for v in wanted - {x[0] for x in plan}:
            plan.append((v, 1000))

    allow = ["*.json", "*.pth"] + [f"voices/{v}.pt" for v, _ in plan]
    local = Path(snapshot_download(repo, revision=revision, allow_patterns=allow))
    model = KModel(repo_id=repo, config=str(local / "config.json"),
                   model=str(next(local.glob("*.pth")))).to(device).eval()

    try:
        fallback = None if args.no_espeak else misaki_espeak.EspeakFallback(british=False)
    except Exception:
        fallback = None
    en_g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback, unk="❓")
    en_callable = lambda t: en_g2p(t)[0]

    pipe = KPipeline(lang_code=code, repo_id=repo, model=model, en_callable=en_callable)

    en_pack = None
    if 0.0 < args.en_blend < 1.0:
        _, en_repo, en_rev, en_voice = LANGS["en"]
        en_local = Path(snapshot_download(en_repo, revision=en_rev,
                                          allow_patterns=[f"voices/{en_voice}.pt"]))
        en_pack = torch.load(str(en_local / "voices" / f"{en_voice}.pt"), weights_only=True)

    all_rows = read_corpus("zh")
    base_dir = args.work / "zh_multi"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 记录已知会被 Kokoro G2P 切碎或含未知词的 uid，避免后续音色重复计算
    bad_uids: set[str] = set()

    plan_idx = {v: i for i, (v, _) in enumerate(VOICE_PLAN)}
    for idx, (voice, target_n) in enumerate(plan):
        voice_pack = pipe.load_voice(str(local / "voices" / f"{voice}.pt"))
        if en_pack is not None:
            alpha = float(args.en_blend)
            voice_pack = (1.0 - alpha) * voice_pack + alpha * en_pack.to(voice_pack.device, dtype=voice_pack.dtype)
            print(f"\n=== [{idx + 1}/{len(plan)}] 音色 {voice}（混入 {alpha:.2f} af_heart，目标 {target_n} 条）===", flush=True)
        else:
            print(f"\n=== [{idx + 1}/{len(plan)}] 音色 {voice}（目标 {target_n} 条）===", flush=True)

        subset = stratified_subset(all_rows, target_n, seed=20261111 + plan_idx.get(voice, idx))
        if args.limit:
            subset = subset[: args.limit]

        out_dir = base_dir / "wavs" / voice
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / f"synth_{voice}.tsv"
        done = set()
        if log_path.exists():
            done = {line.split("\t")[0] for line in log_path.read_text(encoding="utf-8").splitlines() if line}

        todo = [r for r in subset if (r[0] not in done or not (out_dir / f"{r[0]}.wav").exists()) and r[0] not in bad_uids]
        print(f"  {voice} on {device}: 计划 {len(subset)} 条，已完成 {len(subset) - len(todo)} 条，待合成 {len(todo)} 条 → {out_dir}", flush=True)
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
                skipped = [t.text for t in (results[0].tokens or []) if not t.phonemes and any(c.isalpha() for c in t.text)]
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
                    print(f"  [{voice}] {n}/{len(todo)}  已合成 {audio_sec / 3600:.2f} h 音频  用时 {el / 60:.1f} 分  "
                          f"预计当前音色还要 {eta / 60:.1f} 分", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
