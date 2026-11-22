#!/usr/bin/env python3
"""从已训好 200 轮的美音(6)+英音(2)模型衍生「英文为主 + 中文为辅」分支所需的新中文/混排老师音频（方案一）。

做法：
  1. 若 corpus/en_zh_mix.tsv 不存在，自动从 zh.tsv 与 en.tsv 生成 2400 条精选池：
     - word(400) / short(300) / mix(600) / gloss(350) / sent(350)
     - en_mix(400)：英文主句中穿插中文词语/短语（正反双向代码切换）
  2. 用 Kokoro-82M-v1.1-zh 的 zf_032 分别与 8 个英文音色向量混合（默认 --en-blend 0.35）合成；
     纯英文训练样本直接复用 work/en_us_multi 与 work/en_uk_multi 的现有音频，不重复合成。

用法：
  python3 synth_en_zh.py --group all
  python3 synth_en_zh.py --group us --en-blend 0.35
"""

from __future__ import annotations

import argparse
import collections
import random
import re
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

CORPUS_EN_ZH = HERE / "corpus" / "en_zh_mix.tsv"
HAN_ONLY = re.compile(r"^[一-鿿]+$")

EN_MIX_TEMPLATES = [
    ("zh_only", "Please read {zh} carefully."),
    ("zh_only", "A woman read {zh} under a tall tree."),
    ("zh_only", "The phrase {zh} appears in this passage."),
    ("zh_only", "Option A refers to {zh} in this note."),
    ("zh_only", "Look closely at {zh} in the next line."),
    ("zh_only", "She highlighted {zh} while reviewing the text."),
    ("zh_only", "Listen carefully when pronouncing {zh}."),
    ("zh_only", "Do you remember {zh} from yesterday's lesson?"),
    ("en_zh", "In this note, {en} is paired with {zh}."),
    ("en_zh", "Look at {zh} and then say {en}."),
    ("en_zh", "Today we reviewed both {en} and {zh}."),
    ("en_zh", "Try pronouncing {zh} right after {en}."),
    ("en_zh", "The card lists {zh} next to {en}."),
    ("en_zh", "Can you distinguish {en} from {zh} clearly?"),
]

# 36 条原生英文冠词 A/a (/ɐ/) 与字母/序号 A (/ˈeɪ/) 强对比强化句（由纯英文 Kokoro-82M 原声合成）
EN_A_CONTRAST_SENTENCES = [
    ("acont_001", "A woman saw a tall tree by a river."),
    ("acont_002", "A tree grows beside a quiet road."),
    ("acont_003", "A child held a red balloon in a park."),
    ("acont_004", "A bird landed on a wooden fence."),
    ("acont_005", "A student opened a thick notebook."),
    ("acont_006", "A gentle breeze moved through a green valley."),
    ("acont_007", "A teacher wrote a clear example on a board."),
    ("acont_008", "A dog chased a ball across a sunny yard."),
    ("acont_009", "A bright lantern lit a narrow lane."),
    ("acont_010", "A musician played a slow melody on a violin."),
    ("acont_011", "A farmer carried a basket to a local market."),
    ("acont_012", "A painter sketched a bridge over a calm lake."),
    ("acont_013", "Option A is next to a tree."),
    ("acont_014", "Option A is better than a wild guess."),
    ("acont_015", "Plan A gives a clear timeline for a team."),
    ("acont_016", "Grade A reflects a strong effort on a test."),
    ("acont_017", "Vitamin A supports a healthy diet for a child."),
    ("acont_018", "Is it A or B in a quiz like this?"),
    ("acont_019", "Choose A or B before writing a summary."),
    ("acont_020", "Part A explains a basic rule in a chapter."),
    ("acont_021", "Section A covers a classic case study."),
    ("acont_022", "Group A shared a table with a guest."),
    ("acont_023", "Step A begins with a short warm-up."),
    ("acont_024", "The letter A sounds different from a woman."),
    ("acont_025", "A. First, read a short sentence aloud."),
    ("acont_026", "Type A batteries fit a small flashlight."),
    ("acont_027", "Model A uses a lighter frame than a standard unit."),
    ("acont_028", "Line A stops near a busy station."),
    ("acont_029", "Gate A opens a few minutes before boarding."),
    ("acont_030", "Exit A leads directly to a quiet garden."),
    ("acont_031", "Room A has a large window facing a courtyard."),
    ("acont_032", "Class A requires a careful review of a manual."),
    ("acont_033", "Level A introduces a set of common phrases."),
    ("acont_034", "Case A shows a rapid change after a storm."),
    ("acont_035", "Answer A matches a clue in a paragraph."),
    ("acont_036", "Choice A offers a simple path to a solution."),
]

# 仅新合成中文与中英混排部分（各音色另从原 en_*_multi 复用约 2x 数量的纯英文样本）
EN_ZH_GROUPS = {
    "us": {
        "work_name": "en_us_zh_mix",
        "en_source": "en_us_multi",
        "british": False,
        "seed_base": 20261121,
        "voices": [
            ("af_heart", 1200, 2000),
            ("af_bella", 600, 1200),
            ("af_aoede", 600, 1200),
            ("af_jessica", 600, 1200),
            ("af_kore", 600, 1200),
            ("af_sarah", 600, 1200),
        ],
    },
    "uk": {
        "work_name": "en_uk_zh_mix",
        "en_source": "en_uk_multi",
        "british": True,
        "seed_base": 20261131,
        "voices": [
            ("bf_emma", 900, 1800),
            ("bf_isabella", 900, 1800),
        ],
    },
}


def ensure_en_zh_corpus(out_path: Path = CORPUS_EN_ZH, seed: int = 20261121) -> list[tuple[str, str, str]]:
    """生成并缓存 2400 条（中文 + 中英双向混排）语料。"""
    if out_path.exists():
        rows = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cols = line.split("\t")
            rows.append((cols[0], cols[1], cols[2]))
        if rows:
            return rows

    rng = random.Random(seed)
    zh_rows = read_corpus("zh")
    by_kind: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for r in zh_rows:
        by_kind[r[1]].append(r)

    selected: list[tuple[str, str, str]] = []
    selected.extend(by_kind["word"])
    selected.extend(by_kind["short"])
    selected.extend(by_kind["mix"])

    for kind, target_n in (("gloss", 350), ("sent", 350)):
        pool = list(by_kind[kind])
        rng.shuffle(pool)
        selected.extend(pool[:target_n])

    # 构造 400 条英文为主、中文词穿插其中的双向切换语料
    zh_words = [r[2] for r in by_kind["word"] if HAN_ONLY.match(r[2])]
    en_rows = read_corpus("en")
    en_words = [
        r[2] for r in en_rows
        if r[1] == "word" and re.fullmatch(r"[a-z]+", r[2]) and 3 <= len(r[2]) <= 12
    ]
    rng.shuffle(zh_words)
    rng.shuffle(en_words)

    for i in range(400):
        zh_w = zh_words[i % len(zh_words)]
        en_w = en_words[(i * 7 + 3) % len(en_words)]
        tmpl_kind, tmpl = EN_MIX_TEMPLATES[i % len(EN_MIX_TEMPLATES)]
        text = tmpl.format(zh=zh_w) if tmpl_kind == "zh_only" else tmpl.format(zh=zh_w, en=en_w)
        selected.append((f"emix_{i + 1:05d}", "en_mix", text))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(f"{uid}\t{kind}\t{text}" for uid, kind, text in selected) + "\n",
        encoding="utf-8",
    )
    print(f"已生成中文及双向混排语料：{out_path}（共 {len(selected)} 条）", flush=True)
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["us", "uk", "all"], default="all", help="合成口音组（默认 all）")
    ap.add_argument("--voices", help="只跑指定音色（逗号分隔）")
    ap.add_argument("--zh-anchor", default="zf_032", help="Kokoro 中文基准音色（默认 zf_032）")
    ap.add_argument("--en-blend", type=float, default=0.35, help="混入对应英文音色向量的比例（默认 0.35）")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--limit", type=int, help="每个音色最多合成前 N 条（调试用）")
    ap.add_argument("--device", help="cuda / cpu，默认自动选 cuda")
    ap.add_argument("--no-espeak", action="store_true")
    ap.add_argument("--build-corpus-only", action="store_true", help="仅生成 corpus/en_zh_mix.tsv 并退出")
    args = ap.parse_args()

    all_rows = ensure_en_zh_corpus()
    if args.build_corpus_only:
        return 0

    groups = ["us", "uk"] if args.group == "all" else [args.group]
    wanted = {v.strip() for v in args.voices.split(",") if v.strip()} if args.voices else None

    import torch
    from huggingface_hub import snapshot_download
    from misaki import en as misaki_en
    from misaki import espeak as misaki_espeak

    setup_espeak(args.no_espeak)
    from kokoro import KModel, KPipeline

    zh_code, zh_repo, zh_rev, _ = LANGS["zh"]
    _, en_repo, en_rev, _ = LANGS["en"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    zh_local = Path(
        snapshot_download(
            zh_repo,
            revision=zh_rev,
            allow_patterns=["*.json", "*.pth", f"voices/{args.zh_anchor}.pt"],
        )
    )
    target_voices = [
        v for g in groups for v, _, _ in EN_ZH_GROUPS[g]["voices"] if wanted is None or v in wanted
    ]
    if not target_voices:
        print("没有匹配到待合成的音色", flush=True)
        return 1

    en_local = Path(
        snapshot_download(
            en_repo,
            revision=en_rev,
            allow_patterns=["*.json", "*.pth"] + [f"voices/{v}.pt" for v in target_voices],
        )
    )

    model = KModel(
        repo_id=zh_repo,
        config=str(zh_local / "config.json"),
        model=str(next(zh_local.glob("*.pth"))),
    ).to(device).eval()

    zf_pack = torch.load(str(zh_local / "voices" / f"{args.zh_anchor}.pt"), weights_only=True)
    alpha = float(args.en_blend)

    for group_name in groups:
        cfg = EN_ZH_GROUPS[group_name]
        plan = [(v, n) for v, n, _ in cfg["voices"] if wanted is None or v in wanted]
        if not plan:
            continue

        try:
            fb = None if args.no_espeak else misaki_espeak.EspeakFallback(british=cfg["british"])
        except Exception:
            fb = None
        en_g2p = misaki_en.G2P(trf=False, british=cfg["british"], fallback=fb, unk="❓")
        pipe = KPipeline(
            lang_code=zh_code,
            repo_id=zh_repo,
            model=model,
            en_callable=lambda t, g=en_g2p: g(t)[0],
        )

        base_dir = args.work / cfg["work_name"]
        base_dir.mkdir(parents=True, exist_ok=True)
        bad_uids: set[str] = set()
        voice_idx_map = {v: i for i, (v, _, _) in enumerate(cfg["voices"])}

        print(
            f"\n================ 组 {group_name} ({cfg['work_name']}, "
            f"blend={1 - alpha:.2f}*{args.zh_anchor}+{alpha:.2f}*en, british={cfg['british']}) ================",
            flush=True,
        )

        for v_idx, (voice, zh_target) in enumerate(plan):
            en_pack = torch.load(str(en_local / "voices" / f"{voice}.pt"), weights_only=True)
            voice_pack = (1.0 - alpha) * zf_pack + alpha * en_pack.to(zf_pack.device, dtype=zf_pack.dtype)

            seed = cfg["seed_base"] + voice_idx_map.get(voice, v_idx)
            subset = stratified_subset(all_rows, zh_target, seed=seed)
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
                f"\n=== [{group_name} {v_idx + 1}/{len(plan)}] {voice} on {device}: "
                f"计划 {len(subset)} 条中文/混排，已完成 {len(subset) - len(todo)} 条，待合成 {len(todo)} 条 ===",
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
                            f"  [{voice}] {n}/{len(todo)}  已合成 {audio_sec / 3600:.2f} h  "
                            f"用时 {el / 60:.1f} 分  预计本音色还需 {eta / 60:.1f} 分",
                            flush=True,
                        )

    # 额外用纯英文原生 Kokoro-82M 合成 36 条冠词 A/a vs 字母 A 强对比句（零中文混色）
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    en_model = KModel(
        repo_id=en_repo,
        config=str(en_local / "config.json"),
        model=str(next(en_local.glob("*.pth"))),
    ).to(device).eval()

    for group_name in groups:
        cfg = EN_ZH_GROUPS[group_name]
        plan = [v for v, _, _ in cfg["voices"] if wanted is None or v in wanted]
        if not plan:
            continue
        en_pipe = KPipeline(lang_code="b" if cfg["british"] else "a", repo_id=en_repo, model=en_model)
        base_dir = args.work / cfg["work_name"]
        print(f"\n=== [{group_name}] 原生英文 Kokoro-82M 合成 A/a 对比强化句（每音色 {len(EN_A_CONTRAST_SENTENCES)} 条）===", flush=True)
        for voice in plan:
            vp = en_pipe.load_voice(str(en_local / "voices" / f"{voice}.pt"))
            out_dir = base_dir / "wavs" / voice
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = base_dir / f"synth_acont_{voice}.tsv"
            done = set()
            if log_path.exists():
                done = {line.split("\t")[0] for line in log_path.read_text(encoding="utf-8").splitlines() if line}
            todo = [
                (uid, txt)
                for uid, txt in EN_A_CONTRAST_SENTENCES
                if uid not in done or not (out_dir / f"{uid}.wav").exists()
            ]
            if not todo:
                continue
            with log_path.open("a", encoding="utf-8") as log:
                for uid, txt in todo:
                    results = list(en_pipe(txt, voice=vp, speed=1.0, split_pattern=None))
                    if len(results) != 1 or results[0].audio is None:
                        continue
                    audio = trim_edge_silence(results[0].audio.detach().cpu().numpy().astype(np.float32), SAMPLE_RATE)
                    sf.write(out_dir / f"{uid}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
                    log.write(f"{uid}\ta_contrast\t{len(audio) / SAMPLE_RATE:.2f}\t{results[0].phonemes}\n")
            print(f"  [{voice}] 完成 A/a 对比句 {len(todo)} 条", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
