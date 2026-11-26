#!/usr/bin/env python3
"""试验第 0 步（WSL，.venv-synth）：用 Kokoro 原版纯净音色生成
  1) 参考音频 refs/<voice>_r<k>.wav —— 给 IndexTTS2 / VibeVoice / FireRedTTS2 做零样本克隆；
  2) 对照组 out/kokoro/<voice>/<pid>.wav —— Kokoro 原版按阅读器的方式（逐句）朗读测试段落。

用法（在 tts-distill 目录）：
  .venv-synth/bin/python pilot/kokoro_pilot.py --voices zf_046,zf_032
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synth import LANGS, SAMPLE_RATE, setup_espeak, trim_edge_silence  # noqa: E402

# 参考音频用的句子：中速、陈述语气、短句为主（Kokoro 短句读得最自然），不与测试集重复。
REF_TEXTS = [
    "早上的空气有点凉。我泡了一杯茶，坐在窗边，慢慢翻开昨天没读完的那本书。",
    "这条小路两边种满了梧桐树。每到秋天，落叶铺了厚厚一层，踩上去沙沙作响。",
    "其实很多事情并没有那么复杂。只要一步一步来，总能找到解决的办法。",
]

SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="zf_046")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    setup_espeak(False)
    from kokoro import KModel, KPipeline

    code, repo, revision, _ = LANGS["zh"]
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    local = Path(snapshot_download(repo, revision=revision,
                                   allow_patterns=["*.json", "*.pth"] + [f"voices/{v}.pt" for v in voices]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=repo, config=str(local / "config.json"),
                   model=str(next(local.glob("*.pth")))).to(device).eval()
    pipe = KPipeline(lang_code=code, repo_id=repo, model=model)

    texts = []
    for p in (HERE / "texts.tsv", args.out.parent / "texts_book.tsv"):
        if p.exists():
            texts += [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def say(text: str, pack) -> np.ndarray:
        res = list(pipe(text, voice=pack, speed=1.0, split_pattern=None))
        return np.concatenate([r.audio.detach().cpu().numpy() for r in res if r.audio is not None]).astype(np.float32)

    gap = lambda s: np.zeros(int(SAMPLE_RATE * s), np.float32)

    for voice in voices:
        pack = pipe.load_voice(str(local / "voices" / f"{voice}.pt"))  # 纯净音色，不混 af_heart
        ref_dir = args.out.parent / "refs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for k, t in enumerate(REF_TEXTS):
            sents = [s for s in SENT_SPLIT.split(t) if s]
            pcm = np.concatenate(sum([[trim_edge_silence(say(s, pack)), gap(0.3)] for s in sents], []))
            sf.write(ref_dir / f"{voice}_r{k}.wav", pcm, SAMPLE_RATE, subtype="PCM_16")
            print(f"ref {voice}_r{k}: {len(pcm) / SAMPLE_RATE:.1f}s", flush=True)

        vdir = args.out / "kokoro" / voice
        vdir.mkdir(parents=True, exist_ok=True)
        for pid, _kind, text in texts:
            # 阅读器的方式：按句末标点逐句合成，句间 0.35s
            sents = [s for s in SENT_SPLIT.split(text) if s]
            pcm = np.concatenate(sum([[trim_edge_silence(say(s, pack)), gap(0.35)] for s in sents], []))
            sf.write(vdir / f"{pid}.wav", pcm, SAMPLE_RATE, subtype="PCM_16")
        print(f"kokoro {voice}: {len(texts)} 段完成 → {vdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
