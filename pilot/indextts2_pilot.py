#!/usr/bin/env python3
"""试验：IndexTTS2 以 Kokoro 参考音频零样本克隆朗读测试段落（Windows，index-tts 的 .venv）。

  D:\\SharingNetwork\\Junjun\\index-tts\\.venv\\Scripts\\python.exe pilot\\indextts2_pilot.py --voices zf_046,zf_032 --ref r0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work" / "pilot"
INDEX_DIR = Path(r"D:\SharingNetwork\Junjun\index-tts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="zf_046")
    ap.add_argument("--ref", default="r0", help="参考音频编号 r0/r1/r2")
    ap.add_argument("--tag", default="indextts2")
    ap.add_argument("--only", help="只跑这些段落 id，逗号分隔")
    args = ap.parse_args()

    os.chdir(INDEX_DIR)
    sys.path.insert(0, str(INDEX_DIR))
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True)
    texts = [l.split("\t") for l in (HERE / "texts.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        texts = [t for t in texts if t[0] in keep]

    for voice in [v.strip() for v in args.voices.split(",") if v.strip()]:
        ref = WORK / "refs" / f"{voice}_{args.ref}.wav"
        out_dir = WORK / "out" / args.tag / voice
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid, _kind, text in texts:
            out = out_dir / f"{pid}.wav"
            if out.exists():
                continue
            t0 = time.time()
            tts.infer(spk_audio_prompt=str(ref), text=text, output_path=str(out), verbose=False)
            print(f"[{args.tag}] {voice} {pid} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
