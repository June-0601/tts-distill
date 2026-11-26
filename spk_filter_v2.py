#!/usr/bin/env python3
"""v2 音色把关：CAM++ 声纹（Fun-CosyVoice3 自带 campplus.onnx），每条合成音频与该音色 Kokoro 原声的平均声纹
做余弦相似度，写 work/zh_v2/spk_<voice>.tsv（uid  sim）。qa_v2.py 读它来丢掉音色跑偏的条目。
用 CosyVoice3 的环境运行（Windows）：
  D:\\SharingNetwork\\Junjun\\CosyVoice3\\env\\python.exe spk_filter_v2.py [--voices zf_046,...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torchaudio
import torchaudio.compliance.kaldi as kaldi

HERE = Path(__file__).resolve().parent
V2 = HERE / "work" / "zh_v2"
KOKORO = HERE / "work" / "pilot" / "out" / "kokoro"
ONNX = Path(r"D:\SharingNetwork\Junjun\CosyVoice3\pretrained_models\Fun-CosyVoice3-0.5B\campplus.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices")
    args = ap.parse_args()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    def embed(p: Path):
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        e = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        return e / np.linalg.norm(e)

    voices = sorted(p.name for p in (V2 / "wavs").iterdir() if p.is_dir())
    if args.voices:
        voices = [v for v in voices if v in args.voices.split(",")]
    for v in voices:
        c = np.mean([embed(p) for p in sorted((KOKORO / v).glob("*.wav"))], 0)
        c /= np.linalg.norm(c)
        out = V2 / f"spk_{v}.tsv"
        done = {}
        if out.exists():
            done = dict(l.split("\t") for l in out.read_text(encoding="utf-8").splitlines() if l)
        with out.open("a", encoding="utf-8") as f:
            for p in sorted((V2 / "wavs" / v).glob("*.wav")):
                if p.stem in done:
                    continue
                f.write(f"{p.stem}\t{float(embed(p) @ c):.4f}\n")
        sims = [float(x.split("\t")[1]) for x in out.read_text(encoding="utf-8").splitlines() if x]
        print(f"{v}: {len(sims)} 条，相似度中位 {np.median(sims):.3f}，p10 {np.percentile(sims, 10):.3f}", flush=True)


if __name__ == "__main__":
    main()
