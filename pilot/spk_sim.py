#!/usr/bin/env python3
"""音色相似度：CAM++ 声纹（Fun-CosyVoice3 自带 campplus.onnx），与 Kokoro 原声的平均声纹做余弦相似度。
用 CosyVoice3 的环境运行：
  D:\\SharingNetwork\\Junjun\\CosyVoice3\\env\\python.exe spk_sim.py
输出 work/pilot/spk_sim.tsv；越接近 1 越像（同一人通常 >0.75）。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work" / "pilot"
ONNX = Path(r"D:\SharingNetwork\Junjun\CosyVoice3\pretrained_models\Fun-CosyVoice3-0.5B\campplus.onnx")

sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


def embed(path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    e = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
    return e / np.linalg.norm(e)


def main():
    out = WORK / "out"
    centroid = {}
    for vdir in (out / "kokoro").iterdir():
        es = [embed(p) for p in sorted(vdir.glob("*.wav"))]
        c = np.mean(es, 0)
        centroid[vdir.name] = c / np.linalg.norm(c)
    res = defaultdict(list)
    for wav in sorted(out.glob("*/*/*.wav")):
        e, v = wav.parent.parent.name, wav.parent.name
        if v in centroid:
            res[(e, v)].append(float(embed(wav) @ centroid[v]))
    with (WORK / "spk_sim.tsv").open("w", encoding="utf-8") as f:
        f.write("engine\tvoice\tn\tspk_sim\tmin\n")
        for (e, v), s in sorted(res.items()):
            line = f"{e}\t{v}\t{len(s)}\t{np.mean(s):.4f}\t{np.min(s):.4f}"
            f.write(line + "\n")
            print(line)


if __name__ == "__main__":
    main()
