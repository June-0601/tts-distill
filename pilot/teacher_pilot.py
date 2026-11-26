#!/usr/bin/env python3
"""试验：新一代 TTS 以 Kokoro 参考音频零样本克隆，朗读 pilot/texts.tsv 的测试段落。

每个引擎用自己仓库里的环境运行，例如：
  D:\\SharingNetwork\\Junjun\\VoxCPM\\env\\python.exe        teacher_pilot.py --engine voxcpm
  D:\\SharingNetwork\\Junjun\\Qwen3-TTS\\env\\python.exe     teacher_pilot.py --engine qwen3
  D:\\SharingNetwork\\Junjun\\index-tts-2.5\\.venv\\Scripts\\python.exe teacher_pilot.py --engine index25
  D:\\SharingNetwork\\Junjun\\CosyVoice3\\env\\python.exe    teacher_pilot.py --engine cosy3

输出：work/pilot/out/<engine>/<voice>/<pid>.wav（各引擎原生采样率），已存在的跳过。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work" / "pilot"
J = Path(r"D:\SharingNetwork\Junjun")

# 与 kokoro_pilot.py 的 REF_TEXTS 一一对应（参考音频的逐字文本）
REF_TEXTS = [
    "早上的空气有点凉。我泡了一杯茶，坐在窗边，慢慢翻开昨天没读完的那本书。",
    "这条小路两边种满了梧桐树。每到秋天，落叶铺了厚厚一层，踩上去沙沙作响。",
    "其实很多事情并没有那么复杂。只要一步一步来，总能找到解决的办法。",
]


def load_texts(only: str | None):
    rows = []
    for p in (HERE / "texts.tsv", WORK / "texts_book.tsv"):  # 后者是书里挑的段落，只在本地
        if p.exists():
            rows += [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only:
        keep = set(only.split(","))
        rows = [r for r in rows if r[0] in keep]
    return rows


class VoxCPMEngine:
    def __init__(self):
        from voxcpm import VoxCPM
        self.m = VoxCPM.from_pretrained(str(J / "VoxCPM" / "pretrained_models" / "VoxCPM2"), load_denoiser=False)
        self.sr = self.m.tts_model.sample_rate

    def synth(self, text, ref_wav, ref_text, out):
        import soundfile as sf
        # “极致克隆”：参考音频 + 逐字文本做续写，同时作为音色参考
        wav = self.m.generate(text=text, prompt_wav_path=ref_wav, prompt_text=ref_text,
                              reference_wav_path=ref_wav, cfg_value=2.0, inference_timesteps=10, seed=42)
        sf.write(out, wav, self.sr)


class Qwen3Engine:
    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.m = Qwen3TTSModel.from_pretrained(
            str(J / "Qwen3-TTS" / "pretrained_models" / "Qwen3-TTS-12Hz-1.7B-Base"),
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        self.prompts = {}

    def synth(self, text, ref_wav, ref_text, out):
        import soundfile as sf
        if ref_wav not in self.prompts:
            self.prompts[ref_wav] = self.m.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
        wavs, sr = self.m.generate_voice_clone(text=text, language="Chinese", voice_clone_prompt=self.prompts[ref_wav])
        sf.write(out, wavs[0], sr)


class Index25Engine:
    def __init__(self):
        root = J / "index-tts-2.5"
        os.chdir(root)
        sys.path.insert(0, str(root))
        from indextts.infer_v2_5 import IndexTTS2
        self.m = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)

    def synth(self, text, ref_wav, ref_text, out):
        self.m.infer(spk_audio_prompt=ref_wav, text=text, lang="ZH", output_path=out, verbose=False)


class Cosy3Engine:
    def __init__(self):
        root = J / "CosyVoice3"
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "third_party" / "Matcha-TTS")]
        from cosyvoice.cli.cosyvoice import AutoModel
        self.m = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")

    def synth(self, text, ref_wav, ref_text, out):
        import torch
        import torchaudio
        prompt = "You are a helpful assistant.<|endofprompt|>" + ref_text
        parts = [j["tts_speech"] for j in self.m.inference_zero_shot(text, prompt, ref_wav, stream=False)]
        torchaudio.save(out, torch.cat(parts, dim=1), self.m.sample_rate)


ENGINES = {"voxcpm": VoxCPMEngine, "qwen3": Qwen3Engine, "index25": Index25Engine, "cosy3": Cosy3Engine}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--voices", default="zf_046,zf_032")
    ap.add_argument("--ref", type=int, default=0, help="参考音频编号 0/1/2")
    ap.add_argument("--only", help="只跑这些段落 id，逗号分隔")
    args = ap.parse_args()

    texts = load_texts(args.only)
    eng = ENGINES[args.engine]()
    for voice in [v.strip() for v in args.voices.split(",") if v.strip()]:
        ref_wav = str(WORK / "refs" / f"{voice}_r{args.ref}.wav")
        ref_text = REF_TEXTS[args.ref]
        out_dir = WORK / "out" / args.engine / voice
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid, _kind, text in texts:
            out = out_dir / f"{pid}.wav"
            if out.exists():
                continue
            t0 = time.time()
            eng.synth(text, ref_wav, ref_text, str(out))
            print(f"[{args.engine}] {voice} {pid} {time.time() - t0:.1f}s", flush=True)
    print(f"DONE {args.engine}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
