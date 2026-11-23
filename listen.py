#!/usr/bin/env python3
"""用导出的 ONNX 合成一组固定试听句（包括单读的词、中英混排句与纯英文句），并用 Whisper 回听打分。挑 checkpoint 时用。

  python listen.py en work/export/en_US-kokoro_heart-medium.onnx
  python listen.py zh work/export/zh_CN-kokoro_zf001-medium.onnx --whisper large-v3-turbo

在 .venv-piper 里跑。WAV 写到 <onnx 同目录>/listen-<lang>/，打分结果同时打印出来。
英文注音用 piper 自带的 espeak-ng（与网页的 piper-tts-web 同源）；中文（含中英双语）用 pinyin_ids.mjs（与网页同一份）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qa import en_units, levenshtein, mix_units, zh_units  # noqa: E402

HAN = re.compile(r"[一-鿿]")

TESTS = {
    "en": [
        "The evidence was dubious, and the panel remained impartial.",
        "Reading every day builds vocabulary faster than memorizing lists.",
        "She paused at the doorway, unsure whether to knock or simply walk away.",
        "Would you rather read the novel first, or watch the film?",
        "In 1998, forty-two volunteers joined the expedition to the northern glacier.",
        "ubiquitous", "although", "thoroughly", "read", "vocabulary", "photosynthesis", "abandon",
    ],
    "zh": [
        "你好，今天的天气很好。",
        "虽然你很有钱，但是我不相信你很开心。",
        "他去银行取了一些钱，然后回家做饭。",
        "这本书我已经读了三遍，每次都有新的收获。",
        "她每天早上都会喝一杯 coffee，然后再打开电脑开始工作。",
        "这个单词 ubiquitous 的意思是无处不在的。",
        "Reading every day builds vocabulary faster than memorizing lists.",
        "abandon",
        "放弃，抛弃，遗弃",
        "无处不在的",
        "谢谢",
    ],
}


def phoneme_ids(lang: str, texts, config):
    if lang == "en":
        from piper.phoneme_ids import phonemes_to_ids
        from piper.phonemize_espeak import EspeakPhonemizer
        from en_zh_g2p import _phonemize_en_chunk

        ph = EspeakPhonemizer()
        out = []
        for t in texts:
            ids = []
            for sent in _phonemize_en_chunk(ph, config["espeak"]["voice"], t):
                ids += phonemes_to_ids(sent, id_map=config["phoneme_id_map"])
            out.append(ids)
        return out
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.tsv", Path(tmp) / "out.jsonl"
        src.write_text("".join(f"t{i}\tsent\t{t}\n" for i, t in enumerate(texts)), encoding="utf-8")
        subprocess.run(["node", str(HERE / "pinyin_ids.mjs"), str(src), str(dst)], check=True)
        return [json.loads(line)["ids"] for line in dst.read_text(encoding="utf-8").splitlines() if line]


def units_for_test(lang: str, text: str):
    if lang == "en" or not HAN.search(text):
        return en_units(text), "en"
    if re.search(r"[A-Za-z]", text):
        return mix_units(text), "zh"
    return zh_units(text), "zh"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lang", choices=TESTS)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--speaker", help="多音色模型指定音色名或 speaker_id（不指定则测试全部音色）")
    ap.add_argument("--whisper", help="faster-whisper 模型，默认 en=small.en、zh=large-v3-turbo；none 不打分")
    ap.add_argument("--whisper-device", default="auto", help="faster-whisper 运行设备：auto / cuda / cpu")
    args = ap.parse_args()

    import onnxruntime as ort

    config = json.loads(Path(str(args.onnx) + ".json").read_text(encoding="utf-8"))
    rate = config["audio"]["sample_rate"]
    inf = config["inference"]
    num_speakers = config.get("num_speakers", 1)
    speaker_id_map = config.get("speaker_id_map") or {}
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    texts = TESTS[args.lang]
    all_ids = phoneme_ids(args.lang, texts, config)

    if num_speakers > 1 and speaker_id_map:
        if args.speaker is not None:
            if args.speaker in speaker_id_map:
                speakers = [(args.speaker, speaker_id_map[args.speaker])]
            else:
                sid = int(args.speaker)
                inv = {v: k for k, v in speaker_id_map.items()}
                speakers = [(inv.get(sid, f"spk_{sid}"), sid)]
        else:
            speakers = sorted(speaker_id_map.items(), key=lambda x: x[1])
    else:
        speakers = [(None, None)]

    model_name = args.whisper or ("small.en" if args.lang == "en" else "large-v3-turbo")
    whisper_model = None
    if model_name != "none":
        from faster_whisper import WhisperModel
        compute = "int8" if args.whisper_device == "cpu" else "auto"
        whisper_model = WhisperModel(model_name, device=args.whisper_device, compute_type=compute)

    for spk_name, sid in speakers:
        out_dir = args.onnx.parent / (f"listen-{args.lang}-{spk_name}" if spk_name else f"listen-{args.lang}")
        out_dir.mkdir(exist_ok=True)
        paths = []
        for i, (text, ids) in enumerate(zip(texts, all_ids)):
            inputs = {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array([inf["noise_scale"], inf["length_scale"], inf["noise_w"]], dtype=np.float32),
            }
            if num_speakers > 1:
                inputs["sid"] = np.array([sid or 0], dtype=np.int64)
            audio = session.run(None, inputs)[0].squeeze()
            path = out_dir / f"{i:02d}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            paths.append(path)
        label = f"[{spk_name} (sid={sid})] " if spk_name else ""
        print(f"\n{label}已写 {len(paths)} 条到 {out_dir}", flush=True)

        if whisper_model is not None:
            bad = 0
            for text, path in zip(texts, paths):
                ref, w_lang = units_for_test(args.lang, text)
                segs, _ = whisper_model.transcribe(str(path), language=w_lang, beam_size=5, without_timestamps=True)
                hyp_text = " ".join(s.text.strip() for s in segs)
                hyp, _ = units_for_test(args.lang, hyp_text)
                err = levenshtein(ref, hyp)
                bad += err > 0
                print(f"  {'✓' if err == 0 else '✗'} {path.name}  {text}  →  {hyp_text}", flush=True)
            print(f"{label}{len(texts) - bad}/{len(texts)} 条 Whisper 听写完全一致（单词和中文短语听写本身也会出错，结合试听判断）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
