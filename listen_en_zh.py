#!/usr/bin/env python3
"""用导出的 en+zh 混排 ONNX 合成 16 条测试句（纯英、英为主+中、中为主+英、纯中词句），并用 Whisper 回听打分。

用法（在 .venv-piper 里跑）：
  python3 listen_en_zh.py work/export/en_US-kokoro_us_zh_mix-medium.onnx
  python3 listen_en_zh.py work/export/en_GB-kokoro_uk_zh_mix-medium.onnx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from en_zh_g2p import texts_to_en_zh_ids  # noqa: E402
from qa import en_units, levenshtein, mix_units, zh_units  # noqa: E402

HAN = re.compile(r"[一-鿿]")

TESTS = [
    # 00..03: 纯英文回归校验（保证 200 轮英文底子零退化）
    "The evidence was dubious, and the panel remained impartial.",
    "Reading every day builds vocabulary faster than memorizing lists.",
    "ubiquitous",
    "thoroughly",
    # 04..05: 冠词 A/a (/ɐ/) vs 字母/序号 A (/ˈeɪ/) 专项回归校验
    "A woman saw a tall tree by a river.",
    "Option A is next to a tree.",
    # 06..11: 英文主句穿插中文词语/短语（最贴近目标场景）
    "The word abandon means 放弃，抛弃 in Chinese.",
    "Please read 渡鸦 carefully before the meeting.",
    "Today we learned 拍手喝彩 and photosynthesis.",
    "In this chapter, the key phrase is 背信弃义.",
    "She wrote 无处不在的 next to ubiquitous.",
    "Can you pronounce 谢谢 right after although?",
    # 12..14: 中文句穿插英文单词
    "这个单词 ubiquitous 的意思是无处不在的。",
    "她每天早上都会喝一杯 coffee，然后再打开电脑开始工作。",
    "作者在这里用 albeit 来强调尽管，虽然。",
    # 15..17: 纯中文词句
    "你好，今天的天气很好。",
    "放弃，抛弃，遗弃",
    "谢谢",
]


def units_for_bilingual(text: str):
    if not HAN.search(text):
        return en_units(text), "en"
    if re.search(r"[A-Za-z]", text):
        return mix_units(text), "zh"
    return zh_units(text), "zh"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--speaker", help="只测指定 speaker 名字或 sid")
    ap.add_argument("--no-whisper", action="store_true", help="只合成试听 WAV，不跑 Whisper 评分")
    ap.add_argument("--whisper-device", default="auto", help="auto / cuda / cpu")
    args = ap.parse_args()

    import onnxruntime as ort

    config = json.loads(Path(str(args.onnx) + ".json").read_text(encoding="utf-8"))
    rate = config["audio"]["sample_rate"]
    inf = config["inference"]
    num_speakers = config.get("num_speakers", 1)
    speaker_id_map = config.get("speaker_id_map") or {}
    espeak_voice = (config.get("espeak") or {}).get("voice", "en-us")
    id_map = config["phoneme_id_map"]

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    pairs = [(f"t{i:02d}", txt) for i, txt in enumerate(TESTS)]
    g2p_out = texts_to_en_zh_ids(pairs, espeak_voice, id_map)
    all_ids = []
    for i, _txt in enumerate(TESTS):
        ids, bad = g2p_out[f"t{i:02d}"]
        if bad or not ids:
            raise RuntimeError(f"G2P 失败 t{i:02d}: bad={bad}")
        all_ids.append(ids)

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

    whisper_en = None
    whisper_zh = None
    if not args.no_whisper:
        from faster_whisper import WhisperModel
        compute = "int8" if args.whisper_device == "cpu" else "auto"
        whisper_en = WhisperModel("small.en", device=args.whisper_device, compute_type=compute)
        whisper_zh = WhisperModel("large-v3-turbo", device=args.whisper_device, compute_type=compute)

    for spk_name, sid in speakers:
        out_dir = args.onnx.parent / (f"listen-en_zh-{spk_name}" if spk_name else "listen-en_zh")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, ids in enumerate(all_ids):
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

        if not args.no_whisper:
            bad = 0
            for text, path in zip(TESTS, paths):
                ref, w_lang = units_for_bilingual(text)
                wm = whisper_en if w_lang == "en" else whisper_zh
                segs, _ = wm.transcribe(str(path), language=w_lang, beam_size=5, without_timestamps=True)
                hyp_text = " ".join(s.text.strip() for s in segs)
                hyp, _ = units_for_bilingual(hyp_text)
                err = levenshtein(ref, hyp)
                bad += err > 0
                print(f"  {'✓' if err == 0 else '✗'} {path.name}  {text}  →  {hyp_text}", flush=True)
            print(f"{label}{len(TESTS) - bad}/{len(TESTS)} 条 Whisper 听写完全一致（结合试听判断）", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
