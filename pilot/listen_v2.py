#!/usr/bin/env python3
"""v2 长文本试听：从 work/books/heldout.tsv（训练没见过的整章）每本书取一段连续正文，
像阅读器一样按句切开逐句合成、句间留固定停顿，拼成一整段长音频；新模型（带词边界）与旧 zh_multi 并排。

  .venv-piper/bin/python pilot/listen_v2.py --new work/export/zh_CN-kokoro_zh_v2-medium.onnx \
      --old /mnt/d/.../work/export/zh_CN-kokoro_zh_multi-medium.onnx --out /mnt/d/.../work/zh_v2/listen --tag e33

输出 <out>/<tag>/<model>_<voice>_<book>.wav 与 <out>/listen_<tag>.html（音频与书一样只留在本地）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SPEAKERS = ["zf_046", "zf_032", "zf_021", "zf_022", "zf_023", "zf_024", "zf_059", "zf_079"]
TARGET_CHARS = 420           # 每段约 1.5–2 分钟
GAP = {"。": 0.42, "！": 0.42, "？": 0.42, "；": 0.30, "…": 0.40, "\n": 0.70}
SENT = re.compile(r"[^。！？；…]*[。！？；…]+[”’」』）)]*|[^。！？；…]+$")


def passages(args_books):
    rows = [l.split("\t") for l in (args_books / "heldout.tsv").read_text(encoding="utf-8").splitlines() if l]
    out = {}
    for book in ("book3", "book2", "book1"):
        chunks = [t for _u, b, t in rows if b == book]
        # 跳过开头的章节标题/短句，从第一个较长的叙述块开始，取连续若干块
        start = next(i for i, t in enumerate(chunks) if len(t) >= 40)
        text, i = "", start
        while i < len(chunks) and len(text) < TARGET_CHARS:
            text += chunks[i]
            i += 1
        out[book] = text
    return out


def sentences(text):
    return [s.strip() for s in SENT.findall(text) if s.strip()]


def ids_for(texts, word_boundary):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.tsv", Path(tmp) / "out.jsonl"
        src.write_text("".join(f"t{i}\tsent\t{t}\n" for i, t in enumerate(texts)), encoding="utf-8")
        cmd = ["node", str(HERE / "pinyin_ids.mjs")] + (["--word-boundary"] if word_boundary else []) + [str(src), str(dst)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return [json.loads(l)["ids"] for l in dst.read_text(encoding="utf-8").splitlines() if l]


class Model:
    def __init__(self, onnx: Path):
        import onnxruntime as ort
        self.cfg = json.loads(Path(str(onnx) + ".json").read_text(encoding="utf-8"))
        self.sr = self.cfg["audio"]["sample_rate"]
        self.inf = self.cfg["inference"]
        self.spk = self.cfg.get("speaker_id_map") or {}
        self.wb = bool((self.cfg.get("zh_frontend") or {}).get("word_boundary"))
        self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])

    def say(self, ids, voice, length_scale=None):
        x = {"input": np.array([ids], dtype=np.int64), "input_lengths": np.array([len(ids)], dtype=np.int64),
             "scales": np.array([self.inf["noise_scale"], length_scale or self.inf["length_scale"],
                                 self.inf["noise_w"]], dtype=np.float32),
             "sid": np.array([self.spk[voice]], dtype=np.int64)}
        return self.sess.run(None, x)[0].squeeze()


def trim(a, sr, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    pad = int(0.03 * sr)
    return a[max(0, idx[0] - pad): idx[-1] + pad]


def write(path, a, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--books", type=Path, default=HERE / "work/books")
    ap.add_argument("--voices", default=",".join(SPEAKERS))
    args = ap.parse_args()
    voices = args.voices.split(",")

    models = {"new": Model(args.new)}
    if args.old:
        models["old"] = Model(args.old)
    ps = passages(args.books)
    od = args.out / args.tag
    od.mkdir(parents=True, exist_ok=True)
    stats = []
    for book, text in ps.items():
        sents = sentences(text)
        for mname, m in models.items():
            all_ids = ids_for(sents, m.wb)
            for v in voices:
                parts, speech = [], 0.0
                for s, ids in zip(sents, all_ids):
                    a = trim(m.say(ids, v), m.sr)
                    speech += len(a) / m.sr
                    parts += [a, np.zeros(int(GAP.get(s[-1:], GAP.get(s.rstrip("”’」』）)")[-1:], 0.4)) * m.sr), np.float32)]
                write(od / f"{mname}_{v}_{book}.wav", np.concatenate(parts), m.sr)
                n_han = len(re.findall(r"[一-鿿]", text))
                stats.append((mname, v, book, n_han / speech))
                print(f"{mname} {v} {book}: {n_han / speech:.2f} 字/秒", flush=True)

    # 试听页
    cps = {(m, v): np.mean([c for mm, vv, _b, c in stats if mm == m and vv == v]) for m in models for v in voices}
    names = {"book3": "流俗地", "book2": "勒古恩", "book1": "香蕉、沙滩与基地"}
    h = ["<!doctype html><meta charset=utf-8><title>v2 长文本试听 " + args.tag + "</title>",
         "<style>body{font:15px/1.6 system-ui;margin:24px;max-width:1100px}td,th{padding:6px 10px;border-bottom:1px solid #ddd;vertical-align:top}"
         "audio{width:260px}.t{color:#555;font-size:13px;max-width:1000px}</style>",
         f"<h2>v2 长文本试听（{args.tag}）— 训练未见过的章节，逐句合成、固定句间停顿</h2>",
         "<p>new = 新模型（CosyVoice3 / Qwen3 老师 + 电子书语料 + 词边界）；old = 旧 zh_multi（Kokoro 老师）。括号里是语速（汉字/秒）。</p>"]
    for book, text in ps.items():
        h.append(f"<h3>{names[book]}</h3><p class=t>{text}</p><table><tr><th>音色</th>"
                 + "".join(f"<th>{m}</th>" for m in models) + "</tr>")
        for v in voices:
            h.append(f"<tr><td>{v}</td>" + "".join(
                f"<td><audio controls preload=none src='{args.tag}/{m}_{v}_{book}.wav'></audio> ({cps[(m, v)]:.1f})</td>"
                for m in models) + "</tr>")
        h.append("</table>")
    (args.out / f"listen_{args.tag}.html").write_text("\n".join(h), encoding="utf-8")
    print("试听页：", args.out / f"listen_{args.tag}.html")


if __name__ == "__main__":
    main()
