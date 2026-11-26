#!/usr/bin/env python3
"""v2 训练音频合成：每个音色用你试听选定的老师引擎，以 Kokoro 原声参考音频（work/pilot/refs/<voice>_r0.wav）
零样本克隆，朗读 work/books/pool.tsv 里按音色分配的句块。断点续跑，已合成的跳过。

每个引擎用自己仓库的环境运行（脚本只处理分给该引擎的音色）：
  D:\\SharingNetwork\\Junjun\\CosyVoice3\\env\\python.exe synth_teacher.py --engine cosy3
  D:\\SharingNetwork\\Junjun\\Qwen3-TTS\\env\\python.exe  synth_teacher.py --engine qwen3

输出：work/zh_v2/wavs/<voice>/<uid>.wav（引擎原生采样率），work/zh_v2/plan_<voice>.tsv（分配到的句块），
      work/zh_v2/synth_<voice>.tsv（uid  kind  秒数）。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = Path(r"D:\SharingNetwork\Junjun")
WORK = HERE / "work"
OUT = WORK / "zh_v2"
SEED = 20261124

# 试听选定（2026-11-24）：音色 → (引擎, 书句块数, 旧语料条数)。书句块平均约 11 秒，旧语料约 3 秒。
VOICE_PLAN = {
    "zf_046": ("cosy3", 1450, 400),   # 约 5 小时，最喜欢
    "zf_032": ("qwen3", 1150, 350),   # 约 4 小时
    "zf_079": ("qwen3", 870, 300),    # 约 3 小时
    "zf_021": ("cosy3", 580, 250),    # 以下各约 2 小时
    "zf_059": ("cosy3", 580, 250),
    "zf_022": ("qwen3", 580, 250),
    "zf_023": ("qwen3", 580, 250),
    "zf_024": ("qwen3", 580, 250),
}
CHAR_BUDGET = 420  # Qwen3 每批总字数上限（RTX 4070 12GB；8×47 字中位长度刚好一批）
BOOK_MIX ={"book3": 0.40, "book2": 0.35, "book1": 0.25}  # 流俗地 / 勒古恩 / 香蕉沙滩与基地
REF_TEXT = "早上的空气有点凉。我泡了一杯茶，坐在窗边，慢慢翻开昨天没读完的那本书。"


def read_pool():
    rows = [l.split("\t") for l in (WORK / "books" / "pool.tsv").read_text(encoding="utf-8").splitlines() if l]
    return [(u, k, t) for u, k, t in rows]


def make_plans(pool):
    """各音色的书句块互不重复（文本多样性最大）；旧语料各音色独立抽样。顺序固定，结果可复现。"""
    rng = random.Random(SEED)
    by_book = {b: [r for r in pool if r[1] == b] for b in BOOK_MIX}
    for v in by_book.values():
        rng.shuffle(v)
    old = [r for r in pool if r[1] not in BOOK_MIX]
    cursor = {b: 0 for b in BOOK_MIX}
    plans = {}
    for voice, (_eng, n_book, n_old) in VOICE_PLAN.items():
        rows = []
        for b, frac in BOOK_MIX.items():
            k = round(n_book * frac)
            rows += by_book[b][cursor[b]:cursor[b] + k]
            cursor[b] += k
        r2 = random.Random(f"{SEED}-{voice}")
        rows += r2.sample(old, n_old)
        plans[voice] = rows
    return plans


def prompt_text(kind, text):
    # 单读的词补句号，让老师用自然的陈述降调收尾（同 synth.py）
    if kind in ("word", "en_word") and text and text[-1] not in ".!?。！？":
        return text + ("." if text.isascii() else "。")
    return text


class Cosy3:
    batch = 1

    def __init__(self):
        root = J / "CosyVoice3"
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "third_party" / "Matcha-TTS")]
        from cosyvoice.cli.cosyvoice import AutoModel
        self.m = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
        self.sr = self.m.sample_rate

    def synth(self, texts, ref):
        import torch
        p = "You are a helpful assistant.<|endofprompt|>" + REF_TEXT
        out = []
        for t in texts:
            parts = [j["tts_speech"] for j in self.m.inference_zero_shot(t, p, ref, stream=False)]
            out.append(torch.cat(parts, dim=1).squeeze(0).cpu().numpy())
        return out


class Qwen3:
    batch = 8  # 实测 RTX 4070：batch 8 RTF 0.44（单条 1.7），显存 7.8GB

    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.m = Qwen3TTSModel.from_pretrained(
            str(J / "Qwen3-TTS" / "pretrained_models" / "Qwen3-TTS-12Hz-1.7B-Base"),
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        self.prompts = {}
        self.sr = None

    def synth(self, texts, ref):
        if ref not in self.prompts:
            self.prompts[ref] = self.m.create_voice_clone_prompt(ref_audio=ref, ref_text=REF_TEXT)
        p = self.prompts[ref]
        wavs, sr = self.m.generate_voice_clone(text=list(texts), language=["Chinese"] * len(texts),
                                               voice_clone_prompt=p * len(texts) if isinstance(p, list) else p)
        self.sr = sr
        return wavs


ENGINES = {"cosy3": Cosy3, "qwen3": Qwen3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--voices", help="只跑这些音色（默认：分给该引擎的全部音色）")
    ap.add_argument("--limit", type=int, help="每个音色最多合成 N 条（调试）")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    plans = make_plans(read_pool())
    OUT.mkdir(parents=True, exist_ok=True)
    for v, rows in plans.items():
        with (OUT / f"plan_{v}.tsv").open("w", encoding="utf-8") as f:
            f.writelines(f"{u}\t{k}\t{t}\n" for u, k, t in rows)

    voices = [v for v, (e, _, _) in VOICE_PLAN.items() if e == args.engine]
    if args.voices:
        voices = [v for v in voices if v in args.voices.split(",")]
    eng = ENGINES[args.engine]()

    for voice in voices:
        ref = str(WORK / "pilot" / "refs" / f"{voice}_r0.wav")
        wav_dir = OUT / "wavs" / voice
        wav_dir.mkdir(parents=True, exist_ok=True)
        log = OUT / f"synth_{voice}.tsv"
        done = set()
        if log.exists():
            done = {l.split("\t")[0] for l in log.read_text(encoding="utf-8").splitlines() if l}
        todo = [r for r in plans[voice] if r[0] not in done]
        if args.limit:
            todo = todo[: args.limit]
        # 按长度排序再分批，减少批内补齐的浪费
        todo.sort(key=lambda r: len(r[2]))
        print(f"=== {voice} [{args.engine}]：共 {len(plans[voice])}，待合成 {len(todo)}", flush=True)
        t0, audio_sec = time.time(), 0.0
        # 分批：条数不超过 eng.batch，且每批总字数不超过 CHAR_BUDGET（长句块 8 条一批会爆 12GB 显存）
        batches, cur = [], []
        for r in todo:
            if cur and (len(cur) >= eng.batch or sum(len(x[2]) for x in cur) + len(r[2]) > CHAR_BUDGET):
                batches.append(cur)
                cur = []
            cur.append(r)
        if cur:
            batches.append(cur)

        def run(batch):
            """显存不足时清缓存、对半拆开重试，直到单条。"""
            try:
                return list(zip(batch, eng.synth([prompt_text(k, t) for _, k, t in batch], ref)))
            except Exception as e:
                import torch
                torch.cuda.empty_cache()
                if len(batch) > 1 and "out of memory" in repr(e).lower():
                    h = len(batch) // 2
                    return run(batch[:h]) + run(batch[h:])
                print(f"  {batch[0][0]} 等 {len(batch)} 条失败：{e!r}"[:300], flush=True)
                return []

        i = 0
        with log.open("a", encoding="utf-8") as lf:
            for batch in batches:
                i += len(batch)
                for (uid, kind, _t), w in run(batch):
                    w = np.asarray(w, dtype=np.float32).squeeze()
                    sf.write(wav_dir / f"{uid}.wav", w, eng.sr, subtype="PCM_16")
                    sec = len(w) / eng.sr
                    audio_sec += sec
                    lf.write(f"{uid}\t{kind}\t{sec:.2f}\n")
                lf.flush()
                n = i
                if n % 80 < len(batch) or n == len(todo):
                    el = time.time() - t0
                    print(f"  [{voice}] {n}/{len(todo)}  {audio_sec / 3600:.2f} h 音频  用时 {el / 60:.0f} 分  "
                          f"预计还要 {el / n * (len(todo) - n) / 60:.0f} 分", flush=True)
    print(f"DONE {args.engine}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
