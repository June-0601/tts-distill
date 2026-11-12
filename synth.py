#!/usr/bin/env python3
"""用 Kokoro 把 corpus/<lang>.tsv 合成为训练音频（24kHz 单声道 WAV），中断后重跑会跳过已完成的。

  python3 synth.py en              # af_heart → work/en/wavs/
  python3 synth.py zh              # zf_001（默认混入 15% af_heart 改善英文发音）→ work/zh/wavs/
  python3 synth.py zh --voice zf_094 --en-blend 0.2
  python3 synth.py en --limit 20   # 只合成前 20 条，试跑用

输出 work/<lang>/synth.tsv：id、kind、秒数、Kokoro 实际用的音素（排查读错时看）。
合成时：
  1. 单读的词（word / en_word）自动在送入 Kokoro 时补句号诱导自然降调收尾（训练标签仍保持无标点原词）；
  2. 中文管道启用 misaki 英文 G2P 回退（en_callable），使同一个中文女声能自然朗读中英混排与英文句；
  3. 自动裁去 Kokoro 句首 ~0.3s、句尾 ~0.45s 的冗长死寂，统一留出 60ms 头静音 + 120ms 尾静音，避免单读词对齐发散。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
LANGS = {
    # lang: (Kokoro lang_code, 模型仓库, 固定提交, 默认音色)
    "en": ("a", "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", "af_heart"),
    "zh": ("z", "hexgrad/Kokoro-82M-v1.1-zh", "01e7505bd6a7a2ac4975463114c3a7650a9f7218", "zf_001"),
}
SAMPLE_RATE = 24000
HAN = re.compile(r"[一-鿿]")


SYSTEM_ESPEAK = [  # (库, 数据目录)
    ("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
    ("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1", "/usr/lib/aarch64-linux-gnu/espeak-ng-data"),
    ("/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"),
]


def setup_espeak(disable: bool) -> None:
    """Kokoro 英文遇到词典外的词会回退到 espeak-ng。misaki 自带的那份（espeakng-loader）
    把数据目录写死成它打包机器上的路径，在 macOS 上一初始化就让整个进程退出。
    装了系统 espeak-ng（WSL：apt install espeak-ng）就改用系统的；
    --no-espeak 则关掉回退（生词会被跳过，只用于在没有 espeak 的机器上试跑）。"""
    import os

    import espeakng_loader

    lib, data = os.environ.get("ESPEAK_LIBRARY"), os.environ.get("ESPEAK_DATA_PATH")
    if not lib:
        lib, data = next(((l, d) for l, d in SYSTEM_ESPEAK if Path(l).exists() and Path(d).exists()), (None, None))
    if lib:
        espeakng_loader.get_library_path = lambda: lib
        espeakng_loader.get_data_path = lambda: data
        print(f"espeak-ng：用系统的 {lib}", flush=True)
    if disable:
        import misaki.espeak

        class _Off:
            def __init__(self, *a, **k):
                raise RuntimeError("--no-espeak")

        misaki.espeak.EspeakFallback = _Off


def trim_edge_silence(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    lead_sec: float = 0.08,
    tail_sec: float = 0.14,
) -> np.ndarray:
    """裁掉 Kokoro 默认自带的 ~0.3s 头静音与 ~0.45s 尾静音，保留均匀的 80ms 头 + 140ms 尾。"""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < 1e-5:
        return audio
    thr = max(1e-4, peak * 0.005)
    win = max(1, int(round(sr * 0.01)))
    idx = np.flatnonzero(np.abs(audio) > thr)
    if idx.size == 0:
        return audio
    start = max(0, int(idx[0]) - win - int(round(sr * lead_sec)))
    end = min(len(audio), int(idx[-1]) + win + int(round(sr * tail_sec)))
    trimmed = audio[start:end]
    if peak > 0.98:
        trimmed = trimmed * (0.95 / peak)
    return trimmed.astype(np.float32, copy=False)


def prompt_for_kokoro(kind: str, text: str) -> str:
    """单读的词在送给 Kokoro 合成时补上句末句号，让老师用自然的词典降调收尾。"""
    if kind in ("word", "en_word") and text and text[-1] not in ".!?。！？":
        return text + ("。" if HAN.search(text) else ".")
    return text


def read_corpus(lang: str):
    rows = []
    for line in (HERE / "corpus" / f"{lang}.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        rows.append((cols[0], cols[1], cols[2]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lang", choices=LANGS)
    ap.add_argument("--voice", help="Kokoro 音色，默认 en=af_heart、zh=zf_001")
    ap.add_argument("--en-blend", type=float, default=0.15,
                    help="合成 zh 时混入英文女声 af_heart 的比例（默认 0.15，消除中式英语腔且保持音色统一；0 为不混）")
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--limit", type=int, help="只合成前 N 条")
    ap.add_argument("--device", help="cuda / cpu，默认有显卡就用显卡")
    ap.add_argument("--no-espeak", action="store_true", help="关掉英文生词的 espeak 回退（只用于试跑）")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    setup_espeak(args.no_espeak)
    from kokoro import KModel, KPipeline

    code, repo, revision, default_voice = LANGS[args.lang]
    voice = args.voice or default_voice
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 钉住模型版本：先把固定提交下到本地缓存，再从缓存加载
    local = Path(snapshot_download(repo, revision=revision,
                                   allow_patterns=["*.json", "*.pth", f"voices/{voice}.pt"]))
    model = KModel(repo_id=repo, config=str(local / "config.json"),
                   model=str(next(local.glob("*.pth")))).to(device).eval()

    en_callable = None
    if args.lang == "zh":
        from misaki import en as misaki_en
        from misaki import espeak as misaki_espeak

        try:
            fallback = None if args.no_espeak else misaki_espeak.EspeakFallback(british=False)
        except Exception:
            fallback = None
        en_g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback, unk="❓")
        en_callable = lambda t: en_g2p(t)[0]

    pipe = KPipeline(lang_code=code, repo_id=repo, model=model, en_callable=en_callable)
    voice_pack = pipe.load_voice(str(local / "voices" / f"{voice}.pt"))

    if args.lang == "zh" and 0.0 < args.en_blend < 1.0:
        _, en_repo, en_rev, en_voice = LANGS["en"]
        en_local = Path(snapshot_download(en_repo, revision=en_rev,
                                          allow_patterns=[f"voices/{en_voice}.pt"]))
        en_pack = torch.load(str(en_local / "voices" / f"{en_voice}.pt"), weights_only=True)
        alpha = float(args.en_blend)
        voice_pack = (1.0 - alpha) * voice_pack + alpha * en_pack.to(voice_pack.device, dtype=voice_pack.dtype)
        print(f"音色融合：{1.0 - alpha:.2f} * {voice} + {alpha:.2f} * {en_voice}", flush=True)

    out_dir = args.work / args.lang / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.work / args.lang / "synth.tsv"
    done = set()
    if log_path.exists():
        done = {line.split("\t")[0] for line in log_path.read_text(encoding="utf-8").splitlines()}

    rows = read_corpus(args.lang)[: args.limit]
    todo = [r for r in rows if r[0] not in done or not (out_dir / f"{r[0]}.wav").exists()]
    print(f"{args.lang}/{voice} on {device}: 共 {len(rows)} 条，待合成 {len(todo)} 条 → {out_dir}", flush=True)

    t0, audio_sec = time.time(), 0.0
    with log_path.open("a", encoding="utf-8") as log:
        for n, (uid, kind, text) in enumerate(todo, 1):
            synth_text = prompt_for_kokoro(kind, text)
            # split_pattern=None：整句一次合成；超长被 Kokoro 切成几段的跳过，免得拼接处不自然
            results = list(pipe(synth_text, voice=voice_pack, speed=1.0, split_pattern=None))
            if len(results) != 1 or results[0].audio is None:
                print(f"  跳过 {uid}：Kokoro 切成了 {len(results)} 段", flush=True)
                continue
            # 词典外的词 Kokoro 会悄悄跳过（英文 phonemes 为空，中文记成 ❓），这种句子不要
            skipped = [t.text for t in (results[0].tokens or []) if not t.phonemes and any(c.isalpha() for c in t.text)]
            if skipped or "❓" in (results[0].phonemes or ""):
                print(f"  跳过 {uid}：Kokoro 没读出 {skipped or '❓'}", flush=True)
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
                print(f"  {n}/{len(todo)}  已合成 {audio_sec / 3600:.2f} h 音频  用时 {el / 60:.1f} 分  "
                      f"预计还要 {eta / 60:.1f} 分", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
