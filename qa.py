#!/usr/bin/env python3
"""用 Whisper 回听 Kokoro 合成的音频，丢掉读错、漏读、时长异常的，写出 Piper 训练用的 metadata.csv。

  python3 qa.py en            # 默认 small.en
  python3 qa.py zh            # 默认 large-v3-turbo（支持纯中、中英混排、纯英）
  python3 qa.py en --model tiny.en --device cpu   # 试跑

输出：
  work/<lang>/metadata.csv  en：id.wav|text      zh：id.wav|text|phoneme_ids（Piper phoneme_ids 格式）
  work/<lang>/qa.tsv        每条的识别结果、错误率、去留原因（重跑时已有结果的直接复用）

判定：英文按词错率（数字两边都转成英文单词再比）；中文按不带声调的拼音比，
中英混排按「汉字拼音 + 英文词」序列比，繁简、同音字都不算错。单读的词/短语只允许极小差异。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = {"en": "small.en", "zh": "large-v3-turbo"}
# 每个单位（英文词 / 中文音节）的合理时长范围，秒（synth.py 已裁去首尾多余静音）
PER_UNIT = {"en": (0.08, 1.2), "zh": (0.08, 0.6)}


def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def en_units(text: str):
    from num2words import num2words

    text = text.lower().replace("-", " ").replace("’", "'")
    text = re.sub(r"(\d+(\.\d+)?)", lambda m: " " + num2words(float(m.group(1)) if "." in m.group(1)
                                                              else int(m.group(1))) + " ", text)
    text = text.replace("-", " ").replace(",", " ")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def zh_units(text: str):
    from pypinyin import lazy_pinyin

    han = "".join(re.findall(r"[一-鿿]", text))
    return lazy_pinyin(han, errors="ignore")


def mix_units(text: str):
    """中英混排句：按出现顺序提取汉字拼音与英文词。"""
    from pypinyin import lazy_pinyin

    out = []
    for m in re.finditer(r"[一-鿿]+|[A-Za-z0-9]+(?:'[A-Za-z]+)*", text):
        seg = m.group(0)
        if re.match(r"[一-鿿]", seg):
            out.extend(lazy_pinyin(seg, errors="ignore"))
        else:
            out.extend(en_units(seg))
    return out


def units_for(lang: str, kind: str, text: str):
    if lang == "en" or kind in ("en_word", "en_sent"):
        return en_units(text)
    if kind == "mix":
        return mix_units(text)
    return zh_units(text)


def judge(lang: str, kind: str, ref, hyp, sec: float):
    """返回 (保留?, 错误率, 原因)。"""
    if not ref:
        return False, 1.0, "参考文本为空"
    eff_lang = "en" if (lang == "en" or kind in ("en_word", "en_sent")) else "zh"
    err = levenshtein(ref, hyp)
    rate = err / len(ref)
    lo, hi = PER_UNIT[eff_lang]
    per = sec / len(ref)
    # 单个词或短释义：时长下限放宽，上限按词长限制
    if kind in ("word", "en_word", "gloss", "short") and len(ref) <= 4:
        if sec < 0.15:
            return False, rate, "时长过短"
        if sec > 0.8 + 1.0 * len(ref):
            return False, rate, "时长过长"
        if err > (0 if len(ref) == 1 else 1):
            return False, rate, "识别不一致"
        return True, rate, ""
    if per < lo or per > hi:
        return False, rate, f"语速异常（{per:.2f} 秒/{'词' if eff_lang == 'en' else '字'}）"
    # 中英混排句 Whisper 有时会把夹杂的单个英文生词音译成近似汉字，允许略宽容差
    max_err = max(2, round(0.15 * len(ref))) if kind == "mix" else max(1, round(0.08 * len(ref)))
    if err > max_err:
        return False, rate, "识别差异过多"
    return True, rate, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lang", choices=["en", "zh"])
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--model", help="faster-whisper 模型名")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = ap.parse_args()

    import soundfile as sf
    from faster_whisper import WhisperModel

    lang = args.lang
    rows = {}
    for line in (HERE / "corpus" / f"{lang}.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        rows[cols[0]] = cols
    wav_dir = args.work / lang / "wavs"
    qa_path = args.work / lang / "qa.tsv"
    cache = {}
    if qa_path.exists():
        for line in qa_path.read_text(encoding="utf-8").splitlines():
            uid, hyp = line.split("\t")[:2]
            cache[uid] = hyp

    todo = [uid for uid in rows if (wav_dir / f"{uid}.wav").exists()]
    need = [uid for uid in todo if uid not in cache]
    print(f"{lang}: 有音频 {len(todo)} 条，待识别 {len(need)} 条", flush=True)
    if need:
        model = WhisperModel(args.model or DEFAULT_MODEL[lang], device=args.device,
                             compute_type="auto")
        for n, uid in enumerate(need, 1):
            kind = rows[uid][1]
            w_lang = "en" if (lang == "en" or kind in ("en_word", "en_sent")) else "zh"
            segments, _ = model.transcribe(str(wav_dir / f"{uid}.wav"), language=w_lang, beam_size=5,
                                           condition_on_previous_text=False, without_timestamps=True)
            cache[uid] = " ".join(s.text.strip() for s in segments).replace("\t", " ")
            if n % 200 == 0 or n == len(need):
                print(f"  已识别 {n}/{len(need)}", flush=True)

    kept, report, reasons = [], [], {}
    for uid in todo:
        cols = rows[uid]
        kind, text = cols[1], cols[2]
        info = sf.info(str(wav_dir / f"{uid}.wav"))
        ref = units_for(lang, kind, text)
        hyp = units_for(lang, kind, cache[uid])
        ok, rate, why = judge(lang, kind, ref, hyp, info.duration)
        report.append(f"{uid}\t{cache[uid]}\t{rate:.3f}\t{'保留' if ok else why}")
        if ok:
            kept.append(f"{uid}.wav|{text}" + (f"|{cols[3]}" if lang == "zh" else ""))
        else:
            reasons[why] = reasons.get(why, 0) + 1

    qa_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.work / lang / "metadata.csv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{lang}: 保留 {len(kept)}/{len(todo)} 条 → {args.work / lang / 'metadata.csv'}")
    for why, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  丢弃 {n} 条：{why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
