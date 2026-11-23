#!/usr/bin/env python3
"""英文主模型 + 中文辅助音素体系与批量 G2P 工具（必须在 .venv-piper 中调用 Piper）。

音素槽位规划（共 232 个，保持在 num_symbols=256 以内，无损复用 200 轮纯英文权重）：
  - 0..165   : 原样继承 Piper DEFAULT_PHONEME_ID_MAP（英文 espeak IPA、控制符与英文标点）
  - 166..231 : 66 个中文拼音符号（py:Ø、23 声母、37 韵母、声调 py:1..py:5），不与 0..165 冲突
  - 中文全角标点映射到英文已有标点 ID（。→.、？→?、！→!、，/、/—/…→,、：→:、；→;）
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
PHONEMES_PATH = HERE / "corpus" / "en_zh_phonemes.json"
ZH_PHONEMES_PATH = HERE / "corpus" / "zh_phonemes.json"

ZH_PUNCT_MAP = {
    "。": ".",
    "？": "?",
    "！": "!",
    "，": ",",
    "、": ",",
    "：": ":",
    "；": ";",
    "—": ",",
    "…": ",",
    "（": ",",
    "）": ",",
    "【": ",",
    "】": ",",
    "(": ",",
    ")": ",",
    "[": ",",
    "]": ",",
    "“": '"',
    "”": '"',
}
ZH_TRIGGER_RE = re.compile(r"[一-鿿。？！、，：；—…（）【】\[\]]")
ZH_PUNCT_SPLIT_RE = re.compile(r"([。？！、，：；—…（）()【】\[\]“”]+)")
PAUSE_PUNCTS = {",", ":", ";"}

# 明确表示“字母/序号 A”的语境（避免 espeak 在 Option A is ... / A or B / A. First 等处误弱读成冠词 ɐ）
_LETTER_A_LABEL = (
    r"(?:Option|Choice|Part|Section|Group|Answer|Class|Grade|Team|Level|Step|Case|Type|Plan|"
    r"Vitamin|Category|Item|Appendix|Figure|Table|Column|Row|Zone|Phase|Stage|Tier|Version|"
    r"Model|Series|Track|Exit|Gate|Terminal|Block|Building|Wing|Room|Unit|Seat|Line|Route|"
    r"Channel|Point|Node|Side|Factor|Method|Approach|Sample|Set|List|Letter)"
)
_LETTER_A_RE = re.compile(
    rf"(?i)(?<![A-Za-z.])({_LETTER_A_LABEL}\s+)(?-i:A)(?![A-Za-z])"
    rf"|(?<![A-Za-z.])(A)(?=\s+(?:or|and|vs\.?|versus|to)\s+B(?![A-Za-z]))"
    rf"|(?<=[(\[（【])(A)(?=[)\]）】])"
    rf"|(?:^|(?<=\n))(A)(?=\.\s+[A-Z])"
)
_LETTER_A_SENTINEL = "\ue000LETTER_A\ue001"
_EN_AFFIX_PAREN_RE = re.compile(r"(?<=[A-Za-z])\(\s*(s|es|ies|ren|ed|ing)\s*\)", re.I)


def normalize_en_text(text: str) -> str:
    """规整英文换行、字母/序号 A 标记与括号停顿（espeak-ng 会吞掉 '(' ')' '[' ']' 而不发停顿）。"""
    def _repl_a(m: re.Match) -> str:
        if m.group(1) is not None:
            return f"{m.group(1)}{_LETTER_A_SENTINEL}"
        return _LETTER_A_SENTINEL

    s = re.sub(r"\r\n?", "\n", str(text or ""))
    s = _LETTER_A_RE.sub(_repl_a, s)
    s = re.sub(r"([A-Za-z0-9,;:\-\"'\ue001])\n+(?=[A-Za-z0-9(\"'“\ue000])", r"\1 ", s)
    s = re.sub(r"([.?!])\n+", r"\1 ", s)
    s = re.sub(r"\n+", ". ", s)

    affixes: list[str] = []

    def _save_affix(m: re.Match) -> str:
        tok = f"\ue010{len(affixes)}\ue011"
        affixes.append(m.group(0))
        return tok

    s = _EN_AFFIX_PAREN_RE.sub(_save_affix, s)

    def _rep_open(m: re.Match) -> str:
        prev = s[: m.start()]
        return " " if (not prev.strip() or re.search(r'[.?!,;:\-—(\["\'“‘]\s*$', prev)) else "; "

    s = re.sub(r"\s*([(\[（【])\s*", _rep_open, s)

    def _rep_close(m: re.Match) -> str:
        after = s[m.end() :]
        return "" if (not after.strip() or re.match(r'^[.?!,;:)\]"\'”’]', after)) else "; "

    s = re.sub(r"\s*([)\]）】])\s*", _rep_close, s)
    s = re.sub(r"\s+([.?!,;:])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"^[;,\s]+|[;,\s]+$", "", s)

    if affixes:
        s = re.sub(r"\ue010(\d+)\ue011", lambda m: affixes[int(m.group(1))], s)
    return s


def _normalize_en_a_text(text: str) -> str:
    return normalize_en_text(text)


def _phonemize_en_chunk(ep, espeak_voice: str, text: str) -> list[list[str]]:
    """对英文片段做 espeak 注音；遇到标注出的字母 A 时强制使用该口音独立的 'A' 音素（/ˈeɪ/）。"""
    norm = _normalize_en_a_text(text)
    if _LETTER_A_SENTINEL not in norm:
        return ep.phonemize(espeak_voice, norm)

    letter_a_ph = ep.phonemize(espeak_voice, "A")[0]
    parts = norm.split(_LETTER_A_SENTINEL)
    merged: list[list[str]] = []
    cur: list[str] = []

    for idx, part in enumerate(parts):
        if idx > 0:
            if cur and cur[-1] != " ":
                cur.append(" ")
            cur.extend(letter_a_ph)
        chunk = part.strip()
        if not chunk:
            continue
        lead = ""
        while chunk and chunk[0] in ".?!,:;":
            lead += chunk[0]
            chunk = chunk[1:]
        if lead:
            for ch in lead:
                cur.append(ch)
                if ch in ".?!":
                    merged.append(cur)
                    cur = []
                elif ch in PAUSE_PUNCTS:
                    cur.append(" ")
            chunk = chunk.strip()
            if not chunk:
                continue
        sents = ep.phonemize(espeak_voice, chunk)
        for s_i, sent in enumerate(sents):
            if s_i == 0:
                if cur and cur[-1] != " ":
                    cur.append(" ")
                cur.extend(sent)
            else:
                if cur:
                    merged.append(cur)
                cur = list(sent)
    if cur:
        merged.append(cur)
    return merged


def build_en_zh_phoneme_map(out_path: Path = PHONEMES_PATH) -> dict[str, list[int]]:
    """生成并写出兼容 200 轮英文底模的 232 符号音素表。"""
    from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

    id_map: dict[str, list[int]] = {k: list(v) for k, v in DEFAULT_PHONEME_ID_MAP.items()}
    next_id = max(max(v) for v in id_map.values()) + 1
    assert next_id == 166, f"Expected DEFAULT_PHONEME_ID_MAP end at 165, got {next_id - 1}"

    zh_map = json.loads(ZH_PHONEMES_PATH.read_text(encoding="utf-8"))
    zh_core = sorted(
        ((ids[0], sym) for sym, ids in zh_map.items() if 3 <= ids[0] <= 68),
        key=lambda x: x[0],
    )
    assert len(zh_core) == 66, f"Expected 66 core zh symbols, got {len(zh_core)}"

    for _, sym in zh_core:
        id_map[f"py:{sym}"] = [next_id]
        next_id += 1

    # 在 json 中也保留全角中文标点到英文已训标点 ID 的直观别名
    for zh_p, en_p in ZH_PUNCT_MAP.items():
        if en_p in id_map:
            id_map[zh_p] = list(id_map[en_p])

    assert next_id <= 256, f"Exceeds num_symbols=256: {next_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return id_map


def load_en_zh_phoneme_map() -> dict[str, list[int]]:
    if PHONEMES_PATH.exists():
        return json.loads(PHONEMES_PATH.read_text(encoding="utf-8"))
    return build_en_zh_phoneme_map(PHONEMES_PATH)


def _node_zh_segments(items: Sequence[tuple[str, str]]) -> dict[str, tuple[list, list[str]]]:
    """批量调用 zh_pinyin_tokens.mjs，避免重复触发 Node 启动成本。"""
    if not items:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp) / "in.tsv"
        tsv.write_text(
            "\n".join(f"{uid}\t{txt.replace(chr(9), ' ').replace(chr(10), ' ')}" for uid, txt in items) + "\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["node", str(HERE / "zh_pinyin_tokens.mjs"), str(tsv)],
            cwd=str(HERE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    out: dict[str, tuple[list, list[str]]] = {}
    for line in proc.stdout.splitlines():
        if not line:
            continue
        obj = json.loads(line)
        out[obj["id"]] = (obj["segs"], obj["bad"])
    return out


def texts_to_en_zh_ids(
    items: Sequence[tuple[str, str]],
    espeak_voice: str,
    id_map: Optional[dict[str, list[int]]] = None,
) -> dict[str, tuple[list[int], list[str]]]:
    """批量将 (uid, text) 转为 (phoneme_ids, bad_tokens)。

    纯英文文本（无汉字和中文全角标点）走 Piper 原生 EspeakPhonemizer + DEFAULT_PHONEME_ID_MAP，
    产生的 ID 序列与 200 轮纯英文模型比特一致；含中文或混排文本走统一混合管线。
    """
    from piper.phoneme_ids import phonemes_to_ids
    from piper.phonemize_espeak import EspeakPhonemizer

    if id_map is None:
        id_map = load_en_zh_phoneme_map()

    ep = EspeakPhonemizer()
    results: dict[str, tuple[list[int], list[str]]] = {}
    zh_items: list[tuple[str, str]] = []

    for uid, text in items:
        if not ZH_TRIGGER_RE.search(text):
            sents = _phonemize_en_chunk(ep, espeak_voice, text)
            ids = list(itertools.chain(*(phonemes_to_ids(s, id_map=id_map) for s in sents)))
            results[uid] = (ids, [])
        else:
            zh_items.append((uid, text))

    if not zh_items:
        return results

    parsed = _node_zh_segments(zh_items)
    for uid, _text in zh_items:
        segs, bad = parsed.get(uid, ([], ["node_missing"]))
        tokens: list[str] = []
        last_kind: Optional[str] = None  # 'zh' | 'en' | 'punct' | 'space'

        def ensure_boundary(next_kind: str) -> None:
            if last_kind in ("zh", "en") and next_kind in ("zh", "en") and last_kind != next_kind:
                tokens.append(" ")

        for seg in segs:
            if seg[0] == "z":
                _, init, fin, tone = seg
                ensure_boundary("zh")
                tokens.extend([f"py:{init}", f"py:{fin}", f"py:{tone}"])
                last_kind = "zh"
            else:
                raw_chunk = seg[1]
                for sub in ZH_PUNCT_SPLIT_RE.split(raw_chunk):
                    if not sub:
                        continue
                    if ZH_PUNCT_SPLIT_RE.fullmatch(sub):
                        for ch in sub:
                            mapped = ZH_PUNCT_MAP.get(ch)
                            if mapped and mapped in id_map:
                                # 跳过句首多余括号停顿以及连续重复的停顿标点
                                if mapped in PAUSE_PUNCTS and (not tokens or last_kind in ("space", "punct")):
                                    continue
                                tokens.append(mapped)
                                if mapped in PAUSE_PUNCTS:
                                    tokens.append(" ")
                                    last_kind = "space"
                                else:
                                    last_kind = "punct"
                    else:
                        stripped = sub.strip()
                        if not stripped:
                            if tokens and tokens[-1] != " " and last_kind in ("zh", "en"):
                                tokens.append(" ")
                                last_kind = "space"
                            continue
                        sents = _phonemize_en_chunk(ep, espeak_voice, stripped)
                        flat_en: list[str] = []
                        for s_idx, sent in enumerate(sents):
                            if s_idx > 0 and flat_en and flat_en[-1] != " ":
                                flat_en.append(" ")
                            flat_en.extend(sent)
                        if flat_en:
                            ensure_boundary("en")
                            tokens.extend(flat_en)
                            last_kind = "punct" if flat_en[-1] in ".?!,:;'\")" else (
                                "space" if flat_en[-1] == " " else "en"
                            )

        # 去掉末尾多余空格与孤立末尾括号逗号，避免 EOS 前无意义停顿
        while tokens and tokens[-1] in (" ", ","):
            tokens.pop()
        ids = phonemes_to_ids(tokens, id_map=id_map)
        results[uid] = (ids, bad)

    return results


if __name__ == "__main__":
    id_map = build_en_zh_phoneme_map()
    print(f"已生成 {PHONEMES_PATH}：共 {len(id_map)} 键，最大 ID = {max(max(v) for v in id_map.values())}")
    samples = [
        ("s1_en", "Trump supporters trust him more than their friends and family."),
        ("s2_zh", "你好，今天的天气很好。"),
        ("s3_mix", "考试里常考的 perfidy，对应的中文意思是不诚实，背信弃义。"),
        ("s4_en_mix", "The word abandon means 放弃，抛弃 in Chinese."),
    ]
    out = texts_to_en_zh_ids(samples, "en-us", id_map)
    for uid, txt in samples:
        ids, bad = out[uid]
        print(f"  {uid}: len={len(ids)}, bad={bad}, max_id={max(ids)}, head={ids[:18]} ... tail={ids[-10:]}")
