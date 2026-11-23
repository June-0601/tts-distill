#!/usr/bin/env python3
"""Optimized wrapper and voice-pack loader for original Kokoro engines (EN + ZH).

Fixes:
1. Chinese (Kokoro-82M-v1.1-zh):
   - Context polyphone fixes via pypinyin phrase dict (+ optional g2pW)
   - Temperature / percent / fraction / date normalization (eliminates '❓')
   - Parentheses / bracket pause insertion for bilingual names e.g. 中文（English Name）
   - Strips choppy '/' word breaks and redundant post-punctuation spaces
   - Splits long Chinese text along 。？！；\\n (fixing upstream KPipeline regex omission)
   - Loads prosody-optimized voices_zh/*.pt (preserves :128 timbre, cleans 128: pacing)
   - Caps residual internal silences cleanly via 24kHz raised-cosine RMS compression
2. English (Kokoro-82M):
   - Disambiguates capitalized article 'A' ('A tree', 'A woman') -> /ə/ while keeping
     true letter labels ('Option A', 'Grade A', 'A or B', 'A. First') as /ˈeɪ/
   - Normalizes linebreaks that confuse misaki's token aligner
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

OPT_DIR = Path(__file__).resolve().parent
VOICES_ZH_DIR = OPT_DIR / "voices_zh"

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_ZH_DIGITS = "零一二三四五六七八九"

# High-frequency polyphones mispronounced by PaddleSpeech/pypinyin default dictionary in Kokoro
_POLYPHONE_PHRASES = {
    "流血": [["liú"], ["xuè"]],
    "出血": [["chū"], ["xuè"]],
    "吐血": [["tù"], ["xuè"]],
    "鲜血": [["xiān"], ["xuè"]],
    "血管": [["xuè"], ["guǎn"]],
    "血型": [["xuè"], ["xíng"]],
    "血泊": [["xuè"], ["pō"]],
    "血汗": [["xuè"], ["hàn"]],
    "划破": [["huà"], ["pò"]],
    "划分": [["huà"], ["fēn"]],
    "划归": [["huà"], ["guī"]],
    "策划": [["cè"], ["huà"]],
    "计划": [["jì"], ["huà"]],
    "还给": [["huán"], ["gěi"]],
    "还他": [["huán"], ["tā"]],
    "还她": [["huán"], ["tā"]],
    "还我": [["huán"], ["wǒ"]],
    "还您": [["huán"], ["nín"]],
    "还钱": [["huán"], ["qián"]],
    "还书": [["huán"], ["shū"]],
    "归还": [["guī"], ["huán"]],
    "偿还": [["cháng"], ["huán"]],
    "退还": [["tuì"], ["huán"]],
    "第一行": [["dì"], ["yī"], ["háng"]],
    "第二行": [["dì"], ["èr"], ["háng"]],
    "第三行": [["dì"], ["sān"], ["háng"]],
    "上一行": [["shàng"], ["yī"], ["háng"]],
    "下一行": [["xià"], ["yī"], ["háng"]],
    "换了行": [["huàn"], ["le"], ["háng"]],
    "换行": [["huàn"], ["háng"]],
    "一行字": [["yī"], ["háng"], ["zì"]],
    "数一数": [["shǔ"], ["yi"], ["shǔ"]],
    "数不清": [["shǔ"], ["bù"], ["qīng"]],
    "数得着": [["shǔ"], ["de"], ["zháo"]],
    "干什么": [["gàn"], ["shén"], ["me"]],
    "干嘛": [["gàn"], ["má"]],
    "干坏事": [["gàn"], ["huài"], ["shì"]],
    "慢慢地": [["màn"], ["màn"], ["de"]],
    "渐渐地": [["jiàn"], ["jiàn"], ["de"]],
    "轻轻地": [["qīng"], ["qīng"], ["de"]],
    "悄悄地": [["qiāo"], ["qiāo"], ["de"]],
    "静静地": [["jìng"], ["jìng"], ["de"]],
    "仔细地": [["zǐ"], ["xì"], ["de"]],
    "认真地": [["rèn"], ["zhēn"], ["de"]],
    "不断地": [["bù"], ["duàn"], ["de"]],
    "不停地": [["bù"], ["tíng"], ["de"]],
    "高兴地": [["gāo"], ["xìng"], ["de"]],
    "迅速地": [["xùn"], ["sù"], ["de"]],
    "拼命地": [["pīn"], ["mìng"], ["de"]],
    "角色": [["jué"], ["sè"]],
    "校对": [["jiào"], ["duì"]],
    "头发": [["tóu"], ["fa"]],
    "尽量": [["jǐn"], ["liàng"]],
}

_LETTER_A_BEFORE_RE = re.compile(
    r"(?:option|choice|part|section|group|answer|class|grade|team|level|step|case|type|plan|"
    r"vitamin|category|item|appendix|figure|fig\.?|table|column|row|zone|phase|stage|tier|version|"
    r"model|series|track|exit|gate|terminal|block|building|wing|room|unit|seat|line|route|"
    r"channel|point|node|side|factor|method|approach|sample|set|list|letter|exhibit|form|"
    r"schedule|an|hepatitis|marked|rated|scored|from|between)[ \t]+$",
    re.I,
)
_EN_WORD_A_RE = re.compile(r"(?<![A-Za-z0-9'.\-])A(?![A-Za-z0-9'\-])")
_LINE_END_PUNCT_RE = re.compile(r"[。？！，、：；.?!,;:][”’\"')）】》]*$")


def _int_to_zh(n: int) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "负" + _int_to_zh(-n)
    if n >= 100_000_000:
        q, r = divmod(n, 100_000_000)
        return _int_to_zh(q) + "亿" + (("零" if r < 10_000_000 else "") + _int_to_zh(r) if r else "")
    if n >= 10_000:
        q, r = divmod(n, 10_000)
        return _int_to_zh(q) + "万" + (("零" if r < 1_000 else "") + _int_to_zh(r) if r else "")
    units = ["", "十", "百", "千"]
    s = str(n)
    out, zero = "", False
    for i, ch in enumerate(s):
        d, pos = int(ch), len(s) - 1 - i
        if d == 0:
            zero = True
        else:
            if zero:
                out += "零"
                zero = False
            if not (d == 1 and pos == 1 and not out):
                out += _ZH_DIGITS[d]
            out += units[pos]
    return out


def _zh_num_str(num_str: str) -> str:
    if "." in num_str:
        a, b = num_str.split(".", 1)
        return _int_to_zh(int(a)) + "点" + "".join(_ZH_DIGITS[int(d)] for d in b)
    return "".join(_ZH_DIGITS[int(d)] for d in num_str) if len(num_str) > 8 else _int_to_zh(int(num_str))


def normalize_en_a(raw_text: str) -> str:
    """Lowercase standalone capitalized article 'A' (e.g. 'A tree', 'A woman') while keeping letter A and clean brackets."""
    s = re.sub(r"\r\n?", "\n", str(raw_text or ""))
    if re.fullmatch(r'["\'(\s]*A["\').!?\s]*', s.strip()):
        return s

    out = []
    last = 0
    for m in _EN_WORD_A_RE.finditer(s):
        out.append(s[last:m.start()])
        before = s[:m.start()]
        after = s[m.end():]
        keep = False
        if (re.match(r"^\s*$", after) and re.match(r"^[\s(\[\"']*$", before)) or re.match(r"^[)+:/]|^\.(?:\s|\d|&|$)", after):
            keep = True
        elif re.search(r"[(\[]$", before) and re.match(r"^[ \t]*[)\]]", after):
            keep = True
        elif re.match(r"^[ \t]*(?:&\s*[A-Z]|(?:or|and|vs\.?|versus|to|through|,)[ \t]+[B-Z])\b", after):
            keep = True
        elif re.match(r"^[ \t]+(?:stands[ \t]+for|represents|denotes|equals|comes[ \t]+before|is[ \t]+the[ \t]+(?:first|letter))\b", after, re.I):
            keep = True
        elif _LETTER_A_BEFORE_RE.search(before):
            keep = True
        elif re.search(r"(?:\bbetween[ \t]+[B-Z][ \t]+and|\bfrom[ \t]+[B-Z][ \t]+to|(?:grade|score|blood[ \t]+(?:type|group)|option|choice)[ \t]+(?:is|was))[ \t]+$", before, re.I):
            keep = True
        elif re.match(r"^[ \t]*[.!?'\")]*[ \t]*(?:\n|$)", after):
            keep = True
        out.append("A" if keep else "a")
        last = m.end()
    out.append(s[last:])
    res = "".join(out)

    # Fold linebreaks after disambiguating 'A' so heading lines like "First item\nA bird" aren't confused
    res = re.sub(r"([A-Za-z0-9,;:\-\"'])\n+(?=[A-Za-z0-9(\"'“])", r"\1 ", res)
    res = re.sub(r"([.?!])\n+", r"\1 ", res)
    res = re.sub(r"\n+", ". ", res)

    # Convert unsupported English brackets '[' ']' '（' '）' '【' '】' into comma pauses; keep native '(' ')' for Kokoro EN
    def _rep_open(m: re.Match) -> str:
        prev = res[: m.start()]
        return " " if (not prev.strip() or re.search(r'[.?!,;:\-—(\["\'“‘]\s*$', prev)) else ", "

    res = re.sub(r"\s*([\[\uFF08\u3010])\s*", _rep_open, res)

    def _rep_close(m: re.Match) -> str:
        after = res[m.end() :]
        return "" if (not after.strip() or re.match(r'^[.?!,;:)\]"\'”’]', after)) else ", "

    res = re.sub(r"\s*([\]\uFF09\u3011])\s*", _rep_close, res)
    res = re.sub(r"\s+([.?!,;:])", r"\1", res)
    res = re.sub(r"[ \t]{2,}", " ", res)
    return re.sub(r"^[,\s]+|[,\s]+$", "", res)


def normalize_zh_text(raw_text: str, keep_dunhao: bool = False) -> str:
    """Normalize Chinese punctuation, brackets, newlines, and numbers/units before Kokoro G2P."""
    s = re.sub(r"\r\n?", "\n", str(raw_text or ""))
    lines = [l.strip() for l in s.split("\n") if l.strip()]
    if len(lines) > 1:
        s = "".join(
            l if (i == len(lines) - 1 or _LINE_END_PUNCT_RE.search(l)) else f"{l}，"
            for i, l in enumerate(lines)
        )
    else:
        s = lines[0] if lines else ""

    s = re.sub(r"…{2,}|\.{3,}", "，", s)
    s = re.sub(r"[—\-]{2,}", "，", s)
    s = re.sub(r"[（(【\[]", "，", s)
    s = re.sub(r"[）)】\]]", "，", s)
    s = re.sub(r"[，,]{2,}", "，", s)

    if not keep_dunhao:
        def _rep_dunhao(m: re.Match) -> str:
            off = m.start()
            left = re.search(r"[\u4e00-\u9fff]+$", s[:off])
            right = re.match(r"^[\u4e00-\u9fff]+", s[off + 1:])
            if left and right and len(left.group(0)) >= 6 and len(right.group(0)) >= 6:
                return "，"
            return ""
        s = re.sub(r"、", _rep_dunhao, s)

    s = re.sub(r"([：:；;，,])\s*[：:；;，,]+", r"\1", s)
    s = re.sub(r"[：:；;，,]\s*([。？！.?!])", r"\1", s)
    s = re.sub(r"^[，,\s]+|[，,\s]+$", "", s).strip()

    if re.search(r"[\u4e00-\u9fff°℃％%]", s):
        s = re.sub(
            r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
            lambda m: f"{''.join(_ZH_DIGITS[int(c)] for c in m.group(1))}年{_int_to_zh(int(m.group(2)))}月{_int_to_zh(int(m.group(3)))}日",
            s,
        )
        s = re.sub(
            r"([-−])?(\d+(?:\.\d+)?)\s*(?:°\s*([CcFf])|([℃℉]))",
            lambda m: ("零下" if m.group(1) else "")
            + _zh_num_str(m.group(2))
            + ("华氏度" if (m.group(3) or "").upper() == "F" or m.group(4) == "℉" else "摄氏度"),
            s,
        )
        s = re.sub(
            r"([-−])?(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: ("负" if m.group(1) else "") + "百分之" + _zh_num_str(m.group(2)),
            s,
        )
        s = re.sub(
            r"(?<![A-Za-z0-9./\-])(\d+)\s*/\s*(\d+)(?![A-Za-z0-9./\-])",
            lambda m: f"{_zh_num_str(m.group(2))}分之{_zh_num_str(m.group(1))}",
            s,
        )
    return s


def split_zh_clauses(raw_text: str, max_chars: int = 65) -> list[str]:
    norm = normalize_zh_text(raw_text)
    if not norm:
        return []
    boundary_re = re.compile(r"[。？！?!]+|(?<![A-Z0-9])\.(?=\s|$|[”’\"')）】》\u4e00-\u9fff])")
    sents, last = [], 0
    for m in boundary_re.finditer(norm):
        end = m.end()
        tail = re.match(r"[”’\"')）】》]+", norm[end:])
        if tail:
            end += len(tail.group(0))
        piece = norm[last:end].strip()
        if piece:
            sents.append(piece)
        last = end
    if norm[last:].strip():
        sents.append(norm[last:].strip())

    clauses = []
    for sent in sents:
        if len(sent) <= max_chars:
            clauses.append(sent)
            continue
        parts = [p for p in re.split(r"([；;：:，,])", sent) if p]
        cur = ""
        for p in parts:
            if re.fullmatch(r"[；;：:，,]", p):
                cur += p
                if len(cur) >= int(max_chars * 0.55):
                    clauses.append(cur.strip())
                    cur = ""
            elif cur and len(cur) + len(p) > max_chars and len(cur) >= 8:
                clauses.append(cur.strip())
                cur = p
            else:
                cur += p
        if cur.strip():
            clauses.append(cur.strip())
    return clauses


def compress_silence(
    pcm: np.ndarray,
    sr: int = 24000,
    *,
    frame_ms: int = 10,
    threshold: float = 0.008,
    lead_ms: int = 35,
    tail_ms: int = 80,
    min_silence_ms: int = 130,
    max_pause_ms: int = 200,
    target_pause_ms: int = 180,
    fade_ms: int = 8,
) -> np.ndarray:
    """Cap internal silence pauses and trim excess edge silence with cosine crossfade."""
    if pcm.size == 0:
        return pcm
    hop = max(1, int(round(sr * frame_ms / 1000.0)))
    n_frames = len(pcm) // hop
    if n_frames < 6:
        return pcm.astype(np.float32, copy=False)

    frames = pcm[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    is_sil = rms < threshold

    lead_f = max(0, round(lead_ms / frame_ms))
    tail_f = max(0, round(tail_ms / frame_ms))
    min_sil_f = max(1, round(min_silence_ms / frame_ms))
    max_pause_f = max(1, round(max_pause_ms / frame_ms))
    target_f = max(1, min(max_pause_f, round(target_pause_ms / frame_ms)))
    fade_n = min(int(round(sr * fade_ms / 1000.0)), (target_f * hop) // 2)

    chunks: list[np.ndarray] = []
    i = 0
    while i < n_frames:
        sil = bool(is_sil[i])
        j = i + 1
        while j < n_frames and bool(is_sil[j]) == sil:
            j += 1
        run = j - i
        if not sil:
            chunks.append(pcm[i * hop : j * hop])
        elif i == 0:
            keep = min(run, lead_f)
            if keep > 0:
                chunks.append(pcm[(j - keep) * hop : j * hop])
        elif j == n_frames:
            keep = min(run, tail_f)
            if keep > 0:
                chunks.append(pcm[i * hop : (i + keep) * hop])
        elif run > max_pause_f and run >= min_sil_f:
            lf = target_f // 2
            rf = target_f - lf
            left = pcm[i * hop : (i + lf) * hop].copy()
            right = pcm[(j - rf) * hop : j * hop].copy()
            if fade_n > 0 and len(left) >= fade_n and len(right) >= fade_n:
                k = np.arange(1, fade_n + 1, dtype=np.float32)
                w_out = 0.5 * (1.0 + np.cos(math.pi * k / fade_n))
                w_in = 1.0 - w_out
                left[-fade_n:] *= w_out
                right[:fade_n] *= w_in
            chunks.extend((left, right))
        else:
            chunks.append(pcm[i * hop : j * hop])
        i = j

    return np.concatenate(chunks).astype(np.float32, copy=False) if chunks else pcm.astype(np.float32, copy=False)


class OptimizedZhG2P:
    """Patched ZHG2P wrapping misaki: fixes polyphones, removes '/', and tightens punctuation spaces."""

    def __init__(self, en_callable=None, unk: str = "❓"):
        import jieba
        from pypinyin import load_phrases_dict
        from misaki.zh import ZHG2P

        self._base = ZHG2P(version="1.1", en_callable=en_callable, unk=unk)
        # Must run AFTER ZHG2P init because ZHFrontend._init_pypinyin() resets pypinyin's phrase dict
        if getattr(self._base, "frontend", None) is not None:
            self._base.frontend.phrases_dict.update(_POLYPHONE_PHRASES)
        load_phrases_dict(_POLYPHONE_PHRASES)
        for phrase in _POLYPHONE_PHRASES:
            jieba.add_word(phrase, freq=100000)

    def __call__(self, text: str, en_callable=None):
        norm = normalize_zh_text(text)
        ps, toks = self._base(norm, en_callable=en_callable)
        if not ps:
            return "", toks
        # Strip choppy word-boundary '/' and excessive post-punctuation whitespace stalls
        ps = ps.replace("/", "")
        ps = re.sub(r"([,;:.!?])\s+", r"\1", ps)
        ps = re.sub(r"\s+([,;:.!?])", r"\1", ps)
        ps = ps.replace("“ ", "“").replace(" ”", "”")
        return ps.strip(), toks


class OptimizedEnG2P:
    """Wraps misaki.en.G2P with sentence-initial article 'A' normalization."""

    def __init__(self, british: bool = False, trf: bool = False, fallback=None, unk: str = "❓"):
        from misaki import en as misaki_en

        self._base = misaki_en.G2P(trf=trf, british=british, fallback=fallback, unk=unk)

    def __call__(self, text: str):
        return self._base(normalize_en_a(text))


class OptimizedKokoroEngine:
    """High-level inference helper combining optimized Kokoro G2P, voice packs, and PCM silence compression."""

    def __init__(self, device: Optional[str] = None, zh_voices_dir: Path = VOICES_ZH_DIR):
        import torch
        from huggingface_hub import snapshot_download
        from kokoro import KModel, KPipeline
        from misaki import en as misaki_en
        from misaki import espeak as misaki_espeak

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.zh_voices_dir = Path(zh_voices_dir)

        zh_local = Path(snapshot_download("hexgrad/Kokoro-82M-v1.1-zh", revision="01e7505bd6a7a2ac4975463114c3a7650a9f7218"))
        en_local = Path(snapshot_download("hexgrad/Kokoro-82M", revision="f3ff3571791e39611d31c381e3a41a3af07b4987"))
        self.zh_local = zh_local
        self.en_local = en_local

        self.zh_model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", config=str(zh_local / "config.json"),
                               model=str(next(zh_local.glob("*.pth")))).to(self.device).eval()
        self.en_model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(en_local / "config.json"),
                               model=str(next(en_local.glob("*.pth")))).to(self.device).eval()

        try:
            fb_us = misaki_espeak.EspeakFallback(british=False)
        except Exception:
            fb_us = None
        en_g2p_for_zh = misaki_en.G2P(trf=False, british=False, fallback=fb_us, unk="❓")
        self.zh_pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=self.zh_model)
        # Initialize OptimizedZhG2P AFTER KPipeline(lang_code="z") so KPipeline's ZHG2P doesn't reset pypinyin's phrase dict
        self.zh_g2p = OptimizedZhG2P(en_callable=lambda t: en_g2p_for_zh(normalize_en_a(t))[0], unk="❓")
        self.zh_pipe.g2p = self.zh_g2p
        self.en_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=self.en_model)
        self._zh_packs: dict[str, torch.FloatTensor] = {}
        self._en_packs: dict[str, torch.FloatTensor] = {}

    def load_zh_voice(self, voice: str):
        import torch

        if voice not in self._zh_packs:
            opt_pt = self.zh_voices_dir / f"{voice}.pt"
            src = opt_pt if opt_pt.exists() else (self.zh_local / "voices" / f"{voice}.pt")
            self._zh_packs[voice] = torch.load(str(src), map_location="cpu", weights_only=True)
        return self._zh_packs[voice]

    def load_en_voice(self, voice: str):
        import torch

        if voice not in self._en_packs:
            src = self.en_local / "voices" / f"{voice}.pt"
            self._en_packs[voice] = torch.load(str(src), map_location="cpu", weights_only=True)
        return self._en_packs[voice]

    def synth_zh(self, text: str, voice: str = "zf_046", speed: float = 1.05, max_clause_chars: int = 65) -> np.ndarray:
        pack = self.load_zh_voice(voice)
        clauses = split_zh_clauses(text, max_chars=max_clause_chars)
        parts = []
        gap = np.zeros(int(24000 * 0.08), dtype=np.float32)
        for c in clauses:
            ps, _ = self.zh_g2p(c)
            if not ps:
                continue
            for res in self.zh_pipe.generate_from_tokens(ps[:508], pack, speed=speed):
                if res.audio is not None:
                    pcm = compress_silence(res.audio.detach().cpu().numpy(), sr=24000)
                    if parts:
                        parts.append(gap)
                    parts.append(pcm)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def synth_en(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray:
        pack = self.load_en_voice(voice)
        norm = normalize_en_a(text)
        parts = []
        for res in self.en_pipe(norm, voice=pack, speed=speed):
            if res.audio is not None:
                parts.append(compress_silence(res.audio.detach().cpu().numpy(), sr=24000, max_pause_ms=260, target_pause_ms=220))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
