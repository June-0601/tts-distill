#!/usr/bin/env python3
"""v2 语料：把本地电子书（work/books/book*.txt，epub_text.py 抽出）切成朗读句块，
并混入旧语料 corpus/zh.tsv 里的单词/释义/中英混排/英文部分（保留双语能力）。

书的内容受版权保护，输出只写在 work/books/（gitignore），不进仓库：
  work/books/pool.tsv     训练候选：uid  kind  text      kind = book1|book2|book3|word|gloss|mix|en_word|en_sent|short
  work/books/heldout.tsv  每本书留出的一整章（长文本验收用，不参与训练）

句块规则（让训练分布贴近阅读器实际送进模型的样子）：
  - 按 。！？…（及其后的右引号）断句；连续 1–3 句拼成 15–110 字的块，不跨段落；
  - 超过 110 字的单句在 ；/，处切开，切点之前的部分作为以逗号收尾的块（学“未完”的语调）；
  - 丢掉：版权页/广告/网址、汉字比例 < 60%、阿拉伯数字过多、参考文献式括注（如“（Enloe, 1989）”）。
用法：python -I build_book_corpus.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOKS = HERE / "work" / "books"
OLD = HERE / "corpus" / "zh.tsv"
SEED = 20261124

CJK = re.compile(r"[一-鿿]")
SENT = re.compile(r".*?(?:[。！？]+|…{1,2}|……)[”’」』）)]*", re.S)
JUNK = re.compile(r"https?://|www\.|QQ|微信|公众号|ISBN|CIP|出版社|责任编辑|版权所有|定价|印刷|字数|开本|"
                  r"电子书|小编|下载|书名原文|图书在版编目|ireadweek|z-lib", re.I)
CITE = re.compile(r"[（(][^）)]{0,40}\d{4}[^）)]{0,20}[）)]")
FOOTNOTE = re.compile(r"\[\d+\]|［\d+］|[①-⑳]|〔\d+〕")
MIN_LEN, MAX_LEN = 15, 110


def clean(t: str) -> str:
    t = FOOTNOTE.sub("", t)
    t = re.sub(r"(?<=[。！？”’」』，；：])\d{1,3}(?![\d年月日%％])", "", t)  # 句末尾注编号，如“非正义性。39”
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])", "", t)


def ok(t: str) -> bool:
    if len(t) < MIN_LEN or JUNK.search(t):
        return False
    if len(CJK.findall(t)) / len(t) < 0.6:
        return False
    if len(re.findall(r"\d", t)) > 8 or CITE.search(t):
        return False
    return True


def split_long(s: str) -> list[str]:
    """>MAX_LEN 的单句：在 ；或 ，处切成不超过 MAX_LEN 的几块（切点保留标点）。"""
    out, buf = [], ""
    for piece in re.findall(r"[^；，]*[；，]?", s):
        if not piece:
            continue
        if buf and len(buf) + len(piece) > MAX_LEN:
            out.append(buf)
            buf = ""
        buf += piece
    if buf:
        out.append(buf)
    return out


def chunk_paragraph(p: str, rng: random.Random) -> list[str]:
    sents = [s for s in SENT.findall(p)]
    tail = p[sum(len(s) for s in sents):]
    if tail.strip():
        sents.append(tail)
    units = []
    for s in sents:
        units += split_long(s) if len(s) > MAX_LEN else [s]
    chunks, buf, k = [], "", rng.choice((1, 1, 2, 2, 3))
    n = 0
    for u in units:
        if buf and (len(buf) + len(u) > MAX_LEN or n >= k):
            chunks.append(buf)
            buf, n, k = "", 0, rng.choice((1, 1, 2, 2, 3))
        buf += u
        n += 1
    if buf:
        chunks.append(buf)
    # 太短的块并到前一块
    merged = []
    for c in chunks:
        if merged and len(c) < MIN_LEN and len(merged[-1]) + len(c) <= MAX_LEN:
            merged[-1] += c
        else:
            merged.append(c)
    return merged


def read_chapters(path: Path) -> list[list[str]]:
    chapters, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cur.append(clean(line))
        elif cur:
            chapters.append(cur)
            cur = []
    if cur:
        chapters.append(cur)
    return chapters


def main() -> int:
    rng = random.Random(SEED)
    pool, held = [], []
    for book in sorted(BOOKS.glob("book[0-9].txt")):
        name = book.stem
        chapters = read_chapters(book)
        # 有正文的章：至少 2000 字
        body = [i for i, ch in enumerate(chapters) if sum(len(p) for p in ch) >= 2000]
        hold_idx = body[int(len(body) * 0.6)] if body else -1
        n_ch = 0
        for i, ch in enumerate(chapters):
            for p in ch:
                for c in chunk_paragraph(p, rng):
                    if not ok(c):
                        continue
                    (held if i == hold_idx else pool).append((name, c))
            n_ch += 1
        print(f"{name}: {len(chapters)} 章，留出第 {hold_idx} 章做验收", file=sys.stderr)

    # 去重
    seen, uniq = set(), []
    for k, t in pool:
        if t not in seen:
            seen.add(t)
            uniq.append((k, t))
    pool = uniq

    # 旧语料里保留双语能力相关的部分（Tatoeba 例句 sent/long 由书替代）
    keep_kinds = {"word", "gloss", "short", "mix", "en_word", "en_sent"}
    old = [l.split("\t")[:3] for l in OLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    old = [(k, t) for _uid, k, t in old if k in keep_kinds]

    BOOKS.mkdir(parents=True, exist_ok=True)
    with (BOOKS / "pool.tsv").open("w", encoding="utf-8") as f:
        for i, (k, t) in enumerate(pool):
            f.write(f"bk_{k}_{i:05d}\t{k}\t{t}\n")
        for i, (k, t) in enumerate(old):
            f.write(f"old_{k}_{i:05d}\t{k}\t{t}\n")
    with (BOOKS / "heldout.tsv").open("w", encoding="utf-8") as f:
        for i, (k, t) in enumerate(held):
            f.write(f"ho_{k}_{i:05d}\t{k}\t{t}\n")

    from collections import Counter
    cnt = Counter(k for k, _ in pool)
    chars = Counter()
    for k, t in pool:
        chars[k] += len(t)
    lens = sorted(len(t) for _, t in pool)
    print(f"书句块 {len(pool)}：" + "，".join(f"{k} {cnt[k]} 块/{chars[k]} 字" for k in sorted(cnt)), file=sys.stderr)
    print(f"长度 中位 {lens[len(lens) // 2]}，p10 {lens[len(lens) // 10]}，p90 {lens[len(lens) * 9 // 10]}", file=sys.stderr)
    print(f"旧语料保留 {len(old)} 条；留出验收 {len(held)} 块", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
