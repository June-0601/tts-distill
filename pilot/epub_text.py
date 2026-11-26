#!/usr/bin/env python3
"""把 epub 按书脊顺序抽成纯文本（每段一行，章节间空一行）。只用标准库，不执行书里的任何内容。
  python -I epub_text.py <book.epub> <out.txt>
"""
from __future__ import annotations

import html
import posixpath
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

BLOCK = re.compile(r"<(p|h[1-6]|div|li|br)\b[^>]*>", re.I)
TAG = re.compile(r"<[^>]+>")


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(src)
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = next(e.get("full-path") for e in container.iter() if e.tag.endswith("rootfile"))
    opf = ET.fromstring(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    meta = {e.tag.split("}")[1]: (e.text or "") for e in opf.iter() if "}" in e.tag and e.tag.split("}")[1] in ("title", "creator", "language")}
    items = {e.get("id"): e.get("href") for e in opf.iter() if e.tag.endswith("item")}
    spine = [items[e.get("idref")] for e in opf.iter() if e.tag.endswith("itemref")]
    chapters = []
    for href in spine:
        raw = z.read(posixpath.normpath(posixpath.join(base, href))).decode("utf-8", "ignore")
        raw = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", "", raw)
        raw = BLOCK.sub("\n", raw)
        text = html.unescape(TAG.sub("", raw))
        # 连续空白压成一个空格，再去掉紧挨汉字/全角标点的空格（保留英文词间空格）
        paras = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
        paras = [re.sub(r"(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])", "", p) for p in paras]
        paras = [p for p in paras if p]
        if paras:
            chapters.append(paras)
    with open(dst, "w", encoding="utf-8") as f:
        for ch in chapters:
            f.write("\n".join(ch) + "\n\n")
    n = sum(len(p) for ch in chapters for p in ch)
    print(f"title={meta.get('title')} creator={meta.get('creator')} lang={meta.get('language')}")
    print(f"spine={len(spine)} chapters_with_text={len(chapters)} paragraphs={sum(map(len, chapters))} chars={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
