#!/usr/bin/env python3
"""从书的纯文本里挑试验段落（偏向括号英文、数字、长句），写到 work/pilot/texts_book.tsv。
书的内容只留在本地 work/（已 gitignore），不进仓库。
  python -I book_pilot_texts.py <book.txt> [N]
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work" / "pilot" / "texts_book.tsv"
SENT = re.compile(r"[^。！？]*[。！？]+[”」』]?")
CJK = re.compile(r"[一-鿿]")


def main() -> int:
    src = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paras = [p for p in src.read_text(encoding="utf-8").splitlines() if len(p) > 80]
    paras = [p for p in paras if len(CJK.findall(p)) / len(p) > 0.75]
    # 每段取开头 1–3 句，拼成 60–150 字的块
    cands = []
    for p in paras:
        ss = SENT.findall(p)
        buf = ""
        for s in ss:
            if len(buf) + len(s) > 150:
                break
            buf += s
            if len(buf) >= 60:
                cands.append(buf)
                buf = ""
    rng = random.Random(20261123)
    tricky = [c for c in cands if re.search(r"（[A-Za-z]", c) or re.search(r"\d", c)]
    plain = [c for c in cands if c not in tricky]
    rng.shuffle(tricky)
    rng.shuffle(plain)
    pick = tricky[: n // 2] + plain[: n - n // 2]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, t in enumerate(pick, 1):
            f.write(f"b{i:02d}\tbook\t{t}\n")
    print(f"{len(cands)} candidates, wrote {len(pick)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
