#!/usr/bin/env python3
"""生成本地试听页 work/pilot/listen.html：每段一行，各引擎并排，附评测分数。
任意 Python 3 运行：python make_listen_page.py
"""
from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work" / "pilot"
ORDER = ["kokoro", "voxcpm", "qwen3", "index25", "cosy3"]
LABEL = {"kokoro": "Kokoro 原版（对照）", "voxcpm": "VoxCPM2", "qwen3": "Qwen3-TTS", "index25": "IndexTTS-2.5",
         "cosy3": "CosyVoice3"}


def read_tsv(p: Path):
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l]


def main():
    texts = []
    for p in (HERE / "texts.tsv", WORK / "texts_book.tsv"):
        if p.exists():
            texts += [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    per = {(r["engine"], r["voice"], r["pid"]): r for r in read_tsv(WORK / "eval.tsv")}
    summ = {(r["engine"], r["voice"]): r for r in read_tsv(WORK / "summary.tsv")}
    sim = {(r["engine"], r["voice"]): r for r in read_tsv(WORK / "spk_sim.tsv")}
    out = WORK / "out"
    voices = sorted({p.name for p in out.glob("*/*") if p.is_dir()})
    engines = [e for e in ORDER if (out / e).is_dir()]

    h = ["<!doctype html><meta charset=utf-8><title>老师候选试听</title><style>",
         "body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:16px;background:#fafafa;color:#222}",
         "table{border-collapse:collapse;width:100%;margin-bottom:28px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:13px}",
         "th{background:#eee;position:sticky;top:0}audio{width:220px;height:32px}.m{color:#666;font-size:12px}",
         ".t{max-width:340px}.bad{color:#c00}</style>",
         "<h1>老师候选试听（Kokoro 音色克隆 · 长句朗读）</h1>",
         "<p>指标：cer=错字率↓；断句异常=每百字不在标点处的停顿↓；标点不停=标点处没停顿的比例↓；语速=字/秒；音色相似=与 Kokoro 原声的声纹相似度↑。</p>"]
    for v in voices:
        h.append(f"<h2>{v}</h2><table><tr><th>引擎</th><th>cer</th><th>断句异常</th><th>标点不停</th><th>语速</th><th>音色相似</th></tr>")
        for e in engines:
            s, m = summ.get((e, v), {}), sim.get((e, v), {})
            h.append(f"<tr><td>{LABEL[e]}</td><td>{s.get('cer', '')}</td><td>{s.get('bad_pause', '')}</td>"
                     f"<td>{s.get('miss_pause', '')}</td><td>{s.get('cps', '')}</td><td>{m.get('spk_sim', '')}</td></tr>")
        h.append("</table><table><tr><th>段落</th>" + "".join(f"<th>{LABEL[e]}</th>" for e in engines) + "</tr>")
        for pid, kind, text in texts:
            h.append(f"<tr><td class=t><b>{pid}</b> <span class=m>{kind}</span><br>{html.escape(text)}</td>")
            for e in engines:
                f = out / e / v / f"{pid}.wav"
                if not f.exists():
                    h.append("<td>—</td>")
                    continue
                r = per.get((e, v, pid), {})
                metr = (f"cer {float(r['cer']):.2f} · 断句异常 {float(r['bad_pause']):.1f} · 语速 {float(r['cps']):.1f}"
                        if r else "")
                h.append(f"<td><audio controls preload=none src='out/{e}/{v}/{pid}.wav'></audio><div class=m>{metr}</div></td>")
            h.append("</tr>")
        h.append("</table>")
    (WORK / "listen.html").write_text("\n".join(h), encoding="utf-8")
    print(WORK / "listen.html")
    return h


def per_voice_pages():
    """每个音色单独一页（listen_<voice>.html），页面更小，方便逐个音色挑引擎。"""
    import re as _re
    full = "\n".join(main())
    head, *sections = full.split("<h2>")
    for sec in sections:
        v = sec.split("</h2>", 1)[0]
        p = WORK / f"listen_{_re.sub(r'[^A-Za-z0-9_]', '', v)}.html"
        p.write_text(head + "<h2>" + sec, encoding="utf-8")
        print(p)


if __name__ == "__main__":
    per_voice_pages()
