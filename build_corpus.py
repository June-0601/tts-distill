#!/usr/bin/env python3
"""生成 Kokoro → Piper 蒸馏用的文本语料（在 Mac 上跑一次，结果提交进仓库）。

输出（都在 corpus/ 下）：
  en.tsv            id<TAB>kind<TAB>text
  zh.tsv            id<TAB>kind<TAB>text<TAB>phoneme_ids（训练直接用这列，与网页注音同一份逻辑）
  zh_phonemes.json  中英双语音素表（训练时 --data.phonemes_path）
  REPORT.md         条数、预计时长、中文被丢弃的原因

kind：
  en  word 单读单词 / short 2–4 词短句 / sent 5–20 词 / long 21–45 词
  zh  word 单读中文词 / gloss 词典释义短语 / short 2–6 字 / sent 7–30 字 / long 31–80 字
      mix 中英混排句 / en_word 英文单读词 / en_sent 英文短句与常句（供中文女声训出双语朗读能力）

来源：Tatoeba 例句（CC BY 2.0 FR）、本仓库 exam.tsv 考试词表、ECDICT 释义（MIT）。
中文每句都和 Kokoro 自己的注音（misaki）逐字比对：多音字读法不一致、或会被 Kokoro
读成儿化的句子一律丢掉，保证老师念的和学生看到的标签是同一个音。

用法（需要 misaki[zh,en]、opencc-python-reimplemented，以及 node + 本目录 npm install）：
  python3 build_corpus.py --tatoeba /path/to/dir   # 目录里放 eng_sentences.tsv.bz2、cmn_sentences.tsv.bz2
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPUS = HERE / "corpus"
EXAM = REPO / "web_app/assets/dist/exam.tsv.gz"
ECDICT = REPO / "web_app/assets/dist/ecdict.tsv.gz"

SEED = 20261111
EN_TARGET = {"word": 2000, "short": 300, "sent": 3000, "long": 700}
ZH_TARGET = {
    "word": 400,
    "gloss": 1200,
    "short": 300,
    "sent": 2600,
    "long": 450,
    "mix": 600,
    "en_word": 300,
    "en_sent": 650,
}
# 粗估每条时长（秒），只用于报告
EN_SEC = {"word": 0.6, "short": 1.2, "sent": 2.6, "long": 7.0}
ZH_SEC = {
    "word": 0.7,
    "gloss": 1.4,
    "short": 1.2,
    "sent": 3.2,
    "long": 8.0,
    "mix": 3.5,
    "en_word": 0.6,
    "en_sent": 2.4,
}

EN_OK = re.compile(r"^[A-Z0-9\"'][A-Za-z0-9 ,.;:!?'\"()\-]*[.!?]\"?$")
ZH_PUNCT = "，。？！、：；“”"
ZH_OK = re.compile(r"^[一-鿿" + ZH_PUNCT + r"]+$")
HAN = re.compile(r"[一-鿿]")

# 保留「著」字的固定词（读 zhù / zhuó），其余港台正体残留的动态助词「著」统一归并为「着」
ZHU_KEEP_WORDS = (
    "著名", "著作", "显著", "名著", "著称", "著书", "卓著", "专著", "原著",
    "编著", "昭著", "土著", "执著", "著录", "著述", "论著", "译著", "遗著", "新著", "巨著", "合著",
)


def fix_traditional_zhu(text: str) -> str:
    """将 opencc 未转干净的港台助词「著」（如 背著、下著雨、哭著）替换为简体「着」。"""
    if "著" not in text:
        return text
    placeholders = {}
    for i, w in enumerate(ZHU_KEEP_WORDS):
        if w in text:
            tok = f"\ue000{i}\ue001"
            placeholders[tok] = w
            text = text.replace(w, tok)
    text = text.replace("著", "着")
    for tok, w in placeholders.items():
        text = text.replace(tok, w)
    return text


def weighted_pick(items, k, weight, rng):
    """无放回加权抽样（Efraimidis–Spirakis）。"""
    keyed = [(rng.random() ** (1.0 / max(weight(x), 1e-6)), i) for i, x in enumerate(items)]
    keyed.sort(reverse=True)
    return [items[i] for _, i in keyed[:k]]


# ----------------------------------------------------------------------------- 词表
def load_exam_words():
    """exam.tsv：word, ipa, 中文, 英文释义, 例句, 考试标签（空格分隔）…"""
    words = {}
    with gzip.open(EXAM, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            word, tags = cols[0].strip(), set(cols[5].split())
            if re.fullmatch(r"[a-z]{2,}(-[a-z]+)?", word):
                words[word] = tags
    return words


def word_level(tags):
    if tags & {"gk", "cet4"}:
        return "basic"
    if tags & {"cet6", "ky"}:
        return "mid"
    if tags & {"ielts", "toefl"}:
        return "adv"
    return "gre"


# ----------------------------------------------------------------------------- 英文
def load_en_pools(tatoeba: Path):
    pools = {"short": [], "sent": [], "long": []}
    seen = set()
    with bz2.open(tatoeba / "eng_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").split("\t")[-1].strip()
            text = re.sub(r"\s+", " ", text).replace("’", "'").replace("“", '"').replace("”", '"')
            if not EN_OK.match(text) or "..." in text or text.count('"') % 2:
                continue
            key = re.sub(r"[^a-z0-9 ]", "", text.lower())
            if key in seen:
                continue
            seen.add(key)
            n = len(text.split())
            kind = "short" if 2 <= n <= 4 else "sent" if 5 <= n <= 20 else "long" if 21 <= n <= 45 else None
            if kind:
                pools[kind].append(text)
    return pools


def build_en(tatoeba: Path, exam_words, rng, pools=None):
    advanced = {w for w, t in exam_words.items() if word_level(t) in ("mid", "adv", "gre")}
    if pools is None:
        pools = load_en_pools(tatoeba)

    def weight(text):
        w = 1.0
        toks = set(re.findall(r"[a-z]+", text.lower()))
        if toks & advanced:
            w *= 2.5                       # 偏向含考试词的句子
        if re.search(r"\b(Tom|Mary)\b", text):
            w *= 0.08                      # Tatoeba 里四分之一的句子都是 Tom/Mary
        if text.endswith(("?", '?"')):
            w *= 1.3
        if text.endswith(("!", '!"')):
            w *= 1.2
        if re.search(r"\d", text):
            w *= 1.5                       # 数字、年份、时间要有一些
        return w

    rows = []
    for kind in ("short", "sent", "long"):
        for text in weighted_pick(pools[kind], EN_TARGET[kind], weight, rng):
            rows.append((kind, text))

    # 单读单词：按难度分层抽，词卡最常用的基础词多一些
    by_level = {"basic": [], "mid": [], "adv": [], "gre": []}
    for w, tags in sorted(exam_words.items()):
        by_level[word_level(tags)].append(w)
    share = {"basic": 0.35, "mid": 0.30, "adv": 0.20, "gre": 0.15}
    for level, frac in share.items():
        k = round(EN_TARGET["word"] * frac)
        for w in rng.sample(by_level[level], min(k, len(by_level[level]))):
            rows.append(("word", w))

    rng.shuffle(rows)
    counters = Counter()
    out = []
    for kind, text in rows:
        counters[kind] += 1
        out.append((f"en_{kind}_{counters[kind]:05d}", kind, text))
    return out


# ----------------------------------------------------------------------------- 中文与中英双语
def clean_gloss(raw: str):
    """ECDICT 释义 → (单读词候选 2~4 字, 1~3 个义项的短语)。"""
    raw = raw.replace("\\r", "\n").replace("\\n", "\n").replace("\r", "\n")
    single_word = None
    multi_gloss = None
    for seg in re.split(r"\n|/", raw):
        seg = re.sub(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|<[^>]*>", "", seg)
        seg = re.sub(r"^\s*([a-z]{1,6}\.\s*)+", "", seg.strip())
        parts = [fix_traditional_zhu(p.strip()) for p in re.split(r"[;；,，]", seg) if p.strip()]
        parts = [p for p in parts if re.fullmatch(r"[一-鿿]{2,8}", p)]
        if not parts:
            continue
        for p in parts:
            if single_word is None and 2 <= len(p) <= 4 and not p.endswith("的"):
                single_word = p
        if multi_gloss is None and len(parts) >= 2:
            text = "，".join(parts[:3])
            if 4 <= len(text) <= 22:
                multi_gloss = text
        elif multi_gloss is None and len(parts) == 1 and 2 <= len(parts[0]) <= 8:
            multi_gloss = parts[0]
    return single_word, multi_gloss


# 40 个多样化的双语查词/释义句式（英文词出现在句首、句中、逗号前、句尾附近等不同韵律位置）
MIX_TEMPLATES = [
    "单词 {w} 的意思是{g}。",
    "在阅读中遇到 {w}，通常表示{g}。",
    "这句话里的 {w} 可以理解为{g}。",
    "词卡上的 {w}，最常见的含义是{g}。",
    "想要表达{g}的时候，可以用 {w} 这个词。",
    "她把 {w} 记在笔记本上，备注是{g}。",
    "今天复习的词汇是 {w}，也就是{g}。",
    "在英语文章里，{w} 常用来形容或表示{g}。",
    "别忘了 {w} 这个词，它的核心意思是{g}。",
    "上下文里的 {w} 指的是{g}，读起来很自然。",
    "遇到生词 {w} 不用慌，它其实就是{g}。",
    "段落开头的 {w}，在这里翻译成{g}最合适。",
    "老师提醒我们，{w} 在这段话里表示{g}。",
    "字典里把 {w} 解释为{g}。",
    "如果上下文提到 {w}，多半是在说{g}。",
    "这本小说里多次出现 {w}，意思是{g}。",
    "书页边上写着 {w} 的注释：{g}。",
    "朗读到 {w} 的时候稍微停顿一下，它表示{g}。",
    "我们可以用 {w} 来代替普通说法，表示{g}。",
    "考试里常考的 {w}，对应的中文意思是{g}。",
    "你刚查过的 {w}，意思就是{g}。",
    "听力材料里出现的 {w}，准确含义是{g}。",
    "作者在这里用 {w} 来强调{g}。",
    "虽然 {w} 看起来有点长，但意思只是{g}。",
    "把这个句子里的 {w} 换成中文，就是{g}。",
]

# 日常与阅读场景的自然中英混排句（涵盖常见名词、动词、短语在不同句法位置的平滑切换）
DAILY_MIX_SENTENCES = [
    "她每天早上都会喝一杯 coffee，然后再打开电脑开始工作。",
    "我们需要在周五的 deadline 之前把这份报告整理好。",
    "今天下午三点有一个线上 meeting，记得提前测试麦克风。",
    "这本英文小说的第一章读起来很顺畅，vocabulary 难度刚刚好。",
    "出门旅行的时候，我习惯在手机里存几期喜欢的 podcast 路上听。",
    "如果遇到不认识的单词，直接点击屏幕就能看到 context 里的释义。",
    "他在晚饭后花了半个小时复习今天的 flashcards，效果非常明显。",
    "为了保护视力，晚上看书的时候可以把界面切换成 dark mode。",
    "这个新版本的 browser 运行速度很快，打开长文章也不会卡顿。",
    "请把修改后的文档发到我的 email，我今天晚上会仔细看一遍。",
    "她在周末制定了一份详细的 schedule，把阅读和运动都安排得很合理。",
    "即使在没有网络的飞机上，offline 语音包依然可以流畅地朗读整本书。",
    "团队在讨论之后收到不少有价值的 feedback，决定立刻改进设计。",
    "这段代码的 logic 非常清晰，新来的同事只用十分钟就看懂了。",
    "坚持每天阅读英文原版书，是最容易保持的 language 学习习惯。",
    "把常用的词典放进本地 cache 之后，查询生词几乎不需要等待。",
    "我们在设计这个 interface 的时候，把所有的按钮都放到了顺手的位置。",
    "昨晚的雨下得很大，但他还是按时完成了日常的 workout 计划。",
    "她很喜欢这种简约的 style，既没有多余的装饰，看着也很舒服。",
    "在开始新的 project 之前，先把目标和步骤写在纸上会更有条理。",
    "这台旧电脑的 memory 已经不太够用了，同时开几个程序就会变慢。",
    "只要打开自动 backup 功能，换新手机的时候就不会丢失任何阅读进度。",
    "他点了一份蔬菜 salad 和一杯热茶，坐在窗边安静地看了一下午书。",
    "好的 user experience 往往体现在那些平时不容易察觉的小细节里。",
    "如果觉得默认语速太慢，可以在 setting 页面里把倍速调高一点。",
    "教授在今天的 lecture 上推荐了三本非常经典的历史著作。",
    "趁着周末天气晴朗，她们约好去公园野餐，顺便聊聊最近读的 novel。",
    "这个小小的 icon 代表书签功能，轻轻点一下就能记住当前读到的位置。",
    "在安静的 library 里待上一整天，能让人完全沉浸在书本的世界里。",
    "他习惯用不同颜色的标记来区分重点，这样做 review 的时候一目了然。",
    "每次读完一个完整的 chapter，我都会合上书本回想一下刚才的情节。",
    "学会跳出自己的 comfort zone，才能在新的领域里收获更多成长。",
    "这个问题其实只需要一点 common sense 就能想明白，不用想得太复杂。",
    "双方经过几轮友好的沟通，终于在 budget 和时间安排上达成了共识。",
    "她对自己的毕业 thesis 要求很高，光参考文献就整理了厚厚一摞。",
    "清晨的阳光透过窗帘照在桌子上，正是集中 focus 读书的好时候。",
    "我们在本地 database 里保存了上万条精选例句，随时都可以检索。",
    "良好的日常 routine 能帮我们节省精力，把时间留给真正重要的事情。",
    "这套全新的 algorithm 不仅节省了电量，还让声音听起来更加自然。",
    "他顺利通过了最后一轮 interview，下个月就要去新的城市工作了。",
    "读书并不是为了比赛，找到适合自己的 pace 才是最开心的方式。",
    "请在登录页面输入正确的 password，然后再同步云端的书架数据。",
    "她把旅行拍下的所有 photo 都按月份整理好，做成了漂亮的相册。",
    "在长途飞行之后，他花了两天才慢慢倒过 jet lag 恢复正常作息。",
    "只要保持充足的 energy 和好奇心，每天都能学到一点新东西。",
    "这篇文章的 summary 写得非常精炼，三句话就概括了作者的核心观点。",
    "我们应该在工作和生活之间找到平衡，留出一点属于自己的 balance。",
    "遇到复杂的长难句时，先理清主干的 grammar 结构就会容易很多。",
    "他在笔记本的第一个 folder 里放满了平时收集的优美英文段落。",
    "这家小店烘焙的 chocolate 蛋糕味道很浓郁，配苦咖啡刚刚好。",
]


def build_zh_candidates(tatoeba: Path, exam_words, en_pools, rng):
    import opencc

    # 使用 tw2s 将港台正体词汇与助词「著」转为规范简体
    t2s = opencc.OpenCC("tw2s")
    pools = {"short": [], "sent": [], "long": []}
    seen = set()
    with bz2.open(tatoeba / "cmn_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            text = t2s.convert(line.rstrip("\n").split("\t")[-1].strip())
            text = fix_traditional_zhu(text)
            text = text.replace(" ", "").replace(",", "，").replace("?", "？").replace("!", "！")
            text = text.replace("．", "。").replace("「", "“").replace("」", "”")
            if not ZH_OK.match(text) or text in seen:
                continue
            seen.add(text)
            n = len(HAN.findall(text))
            kind = "short" if 2 <= n <= 6 else "sent" if 7 <= n <= 30 else "long" if 31 <= n <= 80 else None
            if kind:
                pools[kind].append(text)

    def weight(text):
        w = 1.0
        if text.endswith("？"):
            w *= 1.3
        if text.endswith("！"):
            w *= 1.2
        if "汤姆" in text or "玛丽" in text:
            w *= 0.15
        return w

    rows = []
    # 预采样留足余量（长句含多音字概率高，取 1.75 倍保证过滤后稳达 450 条）
    for kind in ("short", "sent", "long"):
        mult = 1.75 if kind == "long" else 1.45
        k = int(ZH_TARGET[kind] * mult)
        rows += [(kind, t) for t in weighted_pick(pools[kind], k, weight, rng)]

    single_words = []
    glosses = []
    bilingual_pairs = []
    with gzip.open(ECDICT, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3 and cols[0] in exam_words:
                w = cols[0]
                sw, mg = clean_gloss(cols[2])
                if sw and sw not in seen:
                    seen.add(sw)
                    single_words.append(sw)
                if mg and mg not in seen:
                    seen.add(mg)
                    glosses.append(mg)
                if mg and 2 <= len(mg) <= 10 and re.fullmatch(r"[a-z]{3,12}", w):
                    bilingual_pairs.append((w, mg))

    rng.shuffle(single_words)
    rng.shuffle(glosses)
    rng.shuffle(bilingual_pairs)
    rows += [("word", w) for w in single_words[: int(ZH_TARGET["word"] * 1.45)]]
    rows += [("gloss", g) for g in glosses[: int(ZH_TARGET["gloss"] * 1.45)]]

    # 中英混排句（mix）：日常混排句 + 多模板词汇导读双语句
    mix_candidates = list(DAILY_MIX_SENTENCES)
    for i, (eng_w, zh_g) in enumerate(bilingual_pairs[:900]):
        tpl = MIX_TEMPLATES[i % len(MIX_TEMPLATES)]
        mix_candidates.append(tpl.format(w=eng_w, g=zh_g))
    rng.shuffle(mix_candidates)
    # 让日常自然混排句优先排在前面，再补足模板双语句到 600 条
    ordered_mix = list(DAILY_MIX_SENTENCES) + [s for s in mix_candidates if s not in DAILY_MIX_SENTENCES]
    rows += [("mix", s) for s in ordered_mix[: int(ZH_TARGET["mix"] * 1.5)]]

    # 纯英文单读词（en_word）与纯英文句子（en_sent），供中文女声训出完整双语朗读能力
    by_level = {"basic": [], "mid": [], "adv": [], "gre": []}
    for w, tags in sorted(exam_words.items()):
        if re.fullmatch(r"[a-z]{3,12}", w):
            by_level[word_level(tags)].append(w)
    en_word_cands = (
        rng.sample(by_level["basic"], min(180, len(by_level["basic"])))
        + rng.sample(by_level["mid"], min(140, len(by_level["mid"])))
        + rng.sample(by_level["adv"], min(80, len(by_level["adv"])))
        + rng.sample(by_level["gre"], min(50, len(by_level["gre"])))
    )
    rng.shuffle(en_word_cands)
    rows += [("en_word", w) for w in en_word_cands]

    en_sent_cands = (
        rng.sample(en_pools["short"], min(220, len(en_pools["short"])))
        + rng.sample(en_pools["sent"], min(700, len(en_pools["sent"])))
    )
    rng.shuffle(en_sent_cands)
    rows += [("en_sent", s) for s in en_sent_cands]
    return rows


def misaki_checker():
    """返回 check(text) -> (逐字拼音列表, 是否会被读成儿化, 英文是否含未知词❓)，与 Kokoro v1.1-zh 的注音一致。"""
    import cn2an
    import jieba.posseg as psg
    from misaki import en as misaki_en
    from misaki.zh import ZHG2P
    from misaki.zh_frontend import ZHFrontend

    en_g2p = misaki_en.G2P(trf=False, british=False, fallback=None, unk="❓")
    zh_g2p = ZHG2P(version="1.1", en_callable=lambda t: en_g2p(t)[0], unk="❓")
    fe = ZHFrontend(unk="❓")
    zero = {"i": "yi", "in": "yin", "ing": "ying", "u": "wu", "v": "yu", "ve": "yue", "van": "yuan",
            "vn": "yun", "iou": "you", "uei": "wei", "uen": "wen", "ueng": "weng"}

    def written(c, v):
        tone = v[-1] if v[-1:].isdigit() else "5"
        f = v[:-1] if v[-1:].isdigit() else v
        f = {"ii": "i", "iii": "i"}.get(f, f)
        if not c:
            if f in zero:
                return zero[f] + tone
            if f.startswith("i"):
                return "y" + f[1:] + tone
            if f.startswith("u"):
                return "w" + f[1:] + tone
            return f + tone
        f = {"iou": "iu", "uei": "ui", "uen": "un"}.get(f, f)
        if c in ("j", "q", "x") and f.startswith("v"):
            f = "u" + f[1:]
        return c + f + tone

    def check(text):
        has_zh = bool(HAN.search(text))
        if not has_zh:
            ph, tk = en_g2p(text)
            has_unk = ("❓" in ph) or any(not t.phonemes and any(c.isalpha() for c in t.text) for t in tk)
            return [], False, has_unk

        # 只提取中文片段做汉字注音对齐，整句再过一遍 zh_g2p 检查儿化与英文 ❓
        zh_only = "".join(re.findall(r"[一-鿿，。？！、：；“”]+", cn2an.transform(text, "an2cn")))
        norm = ZHG2P.map_punctuation(zh_only)
        seg = fe.tone_modifier.pre_merge_for_modify(psg.lcut(norm))
        sylls = []
        for word, _pos in seg:
            if not HAN.search(word):
                continue
            initials, finals = fe._get_initials_finals(word)
            sylls += [written(c, v) for c, v in zip(initials, finals)]
        zh_ph, _ = fe(norm)
        full_ph, _ = zh_g2p(text)
        has_erhua = "R" in zh_ph
        has_unk = "❓" in full_ph
        return sylls, has_erhua, has_unk

    return check


def build_zh(tatoeba: Path, exam_words, en_pools, rng):
    rows = build_zh_candidates(tatoeba, exam_words, en_pools, rng)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.tsv", Path(tmp) / "out.jsonl"
        src.write_text("".join(f"c{i}\t{k}\t{t}\n" for i, (k, t) in enumerate(rows)), encoding="utf-8")
        subprocess.run(["node", str(HERE / "pinyin_ids.mjs"), str(src), str(dst)], check=True)
        labels = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines() if line]

    check = misaki_checker()
    dropped = Counter()
    examples = {}
    tolerated = Counter()
    kept = {k: [] for k in ZH_TARGET}
    for (kind, text), lab in zip(rows, labels):
        if len(kept[kind]) >= ZH_TARGET[kind]:
            continue
        reason = None
        if lab["bad"]:
            reason = "无法拆分的音节或英文生词"
        else:
            sylls, erhua, has_unk = check(text)
            hanzi = HAN.findall(text)
            if has_unk:
                reason = "Kokoro 含未知词❓"
            elif erhua:
                reason = "Kokoro 会读成儿化"
            elif len(sylls) != len(lab["plain"]):
                reason = "注音长度对不上"
            else:
                for ch, a, b in zip(hanzi, sylls, lab["plain"]):
                    if a == b:
                        continue
                    if a[:-1] != b[:-1]:
                        reason = "多音字读音不同（声韵母不同）"
                    elif ch in "一不":
                        tolerated["一/不变调"] += 1    # 两边都做一/不变调，标签与老师读音一致
                        continue
                    elif "5" in (a[-1], b[-1]):
                        tolerated["轻声标注不同"] += 1  # 同一个词每次都这样，学生能从音频学会读轻声
                        continue
                    else:
                        reason = "多音字声调不同"
                    examples.setdefault(reason, f"{text}（{ch}：Kokoro {a} / 网页 {b}）")
                    break
        if reason:
            dropped[reason] += 1
            continue
        kept[kind].append((text, " ".join(map(str, lab["ids"]))))
    dropped.update({f"（保留）{k}": n for k, n in tolerated.items()})

    out = []
    for kind, items in kept.items():
        for i, (text, ids) in enumerate(items, 1):
            out.append((f"zh_{kind}_{i:05d}", kind, text, ids))
    rng.shuffle(out)
    return out, dropped, examples


def format_en_report(en_rows) -> list[str]:
    c = Counter(k for _, k, _ in en_rows)
    hours = sum(EN_SEC.get(k, 2.5) * n for k, n in c.items()) / 3600
    return [
        "## 英文 en.tsv",
        "",
        *(f"- {k}: {c[k]}" for k in EN_TARGET if c[k]),
        f"- 合计 {len(en_rows)} 条，预计约 {hours:.1f} 小时音频",
        "",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tatoeba", required=True, type=Path, help="放 eng/cmn_sentences.tsv.bz2 的目录")
    ap.add_argument("--only", choices=["en", "zh"], help="只生成一种语言")
    args = ap.parse_args()

    CORPUS.mkdir(exist_ok=True)
    exam_words = load_exam_words()
    en_pools = load_en_pools(args.tatoeba)
    report = ["# 语料报告", "", f"随机种子 {SEED}；来源见 SOURCES.md。", ""]

    if args.only in (None, "en"):
        en = build_en(args.tatoeba, exam_words, random.Random(SEED), pools=en_pools)
        (CORPUS / "en.tsv").write_text("".join(f"{i}\t{k}\t{t}\n" for i, k, t in en), encoding="utf-8")
        c = Counter(k for _, k, _ in en)
        hours = sum(EN_SEC[k] * n for k, n in c.items()) / 3600
        report += format_en_report(en)
        print(f"en: {len(en)} 条 {dict(c)}，约 {hours:.1f} h")
    elif (CORPUS / "en.tsv").exists():
        en_existing = [line.split("\t")[:3] for line in (CORPUS / "en.tsv").read_text(encoding="utf-8").splitlines() if line]
        report += format_en_report(en_existing)

    if args.only in (None, "zh"):
        zh, dropped, examples = build_zh(args.tatoeba, exam_words, en_pools, random.Random(SEED))
        (CORPUS / "zh.tsv").write_text("".join(f"{i}\t{k}\t{t}\t{ids}\n" for i, k, t, ids in zh), encoding="utf-8")
        node = "import('./pinyin_ids.mjs').then(m=>console.log(JSON.stringify(m.PHONEME_TO_ID)))"
        phonemes = subprocess.run(["node", "-e", node], cwd=HERE, check=True, capture_output=True, text=True).stdout
        (CORPUS / "zh_phonemes.json").write_text(
            json.dumps(json.loads(phonemes), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        c = Counter(k for _, k, _, _ in zh)
        hours = sum(ZH_SEC[k] * n for k, n in c.items()) / 3600
        report += [
            "## 中文（中英双语）zh.tsv",
            "",
            *(f"- {k}: {c[k]}" for k in ZH_TARGET),
            f"- 合计 {len(zh)} 条，预计约 {hours:.1f} 小时音频",
            "",
            "与 Kokoro 注音对比后丢弃（只统计抽样到的候选）：",
            "",
            *(f"- {r}: {n}" for r, n in dropped.most_common()),
            *(f"- 例：{e}" for e in examples.values()),
            "",
        ]
        print(f"zh: {len(zh)} 条 {dict(c)}，约 {hours:.1f} h；丢弃 {dict(dropped)}")

    (CORPUS / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
