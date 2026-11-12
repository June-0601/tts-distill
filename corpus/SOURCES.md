# 语料来源与许可

| 来源 | 用在哪 | 许可 |
|---|---|---|
| [Tatoeba](https://tatoeba.org) 英文、中文例句（2026-11-11 导出） | `en.tsv` 的 short / sent / long，`zh.tsv` 的 short / sent / long | CC BY 2.0 FR（署名：Tatoeba 贡献者） |
| [ECDICT](https://github.com/skywind3000/ECDICT) 中文释义 | `zh.tsv` 的 gloss | MIT |
| 本仓库 `web_app/assets/dist/exam.tsv.gz` 考试词表 | `en.tsv` 的 word（单词本身） | 词表自用整理 |

训练用的音频由 [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) 与
[Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)（Apache-2.0）合成。
底模：英文 LJSpeech medium（数据为公有领域），中文 Chaowen medium（数据 CC0，底模来自 Hi-Fi TTS，CC BY 4.0）。
训练代码 [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) 为 GPL-3.0，只在训练机上运行，不进网页。

没有使用柯林斯 COBUILD 的释义（`web_app/assets/dist/cobuild.tsv`）和考试真题原文，它们的许可不允许这样用。
