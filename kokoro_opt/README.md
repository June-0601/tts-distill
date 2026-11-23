# 原版 Kokoro 中英文优化套件（`kokoro_opt/`）

无需重训 82M 主干权重，即可从 **前端 G2P + 风格向量半区微调 + 24kHz 静音压缩** 三层同时解决原版 Kokoro 的中文断句、怪读音、中英括号连读以及英文冠词 `A` 误读问题：

## 目录结构

* **`kokoro_pre.mjs`**（纯 ESM，零依赖，直接兼容浏览器 Web Worker / `kokoro-js` 与 Node）：
  * `normalizeKokoroEnText(text)`：将 `A tree`、`A woman`、`Chapter 1\nA woman` 等句首或标题中大写的不定冠词 `A` 规整为 `a`（读作 `/ə/`），同时保留 `Option A`、`Plan A`、`Vitamin A`、`A or B`、`A. First`、`(A)`、`an A`、`A-rated` 等字母序号读法 `/ˈeɪ/`；
  * `normalizeKokoroZhText(text)` & `splitKokoroZhText(text)`：补全温度/百分数/分数/日期规整（消除 `❓`）、把中英文括号 `（...）`/`(...)` 与无标点换行转成明确停顿、去重连续破折号/省略号、并沿 `。？！；\n` 切分中文长段（修复上游 `kokoro/pipeline.py` 只按英文 `.!?` 切片的缺陷）；
  * `compressKokoroSilence(pcm, 24000)`：对 24kHz 输出波形做带 8ms 余弦淡入淡出的静音压缩，把超过 200ms 的内部长停顿压回 `~180ms`。
* **`voices_zh/<voice>.pt`** & **`voices_zh_bin/<voice>.bin`**（共 8 个常用女声，每个 522 KB）：
  * 由 `build_opt_voices.py` 在 GPU 上运行生成：**100% 保留每个音色原有的声学音色半区 `[:128]`**，仅用可微时长对齐与逗号帧上限约束微调 **韵律半区 `[128:]`**，将 `zf_046` / `zf_079` 等原声 `1.2s+` 停顿与句中碎片喘气大幅收敛，同时兼容 Python `kokoro`（`.pt`）与网页 `kokoro-js`（`.bin`）。
* **`kokoro_engine.py`**（Python 端完整封装，`.venv-synth`）：
  * 集成 `OptimizedZhG2P`（增强多音字词典修 `流血/划破/还给/第一行/换行/数一数/慢慢地` 等、去除逐词 `/` 硬断与标点多余空格、支持括号停顿）、`OptimizedEnG2P` 及 `OptimizedKokoroEngine`。

## 用法示例

### 1. Python 端（直接使用优化后的音色与 G2P）
```python
from kokoro_opt.kokoro_engine import OptimizedKokoroEngine
import soundfile as sf

eng = OptimizedKokoroEngine()
zh_pcm = eng.synth_zh("国际妇女争取和平与自由联盟（Women's International League for Peace and Freedom）", voice="zf_046")
en_pcm = eng.synth_en("A woman saw a tall tree. Option A is next to a tree.", voice="af_heart")
sf.write("zh.wav", zh_pcm, 24000)
sf.write("en.wav", en_pcm, 24000)
```

### 2. 网页端（`kokoro-js`）
1. 将 `kokoro_opt/voices_zh_bin/*.bin` 替换部署至中文 Kokoro 音色静态目录；
2. 送入 `kokoro-js` 前调用 `kokoro_pre.mjs` 的 `normalizeKokoroEnText`（英文）或 `splitKokoroZhText`（中文），推理返回 `Float32Array` 后调用 `compressKokoroSilence(pcm, 24000)`；若中文引擎直接从文本走 `misaki`，可在前端文本里将 `（...）` 经过 `normalizeKokoroZhText` 转为停顿边界后再送入。
