# Mac 端 `reading-app` 模型接入说明

本仓库已将全部训练脚本、双语 G2P 前端参考实现与 `.onnx.json` 配置提交至 `main`，并将 3 套导出模型（每个约 73.5 MiB）发布到 GitHub Release **`v1.0-models`**（避免大二进制膨胀 Git 历史，也无需安装 `git-lfs`）。

---

## 一、在 Mac 上拉取全部模型

```bash
git pull origin main
bash download_models.sh
```

脚本会将以下 6 个文件下载到 `models/` 并自动完成 SHA256 校验（也可直接在 [Releases `v1.0-models`](https://github.com/June-0601/tts-distill/releases/tag/v1.0-models) 页面手动下载）：

| 模型文件 | 大小 | 定位 | `espeak.voice` | `phoneme_type` | `speaker_id_map` |
|---|---|---|---|---|---|
| `en_US-kokoro_us_zh_mix-medium.onnx` + `.json` | 73.49 MiB | **美音 6 合 1（英文为主 + 中文为辅）** | `en-us` | `text`（混合） | `af_heart:0`, `af_bella:1`, `af_aoede:2`, `af_jessica:3`, `af_kore:4`, `af_sarah:5` |
| `en_GB-kokoro_uk_zh_mix-medium.onnx` + `.json` | 73.49 MiB | **英音 2 合 1（英文为主 + 中文为辅）** | `en-gb-x-rp` | `text`（混合） | `bf_emma:0`, `bf_isabella:1` |
| `zh_CN-kokoro_zh_multi-medium.onnx` + `.json` | 73.50 MiB | **中文 8 女声（中文为主 + 英文双语）** | `cmn` | `pinyin` | `zf_032:0`, `zf_046:1`, `zf_021:2`, `zf_022:3`, `zf_023:4`, `zf_024:5`, `zf_059:6`, `zf_079:7` |

所有 3 套模型的推荐推理参数已写入对应 `.onnx.json`：`en_US` / `en_GB` 仍为 `{"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.6}`，中文 `zh_multi` 已收紧为 `{"noise_scale": 0.45, "length_scale": 0.96, "noise_w": 0.25}`（抑制中文朗读随机拖音/卡顿），`sample_rate = 22050`，均需传入多说话人输入张量 `sid`（`int64[1]`）。

---

## 二、网页 Worker（`02-tts-piper.js`）前端注音对接要点

### 1. 中文 8 女声（`zh_CN-kokoro_zh_multi-medium.onnx`）
* 直接沿用仓库内升级后的 `pinyin_ids.mjs`（`pinyin-pro 3.29.4` + `assets/cmudict.json` + `assets/g2p_model.json`），音素 ID 范围 `0..115`（中文拼音 `0..72`，英文 ARPAbet `73..115`）。本次已补全温度/百分数/分数/日期/负号规整、无标点换行补停顿、**中英文括号（如 `国际妇女争取和平与自由联盟（Women's ...）`）自动插停顿边界**、连续破折号/省略号去重、短列举顿号减停，以及常用多音字（`还给/换行/数一数/地` 等）词典纠偏。
* **断句与静音压缩辅助（`zh_tts_runtime.mjs`，零依赖纯 JS，可直接搬进 Web Worker）**：
  1. 送入文本前调用 `splitZhClauses(text, 48)` 按句末标点、括号停顿、换行与均衡分句切分长段落，避免单次过长导致韵律失真；
  2. 使用 `ZH_PIPER_INFERENCE_DEFAULTS`（`noiseScale: 0.45, lengthScale: 0.96, noiseW: 0.25`，已写入新版 `.onnx.json`）；
  3. 模型推理出 `Float32Array` 音频后调用 `compressInternalSilence(pcm, 22050)`，以带余弦淡入淡出的 10ms RMS 窗口将句内 >`200ms` 的长停顿压缩至 `~180ms`、首尾静音裁至 `<=35ms / <=80ms`，消除从原 Kokoro 继承的 ~1s 死停顿与句中卡顿。

### 2. 美音 / 英音「英文为主 + 中文为辅」（`en_US-kokoro_us_zh_mix-medium.onnx` / `en_GB-kokoro_uk_zh_mix-medium.onnx`）
* **完整参考实现**：见仓库里的 `en_zh_g2p.py`、`zh_pinyin_tokens.mjs` 与 `corpus/en_zh_phonemes.json`（及测试脚本 `listen_en_zh.py`）。
* **音素槽位（共 232 个，在 `num_symbols=256` 内）**：
  * **`0..165`**：标准 Piper `DEFAULT_PHONEME_ID_MAP`（英文 `espeak-ng` IPA、控制符与英文标点）；
  * **`166..231`**：66 个中文拼音符号，键名带前缀 `py:`（`py:Ø`、23 声母 `py:b..py:w`、37 韵母 `py:a..py:ue`、声调 `py:1..py:5`）；
  * 中文全角标点与括号映射至已有英文标点 ID：`。→.`(10)、`？→?`(13)、`！→!`(4)、`，/、/—/…/（/）/【/】/(/)/[/]→,`(8)、`：→:`(11)、`；→;`(12)（`en_zh_g2p.py` 已做首尾及连续逗号去重）。
* **前段路由逻辑（网页端实现仅需约 40 行）**：
  1. **纯英文快速路径 & 纯英文 Piper 模型（`en_US-kokoro_us_multi-medium` / `en_GB-kokoro_uk_multi-medium`）**：`espeak-ng` 默认会吞掉括号 `(...)` / `[...]` 而不发停顿音素！送入 `espeak-ng` 前请统一调用 `zh_tts_runtime.mjs` 的 `normalizeEnPunctuation(text)`（Python 端已内置在 `en_zh_g2p.py` 的 `normalize_en_text`），将 `professional practice (Beauchamp & Childress, 1994; ...).` 中的括号映射为分号停顿 `; `（实测比逗号更稳定，能在美音/英音多说话人模型上稳定拉开 `0.15s–0.44s` 自然停顿，同时自动处理 `1996).`、`1,000` 与 `author(s)`）；同时先执行 `A/a` 规整（将 `Option A` / `A or B` / `A. ` 中的字母 `A` 单独用 `espeak("A")` 输出 `/ˈeɪ/`），再进入 `espeak-ng` → `phonemes_to_ids`。
  2. **含中文/中英混排路径**：先用 `pinyin-pro`（与 `pinyin_ids.mjs` 相同的 `PINYIN_OPTIONS` 与 `normalizeNumbers` / `splitSyllable`）将文本切分为连续的汉字音节与非汉字片段；每个汉字音节转为 `['py:' + init, 'py:' + fin, 'py:' + tone]`，英文片段走上述 `espeak-ng` 逻辑，中英交界处或括号前后补 `,` / `' '`（ID `3`），中文标点按映射表转换（`，/：/；` 后补 `' '`），最后统一经 `phonemes_to_ids` 封装为 `[1, 0, ..., 2]`。

### 3. 原版 Kokoro 中英文引擎优化套件（`kokoro_opt/`）
* 详见 [kokoro_opt/README.md](kokoro_opt/README.md)：
  * **原版 Kokoro 英文 `A tree / A woman` 修复**：调用 `kokoro_opt/kokoro_pre.mjs` 的 `normalizeKokoroEnText(text)`，将句首/标题中大写的量词/冠词 `A` 规范化为 `a`（`/ə/`），保留 `Option A`、`Grade A`、`A or B`、`A. First` 等字母/序号读法（`/ˈeɪ/`）。
  * **原版 Kokoro 中文断句、括号与怪读音优化**：提供 8 个韵律重调的音色文件 `kokoro_opt/voices_zh/*.pt`（Python）与 `kokoro_opt/voices_zh_bin/*.bin`（`kokoro-js`，100% 保留原 `[:128]` 音色，仅收紧 `[128:]` 停顿与时长韵律）、`kokoro_pre.mjs`（括号停顿 + `。？！` 分句 + 24kHz 静音压缩）与 `kokoro_engine.py`（去除逐词 `/` 硬断 + 多音字纠偏）。

---

## 三、部署提示
* 3 个 `.onnx` 均约 73.5 MiB（超过 Cloudflare Pages 单文件 25MB 上限），请上传到 R2 并在 `02-tts-piper.js` 中按用户选择的音色与语种按需拉取缓存。
