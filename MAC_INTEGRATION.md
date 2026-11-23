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
* 直接沿用仓库内升级后的 `pinyin_ids.mjs`（`pinyin-pro 3.29.4` + `assets/cmudict.json` + `assets/g2p_model.json`），音素 ID 范围 `0..115`（中文拼音 `0..72`，英文 ARPAbet `73..115`）。本次已补全温度/百分数/分数/负号规整、无标点换行补停顿、连续破折号/省略号去重、短列举顿号减停，以及常用多音字（`还给/换行/数一数/地` 等）词典纠偏。
* **断句与静音压缩辅助（`zh_tts_runtime.mjs`，零依赖纯 JS，可直接搬进 Web Worker）**：
  1. 送入文本前调用 `splitZhClauses(text, 48)` 按句末标点、换行与均衡分句切分长段落，避免单次过长导致韵律失真；
  2. 使用 `ZH_PIPER_INFERENCE_DEFAULTS`（`noiseScale: 0.45, lengthScale: 0.96, noiseW: 0.25`，已写入新版 `.onnx.json`）；
  3. 模型推理出 `Float32Array` 音频后调用 `compressInternalSilence(pcm, 22050)`，以带余弦淡入淡出的 10ms RMS 窗口将句内 >`200ms` 的长停顿压缩至 `~180ms`、首尾静音裁至 `<=35ms / <=80ms`，消除从原 Kokoro 继承的 ~1s 死停顿与句中卡顿。

### 2. 美音 / 英音「英文为主 + 中文为辅」（`en_US-kokoro_us_zh_mix-medium.onnx` / `en_GB-kokoro_uk_zh_mix-medium.onnx`）
* **完整参考实现**：见仓库里的 `en_zh_g2p.py`、`zh_pinyin_tokens.mjs` 与 `corpus/en_zh_phonemes.json`（及测试脚本 `listen_en_zh.py`）。
* **音素槽位（共 232 个，在 `num_symbols=256` 内）**：
  * **`0..165`**：标准 Piper `DEFAULT_PHONEME_ID_MAP`（英文 `espeak-ng` IPA、控制符与英文标点）；
  * **`166..231`**：66 个中文拼音符号，键名带前缀 `py:`（`py:Ø`、23 声母 `py:b..py:w`、37 韵母 `py:a..py:ue`、声调 `py:1..py:5`）；
  * 中文全角标点映射至已有英文标点 ID：`。→.`(10)、`？→?`(13)、`！→!`(4)、`，/、/—/…→,`(8)、`：→:`(11)、`；→;`(12)。
* **前段路由逻辑（网页端实现仅需约 40 行）**：
  1. **纯英文快速路径**（不含 `[一-鿿。？！、，：；—…]`）：先执行 `A/a` 规整（将句内换行折叠为空格；把明确字母/序号 `A`，如 `Option A` / `Plan A` / `A or B` / 行首 `A. ` 单独用 `espeak("A")` 输出 `/ˈeɪ/`，避免误弱读成 `/ɐ/`），其余直接走现有 `espeak-ng`（`en-us` 或 `en-gb-x-rp`）→ `phonemes_to_ids`（每个音素后补 PAD `0`，首尾包 BOS `1` / EOS `2`）。
  2. **含中文/中英混排路径**：先用 `pinyin-pro`（与 `pinyin_ids.mjs` 相同的 `PINYIN_OPTIONS` 与 `normalizeNumbers` / `splitSyllable`）将文本切分为连续的汉字音节与非汉字片段；每个汉字音节转为 `['py:' + init, 'py:' + fin, 'py:' + tone]`，英文片段走上述 `espeak-ng` 逻辑，中英交界处补空格 `' '`（ID `3`），中文标点按映射表转换（`，/：/；` 后补 `' '`），最后统一经 `phonemes_to_ids` 封装为 `[1, 0, ..., 2]`。

---

## 三、部署提示
* 3 个 `.onnx` 均约 73.5 MiB（超过 Cloudflare Pages 单文件 25MB 上限），请上传到 R2 并在 `02-tts-piper.js` 中按用户选择的音色与语种按需拉取缓存。
