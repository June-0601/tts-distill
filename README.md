# Kokoro → Piper 音色蒸馏（运行手册）

把 Kokoro 的英文 **af_heart**、中文 **zf_001** 音色训练进 Piper medium 模型（约 60MB）。
Piper 在 iPhone 网页上能流畅运行，网页版 Kokoro 跑不动；训好后替换网页里试用的 Kokoro Heart。

做法：Kokoro 把一批文本念成音频（老师），Whisper 回听把念错的去掉，再用这些音频微调 Piper（学生）。
这个文件夹自成一体，拷到训练机上就能跑；只有 `build_corpus.py` 依赖仓库里的词表，已经在 Mac 上跑好了。

## 文件

| 文件 | 作用 | 在哪跑 |
|---|---|---|
| `corpus/en.tsv`、`corpus/zh.tsv` | 训练文本（已生成）：英文 6000 条（约 4 小时）、中文（含中英双语）6500 条（约 5 小时） | — |
| `corpus/REPORT.md`、`SOURCES.md` | 语料统计、来源与许可 | — |
| `build_corpus.py` | 重新生成语料（一般不用再跑） | Mac |
| `pinyin_ids.mjs` | 中英双语 → Piper 拼音 + 英文 ARPAbet 音素 ID；**网页朗读要用同一份** | 任意 |
| `setup_wsl.sh` | 一次性装好环境、下载底模 | WSL |
| `synth.py` | Kokoro 合成训练音频 | WSL（.venv-synth） |
| `qa.py` | Whisper 回听质检，写出训练清单 | WSL（.venv-synth） |
| `train.sh` | 训练 / 断点续训 | WSL（.venv-piper） |
| `export.sh` | 导出 ONNX | WSL（.venv-piper） |
| `listen.py` | 合成固定试听句并让 Whisper 打分，挑 checkpoint | WSL（.venv-piper） |

## 一、Windows 上的一次性准备

1. **NVIDIA 驱动**装最新版（560 以上）。WSL 里不要另装显卡驱动，Windows 的驱动会自动带进去。
2. **装 WSL2 + Ubuntu**：管理员 PowerShell 运行 `wsl --install -d Ubuntu-24.04`，重启后按提示建用户。
3. **给 WSL 足够内存**：在 `C:\Users\<你>\.wslconfig` 写入下面两行（按电脑内存调整，留几 GB 给 Windows），
   然后 PowerShell 里 `wsl --shutdown`：
   ```
   [wsl2]
   memory=24GB
   ```
4. **防止训练被打断**：电源设置里「睡眠」改为「从不」；Windows 更新设为暂停几周，或把「使用时段」设成全天。
5. **把这个文件夹放进 WSL 自己的目录**（放在 `/mnt/c/...` 下读写会慢很多）。两种方式任选：
   - 拷贝：在 Mac 上用 `git archive HEAD tools/tts-distill | gzip > tts-distill.tar.gz` 打包（约 2MB），
     经网盘或 U 盘拷到 Windows，在 Ubuntu 里 `tar xzf /mnt/c/Users/<你>/Downloads/tts-distill.tar.gz -C ~`。
     以后脚本有改动就重新拷一次。
   - git：在 WSL 里装好 `tc` 并登录（与 Mac 同一账号），用 `tc apps repository-credentials` 换临时凭据
     clone 私有仓库；以后 `git pull` 即可同步。
6. **磁盘**：WSL 所在的盘至少留 50GB（两套音频与缓存约 10GB，checkpoint 每个约 850MB、每种语言最多存 11 个）。

然后在 Ubuntu 里：

```bash
cd ~/tools/tts-distill        # 或 clone 下来的 ~/june_zhanread/tools/tts-distill
bash setup_wsl.sh             # 约 20–40 分钟，大部分是下载
```

## 二、合成与质检（每种语言约半小时）

```bash
source .venv-synth/bin/activate
python synth.py en            # Kokoro af_heart；中途中断重跑会接着做
python qa.py en               # Whisper small.en 回听，生成 work/en/metadata.csv
python synth.py zh            # Kokoro zf_001；想换音色：--voice zf_094
python qa.py zh               # Whisper large-v3-turbo（首次下载约 1.6GB）
deactivate
```

`qa.py` 最后会打印保留了多少条、丢弃原因。丢弃超过一成就先停下来看看 `work/<lang>/qa.tsv`。

## 三、训练（每种语言约 1.5–2 天）

```bash
source .venv-piper/bin/activate
./train.sh en                 # 先英文；两种语言不要同时训，显存只够一个
```

- **估时**：训练开始几分钟后日志里会出现每轮（epoch）用时，乘以 1000 轮就是总时长。
  4070 上预计每轮 2–3 分钟。
- **看进度**：另开一个 Ubuntu 窗口运行
  `source .venv-piper/bin/activate && tensorboard --logdir work/runs/en`，
  在 Windows 浏览器打开 http://localhost:6006 ：`val_mel` 应缓慢下降，AUDIO 页可以直接听测试句。
- **中断了**：`./train.sh en --resume` 从最近的 `last.ckpt` 接着训。环境变量 `EPOCHS` 不要改，
  学习率是按总轮数规划的。
- **显存不够**（报 CUDA out of memory）：`BATCH=12 ./train.sh en`，还不够就 `BATCH=8`。
- **想快一点**：可以试 `PRECISION=bf16-mixed`，但还没验证过，出现 nan 就换回默认值。
- 英文训完再 `./train.sh zh`。

## 四、挑 checkpoint 并导出

checkpoint 在 `work/runs/<lang>/lightning_logs/version_*/checkpoints/`：
`last.ckpt` 是最后一轮；文件名带 `val_mos=` 的是自动打分最高的几个，带 `val_mel=` 的是频谱误差最低的几个。
官方经验是最后一轮通常和打分最高的差不多，两个都导出听一听：

```bash
./export.sh en                                   # 导出 last.ckpt
python listen.py en work/export/en_US-kokoro_heart-medium.onnx
./export.sh en work/runs/en/lightning_logs/version_0/checkpoints/<val_mos 最高的>.ckpt
python listen.py en work/export/en_US-kokoro_heart-medium.onnx
```

`listen.py` 会把固定试听句（包括 ubiquitous、although 等单读的词）写到 `work/export/listen-en/`，
在 Windows 里打开 `\\wsl$\Ubuntu-24.04\home\<你>\...` 就能直接听。

## 五、交回网页

把 `work/export/` 里的 `.onnx` 和 `.onnx.json` 拷回 Mac，交给 Claude 接入网页：
- 模型放 R2（Pages 单个文件上限 25MB，大陆访问不了 Hugging Face）；
- 英文照现在 Kokoro Heart 的路子，在 `02-tts-piper.js` 的 `LOCAL_EXTRA` 登记；
- 中文要把 `pinyin_ids.mjs` 的 `pinyinTokens` / `tokensToIds` 搬进网页 Worker，替换现在 Xiao Ya 的注音
  （自带数字转汉字/英文、零声母补 `Ø`、以及复用 `/offline-tts/assets/cmudict.json` + `g2p_model.json` 的英文注音，使中文女声具备中英双语朗读能力）。
- 接入后照例用 Whisper 在页面里回听验收，再部署。

## 语料是怎么来的

`build_corpus.py`（Mac 上跑过，结果已提交）：
- 英文：Tatoeba 例句按长度分三档抽样（含考试词的句子权重更高，Tom/Mary 降权），
  加 2000 个按难度分层抽取的单读单词（专门解决现成模型单词读不准的问题）。
- 中文（中英双语）：Tatoeba 例句（`OpenCC("tw2s")` 转简体并修正港台助词「著→着」）、ECDICT 单读中文词与多义项释义短语、
  600 条中英混排句（日常混排 + 双语词卡导读句）、以及 950 条纯英文单词与句子（使中文女声同时支持中文、中英混排和纯英文朗读）。
  每句用 `pinyin_ids.mjs`（网页的注音）和 Kokoro 自己的注音逐字对比：声韵母不同或声调真正不同的多音字整句丢掉，会被读成儿化的也丢掉；
  一/不变调和轻声的差别保留，学生能从老师的音频里学到正确读法。
- 重建：`python3 build_corpus.py --tatoeba <放 eng/cmn_sentences.tsv.bz2 的目录>`
  （需要 `misaki[zh]`、`opencc-python-reimplemented`，以及在本目录 `npm install`）。

## 版本固定

piper1-gpl 提交 `0471299`、底模仓库 rhasspy/piper-checkpoints 提交 `8d418f0`、Kokoro 英文模型 `f3ff357`、
中文模型 `01e7505`、kokoro / misaki 0.9.4、faster-whisper 1.2.1、pinyin-pro 3.29.4。
升级任何一个都可能让训练与网页的注音不一致，先确认再改。
