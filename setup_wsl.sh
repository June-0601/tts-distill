#!/usr/bin/env bash
# 在 Windows 的 WSL2（Ubuntu 24.04）里运行一次，装好合成、质检、训练要用的全部东西：
#   cd ~/june_zhanread/tools/tts-distill && bash setup_wsl.sh
# 重复运行是安全的：已经装好、下好的会跳过。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
PIPER_COMMIT=0471299744417dae497991f859dd0d3c1daab170          # piper1-gpl，2026-11-09
CKPT_REV=8d418f0288a50aed603547ee751193fa7910bfe3              # rhasspy/piper-checkpoints
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

echo "== 1/6 检查显卡"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "WSL 里看不到 NVIDIA 显卡。先在 Windows 上装最新的 NVIDIA 驱动（自带 WSL 用的 CUDA，WSL 里不要另装驱动），"
  echo "然后在 PowerShell 里运行 wsl --shutdown，再重新打开 Ubuntu。"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
case "$HERE" in
  /mnt/*) echo "注意：仓库放在 Windows 盘（$HERE）上，训练读写会慢很多。建议 clone 到 WSL 自己的目录（例如 ~/）。" ;;
esac

echo "== 2/6 系统依赖"
sudo apt-get update -y
sudo apt-get install -y build-essential cmake ninja-build git curl python3-venv python3-dev \
  espeak-ng libsndfile1 nodejs npm

echo "== 3/6 合成与质检环境 .venv-synth（Kokoro + Whisper）"
[ -x "$HERE/.venv-synth/bin/python" ] || python3 -m venv "$HERE/.venv-synth"
"$HERE/.venv-synth/bin/pip" install -q -U pip
"$HERE/.venv-synth/bin/pip" install -q torch --index-url "$TORCH_INDEX"
"$HERE/.venv-synth/bin/pip" install -q "kokoro==0.9.4" "misaki[en,zh]==0.9.4" soundfile \
  "faster-whisper==1.2.1" "num2words==0.5.14" pypinyin
(cd "$HERE" && npm install --no-audit --no-fund --silent)

echo "== 4/6 训练环境 .venv-piper（piper1-gpl 固定提交 ${PIPER_COMMIT:0:7}）"
mkdir -p "$WORK"
[ -d "$WORK/piper1-gpl/.git" ] || git clone -q https://github.com/OHF-Voice/piper1-gpl.git "$WORK/piper1-gpl"
git -C "$WORK/piper1-gpl" fetch -q origin "$PIPER_COMMIT" 2>/dev/null || git -C "$WORK/piper1-gpl" fetch -q origin
git -C "$WORK/piper1-gpl" checkout -q "$PIPER_COMMIT"
[ -x "$HERE/.venv-piper/bin/python" ] || python3 -m venv "$HERE/.venv-piper"
"$HERE/.venv-piper/bin/pip" install -q -U pip
"$HERE/.venv-piper/bin/pip" install -q torch --index-url "$TORCH_INDEX"
"$HERE/.venv-piper/bin/pip" install -q -e "$WORK/piper1-gpl[train]"
"$HERE/.venv-piper/bin/pip" install -q "faster-whisper==1.2.1" "num2words==0.5.14" pypinyin soundfile
(cd "$WORK/piper1-gpl" && source "$HERE/.venv-piper/bin/activate" \
  && ./build_monotonic_align.sh && python3 setup.py build_ext --inplace >/dev/null)

echo "== 5/6 下载底模（各约 850MB）"
mkdir -p "$WORK/base"
BASE="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/$CKPT_REV"
[ -s "$WORK/base/lj-med_1000.ckpt" ] || \
  curl -L --fail -o "$WORK/base/lj-med_1000.ckpt" "$BASE/en/en_US/ljspeech/medium/lj-med_1000.ckpt"
[ -s "$WORK/base/chaowen-medium.ckpt" ] || \
  curl -L --fail -o "$WORK/base/chaowen-medium.ckpt" "$BASE/zh/zh_CN/chaowen/medium/epoch%3D1999-step%3D88000.ckpt"

echo "== 6/6 自检"
"$HERE/.venv-piper/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "训练环境里 torch 看不到 CUDA"
p = torch.cuda.get_device_properties(0)
print(f"训练环境 OK：{p.name}，{p.total_memory / 2**30:.1f} GB 显存，torch {torch.__version__}")
from piper.train.vits.monotonic_align import maximum_path  # noqa: F401  确认对齐模块已编译
PY
"$HERE/.venv-synth/bin/python" -c "import torch, kokoro, faster_whisper; assert torch.cuda.is_available(); print('合成环境 OK')"
df -h "$WORK" | tail -1 | awk '{print "剩余磁盘：" $4 "（建议至少 50G）"}'
echo "全部装好。下一步：source .venv-synth/bin/activate && python synth.py en"
