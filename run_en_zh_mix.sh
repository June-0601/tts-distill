#!/usr/bin/env bash
# 方案一：从第 200 轮纯英文模型增量微调「英文为主 + 中文为辅」分支（美音 6 合 1 + 英音 2 合 1，各 80 轮）。
#
# 用法：
#   ./run_en_zh_mix.sh
#
# 可用环境变量：
#   EPOCHS=80  BATCH=8  PRECISION=bf16-mixed  EN_BLEND=0.35
#   EXPORT_SYNC_DIR=/mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work/export
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
EN_BLEND="${EN_BLEND:-0.35}"
EXPORT_SYNC_DIR="${EXPORT_SYNC_DIR:-/mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work/export}"

if pgrep -f "piper\.train" >/dev/null 2>&1; then
  echo "检测到已有 piper.train 正在运行，避免抢占 RTX 4070 显存，已停止。"
  exit 1
fi

echo "[$(date '+%F %T')] === 1/6 生成音素表 + 混音合成 8 个英文音色的中文/混排语料 (en-blend=$EN_BLEND) ==="
"$HERE/.venv-piper/bin/python" "$HERE/en_zh_g2p.py" >/dev/null
"$HERE/.venv-synth/bin/python" "$HERE/synth_en_zh.py" --group all --en-blend "$EN_BLEND"

echo "[$(date '+%F %T')] === 2/6 Whisper large-v3-turbo 质检并合并纯英文回放数据 ==="
"$HERE/.venv-synth/bin/python" "$HERE/qa_en_zh.py" --group all

echo "[$(date '+%F %T')] === 3/6 增量微调美音+中文分支 (en_us_zh_mix, EPOCHS=$EPOCHS) ==="
EPOCHS="$EPOCHS" BATCH="$BATCH" PRECISION="$PRECISION" "$HERE/train.sh" en_us_zh_mix

echo "[$(date '+%F %T')] === 4/6 导出并双语试听评测美音+中文分支 ==="
"$HERE/export.sh" en_us_zh_mix
"$HERE/.venv-piper/bin/python" "$HERE/listen_en_zh.py" "$WORK/export/en_US-kokoro_us_zh_mix-medium.onnx"

echo "[$(date '+%F %T')] === 5/6 增量微调英音+中文分支 (en_uk_zh_mix, EPOCHS=$EPOCHS) ==="
EPOCHS="$EPOCHS" BATCH="$BATCH" PRECISION="$PRECISION" "$HERE/train.sh" en_uk_zh_mix

echo "[$(date '+%F %T')] === 6/6 导出并双语试听评测英音+中文分支 ==="
"$HERE/export.sh" en_uk_zh_mix
"$HERE/.venv-piper/bin/python" "$HERE/listen_en_zh.py" "$WORK/export/en_GB-kokoro_uk_zh_mix-medium.onnx"

if [ -d "$(dirname "$EXPORT_SYNC_DIR")" ]; then
  mkdir -p "$EXPORT_SYNC_DIR"
  cp -v "$WORK/export/en_US-kokoro_us_zh_mix-medium.onnx" \
        "$WORK/export/en_US-kokoro_us_zh_mix-medium.onnx.json" \
        "$WORK/export/en_GB-kokoro_uk_zh_mix-medium.onnx" \
        "$WORK/export/en_GB-kokoro_uk_zh_mix-medium.onnx.json" \
        "$EXPORT_SYNC_DIR/"
  for d in "$WORK"/export/listen-en_zh-*; do
    [ -d "$d" ] && rm -rf "$EXPORT_SYNC_DIR/$(basename "$d")" && cp -r "$d" "$EXPORT_SYNC_DIR/"
  done
  echo "[$(date '+%F %T')] 已同步导出模型与双语试听音频至：$EXPORT_SYNC_DIR"
fi

echo "[$(date '+%F %T')] 全部完成！"
