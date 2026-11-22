#!/usr/bin/env bash
# 方案 A（美音 6 合 1 + 英音 2 合 1）英文音色一站式流水线（暂定 200 轮）。
#
# 用法：
#   ./run_en_multi.sh            # 若检测到 zh_multi 仍在训练会拒绝抢占 GPU 并提示
#   ./run_en_multi.sh --wait-zh  # 等待当前 zh_multi 训练彻底结束后自动开跑
#
# 可用环境变量：
#   EPOCHS=200  BATCH=8  PRECISION=bf16-mixed
#   注：en.tsv 英文音素最长 735 / 频谱最长 1864，RTX 4070 12GB 在 BATCH=16 下会 OOM，
#       BATCH=12 峰值达 ~10.5GiB（加上桌面与验证模型余量极紧），BATCH=8 峰值 ~7.6GiB 最稳。
#   EXPORT_SYNC_DIR=/mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work/export
set -euo pipefail

WAIT_ZH=0
if [ "${1:-}" = "--wait-zh" ]; then
  WAIT_ZH=1
  shift
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
EXPORT_SYNC_DIR="${EXPORT_SYNC_DIR:-/mnt/d/SharingNetwork/Junjun/reading-app/tts-distill/work/export}"

if pgrep -f "piper\.train.*zh_multi" >/dev/null 2>&1; then
  if [ "$WAIT_ZH" -eq 1 ]; then
    echo "[$(date '+%F %T')] 检测到 zh_multi 仍在训练，等待其完成后自动启动英文流水线..."
    while pgrep -f "piper\.train.*zh_multi" >/dev/null 2>&1; do
      sleep 60
    done
    echo "[$(date '+%F %T')] zh_multi 已结束，等待 10 秒释放显存..."
    sleep 10
  else
    echo "检测到 zh_multi 仍在训练（RTX 4070 仅 12GB 显存，不可并发）。"
    echo "请先等 zh_multi 完成，或运行：./run_en_multi.sh --wait-zh"
    exit 1
  fi
fi

echo "[$(date '+%F %T')] === 1/6 Kokoro 合成 8 个英文音色（美音 6 + 英音 2） ==="
"$HERE/.venv-synth/bin/python" "$HERE/synth_en_multi.py" --group all

echo "[$(date '+%F %T')] === 2/6 Whisper small.en 质检并生成 metadata.csv ==="
"$HERE/.venv-synth/bin/python" "$HERE/qa_en_multi.py" --group all

echo "[$(date '+%F %T')] === 3/6 训练美音 6 合 1 (en_us_multi, EPOCHS=$EPOCHS) ==="
EPOCHS="$EPOCHS" BATCH="$BATCH" PRECISION="$PRECISION" "$HERE/train.sh" en_us_multi

echo "[$(date '+%F %T')] === 4/6 导出并试听评测美音 6 合 1 ==="
"$HERE/export.sh" en_us_multi
"$HERE/.venv-piper/bin/python" "$HERE/listen.py" en "$WORK/export/en_US-kokoro_us_multi-medium.onnx"

echo "[$(date '+%F %T')] === 5/6 训练英音 2 合 1 (en_uk_multi, EPOCHS=$EPOCHS) ==="
EPOCHS="$EPOCHS" BATCH="$BATCH" PRECISION="$PRECISION" "$HERE/train.sh" en_uk_multi

echo "[$(date '+%F %T')] === 6/6 导出并试听评测英音 2 合 1 ==="
"$HERE/export.sh" en_uk_multi
"$HERE/.venv-piper/bin/python" "$HERE/listen.py" en "$WORK/export/en_GB-kokoro_uk_multi-medium.onnx"

if [ -d "$(dirname "$EXPORT_SYNC_DIR")" ]; then
  mkdir -p "$EXPORT_SYNC_DIR"
  cp -v "$WORK/export/en_US-kokoro_us_multi-medium.onnx" \
        "$WORK/export/en_US-kokoro_us_multi-medium.onnx.json" \
        "$WORK/export/en_GB-kokoro_uk_multi-medium.onnx" \
        "$WORK/export/en_GB-kokoro_uk_multi-medium.onnx.json" \
        "$EXPORT_SYNC_DIR/"
  for d in "$WORK"/export/listen-en-*; do
    [ -d "$d" ] && cp -r "$d" "$EXPORT_SYNC_DIR/"
  done
  echo "[$(date '+%F %T')] 已同步导出模型与试听音频至：$EXPORT_SYNC_DIR"
fi

echo "[$(date '+%F %T')] 全部完成！"
