#!/usr/bin/env bash
# 把 checkpoint 导出成网页能用的 ONNX + 配置（在 .venv-piper 里跑）。
#   ./export.sh en                 用最近的 last.ckpt
#   ./export.sh zh path/to.ckpt    指定某个 checkpoint（比如 val_mos 最高的那个）
# 输出 work/export/<名字>.onnx 和 .onnx.json，拷回 Mac 交给网页接入。
set -euo pipefail

LANG_ID="${1:?用法：./export.sh en|zh [checkpoint]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
[ -n "${VIRTUAL_ENV:-}" ] || source "$HERE/.venv-piper/bin/activate"

case "$LANG_ID" in
  en) NAME=en_US-kokoro_heart-medium ;;
  zh) NAME=zh_CN-kokoro_zf001-medium ;;
  *) echo "语言只能是 en 或 zh"; exit 2 ;;
esac
RUN="$WORK/runs/$LANG_ID"
CKPT="${2:-$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)}"
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "找不到 checkpoint"; exit 1; }

OUT="$WORK/export"
mkdir -p "$OUT"
python3 -m piper.train.export_onnx --checkpoint "$CKPT" --output-file "$OUT/$NAME.onnx"
cp "$RUN/$NAME.onnx.json" "$OUT/$NAME.onnx.json"
python3 - "$LANG_ID" "$OUT/$NAME.onnx.json" <<'PY'
import json, sys
lang_id, path = sys.argv[1], sys.argv[2]
cfg = json.load(open(path, encoding="utf-8"))
# 收紧蒸馏模型的随机时长与噪声尺度，减少气音底噪并让单读单词更稳
cfg["inference"] = {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.6}
if lang_id == "zh":
    # 训练时按 text 读现成的 ID；导出后标回 pinyin，和 Chaowen / Xiao Ya 的配置一致
    cfg["phoneme_type"] = "pinyin"
json.dump(cfg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
echo "导出自 $CKPT"
ls -lh "$OUT/$NAME.onnx" "$OUT/$NAME.onnx.json"
echo "试听并让 Whisper 打分：python listen.py $LANG_ID $OUT/$NAME.onnx"
