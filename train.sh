#!/usr/bin/env bash
# 训练 Piper medium 学生模型（在 .venv-piper 里跑）。
#   ./train.sh en            从底模（LJSpeech medium）开始
#   ./train.sh zh            从底模（Chaowen medium，拼音音素）开始
#   ./train.sh en --resume   断电、重启或手动停止后，从最近一次的 last.ckpt 接着训
#
# 可用环境变量调整（默认值按 12GB 显存的 RTX 4070 设定）：
#   EPOCHS=1000  BATCH=16  WORKERS=6  PRECISION=32  ACCEL=gpu  MOS=utmos
#   续训时 EPOCHS 必须和第一次相同（学习率按总轮数退火）。
set -euo pipefail

LANG_ID="${1:?用法：./train.sh en|zh [--resume]}"
MODE="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
EPOCHS="${EPOCHS:-1000}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-6}"
PRECISION="${PRECISION:-32}"
ACCEL="${ACCEL:-gpu}"
MOS="${MOS:-utmos}"

[ -n "${VIRTUAL_ENV:-}" ] || source "$HERE/.venv-piper/bin/activate"

case "$LANG_ID" in
  en)
    NAME=en_US-kokoro_heart-medium
    BASE_CKPT="$WORK/base/lj-med_1000.ckpt"
    EXTRA=(--data.espeak_voice en-us)
    ;;
  zh)
    NAME=zh_CN-kokoro_zf001-medium
    BASE_CKPT="$WORK/base/chaowen-medium.ckpt"
    # 音素 ID 已在 corpus/zh.tsv 里算好（与网页同一份 pinyin_ids.mjs），训练时不再注音
    EXTRA=(--data.espeak_voice cmn --data.phoneme_type text --data.dataset_type phoneme_ids
           --data.phonemes_path "$HERE/corpus/zh_phonemes.json" --model.use_mrd true)
    ;;
  *) echo "语言只能是 en 或 zh"; exit 2 ;;
esac

RUN="$WORK/runs/$LANG_ID"
META="$WORK/$LANG_ID/metadata.csv"
[ -s "$META" ] || { echo "缺少 $META：先跑 synth.py 和 qa.py"; exit 1; }
mkdir -p "$RUN"

ARGS=(
  fit
  --data.voice_name "$NAME"
  --data.csv_path "$META"
  --data.audio_dir "$WORK/$LANG_ID/wavs"
  --data.cache_dir "$WORK/$LANG_ID/cache"
  --data.config_path "$RUN/$NAME.onnx.json"
  --data.batch_size "$BATCH"
  --data.num_workers "$WORKERS"
  --data.validation_split 0.03
  --data.num_test_examples 8
  --data.length_bucketing true
  --data.trim_silence false
  --model.sample_rate 22050
  --model.mos_metric "$MOS"
  --trainer.max_epochs "$EPOCHS"
  --trainer.accelerator "$ACCEL"
  --trainer.devices 1
  --trainer.precision "$PRECISION"
  --trainer.default_root_dir "$RUN"
  "${EXTRA[@]}"
)

if [ "$MODE" = "--resume" ]; then
  LAST="$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
  [ -n "$LAST" ] || { echo "$RUN 下面还没有 last.ckpt，去掉 --resume 从头开始"; exit 1; }
  echo "从 $LAST 接着训（总轮数 $EPOCHS）"
  exec python3 -m piper.train "${ARGS[@]}" --ckpt_path "$LAST"
fi

[ -s "$BASE_CKPT" ] || { echo "缺少底模 $BASE_CKPT：先跑 setup_wsl.sh"; exit 1; }
echo "从底模 $(basename "$BASE_CKPT") 开始训 $NAME：$EPOCHS 轮，batch $BATCH"
exec python3 -m piper.train "${ARGS[@]}" --model.warmstart_ckpt "$BASE_CKPT"
