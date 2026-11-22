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

LANG_ID="${1:?用法：./train.sh en|zh [--resume] [额外 piper.train 参数...]}"
shift
MODE=""
if [ "${1:-}" = "--resume" ]; then
  MODE="--resume"
  shift
fi
EXTRA_CLI=("$@")
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
EPOCHS="${EPOCHS:-1000}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-6}"
if [ -z "${PRECISION:-}" ]; then
  case "$LANG_ID" in
    *_multi) PRECISION=bf16-mixed ;;
    *) PRECISION=32 ;;
  esac
fi
ACCEL="${ACCEL:-gpu}"
MOS="${MOS:-utmos}"

[ -n "${VIRTUAL_ENV:-}" ] || source "$HERE/.venv-piper/bin/activate"

case "$LANG_ID" in
  en)
    NAME=en_US-kokoro_heart-medium
    BASE_CKPT="$WORK/base/lj-med_1000.ckpt"
    AUDIO_DIR="$WORK/en/wavs"
    META="$WORK/en/metadata.csv"
    CACHE_DIR="$WORK/en/cache"
    EXTRA=(--data.espeak_voice en-us)
    ;;
  en_us_multi)
    NAME=en_US-kokoro_us_multi-medium
    BASE_CKPT="$WORK/base/lj-med_1000.ckpt"
    AUDIO_DIR="$WORK/en_us_multi/wavs"
    META="$WORK/en_us_multi/metadata.csv"
    CACHE_DIR="$WORK/en_us_multi/cache"
    NUM_SPEAKERS="${NUM_SPEAKERS:-6}"
    EXTRA=(--model.num_speakers "$NUM_SPEAKERS" --data.espeak_voice en-us --model.use_mrd true)
    ;;
  en_uk_multi)
    NAME=en_GB-kokoro_uk_multi-medium
    BASE_CKPT="$WORK/base/vctk-medium.ckpt"
    AUDIO_DIR="$WORK/en_uk_multi/wavs"
    META="$WORK/en_uk_multi/metadata.csv"
    CACHE_DIR="$WORK/en_uk_multi/cache"
    NUM_SPEAKERS="${NUM_SPEAKERS:-2}"
    EXTRA=(--model.num_speakers "$NUM_SPEAKERS" --data.espeak_voice en-gb-x-rp --model.use_mrd true)
    ;;
  en_us_zh_mix)
    NAME=en_US-kokoro_us_zh_mix-medium
    BASE_CKPT="${BASE_CKPT:-$WORK/runs/en_us_multi/lightning_logs/version_0/checkpoints/last.ckpt}"
    AUDIO_DIR="$WORK/en_us_zh_mix/wavs"
    META="$WORK/en_us_zh_mix/metadata.csv"
    CACHE_DIR="$WORK/en_us_zh_mix/cache"
    NUM_SPEAKERS="${NUM_SPEAKERS:-6}"
    EXTRA=(--model.num_speakers "$NUM_SPEAKERS" --data.espeak_voice en-us --data.phoneme_type text
           --data.dataset_type phoneme_ids --data.phonemes_path "$HERE/corpus/en_zh_phonemes.json"
           --model.learning_rate 0.00015 --model.learning_rate_d 0.000075 --model.use_mrd true)
    ;;
  en_uk_zh_mix)
    NAME=en_GB-kokoro_uk_zh_mix-medium
    BASE_CKPT="${BASE_CKPT:-$WORK/runs/en_uk_multi/lightning_logs/version_0/checkpoints/last.ckpt}"
    AUDIO_DIR="$WORK/en_uk_zh_mix/wavs"
    META="$WORK/en_uk_zh_mix/metadata.csv"
    CACHE_DIR="$WORK/en_uk_zh_mix/cache"
    NUM_SPEAKERS="${NUM_SPEAKERS:-2}"
    EXTRA=(--model.num_speakers "$NUM_SPEAKERS" --data.espeak_voice en-gb-x-rp --data.phoneme_type text
           --data.dataset_type phoneme_ids --data.phonemes_path "$HERE/corpus/en_zh_phonemes.json"
           --model.learning_rate 0.00015 --model.learning_rate_d 0.000075 --model.use_mrd true)
    ;;
  en_af_*)
    VOICE="${LANG_ID#en_}"
    NAME="en_US-kokoro_${VOICE}-medium"
    BASE_CKPT="${BASE_CKPT:-$WORK/base/lj-med_1000.ckpt}"
    AUDIO_DIR="$WORK/en_us_multi/wavs/$VOICE"
    META="$WORK/en_us_multi/metadata_${VOICE}.csv"
    CACHE_DIR="$WORK/en_us_multi/cache_${VOICE}"
    EXTRA=(--data.espeak_voice en-us --model.use_mrd true)
    ;;
  en_bf_*)
    VOICE="${LANG_ID#en_}"
    NAME="en_GB-kokoro_${VOICE}-medium"
    BASE_CKPT="${BASE_CKPT:-$WORK/base/vctk-medium.ckpt}"
    AUDIO_DIR="$WORK/en_uk_multi/wavs/$VOICE"
    META="$WORK/en_uk_multi/metadata_${VOICE}.csv"
    CACHE_DIR="$WORK/en_uk_multi/cache_${VOICE}"
    EXTRA=(--data.espeak_voice en-gb-x-rp --model.use_mrd true)
    ;;
  zh)
    NAME=zh_CN-kokoro_zf001-medium
    BASE_CKPT="$WORK/base/chaowen-medium.ckpt"
    AUDIO_DIR="$WORK/zh/wavs"
    META="$WORK/zh/metadata.csv"
    CACHE_DIR="$WORK/zh/cache"
    # 音素 ID 已在 corpus/zh.tsv 里算好（与网页同一份 pinyin_ids.mjs），训练时不再注音
    EXTRA=(--data.espeak_voice cmn --data.phoneme_type text --data.dataset_type phoneme_ids
           --data.phonemes_path "$HERE/corpus/zh_phonemes.json" --model.use_mrd true)
    ;;
  zh_multi)
    NAME=zh_CN-kokoro_zh_multi-medium
    BASE_CKPT="$WORK/base/chaowen-medium.ckpt"
    AUDIO_DIR="$WORK/zh_multi/wavs"
    META="$WORK/zh_multi/metadata.csv"
    CACHE_DIR="$WORK/zh_multi/cache"
    NUM_SPEAKERS="${NUM_SPEAKERS:-8}"
    EXTRA=(--model.num_speakers "$NUM_SPEAKERS" --data.espeak_voice cmn --data.phoneme_type text
           --data.dataset_type phoneme_ids --data.phonemes_path "$HERE/corpus/zh_phonemes.json"
           --model.use_mrd true)
    ;;
  zh_zf_*)
    VOICE="${LANG_ID#zh_}"
    NAME="zh_CN-kokoro_${VOICE}-medium"
    BASE_CKPT="${BASE_CKPT:-$WORK/base/chaowen-medium.ckpt}"
    AUDIO_DIR="$WORK/zh_multi/wavs/$VOICE"
    META="$WORK/zh_multi/metadata_${VOICE}.csv"
    CACHE_DIR="$WORK/zh_multi/cache_${VOICE}"
    EXTRA=(--data.espeak_voice cmn --data.phoneme_type text --data.dataset_type phoneme_ids
           --data.phonemes_path "$HERE/corpus/zh_phonemes.json" --model.use_mrd true)
    ;;
  *) echo "语言只能是 en、en_us_multi、en_uk_multi、en_af_xxx、en_bf_xxx、zh、zh_multi 或 zh_zf_xxx"; exit 2 ;;
esac

RUN="$WORK/runs/$LANG_ID"
[ -s "$META" ] || { echo "缺少 $META：先跑 synth*.py 和 qa*.py"; exit 1; }
mkdir -p "$RUN"

ARGS=(
  fit
  --data.voice_name "$NAME"
  --data.csv_path "$META"
  --data.audio_dir "$AUDIO_DIR"
  --data.cache_dir "$CACHE_DIR"
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

if [ -n "${LR_DECAY:-}" ]; then
  ARGS+=(--model.lr_decay "$LR_DECAY" --model.lr_decay_d "$LR_DECAY")
fi
if [ ${#EXTRA_CLI[@]} -gt 0 ]; then
  ARGS+=("${EXTRA_CLI[@]}")
fi

if [ "$MODE" = "--resume" ]; then
  LAST="$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
  [ -n "$LAST" ] || { echo "$RUN 下面还没有 last.ckpt，去掉 --resume 从头开始"; exit 1; }
  echo "从 $LAST 接着训（总轮数 $EPOCHS）"
  exec python3 -m piper.train "${ARGS[@]}" --ckpt_path "$LAST"
fi

[ -s "$BASE_CKPT" ] || { echo "缺少底模 $BASE_CKPT：先跑 setup_wsl.sh"; exit 1; }
echo "从底模 $(basename "$BASE_CKPT") 开始训 $NAME：$EPOCHS 轮，batch $BATCH"
exec python3 -m piper.train "${ARGS[@]}" --model.warmstart_ckpt "$BASE_CKPT"
