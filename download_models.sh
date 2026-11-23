#!/usr/bin/env bash
# 在 Mac 上一键从 GitHub Release (v1.0-models) 下载 3 套蒸馏好的 ONNX 模型到 models/ 并校验 SHA256。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/models"
TAG="${TAG:-v1.0-models}"
BASE_URL="https://github.com/June-0601/tts-distill/releases/download/${TAG}"

mkdir -p "$OUT"

FILES=(
  "en_US-kokoro_us_zh_mix-medium.onnx"
  "en_US-kokoro_us_zh_mix-medium.onnx.json"
  "en_GB-kokoro_uk_zh_mix-medium.onnx"
  "en_GB-kokoro_uk_zh_mix-medium.onnx.json"
  "zh_CN-kokoro_zh_multi-medium.onnx"
  "zh_CN-kokoro_zh_multi-medium.onnx.json"
)

for f in "${FILES[@]}"; do
  # *.onnx.json 已随 Git 仓库一起更新；仅在本地缺失时才回退从 Release 拉取，避免旧 Release 覆盖新配置
  if [[ "$f" == *.onnx.json && -s "$OUT/$f" ]]; then
    echo ">>> 保留仓库内最新的 $f"
    continue
  fi
  echo ">>> 下载 $f ..."
  curl -fL --retry 3 --progress-bar -o "$OUT/$f" "$BASE_URL/$f"
done

echo ">>> 校验 SHA256 ..."
cd "$OUT"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c SHA256SUMS.txt
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c SHA256SUMS.txt
fi
echo "全部模型已就绪：$OUT"
