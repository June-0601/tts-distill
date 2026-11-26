"""下载四个候选老师模型（Hugging Face），断点续传。
用法：python download_teachers.py  （任何装了 huggingface_hub 的 Python 都可以）
"""
from huggingface_hub import snapshot_download

ROOT = r"D:\SharingNetwork\Junjun"
JOBS = [
    ("openbmb/VoxCPM2", rf"{ROOT}\VoxCPM\pretrained_models\VoxCPM2"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", rf"{ROOT}\Qwen3-TTS\pretrained_models\Qwen3-TTS-12Hz-1.7B-Base"),
    ("IndexTeam/IndexTTS-2.5", rf"{ROOT}\index-tts-2.5\checkpoints"),
    ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", rf"{ROOT}\CosyVoice3\pretrained_models\Fun-CosyVoice3-0.5B"),
]

for repo, dst in JOBS:
    print(f"==> {repo} -> {dst}", flush=True)
    snapshot_download(repo, local_dir=dst, max_workers=8)
    print(f"OK  {repo}", flush=True)
