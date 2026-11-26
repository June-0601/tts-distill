# 为四个候选老师各建一个独立环境（放在各自仓库的 env 目录里，不影响旧版）。
# 用法：powershell -File setup_teachers.ps1 -Engine voxcpm|qwen3|index25|cosy3
param([Parameter(Mandatory = $true)][string]$Engine)
$ErrorActionPreference = 'Stop'
$J = 'D:\SharingNetwork\Junjun'
$conda = 'D:\miniconda3\Scripts\conda.exe'
$cu128 = 'https://download.pytorch.org/whl/cu128'

function New-Env($dir, $py) {
    if (-not (Test-Path "$dir\env\python.exe")) {
        & $conda create -y -q -p "$dir\env" "python=$py" pip | Out-Host
    }
    return "$dir\env\python.exe"
}

switch ($Engine) {
    'voxcpm' {
        $py = New-Env "$J\VoxCPM" '3.11'
        & $py -m pip install torch torchaudio --index-url $cu128
        & $py -m pip install -e "$J\VoxCPM"
    }
    'qwen3' {
        $py = New-Env "$J\Qwen3-TTS" '3.11'
        & $py -m pip install torch torchaudio --index-url $cu128
        & $py -m pip install -e "$J\Qwen3-TTS"
    }
    'index25' {
        $py = New-Env "$J\index-tts-2.5" '3.11'
        & $py -m pip install uv
        Push-Location "$J\index-tts-2.5"
        $env:UV_PROJECT_ENVIRONMENT = "$J\index-tts-2.5\.venv"
        & "$J\index-tts-2.5\env\Scripts\uv.exe" sync --python "$J\index-tts-2.5\env\python.exe"
        Pop-Location
        $py = "$J\index-tts-2.5\.venv\Scripts\python.exe"
    }
    'cosy3' {
        $py = New-Env "$J\CosyVoice3" '3.10'
        & $py -m pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        & $py -m pip install -r "$J\CosyVoice3\requirements.txt"
    }
}
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "SETUP DONE: $Engine"
