# 对一批音色跑完整试验：Kokoro 参考音频+对照组 → 四个老师引擎 → 评测 → 音色相似度 → 试听页。
# 用法：powershell -File run_voices.ps1 -Voices zf_021,zf_022
param([Parameter(Mandatory = $true)][string]$Voices)
$env:PYTHONIOENCODING = 'utf-8'; $env:TQDM_DISABLE = '1'
$J = 'D:\SharingNetwork\Junjun'
$P = "$J\reading-app\tts-distill\pilot"
$W = '/mnt/d/SharingNetwork/Junjun/reading-app/tts-distill'

Write-Host "######## kokoro $(Get-Date -f HH:mm)"
wsl -d Ubuntu -- bash -lc "cd /root/tts-distill && cp $W/pilot/*.py pilot/ && HF_HUB_OFFLINE=1 .venv-synth/bin/python pilot/kokoro_pilot.py --voices $Voices --out $W/work/pilot/out 2>&1 | grep -E '^(ref|kokoro)|Error'"

$engines = @(
    @('cosy3', "$J\CosyVoice3\env\python.exe"),
    @('qwen3', "$J\Qwen3-TTS\env\python.exe"),
    @('index25', "$J\index-tts-2.5\.venv\Scripts\python.exe"),
    @('voxcpm', "$J\VoxCPM\env\python.exe")
)
foreach ($e in $engines) {
    Write-Host "######## $($e[0]) $(Get-Date -f HH:mm)"
    & $e[1] "$P\teacher_pilot.py" --engine $e[0] --voices $Voices 2>&1 | Select-String 'DONE|Traceback|Error:' | Select-Object -Last 10
}

Write-Host "######## eval $(Get-Date -f HH:mm)"
wsl -d Ubuntu -- bash -lc "cd /root/tts-distill && HF_HUB_OFFLINE=1 .venv-synth/bin/python $W/pilot/eval_pilot.py $W/work/pilot > /tmp/eval.log 2>&1; sed -n '/==/,`$p' /tmp/eval.log"
Push-Location "$J\CosyVoice3"
& .\env\python.exe "$P\spk_sim.py"
Pop-Location
& D:\miniconda3\python.exe -I "$P\make_listen_page.py"
Write-Host "ALL DONE $(Get-Date -f HH:mm)"
