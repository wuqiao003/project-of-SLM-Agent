# 低负载训练 — 速度慢，但主机不易卡死
# 用法: .\scripts\train_low_load.ps1
# 仍卡: 把 $StepDelay 改为 1.5 或 2.0

$StepDelay = 0.8
Set-Location $PSScriptRoot\..

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "低负载训练启动 (step-delay=${StepDelay}s)" -ForegroundColor Cyan

python run.py train data/prepared/tool_use_train.jsonl `
    --output-dir outputs/tool_use_v1 `
    --epochs 3 `
    --batch-size 1 `
    --no-use-unsloth `
    --low-load `
    --step-delay $StepDelay `
    --cpu-threads 4
