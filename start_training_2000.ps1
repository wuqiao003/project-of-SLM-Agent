# Edge SLM - 2000条数据生成与训练脚本
# 使用方法: 右键 -> 使用 PowerShell 运行
# 或在终端执行: .\start_training_2000.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Edge SLM - 2000条数据生成与训练" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 切换到项目目录
Set-Location $PSScriptRoot

# 检查 Ollama 是否运行
Write-Host "[检查] Ollama 服务状态..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
    Write-Host "[OK] Ollama 正在运行，检测到模型: $($response.models.Count) 个" -ForegroundColor Green
} catch {
    Write-Host "[错误] Ollama 未运行！请先启动 Ollama" -ForegroundColor Red
    Write-Host "运行命令: ollama serve" -ForegroundColor Yellow
    Read-Host "按回车退出"
    exit 1
}

Write-Host ""
Write-Host "[1/2] 开始生成 2000 条训练数据..." -ForegroundColor Cyan
Write-Host "预计耗时: 2-4 小时 (取决于 Ollama 速度)" -ForegroundColor Yellow
Write-Host "支持断点续传，可随时中断后继续" -ForegroundColor Yellow
Write-Host ""

# 记录开始时间
$startTime = Get-Date

# 执行数据生成和训练
python scripts/batch_distill_and_train.py `
    --num-samples 2000 `
    --output-dir data/distilled_2000 `
    --batch-size 100 `
    --model qwen2.5:14b `
    --train

# 计算耗时
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "任务完成！" -ForegroundColor Green
Write-Host "总耗时: $($duration.Hours)小时 $($duration.Minutes)分钟" -ForegroundColor Green
Write-Host "数据保存在: data/distilled_2000/" -ForegroundColor Green
Write-Host "模型保存在: outputs/model_large/" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Read-Host "按回车退出"
