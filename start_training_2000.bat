@echo off
echo ============================================
echo Edge SLM - 2000条数据生成与训练
echo ============================================
echo.

cd /d %~dp0

echo [1/2] 开始生成 2000 条训练数据...
echo 预计耗时: 2-4 小时 (取决于 Ollama 速度)
echo.

python scripts/batch_distill_and_train.py ^
    --num-samples 2000 ^
    --output-dir data/distilled_2000 ^
    --batch-size 100 ^
    --model qwen2.5:14b ^
    --train

echo.
echo ============================================
echo 任务完成！
echo 数据保存在: data/distilled_2000/
echo 模型保存在: outputs/model_large/
echo ============================================
pause
