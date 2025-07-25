#!/bin/bash
# GPU服务器运行脚本

# 设置错误时退出
set -e

echo "=== PyTorch 风格迁移 - GPU版本 ==="
echo "开始时间: $(date)"

# 检查GPU状态
echo -e "\n=== GPU状态检查 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "警告: nvidia-smi 不可用"
fi

# 检查CUDA和PyTorch
echo -e "\n=== 环境检查 ==="
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# 创建结果目录
mkdir -p results logs

# 设置日志文件
LOG_FILE="logs/style_transfer_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n=== 开始风格迁移 ==="
echo "日志文件: $LOG_FILE"

# 运行风格迁移
python3 train.py \
    --content "images/content/tubingen.jpg" \
    --style "images/style/starry_night.jpg" \
    --output "results/tubingen_starry_night_gpu.jpg" \
    --steps 2000 \
    --style-weight 1e6 \
    --content-weight 1 \
    --tv-weight 1e-3 \
    --lr 0.01 \
    --optimizer adam \
    --max-size 512 \
    --gpu \
    --log-file "$LOG_FILE" \
    --show-every 50 \
    --save-every 250

echo -e "\n=== 完成时间: $(date) ==="
echo "结果文件: results/tubingen_starry_night_gpu.jpg"
echo "日志文件: $LOG_FILE"
