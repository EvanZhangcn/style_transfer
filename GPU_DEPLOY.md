# GPU服务器部署指南

## 环境准备

### 1. 检查GPU环境
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version
```

### 2. 安装Python环境
```bash
# 使用conda创建环境
conda create -n style_transfer python=3.9 -y
conda activate style_transfer

# 或使用venv
python3 -m venv style_transfer_env
source style_transfer_env/bin/activate  # Linux
```

### 3. 安装依赖
```bash
# 安装PyTorch (GPU版本)
# 请根据你的CUDA版本选择合适的命令
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install Pillow matplotlib numpy requests
```

### 4. 验证安装
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## 文件传输

### 上传项目到服务器
```bash
# 使用scp上传整个项目
scp -r style_transfer/ username@server:/path/to/your/workspace/

# 或使用rsync
rsync -avz style_transfer/ username@server:/path/to/your/workspace/style_transfer/
```

### 设置权限
```bash
# 设置脚本执行权限
chmod +x *.sh
chmod +x *.py
```

## 快速开始

### 1. 单次运行
```bash
# 基本运行
python3 train.py \
    --content images/content/tubingen.jpg \
    --style images/style/starry_night.jpg \
    --output results/output.jpg \
    --gpu

# 高质量运行
python3 train.py \
    --content images/content/tubingen.jpg \
    --style images/style/starry_night.jpg \
    --output results/high_quality.jpg \
    --steps 2000 \
    --max-size 1024 \
    --optimizer lbfgs \
    --gpu
```

### 2. 使用预配置脚本
```bash
# GPU标准运行
./run_gpu.sh

# 批量处理
./batch_process.sh

# 性能测试
python3 gpu_benchmark.py
```

## 高级配置

### 1. 参数调优指南

**图像尺寸 (max-size)**
- 256: 快速测试，约1-2分钟
- 512: 标准质量，约5-10分钟  
- 1024: 高质量，约20-30分钟
- 2048: 极高质量，约1-2小时

**训练步数 (steps)**
- 500-1000: 基本效果
- 1000-2000: 标准质量
- 2000-5000: 高质量

**损失权重调优**
```bash
# 更强的风格效果
--style-weight 5e6

# 更好的内容保持
--content-weight 5

# 更平滑的图像
--tv-weight 1e-2
```

### 2. 优化器选择

**Adam (推荐)**
- 快速收敛
- 适合大多数情况
- 内存占用较小

**LBFGS**
- 更高质量
- 收敛更稳定
- 内存占用较大

### 3. 多GPU支持 (如果有多个GPU)
```bash
# 指定GPU
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
export CUDA_VISIBLE_DEVICES=1  # 使用第二个GPU
```

## 监控和调试

### 1. 实时监控
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控训练进度
tail -f logs/style_transfer_*.log
```

### 2. 内存优化
```bash
# 如果遇到GPU内存不足
python3 train.py \
    --content images/content/tubingen.jpg \
    --style images/style/starry_night.jpg \
    --output results/output.jpg \
    --max-size 256 \
    --gpu
```

### 3. 错误处理
```bash
# 检查CUDA错误
python3 -c "
import torch
try:
    x = torch.randn(10, 10).cuda()
    print('CUDA工作正常')
except Exception as e:
    print(f'CUDA错误: {e}')
"
```

## 批量处理建议

### 1. 后台运行
```bash
# 使用nohup后台运行
nohup ./batch_process.sh > batch_output.log 2>&1 &

# 使用screen
screen -S style_transfer
./batch_process.sh
# Ctrl+A, D 分离会话
```

### 2. 定时任务
```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天凌晨2点运行）
0 2 * * * cd /path/to/style_transfer && ./batch_process.sh
```

## 性能优化建议

### 1. 硬件建议
- **GPU**: GTX 1080Ti+ 或 RTX 2060+
- **显存**: 至少8GB（处理1024尺寸图像）
- **内存**: 至少16GB
- **存储**: SSD存储（图像I/O更快）

### 2. 软件优化
- 使用最新版本的PyTorch
- 启用混合精度训练（如果支持）
- 合理设置batch size

### 3. 网络优化
如果需要下载模型，确保网络连接稳定：
```bash
# 设置代理（如需要）
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
```

## 常见问题

### Q: CUDA out of memory
A: 
```bash
# 减小图像尺寸
--max-size 256

# 或使用CPU
--cpu
```

### Q: 训练太慢
A:
```bash
# 减少训练步数
--steps 500

# 使用更小的图像
--max-size 256
```

### Q: 结果质量不佳
A:
```bash
# 增加训练步数
--steps 2000

# 调整损失权重
--style-weight 5e6 --content-weight 2

# 使用LBFGS优化器
--optimizer lbfgs
```

## 输出文件说明

- `results/`: 生成的风格迁移图像
- `logs/`: 训练日志文件
- `results/*_step_*.jpg`: 中间训练结果
- `logs/gpu_benchmark.log`: 性能测试日志

## 清理和维护

```bash
# 清理临时文件
find results/ -name "*_step_*.jpg" -delete

# 清理旧日志（保留最近7天）
find logs/ -name "*.log" -mtime +7 -delete

# 清理GPU缓存
python3 -c "import torch; torch.cuda.empty_cache()"
```
