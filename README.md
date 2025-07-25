# 🎨 现代神经网络风格迁移 (Modern Neural Style Transfer)
基于 PyTorch 实现的现代神经网络风格迁移项目，使用最新的 ConvNeXt模型作为特征提取器，可以将任意艺术作品的风格应用到您的照片上。


## 📋 项目特色
- **🚀 GPU 加速**：支持多 GPU 并行处理，快速生成高质量结果
- **🧠 现代架构**：使用 ConvNeXt-2022 替代传统 VGG19，特征表示能力更强
- **🎛️ 多种模式**：快速、标准、高质量三种处理模式
- **💡 简单易用**：交互式界面，无需复杂参数配置
- **⚙️ 高度可定制**：支持丰富的参数调节和优化器选择
- **📊 实时监控**：详细的训练进度和损失函数监控
- **🔧 稳定性优化**：包含 NaN 检测、梯度裁剪等稳定性措施
- **⚡ 高效处理**：ConvNeXt 架构提供更好的效率和质量平衡

### 核心算法
- **内容提取**：使用预训练 ConvNeXt-Base 网络提取图像内容特征
- **风格提取**：通过 Gram 矩阵捕获纹理和风格信息
- **损失函数**：内容损失 + 风格损失 + 总变分损失
- **优化方法**：支持 Adam 和 L-BFGS 优化器

### ConvNeXt 优势
- **现代架构 (2022)**：融合了 Vision Transformer 的设计理念
- **更强特征表示**：相比 VGG19，提供更丰富的语义特征
- **高效计算**：更好的参数效率和计算速度
- **稳定训练**：改进的归一化和激活函数设计

## 📦 安装依赖
```bash
# 克隆项目
git clone https://github.com/EvanZhangcn/style_transfer.git
cd style_transfer

# 安装依赖
pip install torch torchvision pillow numpy matplotlib

# 如需 GPU 支持，请安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速开始

```bash
# 命令行快速使用
chmod +x quick_modern_transfer.sh
./quick_modern_transfer.sh

# 完整参数控制
python modern_train.py \
    --content images/content/photo.jpg \
    --style images/style/artwork.jpg \
    --output results/modern_result.jpg \
    --steps 800 \
    --max-size 512 \
    --style-weight 1e6
```

## 使用自己的图像

1. 将内容图像放入 `images/content/` 文件夹
2. 将风格图像放入 `images/style/` 文件夹
3. 使用GPU训练命令

## 算法原理

风格迁移通过最小化以下损失函数实现：

1. **内容损失**: 保持原图的内容结构
2. **风格损失**: 匹配风格图像的纹理特征（通过Gram矩阵）
3. **总变分损失**: 保持图像平滑性

使用预训练的ConvNeXt网络提取特征，通过梯度下降优化生成图像的像素值。
