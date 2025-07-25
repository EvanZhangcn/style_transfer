#!/bin/bash
# 现代风格迁移快速启动脚本 - ConvNeXt版本
# Modern Style Transfer Quick Start Script using ConvNeXt

echo "🎨 现代神经网络风格迁移 (ConvNeXt-2022)"
echo "==========================================="

# 交互式获取内容图像路径
while true; do
    read -p "请输入内容图像路径: " CONTENT_PATH
    if [ -f "$CONTENT_PATH" ]; then
        break
    else
        echo "❌ 内容图像不存在: $CONTENT_PATH"
        echo "请重新输入有效的文件路径"
    fi
done

# 交互式获取风格图像路径
while true; do
    read -p "请输入风格图像路径: " STYLE_PATH
    if [ -f "$STYLE_PATH" ]; then
        break
    else
        echo "❌ 风格图像不存在: $STYLE_PATH"
        echo "请重新输入有效的文件路径"
    fi
done

# 创建输出目录
mkdir -p results

# 生成输出文件名
CONTENT_NAME=$(basename "$CONTENT_PATH" | cut -d. -f1)
STYLE_NAME=$(basename "$STYLE_PATH" | cut -d. -f1)
OUTPUT_PATH="results/${CONTENT_NAME}_with_${STYLE_NAME}_convnext.jpg"

echo "📸 内容图像: $CONTENT_PATH"
echo "🎭 风格图像: $STYLE_PATH"
echo "💾 输出路径: $OUTPUT_PATH"
echo

# 检测GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    DEVICE_ARG="--device auto"
else
    echo "🖥️  使用CPU模式"
    DEVICE_ARG="--device cpu"
fi

echo
echo "🚀 开始处理..."

# 执行风格迁移
python3 modern_train.py \
    --content "$CONTENT_PATH" \
    --style "$STYLE_PATH" \
    --output "$OUTPUT_PATH" \
    --steps 800 \
    --max-size 512 \
    --style-weight 1e6 \
    --content-weight 1 \
    --tv-weight 1e-3 \
    --lr 0.01 \
    --optimizer adam \
    --save-every 200 \
    $DEVICE_ARG

# 检查结果
if [ -f "$OUTPUT_PATH" ]; then
    echo
    echo "🎉 风格迁移完成！"
    echo "📁 结果保存至: $OUTPUT_PATH"
    
    # 显示文件大小
    FILE_SIZE=$(ls -lh "$OUTPUT_PATH" | awk '{print $5}')
    echo "📏 文件大小: $FILE_SIZE"
    
    # 如果有图像查看器，尝试打开结果
    if command -v eog &> /dev/null; then
        echo "🖼️  正在打开结果图像..."
        eog "$OUTPUT_PATH" &
    elif command -v display &> /dev/null; then
        echo "🖼️  正在打开结果图像..."
        display "$OUTPUT_PATH" &
    elif command -v open &> /dev/null; then
        echo "🖼️  正在打开结果图像..."
        open "$OUTPUT_PATH" &
    fi
else
    echo "❌ 处理失败，未生成输出文件"
    exit 1
fi
