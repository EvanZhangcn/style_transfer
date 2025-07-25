#!/bin/bash
# 批量风格迁移脚本

# 设置错误时退出
set -e

echo "=== 批量风格迁移 ==="
echo "开始时间: $(date)"

# 创建目录
mkdir -p results/batch logs

# 设置参数
CONTENT_DIR="images/content"
STYLE_DIR="images/style"
OUTPUT_DIR="results/batch"
LOG_DIR="logs"

# 检查目录是否存在
if [ ! -d "$CONTENT_DIR" ] || [ ! -d "$STYLE_DIR" ]; then
    echo "错误: 图像目录不存在"
    echo "请确保存在以下目录:"
    echo "  - $CONTENT_DIR"
    echo "  - $STYLE_DIR"
    exit 1
fi

# 获取图像文件列表
CONTENT_FILES=($(find "$CONTENT_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png"))
STYLE_FILES=($(find "$STYLE_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png"))

echo "找到 ${#CONTENT_FILES[@]} 个内容图像"
echo "找到 ${#STYLE_FILES[@]} 个风格图像"

if [ ${#CONTENT_FILES[@]} -eq 0 ] || [ ${#STYLE_FILES[@]} -eq 0 ]; then
    echo "错误: 没有找到图像文件"
    exit 1
fi

# 执行批量处理
total_combinations=$((${#CONTENT_FILES[@]} * ${#STYLE_FILES[@]}))
current=0

for content_file in "${CONTENT_FILES[@]}"; do
    content_name=$(basename "$content_file" | sed 's/\.[^.]*$//')
    
    for style_file in "${STYLE_FILES[@]}"; do
        style_name=$(basename "$style_file" | sed 's/\.[^.]*$//')
        
        current=$((current + 1))
        echo -e "\n=== 处理 $current/$total_combinations ==="
        echo "内容: $content_name"
        echo "风格: $style_name"
        
        # 输出文件名
        output_file="${OUTPUT_DIR}/${content_name}_styled_${style_name}.jpg"
        log_file="${LOG_DIR}/batch_${content_name}_${style_name}_$(date +%Y%m%d_%H%M%S).log"
        
        # 执行风格迁移
        python3 train.py \
            --content "$content_file" \
            --style "$style_file" \
            --output "$output_file" \
            --steps 1000 \
            --style-weight 1e6 \
            --content-weight 1 \
            --tv-weight 1e-3 \
            --lr 0.01 \
            --optimizer adam \
            --max-size 512 \
            --gpu \
            --log-file "$log_file" \
            --show-every 100 \
            --save-every 500
        
        echo "完成: $output_file"
    done
done

echo -e "\n=== 批量处理完成 ==="
echo "完成时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"

# 生成结果总结
echo -e "\n=== 结果总结 ==="
echo "生成的图像:"
find "$OUTPUT_DIR" -name "*.jpg" | sort
