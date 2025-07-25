"""
风格迁移演示脚本
不需要命令行参数，直接运行即可
"""

import torch
import os
from train import style_transfer
from download_images import download_sample_images, create_sample_image

def demo():
    """
    演示风格迁移
    """
    print("=== PyTorch 风格迁移演示 ===\n")
    
    # 检查是否有示例图像
    content_dir = "images/content"
    style_dir = "images/style"
    
    if not os.path.exists(content_dir) or not os.path.exists(style_dir):
        print("准备示例图像...")
        try:
            download_sample_images()
        except:
            print("网络下载失败，创建本地示例图像...")
            create_sample_image()
    
    # 寻找可用的图像
    content_images = []
    style_images = []
    
    if os.path.exists(content_dir):
        content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if os.path.exists(style_dir):
        style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not content_images or not style_images:
        print("没有找到示例图像！请手动添加图像到以下文件夹：")
        print(f"- 内容图像: {content_dir}")
        print(f"- 风格图像: {style_dir}")
        return
    
    print(f"找到 {len(content_images)} 个内容图像")
    print(f"找到 {len(style_images)} 个风格图像")
    
    # 选择第一个可用的图像组合
    content_path = os.path.join(content_dir, content_images[0])
    style_path = os.path.join(style_dir, style_images[0])
    
    print(f"\n使用图像:")
    print(f"内容: {content_path}")
    print(f"风格: {style_path}")
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"计算设备: {device}")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 输出文件名
    content_name = os.path.splitext(content_images[0])[0]
    style_name = os.path.splitext(style_images[0])[0]
    output_path = f"results/{content_name}_styled_with_{style_name}.jpg"
    
    print(f"输出: {output_path}")
    print("\n开始风格迁移...\n")
    
    # 执行风格迁移（快速演示版本）
    try:
        style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            device=device,
            num_steps=500,          # 减少步数以便快速演示
            style_weight=1e6,
            content_weight=1,
            tv_weight=1e-3,
            lr=0.01,
            show_every=100,         # 更频繁地显示进度
            save_every=250          # 保存中间结果
        )
        
        print(f"\n演示完成！请查看结果: {output_path}")
        
    except Exception as e:
        print(f"风格迁移过程中出现错误: {e}")
        print("请检查图像文件是否正确，或尝试使用CPU模式")

def quick_test():
    """
    快速测试（只运行很少的步数）
    """
    print("=== 快速测试模式 ===\n")
    
    # 使用示例图像进行快速测试
    create_sample_image()
    
    content_path = "images/content/sample_content.jpg"
    style_path = "images/style/sample_style.jpg"
    output_path = "results/quick_test.jpg"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("运行快速测试（50步）...")
    
    try:
        style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            device=device,
            num_steps=50,           # 很少的步数用于测试
            style_weight=1e4,       # 降低权重以便快速收敛
            content_weight=1,
            tv_weight=1e-4,
            lr=0.1,                 # 更高的学习率
            show_every=10,
            save_every=25
        )
        
        print(f"快速测试完成！结果: {output_path}")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        demo()
