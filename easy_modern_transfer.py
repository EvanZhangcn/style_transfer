#!/usr/bin/env python3

import os
import torch
from modern_style_transfer import ModernStyleTransfer


def print_banner():
    """打印横幅"""
    print("🎨" + "=" * 60 + "🎨")
    print("        现代神经网络风格迁移工具 (ConvNeXt-2022)")
    print("                 Easy Style Transfer")
    print("🎨" + "=" * 60 + "🎨")
    print()


def check_directories():
    """检查并创建必要的目录"""
    dirs = ['images/content', 'images/style', 'results']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 创建目录: {dir_path}")


def list_files(directory, extensions=['.jpg', '.jpeg', '.png']):
    """列出目录中的图像文件"""
    if not os.path.exists(directory):
        return []
    
    files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            files.append(file)
    return sorted(files)


def select_file(directory, file_type):
    """选择文件的交互界面"""
    files = list_files(directory)
    
    if not files:
        print(f"❌ {directory} 目录中没有找到图像文件")
        print(f"请将{file_type}图像放入 {directory} 目录")
        return None
    
    print(f"\n📂 {directory} 中的文件:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file}")
    
    while True:
        try:
            choice = input(f"\n选择{file_type}图像 (1-{len(files)}) 或输入完整路径: ").strip()
            
            # 如果输入的是数字，选择列表中的文件
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return os.path.join(directory, files[idx])
                else:
                    print(f"❌ 请输入 1-{len(files)} 之间的数字")
            
            # 如果输入的是路径
            elif os.path.exists(choice):
                return choice
            
            else:
                print(f"❌ 文件不存在: {choice}")
                
        except ValueError:
            print("❌ 输入无效")


def select_mode():
    """选择处理模式"""
    modes = {
        1: {
            'name': '快速模式',
            'description': '30秒-1分钟, 384px, 300步',
            'steps': 300,
            'max_size': 384,
            'style_weight': 1e6,
            'lr': 0.02
        },
        2: {
            'name': '标准模式',
            'description': '2-3分钟, 512px, 800步',
            'steps': 800,
            'max_size': 512,
            'style_weight': 1e6,
            'lr': 0.01
        },
        3: {
            'name': '高质量模式',
            'description': '5-8分钟, 768px, 1500步',
            'steps': 1500,
            'max_size': 768,
            'style_weight': 1e6,
            'lr': 0.01
        }
    }
    
    print("\n🎯 选择处理模式:")
    for i, mode in modes.items():
        print(f"  {i}. {mode['name']} - {mode['description']}")
    
    while True:
        try:
            choice = int(input("\n请选择模式 (1-3): "))
            if choice in modes:
                return modes[choice]
            else:
                print("❌ 请输入 1-3 之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")


def generate_output_path(content_path, style_path):
    """生成输出文件路径"""
    content_name = os.path.splitext(os.path.basename(content_path))[0]
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    
    output_name = f"{content_name}_with_{style_name}_convnext.jpg"
    return os.path.join("results", output_name)


def main():
    """主函数"""
    print_banner()
    
    # 检查目录
    check_directories()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    print()
    
    # 选择内容图像
    content_path = select_file("images/content", "内容")
    if not content_path:
        return
    
    # 选择风格图像
    style_path = select_file("images/style", "风格")
    if not style_path:
        return
    
    # 选择处理模式
    mode = select_mode()
    
    # 生成输出路径
    output_path = generate_output_path(content_path, style_path)
    
    # 确认参数
    print(f"\n📋 处理参数:")
    print(f"  内容图像: {content_path}")
    print(f"  风格图像: {style_path}")
    print(f"  输出路径: {output_path}")
    print(f"  处理模式: {mode['name']}")
    print(f"  图像尺寸: {mode['max_size']}px")
    print(f"  训练步数: {mode['steps']}")
    
    confirm = input(f"\n确认开始处理? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ 取消处理")
        return
    
    try:
        # 创建风格迁移对象
        print(f"\n🚀 初始化ConvNeXt模型...")
        style_transfer = ModernStyleTransfer(device=device, max_size=mode['max_size'])
        
        # 执行风格迁移
        print(f"🎨 开始风格迁移 ({mode['name']})...")
        result_img, losses = style_transfer.transfer_style(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            num_steps=mode['steps'],
            style_weight=mode['style_weight'],
            content_weight=1,
            tv_weight=1e-3,
            lr=mode['lr'],
            optimizer_type='adam',
            save_every=mode['steps'] // 4,
            show_progress=True
        )
        
        print(f"\n🎉 风格迁移完成！")
        print(f"📁 结果保存至: {output_path}")
        
        # 询问是否继续
        again = input(f"\n是否继续处理其他图像? (y/N): ").strip().lower()
        if again in ['y', 'yes']:
            main()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()
