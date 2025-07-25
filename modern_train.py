#!/usr/bin/env python3


import argparse
import torch
import os
import sys
from modern_style_transfer import ModernStyleTransfer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='现代神经网络风格迁移 (ConvNeXt)')
    
    # 基本参数
    parser.add_argument('--content', type=str, required=True,
                        help='内容图像路径')
    parser.add_argument('--style', type=str, required=True,
                        help='风格图像路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像路径')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=1000,
                        help='训练步数 (默认: 1000)')
    parser.add_argument('--max-size', type=int, default=512,
                        help='图像最大尺寸 (默认: 512)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 (默认: 0.01)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'lbfgs'],
                        help='优化器类型 (默认: adam)')
    
    # 损失权重
    parser.add_argument('--style-weight', type=float, default=1e6,
                        help='风格损失权重 (默认: 1e6)')
    parser.add_argument('--content-weight', type=float, default=1,
                        help='内容损失权重 (默认: 1)')
    parser.add_argument('--tv-weight', type=float, default=1e-3,
                        help='总变分损失权重 (默认: 1e-3)')
    
    # 其他参数
    parser.add_argument('--save-every', type=int, default=200,
                        help='保存中间结果的间隔步数 (默认: 200)')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备选择 (auto/cpu/cuda:0/cuda:1等)')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，减少输出')
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置计算设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🎮 自动选择GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("🖥️  使用CPU (未检测到可用GPU)")
    else:
        device = torch.device(device_arg)
        print(f"🔧 手动指定设备: {device}")
    
    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
    
    return device


def validate_paths(content_path, style_path, output_path):
    """验证文件路径"""
    if not os.path.exists(content_path):
        print(f"❌ 内容图像不存在: {content_path}")
        sys.exit(1)
    
    if not os.path.exists(style_path):
        print(f"❌ 风格图像不存在: {style_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")


def main():
    """主函数"""
    args = parse_args()
    
    print("🎨 现代神经网络风格迁移 (ConvNeXt-2022)")
    print("=" * 50)
    
    # 设置设备
    device = setup_device(args.device)
    
    # 验证路径
    validate_paths(args.content, args.style, args.output)
    
    # 显示参数
    if not args.quiet:
        print(f"📸 内容图像: {args.content}")
        print(f"🎭 风格图像: {args.style}")
        print(f"💾 输出路径: {args.output}")
        print(f"🔢 训练步数: {args.steps}")
        print(f"📏 最大尺寸: {args.max_size}")
        print(f"⚙️ 优化器: {args.optimizer}")
        print(f"📊 权重 - 风格: {args.style_weight}, 内容: {args.content_weight}, TV: {args.tv_weight}")
        print("-" * 50)
    
    try:
        # 创建风格迁移对象
        style_transfer = ModernStyleTransfer(device=device, max_size=args.max_size)
        
        # 执行风格迁移
        result_img, losses = style_transfer.transfer_style(
            content_path=args.content,
            style_path=args.style,
            output_path=args.output,
            num_steps=args.steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            tv_weight=args.tv_weight,
            lr=args.lr,
            optimizer_type=args.optimizer,
            save_every=args.save_every,
            show_progress=not args.quiet
        )
        
        print("🎉 风格迁移完成！")
        
        # 显示最终损失
        if not args.quiet and losses['total']:
            final_loss = losses['total'][-1]
            print(f"📈 最终损失: {final_loss:.2e}")
            print(f"📁 结果保存至: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
