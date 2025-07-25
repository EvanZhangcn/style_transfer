"""
风格迁移训练脚本
优化版本，支持GPU加速和批处理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import logging
import sys
from pathlib import Path

def setup_logging(log_file=None):
    """
    设置日志记录
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def get_device():
    """
    智能设备选择
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
        logging.info(f"CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU计算")
    
    return device

def load_image(image_path, max_size=512, device='cpu'):
    """
    加载并预处理图像，增强版本
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        logging.info(f"加载图像: {image_path}, 原始尺寸: {image.size}")
    except Exception as e:
        raise ValueError(f"无法加载图像 {image_path}: {e}")
    
    # 智能尺寸调整
    original_size = max(image.size)
    if original_size > max_size:
        size = max_size
        logging.info(f"图像尺寸从 {original_size} 调整到 {size}")
    else:
        size = original_size
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor, path):
    """
    保存张量为图像，改进版本
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    tensor = tensor.cpu().clone().squeeze(0)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像并保存
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(path, quality=95)  # 高质量保存
    logging.info(f"图像保存到: {path}")

class VGGFeatureExtractor(nn.Module):
    """
    使用VGG19提取特征，GPU优化版本
    """
    def __init__(self, device='cpu'):
        super(VGGFeatureExtractor, self).__init__()
        
        # 加载预训练的VGG19
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            logging.info("VGG19模型加载成功")
        except Exception as e:
            logging.error(f"VGG19模型加载失败: {e}")
            raise
        
        self.model = vgg.to(device).eval()
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 定义要提取特征的层
        self.feature_layers = {
            '0': 'conv1_1',   # 风格层
            '5': 'conv2_1',   # 风格层  
            '10': 'conv3_1',  # 风格层
            '19': 'conv4_1',  # 风格层和内容层
            '21': 'conv4_2',  # 内容层
            '28': 'conv5_1'   # 风格层
        }
        
        # 预热GPU（如果使用GPU）
        if device.type == 'cuda':
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = self.forward(dummy_input)
            logging.info("GPU预热完成")
    
    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.feature_layers:
                features[self.feature_layers[name]] = x
        return features

def gram_matrix(tensor):
    """
    计算格拉姆矩阵
    """
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)

def style_transfer(content_path, style_path, output_path, device='cpu', 
                  num_steps=1000, style_weight=1e6, content_weight=1, 
                  tv_weight=1e-3, lr=0.01, show_every=200, save_every=500,
                  optimizer_type='adam', max_size=512):
    """
    执行风格迁移，GPU优化版本
    """
    logging.info("="*60)
    logging.info("开始风格迁移")
    logging.info(f"设备: {device}")
    logging.info(f"内容图像: {content_path}")
    logging.info(f"风格图像: {style_path}")
    logging.info(f"输出路径: {output_path}")
    logging.info(f"训练步数: {num_steps}")
    logging.info(f"最大尺寸: {max_size}")
    logging.info("="*60)
    
    # 记录GPU内存使用情况
    if device.type == 'cuda':
        logging.info(f"GPU内存使用: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    # 加载图像
    try:
        content_img = load_image(content_path, max_size=max_size, device=device)
        style_img = load_image(style_path, max_size=max_size, device=device)
    except Exception as e:
        logging.error(f"图像加载失败: {e}")
        return False
    
    logging.info(f"内容图像尺寸: {content_img.shape}")
    logging.info(f"风格图像尺寸: {style_img.shape}")
    
    # 初始化生成图像（从内容图像开始）
    generated_img = content_img.clone().requires_grad_(True)
    
    # 加载VGG模型
    try:
        vgg = VGGFeatureExtractor(device)
    except Exception as e:
        logging.error(f"VGG模型加载失败: {e}")
        return False
    
    # 获取目标特征
    logging.info("计算目标特征...")
    with torch.no_grad():
        content_features = vgg(content_img)
        style_features = vgg(style_img)
    
    # 计算风格目标（格拉姆矩阵）
    style_targets = {}
    for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
        style_targets[layer] = gram_matrix(style_features[layer])
    
    # 内容目标
    content_target = content_features['conv4_2']
    
    # 优化器选择
    if optimizer_type.lower() == 'lbfgs':
        optimizer = optim.LBFGS([generated_img], lr=lr, max_iter=20)
        logging.info("使用LBFGS优化器")
    else:
        optimizer = optim.Adam([generated_img], lr=lr)
        logging.info(f"使用Adam优化器，学习率: {lr}")
    
    logging.info("开始训练...")
    start_time = time.time()
    best_loss = float('inf')
    
    # 训练循环
    for step in range(num_steps):
        def closure():
            optimizer.zero_grad()
            
            # 获取生成图像的特征
            generated_features = vgg(generated_img)
            
            # 计算内容损失
            content_loss = torch.nn.functional.mse_loss(
                generated_features['conv4_2'], content_target)
            
            # 计算风格损失
            style_loss = 0
            for layer in style_targets:
                generated_gram = gram_matrix(generated_features[layer])
                style_loss += torch.nn.functional.mse_loss(
                    generated_gram, style_targets[layer])
            
            # 计算总变分损失（平滑性）
            y_tv = torch.sum(torch.abs(generated_img[:, :, 1:, :] - generated_img[:, :, :-1, :]))
            x_tv = torch.sum(torch.abs(generated_img[:, :, :, 1:] - generated_img[:, :, :, :-1]))
            tv_loss = x_tv + y_tv
            
            # 总损失
            total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
            
            total_loss.backward()
            return total_loss
        
        # 优化步骤
        if optimizer_type.lower() == 'lbfgs':
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()
        
        # 限制像素值
        with torch.no_grad():
            generated_img.clamp_(-2.5, 2.5)
        
        # 记录最佳损失
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
        
        # 打印进度
        if step % show_every == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (step + 1) * (num_steps - step - 1)
            
            # 分别计算各部分损失用于显示
            with torch.no_grad():
                generated_features = vgg(generated_img)
                content_loss_val = torch.nn.functional.mse_loss(
                    generated_features['conv4_2'], content_target).item()
                
                style_loss_val = 0
                for layer in style_targets:
                    generated_gram = gram_matrix(generated_features[layer])
                    style_loss_val += torch.nn.functional.mse_loss(
                        generated_gram, style_targets[layer]).item()
                
                y_tv = torch.sum(torch.abs(generated_img[:, :, 1:, :] - generated_img[:, :, :-1, :]))
                x_tv = torch.sum(torch.abs(generated_img[:, :, :, 1:] - generated_img[:, :, :, :-1]))
                tv_loss_val = (x_tv + y_tv).item()
            
            logging.info(f"步骤 {step:4d}/{num_steps}: "
                        f"总损失={current_loss:.2f}, "
                        f"内容={content_loss_val:.2f}, "
                        f"风格={style_loss_val:.2e}, "
                        f"TV={tv_loss_val:.2e}, "
                        f"时间={elapsed_time:.1f}s, "
                        f"ETA={eta:.1f}s")
            
            # GPU内存使用情况
            if device.type == 'cuda' and step % (show_every * 2) == 0:
                logging.info(f"GPU内存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # 保存中间结果
        if step > 0 and step % save_every == 0:
            temp_path = output_path.replace('.jpg', f'_step_{step}.jpg')
            save_image(generated_img, temp_path)
            logging.info(f"保存中间结果: {temp_path}")
        
        # 清理GPU缓存
        if device.type == 'cuda' and step % 100 == 0:
            torch.cuda.empty_cache()
    
    # 保存最终结果
    save_image(generated_img, output_path)
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info("风格迁移完成！")
    logging.info(f"总用时: {total_time:.1f}秒")
    logging.info(f"平均每步: {total_time/num_steps:.3f}秒")
    logging.info(f"最佳损失: {best_loss:.2f}")
    logging.info(f"结果保存至: {output_path}")
    logging.info("="*60)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='PyTorch Neural Style Transfer - GPU优化版本')
    parser.add_argument('--content', type=str, required=True,
                      help='内容图像路径')
    parser.add_argument('--style', type=str, required=True,
                      help='风格图像路径')
    parser.add_argument('--output', type=str, default='results/output.jpg',
                      help='输出图像路径')
    parser.add_argument('--steps', type=int, default=1000,
                      help='训练步数')
    parser.add_argument('--style-weight', type=float, default=1e6,
                      help='风格损失权重')
    parser.add_argument('--content-weight', type=float, default=1,
                      help='内容损失权重')
    parser.add_argument('--tv-weight', type=float, default=1e-3,
                      help='总变分损失权重')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'lbfgs'],
                      help='优化器类型')
    parser.add_argument('--max-size', type=int, default=512,
                      help='图像最大尺寸')
    parser.add_argument('--gpu', action='store_true',
                      help='强制使用GPU')
    parser.add_argument('--cpu', action='store_true',
                      help='强制使用CPU')
    parser.add_argument('--log-file', type=str, 
                      help='日志文件路径')
    parser.add_argument('--save-every', type=int, default=500,
                      help='中间结果保存间隔')
    parser.add_argument('--show-every', type=int, default=100,
                      help='进度显示间隔')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_file)
    
    # 设备选择
    if args.cpu:
        device = torch.device('cpu')
        logging.info("强制使用CPU")
    elif args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.warning("GPU不可用，使用CPU")
            device = torch.device('cpu')
    else:
        device = get_device()
    
    # 检查输入文件
    if not os.path.exists(args.content):
        logging.error(f"内容图像不存在: {args.content}")
        return
    
    if not os.path.exists(args.style):
        logging.error(f"风格图像不存在: {args.style}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 记录参数
    logging.info("训练参数:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # 执行风格迁移
    success = style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        device=device,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
        optimizer_type=args.optimizer,
        max_size=args.max_size,
        show_every=args.show_every,
        save_every=args.save_every
    )
    
    if success:
        logging.info("程序正常结束")
    else:
        logging.error("程序异常结束")
        sys.exit(1)

if __name__ == '__main__':
    main()
