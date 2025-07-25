"""
基于ConvNeXt的现代神经网络风格迁移实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')


class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt特征提取器
    """
    
    def __init__(self, device='cpu'):
        super(ConvNeXtFeatureExtractor, self).__init__()
        self.device = device
        
        # 加载预训练的ConvNeXt模型
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.model = convnext_base(weights=weights).to(device)
        self.model.eval()
        
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # ConvNeXt的特征层定义
        # ConvNeXt分为4个stage，每个stage包含多个block
        self.feature_layers = {
            'stage1_block0': 'features.1.0',      # 早期特征，纹理
            'stage1_block2': 'features.1.2',      # 早期特征，纹理
            'stage2_block0': 'features.3.0',      # 中层特征，局部模式
            'stage2_block2': 'features.3.2',      # 中层特征，局部模式
            'stage3_block0': 'features.5.0',      # 高层特征，内容
            'stage3_block9': 'features.5.9',      # 高层特征，内容
            'stage4_block0': 'features.7.0',      # 最高层特征，语义
            'stage4_block2': 'features.7.2',      # 最高层特征，语义
        }
        
        # 风格层：多个层次的特征用于捕获不同尺度的纹理
        self.style_layers = [
            'stage1_block0',  # 低层纹理
            'stage1_block2',  # 低层纹理
            'stage2_block0',  # 中层模式
            'stage2_block2',  # 中层模式
            'stage3_block0',  # 高层模式
        ]
        
        # 内容层：高层特征用于保持语义内容
        self.content_layers = [
            'stage3_block9',  # 主要内容特征
        ]
        
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    def get_features(self, x, layers):
        """提取指定层的特征"""
        features = {}
        
        # ConvNeXt前向传播并提取中间特征
        x = self.model.features[0](x)  # stem layer
        
        # Stage 1
        for i, layer in enumerate(self.model.features[1]):
            x = layer(x)
            layer_name = f'stage1_block{i}'
            if layer_name in layers:
                features[layer_name] = x
        
        x = self.model.features[2](x)  # downsample
        
        # Stage 2
        for i, layer in enumerate(self.model.features[3]):
            x = layer(x)
            layer_name = f'stage2_block{i}'
            if layer_name in layers:
                features[layer_name] = x
        
        x = self.model.features[4](x)  # downsample
        
        # Stage 3
        for i, layer in enumerate(self.model.features[5]):
            x = layer(x)
            layer_name = f'stage3_block{i}'
            if layer_name in layers:
                features[layer_name] = x
        
        x = self.model.features[6](x)  # downsample
        
        # Stage 4
        for i, layer in enumerate(self.model.features[7]):
            x = layer(x)
            layer_name = f'stage4_block{i}'
            if layer_name in layers:
                features[layer_name] = x
        
        return features


class ModernStyleTransfer:
    """
    基于ConvNeXt的现代风格迁移类
    """
    
    def __init__(self, device='cpu', max_size=512):
        self.device = device
        self.max_size = max_size
        self.feature_extractor = ConvNeXtFeatureExtractor(device)
        
        print(f"🚀 Modern Style Transfer 初始化完成")
        print(f"📱 设备: {device}")
        print(f"🖼️  最大图像尺寸: {max_size}")
        print(f"🧠 特征提取器: ConvNeXt-Base (2022)")
        
    def load_image(self, image_path, size=None):
        """加载并预处理图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if size is None:
            size = min(self.max_size, max(image.size))
        
        # 保持宽高比的resize
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def save_image(self, tensor, path):
        """保存图像"""
        image = tensor.cpu().clone().detach()
        image = image.squeeze(0)
        image = torch.clamp(image, 0, 1)
        
        transform = transforms.ToPILImage()
        image = transform(image)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
    
    def gram_matrix(self, tensor):
        """
        计算Gram矩阵，用于捕获风格特征
        Gram矩阵能够捕获特征之间的相关性，代表纹理信息
        """
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def content_loss(self, target_features, content_features):
        """计算内容损失"""
        loss = 0
        for layer in self.feature_extractor.content_layers:
            target_feature = target_features[layer]
            content_feature = content_features[layer]
            loss += torch.mean((target_feature - content_feature) ** 2)
        return loss
    
    def style_loss(self, target_features, style_grams):
        """计算风格损失"""
        loss = 0
        for layer in self.feature_extractor.style_layers:
            target_feature = target_features[layer]
            target_gram = self.gram_matrix(target_feature)
            style_gram = style_grams[layer]
            
            # 归一化Gram矩阵
            _, d, h, w = target_feature.shape
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            layer_loss = layer_loss / (d * h * w)
            
            loss += layer_loss
        return loss
    
    def total_variation_loss(self, image):
        """
        总变分损失，用于图像平滑
        减少噪声，使生成的图像更自然
        """
        tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        return tv_h + tv_w
    
    def transfer_style(self, content_path, style_path, output_path=None,
                      num_steps=1000, style_weight=1e6, content_weight=1,
                      tv_weight=1e-3, lr=0.01, optimizer_type='adam',
                      save_every=100, show_progress=True):
        """
        执行风格迁移
        
        Args:
            content_path: 内容图像路径
            style_path: 风格图像路径
            output_path: 输出路径
            num_steps: 优化步数
            style_weight: 风格损失权重
            content_weight: 内容损失权重
            tv_weight: 总变分损失权重
            lr: 学习率
            optimizer_type: 优化器类型 ('adam', 'lbfgs')
            save_every: 每隔多少步保存中间结果
            show_progress: 是否显示进度
        """
        
        print("🎨 开始现代风格迁移...")
        start_time = time.time()
        
        # 加载图像
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        print(f"📸 内容图像尺寸: {content_img.shape}")
        print(f"🎭 风格图像尺寸: {style_img.shape}")
        
        # 初始化目标图像
        target_img = content_img.clone().requires_grad_(True)
        
        # 提取特征
        all_layers = self.feature_extractor.content_layers + self.feature_extractor.style_layers
        
        content_features = self.feature_extractor.get_features(content_img, all_layers)
        style_features = self.feature_extractor.get_features(style_img, all_layers)
        
        # 预计算风格特征的Gram矩阵
        style_grams = {}
        for layer in self.feature_extractor.style_layers:
            style_grams[layer] = self.gram_matrix(style_features[layer])
        
        # 设置优化器
        if optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS([target_img], lr=lr, max_iter=20)
        else:
            optimizer = optim.Adam([target_img], lr=lr)
        
        print(f"⚙️ 优化器: {optimizer.__class__.__name__}")
        print(f"📊 权重设置 - 风格: {style_weight}, 内容: {content_weight}, TV: {tv_weight}")
        
        # 损失记录
        losses = {
            'total': [],
            'content': [],
            'style': [],
            'tv': []
        }
        
        def closure():
            # 清除梯度
            optimizer.zero_grad()
            
            # 提取目标图像特征
            target_features = self.feature_extractor.get_features(target_img, all_layers)
            
            # 计算各项损失
            c_loss = self.content_loss(target_features, content_features)
            s_loss = self.style_loss(target_features, style_grams)
            tv_loss = self.total_variation_loss(target_img)
            
            # 加权总损失
            total_loss = (content_weight * c_loss + 
                         style_weight * s_loss + 
                         tv_weight * tv_loss)
            
            # 反向传播
            total_loss.backward()
            
            # 记录损失
            losses['total'].append(total_loss.item())
            losses['content'].append(c_loss.item())
            losses['style'].append(s_loss.item())
            losses['tv'].append(tv_loss.item())
            
            return total_loss
        
        # 训练循环
        for step in range(num_steps):
            if optimizer_type.lower() == 'lbfgs':
                optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            # 限制像素值范围
            with torch.no_grad():
                target_img.clamp_(0, 1)
            
            # 显示进度
            if show_progress and step % 50 == 0:
                current_loss = losses['total'][-1]
                elapsed = time.time() - start_time
                print(f"步骤 {step:4d}/{num_steps} | "
                      f"总损失: {current_loss:.2e} | "
                      f"耗时: {elapsed:.1f}s")
            
            # 保存中间结果
            if save_every > 0 and step > 0 and step % save_every == 0 and output_path:
                intermediate_path = output_path.replace('.', f'_step_{step}.')
                self.save_image(target_img, intermediate_path)
        
        # 保存最终结果
        if output_path:
            self.save_image(target_img, output_path)
            print(f"✅ 风格迁移完成！结果保存至: {output_path}")
        
        total_time = time.time() - start_time
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        
        return target_img, losses
    
    def plot_losses(self, losses):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(losses['total'])
        plt.title('Total Loss')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(losses['content'])
        plt.title('Content Loss')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(losses['style'])
        plt.title('Style Loss')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.plot(losses['tv'])
        plt.title('Total Variation Loss')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    现代风格迁移演示
    """
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 创建现代风格迁移对象
    style_transfer = ModernStyleTransfer(device=device, max_size=512)
    
    # 设置路径
    content_path = "images/content/photo.jpg"
    style_path = "images/style/artwork.jpg"
    output_path = "results/modern_result.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(content_path):
        print(f"❌ 内容图像不存在: {content_path}")
        print("请将内容图像放在 images/content/ 目录下")
        return
    
    if not os.path.exists(style_path):
        print(f"❌ 风格图像不存在: {style_path}")
        print("请将风格图像放在 images/style/ 目录下")
        return
    
    try:
        # 执行风格迁移
        result_img, losses = style_transfer.transfer_style(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            num_steps=800,
            style_weight=1e6,
            content_weight=1,
            tv_weight=1e-3,
            lr=0.01,
            optimizer_type='adam',
            save_every=200
        )
        
        # 绘制损失曲线
        style_transfer.plot_losses(losses)
        
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
