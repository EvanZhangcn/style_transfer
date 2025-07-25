"""
Neural Style Transfer Implementation in PyTorch
Based on "Image Style Transfer Using Convolutional Neural Networks" (Gatys et al., CVPR 2015)
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

def hello():
    print("Hello from style_transfer.py!")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 图像预处理和后处理函数
def preprocess(image, size=512):
    """
    预处理图像：调整大小、转换为张量、标准化
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def deprocess(tensor):
    """
    后处理张量：反标准化、转换为PIL图像
    """
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    tensor = tensor.clone().squeeze(0).cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # 限制像素值范围
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    transform = transforms.ToPILImage()
    return transform(tensor)

class VGGFeatureExtractor(nn.Module):
    """
    使用预训练的VGG19模型提取特征
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # 加载预训练的VGG19模型
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # 我们只需要前30层
        self.features = nn.Sequential()
        for i in range(30):
            self.features.add_module(str(i), vgg[i])
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        返回指定层的特征图
        """
        features = {}
        for name, layer in self.features.named_children():
            x = layer(x)
            if name in ['0', '5', '10', '19', '28']:  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
                features[name] = x
        return features

def gram_matrix(tensor):
    """
    计算格拉姆矩阵，用于表示风格特征
    """
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)

def content_loss(target_feature, generated_feature):
    """
    计算内容损失
    """
    return nn.functional.mse_loss(generated_feature, target_feature)

def style_loss(target_gram, generated_feature):
    """
    计算风格损失
    """
    generated_gram = gram_matrix(generated_feature)
    return nn.functional.mse_loss(generated_gram, target_gram)

def total_variation_loss(image):
    """
    计算总变分损失，用于图像平滑
    """
    batch_size, channels, height, width = image.size()
    
    # 水平方向的变分
    h_variation = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    
    # 垂直方向的变分
    w_variation = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    
    return (h_variation + w_variation) / (batch_size * channels * height * width)

class StyleTransfer:
    """
    风格迁移主类
    """
    def __init__(self, content_weight=1e4, style_weight=1e-2, tv_weight=1e-7):
        self.device = device
        self.vgg = VGGFeatureExtractor().to(device)
        
        # 损失权重
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # 内容层和风格层
        self.content_layers = ['19']  # conv4_1
        self.style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    
    def load_image(self, image_path, size=512):
        """
        加载并预处理图像
        """
        image = Image.open(image_path).convert('RGB')
        return preprocess(image, size).to(self.device)
    
    def get_style_targets(self, style_image):
        """
        获取风格图像的目标特征
        """
        with torch.no_grad():
            style_features = self.vgg(style_image)
            style_targets = {}
            for layer in self.style_layers:
                style_targets[layer] = gram_matrix(style_features[layer])
        return style_targets
    
    def get_content_targets(self, content_image):
        """
        获取内容图像的目标特征
        """
        with torch.no_grad():
            content_features = self.vgg(content_image)
            content_targets = {}
            for layer in self.content_layers:
                content_targets[layer] = content_features[layer]
        return content_targets
    
    def compute_loss(self, generated_image, content_targets, style_targets):
        """
        计算总损失
        """
        generated_features = self.vgg(generated_image)
        
        # 内容损失
        content_loss_value = 0
        for layer in self.content_layers:
            content_loss_value += content_loss(content_targets[layer], 
                                             generated_features[layer])
        
        # 风格损失
        style_loss_value = 0
        for layer in self.style_layers:
            style_loss_value += style_loss(style_targets[layer], 
                                         generated_features[layer])
        
        # 总变分损失
        tv_loss_value = total_variation_loss(generated_image)
        
        # 总损失
        total_loss = (self.content_weight * content_loss_value + 
                     self.style_weight * style_loss_value + 
                     self.tv_weight * tv_loss_value)
        
        return total_loss, content_loss_value, style_loss_value, tv_loss_value
    
    def transfer(self, content_path, style_path, output_path=None, 
                num_epochs=500, learning_rate=0.01, size=512, save_steps=100):
        """
        执行风格迁移
        """
        # 加载图像
        content_image = self.load_image(content_path, size)
        style_image = self.load_image(style_path, size)
        
        # 获取目标特征
        content_targets = self.get_content_targets(content_image)
        style_targets = self.get_style_targets(style_image)
        
        # 初始化生成图像（从内容图像开始）
        generated_image = content_image.clone().requires_grad_(True)
        
        # 优化器
        optimizer = optim.LBFGS([generated_image], lr=learning_rate)
        
        # 显示原始图像
        self.show_images(content_image, style_image, "原始图像")
        
        epoch = [0]  # 使用列表以便在闭包中修改
        
        def closure():
            optimizer.zero_grad()
            
            # 限制像素值范围
            generated_image.data.clamp_(0, 1)
            
            # 计算损失
            total_loss, content_loss_val, style_loss_val, tv_loss_val = \
                self.compute_loss(generated_image, content_targets, style_targets)
            
            total_loss.backward()
            
            # 打印损失
            if epoch[0] % 50 == 0:
                print(f'Epoch {epoch[0]:4d}: '
                      f'Total Loss: {total_loss.item():.2f}, '
                      f'Content: {content_loss_val.item():.2f}, '
                      f'Style: {style_loss_val.item():.6f}, '
                      f'TV: {tv_loss_val.item():.6f}')
            
            # 保存中间结果
            if epoch[0] % save_steps == 0:
                self.show_generated_image(generated_image, f"Epoch {epoch[0]}")
            
            epoch[0] += 1
            return total_loss
        
        # 训练循环
        for i in range(num_epochs):
            optimizer.step(closure)
        
        # 最终结果
        generated_image.data.clamp_(0, 1)
        self.show_generated_image(generated_image, "最终结果")
        
        # 保存结果
        if output_path:
            result_image = deprocess(generated_image)
            result_image.save(output_path)
            print(f"结果已保存到: {output_path}")
        
        return generated_image
    
    def show_images(self, content_image, style_image, title):
        """
        显示内容图像和风格图像
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(deprocess(content_image))
        ax1.set_title("内容图像")
        ax1.axis('off')
        
        ax2.imshow(deprocess(style_image))
        ax2.set_title("风格图像")
        ax2.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_generated_image(self, generated_image, title):
        """
        显示生成的图像
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(deprocess(generated_image))
        plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    """
    主函数 - 演示风格迁移
    """
    # 创建风格迁移对象
    style_transfer = StyleTransfer(
        content_weight=1e4,
        style_weight=1e-2,
        tv_weight=1e-7
    )
    
    # 执行风格迁移
    # 注意：需要准备内容图像和风格图像
    content_path = "images/content.jpg"  # 替换为您的内容图像路径
    style_path = "images/style.jpg"      # 替换为您的风格图像路径
    output_path = "results/result.jpg"   # 输出路径
    
    if os.path.exists(content_path) and os.path.exists(style_path):
        style_transfer.transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            num_epochs=500,
            learning_rate=1.0,
            size=512
        )
    else:
        print("请先准备内容图像和风格图像文件")

if __name__ == "__main__":
    main()
