"""
下载示例图像用于风格迁移
"""

import os
import requests
from PIL import Image
import io

def download_image(url, filename, folder="images"):
    """
    从URL下载图像并保存到本地
    """
    try:
        # 定义一个看起来像浏览器的请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 在请求中加入headers
        response = requests.get(url, headers=headers, timeout=30) # <--- 修改这里
        response.raise_for_status()
        
        # 创建文件夹（如果不存在）
        os.makedirs(folder, exist_ok=True)
        
        # 保存图像
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img.save(os.path.join(folder, filename))
        
        print(f"成功下载: {filename}")
        return True
    except Exception as e:
        print(f"下载失败 {filename}: {e}")
        return False
    
def download_sample_images():
    """
    下载一些示例图像
    """
    # 示例内容图像
    content_images = {
        "tubingen.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Tuebingen_Neckarfront.jpg/512px-Tuebingen_Neckarfront.jpg",
        "dancing.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/512px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
        "golden_gate.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/512px-GoldenGateBridge-001.jpg"
    }
    
    # 示例风格图像
    style_images = {
        "starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/512px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "the_scream.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/512px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
        "great_wave.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wave_off_Kanagawa.jpg/512px-The_Great_Wave_off_Kanagawa.jpg",
        "picasso.jpg": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Les_Demoiselles_d%27Avignon.jpg/512px-Les_Demoiselles_d%27Avignon.jpg"
    }
    
    print("开始下载内容图像...")
    success_count = 0
    for filename, url in content_images.items():
        if download_image(url, filename, "images/content"):
            success_count += 1
    
    print(f"\n内容图像下载完成: {success_count}/{len(content_images)}")
    
    print("\n开始下载风格图像...")
    success_count = 0
    for filename, url in style_images.items():
        if download_image(url, filename, "images/style"):
            success_count += 1
    
    print(f"\n风格图像下载完成: {success_count}/{len(style_images)}")

def create_sample_image():
    """
    如果下载失败，创建一个简单的示例图像
    """
    from PIL import Image, ImageDraw
    import random
    
    # 创建内容图像（简单的几何图形）
    content_img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(content_img)
    
    # 绘制一些简单的形状
    draw.rectangle([100, 100, 400, 400], fill='blue', outline='black', width=3)
    draw.ellipse([200, 200, 300, 300], fill='red')
    
    os.makedirs("images/content", exist_ok=True)
    content_img.save("images/content/sample_content.jpg")
    
    # 创建风格图像（随机噪声风格）
    style_img = Image.new('RGB', (512, 512))
    pixels = []
    for _ in range(512 * 512):
        pixels.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    style_img.putdata(pixels)
    
    os.makedirs("images/style", exist_ok=True)
    style_img.save("images/style/sample_style.jpg")
    
    print("创建了示例图像:")
    print("- images/content/sample_content.jpg")
    print("- images/style/sample_style.jpg")

if __name__ == "__main__":
    try:
        download_sample_images()
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        print("将创建示例图像...")
        create_sample_image()
