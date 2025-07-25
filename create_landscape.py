from PIL import Image, ImageDraw
import os

# 创建一个更有趣的内容图像
content_img = Image.new('RGB', (512, 512), (135, 206, 235))  # 天蓝色背景
draw = ImageDraw.Draw(content_img)

# 绘制太阳
draw.ellipse([350, 50, 450, 150], fill='yellow', outline='orange', width=3)

# 绘制山脉
points = [(0, 400), (100, 300), (200, 350), (300, 280), (400, 320), (512, 270), (512, 512), (0, 512)]
draw.polygon(points, fill='darkgreen')

# 绘制房子
draw.rectangle([150, 350, 250, 450], fill='brown', outline='black', width=2)
# 屋顶
draw.polygon([(140, 350), (200, 300), (260, 350)], fill='red', outline='black', width=2)
# 门
draw.rectangle([180, 400, 220, 450], fill=(101, 67, 33))
# 窗户
draw.rectangle([160, 370, 190, 390], fill='lightblue', outline='black', width=1)
draw.rectangle([210, 370, 240, 390], fill='lightblue', outline='black', width=1)

# 绘制树
draw.ellipse([50, 320, 120, 390], fill='green')
draw.rectangle([80, 390, 90, 450], fill='brown')

# 保存图像
os.makedirs('images/content', exist_ok=True)
content_img.save('images/content/landscape.jpg')
print('创建了新的内容图像: images/content/landscape.jpg')
