#!/usr/bin/env python3
"""
服务器环境检查脚本
在部署到GPU服务器前运行此脚本来检查环境
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_python():
    """检查Python版本"""
    print("=== Python环境检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查Python版本是否满足要求
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，需要3.7+")
        return False
    else:
        print("✅ Python版本满足要求")
        return True

def check_gpu():
    """检查GPU环境"""
    print("\n=== GPU环境检查 ===")
    
    # 检查nvidia-smi
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✅ nvidia-smi可用")
        print("GPU信息:")
        lines = output.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'GeForce' in line or 'Tesla' in line or 'RTX' in line or 'GTX' in line:
                print(f"  {line.strip()}")
    else:
        print("❌ nvidia-smi不可用，可能没有GPU或驱动未安装")
        return False
    
    # 检查CUDA
    success, output, error = run_command("nvcc --version")
    if success:
        print("✅ CUDA编译器可用")
        for line in output.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
    else:
        print("⚠️  CUDA编译器不可用（可能不影响PyTorch使用）")
    
    return True

def check_python_packages():
    """检查Python包"""
    print("\n=== Python包检查 ===")
    
    required_packages = {
        'torch': '1.9.0',
        'torchvision': '0.10.0',
        'PIL': '8.0.0',
        'matplotlib': '3.3.0',
        'numpy': '1.19.0',
        'requests': '2.25.0'
    }
    
    all_good = True
    
    for package, min_version in required_packages.items():
        try:
            if package == 'PIL':
                module = importlib.import_module('PIL')
                # PIL的版本检查方式不同
                version = getattr(module, '__version__', 'unknown')
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package}: {version}")
            
            # 特别检查PyTorch的CUDA支持
            if package == 'torch':
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    print(f"  ✅ CUDA支持: 可用")
                    print(f"  ✅ CUDA版本: {torch.version.cuda}")
                    print(f"  ✅ GPU数量: {torch.cuda.device_count()}")
                    if torch.cuda.device_count() > 0:
                        print(f"  ✅ GPU名称: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"  ❌ CUDA支持: 不可用")
                    all_good = False
                    
        except ImportError:
            print(f"❌ {package}: 未安装")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package}: 检查时出错 - {e}")
    
    return all_good

def check_project_files():
    """检查项目文件"""
    print("\n=== 项目文件检查 ===")
    
    required_files = [
        'train.py',
        'style_transfer.py',
        'demo.py',
        'requirements.txt',
        'run_gpu.sh',
        'batch_process.sh',
        'gpu_benchmark.py'
    ]
    
    required_dirs = [
        'images',
        'images/content',
        'images/style',
        'results'
    ]
    
    all_good = True
    
    # 检查文件
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 缺失")
            all_good = False
    
    # 检查目录
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/")
        else:
            print(f"❌ {dir}/ 缺失")
            all_good = False
    
    # 检查示例图像
    content_images = []
    style_images = []
    
    if os.path.exists('images/content'):
        content_images = [f for f in os.listdir('images/content') 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if os.path.exists('images/style'):
        style_images = [f for f in os.listdir('images/style') 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"📁 内容图像: {len(content_images)} 个")
    print(f"📁 风格图像: {len(style_images)} 个")
    
    if len(content_images) == 0 or len(style_images) == 0:
        print("⚠️  建议运行 python download_images.py 下载示例图像")
    
    return all_good

def check_permissions():
    """检查文件权限"""
    print("\n=== 文件权限检查 ===")
    
    script_files = ['run_gpu.sh', 'batch_process.sh']
    
    for script in script_files:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✅ {script} 可执行")
            else:
                print(f"⚠️  {script} 不可执行，运行: chmod +x {script}")
        else:
            print(f"❌ {script} 不存在")

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 基本功能测试 ===")
    
    try:
        import torch
        
        # 测试CPU张量
        x = torch.randn(10, 10)
        print("✅ CPU张量创建正常")
        
        # 测试GPU张量（如果可用）
        if torch.cuda.is_available():
            x_gpu = torch.randn(10, 10).cuda()
            print("✅ GPU张量创建正常")
            
            # 测试简单计算
            y = torch.mm(x_gpu, x_gpu)
            print("✅ GPU矩阵运算正常")
        else:
            print("⚠️  GPU不可用，跳过GPU测试")
        
        # 测试VGG模型加载
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        print("✅ VGG19模型加载正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 PyTorch风格迁移项目 - 环境检查")
    print("="*60)
    
    checks = [
        ("Python环境", check_python),
        ("GPU环境", check_gpu),
        ("Python包", check_python_packages),
        ("项目文件", check_project_files),
        ("文件权限", check_permissions),
        ("基本功能", test_basic_functionality)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}检查时出错: {e}")
            results.append((name, False))
    
    # 输出总结
    print("\n" + "="*60)
    print("📋 检查总结")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:12s}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有检查通过！可以开始风格迁移训练")
        print("\n推荐运行命令:")
        print("  # 快速测试")
        print("  python demo.py test")
        print("\n  # GPU训练")
        print("  ./run_gpu.sh")
        print("\n  # 性能测试")
        print("  python gpu_benchmark.py")
    else:
        print("⚠️  部分检查未通过，请修复后再运行")
        print("\n常见解决方案:")
        print("  # 安装PyTorch GPU版本")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n  # 下载示例图像")
        print("  python download_images.py")
        print("\n  # 设置脚本权限")
        print("  chmod +x *.sh")
    
    print("="*60)

if __name__ == "__main__":
    main()
