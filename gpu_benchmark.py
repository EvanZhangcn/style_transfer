#!/usr/bin/env python3
"""
GPU性能测试脚本
测试不同参数设置下的性能表现
"""

import torch
import time
import os
import logging
from train import style_transfer, setup_logging, get_device

def benchmark_gpu():
    """
    GPU性能基准测试
    """
    print("=== GPU性能基准测试 ===")
    
    # 设置日志
    setup_logging("logs/gpu_benchmark.log")
    
    # 获取设备信息
    device = get_device()
    
    if device.type != 'cuda':
        print("错误: 需要GPU来运行性能测试")
        return
    
    # 测试配置
    test_configs = [
        {
            'name': '小图像_快速',
            'max_size': 256,
            'steps': 100,
            'style_weight': 1e5,
        },
        {
            'name': '中等图像_标准',
            'max_size': 512,
            'steps': 500,
            'style_weight': 1e6,
        },
        {
            'name': '大图像_高质量',
            'max_size': 1024,
            'steps': 1000,
            'style_weight': 1e6,
        }
    ]
    
    # 使用示例图像
    content_path = "images/content/tubingen.jpg"
    style_path = "images/style/starry_night.jpg"
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"错误: 找不到测试图像")
        print(f"内容图像: {content_path}")
        print(f"风格图像: {style_path}")
        return
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n=== 测试 {i+1}/{len(test_configs)}: {config['name']} ===")
        
        output_path = f"results/benchmark_{config['name']}.jpg"
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        try:
            success = style_transfer(
                content_path=content_path,
                style_path=style_path,
                output_path=output_path,
                device=device,
                num_steps=config['steps'],
                style_weight=config['style_weight'],
                max_size=config['max_size'],
                show_every=50,
                save_every=999999  # 不保存中间结果
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if success:
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                avg_time_per_step = total_time / config['steps']
                
                result = {
                    'config': config['name'],
                    'success': True,
                    'total_time': total_time,
                    'avg_time_per_step': avg_time_per_step,
                    'peak_memory_gb': peak_memory,
                    'max_size': config['max_size'],
                    'steps': config['steps']
                }
                
                print(f"成功完成!")
                print(f"总时间: {total_time:.1f}秒")
                print(f"平均每步: {avg_time_per_step:.3f}秒")
                print(f"峰值GPU内存: {peak_memory:.2f}GB")
                
            else:
                result = {'config': config['name'], 'success': False}
                print(f"测试失败!")
                
        except Exception as e:
            print(f"测试出错: {e}")
            result = {'config': config['name'], 'success': False, 'error': str(e)}
        
        results.append(result)
    
    # 输出测试总结
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    
    for result in results:
        if result['success']:
            print(f"{result['config']:15s}: "
                  f"{result['total_time']:6.1f}s, "
                  f"{result['avg_time_per_step']:6.3f}s/step, "
                  f"{result['peak_memory_gb']:5.2f}GB")
        else:
            print(f"{result['config']:15s}: 失败")
    
    return results

def test_different_optimizers():
    """
    测试不同优化器的性能
    """
    print("\n=== 优化器性能对比 ===")
    
    device = get_device()
    if device.type != 'cuda':
        print("跳过: 需要GPU")
        return
    
    optimizers = ['adam', 'lbfgs']
    content_path = "images/content/tubingen.jpg"
    style_path = "images/style/starry_night.jpg"
    
    for opt in optimizers:
        print(f"\n测试优化器: {opt}")
        
        output_path = f"results/optimizer_test_{opt}.jpg"
        
        torch.cuda.empty_cache()
        start_time = time.time()
        
        try:
            success = style_transfer(
                content_path=content_path,
                style_path=style_path,
                output_path=output_path,
                device=device,
                num_steps=200,
                style_weight=1e6,
                max_size=512,
                optimizer_type=opt,
                show_every=50,
                save_every=999999
            )
            
            if success:
                total_time = time.time() - start_time
                print(f"{opt}: {total_time:.1f}秒")
            else:
                print(f"{opt}: 失败")
                
        except Exception as e:
            print(f"{opt}: 错误 - {e}")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 运行基准测试
    benchmark_gpu()
    
    # 测试优化器
    test_different_optimizers()
    
    print("\n=== 测试完成 ===")
    print("结果文件保存在 results/ 目录")
    print("日志文件保存在 logs/ 目录")
