#!/usr/bin/env python3
"""
本地系统测试脚本
不依赖网络连接，测试基本功能
"""

import os
import sys
import torch
import ray

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_system_info():
    """测试系统信息"""
    print("🔍 系统信息...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"当前目录: {os.getcwd()}")
    return True

def test_ray_local():
    """测试Ray本地功能"""
    print("🔍 测试Ray本地功能...")
    try:
        # 停止现有Ray进程
        ray.shutdown()
        
        # 启动本地Ray
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            object_store_memory=500000000,
            _memory=1000000000,
            local_mode=True  # 本地模式，不启动集群
        )
        
        # 测试简单的Ray任务
        @ray.remote
        def simple_task(x):
            return x * 2
        
        result = ray.get(simple_task.remote(5))
        print(f"✅ Ray本地任务测试成功: {result}")
        
        ray.shutdown()
        return True
    except Exception as e:
        print(f"❌ Ray本地测试失败: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    print("🔍 测试项目结构...")
    required_dirs = ['algo', 'runner', 'utils', 'dataset']
    required_files = ['config.py', 'requirements.txt']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ 目录存在: {dir_name}")
        else:
            print(f"❌ 目录缺失: {dir_name}")
            return False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"❌ 文件缺失: {file_name}")
            return False
    
    return True

def test_basic_imports():
    """测试基本导入（不依赖网络）"""
    print("🔍 测试基本导入...")
    try:
        import numpy as np
        import pandas as pd
        import transformers
        import peft
        import datasets
        print("✅ 核心依赖导入成功")
        return True
    except Exception as e:
        print(f"❌ 依赖导入失败: {e}")
        return False

def test_memory_usage():
    """测试内存使用"""
    print("🔍 测试内存使用...")
    try:
        # 创建一个简单的模型来测试内存
        import torch.nn as nn
        
        # 创建一个小的神经网络
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # 测试前向传播
        x = torch.randn(32, 100)
        output = model(x)
        print(f"✅ 模型测试成功，输出形状: {output.shape}")
        
        # 清理内存
        del model, x, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始本地系统测试...")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    # 运行测试
    tests = [
        test_system_info,
        test_project_structure,
        test_basic_imports,
        test_ray_local,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
        print("-" * 30)
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有本地测试通过！")
        print("💡 建议：")
        print("1. 配置网络代理或使用国内镜像")
        print("2. 下载预训练模型到本地")
        print("3. 准备本地数据集")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 