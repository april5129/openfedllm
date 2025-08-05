#!/usr/bin/env python3
"""
Ray分布式联邦学习安装测试脚本
"""

import sys
import subprocess

def test_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    
    try:
        import ray
        print(f"✅ Ray版本: {ray.__version__}")
    except ImportError as e:
        print(f"❌ Ray导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers版本: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers导入失败: {e}")
        return False
    
    try:
        import peft
        print(f"✅ PEFT版本: {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT导入失败: {e}")
        return False
    
    return True

def test_ray_basic():
    """测试Ray基本功能"""
    print("\n🔍 测试Ray基本功能...")
    
    try:
        import ray
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        print("✅ Ray初始化成功")
        
        # 测试远程函数
        @ray.remote
        def hello():
            return "Hello from Ray!"
        
        result = ray.get(hello.remote())
        print(f"✅ 远程函数测试: {result}")
        
        # 测试Actor
        @ray.remote
        class Counter:
            def __init__(self):
                self.value = 0
            
            def increment(self):
                self.value += 1
                return self.value
        
        counter = Counter.remote()
        result = ray.get(counter.increment.remote())
        print(f"✅ Actor测试: {result}")
        
        # 清理
        ray.shutdown()
        print("✅ Ray清理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Ray功能测试失败: {e}")
        return False

def test_gpu():
    """测试GPU支持"""
    print("\n🔍 测试GPU支持...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            
            # 测试GPU计算
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ GPU计算测试成功")
            
        else:
            print("⚠️  GPU不可用，将使用CPU模式")
            
        return True
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def test_ray_gpu():
    """测试Ray GPU支持"""
    print("\n🔍 测试Ray GPU支持...")
    
    try:
        import ray
        import torch
        
        if torch.cuda.is_available():
            # 初始化Ray with GPU
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_gpus=1)
            
            @ray.remote(num_gpus=0.5)
            def gpu_task():
                import torch
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                return torch.mm(x, y).sum().item()
            
            result = ray.get(gpu_task.remote())
            print(f"✅ Ray GPU任务测试: {result}")
            
            ray.shutdown()
        else:
            print("⚠️  GPU不可用，跳过Ray GPU测试")
            
        return True
        
    except Exception as e:
        print(f"❌ Ray GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Ray分布式联邦学习安装测试")
    print("=" * 50)
    
    # 测试Python版本
    print(f"Python版本: {sys.version}")
    
    # 运行所有测试
    tests = [
        test_imports,
        test_ray_basic,
        test_gpu,
        test_ray_gpu
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！Ray分布式联邦学习环境配置成功！")
        print("\n下一步：")
        print("1. 启动Ray集群: ./start_ray_cluster.sh")
        print("2. 运行分布式训练: ./run_distributed_fedavg.sh")
    else:
        print("⚠️  部分测试失败，请检查安装配置")
        print("\n建议：")
        print("1. 重新运行安装脚本: ./install_ray_dependencies.sh")
        print("2. 检查Python版本和CUDA版本")
        print("3. 使用虚拟环境避免依赖冲突")

if __name__ == "__main__":
    main() 