#!/usr/bin/env python3
"""
最小化测试脚本
避免内存问题，只测试基本功能
"""

import os
import sys

def test_basic_system():
    """测试基本系统"""
    print("🔍 基本系统测试...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print(f"CPU核心数: {os.cpu_count()}")
    return True

def test_project_files():
    """测试项目文件"""
    print("🔍 项目文件测试...")
    files = ['config.py', 'requirements.txt']
    dirs = ['algo', 'runner', 'utils', 'dataset']
    
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f} 存在")
        else:
            print(f"❌ {f} 缺失")
            return False
    
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d}/ 存在")
        else:
            print(f"❌ {d}/ 缺失")
            return False
    
    return True

def test_basic_imports():
    """测试基本导入"""
    print("🔍 基本导入测试...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print("✅ Transformers")
        
        import peft
        print("✅ PEFT")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 最小化系统测试...")
    print("=" * 40)
    
    tests = [
        test_basic_system,
        test_project_files,
        test_basic_imports
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
        print("-" * 20)
    
    print(f"📊 测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 基本系统测试通过！")
        print("💡 系统已准备就绪，可以运行联邦学习项目。")
        return True
    else:
        print("⚠️  部分测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 