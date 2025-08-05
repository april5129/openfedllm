#!/usr/bin/env python3
"""
简单的联邦学习测试脚本
用于验证系统是否正常工作
"""

import os
import sys
import torch
import ray
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    try:
        from algo.FedFT.base_client import BaseClient
        from algo.FedFT.fedavg.server import FedAvgServer
        from dataset.process_dataset import process_sft_dataset
        from utils.template import get_formatting_prompts_func
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_ray_cluster():
    """测试Ray集群"""
    print("🔍 测试Ray集群...")
    try:
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=2,
                object_store_memory=500000000,
                _memory=1000000000
            )
        print(f"✅ Ray集群状态: {ray.status()}")
        return True
    except Exception as e:
        print(f"❌ Ray集群测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    try:
        # 使用较小的模型进行测试
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 测试PEFT配置
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print("✅ 模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_dataset_processing():
    """测试数据集处理"""
    print("🔍 测试数据集处理...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
        print(f"✅ 数据集加载成功，样本数: {len(dataset)}")
        return True
    except Exception as e:
        print(f"❌ 数据集处理失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始联邦学习系统测试...")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    # 运行测试
    tests = [
        test_basic_imports,
        test_ray_cluster,
        test_model_loading,
        test_dataset_processing
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
        print("🎉 所有测试通过！系统准备就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        return False

if __name__ == "__main__":
    success = main()
    if ray.is_initialized():
        ray.shutdown()
    sys.exit(0 if success else 1) 