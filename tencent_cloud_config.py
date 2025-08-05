#!/usr/bin/env python3
"""
腾讯云服务器专用配置
针对腾讯云服务器的硬件配置优化
"""

import os
import torch

# 腾讯云服务器配置
TENCENT_CLOUD_CONFIG = {
    # 服务器基本信息
    "server_ip": "106.52.36.202",
    "server_user": "root",
    "server_password": "@Dsdq0722",
    
    # 硬件配置（根据实际服务器配置调整）
    "cpu_cores": 8,  # CPU核心数
    "memory_gb": 32,  # 内存大小
    "gpu_count": 1,   # GPU数量（如果有）
    "gpu_memory_gb": 8,  # 每个GPU内存
    
    # Ray集群配置
    "ray_head_port": 6379,
    "ray_dashboard_port": 8265,
    "ray_object_store_memory": "2000000000",  # 2GB
    "ray_memory": "4000000000",  # 4GB
    
    # 训练配置
    "num_clients": 4,  # 客户端数量
    "sample_clients": 2,  # 每轮参与训练的客户端数
    "num_rounds": 50,  # 训练轮数
    "batch_size": 8,  # 批次大小
    "learning_rate": 2e-5,  # 学习率
    "max_steps": 100,  # 每轮最大步数
    
    # 模型配置
    "model_name": "meta-llama/Llama-2-7b-hf",  # 模型名称
    "use_peft": True,  # 使用PEFT
    "peft_lora_r": 8,  # LoRA rank
    "peft_lora_alpha": 16,  # LoRA alpha
    
    # 数据集配置
    "dataset_name": "lucasmccabe-lmi/CodeAlpaca-20k",
    "dataset_sample": 5000,  # 数据集采样数量
    
    # 输出配置
    "output_dir": "/root/openfedllm/output/tencent_cloud_training",
    "save_model_freq": 10,  # 保存模型频率
}

def get_optimized_training_args():
    """获取针对腾讯云服务器优化的训练参数"""
    return {
        "num_rounds": TENCENT_CLOUD_CONFIG["num_rounds"],
        "num_clients": TENCENT_CLOUD_CONFIG["num_clients"],
        "sample_clients": TENCENT_CLOUD_CONFIG["sample_clients"],
        "batch_size": TENCENT_CLOUD_CONFIG["batch_size"],
        "learning_rate": TENCENT_CLOUD_CONFIG["learning_rate"],
        "max_steps": TENCENT_CLOUD_CONFIG["max_steps"],
        "save_model_freq": TENCENT_CLOUD_CONFIG["save_model_freq"],
    }

def get_ray_config():
    """获取Ray集群配置"""
    return {
        "head_port": TENCENT_CLOUD_CONFIG["ray_head_port"],
        "dashboard_port": TENCENT_CLOUD_CONFIG["ray_dashboard_port"],
        "object_store_memory": TENCENT_CLOUD_CONFIG["ray_object_store_memory"],
        "memory": TENCENT_CLOUD_CONFIG["ray_memory"],
        "num_cpus": TENCENT_CLOUD_CONFIG["cpu_cores"],
        "num_gpus": TENCENT_CLOUD_CONFIG["gpu_count"] if torch.cuda.is_available() else 0,
    }

def get_model_config():
    """获取模型配置"""
    return {
        "model_name_or_path": TENCENT_CLOUD_CONFIG["model_name"],
        "use_peft": TENCENT_CLOUD_CONFIG["use_peft"],
        "peft_lora_r": TENCENT_CLOUD_CONFIG["peft_lora_r"],
        "peft_lora_alpha": TENCENT_CLOUD_CONFIG["peft_lora_alpha"],
        "dataset_name": TENCENT_CLOUD_CONFIG["dataset_name"],
        "dataset_sample": TENCENT_CLOUD_CONFIG["dataset_sample"],
        "output_dir": TENCENT_CLOUD_CONFIG["output_dir"],
    }

def create_tencent_cloud_runner_script():
    """创建腾讯云服务器专用的运行脚本"""
    script_content = f'''#!/bin/bash

# 腾讯云服务器专用分布式联邦学习运行脚本

echo "🚀 在腾讯云服务器上启动分布式联邦学习训练..."

# 设置环境变量
export PYTHONPATH="/root/openfedllm:$PYTHONPATH"
export RAY_DISABLE_IMPORT_WARNING=1

# 创建输出目录
mkdir -p {TENCENT_CLOUD_CONFIG["output_dir"]}

# 启动Ray集群
echo "启动Ray集群..."
ray start --head --port={TENCENT_CLOUD_CONFIG["ray_head_port"]} \\
    --dashboard-host=0.0.0.0 --dashboard-port={TENCENT_CLOUD_CONFIG["ray_dashboard_port"]} \\
    --object-store-memory={TENCENT_CLOUD_CONFIG["ray_object_store_memory"]} \\
    --memory={TENCENT_CLOUD_CONFIG["ray_memory"]}

# 等待集群启动
sleep 10

# 显示集群状态
echo "Ray集群状态："
ray status

# 运行分布式训练
echo "开始分布式联邦学习训练..."
python3 runner/FedFT/distributed_fedavg_runner.py \\
    --model_name_or_path "{TENCENT_CLOUD_CONFIG["model_name"]}" \\
    --dataset_name "{TENCENT_CLOUD_CONFIG["dataset_name"]}" \\
    --num_rounds {TENCENT_CLOUD_CONFIG["num_rounds"]} \\
    --num_clients {TENCENT_CLOUD_CONFIG["num_clients"]} \\
    --sample_clients {TENCENT_CLOUD_CONFIG["sample_clients"]} \\
    --learning_rate {TENCENT_CLOUD_CONFIG["learning_rate"]} \\
    --batch_size {TENCENT_CLOUD_CONFIG["batch_size"]} \\
    --seq_length 512 \\
    --num_train_epochs 1 \\
    --max_steps {TENCENT_CLOUD_CONFIG["max_steps"]} \\
    --output_dir "{TENCENT_CLOUD_CONFIG["output_dir"]}" \\
    --use_peft \\
    --peft_lora_r {TENCENT_CLOUD_CONFIG["peft_lora_r"]} \\
    --peft_lora_alpha {TENCENT_CLOUD_CONFIG["peft_lora_alpha"]} \\
    --template "alpaca" \\
    --dataset_sample {TENCENT_CLOUD_CONFIG["dataset_sample"]}

echo "✅ 分布式联邦学习训练完成！"

# 停止Ray集群
ray stop

echo "📊 训练结果保存在: {TENCENT_CLOUD_CONFIG["output_dir"]}"
echo "📈 查看Ray Dashboard: http://{TENCENT_CLOUD_CONFIG["server_ip"]}:{TENCENT_CLOUD_CONFIG["ray_dashboard_port"]}"
'''
    
    with open("run_tencent_cloud_training.sh", "w") as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod("run_tencent_cloud_training.sh", 0o755)
    print("✅ 腾讯云服务器专用运行脚本已创建: run_tencent_cloud_training.sh")

def create_monitoring_script():
    """创建监控脚本"""
    script_content = f'''#!/bin/bash

# 腾讯云服务器监控脚本

echo "📊 腾讯云服务器分布式训练监控"
echo "=================================="

# 检查Ray集群状态
echo "🔍 检查Ray集群状态..."
if pgrep -f "ray" > /dev/null; then
    echo "✅ Ray集群正在运行"
    ray status
else
    echo "❌ Ray集群未运行"
fi

# 检查GPU使用情况
echo ""
echo "🔍 检查GPU使用情况..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️  NVIDIA-SMI不可用"
fi

# 检查系统资源
echo ""
echo "🔍 检查系统资源..."
echo "CPU使用率:"
top -bn1 | grep "Cpu(s)" | awk '{{print $2}}' | cut -d'%' -f1

echo "内存使用情况:"
free -h

echo "磁盘使用情况:"
df -h

# 检查训练进程
echo ""
echo "🔍 检查训练进程..."
ps aux | grep -E "(python|ray)" | grep -v grep

# 检查输出目录
echo ""
echo "🔍 检查输出目录..."
if [ -d "{TENCENT_CLOUD_CONFIG["output_dir"]}" ]; then
    echo "✅ 输出目录存在"
    ls -la {TENCENT_CLOUD_CONFIG["output_dir"]}
else
    echo "❌ 输出目录不存在"
fi

echo ""
echo "📈 Ray Dashboard: http://{TENCENT_CLOUD_CONFIG["server_ip"]}:{TENCENT_CLOUD_CONFIG["ray_dashboard_port"]}"
'''
    
    with open("monitor_tencent_cloud.sh", "w") as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod("monitor_tencent_cloud.sh", 0o755)
    print("✅ 监控脚本已创建: monitor_tencent_cloud.sh")

if __name__ == "__main__":
    print("🔧 创建腾讯云服务器专用配置...")
    create_tencent_cloud_runner_script()
    create_monitoring_script()
    print("✅ 腾讯云服务器配置完成！") 