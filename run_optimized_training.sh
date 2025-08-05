#!/bin/bash

# 针对低内存服务器优化的联邦学习训练脚本
# 适用于3.7GB内存的腾讯云服务器

echo "🚀 启动优化的联邦学习训练（低内存配置）..."

# 设置环境变量
export PYTHONPATH="/root/openfedllm:$PYTHONPATH"
export RAY_DISABLE_IMPORT_WARNING=1

# 创建输出目录
mkdir -p /root/openfedllm/output/optimized_training

# 检查Ray集群状态
echo "检查Ray集群状态..."
ray status

# 运行优化的分布式训练
echo "开始优化的分布式联邦学习训练..."
python3 runner/FedFT/distributed_fedavg_runner.py \
    --model_name_or_path "microsoft/DialoGPT-medium" \
    --dataset_name "lucasmccabe-lmi/CodeAlpaca-20k" \
    --num_rounds 10 \
    --num_clients 2 \
    --sample_clients 1 \
    --learning_rate 5e-5 \
    --batch_size 4 \
    --seq_length 256 \
    --num_train_epochs 1 \
    --max_steps 50 \
    --output_dir "/root/openfedllm/output/optimized_training" \
    --use_peft \
    --peft_lora_r 4 \
    --peft_lora_alpha 8 \
    --template "alpaca" \
    --dataset_sample 1000 \
    --gradient_accumulation_steps 2 \
    --save_steps 25 \
    --logging_steps 10 \
    --warmup_steps 10 \
    --load_in_8bit

echo "✅ 优化的联邦学习训练完成！"

# 停止Ray集群
echo "停止Ray集群..."
ray stop

echo "📊 训练结果保存在: /root/openfedllm/output/optimized_training" 