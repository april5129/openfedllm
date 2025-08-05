#!/bin/bash

# 腾讯云服务器专用分布式联邦学习运行脚本

echo "🚀 在腾讯云服务器上启动分布式联邦学习训练..."

# 设置环境变量
export PYTHONPATH="/root/openfedllm:$PYTHONPATH"
export RAY_DISABLE_IMPORT_WARNING=1

# 创建输出目录
mkdir -p /root/openfedllm/output/tencent_cloud_training

# 启动Ray集群
echo "启动Ray集群..."
ray start --head --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --object-store-memory=2000000000 \
    --memory=4000000000

# 等待集群启动
sleep 10

# 显示集群状态
echo "Ray集群状态："
ray status

# 运行分布式训练
echo "开始分布式联邦学习训练..."
python3 runner/FedFT/distributed_fedavg_runner.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "lucasmccabe-lmi/CodeAlpaca-20k" \
    --num_rounds 50 \
    --num_clients 4 \
    --sample_clients 2 \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --seq_length 512 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --output_dir "/root/openfedllm/output/tencent_cloud_training" \
    --use_peft \
    --peft_lora_r 8 \
    --peft_lora_alpha 16 \
    --template "alpaca" \
    --dataset_sample 5000

echo "✅ 分布式联邦学习训练完成！"

# 停止Ray集群
ray stop

echo "📊 训练结果保存在: /root/openfedllm/output/tencent_cloud_training"
echo "📈 查看Ray Dashboard: http://106.52.36.202:8265" 