#!/bin/bash

# 分布式联邦学习训练启动脚本

echo "启动分布式联邦学习训练..."

# 检查Ray是否已安装
if ! python -c "import ray" &> /dev/null; then
    echo "Ray未安装，正在安装..."
    pip install -r requirements_ray.txt
fi

# 停止现有的Ray进程
ray stop

# 启动Ray集群
echo "启动Ray集群..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# 等待集群启动
sleep 5

# 显示集群状态
echo "Ray集群状态："
ray status

# 运行分布式联邦学习训练
echo "开始分布式联邦学习训练..."
python runner/FedFT/distributed_fedavg_runner.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "lucasmccabe-lmi/CodeAlpaca-20k" \
    --num_rounds 10 \
    --num_clients 4 \
    --sample_clients 2 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --seq_length 512 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --output_dir "output/distributed_fedavg" \
    --use_peft \
    --peft_lora_r 8 \
    --peft_lora_alpha 16 \
    --template "alpaca" \
    --dataset_sample 1000

echo "分布式联邦学习训练完成！"

# 停止Ray集群
ray stop 