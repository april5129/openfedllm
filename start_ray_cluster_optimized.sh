#!/bin/bash

# 针对低内存服务器优化的Ray集群启动脚本
# 适用于3.7GB内存的腾讯云服务器

echo "🚀 启动优化的Ray集群（低内存配置）..."

# 设置环境变量
export PYTHONPATH="/root/openfedllm:$PYTHONPATH"
export RAY_DISABLE_IMPORT_WARNING=1

# 停止现有的Ray进程
echo "停止现有Ray进程..."
ray stop

# 清理临时文件
echo "清理临时文件..."
rm -rf /tmp/ray/session_latest
mkdir -p /tmp/ray/spill

# 设置系统限制
echo "设置系统限制..."
ulimit -n 65536

# 启动Ray集群（低内存配置）
echo "启动Ray集群（低内存配置）..."
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --object-store-memory=500000000 \
    --memory=1000000000 \
    --num-cpus=2 \
    --temp-dir=/tmp/ray \
    --system-config='{"object_spilling_config": "{\"type\": \"filesystem\", \"params\": {\"directory_path\": \"/tmp/ray/spill\"}}", "task_retry_delay_ms": 1000, "object_timeout_milliseconds": 1000}'

# 等待集群启动
echo "等待集群启动..."
sleep 10

# 检查集群状态
echo "Ray集群状态："
ray status

echo "✅ Ray集群启动完成！"
echo "📊 Ray Dashboard: http://$(hostname -I | awk '{print $1}'):8265" 