#!/bin/bash

# 腾讯云服务器联邦学习项目快速启动脚本

echo "🚀 腾讯云服务器联邦学习项目快速启动"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="/root/openfedllm:$PYTHONPATH"
export RAY_DISABLE_IMPORT_WARNING=1
export HF_ENDPOINT=https://hf-mirror.com

# 检查系统状态
echo "📋 检查系统状态..."
python3 test_minimal.py

if [ $? -ne 0 ]; then
    echo "❌ 系统检查失败，请检查配置"
    exit 1
fi

echo "✅ 系统检查通过"

# 停止现有Ray进程
echo "🛑 停止现有Ray进程..."
ray stop 2>/dev/null || true

# 启动Ray集群
echo "⚡ 启动Ray集群..."
./start_ray_cluster_optimized.sh

if [ $? -ne 0 ]; then
    echo "❌ Ray集群启动失败"
    exit 1
fi

echo "✅ Ray集群启动成功"

# 等待集群稳定
echo "⏳ 等待集群稳定..."
sleep 5

# 检查集群状态
echo "📊 检查集群状态..."
ray status

# 询问是否开始训练
echo ""
echo "🎯 是否开始联邦学习训练？"
echo "1. 是 - 开始训练"
echo "2. 否 - 仅启动集群"
read -p "请选择 (1/2): " choice

case $choice in
    1)
        echo "🚀 开始联邦学习训练..."
        ./run_optimized_training.sh
        ;;
    2)
        echo "✅ 集群已启动，可以手动运行训练"
        echo "运行命令: ./run_optimized_training.sh"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 快速启动完成！"
echo "📊 Ray Dashboard: http://$(hostname -I | awk '{print $1}'):8265"
echo "📁 项目目录: /root/openfedllm" 