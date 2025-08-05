#!/bin/bash

# Ray集群启动脚本
# 用于分布式联邦学习

echo "启动Ray分布式集群..."

# 检查Ray是否已安装
if ! command -v ray &> /dev/null; then
    echo "Ray未安装，正在安装..."
    pip install -r requirements_ray.txt
fi

# 停止现有的Ray进程
ray stop

# 启动Ray集群
echo "启动Ray头节点..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# 等待头节点启动
sleep 5

# 显示集群状态
echo "Ray集群状态："
ray status

echo "Ray集群启动完成！"
echo "Dashboard地址: http://localhost:8265"
echo "要连接到集群，使用: ray start --address='localhost:6379'" 