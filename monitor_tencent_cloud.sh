#!/bin/bash

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
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

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
if [ -d "/root/openfedllm/output/tencent_cloud_training" ]; then
    echo "✅ 输出目录存在"
    ls -la /root/openfedllm/output/tencent_cloud_training
else
    echo "❌ 输出目录不存在"
fi

echo ""
echo "📈 Ray Dashboard: http://106.52.36.202:8265" 