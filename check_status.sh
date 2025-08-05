#!/bin/bash

# 腾讯云服务器联邦学习项目状态检查脚本

echo "📊 腾讯云服务器联邦学习项目状态检查"
echo "======================================"

# 系统信息
echo "🔍 系统信息:"
echo "  操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  Python版本: $(python3 --version)"
echo "  CPU核心数: $(nproc)"
echo "  内存: $(cat /proc/meminfo | grep MemTotal | awk '{print $2/1024/1024 " GB"}')"
echo "  磁盘使用: $(df -h / | tail -1 | awk '{print $5}')"

# 检查项目文件
echo ""
echo "📁 项目文件检查:"
if [ -f "config.py" ]; then
    echo "  ✅ config.py 存在"
else
    echo "  ❌ config.py 缺失"
fi

if [ -f "requirements.txt" ]; then
    echo "  ✅ requirements.txt 存在"
else
    echo "  ❌ requirements.txt 缺失"
fi

# 检查目录
for dir in algo runner utils dataset; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ 存在"
    else
        echo "  ❌ $dir/ 缺失"
    fi
done

# 检查Ray集群状态
echo ""
echo "⚡ Ray集群状态:"
if command -v ray &> /dev/null; then
    echo "  ✅ Ray已安装"
    if ray status &> /dev/null; then
        echo "  ✅ Ray集群运行中"
        ray status | head -10
    else
        echo "  ⚠️  Ray集群未运行"
    fi
else
    echo "  ❌ Ray未安装"
fi

# 检查Python依赖
echo ""
echo "🐍 Python依赖检查:"
python3 -c "import torch; print('  ✅ PyTorch', torch.__version__)" 2>/dev/null || echo "  ❌ PyTorch未安装"
python3 -c "import transformers; print('  ✅ Transformers')" 2>/dev/null || echo "  ❌ Transformers未安装"
python3 -c "import peft; print('  ✅ PEFT')" 2>/dev/null || echo "  ❌ PEFT未安装"
python3 -c "import ray; print('  ✅ Ray', ray.__version__)" 2>/dev/null || echo "  ❌ Ray未安装"

# 检查输出目录
echo ""
echo "📂 输出目录检查:"
if [ -d "output" ]; then
    echo "  ✅ output/ 目录存在"
    echo "  内容:"
    ls -la output/ 2>/dev/null | head -5
else
    echo "  ⚠️  output/ 目录不存在"
fi

# 检查网络连接
echo ""
echo "🌐 网络连接检查:"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "  ✅ 网络连接正常"
else
    echo "  ❌ 网络连接异常"
fi

# 检查端口
echo ""
echo "🔌 端口检查:"
if netstat -tlnp 2>/dev/null | grep :8265; then
    echo "  ✅ Ray Dashboard端口8265开放"
else
    echo "  ⚠️  Ray Dashboard端口8265未开放"
fi

if netstat -tlnp 2>/dev/null | grep :6379; then
    echo "  ✅ Ray集群端口6379开放"
else
    echo "  ⚠️  Ray集群端口6379未开放"
fi

echo ""
echo "🎯 快速操作:"
echo "  启动集群: ./start_ray_cluster_optimized.sh"
echo "  运行训练: ./run_optimized_training.sh"
echo "  快速启动: ./quick_start.sh"
echo "  停止集群: ray stop" 