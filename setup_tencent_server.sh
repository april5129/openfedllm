#!/bin/bash

# 腾讯云服务器直接设置脚本
# 在腾讯云服务器上运行此脚本

echo "🚀 在腾讯云服务器上设置分布式联邦学习环境..."

# 检查是否在腾讯云服务器上
if [[ $(hostname) != *"VM-20-3-opencloudos"* ]]; then
    echo "⚠️  请确保在腾讯云服务器上运行此脚本"
    exit 1
fi

echo "✅ 确认在腾讯云服务器上运行"

# 更新系统
echo "📦 更新系统包..."
yum update -y

# 安装基础工具
echo "🔧 安装基础工具..."
yum install -y git wget curl python3 python3-pip

# 创建项目目录
echo "📁 创建项目目录..."
mkdir -p /root/openfedllm
cd /root/openfedllm

# 安装Python依赖
echo "🐍 安装Python依赖..."
pip3 install --upgrade pip

# 安装Ray分布式框架
echo "⚡ 安装Ray分布式框架..."
pip3 install "ray[default]>=2.9.0"

# 安装其他必要依赖
echo "📚 安装其他依赖..."
pip3 install torch>=2.0.0 torchvision torchaudio
pip3 install transformers>=4.31.0
pip3 install peft>=0.4.0
pip3 install trl>=0.7.0
pip3 install accelerate>=0.21.0
pip3 install datasets>=2.13.0
pip3 install tokenizers>=0.13.0
pip3 install numpy>=1.25.0
pip3 install pandas>=2.1.0
pip3 install scipy>=1.11.0
pip3 install tqdm>=4.66.0
pip3 install psutil>=5.9.0
pip3 install rich>=13.6.0
pip3 install tyro>=0.5.0

# 创建Ray集群配置
echo "⚙️  配置Ray集群..."
mkdir -p /tmp/ray/session_latest/logs

# 设置环境变量
echo "🔧 设置环境变量..."
echo 'export PYTHONPATH="/root/openfedllm:$PYTHONPATH"' >> ~/.bashrc
echo 'export RAY_DISABLE_IMPORT_WARNING=1' >> ~/.bashrc
source ~/.bashrc

echo "✅ 腾讯云服务器环境设置完成！"

# 显示系统信息
echo ""
echo "📊 系统信息："
echo "主机名: $(hostname)"
echo "操作系统: $(cat /etc/os-release | grep PRETTY_NAME)"
echo "CPU核心数: $(nproc)"
echo "内存: $(free -h | grep Mem | awk '{print $2}')"
echo "Python版本: $(python3 --version)"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
else
    echo "GPU: 未检测到NVIDIA GPU"
fi

echo ""
echo "🎯 下一步："
echo "1. 上传项目文件到 /root/openfedllm/"
echo "2. 运行测试: python3 test_ray_installation.py"
echo "3. 启动训练: ./run_tencent_cloud_training.sh" 