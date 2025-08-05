#!/bin/bash

# Ray分布式联邦学习依赖安装脚本
# 解决依赖冲突问题

echo "开始安装Ray分布式联邦学习依赖..."

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv ray_fed_env
    source ray_fen_env/bin/activate
    echo "虚拟环境已激活"
fi

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
pip install torch>=2.0.0 torchvision torchaudio
pip install transformers>=4.31.0
pip install peft>=0.4.0
pip install trl>=0.7.0
pip install accelerate>=0.21.0
pip install datasets>=2.13.0
pip install tokenizers>=0.13.0

# 安装Ray（使用兼容版本）
echo "安装Ray分布式框架..."
pip install "ray[default]>=2.9.0"

# 安装其他必要依赖
echo "安装其他依赖..."
pip install numpy>=1.25.0
pip install pandas>=2.1.0
pip install scipy>=1.11.0
pip install tqdm>=4.66.0
pip install psutil>=5.9.0
pip install rich>=13.6.0
pip install tyro>=0.5.0

# 验证安装
echo "验证安装..."
python3 -c "
import ray
import torch
import transformers
import peft
import trl
print('✅ 所有依赖安装成功！')
print(f'Ray版本: {ray.__version__}')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
"

echo "安装完成！"
echo ""
echo "使用方法："
echo "1. 启动Ray集群: ./start_ray_cluster.sh"
echo "2. 运行分布式训练: ./run_distributed_fedavg.sh"
echo ""
echo "如果遇到问题，请检查："
echo "- Python版本 >= 3.8"
echo "- CUDA版本（如果使用GPU）"
echo "- 网络连接（用于下载模型）" 