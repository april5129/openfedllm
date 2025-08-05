#!/bin/bash

# 腾讯云服务器部署脚本
# 用于在腾讯云服务器上设置分布式联邦学习环境

TENCENT_SERVER="106.52.36.202"
TENCENT_USER="root"
TENCENT_PASSWORD="@Dsdq0722"

echo "🚀 开始部署分布式联邦学习到腾讯云服务器..."
echo "服务器地址: $TENCENT_SERVER"
echo "用户名: $TENCENT_USER"

# 检查是否安装了sshpass
if ! command -v sshpass &> /dev/null; then
    echo "❌ 需要安装sshpass来支持密码认证"
    echo "请运行: sudo apt-get install sshpass (Ubuntu/Debian)"
    echo "或: sudo yum install sshpass (CentOS/RHEL)"
    exit 1
fi

# 创建远程部署脚本
cat > remote_setup.sh << 'EOF'
#!/bin/bash

echo "🔧 在腾讯云服务器上设置分布式联邦学习环境..."

# 更新系统
echo "更新系统包..."
yum update -y || apt-get update -y

# 安装基础工具
echo "安装基础工具..."
yum install -y git wget curl python3 python3-pip || apt-get install -y git wget curl python3 python3-pip

# 创建项目目录
echo "创建项目目录..."
mkdir -p /root/openfedllm
cd /root/openfedllm

# 克隆项目（如果是从Git仓库）
# git clone <your-repo-url> .

# 或者直接复制文件（通过scp）

# 安装Python依赖
echo "安装Python依赖..."
pip3 install --upgrade pip

# 安装Ray分布式框架
echo "安装Ray分布式框架..."
pip3 install "ray[default]>=2.9.0"

# 安装其他必要依赖
echo "安装其他依赖..."
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
echo "配置Ray集群..."
mkdir -p /tmp/ray/session_latest/logs

# 设置环境变量
echo "设置环境变量..."
echo 'export PYTHONPATH="/root/openfedllm:$PYTHONPATH"' >> ~/.bashrc
echo 'export RAY_DISABLE_IMPORT_WARNING=1' >> ~/.bashrc
source ~/.bashrc

echo "✅ 腾讯云服务器环境设置完成！"
EOF

# 上传部署脚本到服务器
echo "📤 上传部署脚本到服务器..."
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no remote_setup.sh $TENCENT_USER@$TENCENT_SERVER:/root/

# 在服务器上执行部署脚本
echo "🔧 在服务器上执行部署脚本..."
sshpass -p "$TENCENT_PASSWORD" ssh -o StrictHostKeyChecking=no $TENCENT_USER@$TENCENT_SERVER "chmod +x /root/remote_setup.sh && /root/remote_setup.sh"

# 上传项目文件
echo "📤 上传项目文件到服务器..."
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r runner/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r algo/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r utils/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r dataset/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no config.py $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no requirements_ray.txt $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no start_ray_cluster.sh $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no run_distributed_fedavg.sh $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no test_ray_installation.py $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/

# 给脚本添加执行权限
echo "🔧 设置脚本权限..."
sshpass -p "$TENCENT_PASSWORD" ssh -o StrictHostKeyChecking=no $TENCENT_USER@$TENCENT_SERVER "cd /root/openfedllm && chmod +x *.sh"

# 测试连接
echo "🧪 测试Ray安装..."
sshpass -p "$TENCENT_PASSWORD" ssh -o StrictHostKeyChecking=no $TENCENT_USER@$TENCENT_SERVER "cd /root/openfedllm && python3 test_ray_installation.py"

echo "✅ 部署完成！"
echo ""
echo "🎯 下一步操作："
echo "1. 连接到服务器: ssh $TENCENT_USER@$TENCENT_SERVER"
echo "2. 进入项目目录: cd /root/openfedllm"
echo "3. 启动Ray集群: ./start_ray_cluster.sh"
echo "4. 运行分布式训练: ./run_distributed_fedavg.sh"
echo ""
echo "📊 监控Ray集群: http://$TENCENT_SERVER:8265"

# 清理本地临时文件
rm -f remote_setup.sh 