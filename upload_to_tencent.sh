#!/bin/bash

# 简化的腾讯云服务器文件上传脚本

TENCENT_SERVER="106.52.36.202"
TENCENT_USER="root"
TENCENT_PASSWORD="@Dsdq0722"

echo "📤 上传文件到腾讯云服务器..."
echo "服务器地址: $TENCENT_SERVER"

# 检查sshpass是否安装
if ! command -v sshpass &> /dev/null; then
    echo "❌ 需要安装sshpass"
    echo "请运行: yum install -y sshpass"
    exit 1
fi

# 上传核心文件
echo "📁 上传项目文件..."

# 上传目录
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r runner/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r algo/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r utils/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no -r dataset/ $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/

# 上传单个文件
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no config.py $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no requirements_ray.txt $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no run_tencent_cloud_training.sh $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no monitor_tencent_cloud.sh $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/
sshpass -p "$TENCENT_PASSWORD" scp -o StrictHostKeyChecking=no test_ray_installation.py $TENCENT_USER@$TENCENT_SERVER:/root/openfedllm/

# 设置执行权限
echo "🔧 设置文件权限..."
sshpass -p "$TENCENT_PASSWORD" ssh -o StrictHostKeyChecking=no $TENCENT_USER@$TENCENT_SERVER "cd /root/openfedllm && chmod +x *.sh"

echo "✅ 文件上传完成！"
echo ""
echo "🎯 下一步操作："
echo "1. 连接到服务器: ssh $TENCENT_USER@$TENCENT_SERVER"
echo "2. 进入项目目录: cd /root/openfedllm"
echo "3. 运行测试: python3 test_ray_installation.py"
echo "4. 启动训练: ./run_tencent_cloud_training.sh" 