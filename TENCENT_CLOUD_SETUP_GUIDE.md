# 腾讯云服务器联邦学习项目部署指南

## 🚀 项目概述

本项目是一个基于联邦学习的大语言模型训练系统，已在腾讯云服务器上成功部署。

## 📋 系统信息

- **操作系统**: AlmaLinux 8.10
- **Python版本**: 3.9.20
- **PyTorch版本**: 2.0.1+cu117
- **CPU核心数**: 4
- **内存**: 3.7GB
- **存储**: 40GB

## ✅ 已完成的配置

### 1. 环境设置
- ✅ Python 3.9.20 已安装
- ✅ PyTorch 2.0.1 已安装
- ✅ Transformers 库已安装
- ✅ PEFT 库已安装
- ✅ Ray 分布式框架已安装
- ✅ 其他依赖库已安装

### 2. 项目结构
- ✅ 项目文件完整
- ✅ 配置文件就绪
- ✅ 算法模块就绪
- ✅ 工具模块就绪

### 3. 网络配置
- ✅ 配置了清华PyPI镜像源
- ✅ 设置了Hugging Face镜像

## 🎯 运行步骤

### 步骤1: 启动Ray集群
```bash
# 启动优化的Ray集群（低内存配置）
./start_ray_cluster_optimized.sh
```

### 步骤2: 运行联邦学习训练
```bash
# 运行优化的训练脚本
./run_optimized_training.sh
```

### 步骤3: 监控训练过程
```bash
# 查看Ray集群状态
ray status

# 查看Ray Dashboard（如果网络可达）
# http://服务器IP:8265
```

## 📊 优化配置

### 内存优化
- Ray对象存储: 500MB
- Ray内存限制: 1GB
- 批次大小: 4
- 序列长度: 256
- 客户端数量: 2

### 模型配置
- 使用较小的模型: microsoft/DialoGPT-medium
- 启用8位量化: --load_in_8bit
- LoRA配置: r=4, alpha=8
- 数据集采样: 1000样本

## 🔧 故障排除

### 1. 内存不足
如果遇到内存不足问题：
```bash
# 减少批次大小
--batch_size 2

# 减少序列长度
--seq_length 128

# 减少客户端数量
--num_clients 1
```

### 2. 网络连接问题
如果无法下载模型：
```bash
# 设置代理（如果有）
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 或使用离线模式
# 预先下载模型到本地目录
```

### 3. Ray集群问题
```bash
# 重启Ray集群
ray stop
./start_ray_cluster_optimized.sh
```

## 📁 重要文件

- `start_ray_cluster_optimized.sh` - 优化的Ray集群启动脚本
- `run_optimized_training.sh` - 优化的训练脚本
- `test_minimal.py` - 系统测试脚本
- `config.py` - 项目配置文件
- `requirements_ray.txt` - Ray专用依赖文件

## 🎉 成功标志

当看到以下输出时，表示系统运行成功：
```
✅ Ray集群启动完成！
✅ 联邦学习训练完成！
📊 训练结果保存在: /root/openfedllm/output/optimized_training
```

## 📞 技术支持

如果遇到问题，请检查：
1. 系统内存使用情况
2. Ray集群状态
3. 网络连接状态
4. 日志文件输出

## 🔄 下一步

1. 配置网络代理以访问Hugging Face
2. 下载更大的预训练模型
3. 准备更多训练数据
4. 调整训练参数以获得更好的性能 