import ray
import time
import copy
import torch
import numpy as np
import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    round_num: int
    training_time: float
    communication_time: float
    aggregation_time: float
    client_losses: List[float]
    global_loss: float = 0.0

@ray.remote
class DistributedCoordinator:
    """分布式协调器，管理联邦学习的整体流程"""
    
    def __init__(self, fed_args, script_args):
        self.fed_args = fed_args
        self.script_args = script_args
        self.metrics_history = []
        self.global_model_state = None
        self.clients_ready = set()
        
    def register_client(self, client_id: int):
        """注册客户端"""
        self.clients_ready.add(client_id)
        logger.info(f"客户端 {client_id} 已注册")
        
    def set_global_model(self, global_model_state: Dict):
        """设置全局模型状态"""
        self.global_model_state = copy.deepcopy(global_model_state)
        
    def get_global_model(self):
        """获取全局模型状态"""
        return copy.deepcopy(self.global_model_state)
        
    def record_metrics(self, metrics: TrainingMetrics):
        """记录训练指标"""
        self.metrics_history.append(metrics)
        
    def get_metrics_history(self):
        """获取指标历史"""
        return self.metrics_history
        
    def get_ready_clients(self):
        """获取已就绪的客户端列表"""
        return list(self.clients_ready)

@ray.remote
class ModelAggregator:
    """模型聚合器，负责聚合客户端模型"""
    
    def __init__(self, fed_args):
        self.fed_args = fed_args
        
    def aggregate_models(self, global_dict: Dict, local_dicts: List[Dict], 
                        sample_nums: List[int], clients_this_round: List[int]) -> Dict:
        """聚合客户端模型"""
        from algo.FedFT.fedavg.server import global_aggregate
        
        # 创建本地字典列表
        local_dict_list = [None] * self.fed_args.num_clients
        for i, client_id in enumerate(clients_this_round):
            if local_dicts[i] is not None:
                local_dict_list[client_id] = local_dicts[i]
        
        # 执行聚合
        aggregated_dict, _ = global_aggregate(
            global_dict, local_dict_list, sample_nums, clients_this_round
        )
        
        return aggregated_dict

@ray.remote
class ResourceMonitor:
    """资源监控器，监控集群资源使用情况"""
    
    def __init__(self):
        self.resource_usage = {}
        
    def update_usage(self, node_id: str, cpu_usage: float, memory_usage: float, 
                    gpu_usage: float = 0.0):
        """更新资源使用情况"""
        self.resource_usage[node_id] = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'timestamp': time.time()
        }
        
    def get_cluster_status(self):
        """获取集群状态"""
        return self.resource_usage
        
    def get_optimal_client_allocation(self, num_clients: int):
        """获取最优的客户端分配策略"""
        # 基于资源使用情况，返回最优的客户端分配
        available_nodes = []
        for node_id, usage in self.resource_usage.items():
            if usage['cpu_usage'] < 0.8 and usage['memory_usage'] < 0.8:
                available_nodes.append(node_id)
        
        # 简单的轮询分配
        return available_nodes[:num_clients]

class DistributedFedTrainer:
    """分布式联邦学习训练器"""
    
    def __init__(self, fed_args, script_args, peft_config, tokenizer, 
                 formatting_prompts_func, data_collator, local_datasets):
        self.fed_args = fed_args
        self.script_args = script_args
        self.peft_config = peft_config
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.local_datasets = local_datasets
        
        # 初始化Ray组件
        self.coordinator = DistributedCoordinator.remote(fed_args, script_args)
        self.aggregator = ModelAggregator.remote(fed_args)
        self.monitor = ResourceMonitor.remote()
        
        # 客户端列表
        self.clients = []
        
    def initialize_clients(self):
        """初始化客户端"""
        logger.info("初始化分布式客户端...")
        
        # 从fedavg_runner.py导入FedClient
        from runner.FedFT.fedavg_runner import FedClient
        
        for client_id in range(self.fed_args.num_clients):
            client = FedClient.remote(
                client_id, self.script_args, self.fed_args, self.peft_config,
                self.tokenizer, self.formatting_prompts_func, self.data_collator,
                self.local_datasets[client_id]
            )
            self.clients.append(client)
            
            # 注册客户端
            ray.get(self.coordinator.register_client.remote(client_id))
            
        logger.info(f"已初始化 {len(self.clients)} 个客户端")
        
    def train_round(self, round_num: int, global_dict: Dict) -> TrainingMetrics:
        """执行一轮训练"""
        from algo.FedFT.fedavg.server import get_clients_this_round
        
        # 获取本轮参与的客户端
        clients_this_round = get_clients_this_round(self.fed_args, round_num)
        logger.info(f"第 {round_num + 1} 轮训练，参与客户端: {clients_this_round}")
        
        # 设置全局模型
        ray.get(self.coordinator.set_global_model.remote(global_dict))
        
        # 提交训练任务
        training_futures = []
        for client_id in clients_this_round:
            future = self.clients[client_id].train.remote(
                copy.deepcopy(global_dict), round_num, clients_this_round
            )
            training_futures.append((client_id, future))
        
        # 收集训练结果
        local_dicts = []
        client_losses = [-1] * self.fed_args.num_clients
        max_training_time = 0.0
        max_communication_time = 0.0
        
        for client_id, future in training_futures:
            try:
                local_dict, train_time, comm_time = ray.get(future)
                if local_dict is not None:
                    local_dicts.append(local_dict)
                    max_training_time = max(max_training_time, train_time)
                    max_communication_time = max(max_communication_time, comm_time)
                    
                    # 获取客户端损失
                    client_loss = ray.get(self.clients[client_id].get_training_loss.remote())
                    if client_loss and len(client_loss) > round_num:
                        client_losses[client_id] = client_loss[round_num]
                        
            except Exception as e:
                logger.error(f"客户端 {client_id} 训练失败: {e}")
        
        # 聚合模型
        agg_start = time.time()
        sample_nums = [len(self.local_datasets[i]) for i in range(self.fed_args.num_clients)]
        aggregated_dict = ray.get(self.aggregator.aggregate_models.remote(
            global_dict, local_dicts, sample_nums, clients_this_round
        ))
        agg_end = time.time()
        aggregation_time = agg_end - agg_start
        
        # 创建训练指标
        metrics = TrainingMetrics(
            round_num=round_num,
            training_time=max_training_time,
            communication_time=max_communication_time,
            aggregation_time=aggregation_time,
            client_losses=client_losses
        )
        
        # 记录指标
        ray.get(self.coordinator.record_metrics.remote(metrics))
        
        logger.info(f"第 {round_num + 1} 轮完成 - "
                   f"训练: {max_training_time:.2f}s, "
                   f"通信: {max_communication_time:.2f}s, "
                   f"聚合: {aggregation_time:.2f}s")
        
        return aggregated_dict, metrics
        
    def train(self, global_dict: Dict, num_rounds: int) -> Dict:
        """执行完整的联邦学习训练"""
        logger.info(f"开始分布式联邦学习训练，共 {num_rounds} 轮")
        
        # 初始化客户端
        self.initialize_clients()
        
        # 执行训练轮次
        for round_num in range(num_rounds):
            global_dict, metrics = self.train_round(round_num, global_dict)
            
            # 保存模型检查点
            if (round_num + 1) % self.fed_args.save_model_freq == 0:
                self.save_checkpoint(global_dict, round_num + 1)
        
        logger.info("分布式联邦学习训练完成")
        return global_dict
        
    def save_checkpoint(self, global_dict: Dict, round_num: int):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.script_args.output_dir, f"checkpoint-{round_num}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 这里需要将global_dict转换回模型并保存
        # 具体实现取决于模型类型
        logger.info(f"保存检查点到 {checkpoint_dir}")
        
    def get_training_summary(self):
        """获取训练摘要"""
        metrics_history = ray.get(self.coordinator.get_metrics_history.remote())
        
        total_training_time = sum(m.training_time for m in metrics_history)
        total_communication_time = sum(m.communication_time for m in metrics_history)
        total_aggregation_time = sum(m.aggregation_time for m in metrics_history)
        
        return {
            'total_rounds': len(metrics_history),
            'total_training_time': total_training_time,
            'total_communication_time': total_communication_time,
            'total_aggregation_time': total_aggregation_time,
            'total_time': total_training_time + total_communication_time + total_aggregation_time,
            'metrics_history': metrics_history
        } 