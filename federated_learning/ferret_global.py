from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from evaluation.rouge_evaluations import *
from federated_learning.utils_data.default_tokens import DefaultToken
from copy import deepcopy
import os
import math
from .ferret_optimizer import *

# softmax函数实现
def softmax(vec):
    vec = vec - np.max(vec)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# 最小-最大归一化
def min_max_norm(vec):
    min_val = np.min(vec)
    return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)

# 联邦学习服务器端实现
class Server(object):
    # 修改设备初始化逻辑
    def __init__(self, args, eval_loader, candidate_seeds, log_dir):
        # 初始化，保存参数、评估数据、候选种子、日志目录等
        self.args = args
        self.eval_loader = eval_loader
        self.candidate_seeds = candidate_seeds
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        self.log_dir = log_dir
        self.tokenizer.model_max_length = self.args.max_length
        special_tokens = dict()
        # 检查并补充特殊token
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)
        
        # 根据设备类型加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map='cpu' if args.device < 0 else 'cuda',
            torch_dtype=torch.float32 if args.device < 0 else torch.float16,
            trust_remote_code=True
        )
        
        # 初始化全局种子池
        self.seed_pool = {seed: 0.0 for seed in self.candidate_seeds}
        
        self.device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    def aggregate_seed_pool(self, selected_client_list, cur_round=1):
        # 聚合所有客户端上传的低维梯度/参数更新
        for seed in self.candidate_seeds:
            self.seed_pool[seed] *= self.args.momentum  # 动量机制
        
        # 计算聚合权重
        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))
        # 累加每个客户端的本地更新
        for client_idx in range(len(selected_client_list)):
            local_seed_pool = selected_client_list[client_idx].local_seed_pool
            for seed, grad in local_seed_pool.items():
                self.seed_pool[seed] += grad * weight_array[client_idx]
        
        # 清理客户端模型，节省内存
        for client in selected_client_list:
            client.clear_model()

    def update_global_model_by_seed_pool(self):
        # 用聚合后的低维梯度/参数重构全局模型
        self.model.to(self.device)
        framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
        progress_bar = tqdm(range(len(self.seed_pool))) 
        # 遍历所有种子，逐步更新全局模型
        for seed, grad in self.seed_pool.items():
            framework.update(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description(f'server update global model')
        self.model.to("cpu")

    def eval(self, cur_round, eval_avg_acc):
        # 评估当前全局模型
        if self.args.eval_metric == 'loss':
            eval_metric = self.eval_loss(cur_round)
        else:
            eval_metric =  self.eval_generate(cur_round)
        
        # 保存模型
        if self.args.save and cur_round > 0:
            save_dir = self.log_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存最佳模型
            if (self.args.eval_metric == 'loss' and eval_metric < np.min(eval_avg_acc)) or (self.args.eval_metric != 'none' and eval_metric > np.max(eval_avg_acc)):
                for file_name in os.listdir(save_dir):
                    if 'best' in file_name:
                        os.remove(os.path.join(save_dir, file_name))  
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_best_round{cur_round}.bin'))
            # 保存最终模型
            for file_name in os.listdir(save_dir):
                if 'final' in file_name:
                    os.remove(os.path.join(save_dir, file_name)) 
            torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_final_round{cur_round}.bin'))
        return eval_metric

    def eval_loss(self, cur_round):
        # 用loss评估模型
        self.model = self.model.to(self.device)
        self.model.eval()
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0
        
        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device) 
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                progress_bar_eval.update(1)
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, loss: {loss_total_eval / num_eval}')
        print()
        print()
        self.model = self.model.cpu()
        return (loss_total_eval / num_eval).item()

    def eval_generate(self, cur_round):
        # 用生成任务评估模型
        self.model = self.model.to(self.device)
        self.model.eval()
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        acc_total_eval = 0.0
        num_eval = 0
        
        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                label_ids = batch['labels'].to(self.device)
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    num_beams=1,
                )
                # 计算rouge分数
                acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, metric: {acc_total_eval / num_eval}')
        print()
        print()
        self.model = self.model.cpu()
        return acc_total_eval / num_eval