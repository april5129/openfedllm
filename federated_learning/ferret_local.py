from .ferret_optimizer import *
from tqdm import tqdm
from copy import deepcopy

# 联邦学习客户端实现
class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader):
        # 初始化，保存客户端编号、参数、候选种子、本地数据等
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None

        self.device = torch.device(f'cuda:{args.device}')
        self.candidate_seeds = candidate_seeds

    def local_train_with_seed_pool(self, pulled_model, cur_round):
        # 本地训练主流程
        self.model = pulled_model
        # 记录旧参数用于后续投影
        old_params = [(name, deepcopy(param.data)) for name, param in self.model.named_parameters() if param.requires_grad]
        
        self.model.to(self.device)
        
        # 初始化本地种子池
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = self.args.lr
        
        # 计算本地训练步数
        if self.args.batch_or_epoch == 'epoch':
            iter_steps = self.args.local_step * len(self.train_loader)
            print("iter_steps:", iter_steps)
        else:
            iter_steps = self.args.local_step
           
        # 初始化Ferret算法框架
        framework = FerretFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)
        self.model.train()
        self.model.zero_grad()
        
        # 训练进度条
        if self.args.batch_or_epoch == 'batch':
                loss_total_train = 0.0
                num_trained = 0
                progress_bar = tqdm(range(iter_steps))
                
        for cur_step in range(iter_steps):
            # epoch模式下每轮重置进度条
            if self.args.batch_or_epoch == 'epoch':
                if cur_step % len(self.train_loader) == 0:
                    loss_total_train = 0.0
                    num_trained = 0
                    progress_bar = tqdm(range(len(self.train_loader)))
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)
            # 将数据转移到GPU
            batch = {
                'input_ids': batch['input_ids'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device) 
            }
            
            # 梯度累积步数控制
            if cur_step % self.args.n_accum == self.args.n_accum - 1:
                apply_optim_step = True
            else:
                apply_optim_step = False
            
            logits, loss = framework.step(batch, apply_optim_step=apply_optim_step)
            
            progress_bar.update(1)
            if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                loss_total_train += loss
                num_trained += len(batch['input_ids'])
            if self.args.batch_or_epoch == 'epoch':
                progress_bar.set_description(f'client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
            else:
                progress_bar.set_description(f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')

        # 本地参数更新投影到低维空间
        self.local_seed_pool = framework.project_update(dict(old_params))
            
        # 释放内存
        del old_params, framework
        self.model = None

    def clear_model(self):
        # 清理模型，节省内存
        self.model = None

    def migrate(self, device):
        """
        将客户端迁移到新设备
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        从服务器拉取全局模型
        """
        self.model = forked_global_model