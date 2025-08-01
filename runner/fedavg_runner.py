import copy
import os
import sys
import time  # 新增
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import (
    get_peft_model, 
    get_peft_model_state_dict, 
    set_peft_model_state_dict, 
    prepare_model_for_kbit_training
)

from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algo.FT.fedavg.client import *
from algo.FT.fedavg.server import *
from config import get_config, save_config, get_model_config, get_training_args
from dataset.split_dataset import *
from utils import *
from utils.fed_utils import get_proxy_dict, get_auxiliary_dict

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
fed_args.fed_alg = "fedavg" # Force the fed_alg parameter to be 'fedavg'
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
def calculate_model_size_gb(model_dict):
    """计算模型参数的字节大小，返回GB单位"""
    total_bytes = 0
    for param_tensor in model_dict.values():
        if isinstance(param_tensor, torch.Tensor):
            # 计算参数数量 × 每个参数的字节数
            total_bytes += param_tensor.numel() * param_tensor.element_size()
    return total_bytes / (1024**3)  # 转换为GB

# 在初始化部分添加通信量统计变量
total_download_volume = 0.0  # 总下行通信量 (GB)
total_upload_volume = 0.0    # 总上行通信量 (GB)

# 计算单个模型的大小
model_size_gb = calculate_model_size_gb(global_dict)
print(f"Model parameter size: {model_size_gb:.4f} GB")

training_loss = [[] for i in range(fed_args.num_clients)]
total_training_time = 0.0
total_communication_time = 0.0
total_aggregation_time = 0.0

# Define the client training function
def train_client(client_id, global_dict, local_datasets, round, fed_args, script_args, 
                 tokenizer, formatting_prompts_func, data_collator, training_loss, 
                 local_dict_list, clients_this_round):
    """
    Single-client training function
    """
    if client_id not in clients_this_round:
        training_loss[client_id].append(-1)  # -1 indicates that the client did not participate in training
        return None, 0.0, 0.0  # 返回训练时间和通信时间
    
    training_time = 0.0
    communication_time = 0.0
    
    try:
        # Create model replicas (每个线程requires an independent model instance)
        device_map, quantization_config, torch_dtype = get_model_config(script_args)
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        
        if script_args.load_in_8bit or script_args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        # Sync global model to local
        comm_start = time.time()
        set_peft_model_state_dict(model, global_dict)
        comm_end = time.time()
        communication_time += comm_end - comm_start
        
        # Get current round's dataset
        sub_dataset = get_dataset_this_round(local_datasets[client_id], round, fed_args, script_args)
        
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6) # Cosine LR decay
        new_training_args = get_training_args(script_args, new_lr)
        
        # Creat trainer
        trainer = get_fedavg_local_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=new_training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            script_args=script_args,
        )
        
        # Training the model (训练时间)
        train_start = time.time()
        results = trainer.train()
        train_end = time.time()
        training_time = train_end - train_start
        
        training_loss[client_id].append(results.training_loss)
        
        # Get the local model parameters (通信时间)
        comm_start = time.time()
        local_dict_list[client_id] = copy.deepcopy(get_peft_model_state_dict(model))
        comm_end = time.time()
        communication_time += comm_end - comm_start
        
        print(f"Client {client_id} training completed")
        return None, training_time, communication_time
        
    except Exception as e:
        print(f"Error in client {client_id} training: {e}")
        training_loss[client_id].append(-1)
        return None, 0.0, 0.0


for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    
    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # 计算本轮下行通信量（服务端→客户端）
    round_download_volume = len(clients_this_round) * model_size_gb
    total_download_volume += round_download_volume
    
    # Use a thread pool to execute client training tasks
    max_workers = min(len(clients_this_round), 100)
    
    round_training_time = 0.0
    round_communication_time = 0.0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all client training tasks
        future_to_client = {
            executor.submit(
                train_client, 
                client, 
                copy.deepcopy(global_dict),
                local_datasets, 
                round, 
                fed_args, 
                script_args,
                tokenizer, 
                formatting_prompts_func, 
                data_collator, 
                training_loss, 
                local_dict_list, 
                clients_this_round
            ): client for client in clients_this_round
        }
        
        # Collect results from all clients
        for future in as_completed(future_to_client):
            client = future_to_client[future]
            try:
                _, train_time, comm_time = future.result()
                if train_time is not None:
                    round_training_time = max(round_training_time, train_time)
                    round_communication_time = max(round_communication_time, comm_time)
            except Exception as exc:
                print(f'Client {client} generated an exception: {exc}')
    
    total_training_time += round_training_time
    total_communication_time += round_communication_time
    

    # 计算本轮上行通信量（客户端→服务端）
    round_upload_volume = len(clients_this_round) * model_size_gb
    total_upload_volume += round_upload_volume
    
    # ===== Server aggregates the local models =====
    agg_start = time.time()
    global_dict, global_auxiliary = global_aggregate(
        global_dict, local_dict_list, sample_num_list, clients_this_round)
    set_peft_model_state_dict(model, global_dict)   # Update global model
    agg_end = time.time()
    round_aggregation_time = agg_end - agg_start
    total_aggregation_time += round_aggregation_time
    
    print(f"Round {round+1} - Training: {round_training_time:.2f}s, Communication: {round_communication_time:.2f}s, Aggregation: {round_aggregation_time:.2f}s")
    print(f"Round {round+1} - Download: {round_download_volume:.4f}GB, Upload: {round_upload_volume:.4f}GB")
    
    # ===== Save the global model =====
    if (round+1) % fed_args.save_model_freq == 0:
        model.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        tokenizer.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss, dtype=object))

# ===== Print timing statistics =====
# 在最终统计中添加通信量信息
print("\n" + "="*50)
print("FEDERATED LEARNING STATISTICS")
print("="*50)
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Total Communication Time: {total_communication_time:.2f} seconds")
print(f"Total Aggregation Time: {total_aggregation_time:.2f} seconds")
print(f"Total Time: {total_training_time + total_communication_time + total_aggregation_time:.2f} seconds")
print("-"*50)
print(f"Total Download Volume (Server→Clients): {total_download_volume:.4f} GB")
print(f"Total Upload Volume (Clients→Server): {total_upload_volume:.4f} GB")
print(f"Total Communication Volume: {total_download_volume + total_upload_volume:.4f} GB")
print(f"Average Volume per Round: {(total_download_volume + total_upload_volume)/fed_args.num_rounds:.4f} GB")
print("="*50)

print("Fedavg federated training finished!")