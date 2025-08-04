import sys
import copy
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algo.DPO.base_client import *
from algo.FedFT.local.server import *
from config import get_config, save_config, get_model_config, get_training_args
from dataset.split_dataset import *
from utils import *
from utils.fed_utils import get_proxy_dict, get_auxiliary_dict

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
fed_args.fed_alg = "local" # Force the fed_alg parameter to be 'local'
fed_args.num_clients = 1 # Force the num_clients parameter to be '1'
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_dpo_dataset(script_args.dataset_name, dataset, script_args.template, script_args.dataset_sample)

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

if script_args.use_peft == True:
    model_ref = None
else:
    # construct a reference model with the identical original parameters
    # e.g. DPO need a reference model to compute the discrepancy loss
    model_ref = AutoModelForCausalLM.from_pretrained(
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

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
total_training_time = 0.0
total_communication_time = 0.0
total_aggregation_time = 0.0

clients_this_round = [0]
client_id = 0
com_time_start = time.time()
set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
com_time_end = time.time()
total_communication_time += com_time_end - com_time_start

for round in tqdm(range(fed_args.num_rounds)):
    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # Get current round's dataset
    sub_dataset = get_dataset_this_round(local_datasets[client_id], round, fed_args, script_args)      # get the required sub-dataset for this round
    new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)      # manually schedule the learning rate
    new_training_args = get_training_args(script_args, new_lr)

    # Creat trainer
    trainer = get_fed_local_dpo_trainer(
        script_args=script_args,
        model=model,
        model_ref=model_ref,
        tokenizer=tokenizer,
        training_args=new_training_args,
        local_dataset=sub_dataset,
    )
        
    # Training the model
    train_start = time.time()
    results = trainer.train()
    train_end = time.time()
    training_time = train_end - train_start

    training_loss[client_id].append(results.training_loss)
        
# ===== Client transmits local information to server =====
com_time_start = time.time()
local_dict_list[client_id] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
com_time_end = time.time()
total_communication_time += com_time_end - com_time_start

# ===== Server aggregate the local models =====
agg_time_start = time.time()
global_dict, global_auxiliary = global_aggregate(
    global_dict, local_dict_list, sample_num_list, clients_this_round
)
set_peft_model_state_dict(model, global_dict)   # Update global model
agg_time_end = time.time()
total_aggregation_time += agg_time_end - agg_time_start
    
# ===== Save the model =====
if (round+1) % fed_args.save_model_freq == 0:
    trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss[client_id]))

def calculate_model_size_gb(model_dict):
    """Calculate model parameter byte size, return in GB units"""
    total_bytes = 0
    for param_tensor in model_dict.values():
        if isinstance(param_tensor, torch.Tensor):
            # Calculate parameter bytes: number of parameters × bytes per element
            total_bytes += param_tensor.numel() * param_tensor.element_size()
    
    return total_bytes / (1024**3)  # Convert to GB

communication_volume = calculate_model_size_gb(global_dict)  # Single communication volume
communication_volume *= 2  # Upload and download
communication_volume *= len(clients_this_round) # Distribute to multiple clients
# communication_volume *= fed_args.num_rounds # Total rounds
print("\n" + "="*50)
print("FEDERATED LEARNING STATISTICS")
print("="*50)
print(f"Total Training Time: {total_training_time:.8f} seconds")
print(f"Total Communication Time: {total_communication_time:.8f} seconds")
print(f"Total Aggregation Time: {total_aggregation_time:.8f} seconds")
print(f"Total Time: {total_training_time + total_communication_time + total_aggregation_time:.8f} seconds")
print("-"*50)
print(f"Total Communication Volume: {communication_volume:.8f} GB")
print("-"*50)
print("Local federated training finished!")
