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

from algo.DPO.scaffold.client import *
from algo.FedFT.scaffold.server import *
from config import get_config, save_config, get_model_config, get_training_args
from dataset.split_dataset import *
from dataset.process_dataset import *
from utils import *
from utils.fed_utils import get_proxy_dict, get_auxiliary_dict

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
fed_args.fed_alg = "scaffold" # Force the fed_alg parameter to be 'scaffold'
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

# Define the client training function
def train_client(client_id, global_dict, local_datasets, round, fed_args, script_args, 
                 tokenizer, training_loss, local_dict_list, clients_this_round):
    """
    Single-client training function
    """
    if client_id not in clients_this_round:
        training_loss[client_id].append(-1)  # -1 indicates that the client did not participate in training
        return None, 0.0, 0.0  # Return training time and communication time

    training_time = 0.0
    communication_time = 0.0
    
    try:
        # Create model replicas (each thread requires an independent model instance)
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
        sub_dataset = get_dataset_this_round(local_datasets[client_id], round, script_args)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)      # manually schedule the learning rate
        new_training_args = get_training_args(script_args, new_lr)
    
        # Creat trainer
        trainer = get_fed_local_dpo_trainer(
            script_args=script_args,
            fed_args=fed_args,
            model=model,
            model_ref=model_ref,
            tokenizer=tokenizer,
            training_args=new_training_args,
            local_dataset=sub_dataset,
            global_dict=global_dict,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )
        
        # Training the model
        train_start = time.time()
        results = trainer.train()
        train_end = time.time()
        training_time = train_end - train_start

        training_loss[client_id].append(results.training_loss)
        
        auxiliary_model_list[client_id], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        # Get the local model parameters
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
    # Select the clients participating in this round of training
    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # Use a thread pool to execute client training tasks
    max_workers = min(len(clients_this_round), 100)  # Limit the number of concurrent threads to avoid  out-of-memory errors

    round_training_time = 0.0
    round_communication_time = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all client training tasks
        future_to_client = {
            executor.submit(
                train_client, 
                client, 
                copy.deepcopy(global_dict),  # Each thread gets its own independent replica of the global model
                local_datasets, 
                round, 
                fed_args, 
                script_args,
                tokenizer, 
                training_loss, 
                local_dict_list, 
                clients_this_round
            ): client for client in range(fed_args.num_clients)
        }
        
        # Wait for all tasks to complete
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

    # ===== Server aggregate the local models =====
    agg_start = time.time()
    global_dict, global_auxiliary = global_aggregate(
        global_dict, local_dict_list, sample_num_list, clients_this_round)
    set_peft_model_state_dict(model, global_dict)   # Update global model
    agg_end = time.time()
    round_aggregation_time = agg_end - agg_start
    total_aggregation_time += round_aggregation_time
    
    print(f"Round {round+1} - Training: {round_training_time:.2f}s, Communication: {round_communication_time:.2f}s, Aggregation: {round_aggregation_time:.2f}s")

    # ===== Save the global model =====
    if (round+1) % fed_args.save_model_freq == 0:
        model.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        tokenizer.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

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
communication_volume *= fed_args.num_rounds # Total rounds

print("\n" + "="*50)
print("FEDERATED LEARNING STATISTICS")
print("="*50)
print(f"Total Training Time: {total_training_time:.8f} seconds")
print(f"Total Communication Time: {total_communication_time:.8f} seconds")
print(f"Total Aggregation Time: {total_aggregation_time:.8f} seconds")
print(f"Total Time: {total_training_time + total_communication_time + total_aggregation_time:.8f} seconds")
print("-"*50)
print(f"Total Communication Volume: {communication_volume:.8f} GB")
print("="*50)
print("Scaffold federated training finished!")
