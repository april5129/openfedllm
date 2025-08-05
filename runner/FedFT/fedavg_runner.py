import copy
import os
import sys
import time
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

import ray
from ray import serve
from ray.util.queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algo.FedFT.base_client import *
from algo.FedFT.fedavg.server import *
from config import get_config, save_config, get_model_config, get_training_args
from dataset.split_dataset import *
from dataset.process_dataset import *
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

# ===== Initialize Ray =====
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# ===== Define Ray Client Actor =====
@ray.remote(num_cpus=1, num_gpus=0.5 if torch.cuda.is_available() else 0)
class FedClient:
    def __init__(self, client_id, script_args, fed_args, peft_config, tokenizer, 
                 formatting_prompts_func, data_collator, local_dataset):
        self.client_id = client_id
        self.script_args = script_args
        self.fed_args = fed_args
        self.peft_config = peft_config
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.local_dataset = local_dataset
        self.model = None
        self.training_loss = []
        
    def initialize_model(self):
        """Initialize the model for this client"""
        from config import get_model_config, get_training_args
        
        device_map, quantization_config, torch_dtype = get_model_config(self.script_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.script_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=self.script_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        
        if self.script_args.load_in_8bit or self.script_args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.config.use_cache = False
        
        if True:  # gradient_checkpointing
            self.model.enable_input_require_grads()
    
    def train(self, global_dict, round_num, clients_this_round):
        """Train the client model"""
        if self.client_id not in clients_this_round:
            self.training_loss.append(-1)
            return None, 0.0, 0.0
        
        if self.model is None:
            self.initialize_model()
        
        training_time = 0.0
        communication_time = 0.0
        
        try:
            # Sync global model to local
            comm_start = time.time()
            set_peft_model_state_dict(self.model, global_dict)
            comm_end = time.time()
            communication_time += comm_end - comm_start
            
            # Get current round's dataset
            from dataset.split_dataset import get_dataset_this_round
            from utils.fed_utils import cosine_learning_rate
            from config import get_training_args
            from algo.FedFT.base_client import get_base_local_trainer
            
            sub_dataset = get_dataset_this_round(self.local_dataset, round_num, self.script_args)
            
            new_lr = cosine_learning_rate(round_num, self.fed_args.num_rounds, 
                                        self.script_args.learning_rate, 1e-6)
            new_training_args = get_training_args(self.script_args, new_lr)
            
            # Create trainer
            trainer = get_base_local_trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                training_args=new_training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=self.formatting_prompts_func,
                data_collator=self.data_collator,
                script_args=self.script_args,
            )
            
            # Training the model
            train_start = time.time()
            results = trainer.train()
            train_end = time.time()
            training_time = train_end - train_start
            
            self.training_loss.append(results.training_loss)
            
            # Get the local model parameters
            comm_start = time.time()
            local_dict = copy.deepcopy(get_peft_model_state_dict(self.model))
            comm_end = time.time()
            communication_time += comm_end - comm_start
            
            return local_dict, training_time, communication_time
            
        except Exception as e:
            print(f"Error in client {self.client_id} training: {e}")
            self.training_loss.append(-1)
            return None, 0.0, 0.0
    
    def get_training_loss(self):
        """Get training loss history"""
        return self.training_loss

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
total_training_time = 0.0
total_communication_time = 0.0
total_aggregation_time = 0.0




# ===== Create Ray clients =====
print("Creating Ray client actors...")
ray_clients = []
for client_id in range(fed_args.num_clients):
    client = FedClient.remote(
        client_id, script_args, fed_args, peft_config, tokenizer,
        formatting_prompts_func, data_collator, local_datasets[client_id]
    )
    ray_clients.append(client)

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    
    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    round_training_time = 0.0
    round_communication_time = 0.0
    
    # Submit training tasks to Ray clients
    training_futures = []
    for client_id in clients_this_round:
        future = ray_clients[client_id].train.remote(
            copy.deepcopy(global_dict), round, clients_this_round
        )
        training_futures.append((client_id, future))
    
    # Collect results from all clients
    local_dict_list = [None] * fed_args.num_clients
    for client_id, future in training_futures:
        try:
            local_dict, train_time, comm_time = ray.get(future)
            if local_dict is not None:
                local_dict_list[client_id] = local_dict
                round_training_time = max(round_training_time, train_time)
                round_communication_time = max(round_communication_time, comm_time)
        except Exception as exc:
            print(f'Client {client_id} generated an exception: {exc}')
    
    total_training_time += round_training_time
    total_communication_time += round_communication_time

    
    # ===== Server aggregates the local models =====
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
    
    # Collect training loss from all clients
    for client_id in range(fed_args.num_clients):
        training_loss[client_id] = ray.get(ray_clients[client_id].get_training_loss.remote())
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss, dtype=object))

    
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

# ===== Print timing statistics =====
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
print("Fedavg federated training finished!")

# ===== Cleanup Ray =====
ray.shutdown()