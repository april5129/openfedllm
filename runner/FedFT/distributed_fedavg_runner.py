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
from runner.FedFT.distributed_manager import DistributedFedTrainer

def main():
    # ===== Define the arguments =====
    script_args, fed_args, peft_config = get_config()
    fed_args.fed_alg = "fedavg" # Force the fed_alg parameter to be 'fedavg'
    training_args = get_training_args(script_args, script_args.learning_rate)
    save_config(script_args, fed_args)
    print(script_args, fed_args)

    # ===== Initialize Ray =====
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,  # 根据实际CPU核心数调整
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            object_store_memory=1000000000,  # 1GB
            _memory=2000000000  # 2GB
        )
    print("Ray集群已初始化")

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

    # ===== Create distributed trainer =====
    print("创建分布式联邦学习训练器...")
    distributed_trainer = DistributedFedTrainer(
        fed_args=fed_args,
        script_args=script_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_prompts_func=formatting_prompts_func,
        data_collator=data_collator,
        local_datasets=local_datasets
    )

    # ===== Start distributed federated training =====
    print("开始分布式联邦学习训练...")
    start_time = time.time()
    
    # 执行分布式训练
    final_global_dict = distributed_trainer.train(global_dict, fed_args.num_rounds)
    
    end_time = time.time()
    total_time = end_time - start_time

    # ===== Update global model with final parameters =====
    set_peft_model_state_dict(model, final_global_dict)

    # ===== Save the final model =====
    model.save_pretrained(os.path.join(script_args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "final_model"))

    # ===== Get training summary =====
    training_summary = distributed_trainer.get_training_summary()
    
    # ===== Calculate communication volume =====
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
    communication_volume *= fed_args.sample_clients  # Distribute to multiple clients
    communication_volume *= fed_args.num_rounds  # Total rounds

    # ===== Print timing statistics =====
    print("\n" + "="*50)
    print("分布式联邦学习统计")
    print("="*50)
    print(f"总训练时间: {training_summary['total_training_time']:.8f} 秒")
    print(f"总通信时间: {training_summary['total_communication_time']:.8f} 秒")
    print(f"总聚合时间: {training_summary['total_aggregation_time']:.8f} 秒")
    print(f"总时间: {training_summary['total_time']:.8f} 秒")
    print(f"实际总时间: {total_time:.8f} 秒")
    print("-"*50)
    print(f"总通信量: {communication_volume:.8f} GB")
    print(f"训练轮次: {training_summary['total_rounds']}")
    print("="*50)
    print("分布式联邦学习训练完成！")

    # ===== Cleanup Ray =====
    ray.shutdown()

if __name__ == "__main__":
    main() 