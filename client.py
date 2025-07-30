import copy
import time
import numpy as np
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

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

# 定义训练参数
training_args = get_training_args(script_args, script_args.learning_rate)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=script_args.gradient_checkpointing
            )

# 只有在使用PEFT时才应用PEFT配置
if script_args.use_peft and peft_config is not None:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if script_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Define the global and local models =====
# 只有在使用PEFT时才使用get_peft_model_state_dict，否则使用model.state_dict
if script_args.use_peft and peft_config is not None:
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
else:
    global_dict = copy.deepcopy(model.state_dict())

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Client training loop =====
training_loss = [[] for i in range(fed_args.num_clients)]
total_training_time = 0.0
total_communication_time = 0.0

# 使用配置文件中的client_id
client_id = fed_args.client_id

# 服务端地址
SERVER_URL = "http://localhost:5000"

def get_global_model_from_server():
    """从服务端获取全局模型参数"""
    try:
        response = requests.get(f"{SERVER_URL}/get_global_model")
        if response.status_code == 200:
            return response.json()["global_dict"]
        else:
            print(f"Failed to get global model: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting global model: {e}")
        return None

def send_model_to_server(local_dict, client_id):
    """发送本地模型参数到服务端"""
    try:
        data = {
            "client_id": client_id,
            "local_dict": local_dict
        }
        response = requests.post(f"{SERVER_URL}/send_model", json=data)
        if response.status_code == 200:
            print("Successfully sent model to server")
            return True
        else:
            print(f"Failed to send model to server: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending model to server: {e}")
        return False

for round in range(fed_args.num_rounds):
    # 从服务端接收全局模型参数
    global_dict = get_global_model_from_server()
    if global_dict is None:
        print("Failed to get global model, skipping round")
        continue
    
    # 获取本轮需要训练的客户端
    clients_this_round = get_clients_this_round(fed_args, round)
    
    # 如果当前客户端需要参与训练
    if client_id in clients_this_round:
        com_time_start = time.time()
        # 同步全局模型到本地
        # 只有在使用PEFT时才使用set_peft_model_state_dict
        if script_args.use_peft and peft_config is not None:
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        else:
            model.load_state_dict(global_dict)   # sync the global model to the local model
        com_time_end = time.time()
        total_communication_time += com_time_end - com_time_start
        
        sub_dataset = get_dataset_this_round(local_datasets[client_id], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client_id],
            global_auxiliary=global_auxiliary,
        )

        train_time_start = time.time()
        results = trainer.train()
        train_time_end = time.time()
        total_training_time += train_time_end - train_time_start

        training_loss[client_id].append(results.training_loss)

        # ===== Client transmits local information to server =====
        com_time_start = time.time()
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client_id], auxiliary_delta_dict[client_id] = trainer.get_auxiliary_param()

        # 只有在使用PEFT时才使用get_peft_model_state_dict
        if script_args.use_peft and peft_config is not None:
            local_dict = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
        else:
            local_dict = copy.deepcopy(model.state_dict())   # copy is needed!
        com_time_end = time.time()
        total_communication_time += com_time_end - com_time_start
        
        # 将本地模型参数发送给服务端
        success = send_model_to_server(local_dict, client_id)
        if not success:
            print("Failed to send model to server")
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

print("Client training finished!")
print(f"total_training_time: {total_training_time:.2f}s")
print(f"total_communication_time: {total_communication_time:.2f}s")