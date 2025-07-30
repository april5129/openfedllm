import copy
import os
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from flask import Flask, request, jsonify
import threading
import torch

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

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

# 保存配置
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Define the global and local models =====
# 只有在使用PEFT时才使用get_peft_model_state_dict，否则使用model.state_dict
if script_args.use_peft and peft_config is not None:
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
else:
    global_dict = copy.deepcopy(model.state_dict())

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== HTTP Server for communication =====
app = Flask(__name__)

# 全局状态变量
clients_data = {}
clients_ready = 0
round_counter = 0
num_clients = fed_args.num_clients
sample_clients = fed_args.sample_clients

# 将张量转换为可序列化的格式
def serialize_state_dict(state_dict):
    serialized_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            serialized_dict[key] = value.cpu().numpy().tolist()
        else:
            serialized_dict[key] = value
    return serialized_dict

# 将序列化的数据转换回张量
def deserialize_state_dict(serialized_dict):
    state_dict = {}
    for key, value in serialized_dict.items():
        if isinstance(value, list):
            state_dict[key] = torch.tensor(value)
        else:
            state_dict[key] = value
    return state_dict

@app.route('/send_model', methods=['POST'])
def receive_model_from_client():
    global clients_data, clients_ready
    
    # 获取客户端ID和模型数据
    client_id = request.json.get('client_id')
    local_dict = request.json.get('local_dict')
    
    # 将序列化的数据转换回张量
    clients_data[client_id] = deserialize_state_dict(local_dict)
    clients_ready += 1
    
    print(f"Received model from client {client_id}")
    
    return jsonify({"status": "success"})

@app.route('/get_global_model', methods=['GET'])
def send_global_model_to_client():
    # 将global_dict中的张量序列化为列表
    serialized_global_dict = serialize_state_dict(global_dict)
    # 发送全局模型给客户端
    return jsonify({
        "global_dict": serialized_global_dict,
        "round": round_counter
    })

# ===== Start federated training =====
total_aggregation_time = 0.0

# 启动Flask服务线程
def start_flask_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

flask_thread = threading.Thread(target=start_flask_server)
flask_thread.daemon = True
flask_thread.start()

print("HTTP server started on port 5000")

for round in range(fed_args.num_rounds):
    # 选择参与本轮训练的客户端
    clients_this_round = get_clients_this_round(fed_args, round)
    round_counter = round

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # 等待客户端上传模型参数
    # 等待足够数量的客户端上传模型
    while clients_ready < len(clients_this_round):
        time.sleep(1)  # 等待1秒后再次检查
    
    # 转换clients_data为local_dict_list格式
    local_dict_list = [clients_data.get(i, {}) for i in range(num_clients)]
    
    # ===== Server aggregates the local models =====
    agg_time_start = time.time()
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    # 只有在使用PEFT时才使用set_peft_model_state_dict
    if script_args.use_peft and peft_config is not None:
        set_peft_model_state_dict(model, global_dict)   # Update global model
    else:
        model.load_state_dict(global_dict)   # Update global model
    agg_time_end = time.time()
    total_aggregation_time += agg_time_end - agg_time_start

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        # 保存模型逻辑
        pass
    
    # 重置状态以准备下一轮
    clients_ready = 0
    clients_data = {}

print("Federated training finished!")
print(f"total_aggregation_time: {total_aggregation_time:.2f}s")