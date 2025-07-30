from flask import Flask, request, jsonify
import threading
import time
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用CORS以支持跨域请求

# 全局状态变量
clients_data = {}
global_model = None
round_counter = 0
num_clients = 3
clients_ready = 0

@app.route('/send_model', methods=['POST'])
def receive_model_from_client():
    global clients_data, clients_ready
    
    # 获取客户端ID和模型数据
    client_id = request.json.get('client_id')
    model_data = request.json.get('local_dict')  # 使用local_dict作为键名以匹配客户端
    
    # 保存客户端上传的模型数据
    clients_data[client_id] = model_data
    clients_ready += 1
    
    print(f"Received model from client {client_id}")
    
    return jsonify({"status": "success"})

@app.route('/get_global_model', methods=['GET'])
def send_global_model_to_client():
    # 发送全局模型给客户端
    return jsonify({
        "global_dict": global_model,  # 使用global_dict作为键名以匹配客户端
        "round": round_counter
    })

@app.route('/start_round', methods=['POST'])
def start_new_round():
    global round_counter, clients_ready, clients_data
    
    # 检查是否所有客户端都已上传模型
    if clients_ready >= num_clients:
        # 执行模型聚合
        aggregate_models()
        
        # 重置状态
        clients_ready = 0
        clients_data = {}
        round_counter += 1
        
        return jsonify({"status": "round completed", "round": round_counter})
    else:
        return jsonify({"status": "waiting for more clients"})

def aggregate_models():
    global global_model
    # 这里实现模型聚合逻辑
    # 实现联邦平均算法（FedAvg）
    if clients_data:
        # 获取客户端数量
        num_clients = len(clients_data)
        
        # 初始化全局模型
        global_model = {}
        
        # 对于模型中的每个参数
        # 假设所有客户端的模型结构相同，使用第一个客户端的模型作为参考
        first_client_data = next(iter(clients_data.values()))
        
        # 对于每个参数名称
        for param_name in first_client_data.keys():
            # 计算所有客户端该参数的平均值
            param_sum = None
            for client_data in clients_data.values():
                if param_name in client_data:
                    param_tensor = torch.tensor(client_data[param_name])
                    if param_sum is None:
                        param_sum = param_tensor.clone()
                    else:
                        param_sum += param_tensor
            
            # 计算平均值
            if param_sum is not None:
                global_model[param_name] = (param_sum / num_clients).tolist()
        
        print(f"Aggregated global model for {num_clients} clients")

if __name__ == '__main__':
    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000, debug=True)