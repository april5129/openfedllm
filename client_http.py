import requests
import time
import json
import torch

# 服务端地址
SERVER_URL = "http://localhost:5000"

# 客户端ID（实际部署时应通过命令行参数传入）
CLIENT_ID = 0

def send_model_to_server(model_data):
    """发送本地模型到服务端"""
    url = f"{SERVER_URL}/send_model"
    payload = {
        "client_id": CLIENT_ID,
        "model_data": model_data
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Successfully sent model to server: {response.json()}")
            return True
        else:
            print(f"Failed to send model to server: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending model to server: {e}")
        return False

def get_global_model_from_server():
    """从服务端获取全局模型"""
    url = f"{SERVER_URL}/get_global_model"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Received global model from server: round {data['round']}")
            return data['global_model']
        else:
            print(f"Failed to get global model from server: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting global model from server: {e}")
        return None

def simulate_training(model_params, local_data):
    """模拟本地训练过程"""
    # 这里应该是实际的模型训练代码
    # 简单示例：对模型参数进行小幅度调整
    updated_params = {}
    for key, value in model_params.items():
        # 将参数转换为张量
        param_tensor = torch.tensor(value)
        # 添加一个小的随机扰动来模拟训练效果
        noise = torch.randn_like(param_tensor) * 0.01
        updated_params[key] = (param_tensor + noise).tolist()
    
    print("Simulated training completed with parameter updates")
    return updated_params

def main():
    # 初始化本地模型参数
    # 在实际应用中，这应该是从预训练模型中获取的参数
    local_model_params = {
        "param1": [0.1, 0.2, 0.3],
        "param2": [0.4, 0.5, 0.6]
    }
    
    # 模拟本地数据
    local_data = [1, 2, 3, 4, 5]  # 实际项目中这里应该是训练数据
    
    # 联邦学习训练循环
    for round_num in range(10):  # 假设训练10轮
        print(f"\n=== Round {round_num + 1} ===")
        
        # 1. 从服务端获取全局模型
        global_model = get_global_model_from_server()
        if global_model is not None:
            print(f"Using global model")
            # 使用服务端的全局模型参数
            local_model_params = global_model
        
        # 2. 执行本地训练
        local_model_params = simulate_training(local_model_params, local_data)
        
        # 3. 将本地模型发送给服务端
        success = send_model_to_server(local_model_params)
        
        if success:
            print("Round completed successfully")
        else:
            print("Round failed")
        
        # 等待一段时间再进行下一轮
        time.sleep(2)

if __name__ == "__main__":
    main()