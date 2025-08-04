max_steps=1 # 10
num_rounds=1 # 200
batch_size=1 # 16
gradient_accumulation_steps=1
seq_length=16  # 512
num_clients=2 # 5
sample_clients=2 # 2
lr=5e-4

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="Anthropic/hh-rlhf"  # Requires special access
dataset_sample=10 # 20000
model_name_or_path="openai-community/gpt2"  # Changed to smaller model
output_dir=./output
fed_alg="fedprox"

# gpu=2
# CUDA_VISIBLE_DEVICES=$gpu python3 runner/DPO/fedprox_runner.py \
python3 runner/DPO/fedprox_runner.py \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --learning_rate $lr \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "vicuna_v1.1" \