export CUDA_VISIBLE_DEVICES=5,6
export TRANSFORMERS_CACHE="/local/data/mt/qq/.model_cache/"

# quick testing
# CUDA_VISIBLE_DEVICES=2,3 torchrun  --nproc_per_node 2 --master_port 5204 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_2000_train_zach_filtered_tatsu_E4.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 2 --estring DEBUG --gt_subset

# gpt
# torchrun --nproc_per_node 2 --master_port 55221 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/jeopardy/jeopardy_gpt_1024_train_tatsu.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 1024

# zach E1
# torchrun --nproc_per_node 2 --master_port 55223 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_1000_train_zach_filtered_tatsu_E1.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 1024 --estring E1

# zach E2
# torchrun --nproc_per_node 2 --master_port 55229 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_1000_train_zach_filtered_tatsu_E2.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 500 --estring E2


# zach E3
# torchrun --nproc_per_node 2 --master_port 55225 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_1000_train_zach_filtered_tatsu_E1.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 500 --estring E3

# zach E4
# torchrun --nproc_per_node 2 --master_port 5204 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_2000_train_zach_filtered_tatsu_E4.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 500 --estring E4

# zach E5
# torchrun --nproc_per_node 2 --master_port 5206 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_1000_train_zach_filtered_tatsu_E5.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 500 --estring E5

# GPT-4 E7
for round_cutoff in {0..39}
do
    torchrun --nproc_per_node 2 --master_port 5204 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_2000_train_zach_filtered_tatsu_E7.jsonl --bf16 True --output_dir models/llama3_ft/E7 --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 500 --estring E7 --round_cutoff $round_cutoff
done