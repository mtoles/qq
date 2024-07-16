export CUDA_VISIBLE_DEVICES=1,2
export TRANSFORMERS_CACHE="/local/data/mt/qq/.model_cache/"

# gpt
# torchrun --nproc_per_node 2 --master_port 55221 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/jeopardy/jeopardy_gpt_1024_train_tatsu.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 1024

# zach
torchrun --nproc_per_node 2 --master_port 55223 llama_train.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --data_path data/zach/jeopardy_1000_train_zach_filtered_tatsu.jsonl --bf16 True --output_dir models/llama3_ft --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 1024
