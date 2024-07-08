export CUDA_VISIBLE_DEVICES="6,7"
export TRANSFORMERS_CACHE="/local-scratch1/data/mt/code/qq/.model_cache/"

DS_SIZE="6000"

# create the jeopardy dataset
# python3 /local-scratch1/data/mt/code/qq/make_jeopardy_ds.py --split train --downsample_pt_size $DS_SIZE --save_dir data/jeopardy/ --active_filter 

# our jeopardy data
# torchrun --nproc_per_node 2 --master_port 55222 train_stock.py --model_name_or_path .model_cache/alpaca/tuned --data_path data/jeopardy/jeopardy_${DS_SIZE}_train_active_filtered_tatsu.jsonl --bf16 True --output_dir models/alexpaca_tatsu --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap\ offload --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 256


torchrun --nproc_per_node 2 --master_port 55222 train_stock.py --model_name_or_path .model_cache/alpaca/tuned --data_path data/jeopardy/jeopardy_full_train_tatsu.jsonl --bf16 True --output_dir models/alexpaca_tatsu --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --fsdp full_shard\ auto_wrap\ offload --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --tf32 True --examples 1000