#!/bin/bash

export TRANSFORMERS_CACHE=./.model_cache 
# export CUDA_VISIBLE_DEVICES=1
# for quick testing
# DOWNSAMPLE_PT_SIZE=10
# for the real run
# DOWNSAMPLE_PT_SIZE="None"

### prompt ablation

## llama3
## p3 works best
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p4 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p5 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch llama3 --template_id p6 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/llama3 --downsample_pt_size 400 

## gpt 3.5
## p3 works best 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p4 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p5 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p6 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-3.5-turbo --downsample_pt_size 400 

## gpt 4
## p3 works best
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p4 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p5 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --split train --m1_arch t5-base --m2_arch gpt-4 --template_id p6 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/gpt-4 --downsample_pt_size 400 


# ### Generate main figure ground truth subset results
# CUDA_VISIBLE_DEVICES=1
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch repeater --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-4 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ --alexpaca_path models/llama3_ft/500_07_24-03:22:17_E4_GOOD --gt_subset


# # # Generate the main results figure
# CUDA_VISIBLE_DEVICES=2
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch repeater --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-4 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ --alexpaca_path models/llama3_ft/500_07_24-03:22:17_E4_GOOD


# # ### generate the oracle size ablation results
# export CUDA_VISIBLE_DEVICES=2
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size small --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size large --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size xl --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 8 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size xxl --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch gpt-3.5-turbo --template_id p3 --save_dir results/1_paper/oracle/ 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch gpt-4 --template_id p2 --save_dir results/1_paper/oracle/ 


# # ### generate the m1 size ablation results
# export CUDA_VISIBLE_DEVICES=1
# # python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-small --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-large --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-xl --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-xxl --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch gpt-3.5-turbo --m2_arch gt --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch gpt-4 --m2_arch gt --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ --supp_eval 

### E9 seeds
### full
# export CUDA_VISIBLE_DEVICES=1
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s0 --gt_subset --alexpaca_path models/llama3_ft/E9/500_07_28-22:01:57_s0

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s100 --gt_subset --alexpaca_path models/llama3_ft/E9/500_07_29-00:18:35_s100

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s200 --gt_subset --alexpaca_path models/llama3_ft/E9/500_07_29-07:01:29_s200

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s300 --gt_subset --alexpaca_path models/llama3_ft/E9/500_07_29-01:29:13_s300

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s400 --gt_subset --alexpaca_path models/llama3_ft/E9/500_07_29-02:03:44_s400

# ### full

CUDA_VISIBLE_DEVICES=1 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s0 --alexpaca_path models/llama3_ft/E9/500_07_30-07:54:55 &

CUDA_VISIBLE_DEVICES=2 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s100 --alexpaca_path models/llama3_ft/E9/500_07_30-08:28:28 &

CUDA_VISIBLE_DEVICES=3 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s200 --alexpaca_path models/llama3_ft/E9/500_07_30-09:02:23 &

CUDA_VISIBLE_DEVICES=4 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s300 --alexpaca_path models/llama3_ft/E9/500_07_30-09:36:14 &

CUDA_VISIBLE_DEVICES=6 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E9/s400 --alexpaca_path models/llama3_ft/E9/500_07_30-10:10:49 &


### E7 Round cutoff
# export CUDA_VISIBLE_DEVICES=2
# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r0 --alexpaca_path models/llama3_ft/E7/500_cuttoff_0_07_29-02:26:13

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r5 --alexpaca_path models/llama3_ft/E7/500_cuttoff_5_07_29-03:00:19

# python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r10 --alexpaca_path models/llama3_ft/E7/500_cuttoff_10_07_29-03:34:16

# CUDA_VISIBLE_DEVICES=1 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r15 --alexpaca_path models/llama3_ft/E7/500_cuttoff_15_07_29-04:08:55 &

# CUDA_VISIBLE_DEVICES=2 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r20 --alexpaca_path models/llama3_ft/E7/500_cuttoff_20_07_29-04:43:01 &

# CUDA_VISIBLE_DEVICES=3 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r25 --alexpaca_path models/llama3_ft/E7/500_cuttoff_25_07_29-05:17:58 &

# CUDA_VISIBLE_DEVICES=4 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r30 --alexpaca_path models/llama3_ft/E7/500_cuttoff_30_07_29-05:52:38 &

# CUDA_VISIBLE_DEVICES=6 python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/E7/r35 --alexpaca_path models/llama3_ft/E7/500_cuttoff_35_07_29-06:26:59 &