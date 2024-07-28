export TRANSFORMERS_CACHE=./.model_cache 
export CUDA_VISIBLE_DEVICES=0
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


### Generate main figure ground truth subset results

python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ --alexpaca_path models/llama3_ft/500_07_24-03:22:17_E4_GOOD --gt_subset
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch repeater --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-4 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 


# # # Generate the main results figure

python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ --alexpaca_path models/llama3_ft/500_07_24-03:22:17_E4_GOOD
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch llama3 --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch repeater --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-3.5-turbo --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gpt-4 --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 


### generate the oracle size ablation results

python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size small --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size large --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size xl --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size xxl --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch gpt-3.5-turbo --template_id p3 --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch gpt-4 --template_id p2 --save_dir results/1_paper/oracle/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch llama3 --template_id p3 --save_dir results/1_paper/oracle/ 


# ### generate the m1 size ablation results

python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-small --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-base --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-large --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-xl --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch t5-xxl --m2_arch gt --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch gpt-3.5-turbo --m2_arch gt --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_eval_batch_size 64 --oracle_eval_batch_size 16 --m1_arch gpt-4 --m2_arch gt --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 

