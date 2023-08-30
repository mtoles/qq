TRANSFORMERS_CACHE=./model_cache 
CUDA_VISIBLE_DEVICES=7 
# for quick testing
# DOWNSAMPLE_PT_SIZE=10
# for the real run
# DOWNSAMPLE_PT_SIZE="None"

# ### prompt ablation

# # chatgpt
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p4 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p5 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch openai --template_id p6 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/openai --gt_subset 

# # alpaca
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p2 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p3 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p4 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p5 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 
# python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p6 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/prompt/alpaca --gt_subset 

# Generate the main results figure

python3 main.py --m1_arch t5-base --m2_arch repeater --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
python3 main.py --m1_arch t5-base --m2_arch openai --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/ 
python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/main/

### Generate main figure ground truth subset results

python3 main.py --m1_arch t5-base --m2_arch repeater --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_arch t5-base --m2_arch openai --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 
python3 main.py --m1_arch t5-base --m2_arch alpaca --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/gt/ --gt_subset 

### generate the oracle size ablation results

python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size small --save_dir results/1_paper/oracle/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/oracle/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size large --save_dir results/1_paper/oracle/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size xl --save_dir results/1_paper/oracle/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size xxl --save_dir results/1_paper/oracle/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch openai --save_dir results/1_paper/oracle/ 


### generate the m1 size ablation results

python3 main.py --m1_arch t5-small --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_arch t5-base --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_arch t5-large --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_arch t5-xl --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_arch t5-xxl --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 
python3 main.py --m1_arch openai --m2_arch gt --template_id p1 --oracle_arch t5 --oracle_size base --save_dir results/1_paper/m1/ 

