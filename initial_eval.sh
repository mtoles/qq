now=$(date +"%y%m%d-%H%M%S")
results_filename="bash_initial_eval_"$now

# Fast or slow mode
downsample_pt_size=100
# downsample_pt_size=None
load_from_cache=True

# Preprocess all the data
python3 /local-scratch1/data/mt/code/qq/preprocess.py --split validation --dataset hotpot --distract_or_focus focus --cache_dir /local-scratch1/data/mt/code/qq/.cache --load_from_cache $load_from_cache --masking_schemes randomsentence 

# Run BB
python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --pm_paths models/good_models/hotpot-final-model-exp-20230126-170030-bigbird-5epoch --pm_arch bigbird --masking_scheme None --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --pm_paths models/good_models/hotpot-final-model-exp-20230126-170030-bigbird-5epoch --pm_arch bigbird --masking_scheme supporting --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --pm_paths models/good_models/hotpot-final-model-exp-20230126-170030-bigbird-5epoch --pm_arch bigbird --masking_scheme randomsentence --downsample_pt_size $downsample_pt_size --results_filename $results_filename

# Run t5-small
python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --pm_arch t5-small --masking_scheme None --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus  --pm_arch t5-small --masking_scheme supporting --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus  --pm_arch t5-small --masking_scheme randomsentence --downsample_pt_size $downsample_pt_size --results_filename $results_filename

# Run t5-xxl
python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --pm_arch t5-xxl --masking_scheme None --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus  --pm_arch t5-xxl --masking_scheme supporting --downsample_pt_size $downsample_pt_size --results_filename $results_filename

python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus  --pm_arch t5-xxl --masking_scheme randomsentence --downsample_pt_size $downsample_pt_size --results_filename $results_filename