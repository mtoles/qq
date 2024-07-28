export CUDA_VISIBLE_DEVICES=7
export TRANSFORMERS_CACHE="/local/data/mt/qq/.model_cache/"

# E4
# python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40

# E9
# python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 100
python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 200
python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 300
python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 400

# E10
# python3 /local/data/mt/qq/make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --do_temp_scaling False