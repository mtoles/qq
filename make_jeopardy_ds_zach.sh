export CUDA_VISIBLE_DEVICES=6
export TRANSFORMERS_CACHE="./.model_cache/"

# E4
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40

# E9
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 0
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 100
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 200
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 300
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 400

# E10
# python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --do_temp_scaling False

### E12
python3 ./make_jeopardy_ds_zach.py --split train --downsample_pt_size 2000 --rounds 40 --initial_seed 0
