export TRANSFORMERS_CACHE='/local-scratch1/data/ykhuang/cache/transformers'
export HF_DATASETS_CACHE='/local-scratch1/data/ykhuang/cache/hf_datasets'
export TFDS_DATA_DIR='/local-scratch0/data/ykhuang/cache/tensorflow_datasets'

CACHE_DIR="/local-scratch1/data/ykhuang/cache/qq"
export CUDA_VISIBLE_DEVICES=2


python3 preprocess.py --split validation --dataset hotpot --distract_or_focus focus --cache_dir $CACHE_DIR --load_from_cache False --masking_schemes randomsentence





