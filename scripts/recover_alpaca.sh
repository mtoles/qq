export TRANSFORMERS_CACHE='/local-scratch1/data/ykhuang/cache/transformers'
export HF_DATASETS_CACHE='/local-scratch0/data/ykhuang/cache/hf_datasets'
export TFDS_DATA_DIR='/local-scratch0/data/ykhuang/cache/tensorflow_datasets'


export CUDA_VISIBLE_DEVICES=7

python3 main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus --m1_arch t5-small --m2_arch repeater --oracle_arch t5-small --pm_eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence --adversarial_drop_thresh 0.5 --max_adversarial_examples 3





