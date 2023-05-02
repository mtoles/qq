## Table of Contents
- [Install](#install)
- [Prepare](#prepare)
  - [Prepare Configures](#prepare-configures)
  - [Prepare Models](#prepare-models)
  - [Prepare Datasets](#prepare-datasets)
- [Main](#main)

## Install
Install environment.yml with 
```
conda env create --force
```

## Prepare
### Prepare Configures
Create a config.ini file (if you are using an Open AI model). It should look like:
```
[API_KEYS]
openai_api_key = YOUR_API_KEY
```
### Prepare Models
To get the alpaca-7B model, 
1. modify the path in [scripts/recover_alpaca.sh](scripts/recover_alpaca.sh)
```
TRANSFORMERS_CACHE="<YOUR_TRANSFORMERS_CACHE_PATH>" 
LLAMA_PATH_RAW="<YOUR_LLAMA_PATH>"
```
 
2. then run the script with 
```
bash scripts/recover_alpaca.sh
```
### Prepare Datasets
Run preprocess with 
```
python3 preprocess.py --split validation --dataset hotpot --distract_or_focus focus --cache_dir .../qq/.cache --load_from_cache False --masking_schemes randomsentence
```

## Main
Run main with default setting:
```
python3 main.py --pt_dataset_path data/preprocess/hotpot-validation-None-None-focus --m1_arch t5-small --m2_arch repeater --oracle_arch t5-small --pm_eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence --adversarial_drop_thresh 0.5 --max_adversarial_examples 3
```

Set flan-t5 as m2 model
```
python3 main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus \
--m1_arch t5-small --m2_arch flan-t5-large --oracle_arch t5-small \
--pm_eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence \
--adversarial_drop_thresh 0.5 --max_adversarial_examples 3 --downsample_pt_size 100 
```

Set alpaca as m2 model
```
python3 main.py --pt_dataset_path data/preprocess/hotpot-validation-None-randomsentenceNone-focus \
--m1_arch t5-small --m2_arch alpaca --oracle_arch t5-small \
--pm_eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence \
--adversarial_drop_thresh 0.5 --max_adversarial_examples 3 --downsample_pt_size 100 \
--alpaca_model_path <YOUR_ALPACA_MODEL_PATH>
```
