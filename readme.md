0. Install environment.yml with `conda env create --force`
1. Create a config.ini file (if you are using an Open AI model). It should look like:

[API_KEYS]
openai_api_key = YOUR_API_KEY

2. Run preprocess with `python3 preprocess.py --split validation --dataset hotpot --distract_or_focus focus --cache_dir .../qq/.cache --load_from_cache False --masking_schemes randomsentence`
3. Run main with `python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-None-focus --m1_arch t5-small --m2_arch repeater --oracle_arch t5-small --pm_eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence --adversarial_drop_thresh 0.5 --max_adversarial_examples 3`