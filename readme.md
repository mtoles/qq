This is the repo for the upcoming paper "What is a good question? Task oriented question generation with fact-level
masking". It is a work in progress. It implements the TOQG pipeline including:

1. Primary model (Flan-T5)
2. Secondary models (ChatGPT, Alpaca (WIP), FlanT5 (WIP))
3. Oracle (Flan-T5)
4. Adversarial dataset generation algorithm
5. HotpotQA dataset ingestion and evaluation

Key files:

- `preprocess.py` generates a cached and prepped version of the HotpotQA dataset
- `main.py` runs the entire pipeline
- `primary_models.py` contains primary model implementations
- `secondary_models.py` contains secondary model implementations
- `oracles.py` contains oracle implementations
- `masking.py` contains the majority of adversarial generation code
- `.cache/` and `.model_cache` are used for caching

1. Install environment.yml with `conda env create --force`
2. Create a config.ini file (if you are using an Open AI model). It should look like:

```
[API_KEYS]
openai_api_key = YOUR_API_KEY
```

2. Run preprocess with `python3 preprocess.py --split validation --dataset hotpot --distract_or_focus focus --cache_dir .../qq/.cache --load_from_cache False --masking_schemes bfsentence`
3. Run main with `python3 /local-scratch1/data/mt/code/qq/main.py --pt_dataset_path data/preprocess/hotpot-validation-None-None-focus --m1_arch t5-small --m2_arch openai --oracle_arch t5-small --eval_batch_size 12 --oracle_eval_batch_size 12 --masking_scheme bfsentence --adversarial_drop_thresh 0.5 --max_adversarial_examples 3`

# Results
| Improvement | Section    | Before     | After     | Speedup   | 
|-------------|------------|------------|-----------|-----------|
| Baseline    | -          | 6353.589   | -         | -         |
| M1 Batching | M1         | 57.859     | 40.529    | 1.4       |
| Algorithm   | Adv. Gen.  | 4753.363   | 256.604   | 18.5      |
| M2 Caching  | M2         | 249.873    | 2.01      | 124.3     |
| Multi GPU   | M1         | 40.529     | 24.502    | 1.7       |