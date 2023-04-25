export TRANSFORMERS_CACHE='/local-scratch1/data/ykhuang/cache/transformers'
#export HF_DATASETS_CACHE='/local-scratch0/data/ykhuang/cache/hf_datasets'
#export TFDS_DATA_DIR='/local-scratch0/data/ykhuang/cache/tensorflow_datasets'
#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export CUDA_VISIBLE_DEVICES=2

# https://github.com/tatsu-lab/stanford_alpaca
LLAMA_PATH_RAW="/local-scratch1/data/shared/llama/model_weights"
LLAMA_PATH_HF="$TRANSFORMERS_CACHE/llama"
ALPACA_PATH_DIFF="$TRANSFORMERS_CACHE/alpaca/diff"
ALPACA_PATH_TUNED="$TRANSFORMERS_CACHE/alpaca/tuned"


if [ -d "$LLAMA_PATH_HF" ]
then
  echo "llama model in huggingface format already exists"
else
  echo "llama files in huggingface format doesn't exist"
  python alpaca/convert_llama_weights_to_hf.py \
      --input_dir $LLAMA_PATH_RAW --model_size 7B --output_dir $LLAMA_PATH_HF
fi

if [ -f "$ALPACA_PATH_DIFF/config.json" ]
then
  echo "weights difference already exists"
else
  echo "Downloading weight difference to $ALPACA_PATH_DIFF"
  git clone https://huggingface.co/tatsu-lab/alpaca-7b-wdiff $ALPACA_PATH_DIFF
fi

python alpaca/weight_diff.py recover --path_raw $LLAMA_PATH_HF \
      --path_diff $ALPACA_PATH_DIFF --path_tuned $ALPACA_PATH_TUNED







