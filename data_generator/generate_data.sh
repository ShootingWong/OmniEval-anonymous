#!/bin/bash
NODE_ROOT="corpus/nodes_dir" # the root to save the built knowledge corpus
SAVE_NAME="few_shot_test" 
CHUNK_SIZE=2048
CHUNK_OVERLAP=256
NODE_NAME="${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}" # the save name of your built knowledge corpus (same as the one in build_corpus.sh)
API_KEY="" # your own api key
DATA_SUFFIX="test" # the indication of your current generation.

# Notion:
# 1. when the api request is broken, you can set data_gen_bg_idx to the last data_index to continue the previous generation and avoid generating data samples from the same documents
# 2. I recommend to user gpt4 to generate rather than gpt3.5

python -u data_generator/generator_matrix.py \
    --node_root "$NODE_ROOT/${NODE_NAME}_nodes" \
    --rawdoc_root "$NODE_ROOT/${NODE_NAME}_docs" \
    --data_gen_suffix $DATA_SUFFIX \
    --data_gen_bg_idx 0 \
    --thread 20 \
    --infer_type gpt \
    --gpt_version gpt-3.5-turbo-0125 \
    --openai_api https://api.openai.com \
    --apikey $API_KEY
    #  https://api.openai.com/v1/chat/completions
