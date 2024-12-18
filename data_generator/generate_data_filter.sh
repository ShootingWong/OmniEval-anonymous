#!/bin/bash
NODE_ROOT="corpus/nodes_dir" # the root to save the built knowledge corpus
SAVE_NAME="few_shot_test" 
CHUNK_SIZE=2048
CHUNK_OVERLAP=256
NODE_NAME="${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}" # the save name of your built knowledge corpus (same as the one in build_corpus.sh)
API_KEY="" # your own api key
DATA_SUFFIX="test"
GEN_TYPE="filter" # when you need to do data quality inspection, sest this parameter as "filter"
python data_generator/generator_matrix.py \
    --node_root "$NODE_ROOT/${NODE_NAME}_nodes" \
    --rawdoc_root "$NODE_ROOT/${NODE_NAME}_docs" \
    --data_gen_suffix $DATA_SUFFIX \
    --gpt_version gpt-3.5-turbo-0125 \
    --data_gen_type $GEN_TYPE \
    --thread 20 \
    --apikey $API_KEY