DATA_ROOT="corpus" # the root to save all retrieval data information
DATA_DIR="few_shot" # the dirname of the source documents. You should first put your documents in the $DATA_ROOT/$DATA_DIR.
SAVE_NAME="few_shot_test" #the dirname to save the built knowledge corpus. 
CHUNK_SIZE=2048
CHUNK_OVERLAP=256
python corpus_builder/build_corpus.py \
    --transcript_directory $DATA_ROOT/$DATA_DIR \
    --rawdoc_root "$DATA_ROOT/nodes_dir/${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}_docs" \
    --node_root "$DATA_ROOT/nodes_dir/${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}_nodes" \
    --DEFAULT_CHUNK_SIZE $CHUNK_SIZE \
    --SENTENCE_CHUNK_OVERLAP $CHUNK_OVERLAP
