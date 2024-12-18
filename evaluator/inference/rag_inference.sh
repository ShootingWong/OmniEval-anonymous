#!/bin/bash
NODE_ROOT="corpus/nodes_dir" # the root to save the built knowledge corpus
#
SAVE_NAME="few_shot_test"
CHUNK_SIZE=2048
CHUNK_OVERLAP=256
NODE_NAME="${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}_nodes"
INDEX_ROOT="corpus/flash_rag_index"
PRED_RESULT_ROOT="evaluator/pred_results"
RAG_FRAMEWORK="vllm-openai"
INFER_BATCH=4
PER_GPU_EMBEDDER_BATCH_SIZE=32

generators=("llama3-70b-instruct" "qwen2-72b" "yi15-34b" "deepseek-v2-chat")
retrievers=("bge-m3" "bge-large-zh" "e5-mistral-7b" "gte-qwen2-1.5b" "jina-zh")

for generator in "${generators[@]}"
do
  for retriever in "${retrievers[@]}"
  do
    if [[ $retriever == "gte-qwen2-1.5b" || $retriever == "gte-qwen2-7b" || $retriever == "jina-zh" ]]; then
      retrieval_use_fp16=false
    else
      retrieval_use_fp16=true
    fi
    python evaluator/inference/rag_model.py \
      --node_root "${NODE_ROOT}/${NODE_NAME}" \
      --rawdoc_root "${NODE_ROOT}/${NODE_NAME}" \
      --data_gen_root "data_generator/${EVAL_GENDATA_NAME}" \
      --index_path $INDEX_ROOT \
      --retrieval_method $retriever \
      --generator_model $generator \
      --rag_framework $RAG_FRAMEWORK \
      --infer_batch $INFER_BATCH \
      --per_gpu_embedder_batch_size $PER_GPU_EMBEDDER_BATCH_SIZE \
      --pipline sequential \
      --pred_result_root $PRED_RESULT_ROOT \
      --use_sentence_transformer \
      --retrieval_topk 5 \
      --rag_system_prompt "{reference}\n\n 这是搜索增强的场景，助手需要使用资料原文回答用户的问题，不要做任何修改、归纳或总结。\n\n" \
      --rag_user_prompt "问题: {question}\n答案:" \
      --rag_conversation_prompt "\n历史对话问题记录：{}\n当前问题：{}" \
      --save_pred_result \
      --retrieval_use_fp16 $retrieval_use_fp16
  done
done
        