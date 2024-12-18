#!/bin/bash
NODE_ROOT="corpus/nodes_dir" # the root to save the built knowledge corpus
# 
SAVE_NAME="few_shot_test" 
CHUNK_SIZE=2048
CHUNK_OVERLAP=256
NODE_NAME="${SAVE_NAME}-${CHUNK_SIZE}-${CHUNK_OVERLAP}_nodes"
EVAL_GENDATA_NAME="gen_datas_${your_fix}" 

retriever="bge-large-zh,gte-qwen2-1.5b" # set your target evaluation retriever model names, split by ','. The corresponding name2path map should be set in configs/model2path.json
generator_model="deepseek-v2-chat,yi15-34b" # set your target evaluation generation model names, split by ','. The corresponding name2path map should be set in configs/model2path.json

retrieval_topk=5
judge_type="rule" # model or rule
hallu_eval=0 # hallucination evaluator is different to other evaluator. 1 = use do hallucination evaluation, 0 = do other model-based metric evaluation. This parameter is only valid when judge_type="model"

MODEL_EVAL_API=http://localhost:8000
HALLU_EVAL_API=http://localhost:8000
if [ $judge_type = "model" ]; then 
  echo "model-based evaluate!!"
  if [ $hallu_eval = 1 ]; then
    echo "evaluate hallucination!!"
    eval_suffix="qwen-eval-hallucination"
    openai_api=$HALLU_EVAL_API
    model_eval_metrics="hallucination"
    eval_batch_size=4
  else 
    echo "evaluate except hallucination!!"
    eval_suffix="qwen-eval"
    openai_api=$MODEL_EVAL_API
    model_eval_metrics="accuracy,completeness,utilization,numerical_accuracy"
    eval_batch_size=4
  fi
else
  echo "rule-based evaluate!!"
  eval_suffix="none"
  openai_api=$MODEL_EVAL_API # useless
  model_eval_metrics="accuracy,completeness,utilization,numerical_accuracy" # useless
  eval_batch_size=2 # useless
fi

python -u evaluator/judgement/judger.py \
  --thread 50 \
  --node_root "$NODE_ROOT/${NODE_NAME}" \
  --data_gen_root data_generator/$EVAL_GENDATA_NAME \
  --retrieval_method $retriever \
  --retrieval_topk $retrieval_topk \
  --generator_model $generator_model \
  --pred_result_root evaluator/pred_results \
  --infer_type vllm_server \
  --temperature 0 \
  --judge_type $judge_type \
  --eval_batch_size $eval_batch_size \
  --openai_api $openai_api \
  --eval_suffix $eval_suffix \
  --model_eval_metrics $model_eval_metrics \
  # --close_book # turn it on for the close-book setting.
  