import os
import json
import sys 
from tqdm import tqdm 
import numpy as np 
from collections import defaultdict

def load_data(path):
    datas = [json.loads(line) for line in open(path, encoding='utf8').readlines()]
    if 'all_avg_retrieval_res_map' in datas[-1]:
        datas = datas[:-1]
    return datas

def get_valid_mean(list_):
    if isinstance(list_[0], list) : # rouge
        return np.mean(np.array(list_), axis=0)

    valid_list = [item for item in list_ if item >= 0]
    if len(valid_list) == 0: return -1
    return np.mean(np.array(valid_list), axis=0)


def extract_target_results(datas, key_metrics, merge_target, merge_result):
    for data in tqdm(datas):
        dataset = list(data.keys())[0]
        result = data[dataset][merge_target]
        if dataset not in merge_result:
            merge_result[dataset] = dict()
        if merge_target not in merge_result[dataset]:
            merge_result[dataset][merge_target] = dict()
        for metric in key_metrics:
            if metric in result:
                merge_result[dataset][merge_target][metric] = result[metric]

    return merge_result

def all_average(raw_result):
    sort_result = sorted(raw_result, key=lambda x: list(x.keys())[0])
                    
    # all average:
    all_ret_res = defaultdict(list)
    all_gen_res = defaultdict(list)
    valid_ret = True
    valid_gen = True
    for result_ in sort_result:
        result = list(result_.values())[0]
        assert result is not None, f"result is None, result_ = {result_}"
        if 'retrieval_res_map' not in result or result['retrieval_res_map'] is None:
            valid_ret = False
        else:
            ret_res = result['retrieval_res_map']
            for subkey in ret_res:
                all_ret_res[subkey] += [float(v) if '-' not in v else [float(v_) for v_ in v.split('-')] for v in ret_res[subkey]['list'].split(',')]
        if 'generation_res_map' not in result:
            valid_gen = False
        else:
            gen_res = result['generation_res_map']
            for subkey in gen_res:
                new_res = []
                for v in gen_res[subkey]['list'].split(','):
                    try:
                        v = float(v)
                    except:
                        v = [float(v_) for v_ in v.split('-')]
                    new_res.append(v)
                all_gen_res[subkey] += new_res
                
    if valid_ret:
        for key in all_ret_res:
            all_ret_res[key] = get_valid_mean(all_ret_res[key]) #np.mean(np.array(all_ret_res[key]), axis=0)
            if len(all_ret_res[key].shape) == 0:
                all_ret_res[key] = float(all_ret_res[key])
            else:
                all_ret_res[key] = ','.join([str(v) for v in all_ret_res[key].tolist()])
    if valid_gen:
        for key in all_gen_res:
            # loguru.logger.info(f'=== np.array(all_gen_res[{key}]) shape = {np.array(all_gen_res[key]).shape} data[0] = {np.array(all_gen_res[key])[0]}')
            all_gen_res[key] = get_valid_mean(all_gen_res[key]) # np.mean(np.array(all_gen_res[key]), axis=0)
            if len(all_gen_res[key].shape) == 0:
                all_gen_res[key] = float(all_gen_res[key])
            else:
                all_gen_res[key]  = ','.join([str(v) for v in all_gen_res[key].tolist()])
    all_avg_data = {
        "all_avg_retrieval_res_map": all_ret_res if valid_ret else None, 
        "all_avg_generation_res_map": all_gen_res if valid_gen else None,
    }
    raw_result.append(all_avg_data)
    return raw_result


def post_process(merge_result):
    new_result_list = []
    for dataset in merge_result:
        for target_key in all_target_key:
            if target_key not in merge_result[dataset]:
                merge_result[dataset][target_key] = None
        new_result_list.append({dataset: merge_result[dataset]})
    return new_result_list

def save_jsonl(datas, save_path):
    with open(save_path, 'w', encoding='utf8') as wf:
        for data in datas:
            wf.write(json.dumps(data, ensure_ascii=False) + '\n')

    
def merge(result_path, old_result_path, save_path):    
    results = load_data(result_path)
    old_results = load_data(old_result_path)
    merge_result = dict()
    for i, merge_target in enumerate(merge_target_key):
        merge_result = extract_target_results(results, result_keys[i], merge_target, merge_result)
        merge_result = extract_target_results(old_results, old_result_keys[i], merge_target, merge_result)
    
    merge_result = post_process(merge_result)
    merge_result = all_average(merge_result)
    save_jsonl(merge_result, save_path)

def merge_all(root, file_new, file_old, file_save, skip_close=False):
    models = os.listdir(root)
    for model in tqdm(models):
        if skip_close and 'CLOSE' in model : continue 
        path_new = os.path.join(root, model, file_new)
        path_old = os.path.join(root, model, file_old)
        path_save = os.path.join(root, model, file_save)
        merge(path_new, path_old, path_save)
        # print(f'=====> ')


result_keys = [["hallucination"]] 
old_result_keys = [["accuracy", "completeness", "utilization", "numerical_accuracy"]]
merge_target_key = ["generation_res_map"]
all_target_key = ["retrieval_res_map", "generation_res_map"]

file_new = "evaluation_result_model_qwen-eval-hallucination.jsonl"
file_old = "evaluation_result_model_qwen-eval.jsonl"
file_save = "evaluation_result_model_qwen-eval-merge.jsonl"

root = "evaluator/pred_results/gen_datas_${your_suffix}"
skip_close=True
merge_all(root, file_new, file_old, file_save, skip_close=skip_close)