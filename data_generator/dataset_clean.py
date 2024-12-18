import json
import pickle
import os
from collections import defaultdict

def clean_per_data(data):
    # question
    if isinstance(data['question'], list):
        flag = True
        if not isinstance(data['answer'], list) or len(data['answer']) != data['question']:
            flag = False
        if not isinstance(data['relevant_passage'], list) or len(data['relevant_passage']) != data['question']:
            flag = False
        if flag:
            new_datas = []
            for i in range(len(data['question'])):
                new_data = {
                    "though_process": data["though_process"],
                    "question": data["question"][i],
                    "answer": data["answer"][i],
                    "relevant_passage": data["relevant_passage"][i],
                    "topic_name": data["topic_name"],
                    "task_name": data["task_name"],
                    "relevant_node": data["relevant_node"],
                }
                new_data = clean_per_data(new_data)
                if new_data is None:
                    flag=False
                    break
                new_datas.append(new_data)
            if flag:
                return new_datas
            else:
                return None
        else:
            return None

    if data['question'].strip() == '': 
        return None

    data['question'] = data['question'].strip()

    # answer
    if isinstance(data['answer'], str):
        if data['answer'].strip() == '': 
            return None
    elif isinstance(data['answer'], list):
        new_answers = []
        for ans in data['answer']:
            if ans.strip() == '': 
                return None
            new_answers.append(ans.strip())
        if len(new_answers) == 0: 
            return None
        else:
            data['answer'] = new_answers
    else:
        return None

    # relevant_passage
    if isinstance(data['relevant_passage'], str):
        if data['relevant_passage'].strip() == '': return None
        data['relevant_passage'] = data['relevant_passage'].strip()
        data['relevant_passage'] = [data['relevant_passage']]
    elif isinstance(data['relevant_passage'], list):
        new_rel = []
        for rel in data['relevant_passage']:
            if rel.strip() == '': continue
            new_rel.append(rel.strip())
        data['relevant_passage'] = new_rel
    else:
        return None
        
    # relevant_node
    if isinstance(data['relevant_node'], str):
        data['relevant_node'] = [data['relevant_node']]
    elif not isinstance(data['relevant_node'], list):
        return None

    return data

def clean_data(datas):
    new_datas = []
    for data in datas:
        data = clean_per_data(data)
        if data is not None:
            new_datas.append(data)
    return new_datas

def clean_data_conversation(datas):
    new_datas = []
    for data in datas:
        if not isinstance(data, list): continue
        flag = True
        new_data = []
        for d in data:
            d = clean_per_data(d)
            if d is None:
                flag = False
                break
            else:
                if isinstance(d, list):
                    new_data = d
                    break
                else:
                    new_data.append(d)
        new_datas.append(new_data)

    return new_datas

def get_data_dict(root, new_root, topic_list, res_dict):
    this_root = os.path.join(root, '/'.join(topic_list))
    files = os.listdir(this_root)
    for f in files:
        path = os.path.join(this_root, f)
        if os.path.isdir(path):
            get_data_dict(root, new_root, topic_list + [f], res_dict)
        else:
            
            datas = [json.loads(line) for line in open(path, 'r', encoding='utf8').readlines()]

            print(f'=====> Clean for {" - ".join(topic_list)} : {f}')
            if '多轮对话能力' in f:
                new_datas = clean_data_conversation(datas)
            else:
                new_datas = clean_data(datas)
            res_dict['/'.join(topic_list)] = new_datas

            print(f'=====> Save for {" - ".join(topic_list)} : {f}')
            for i in range(1, len(topic_list)+1):
                inner_root = os.path.join(new_root, '/'.join(topic_list[:i]))
                print(f'i:{i} inner_root ={inner_root}')
                if not os.path.exists(inner_root):
                    os.mkdir(inner_root)

            save_path = os.path.join(new_root, '/'.join(topic_list), f)
            with open(save_path, 'w', encoding='utf8') as wf:
                for data in new_datas:
                    wf.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f'=====> Save for {" - ".join(topic_list)} : {f} OVER')
    # return res_dict

def root_clean(root, new_root):
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    topic_list = []
    res_dict = {}
    get_data_dict(root, new_root, topic_list, res_dict)
    print(f'=====> Inner root_clean  Save all over')


def get_data_dict_forcsv(path, new_root, res_dict):
    # this_root = os.path.join(root, '/'.join(topic_list))
 
    datas = [json.loads(line) for line in open(path, 'r', encoding='utf8').readlines()]
    topic2datas = defaultdict(dict)
    for data in datas:
        topic_name = data['topic_name'] if isinstance(data, dict) else data[0]['topic_name'] 
        # topic_list = topic_name.split(' - ')[1:]
        task_name = data['task_name'] if isinstance(data, dict) else data[0]['task_name'] 
        if task_name not in topic2datas[topic_name]:
            topic2datas[topic_name][task_name] = []
        topic2datas[topic_name][task_name].append(data)
    
    for topic_name in  topic2datas:
        for task_name in topic2datas[topic_name]:
            datas = topic2datas[topic_name][task_name]
            topic_list = topic_name.split(' - ')[1:]
            f = task_name + '.jsonl'
            print(f'=====> Clean for {task_name} : {f}')
            if '多轮对话能力' in f:
                new_datas = clean_data_conversation(datas)
            else:
                new_datas = clean_data(datas)
            res_dict['/'.join(topic_list)] = new_datas

            print(f'=====> Save for {" - ".join(topic_list)} : {f}, newdata cnt = {len(new_datas)} pre datacnt = {len(datas)}')
            for i in range(1, len(topic_list)+1):
                inner_root = os.path.join(new_root, '/'.join(topic_list[:i]))
                print(f'i:{i} inner_root ={inner_root}')
                if not os.path.exists(inner_root):
                    os.mkdir(inner_root)

            save_path = os.path.join(new_root, '/'.join(topic_list), f)
            if len(new_datas) == 0:
                print(f'=====> valid data cnt of {topic_name} {task_name} is 0, skip to save it... ')
                continue
            with open(save_path, 'w', encoding='utf8') as wf:
                for data in new_datas:
                    wf.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f'=====> Save for {topic_name} {task_name} : {f} OVER')
        # return res_dict

def root_clean_forcsv(path, new_root):
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    res_dict = {}
    get_data_dict_forcsv(path, new_root, res_dict)
    print(f'=====> Inner root_clean for csv, Save for {path} over')

if __name__ == '__main__':
    # root = 'data_generator/gen_datas_useadd2048_matrix_merge'
    # new_root = 'data_generator/gen_datas_useadd2048_matrix_merge_clean'
    # root_clean(root, new_root)
    path = 'data_generator/human_annotated/v1022/v1022.jsonl'
    new_root = 'data_generator/human_annotated_v1'
    root_clean_forcsv(path, new_root)