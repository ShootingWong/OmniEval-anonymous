import sys
sys.path.append('.')
from utils.options import get_options
from utils.llm_models import MyLLM
from utils.corpus_loader import Corpus
import json
import os
from tqdm import tqdm
from multiprocessing import Lock
import json_repair
from data_generator.prompt_matrix import *
import copy
from llama_index.core.schema import NodeWithScore, MetadataMode
import numpy as np
from data_generator.task_requirements import task_tree, data_format_tree
from data_generator.dataset_clean import root_clean
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
import re
import loguru 

class DataGenerator:
    def __init__(self, opt, corpus=None):
        self.opt = opt
        self.llm = MyLLM(opt)
        # self.llm.load_llm()
        if corpus is None:
            loguru.logger.info(f'=====> DataGenerator Init Provide NO corpus, load now!')
            self.corpus = Corpus(opt)
            self.corpus.load_corpus_from_nodes(thread=self.opt.thread)
        else:
            loguru.logger.info(f'=====> DataGenerator Init Provide corpus!')
            self.corpus = corpus

        self.task_tree = task_tree 
        self.read_topic_tree()
        loguru.logger.info(f'========> Inner initial, Before get_task_names')
        self.get_task_names()
        loguru.logger.info(f'========> Inner initial, After get_task_names')
        
        self.data_format_tree = data_format_tree

        self.lock = Lock()

        self.topic_classify_system = topic_classify_system
        self.topic_classify_user = topic_classify_user

        self.task_classify_system = task_classify_system
        self.task_classify_user = task_classify_user

        self.data_generation_system = data_generation_system
        self.data_generation_user = data_generation_user

        self.data_filter_system = data_filter_system
        self.data_filter_user = data_filter_user
        
    def read_topic_tree(self):
        
        topic_tree_path = self.opt.topic_tree_path
        topic_tree = json.load(open(topic_tree_path))
        loguru.logger.info(f'======> Inner read_topic_tree topic_tree = {topic_tree} self.opt.max_topic_depth = {self.opt.max_topic_depth}')
        def parse_tree(tree, prefix, depth):
            all_trees = []
            loguru.logger.info(f'Depth = {depth} prefix = {prefix}')
            
            if prefix == '':
                new_fix = tree['topic_name']
            else:
                new_fix = prefix + ' - ' + tree['topic_name']
            if depth >= self.opt.max_topic_depth: 
                return [new_fix]
            
            for sub_tree in tree['sub_topics']:
                all_trees += parse_tree(sub_tree, new_fix, depth+1)
            return all_trees
        
        self.topics = parse_tree(topic_tree, '', 0)
        
    def get_task_names(self):
        self.task_names = []
        loguru.logger.info(f'======> inner get_task_names self.task_tree = {self.task_tree}')
        for task in self.task_tree:
            self.task_names.append(task)
        loguru.logger.info(f'=====> get_task_names over, self.task_names = {self.task_names}')

    def get_task_info_for_classify(self):
        all_tasks = []
        for i, task in enumerate(self.task_tree):
            desc = self.task_tree[task]
            
            desc = desc.split("### 任务要求")[1].strip()
            data = json.dumps({
                "id": i+1,
                "name": task,
                "description": desc
            }, ensure_ascii=False)
            all_tasks.append(data)
        
        return '\n'.join(all_tasks), self.task_names

    
    def get_topics_strs(self):
        all_datas = []
        loguru.logger.info(f'======> Inner get_topics_strs self.topics= {self.topics}')
        for i, topic in enumerate(self.topics):
            all_datas.append(json.dumps({
                "id": i+1,
                "topic_name": topic
            }, ensure_ascii=False))            
        loguru.logger.info(f'======> Inner get_topics_strs all_datas = {all_datas}')
        return '\n'.join(all_datas)
    
    # need llm
    def topic_classify_from_doc(self, doc):
        content = doc.get_content(MetadataMode.NONE)
        title = doc.metadata['Title']
        loguru.logger.info(f'=======> Classify doc title = {title} ')
        topics = self.get_topics_strs()
        loguru.logger.info(f'======> Inner topic_classify_system topics = {topics}')
        user_input = self.topic_classify_user.format(title=title, content=content, topics_str=topics)
        loguru.logger.info(f'======> Inner topic_classify_system user_input = {user_input}')
        output = self.llm.get_llm_output([self.topic_classify_system, user_input])[0]
        loguru.logger.info(f'=======> Inner topic_classify_system output = {output}')

        try:
            topic_id = int(json_repair.loads(output)['topic_id']) - 1
        except:
            loguru.logger.info('========> Load topic classified id Error! Skip...')
            return None

        if topic_id < 0 or topic_id >= len(self.topics):
            loguru.logger.info('========> Invalid topic classified id! Skip...')
            return None
        # loguru.logger.info(f'=======> doc-metadata = {doc.metadata}\nsel_topic = {self.topics[topic_id]}')
        return topic_id
    

    # need llm
    def task_classify_from_doc(self, doc, topic_id):
        content = doc.get_content(MetadataMode.NONE)
        metadata = doc.metadata

        def valid_check(output):
            if not isinstance(output, list): return False
            try:
                for idx in output:
                    if int(idx) > len(task_names): return False
            except:
                return False
            return True
            
        all_task_strs, task_names = self.get_task_info_for_classify()
        # loguru.logger.info(f'==========doc metadata = {metadata} all_task_strs = {all_task_strs}\n=======title = {doc.metadata["Title"]}')
        user_input = self.task_classify_user.format(title=doc.metadata['Title'], content=content, task_str=all_task_strs, topic_str=self.topics[topic_id])
        
        # loguru.logger.info(f'======> Inner task_classify_from_doc user_input = {user_input}')
        output = self.llm.get_llm_output([self.task_classify_system, user_input])[0]
        # loguru.logger.info(f'======> Inner task_classify_from_doc output = {output}')
        try:
            output = json_repair.loads(output)['task_id_list']
        except:
            loguru.logger.info('========> Load task classified id Error! Skip...')
            output = None
        if not valid_check(output):
            class_ids = None
        else:
            class_ids = [int(o)-1 for o in output]
        
        # class_ids = list(set(class_ids) & set([7,8]))

        loguru.logger.info(f'=======> doc-metadata = {metadata}\nsel_class = {[task_names[ids] for ids in class_ids]}')

        return class_ids
    
    # need llm
    def _data_generation(self, doc, topic_ids=None, class_ids=None, need_topic_tasks=None):
        def parse_data(data):
            if not isinstance(data, dict): return None
            if not('question' in data and 'answer' in data and 'relevant_passage' in data): return None
            return data
        
        def parse_output(output):
            try:
                output = json_repair.loads(output)
            except:
                loguru.logger.info('========> Inner _data_generation, json_repair LOAD OUTPUT ERROR')
                return None
            
            if not isinstance(output, list):
                output = parse_data(output)
                if output is not None:
                    new_output = [output]
                else:
                    new_output = []
            else:
                new_output = []
                for out in output:
                    out = parse_data(out)
                    if out is None: continue
                    new_output.append(out)
            if len(new_output) == 0:
                new_output = None
            return new_output
        
        def generation(system, user, sel_topic, task_name, docid_):
            gen_datas = []
            output = self.llm.get_llm_output([system, user], use_batch=False)[0]
            loguru.logger.info(f'=========> Inner _data_generation llm output = {output}')
            output = parse_output(output)
            if output:
                for out in output:
                    out['topic_name'] = sel_topic
                    out['task_name'] = task_name
                    out['relevant_node'] = docid_
                if task_name == '多轮对话能力':
                    gen_datas += [output]
                else:
                    gen_datas += output
            return gen_datas
        
        content = doc.get_content(metadata_mode=MetadataMode.NONE)
        title = doc.metadata['Title']
        
        only_structure = False #True #False
        skip_stock = False #True
        if only_structure:
            loguru.logger.info(f'=====> Only generation for structural !!')
            doc_is_structure = self.contains_markdown_table(content) or self.contains_html_table(content)
            if skip_stock:
                stock = '股票招股意向书' in title
            else:
                stock = False
            if not doc_is_structure: 
                loguru.logger.info(f'=====> {title} don\'t contain structural information, skip to generate 结构化知识问答 ')
                return []
            if stock:
                loguru.logger.info(f'=====> {title} contain stock-related structural info, skip... ')
                return []
        
        topic_id = self.topic_classify_from_doc(doc)
        if topic_id is None:
            loguru.logger.info(f'======> Doc: title: {doc.metadata["Title"]} do not belong to any subtopics, skip...')
            return []
        
        need_topic_ids = list(need_topic_tasks.keys())  if need_topic_tasks is not None else None
        
        if need_topic_ids is not None and topic_id  not in need_topic_ids:
            loguru.logger.info(f'=====> Skip {self.topics[topic_id], topic_id}')
            return []
        
        if topic_ids is not None:
            if topic_id not in topic_ids:
                loguru.logger.info(f'=====> Only need Topic: {self.topics[i] for i in topic_ids}, skip for current topic: {self.topics[topic_id]}')
                return [] 

        if class_ids is None:
            class_ids = np.arange(len(self.task_names)).tolist()

        if need_topic_tasks is not None:
            class_ids = need_topic_tasks[topic_id]
        if class_ids is None or len(class_ids) == 0:
            loguru.logger.info(f'=========> Doc: title: {doc.metadata["title"]} cannot be classfied and generate data! Skip')
            return []

        all_gen_datas = []
        
        loguru.logger.info(f'======> For data generate, title = {title}, topic_id = {topic_id} topic = {self.topics[topic_id]} class_ids = {class_ids} class = {[self.task_names[id_] for id_ in class_ids]}')

        sel_topic = self.topics[topic_id]
        
        for class_id in class_ids:
            task_name = self.task_names[class_id]
            task_desc = self.task_tree[task_name]
            gen_datas = []

            if task_name in ['对比类问答', '长答案形式问答', '多轮对话能力', '多跳推理类问答']:
                later_docs = self.corpus.get_later_neighbor_nodes(doc)
                loguru.logger.info(f'======> len(later_docs) = {len(later_docs)}')

                for later_doc in later_docs[:5]:
                    later_content = later_doc.get_content(metadata_mode=MetadataMode.NONE)
                    later_title = later_doc.metadata['Title']
                    multi_doc_str = multi_doc_format.format(doc_str_1=doc_str_format.format(title=title, content=content), doc_str_2=doc_str_format.format(title=later_title, content=later_content))
                    if task_name == "多跳推理类问答":
                        input_task_name = task_name + "(“实体-关系”链路类)"
                    else:
                        input_task_name = task_name 
                    user_input = self.data_generation_user.format(topic_name=sel_topic, task_name=input_task_name, task_require=task_desc, doc_str=multi_doc_str)
                    loguru.logger.info(f'=========> Inner _data_generation user_input = {user_input}')
                    gen_datas += generation(self.data_generation_system, user_input, sel_topic, task_name, [doc.id_, later_doc.id_])
            # else:
            if task_name in ['抽取类问答', '多跳推理类问答']:
                # continue
                if task_name == "多跳推理类问答":
                    input_task_name = task_name + "(金融计算类)"
                    loguru.logger.info(f"=====> Try for {input_task_name}")
                else:
                    input_task_name = task_name
                user_input = self.data_generation_user.format(topic_name=sel_topic, task_name=input_task_name, task_require=task_desc, doc_str=doc_str_format.format(title=title, content=content))
                loguru.logger.info(f'=========> Inner _data_generation user_input = {user_input}')
                gen_datas = generation(self.data_generation_system, user_input, sel_topic, task_name, doc.id_)
            
            if len(gen_datas) == 0:
                loguru.logger.info(f'=========> For title {title} od {doc.id_} cannot generate Topic: {sel_topic} Task:{input_task_name} data! Skip')
            all_gen_datas += gen_datas

        return all_gen_datas


    def save_gen_task_data(self, gen_data, task_save_path):
        if isinstance(gen_data, dict):
            sel_topic_name = gen_data['topic_name']
            sel_task_name = gen_data['task_name']
        elif isinstance(gen_data, list):
            sel_topic_name = gen_data[0]['topic_name']
            sel_task_name = gen_data[0]['task_name']
        else:
            raise TypeError("=======> Invalid gen_data type! skip..")
        save_path = os.path.join(task_save_path[sel_topic_name], f'{sel_task_name}.jsonl')
        self.lock.acquire()
        wf = open(save_path, 'a+', encoding='utf8')
        wf.write(json.dumps(gen_data, ensure_ascii=False)+'\n')
        self.lock.release()


    def get_task2save_path(self):
        save_root = self.opt.data_gen_root + f'_{self.opt.data_gen_suffix}'
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        task_save_path = {}
        for topic in self.topics:
            sub_topics = topic.split(' - ')
            if sub_topics[0] == '金融':
                sub_topics = sub_topics[1:]
            
            father_topics = []
            for sub_topic in sub_topics:
                father_topics.append(sub_topic)
                cur_root = os.path.join(save_root, '/'.join(father_topics))
                if not os.path.exists(cur_root):
                    os.mkdir(cur_root)
            father_root = os.path.join(save_root, '/'.join(father_topics))
            task_save_path[topic] = father_root
            
        loguru.logger.info(f'===========> task_save_path = {task_save_path}')
        return task_save_path
    
    def _gen_data_filter(self, data):
        def replace(data, new_data):
            for key in new_data:
                data[key] = new_data[key]
            return data

        if isinstance(data, list):
            # conversation data
            topic_name = data[0]["topic_name"]
            task_name = data[0]["task_name"]
            relevant_node = [self.corpus.node_map[node] for node in data[0]["relevant_node"]]
            data_input = copy.deepcopy(data)
            for d in data_input:
                del d["topic_name"]
                del d["task_name"]
                del d["relevant_node"]
        else:
            data_input = copy.deepcopy(data)
            # del data_input["thought_process"]
            del data_input["topic_name"]
            del data_input["task_name"]
            del data_input["relevant_node"]
            topic_name = data["topic_name"]
            task_name = data["task_name"]
            relevant_node = [self.corpus.node_map[node] for node in data["relevant_node"]]
            
        if len(relevant_node) == 1:
            content = relevant_node[0].get_content(metadata_mode=MetadataMode.NONE)
            title = relevant_node[0].metadata['Title']
            doc_str = doc_str_format.format(title=title, content=content)

        elif len(relevant_node) == 2:
            content = relevant_node[0].get_content(metadata_mode=MetadataMode.NONE)
            title = relevant_node[0].metadata['Title']

            later_content = relevant_node[1].get_content(metadata_mode=MetadataMode.NONE)
            later_title = relevant_node[1].metadata['Title']

            doc_str = multi_doc_format.format(doc_str_1=doc_str_format.format(title=title, content=content), doc_str_2=doc_str_format.format(title=later_title, content=later_content))
        

        system = self.data_filter_system 
        task_desc = self.task_tree[task_name]

        user_input = self.data_filter_user.format(doc_str=doc_str, topic_name=topic_name, task_name=task_name, task_require=task_desc, gen_datas=json.dumps(data_input, ensure_ascii=False))

        output = self.llm.get_llm_output([system, user_input], use_batch=False)[0]
        result = json_repair.loads(output)

        if not isinstance(result, dict) or "evaluation" not in result:
            result = {"evaluation": -1}
            
        if result["evaluation"] == 0:
            return None
        elif result["evaluation"] == 1:
            corrected_result = result["corrected_result"]
            flag = True
            try:
                if isinstance(data_input, dict) and isinstance(corrected_result, list):
                    corrected_result = corrected_result[0]
                    
                assert type(data_input) == type(corrected_result)
                if isinstance(data_input, list):
                    assert len(data_input) == len(corrected_result)
                    for i in range(len(data_input)):
                        assert type(data_input[i]) == type(corrected_result[i])
                        assert list(sorted(list(data_input[i].keys()))) == list(sorted(list(corrected_result[i].keys())))
                else:
                    assert type(data_input) == type(corrected_result)
                    assert list(sorted(list(data_input.keys()))) == list(sorted(list(corrected_result.keys())))

            except:
                flag = False
                loguru.logger.info(f"======> Corrected Error, use original data.. corrected_result = {corrected_result}")
            if flag:
                if isinstance(data, list):
                    for i in range(len(data)):
                        data[i] = replace(data[i], corrected_result[i])
                else:
                    data = replace(data, corrected_result)
        elif result["evaluation"] != 2:
            result["evaluation"] = -1
            loguru.logger.info(f"======> Evaluation Error, use original data..")

        if isinstance(data, list):
            for d in data:
                d['evaluation'] = result["evaluation"]
        else:
            data['evaluation'] = result["evaluation"]
        
        return data

    def gen_data_filter(self):
        root = self.opt.data_gen_root + f'_{self.opt.data_gen_suffix}'
        clean_root = root + '_clean'
        newroot = root + '_filter'
        loguru.logger.info(f'=====> Inner gen_data_filter, root = {root} clean_root = {clean_root} newroot = {newroot}')
        if not os.path.exists(clean_root):
            root_clean(root, clean_root)
            loguru.logger.info(f'=====> Inner gen_data_filter, build clean root over')
        else:
            loguru.logger.info(f'=====> Inner gen_data_filter, Clean root exists')

        for sub_root, dirs, files in os.walk(clean_root):
            if len(files) > 0:
                topic_lines = [item for item in sub_root.split(clean_root)[1].split('/') if item.strip() != '']
                for i in range(len(topic_lines)+1):
                    new_sub_root = os.path.join(newroot, '/'.join(topic_lines[:i]))
                    if not os.path.exists(new_sub_root):
                        os.mkdir(new_sub_root)
                new_sub_root = os.path.join(newroot, '/'.join(topic_lines))

                for f in files:
                    datas = [json.loads(line) for line in open(os.path.join(sub_root, f), encoding='utf8').readlines()]
                    valid_cnt = 0
                    
                    new_datas = []
                    new_save_path = os.path.join(new_sub_root, f)
                    if os.path.exists(new_save_path):
                        loguru.logger.info(f'=====> {new_save_path} exists, skip to filter it.')
                        continue
                    for i, data in enumerate(datas):
                        if isinstance(data, list) and len(data) == 0:
                            loguru.logger.info(f'=====> path of {os.path.join(sub_root, f)} at line {i+1} have empty list, skip')
                            continue
                        data = self._gen_data_filter(data)
                        if data is not None:
                            new_datas.append(data)
                            valid_cnt += 1

                    with open(new_save_path, 'w', encoding='utf8') as wf:
                        for data in new_datas:
                            wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                    loguru.logger.info(f'======> Inner data filter, Save for {str(os.path.join(new_sub_root, f))} over | Raw data cnt = {len(datas)} Fiter left cnt = {valid_cnt}')
                    
        loguru.logger.info(f'=====> Inner gen_data_filter, Build fitler root Over')
        final_root = root + '_final'
        if not os.path.exists(final_root):
            root_clean(newroot, final_root)
        else:
            loguru.logger.info(f'=====> Inner gen_data_filter, Cleaned fitler root exists')
        loguru.logger.info(f'=====> Inner gen_data_filter, Clean fitler root Over')

    def data_generation(self, start_idx=0, end_idx=None, topic_ids=None, class_ids=None, need_topic_tasks=None):
        task_save_path = self.get_task2save_path()

        sel_documents = self.corpus.nodes
        if end_idx is None:
            end_idx = len(sel_documents)

        loguru.logger.info('=========Begin to Generate Data=========')
        i = 0
        for doc in tqdm(sel_documents[:end_idx]):
            if i < start_idx:
                i += 1
                continue
            later_neighbor = self.corpus.get_later_neighbor_nodes(doc)
            loguru.logger.info(f'=======> For {i}-th data generation later_neighbor cnt = {len(later_neighbor)}')
            if len(later_neighbor) > 0:
                gen_datas = self._data_generation(doc, topic_ids, class_ids, need_topic_tasks)

                for data in gen_datas:
                    self.save_gen_task_data(data, task_save_path)
            i += 1
        loguru.logger.info('=========Data Generation Over=========')


if __name__ == '__main__':
    opt = get_options()
    loguru.logger.info(opt)
    
    corpus = Corpus(opt)
    sub_cnt = opt.data_gen_node_cnt
    bg_idx = 0
    loguru.logger.info(f'=====> Ready Previously Load {sub_cnt if sub_cnt is not None else "all"} data bg_idx = {bg_idx}')
    corpus.load_corpus_from_nodes(data_cnt=sub_cnt, bg=0, thread=opt.thread)
    corpus.recover_document_tree()

    sub_cnt = len(corpus.nodes)
    loguru.logger.info(f'=====> Previously Load {sub_cnt} data Over')
    data_generator = DataGenerator(opt, corpus)
    
    for i, task in enumerate(data_generator.topics):
        loguru.logger.info(f'ID: {i} topic: {data_generator.topics[i]}')
    for i, task in enumerate(data_generator.task_names):
        loguru.logger.info(f'ID: {i} task_names: {data_generator.task_names[i]}')
        
    start_idx = opt.data_gen_bg_idx 
    loguru.logger.info(f'=====> Generate data from index: {sub_cnt} data Over')

    if opt.data_gen_type == 'filter':
        data_generator.gen_data_filter()
    else:
        data_generator.data_generation(start_idx=start_idx, end_idx=None)
    
    """
    ID: 0 topic: 金融 - 银行业务 - 零售银行
    ID: 1 topic: 金融 - 银行业务 - 商业银行
    ID: 2 topic: 金融 - 银行业务 - 投资银行
    ID: 3 topic: 金融 - 投资 - 股票市场
    ID: 4 topic: 金融 - 投资 - 债券市场
    ID: 5 topic: 金融 - 投资 - 基金
    ID: 6 topic: 金融 - 投资 - 衍生品市场
    ID: 7 topic: 金融 - 保险 - 人寿保险
    ID: 8 topic: 金融 - 保险 - 财产保险
    ID: 9 topic: 金融 - 保险 - 健康保险
    ID: 10 topic: 金融 - 金融科技 - 区块链
    ID: 11 topic: 金融 - 金融科技 - 人工智能
    ID: 12 topic: 金融 - 金融科技 - 大数据
    ID: 13 topic: 金融 - 监管与合规 - 反洗钱（AML）
    ID: 14 topic: 金融 - 监管与合规 - 合规审计
    ID: 15 topic: 金融 - 监管与合规 - 监管报告

    ID: 0 task_names: 抽取类问答
    ID: 1 task_names: 多跳推理类问答
    ID: 2 task_names: 对比类问答
    ID: 3 task_names: 长答案形式问答
    ID: 4 task_names: 多轮对话能力
    """