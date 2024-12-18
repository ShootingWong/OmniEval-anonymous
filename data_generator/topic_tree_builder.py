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
from data_generator.prompt import *
import copy
from llama_index.core.schema import NodeWithScore, MetadataMode
import numpy as np

class TopicTreeBuilder:
    def __init__(self, opt):
        self.opt = opt
        self.llm = MyLLM(opt)
        
        self.all_leaf_tasks = None
        self.lock = Lock()

        self.system = topic_tree_system
        self.user = topic_tree_user
    
    def show_task_tree(self, task_tree, step):
        pre_fix = '\t' * step
        print(pre_fix + task_tree['topic_name'])
        for tree in task_tree['sub_topics']:
            self.show_task_tree(tree, step+1)
            
    def build_task_tree(self, domain_name, refresh=False):
        save_path = self.opt.topic_tree_path

        if os.path.exists(save_path) and not refresh:
            json_output = json.loads(open(save_path, 'r').read())
            return json_output
        
        user_input = self.user.format(domain_name)
        # print(f'=========Inner task_instruction_generation user_input = {user_input}')
        output = self.llm.get_llm_output([self.system, user_input], use_batch=False)[0]
        json_output = json_repair.loads(output)

        print(json_output)
        with open(save_path, 'w') as wf:
            wf.write(json.dumps(json_output, ensure_ascii=False))

        return json_output
    
if __name__ == '__main__':
    opt = get_options()
    opt.topic_tree_path = 'configs/topic_tree2.json'
    print(opt)
    builder = TopicTreeBuilder(opt)
    task_tree = builder.build_task_tree('金融')
    builder.show_task_tree(task_tree, 0)