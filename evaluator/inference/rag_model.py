import json
import os.path
import sys
from dataclasses import asdict

import loguru

sys.path.append('.')
from utils.options import get_options
from utils.llm_models import MyLLM
from utils.utils import convert_to_float

from flashrag.config import Config
from flashrag.retriever.index_builder import *
from flashrag.dataset import Dataset

from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from transformers import AutoTokenizer
from flashrag.utils import get_generator, get_retriever
from flashrag.generator import OpenaiGenerator
from typing import List

# corpus_file = 'corpus.old.jsonl'
corpus_file = 'corpus.jsonl'

class OpenaiGenerator_Self(OpenaiGenerator):
    def __init__(self, opt):
        # self.model_name = config["generator_model"]
        # self.batch_size = config["generator_batch_size"]
        # self.generation_params = config["generation_params"]

        # self.openai_setting = config["openai_setting"]
        self.llm = MyLLM(opt)
    
    def process_input(self, input: List[dict]) -> List[str]:
        loguru.logger.info(f'input len = {len(input)} len(input[0]) = {len(input[0])} type input[0] = {type(input[0])} input[0] = {input[0]}')
        sys_input = input[0]['content']
        user_input = input[1]['content']
        return [sys_input, user_input]

    def process_input_list(self, input_list: List) -> List:
        loguru.logger.info(f'input_list len = {len(input_list)} len(input[0]) = {len(input_list[0])} type input_list[0] = {type(input_list[0])}')
        if isinstance(input_list[0], list):
            new_input = [self.process_input(inputs) for inputs in input_list]
        else:
            new_input = input_list
        return new_input

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        response = self.llm.get_llm_output(self.process_input_list(input_list))
        
        return response
    
class RAG:
    pipline_map = {'sequential': SequentialPipeline}
    def __init__(self, opt, pipline):
        self.opt = opt
        index_root = os.path.join(opt.index_path, opt.node_root.split('/')[-1])
        if not os.path.exists(opt.index_path):
            os.mkdir(opt.index_path)
        if not os.path.exists(index_root):
            os.mkdir(index_root)
        self.index_save_dir =  os.path.join(opt.index_path, opt.node_root.split('/')[-1])
        self.final_index_path = os.path.join(self.index_save_dir, f"{opt.retrieval_method}_{opt.faiss_type}.index")
        model2path = json.load(open(self.opt.model2path))
        query_instruction=json.load(open(os.path.join('configs/query_instruction.json')))[opt.retrieval_method]
        opt.retriever_path = model2path[opt.retrieval_method]
        opt.llm_model_path = model2path[opt.generator_model]
        loguru.logger.info(f"opt={opt}")

        if not os.path.exists(self.final_index_path):
            # build index
            self.build_index()
        else:
            loguru.logger.info(f'Index path: {self.final_index_path} exists')
        corpus_path = os.path.join(opt.node_root, corpus_file)
        self.opt.corpus_path = corpus_path


        loguru.logger.info(f'For RAG setting, retriever model = {opt.retrieval_method} retriever_path = {opt.retriever_path} generator_model model = {opt.generator_model} llm_model_path = {opt.llm_model_path}')
        flashrag_config = {
            "data_dir": opt.data_gen_root,
            "index_path": self.final_index_path,
            "corpus_path": opt.corpus_path,
            "model2path": {opt.retrieval_method: opt.retriever_path, opt.generator_model: opt.llm_model_path},
            "generator_model": opt.generator_model,
            "retrieval_method": opt.retrieval_method,
            "use_sentence_transformer": opt.use_sentence_transformer,
            "use_flag_embedding": opt.use_flag_embedding,
            "query_instruction": query_instruction,
            "metrics": ["em", "f1", "acc"], #useless
            "retrieval_topk": opt.retrieval_topk,
            "save_intermediate_data": True,
            "framework": opt.rag_framework,
            # "generator_model": opt.gpt_version if opt.model_name == 'gpt4' else opt.model_name, #gpt model name
            "generator_batch_size": opt.infer_batch, # batch size for generation, invalid for vllm
            "generation_params":{"do_sample": False,"max_tokens": 32,},
            "openai_setting": {"api_key": self.opt.apikey},
            "gpu_memory_utilization": 0.95,
            "retrieval_use_fp16": opt.retrieval_use_fp16,
        }
        os.environ['OPENAI_API_KEY'] = opt.apikey

        self.config_dict = flashrag_config
        self.config = Config(config_dict=flashrag_config)
        loguru.logger.info(f"config={self.config}")
        loguru.logger.info(f'Inner RAG Model Initial, self.opt.corpus_path = {self.opt.corpus_path} self.config.corpus_path = {self.config.corpus_path}')

        all_split = self.get_dataset(self.config)
        self.test_data = all_split #["test"]

        self.pipline = pipline
        self.prompt_template = PromptTemplate(
            self.config, 
            # system_prompt="Answer the question based on the given document. \
            #         Only give me the answer and do not output any other words. \
            #         \nThe following are given documents.\n\n{reference}",
            # user_prompt="Question: {question}\nAnswer:",
            system_prompt=self.opt.rag_system_prompt, #"根据给定的文档回答问题。只给我答案，不要输出任何其他的话。\n以下是给定的文件。\n\n{reference}",
            user_prompt=self.opt.rag_user_prompt, #"问题: {question}\n答案:",
        )
        self.conv_prompt = self.opt.rag_conversation_prompt #"\n历史对话问题记录：{}\n当前问题：{}"
        inputs = self.prompt_template.get_string(question="奥巴马夫人是谁", retrieval_result=[{"contents": "奥巴马\n奥巴马夫人是米歇尔·奥巴马"}])
        loguru.logger.info(f'output of prompt_template = {inputs}')
        if not self.prompt_template.is_openai:
            self.tokenizer = AutoTokenizer.from_pretrained(opt.llm_model_path)
        else:
            self.tokenizer = None
        self.retriever = get_retriever(self.config)
        self.generator = get_generator(self.config) if 'gpt' not in opt.generator_model else OpenaiGenerator_Self(opt) #get_generator(self.config)
        self.generator.tokenizer = self.tokenizer

        self.pipeline = self.pipline_map[pipline](self.config, prompt_template=self.prompt_template, retriever=self.retriever, generator=self.generator, conv_prompt=self.conv_prompt)
        print(f"\nquery instruction: {self.retriever.query_instruction}\n")

    def get_dataset(self, config):
        """Load dataset from config."""
        if config["data_dir"].endswith("/"):
            dir_name=os.path.basename(config["data_dir"][:-1])
        else:
            dir_name = os.path.basename(config["data_dir"])
        dir_name=os.path.basename(dir_name)
        if dir_name in ["v1022"]:
            version = dir_name
            file_path=f"./data_generator/human_annotated/{version}/{version}.jsonl"
            split_dict = {}
            split_dict[version] = Dataset(config, file_path)
        else:
            def parse_dataroot(root, topic_prefix=[]):
                all_tasks = []
                all_paths = []
                paths = os.listdir(root)
                for path in paths:
                    full_path = os.path.join(root, path)
                    if os.path.isdir(full_path):
                        tasks, paths = parse_dataroot(full_path, topic_prefix + [path])
                        all_tasks += tasks
                        all_paths += paths
                    else:
                        task = path.split('.jsonl')[0]
                        all_tasks.append(' - '.join(topic_prefix + [task]))
                        all_paths.append(full_path)
                return all_tasks, all_paths

            all_split, dataset_paths = parse_dataroot(config["data_dir"], [])
            loguru.logger.info(f'all_split = {all_split}')

            split_dict = {split: None for split in all_split}
            for i, split in enumerate(all_split):
                split_path = dataset_paths[i]
                loguru.logger.info(f'Inner load dataset, split = {split}')
                if not os.path.exists(split_path):
                    loguru.logger.info(f"{split} file not exists!")
                    continue
                # if '多轮对话能力' not in split: continue

                loguru.logger.info(f'Load data for {split}')
                if split in ["test", "val", "dev"]:
                    split_dict[split] = Dataset(
                        config, split_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
                    )
                else:
                    split_dict[split] = Dataset(config, split_path)

        return split_dict

    def build_index(self):
        pooling_method = None
        try:
            if "bm25" in self.opt.retrieval_method:
                pooling_method = None
            else:
                #  read pooling method from 1_Pooling/config.json
                pooling_config = json.load(open(os.path.join(self.opt.retriever_path, "1_Pooling/config.json")))
                for k, v in pooling_config.items():
                    if k.startswith("pooling_mode") and v == True:
                        pooling_method = k.split("pooling_mode_")[-1]
                        break
        except:
            raise ValueError(f"Pooling method not found in {self.opt.retriever_path}")

        self.opt.corpus_path = os.path.join(opt.node_root, 'corpus.jsonl')
        embedding_path = os.path.join(self.index_save_dir, f"emb_{self.opt.retrieval_method}.memmap")
        index_builder = Index_Builder(
            retrieval_method=self.opt.retrieval_method,
            model_path=self.opt.retriever_path,
            corpus_path=self.opt.corpus_path,
            save_dir=self.index_save_dir,
            max_length=self.opt.retriever_max_length,
            batch_size=self.opt.per_gpu_embedder_batch_size,
            use_fp16=self.opt.retrieval_use_fp16,
            pooling_method=pooling_method,
            faiss_type=self.opt.faiss_type,
            embedding_path=embedding_path if os.path.exists(embedding_path) else None,
            save_embedding=self.opt.save_embedding,
            faiss_gpu=self.opt.faiss_gpu,
            use_sentence_transformer=self.opt.use_sentence_transformer,
            use_flag_embedding=self.opt.use_flag_embedding,
        )
        index_builder.build_index()
        
    def mkdir(self, root_list):
        for i in range(len(root_list)):
            root_path = os.path.join(*root_list[:i+1])
            if not os.path.exists(root_path):
                os.mkdir(root_path)
        
    def run(self, save=True):
        
        all_output = {}
        for i, task in enumerate(self.test_data):
            loguru.logger.info(f'Ready for run {task} data cnt = {len(self.test_data[task])}')
            test_data = self.test_data[task]

            if test_data is None: continue
            if len(self.test_data[task]) == 0:
                loguru.logger.info(f'{task} is empty, skip...')
                continue
            loguru.logger.info(f'self.opt.close_domain = {self.opt.close_domain}')
            if save:
                data_root_name = self.opt.data_gen_root.split('/')[-1]
                if self.opt.close_domain:
                    model_name = 'CLOSE-' + self.opt.generator_model
                else:
                    model_name = self.opt.retrieval_method + f'_TOP{self.opt.retrieval_topk}' + '-' + self.opt.generator_model
                
                self.mkdir([self.opt.pred_result_root, data_root_name, model_name])
                save_path = os.path.join(self.opt.pred_result_root, data_root_name, model_name, f'{task}.pred.jsonl')
                if os.path.exists(save_path):
                    loguru.logger.info(f'{save_path} have been evaluated, skip it...')
                    continue
            if self.opt.close_domain:
                output_dataset = self.pipeline.naive_run(test_data, do_eval=False)
            else:
                output_dataset = self.pipeline.run(test_data, do_eval=False)
                
            loguru.logger.info(f"---generation for task {task} output Over---")
            if save:
                loguru.logger.info(f'Save pred data for task {task} to {save_path}')
                with open(save_path, 'w', encoding='utf8') as wf:
                    for item in output_dataset.data:
                        item_valid = convert_to_float(item)
                        wf.write(json.dumps(item_valid, ensure_ascii=False) + '\n')
                    
                loguru.logger.info(f'---Save pred data for task {task} Over---')
            all_output[task] = output_dataset
            # if i == 0: break
        return all_output

if __name__ == '__main__':
    opt = get_options()
    # opt.node_root = 'datas/corpus/nodes_dir/few_shot_test-2048-256_nodes'
    opt.node_root = opt.node_root # 'datas/corpus/nodes_dir/few_shot_test-2048-256_nodes'
    # opt.corpus_path = os.path.join(opt.node_root, 'corpus.jsonl')
    # datas = [json.loads(line) for line in open(opt.corpus_path).readlines()]
    # lens = [len(data['contents']) for data in datas]
    # loguru.logger.info(f'max corpus lens = {max(lens)}')
    # opt.data_gen_root = opt.data_gen_root #'./data_generator/gen_datas_small'
    # opt.generator_model = opt.generator_model# "llama3-8B-instruct" #"gpt-4-1106-preview" 
    # opt.rag_framework = opt.rag_framework #"vllm"


    pipline = opt.pipline #'sequential'
    rag = RAG(opt, pipline)
    # opt.save_pred_result = True
    results = rag.run(save=opt.save_pred_result)
    for key in results:
        loguru.logger.info(f'{key} result = {results[key]} results[key] = {results[key][0]}')
    # loguru.logger.info(f'results = {results}')
    import pickle
    pickle.dump(results, open('test_rag_result.pkl', 'wb'))