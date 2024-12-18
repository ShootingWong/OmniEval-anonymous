# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        self.add_rag_params()
        self.add_build_corpus_params()
        self.add_judgement_params()

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--model_name", type=str, default="llama2-7b-chat", help="name of the llm, used for load local llm "
        )
        self.parser.add_argument( 
            "--gpt_version", default="gpt-4-1106-preview", type=str, help="name of the gpt-version, used for openai api "
        )
        self.parser.add_argument(
            "--log_path", type=str, default="", help="output path"
        )
        self.parser.add_argument( 
            "--max_input_length_llm", default=4096, type=int,
        )
        self.parser.add_argument( 
            "--max_topic_depth", default=2, type=int,
        )
        
        self.parser.add_argument(
            "--openai_api",
            type=str, 
            default="",
            help="",
        )
        
        self.parser.add_argument(
            "--apikey",
            type=str, 
            default="",
            help="",
        )
        self.parser.add_argument(
            "--infer_batch",
            type=int,
            default=2,
            help="",
        )

        self.parser.add_argument(
            "--infer_type",
            type=str,
            default="gpt",
            choices=["vllm", "gpt", "vllm_server"],
            help='',
        )
        
        self.parser.add_argument(
            "--vllm",
            action="store_true",
            help='uses vllm to load LLM',
        )
        self.parser.add_argument(
            "--gpt",
            action="store_true",
            help='uses vllm to load LLM',
        )
        self.parser.add_argument(
            "--vllm_server",
            action="store_true",
            help='uses vllm_server to load LLM',
        )
        self.parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=1.05,
            help="repetition_penalty for generation",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.01,
            help="temperature for generation",
        )
        self.parser.add_argument(
            "--top_k",
            type=int,
            default=5,
            help="top_k for generation",
        )
        self.parser.add_argument(
            "--top_p",
            type=float,
            default=0.85,
            help="top_p for generation",
        )
        self.parser.add_argument(
            "--max_new_tokens",
            type=int,
            default=1000,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--do_sample",
            type=bool,
            default=True,
            help="whether to sample",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=1815,
            help="seed for generation",
        )
        self.parser.add_argument(
            "--retry_times",
            type=int, 
            default=3,
            help="",
        )

        self.parser.add_argument(
            "--data_gen_root",
            type=str, 
            default='./data_generator/gen_datas',
            help="path of generated_datas",
        )
        self.parser.add_argument(
            "--data_gen_suffix",
            type=str, 
            default='',
            help="suffix of generated datas' path",
        )

        self.parser.add_argument(
            "--topic_tree_path",
            type=str, 
            default='configs/topic_tree.json',
            help="path of topic tree json",
        )

        self.parser.add_argument(
            "--data_gen_bg_idx",
            type=int, 
            default=0,
            help="Generation may interrupt, this parameter is used to set a new generation should start from which node.",
        )
        self.parser.add_argument(
            "--data_gen_node_cnt",
            type=int, 
            default=None,
            help="",
        )

        self.parser.add_argument(
            "--data_gen_type",
            type=str, 
            default='normal',
            choices=["rejection", 'normal', "filter"],
            help="this parameter is used to control what types of data to be generated.",
        )
        self.parser.add_argument(
            "--model2path",
            type=str, 
            default='configs/model2path.json',
            help="this parameter is used to set the model path of LLMs/Retrievers.",
        )
        self.parser.add_argument(
            "--thread",
            type=int, 
            default=1,
            help="",
        )
        
    def add_build_corpus_params(self):
        self.parser.add_argument(
            "--transcript_directory", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--node_root",
            type=str, 
            default="",
            help="",
        )
        self.parser.add_argument(
            "--rawdoc_root",
            type=str, 
            default="",
            help="",
        )
        self.parser.add_argument(
            "--DEFAULT_CHUNK_SIZE",
            type=int, 
            default=2048,
            help="",
        )
        self.parser.add_argument(
            "--SENTENCE_CHUNK_OVERLAP",
            type=int, 
            default=256,
            help="",
        )
        
         
    def add_rag_params(self):
        # basic parameters
        self.parser.add_argument(
            "--index_path", type=str, default="datas/corpus/flash_rag_index", help=""
        )
        self.parser.add_argument(
            "--retrieval_topk", type=int, default=5, help=" "
        )
        self.parser.add_argument(
            "--per_gpu_embedder_batch_size", type=int, default=32, help=""
        )
        
        self.parser.add_argument(
            "--pred_result_root", type=str, default="evaluator/pred_results", help=""
        )
        self.parser.add_argument(
            "--generator_model", type=str, default="llama3-8B-instruct", help=""
        )
        
        self.parser.add_argument(
            "--retrieval_method", type=str, default="bge-m3", help=""
        )
        
        self.parser.add_argument(
            "--retriever_path", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--query_instruction", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--retriever_max_length", type=int, default=4096, help=""
        )
        self.parser.add_argument(
            "--use_fp16", type=str2bool, default=True, help=""
        )
        self.parser.add_argument(
            "--retrieval_use_fp16", type=str2bool, default=True, help=""
        )
        self.parser.add_argument(
            "--faiss_type", type=str, default='Flat', help=""
        )
        self.parser.add_argument(
            "--save_embedding", type=bool, default=True, help=""
        )
        self.parser.add_argument(
            "--embedding_path", type=str, default='', help=""
        )
        
        self.parser.add_argument(
            "--faiss_gpu", action='store_true', help=""
        )
        self.parser.add_argument(
            "--use_sentence_transformer", action='store_true', help=""
        )
        self.parser.add_argument(
            "--use_flag_embedding", action='store_true', help=""
        )
        self.parser.add_argument(
            "--pooling_method", type=str, default=None, help=""
        )
        self.parser.add_argument(
            "--rag_framework", type=str, default="openai", help=""
        )
        self.parser.add_argument(
            "--save_pred_result", action='store_true', help=""
        )
        self.parser.add_argument(
            "--pipline", type=str, default="sequential", help="used for flashrag to set the rag pipeline"
        )
        self.parser.add_argument(
            "--rag_system_prompt", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--rag_user_prompt", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--rag_conversation_prompt", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--close_book", action='store_true', help="whether use retriever"
        )
        self.parser.add_argument(
            "--generate_batch_size", type=int, default=1, help=""
        )
    def add_judgement_params(self):
        # basic parameters
        self.parser.add_argument(
            "--judge_type", type=str, default="all", choices=["all", "rule", "model"], help=""
        )
        self.parser.add_argument(
            "--eval_batch_size", type=int, default=1, help=""
        )
        self.parser.add_argument(
            "--eval_suffix", type=str, default="", help=""
        )
        self.parser.add_argument(
            "--model_eval_metrics", type=str, default="accuracy,completeness,hallucination,utilization,numerical_accuracy", help=""
        )
        # # "accuracy": [],
        #     # "completeness": [],
        #     "hallucination": [],
        #     # "utilization": [],
        #     # "numerical_accuracy": [],
        
        
    def parse(self):
        opt = self.parser.parse_args()
        
        return opt



    
def trans_infer_type(opt):
    
    opt.vllm = False
    opt.gpt = False
    opt.vllm_server = False
    if opt.infer_type == 'vllm':
        opt.vllm = True
    elif opt.infer_type == 'gpt':
        opt.gpt = True
    elif opt.infer_type == 'vllm_server':
        opt.vllm_server = True

    return opt

def get_options():
    options = Options()
    opt = options.parse()
    opt = trans_infer_type(opt)
    return opt

if __name__ == '__main__':
    opt = get_options()
    print(opt)