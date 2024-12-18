# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import faiss
import os
import time
from collections import defaultdict
from tqdm import tqdm
import gzip
import json
import math
import copy
import time
import sys

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModel
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer
sys.path.append('.')
from utils.options import get_options
from utils.llm_models import MyLLM
from utils.corpus_loader import Corpus
from FlagEmbedding import BGEM3FlagModel
from data_generator.generator import DataGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

logger = logging.getLogger(__name__)
device = torch.device('cuda:0')

class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, encoded_corpus, index_factory, metric):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")
        
        index = faiss.index_factory(encoded_corpus.shape[1], index_factory, metric)
        
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        logger.info("training index...")
        index.train(encoded_corpus)
        logger.info("adding embeddings...")
        index.add(encoded_corpus)
        self.index = index
        return index

    def load(self, index_path):
        logger.info(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index
        return index
    
    def save(self, index_path):
        logger.info(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        per_cnt = 5000
        all_cnt = query.shape[0]
        all_D, all_I = [], []
        for i in range(0, all_cnt, per_cnt):
            bg = i
            ed = min(i + per_cnt, all_cnt)
            D, I = self.index.search(query[bg:ed], k=hits)
            all_D += D.tolist()
            all_I += I.tolist()
        print(f'np.array(all_D) shape = {np.array(all_D).shape} np.array(all_I) shape = {np.array(all_I).shape}')
        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int32)#self.index.search(query, k=hits)


class DenseRetriever:
    def __init__(self, opt, device, corpus=None):
        self.opt = opt
        self.k = opt.retrieval_topk
        self.device = device

        # Load document corpuse
        if corpus is None:
            self.corpus = Corpus(opt)
            self.corpus.load_corpus_from_nodes()
        else:
            self.corpus = corpus
        self.doc_contents = self.corpus.get_document_content_from_nodes()
        print('Load doc_contents Over len of doc_contents = ', len(self.doc_contents))

        # Load dense retriever 
        model_path = opt.retriever_path
        self.retriever = BGEM3FlagModel(model_path, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        # self.retriever = self.retriever.cuda()
        # self.retriever.eval() ##!!!
        print(f'Init and Load retriever form {model_path} Over')

    def get_embeddings(self, sentences, save_path=None, is_passages=False, gpu=True):
        psg_cnt = len(sentences)
        d = self.retriever.model.model.config.hidden_size
        
        batch_cnt = math.ceil(psg_cnt / self.opt.per_gpu_embedder_batch_size)
        save_batch = 1000 if save_path is not None else len(sentences) + 1 
        sub_idx = 0
        use_sub = False
        sub_cnt = save_batch * self.opt.per_gpu_embedder_batch_size
        if save_path is not None and batch_cnt > save_batch:
            use_sub = True
            # sub_vec = torch.zeros(sub_cnt, d).cuda()
            sub_vec = np.zeros([sub_cnt, d])
            cpu_vec_list = []
        else:
            vec = np.zeros([psg_cnt, d]) #torch.zeros(psg_cnt, d)
        # if gpu:
            # vec = vec.cuda()
        for i in tqdm(range(batch_cnt)):
            
            if save_path is not None and i > 0 and i % save_batch == 0:
                sub_vec_cpu = sub_vec.cpu().clone()
                cpu_vec_list.append(sub_vec_cpu)
                torch.save(sub_vec_cpu, open(save_path+'-{}'.format(sub_idx), 'wb'))
                sub_idx += 1
                sub_vec = sub_vec * 0
                if i == batch_cnt - 1:
                    have_saved_cnt = sub_idx * save_batch * self.opt.per_gpu_embedder_batch_size
                    left_cnt = psg_cnt - have_saved_cnt
                    sub_vec = sub_vec[:left_cnt]

            bg = i * self.opt.per_gpu_embedder_batch_size
            ed = (i+1) * self.opt.per_gpu_embedder_batch_size if i < batch_cnt - 1 else psg_cnt

            batch = sentences[bg: ed]
            with torch.no_grad():
                embeddings = self.retriever.encode(
                    batch, 
                    batch_size=batch_cnt, 
                    max_length=4096, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                )['dense_vecs']

            if use_sub:
                sbg = bg % sub_cnt
                sed = ed % sub_cnt
                if sed == 0:
                    sed = sub_cnt
                print('bg = {} ed = {} sbg = {} sed = {}'.format(bg, ed, sbg, sed))
                sub_vec[sbg:sed] = embeddings
            else:
                vec[bg:ed, :] = embeddings
            
        if not use_sub:
            # vec = vec.cpu()
            vec = torch.Tensor(vec)
            if save_path is not None:
                torch.save(vec, open(save_path, 'wb'))
            return np.asarray(vec)
        else:
            if save_path is not None:
                have_saved_cnt = sub_idx * save_batch * self.opt.per_gpu_embedder_batch_size
                if have_saved_cnt < psg_cnt:
                    sub_vec_cpu = sub_vec.cpu().clone()
                    cpu_vec_list.append(sub_vec_cpu)
                    torch.save(sub_vec_cpu, open(save_path+'-{}'.format(sub_idx), 'wb'))
                    sub_idx += 1
            
            cpu_vec_list = torch.cat(cpu_vec_list, dim=0)
            return np.asarray(cpu_vec_list)
        
    def build_index(self):
        node_root = self.opt.node_root.split('/')[-1]
        save_root = os.path.join(opt.index_path, '{}'.format(node_root))
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        time1 = time.time()
        self.index = FaissIndex(self.device)
        
        doc_vec_path = os.path.join(save_root, 'doc_embs.pth')
        index_path = os.path.join(save_root, f"{opt.retrieval_method}_{opt.faiss_type}.index")

        print('=========> Begin to build index')
        if os.path.exists(index_path):
            t1 = time.time()
            self.index.load(index_path)
            t2 = time.time()
            print('=========> Load Index OVer, COST ', t2-t1)
        else:
            if os.path.exists(doc_vec_path):
                print('=========> Load Doc Emb from path')
                doc_vec = np.asarray(torch.load(doc_vec_path))
                print('=========> Load Doc Emb Over')
            elif os.path.exists(doc_vec_path+'-0'):
                # all_cnt = 65
                datas = [torch.load(doc_vec_path+'-{}'.format(i)) for i in range(all_cnt)]

                doc_vec = torch.cat(datas, dim=0)
                print('=========> concat doc_vec size = {} passage len = {}'.format(doc_vec.size(), len(self.doc_contents)))
                doc_vec = doc_vec[:len(self.doc_contents)]
                print('=========> new doc_vec size = ', doc_vec.size())
                torch.save(doc_vec, open(doc_vec_path, 'wb'))
                doc_vec = np.asarray(doc_vec)
            else:
                print('=========> Generate Doc Emb')
                # doc_vec = get_embeddings(doc_contents, doc_vec_path, opt, is_passages=True, gpu=False)
                doc_vec = self.get_embeddings(self.doc_contents, doc_vec_path)
                print('=========> Generate Doc Emb Over')

            time2 = time.time()
            print('=========> Load embeddings Cost ', time2-time1)

            self.index.build(doc_vec, "Flat", "ip")
            time23 = time.time()
            self.index.save(index_path)

            time33 = time.time()
            print('=========> Build Index OVER cost {} Save Index Cost {} | Begin to Retrieve'.format(time23-time2, time33-time23))
        
    def save_retrieval_for_task(self, task, D, I, datas):
        print(f'=========> Begin to Save Retrieval result for {task}')
        task2path = json.load(open(self.opt.task_data_path, 'r', encoding='utf8'))
        task_path = task2path[task]
        save_path = task_path.split('/')[-1].split('.jsonl')[1] + '_DRres.jsonl'
        
        docs_withid = np.array([{'id': doc.id_} for doc in self.corpus.nodes])
        with open(save_path, 'w', encoding='utf8') as wf:
            # for data in tqdm(datas):
            for i in tqdm(range(I.shape[0])):
                index_list = list(I[i])
                
                sel_passages = list(docs_withid[index_list])
                for idx in range(len(index_list)):
                    sel_passages[idx]['score'] = str(D[i][idx])
                datas['dr_results'] = sel_passages

                wf.write(json.dumps(datas, ensure_ascii=False) + '\n')
        print(f'=========> Save Retrieval result for {task} OVER')

    def get_task_datas(self, task):
        
        task2path = json.load(open(self.opt.task_data_path, 'r', encoding='utf8'))
        task_path = task2path[task]
        datas = [json.loads(line) for line in open(task_path, 'r', encoding='utf8').readlines()]

        return datas

    def search_for_task(self, task, save=True):
        print(f'=========> Begin to Retrieval for {task}')
        time1 = time.time()
        datas = self.get_task_datas(task)
        querys = [data['question'] for data in datas]
        
        save_root = self.opt.save_corpus_embedding_root
        query_vec_path = os.path.join(save_root, '{}_query_embs.pth'.format(task))

        print('=========> query_vec_path = ', query_vec_path)

        if os.path.exists(query_vec_path):
            print('=========> Load Query Emb from path')
            query_vec = np.asarray(torch.load(query_vec_path))
            print('=========> Load Query Emb Over')
        else:
            print('=========> Generate Query Emb')
            query_vec = self.get_embeddings(querys, query_vec_path, is_passages=False, gpu=True)
            print('=========> Generate Query Emb Over')

        D, I = self.index.search(query_vec, self.k)
        
        time2 = time.time()

        print(f'=========> Retrieval for {task} OVER cost {time2-time1}s | Begin to Save')

        if save:
            self.save_retrieval_for_task(task, D, I, datas)

        return D, I
    
    def search(self, querys):
        print(f'=========> Begin to Retrieval ')
        time1 = time()
        query_vec = self.get_embeddings(querys, query_vec_path=None, is_passages=False, gpu=True)

        D, I = self.index.search(query_vec, self.k)
        time2 = time()
        print(f'=========> Retrieval OVER cost {time2-time1}s')

        return D, I

    

if __name__ == "__main__":
    print(f'======> Inner faiss_retriever')
    opt = get_options()
    torch.manual_seed(opt.seed)
    # slurm.init_distributed_mode(opt)

    
    
    opt.node_root = 'datas/corpus/nodes_dir/few_shot_test-2048-256_nodes'
    print(f'======> opt.node_root = {opt.node_root}')
    corpus = Corpus(opt)
    corpus.load_corpus_from_nodes()
    
    datagen = DataGenerator(opt, corpus=corpus)
    datagen.task_instruction_generation()
    opt.task_data_path = datagen.get_task2save_path()
    dense_retriever = DenseRetriever(opt, device, corpus=corpus)
    dense_retriever.build_index()
    # datagen.read_task_tree()
    # task = datagen.all_leaf_tasks[0]['name']
    # print(f'========> datagen.all_leaf_tasks = {datagen.all_leaf_tasks} task = {task}')
    # dense_retriever.search_for_task(task, save=True)