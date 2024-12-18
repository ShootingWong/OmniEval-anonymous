import os
import copy
import json
import sys
sys.path.append('.')
from corpus_builder.build_corpus import load_nodes, shuffle_file, corpus_file 
from llama_index.core.schema import NodeWithScore, MetadataMode
from multiprocessing import Lock, Pool
import numpy as np 
from tqdm import tqdm
import collections
import pickle
import loguru 

path2doc = None
path2node = None

def get_insert_idx(start_list, start_idx):
    return (np.array(start_list) < start_idx).sum()
    

def recover_sub(nodes):
    global path2doc, path2node
    doc_trees = {}
    node2psg_idx = {}
    for node in tqdm(nodes):
        file_path = node.metadata['file_path']
        node_text = node.text 
        file_docs = path2doc[file_path]
        find = False
        for i, doc in enumerate(file_docs):
            if node_text in doc.text:
                key = file_path + '\t' + str(i)
                node2psg_idx[node.id_] = i
                all_text = doc.text
                find = True
                break

        if not find: 
            loguru.logger.info(f'======> file_path = {file_path} not find node {node_text}')
            continue
            
        if key not in doc_trees:
            start_idxs = all_text.index(node_text)
            doc_trees[key] = {
                'start_idxs': [start_idxs],
                'node_ids': [node.id_]
            }
        else:
            start_idxs = all_text.index(node_text)
            insert_idx = get_insert_idx(doc_trees[key]['start_idxs'], start_idxs)
            doc_trees[key]['start_idxs'].insert(insert_idx, start_idxs)
            doc_trees[key]['node_ids'].insert(insert_idx, node.id_)

    return [doc_trees, node2psg_idx]

class Corpus:
    def __init__(self, opt):
        self.opt = opt
    
    def get_later_neighbor_nodes(self, node):
        file_path = node.metadata['file_path']
        loguru.logger.info(f'======> self.node2psg_idx len = {len(self.node2psg_idx)}')
        psg_idx = self.node2psg_idx[node.id_] if node.id_ in self.node2psg_idx else 'none'
        key = file_path + '\t' + str(psg_idx)
        if key in self.doc_trees:
            tree_info = self.doc_trees[key]
            split_node_ids = tree_info['node_ids']
            self_idx = split_node_ids.index(node.id_)
            later_node_ids = split_node_ids[self_idx+1:]
        else:
            later_node_ids = []
        return [self.node_map[nodeid] for nodeid in later_node_ids]
    
    def recover_document_tree(self):
        global path2doc, path2node
        def merge_doc_trees(doc_trees_list):
            final_tree = {}
            for tree in doc_trees_list:
                for key in tree:
                    if key in final_tree:
                        for i, start_idx in enumerate(tree[key]['start_idxs']):
                            insert_idx = get_insert_idx(final_tree[key]['start_idxs'], start_idx)
                            final_tree[key]['start_idxs'].insert(insert_idx, start_idx)
                            final_tree[key]['node_ids'].insert(insert_idx, tree[key]['node_ids'][i])
                    else:
                        final_tree[key] = tree[key]
            loguru.logger.info(f'=====> Merge doc tree list OVER')
            return final_tree

        self.raw_nodes = self.load_nodes_list(self.opt.rawdoc_root, thread=self.opt.thread)
        print('======> Inner load_corpus_from_nodes Load raw_nodes over')

        self.path2doc = collections.defaultdict(list)
        self.path2node = collections.defaultdict(list)
        loguru.logger.info(f'=====> len(raw_nodes) = {len(self.raw_nodes)}')
        for doc in self.raw_nodes:
            self.path2doc[doc.metadata['file_path']].append(doc)
        for node in self.nodes:
            self.path2node[node.metadata['file_path']].append(node)

        loguru.logger.info(f'=====> self.path2doc = {list(self.path2doc.keys())[:10]}')
        loguru.logger.info(f'=====> Begin to build document tree')
        path2doc = self.path2doc
        path2node = self.path2node
        doc_tree_path = os.path.join(self.opt.rawdoc_root, 'doc_trees.json')
        node2psg_idx_path = os.path.join(self.opt.rawdoc_root, 'node2psg_idx.json')
        if os.path.exists(doc_tree_path):
            doc_trees = json.loads(open(doc_tree_path, 'r', encoding='utf8').read())
            node2psg_idx = pickle.load(open(node2psg_idx_path, 'rb'))
        else:
            input_lists = []
            thread = self.opt.thread
            nodes = self.nodes 
            per_cnt = int(np.ceil(len(nodes) / thread))
            for i, bg in enumerate(range(0, len(nodes), per_cnt)):
                ed = int(min(bg + per_cnt, len(nodes)))
                input_lists.append(nodes[bg:ed])
            pool = Pool(thread)
            loguru.logger.info(f'======> Build doc tree pool over')
            results = pool.map(recover_sub, input_lists)
            loguru.logger.info(f'======> Build doc tree pool map over')
            del pool
            
            doc_trees_list = []
            node2psg_idx = {}
            for res in results:
                doc_trees_list.append(res[0])
                node2psg_idx.update(res[1])
            
            doc_trees = merge_doc_trees(doc_trees_list)
            
            with open(doc_tree_path, 'w', encoding='utf8') as wf:
                wf.write(json.dumps(doc_trees, ensure_ascii=False))
            pickle.dump(node2psg_idx, open(node2psg_idx_path, 'wb'))

        self.doc_trees = doc_trees
        self.node2psg_idx = node2psg_idx

        neighbor_nodes = self.get_later_neighbor_nodes(self.nodes[0])
        loguru.logger.info(f'=====> {self.nodes[0].id_} has {len(neighbor_nodes)} Neighbors')

        loguru.logger.info(f'=====> Build document tree over')

    def load_nodes_list_jsonl(self, node_root, data_cnt=None, bg=0, thread=None):
        datas = [json.loads(line) for line in open(os.path.join(node_root, corpus_file), encoding='utf8').readlines()]
        if shuffle_file in node_root:
            idxs = pickle.load(open(os.path.join(node_root, shuffle_file), 'rb'))
        else:
            idxs = np.arange(len(datas))
        datas = list(np.array(datas)[idxs])

        if data_cnt is None:
            data_cnt = len(datas)
        return datas[bg: bg+data_cnt]

    def load_nodes_list(self, node_root, data_cnt=None, bg=0, thread=1):
        files = [f for f in os.listdir(node_root) if '.pkl' in f]
        if data_cnt is None:
            data_cnt = len(files[bg:])
        if thread >= data_cnt:
            thread = data_cnt
        pool_size = int(np.ceil(data_cnt / thread))
        thread = int(np.ceil(data_cnt / pool_size))
        input_list = []
        loguru.logger.info(f'=====> data_cnt = {data_cnt} thread = {thread} pool_size = {pool_size}')
        sum_sub_cnt = 0
        for i in range(thread):
            sub_bg = i * pool_size + bg
            sub_ed = min((i+1) * pool_size, data_cnt) + bg
            sum_sub_cnt += sub_ed-sub_bg
            input_list.append([node_root, sub_ed-sub_bg, sub_bg])

        with Pool(thread) as p:

            res_list = [res for res in p.imap(load_nodes, input_list)] 
            '''
            for i, input_ in enumerate(input_list):
                loguru.logger.info(f'=====> input_-{i} = {input_}') 
                per_res = p.imap(load_nodes, input_)
                loguru.logger.info(f'=====> Thread-{i} Over')
                res_list.append(per_res)
            '''
        '''
        pool = Pool(thread)
        loguru.logger.info(f'======> Inner load_nodes_list build Pool(size={thread}) sum_sub_cnt = {sum_sub_cnt} over')
        res_list = pool.map(load_nodes, input_list)
        '''
        # pool.close() 
        # loguru.logger.info(f'=====> pool close')
        # pool.join()
        # loguru.logger.info(f'=====> pool join')
        # pool.terminate()
        final_result = sum(res_list, [])
        loguru.logger.info(f'=====> final_result(All node cnt) = {len(final_result)}')
        # del pool
        return final_result
    
    def load_corpus_from_nodes(self, data_cnt=None, bg=0, thread=1):
        loguru.logger.info(f'======> Inner load_corpus_from_nodes cnt = {data_cnt} bg = {bg} thread = {thread}')
        self.nodes = self.load_nodes_list(self.opt.node_root, data_cnt, bg, thread)
        print('======> Inner load_corpus_from_nodes Load nodes over')
        
        for document in self.nodes:
            if 'Title' not in document.metadata:
                file_path = document.metadata['file_path']
                title = str(file_path).split('/')[-1].split('.')[0]
                document.metadata['Title'] = title
            
        self.node_map = {node.id_: node for node in self.nodes}
        loguru.logger.info(f'=====> len of self.node_map = {len(self.node_map)} len of self.nodes = {len(self.nodes)} ')
        # self.recover_document_tree()
        
        # return nodes
    def get_document_content_from_nodes(self):
        self.doc_contents = []
        for node in self.nodes:
            title = node.metadata['Title'] 
            content = node.get_content(metadata_mode=MetadataMode.EMBED)
            self.doc_contents.append(f"Title: {title}\nContent: {content}")
        return self.doc_contents
    
    def load_corpus(self):
        def get_folder(root, root_list):
            root_list.append(root)
            files = os.listdir(root)
            # file_root_lists = []
            data_lists = []
            for f in files:
                path = os.path.join(root, f)
                if os.path.isdir(path):
                    son_datas = get_folder(path, copy.deepcopy(root_list))
                    data_lists += son_datas

                else:
                    document = open(path).read()
                    data = {
                        'root_list': root_list,
                        'filename': path,
                        'text': document
                    }
                    data_lists.append(data)

            return data_lists
        
        root = self.opt.corpus_root
        documents = get_folder(root, [])
        
        self.documents = documents
        
    
    