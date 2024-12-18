import os
import copy
import json
import pickle
import numpy as np
from tqdm import tqdm

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core import QueryBundle, Document
# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import (
    BaseNode,
)

from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.readers.base import BaseReader
from typing import (
    List, 
    Union, 
    Optional, 
    Iterable, 
    Any,
    Sequence,
    Dict,
    Callable
)

from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)

from llama_index.core.utils import get_tqdm_iterable
from pathlib import Path
import sys
sys.path.append('.')
from utils import get_options
import loguru

shuffle_file = 'shuffle_index.datapkl'
corpus_file = 'corpus.jsonl'

class JsonlReader(BaseReader):
    
    def lazy_load_data(self, file_name: Path, extra_info: Dict = None, file_metadata: Callable[[str], Dict] = None, filename_as_id: bool = False, encoding='utf8', **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
    
        bsz = 1
        loguru.logger.info(f'Inner JsonlReader file_metadata = {file_metadata}')
        if file_metadata is not None:
            metadata = file_metadata(str(file_name))
        if extra_info is not None:
            metadata = extra_info

        title = str(file_name).split('.jsonl')[0].split('/')[-1]
        datas = [json.loads(line) for line in open(file_name, encoding=encoding).readlines()]
        documents = []
        for i in range(0, len(datas)):
            
            psg_text = json.dumps(datas[i], ensure_ascii=False) if 'json_dataset' in str(file_name) else datas[i]['text'] 
            metadata['Title']= title
            document = Document(
                text=psg_text, 
                metadata=metadata,
            )
            documents.append(document)
        if filename_as_id:
            for i, doc in enumerate(documents):
                doc.id_ = f"{file_name!s}_part_{i}"
        return documents

def flatten_folder(root):
    files = os.listdir(root)
    data_lists = []
    for f in files:
        path = os.path.join(root, f)
        if os.path.isdir(path):
            son_datas = flatten_folder(path)
            data_lists += son_datas

        else:
            data_lists.append(path)

    return data_lists

def process_for_documents(documents):
    for document in documents:
        if 'Title' not in document.metadata:
            file_path = document.metadata['file_path']
            title = str(file_path).split('/')[-1].split('.')[0]
            document.metadata['Title'] = title

        document.excluded_llm_metadata_keys.extend(['file_path', 'id'])
        document.metadata_seperator="::"
        document.metadata_template="{key}=>{value}"
        document.text_template="Metadata: {metadata_str}\n-----\n{content}"
        document.text_template="{content}"

    return documents

def load_documents(transcript_directory):
    file_extractor = {'.jsonl': JsonlReader()} 
    path_list = flatten_folder(transcript_directory)
    loguru.logger.info(f'======> path_list = {path_list}')
    documents = SimpleDirectoryReader(input_files=path_list, file_extractor=file_extractor).load_data(show_progress=True, num_workers=100)
    valid_documents = []
    empty_path = []
    empty_cnt = 0
    for doc in documents:
        if doc.get_content() == '':
            empty_cnt += 1
            empty_path.append(doc.metadata['file_path'])
            continue
        valid_documents.append(doc)
    valid_documents = process_for_documents(valid_documents)
    loguru.logger.info(f'valid_documents[0] = {valid_documents[0]}\n Len of valid_documents = {len(valid_documents)} empty_path = {empty_path} empty_cnt = {empty_cnt}')
    # import sys
    # sys.exit(0)
    return valid_documents

def load_nodes(input_list):
    nodes_path = input_list[0]
    if len(input_list) > 1:
        cnt = input_list[1]
    else:
        cnt = None
    if len(input_list) > 2:
        bg = input_list[2]
    else:
        bg = None

    files = [f for f in os.listdir(nodes_path) if '.pkl' in f]
    loguru.logger.info(f'--------Load Nodes from {nodes_path} All cnt = {len(files)}------')
    nodes = []
    if cnt is None:
        cnt = len(files)
        bg = 0
        
    if shuffle_file not in files:
        loguru.logger.info(f'======> Inner load_nodes shuffle_file not exists , use original one')
        # idxs = np.arange(len(files))
        # np.random.shuffle(idxs)
        # pickle.dump(idxs, open(os.path.join(nodes_path, shuffle_file), 'wb'))
        idxs = np.arange(len(files))
    else:
        loguru.logger.info(f'======> Inner load_nodes shuffle_file exists , load now')
        idxs = pickle.load(open(os.path.join(nodes_path, shuffle_file), 'rb'))
    files = list(np.array(files)[idxs])
    for f in tqdm(files[bg:cnt+bg]):
        if f.endswith('.pkl'):
            nodes.append(pickle.load(open(os.path.join(nodes_path, f), 'rb')))
    loguru.logger.info(f'--------Load Nodes Over Load cnt = {cnt}------')
    return nodes

def save_nodes(nodes, nodes_path):
    os.mkdir(nodes_path)
    loguru.logger.info(f'------Save nodes to {nodes_path}------')
    i = 0
    
    json_path = os.path.join(nodes_path, corpus_file)
    with open(json_path, 'w', encoding='utf8') as wf:
        for node in tqdm(nodes):
            pickle.dump(node, open(os.path.join(nodes_path, f'{i}.pkl'), 'wb'))
            i += 1

            title = node.metadata['Title']
            content = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = node.metadata
            new_data = {
                'id': node.id_,
                'title': title,
                'contents': content,
                # 'metadata': metadata,
            }
            
            wf.write(json.dumps(new_data, ensure_ascii=False) + '\n')

    idx = np.arange(len(nodes))
    np.random.shuffle(idx)
    pickle.dump(idx, open(os.path.join(nodes_path, shuffle_file), 'wb'))

    loguru.logger.info(f'------Save json data of nodes in {nodes_path} OVER------')


def check_short_doc(doc, DEFAULT_CHUNK_SIZE):
    text = doc.get_content(metadata_mode=MetadataMode.NONE)
    if '.jsonl' in doc.metadata['file_path']:
        if '基金' in doc.metadata['file_path']:
            if len(text) > DEFAULT_CHUNK_SIZE: return None
            return True
        if len(text) <= DEFAULT_CHUNK_SIZE:
            return True
    else:
        if len(text) <= DEFAULT_CHUNK_SIZE:
            return True
    return False

def corpus_builder(transcript_directory, rawdoc_paths, nodes_path, DEFAULT_CHUNK_SIZE, SENTENCE_CHUNK_OVERLAP):
    loguru.logger.info(f'Inner corpus_builder DEFAULT_CHUNK_SIZE = {DEFAULT_CHUNK_SIZE}')
    service_context = None
    documents = load_documents(transcript_directory)
    if not os.path.exists(rawdoc_paths):
        save_nodes(documents, rawdoc_paths)
    if os.path.exists(nodes_path):
        nodes = load_nodes([nodes_path])
    else:
        splitter = TokenTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE,chunk_overlap=SENTENCE_CHUNK_OVERLAP)

        long_documents = []
        short_documents = []
        for doc in documents:
            check = check_short_doc(doc, DEFAULT_CHUNK_SIZE)
            if check:
                short_documents.append(doc)
            elif check is None:
                continue
            else:
                long_documents.append(doc)
  
        nodes = splitter.get_nodes_from_documents(long_documents, show_progress=True, num_workers=128) + short_documents
        node_lens = [len(node.get_content(metadata_mode=MetadataMode.NONE)) for node in nodes]
        loguru.logger.info(f'=========> short_documents cnt = {len(short_documents)} node cnt = {len(nodes)} max(node_lens) = {max(node_lens)}')
        empty_cnt = 0
        empty_path = []
        for node in nodes:
            if node.get_content(metadata_mode=MetadataMode.EMBED) == "" :
                empty_cnt += 1
                empty_path.append(node.metadata['file_path'])

        loguru.logger.info(f'empty_path = {empty_path} empty_cnt = {empty_cnt}')
        save_nodes(nodes, nodes_path)
    return nodes


if __name__ == '__main__':
    opt = get_options()
    transcript_directory = opt.transcript_directory
    rawdoc_paths = opt.rawdoc_root
    nodes_path = opt.node_root 
    DEFAULT_CHUNK_SIZE = opt.DEFAULT_CHUNK_SIZE 
    SENTENCE_CHUNK_OVERLAP = opt.SENTENCE_CHUNK_OVERLAP 

    nodes = corpus_builder(transcript_directory, rawdoc_paths, nodes_path, DEFAULT_CHUNK_SIZE, SENTENCE_CHUNK_OVERLAP)
    loguru.logger.info(f'======> All nodes cnt = {len(nodes)}')
   