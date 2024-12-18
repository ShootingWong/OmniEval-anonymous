import json
import datasets
import loguru
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    # loguru.logger.info(f"Loading corpus from {corpus_path}")
    df = pd.read_json(corpus_path, lines=True)
    corpus = datasets.Dataset.from_pandas(df)
    # corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    # print(f'======> max doc_idxs = {max(doc_idxs)} len(corpus) = {len(corpus)}')
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results
