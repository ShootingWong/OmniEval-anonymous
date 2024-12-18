import json
import random
import sys
import threading
import time
from typing import List

import loguru
import numpy as np
import requests
import tqdm
from numpy import ndarray
from requests.adapters import HTTPAdapter
from urllib3 import Retry

sys.path.append('..')
from flashrag.generator import BaseGenerator
from utils.api_config import api_config



class APIGenerator(BaseGenerator):

    def __init__(self, config):
        super().__init__(config)
        api_server=api_config[self.model_name]
        loguru.logger.info(f"Model name: {self.model_name}, API server: {api_server}")
        self.api_server = api_server
        self.generator_batch_size = config.get("generator_batch_size", 1)
        self.request_pool=[]
        self.results=[]

    def construct_request(self, request_data):
        return request_data

    def api_call(self, prompt, repetition_penalty=-1, temperature=-1, top_k=-1, top_p=-1,
                      max_new_tokens=-1, do_sample=True, seed=None):
        retry_strategy = Retry(
            total=1,  # 最大重试次数（包括首次请求）
            backoff_factor=1,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
            allowed_methods=["POST"]  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        request_data = {
            'prompt': prompt,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if seed != None:
            request_data['seed'] = seed
        if top_p != -1 and top_p != None:
            request_data["top_p"] = top_p
        if top_k != -1 and top_k != None:
            request_data["top_k"] = top_k
        request_data=self.construct_request(request_data)
        try:
            response = session.post(self.api_server, json=request_data)
            data = json.loads(response.text)
            if "text" in data:
                ans = data["text"][0][len(prompt):].strip()
            elif "generated_text" in data:
                ans = data["generated_text"].strip()
            elif "choices" in data:
                ans = data["choices"][0]["text"].strip()
            else:
                raise Exception(f"Unexpected response: {data}")
        except Exception as e:
            try:
                loguru.logger.error(response.text)
                # loguru.logger.info(str(e))
                ans = response.text
            except:
                ans = str(e)
        return ans

    def get_output(self,idx, model_input, repetition_penalty, temperature, top_k, top_p, max_new_tokens, do_sample, seed):
        while True:
            res = self.api_call(model_input, repetition_penalty,
                temperature, top_k, top_p, max_new_tokens, do_sample, seed)
            if "stream timeout" in res or "Max retries exceeded with url" in res:
                loguru.logger.info(res)
                # loguru.logger.info(f"stream timeout, retrying...")
                time.sleep(random.random() * 5 + 3)
            else:
                self.results[idx] = res
                break

    def generate(self, input_list: list, **params) -> List[str]:
        max_request_len=self.generator_batch_size
        self.results = ["" for _ in range(len(input_list))]

        repetition_penalty=params.get("repetition_penalty", 1.05)
        temperature=params.get("temperature", 0.3)
        top_k=params.get("top_k", 5)
        top_p=params.get("top_p", 0.85)
        max_new_tokens=params.get("max_new_tokens", 1024)
        do_sample=params.get("do_sample", True)
        seed=params.get("seed", 3)

        for i in tqdm.tqdm(range(len(input_list)), desc="VLLM API Generation"):
            thread=threading.Thread(target=self.get_output, args=(i, input_list[i], repetition_penalty,
                temperature, top_k, top_p, max_new_tokens, do_sample, seed))
            self.request_pool.append(thread)
            thread.start()

            if len(self.request_pool)>=max_request_len:
                for thread in self.request_pool:
                    thread.join()
                self.request_pool=[]

        if len(self.request_pool)>0:
            for thread in self.request_pool:
                thread.join()

            self.request_pool=[]

        return self.results

class VLLMAPIGenerator(APIGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.api_server += "/generate"

    def construct_request(self, request_data):
        return request_data

class TGIAPIGenerator(APIGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.api_server += "/generate"

    def construct_request(self, request_data):
        _request_data = {
            "inputs": request_data.pop("prompt"),
            "parameters": {
                "max_new_tokens": request_data.pop("max_tokens"),
                **request_data
            }
        }
        return _request_data

class VLLMOpenAIGenerator(APIGenerator):
    def __init__(self, config):
        super().__init__(config)
        try:
            api = self.api_server + "/v1/models"
            response = requests.get(api)
            self.serve_model_name=response.json()["data"][0]["root"]
        except:
            raise NotImplementedError(f"API server {self.api_server} is not ready!")
        self.api_server += "/v1/completions"

    def construct_request(self, request_data):
        _request_data = {
            "model": self.serve_model_name,
            "prompt": request_data.pop("prompt"),
            "max_tokens": request_data.pop("max_tokens"),
            "temperature": request_data.pop("temperature"),
            "top_p": request_data.pop("top_p"),
            "top_k": request_data.pop("top_k"),
            "repetition_penalty": request_data.pop("repetition_penalty"),
            "seed": request_data.pop("seed"),
        }

        return _request_data


class APIEncoder:
    def __init__(self, retrieval_method):
        self.retrieval_method = retrieval_method
        self.api_server = api_config[retrieval_method]
        self.request_pool = []
        self.results = []
        loguru.logger.info(f"Model name: {self.retrieval_method}, API server: {self.api_server}")

    def get_hidden_size(self) -> int:
        embed=self.encode(["test"])
        return embed.shape[1]

    def api_call(self, prompt):
        retry_strategy = Retry(
            total=1,  # 最大重试次数（包括首次请求）
            backoff_factor=1,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
            allowed_methods=["POST"]  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        request_data = {
            "control": {
                "handle_name": 0,
                "result_type": 0
            },
            "batch_items": [
                {
                    "item_id": 3333,
                    "request_text": prompt,
                }
            ]
        }
        try:
            response = session.post(self.api_server, json=request_data)
            data = json.loads(response.text)
            embedding=data["infer_results"][0]["item_result"]["embeddings"][0]["float_vector_value"]["values"]
        except Exception as e:
            try:
                loguru.logger.error(response.text)
                # loguru.logger.info(str(e))
                embedding = response.text
            except:
                embedding = str(e)
        return embedding

    def get_output(self, query, idx):
        while True:
            res = self.api_call(query)
            if "stream timeout" in res or "Max retries exceeded with url" in res:
                loguru.logger.info(res)
                # loguru.logger.info(f"stream timeout, retrying...")
                time.sleep(random.random() * 5 + 3)
            else:
                self.results[idx] = res
                break

    def encode(self, query_batch: list) -> ndarray:
        self.results=[None for _ in range(len(query_batch))]
        max_request_len=256

        for i in range(len(query_batch)):
            thread=threading.Thread(target=self.get_output, args=(query_batch[i], i))
            self.request_pool.append(thread)
            thread.start()

            if len(self.request_pool) >= max_request_len:
                for thread in self.request_pool:
                    thread.join()

                self.request_pool = []

        if len(self.request_pool) > 0:
            for thread in self.request_pool:
                thread.join()
            self.request_pool = []

        #. convert to numpy array
        res=np.array(self.results, dtype=np.float32)
        # print(res.shape)
        return res



