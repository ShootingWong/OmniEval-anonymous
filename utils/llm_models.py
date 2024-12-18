import torch
from tqdm import tqdm
from multiprocessing import Lock, Pool
import json
import sys
import os
import requests
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import loguru
from openai import OpenAI 
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def vllm_server_generate_str(url, serve_model_name, pmts, parameters):
    if isinstance(pmts, list):
        input_str = '\n'.join(pmts)
    else:
        input_str = pmts

    parameters = json.loads(parameters)
    data = {
        "prompt":input_str,
        "model": serve_model_name,
    }

    for key in parameters:
        data[key] = parameters[key]

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
    response = session.post(url, json=data)
    data = json.loads(response.text)
    if "text" in data:
        ans = data["text"][0][len(prompt):].strip()
    elif "choices" in data:
        ans = data["choices"][0]["text"].strip()
    else:
        raise Exception(f"Unexpected response: {data}")
    # generated_text = result.json()[0]['generated_text']

    return ans

 
def vllm_server_generate(url, serve_model_name, inputs, pool_size, parameters):
    input_lists = []
    bsz = len(inputs)
    if pool_size > bsz: pool_size = bsz
    per_cnt = bsz // pool_size 
    parameters = json.dumps(parameters)

    for i in range(pool_size):
        bg = i * per_cnt
        ed = (i+1) * per_cnt if i < pool_size-1 else bsz
        input_lists.append([url, serve_model_name, inputs[bg:ed], parameters])

    pool = Pool(pool_size)
    res_list = pool.map(runProcess_vllm_server, input_lists)
    final_list = sum(res_list, [])

    return final_list

def runProcess_vllm_server(inputs):
    url, serve_model_name, strs, parameters = inputs[:]
    res_list = []
    for s in strs:
        res = vllm_server_generate_str(url, serve_model_name, s, parameters)
        res_list.append(res)

    return res_list

class OpenAIApiProxy():
    def __init__(self, openai_api, api_key=None):
        self.client = OpenAI(
            api_key=api_key,
        )
        self.openai_api = openai_api
        self.api_key = api_key
    def call(self, params_gpt):
        response = self.client.chat.completions.create(**params_gpt)
        content = response.choices[0].message.content
        return content

class MyLLM:
    def __init__(self, opt):
        self.opt = opt
        self.infer_batch = opt.infer_batch
        self.model_name = opt.model_name
        if self.opt.vllm:
            print(f'self.model_name = {self.model_name}')
        self.load_llm()

    def load_llm(self):
        current_directory = os.path.dirname(__file__)
        if self.opt.vllm :
            model2path = json.load(open(self.opt.model2path))
            reader_model_path = model2path[self.model_name]

            num_gpus=torch.cuda.device_count()
            
            llm_model = LLM(
                model=reader_model_path,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.5,
                trust_remote_code=True
            )
            sampling_params = SamplingParams(
                temperature=self.opt.temperature, 
            )
            llm_tokenizer = AutoTokenizer.from_pretrained(reader_model_path, trust_remote_code=True)

            self.model = llm_model
            self.reader_tokenizer = llm_tokenizer 
            self.sample_params = sampling_params

            self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token
            print(f'self.reader_tokenizer.eos_token = {self.reader_tokenizer.eos_token}')
            print(f'self.reader_tokenizer.pad_token = {self.reader_tokenizer.pad_token}')
            self.reader_tokenizer.padding_side = "left"
            self.max_input_length = self.opt.max_input_length_llm
            
            print(f'for Model {self.model_name}, self.max_input_length = {self.max_input_length}')

        elif self.opt.gpt:
            self.model = self.opt.gpt_version
            self.reader_tokenizer = None 
            self.sample_params = None
            self.proxy = OpenAIApiProxy(self.opt.openai_api, api_key=self.opt.apikey)
            self.pool_size = self.opt.infer_batch
            loguru.logger.info(f'====> self.opt.gpt = {self.opt.gpt} self.model = {self.model}')
        elif self.opt.vllm_server:
            model2path = json.load(open(self.opt.model2path))
            self.model = None
            self.reader_tokenizer = None

            url = self.opt.openai_api
            api = url + "/v1/models"
            response = requests.get(api)
            self.serve_model_name = response.json()["data"][-1]["id"]
            print(f"=====> response = {response.text} self.serve_model_name = {self.serve_model_name} response.json()['data'][-1] = {response.json()['data'][-1]}")
            # import sys
            # sys.exit(0)
            load_model_name = "qwen2.5-7b" 
            load_model_path = model2path[load_model_name]
            print(f'=====> load_model_path = {load_model_path}')
            self.reader_tokenizer = AutoTokenizer.from_pretrained(load_model_path)
            self.url = url + "/v1/completions"
            loguru.logger.info(f'=====> serve_model_name = {self.serve_model_name} | url = {self.url}')
        
            sample_params = {
                # "repetition_penalty":1.05,
                "temperature":self.opt.temperature,
                "top_k":1,
                # "top_p":0.85,
                # "max_tokens":self.opt.max_new_tokens,
                # "do_sample":False, 
                "seed": 0
            }
            self.sample_params = sample_params
            self.pool_size = self.opt.infer_batch
        else:
            assert False, "Invalid infer_type!"

    def parse_generate(self, batch_input_ids, batch_generate_ids):
        responses = []
        # bsz, max_input = batch_input_ids.size()[:2]
        for i, generated_sequence in enumerate(batch_generate_ids):
            input_ids = batch_input_ids[i]
            text = self.reader_tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    self.reader_tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text.strip())

        return responses

    def get_input_template(self, inputs):
        reader_input_strs = []
        for input_ in inputs:
            messages = [
                {"role": "system", "content": input_[0]},
                {"role": "user", "content": input_[1]}
            ]
            text = self.reader_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            reader_input_strs.append(text)
        
        return reader_input_strs
    
    def get_local_output(self, inputs):
        bsz = len(inputs)
        reader_input_strs = self.get_input_template(inputs)
        out_txt_seqs = []
        
        if self.opt.fastchat:
            reader_inputs = self.reader_tokenizer(reader_input_strs, padding='longest', return_tensors='pt') # 
       
            for i in tqdm(range(0, bsz, self.infer_batch)):
                bg, ed = i, i+self.infer_batch
                max_input_tokens = self.max_input_length - self.opt.max_new_tokens
                reader_input_ids = reader_inputs['input_ids'][bg:ed].to(self.model.device)[:, -max_input_tokens:]
                
                reader_attetion_mask = reader_inputs['attention_mask'][bg:ed].to(self.model.device)[:, -max_input_tokens:]
                
                reader_output = self.model.generate(
                    input_ids=reader_input_ids,
                    attention_mask=reader_attetion_mask,
                    max_new_tokens=self.opt.max_new_tokens,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    do_sample=False, 
                    # output_scores=True,
                    use_cache=True,
                ) 
                
                reader_output = reader_output['sequences']
                per_out_txt_seqs = self.parse_generate(reader_input_ids, reader_output)
                out_txt_seqs += per_out_txt_seqs

        elif self.opt.vllm:
            for i in range(0, bsz, self.infer_batch):
                bg, ed = i, min(i+self.infer_batch, bsz)
            
                prompt_token_ids = self.reader_tokenizer(reader_input_strs[bg: ed], return_tensors='pt', padding=True, truncation=True)['input_ids']
                max_input_tokens = self.max_input_length - self.opt.max_new_tokens
                prompt_token_ids = prompt_token_ids[:, -max_input_tokens:]

                reader_output = self.model.generate(prompt_token_ids=prompt_token_ids.tolist(), sampling_params=self.sample_params)
                
                per_out_txt_seqs = [output.outputs[0].text for output in reader_output]
                out_txt_seqs += per_out_txt_seqs
        return out_txt_seqs
    
    def get_gpt_output(self, inputs):
        prompt = inputs[0]
        prompt_dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content":prompt[0]},
                {"role": "user", "content":prompt[1]}
            ],
            "temperature": 0.01
        }
        resp = self.proxy.call(prompt_dict)

        return resp

    @torch.no_grad()
    def get_llm_output(self, inputs, use_batch=False):
        if not use_batch:
            inputs = [inputs]
            
        bsz = len(inputs)
        if self.opt.vllm:
            out_txt_seqs = self.get_local_output(inputs)
        elif self.opt.gpt:
            out_txt_seqs = self.get_gpt_output(inputs)
            out_txt_seqs = [out_txt_seqs]
        elif self.opt.vllm_server:
            
            inputs = self.get_input_template(inputs)
            out_txt_seqs = vllm_server_generate(self.url, self.serve_model_name, inputs, self.pool_size, self.sample_params)

        return out_txt_seqs
    