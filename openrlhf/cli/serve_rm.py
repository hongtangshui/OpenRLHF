import argparse
import re
import json
import jsonlines
from datasets import load_from_disk
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from math_verify import parse, verify

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.check.qwen_equal import math_equal
from multiprocessing import Pool
from transformers import AutoTokenizer
logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


                
                
class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


# class RuleBasedRMProxy:
#     def __init__(self, args):
#         self.args=args
#         self.prompt2answer={}
        
#         dataset = load_from_disk(args.data_path)
#         train_list = list(dataset["train"])
#         validation_list = list(dataset["test"])
        
#         for line in train_list:
#             self.prompt2answer[line['context'].strip()]=line['answer']
#         for line in validation_list:
#             self.prompt2answer[line['context'].strip()]=line['answer']
            
#         self.tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path)
            
#     def correctness_score(self, qa_pair):
#         prompt, response, _=qa_pair
#         matches = re.findall(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})", response)
#         if len(matches)==0:
#             return 0.0
#         else:
#             pred=matches[-1][:-1]
#         if prompt not in self.prompt2answer: 
#             return 0.0
#         if self.prompt2answer[prompt].strip()==pred.strip():
#             return 1.0
#         else:
#             return 0.1
        
    
#     def split_and_tokenize(self, query):
#         splitted=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")
#         prompt, response=splitted[0], splitted[1]
#         encoded_response=self.tokenizer.encode(response)
#         print("encoded_response:", encoded_response)
#         return (prompt.strip(), response.strip(), encoded_response)

#     def score(self, qa_pair):
#         # qa_pair=(prompt, response, encoded_response)
#         prompt, response, encoded_response=qa_pair
#         # too long penalty
#         if f"boxed" not in response and len(encoded_response)>self.args.max_gen_len-100: 
#             return -1 
#         return self.correctness_score(qa_pair)
    
#     def get_reward(self, queries):
#         batch_size=len(queries)
#         scores=[]
#         qa_pairs=[]
#         # split
#         with Pool(processes=batch_size) as p:
#             splitted=p.map(self.split_and_tokenize, queries)
#         print("rm splitted:", splitted)
#         with Pool(processes=batch_size) as p:
#             scores=p.map(self.score, splitted)
#         print("scores:", scores)
#         return scores

def math_equal(gold, answer):
    # gold=parse(gold)
    # answer=parse(answer)
    return gold.strip()==answer.strip()

def math_equal2(gold, answer):
    try:
        gold=parse(gold)
        answer=parse(answer)
        return verify(gold, answer)
    except:
        return False

# class RuleBasedRMProxy:
#     def __init__(self, args):
#         self.args=args
#         self.prompt2answer={}
        
#         dataset = load_from_disk(args.data_path)
#         train_list = list(dataset["train"])
#         validation_list = list(dataset["test"])
        
#         for line in train_list:
#             self.prompt2answer[line['context'].strip()]=line['answer']
#         for line in validation_list:
#             self.prompt2answer[line['context'].strip()]=line['answer']
            
#         self.tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path)
            
#     def correctness_score(self, qa_pair):
#         prompt, response=qa_pair
#         matches = re.findall(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})", response)
#         if len(matches)==0:
#             return -1
#         else:
#             pred=matches[-1][:-1]
#         if prompt not in self.prompt2answer: 
#             return -1
#         if math_equal(self.prompt2answer[prompt], pred):
#             return 1
#         else:
#             return -0.5
        
    
#     def split_and_tokenize(self, query):
#         splitted=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")
#         prompt, response=splitted[0], splitted[1]
#         encoded_response=self.tokenizer.encode(response)
#         print("encoded_response:", encoded_response)
#         return (prompt.strip(), response.strip(), encoded_response)

#     def score(self, qa_pair):
#         # qa_pair=(prompt, response, encoded_response)
#         prompt, response, encoded_response=qa_pair
#         # too long penalty
#         if f"boxed" not in response and len(encoded_response)>self.args.max_gen_len-100: 
#             return -1 
#         return self.correctness_score(qa_pair)
    
#     def get_reward(self, queries):
#         batch_size=len(queries)
#         scores=[]
#         qa_pairs=[]
#         responses=[]
#         for query in queries:
#             splitted=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")
#             prompt, response=splitted[0], splitted[1]
#             qa_pairs.append((prompt, response))    
#             responses.append(response)
#         lengths=[len(ids) for ids in self.tokenizer.batch_encode_plus(responses, padding=False, truncation=False)["input_ids"]]
#         for qa_pair, length in zip(qa_pairs, lengths):
#             prompt, response=qa_pair
#             if f"boxed" not in response and length>self.args.max_gen_len-100: 
#                 scores.append(-1) 
#             else:
#                 scores.append(self.correctness_score(qa_pair))
#         return scores


class RuleBasedRMProxy:
    def __init__(self, args):
        self.args=args
        self.prompt2answer={}
        
        dataset = load_from_disk(args.data_path)
        train_list = list(dataset["train"])
        validation_list = list(dataset["test"])
        
        for line in train_list:
            self.prompt2answer[line['context'].strip()]=line['answer']
        for line in validation_list:
            self.prompt2answer[line['context'].strip()]=line['answer']
            
            
    def correctness_score(self, qa_pair):
        prompt, response=qa_pair
        matches = re.findall(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})", response)
        if len(matches)==0:
            return -1
        else:
            pred=matches[-1][:-1]
        if prompt not in self.prompt2answer: 
            return -1
        if math_equal2(self.prompt2answer[prompt], pred):
            return 1
        else:
            return -0.5
    
    # def split_and_tokenize(self, query):
    #     splitted=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")
    #     prompt, response=splitted[0], splitted[1]
    #     encoded_response=self.tokenizer.encode(response)
    #     print("encoded_response:", encoded_response)
    #     return (prompt.strip(), response.strip(), encoded_response)

    # def score(self, qa_pair):
    #     # qa_pair=(prompt, response, encoded_response)
    #     prompt, response, encoded_response=qa_pair
    #     # too long penalty
    #     if f"boxed" not in response and len(encoded_response)>self.args.max_gen_len-100: 
    #         return -1 
    #     return self.correctness_score(qa_pair)
    
    def get_reward(self, queries):
        batch_size=len(queries)
        scores=[]
        qa_pairs=[]
        responses=[]
        for query in queries:
            splitted=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")
            prompt, response=splitted[0].strip(), splitted[1].strip()
            qa_pairs.append((prompt, response))    
        for qa_pair in qa_pairs:
            prompt, response=qa_pair
            scores.append(float(self.correctness_score(qa_pair)))
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rule")
    # RuleBasedRM Parameters
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_gen_len", type=int)
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)    # for 
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if args.mode=="model":
        reward_model = RewardModelProxy(args)
    else:
        reward_model = RuleBasedRMProxy(args)
    
    # test_case="<im_start>\nsystem\nnihao<|im_end|>\n<|im_start|>user\n1+1<|im_end|>\n<|im_start|>assistant\n1+1=\\boxed{2}<im_end>"
    # reward=reward_model.get_reward([test_case for _ in range(4)])
    # print(reward)
    # exit()
    
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        client_host = request.client.host
        logger.info(f"client_ip: {client_host}")
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
