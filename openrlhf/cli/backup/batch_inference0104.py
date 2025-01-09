import argparse
import os
from datetime import timedelta
import dataset
import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
import time
from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer

import gc
import contextlib
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

import os
import gc
import contextlib
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

class Model:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float,
        gpus: int,
        n_sampling: int,
        max_new_tokens: int = 16384,
        trust_remote_code: bool = True,
        seed: int = 42,
        enable_prefix_caching: bool = False,
        swap_space: int = 64,
        max_num_seqs: int = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        skip_special_tokens: bool = True,
        prompt_max_len: int = None,
        include_stop_str_in_output: bool = False,
        stop_tokens: list = None
    ):
        """
        Initialize the model with extended configuration options.
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to the model weights
            temperature: Sampling temperature
            gpus: Number of GPUs for tensor parallelism
            n_sampling: Number of sequences to generate
            max_new_tokens: Maximum number of tokens to generate
            trust_remote_code: Whether to trust remote code in model repos
            seed: Random seed for reproducibility
            enable_prefix_caching: Whether to enable prefix caching
            swap_space: GPU memory to reserve for swapping
            max_num_seqs: Maximum number of sequences to process in parallel
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for token repetition
            skip_special_tokens: Whether to skip special tokens in output
            prompt_max_len: Maximum length for input prompts
            include_stop_str_in_output: Whether to include stop strings in output
            stop_tokens: List of stop tokens for generation
        """
        self.model_name = model_name
        self.model_path = model_path
        self.gpus = gpus
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.n_sampling = n_sampling
        self.chat_history = None
        self.sampling_params = None
        self.seed = seed
        
        # Store additional sampling parameters
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.prompt_max_len = prompt_max_len
        self.include_stop_str_in_output = include_stop_str_in_output
        self.stop_tokens = stop_tokens if stop_tokens else ["<|im_end|>"]
        
        # Initialize VLLM model with extended parameters
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.gpus,
            trust_remote_code=trust_remote_code,
            seed=self.seed,
            enable_prefix_caching=enable_prefix_caching,
            swap_space=swap_space,
            max_num_seqs=max_num_seqs
        )

    def init_system_message(
        self,
        message: str | list[str],
        role: str = "system",
        batch_size: int = 16,
        seed: int = 0
    ):
        """
        Initialize system message and configure sampling parameters.
        
        Args:
            message: System message(s) to initialize chat
            role: Role for the system message
            batch_size: Size of batches for processing
            seed: Random seed for sampling
        """
        self.seed = seed
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            n=self.n_sampling,
            seed=self.seed,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            skip_special_tokens=self.skip_special_tokens,
            truncate_prompt_tokens=self.prompt_max_len,
            include_stop_str_in_output=self.include_stop_str_in_output,
            stop=self.stop_tokens
        )
        
        # Initialize chat history for batch processing
        self.chat_history = [[] for _ in range(batch_size)]

        if isinstance(message, str):
            for id_, _ in enumerate(self.chat_history):
                self.chat_history[id_].append({"role": role, "content": message})
        else:
            if len(message) != batch_size:
                raise ValueError(f"message length should be {batch_size}, but got {len(message)}")
            for id_, _ in enumerate(self.chat_history):
                self.chat_history[id_].append({"role": role, "content": message[id_]})

    def chat(
        self,
        input_texts: list[str] | str,
        role_user: str = "user",
        role_model: str = "assistant"
    ):
        """
        Generate responses for input texts using batched inference.
        
        Args:
            input_texts: Single input text or list of input texts
            role_user: Role identifier for user messages
            role_model: Role identifier for model responses
            
        Returns:
            Generated responses from the model
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        if len(input_texts) != len(self.chat_history):
            raise ValueError(
                f"input_texts length should be {len(self.chat_history)}, but got {len(input_texts)}"
            )

        for id_, input_text in enumerate(input_texts):
            self.chat_history[id_].append({"role": role_user, "content": input_text})
        
        print(self.sampling_params)
        responses = self.model.chat(
            messages=self.chat_history,
            sampling_params=self.sampling_params,
        )

        return responses

    def terminate(self):
        """Clean up resources and release memory"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.model.llm_engine.model_executor
        del self.model
        
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
            
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        
def batch_generate_vllm(args):
    from vllm import LLM, SamplingParams
    import os
    import fcntl
    import json
        
    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args
    
    # 获取节点信息
    node_rank = int(os.environ.get("PET_NODE_RANK", "0"))
    num_nodes = int(os.environ.get("PET_NNODES", "1"))
    
    print(f"Node {node_rank}/{num_nodes} initializing model")

    if os.path.exists(args.output_path) and node_rank == num_nodes - 1:
        os.remove(args.output_path)
    
    llm = Model(
        model_name="Qwen",         
        model_path=args.pretrain,   
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        enable_prefix_caching=args.enable_prefix_caching, 
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, 
        max_num_seqs=args.max_num_seqs,
        gpus=args.tp_size,
        prompt_max_len=args.prompt_max_len,                   
        n_sampling=args.best_of_n,                
        stop_tokens=["<|im_end|>"], 
        skip_special_tokens=True,
    )
    # # 配置 vLLM
    # llm = LLM(
    #     model=args.pretrain,
    #     tensor_parallel_size=args.tp_size,
    #     trust_remote_code=True,
    #     seed=args.seed,
    #     # max_num_seqs=args.max_num_seqs,
    #     enable_prefix_caching=args.enable_prefix_caching,
    #     swap_space=64
    # )

    # 配置采样参数
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    # sampling_params = SamplingParams(
    #     max_tokens=args.max_new_tokens,
    #     top_p=args.top_p,
    #     temperature=args.temperature,
    #     repetition_penalty=args.repetition_penalty,
    #     skip_special_tokens=True,
    #     truncate_prompt_tokens=args.prompt_max_len,
    #     include_stop_str_in_output=False,
    #     n=args.best_of_n,
    #     stop=["<|im_end|>"]
    # )

    # 加载和切分数据
    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )

    total_samples = len(prompts_data) if args.iter is None else args.rollout_batch_size
    samples_per_node = total_samples // num_nodes
    remainder = total_samples % num_nodes
    
    start_idx = node_rank * samples_per_node + min(node_rank, remainder)
    end_idx = start_idx + samples_per_node + (1 if node_rank < remainder else 0)
    if args.iter is not None:
        start_idx += args.iter * args.rollout_batch_size
        end_idx += args.iter * args.rollout_batch_size
    
    print(f"Node {node_rank} processing samples {start_idx} to {end_idx}")
    
    # 选择数据
    prompts_data = prompts_data.select(range(start_idx, end_idx))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = list(prompts_dataset)

    if args.enable_csft:
        prompts = [p + args.csft_prompt.strip() + " " for p in prompts]
    
    print("##################")
    print(prompts)
    # 为每个batch初始化system message
    llm.init_system_message(
        """Please reason step by step, and put your final answer within \\boxed{}.""", 
        role="system",
        batch_size=len(prompts),
        seed=args.seed+node_rank
    )
    # 生成完成
    completions = llm.chat(prompts)
    
    # 准备结果
    output_dataset = []
    for i in range(len(prompts_data)):
        output_dataset.append({
            "problem": prompts_data[i]["problem"],
            "solution": prompts_data[i].get("solution", ""),
            "generated_responses": [completions[i].outputs[j].text for j in range(len(completions[i].outputs))],
            "answer": prompts_data[i].get("answer", ""),
            "original_index": start_idx + i
        })

    def safe_write_results(file_path, results):
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 读取现有数据并合并新数据
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_data.append(json.loads(line))
        
        # 添加新数据
        existing_data.extend(results)
        
        # 按original_index排序
        existing_data.sort(key=lambda x: x["original_index"])
        
        # 写回文件
        with open(file_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 获取独占锁
            try:
                # 写入jsonl格式
                for item in existing_data:
                    f.write(json.dumps(item) + '\n')
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁

    # 写入结果
    
    safe_write_results(args.output_path, output_dataset)
    print(f"Node {node_rank}: Successfully wrote results to {args.output_path}")
    # 写入结果后
    while True:
        with open(args.output_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 获取共享锁
            try:
                completed = sum(1 for line in f if line.strip())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  
        
        if completed == total_samples:
            break
        time.sleep(10)
    print(f"Node {node_rank}: All nodes have completed their work")
        

def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []

    for prompts in pbar:
        # Conditional SFT inference
        if args.enable_csft:
            for i in range(len(prompts)):
                prompts[i] += args.csft_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=True,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

        dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)

def batch_rm_inference(args):
    import os
    import fcntl
    import json
    import torch.distributed as dist
    from datetime import timedelta
    from tqdm import tqdm
    
    # 获取节点信息
    node_rank = int(os.environ.get("PET_NODE_RANK", "0"))
    num_nodes = int(os.environ.get("PET_NNODES", "1"))
    
    print(f"Node {node_rank}/{num_nodes} initializing model")

    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # configure model
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # 加载完整数据集
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # 展平生成的responses，保留所有原始字段
    flat_dataset = []
    for item in dataset:
        item_without_responses = {k: v for k, v in item.items() if k != 'generated_responses'}
        for response in item['generated_responses']:
            flat_item = item_without_responses.copy()
            flat_item['input'] = item.get('problem', item.get('question', ''))
            flat_item['output'] = response
            flat_dataset.append(flat_item)

    # 按节点划分数据
    total_samples = len(flat_dataset)
    samples_per_node = total_samples // num_nodes
    remainder = total_samples % num_nodes
    
    start_idx = node_rank * samples_per_node + min(node_rank, remainder)
    end_idx = start_idx + samples_per_node + (1 if node_rank < remainder else 0)
    
    print(f"Node {node_rank} processing samples {start_idx} to {end_idx}")
    
    # 选择该节点要处理的数据
    node_dataset = dataset.from_list(flat_dataset[start_idx:end_idx])
    
    dataset = SFTDataset(
        node_dataset, tokenizer, args.max_len, strategy, pretrain_mode=False, input_template=args.input_template
    )
    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        desc="Rewarding",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    # 推理阶段
    rewards_mapping = {}  # 使用字典存储，键为(problem, response)元组

    with torch.no_grad():
        for _, input_ids, attention_masks, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = model(input_ids, attention_masks)
            
            for prompt, response, reward in zip(info["input"], info["output"], rewards):
                rewards_mapping[(prompt, response)] = reward.item()

        # 写入临时文件
        tmp_output_path = args.output_path + f".tmp.{node_rank}.{strategy.get_rank()}"
        with jsonlines.open(tmp_output_path, mode="w") as writer:
            writer.write({
                "mappings": [
                    {
                        "problem": problem,
                        "response": response,
                        "reward": reward
                    }
                    for (problem, response), reward in rewards_mapping.items()
                ]
            })

    dist.barrier()

    # 在最后一个节点的rank 0进行最终处理
    if node_rank == num_nodes - 1 and strategy.is_rank_0():
        # 等待所有临时文件生成完成
        while True:
            all_done = True
            for n in range(num_nodes):
                for r in range(dist.get_world_size()):
                    if not os.path.exists(args.output_path + f".tmp.{n}.{r}"):
                        all_done = False
                        break
            if all_done:
                break
            time.sleep(1)
        
        # 1. 收集所有rewards映射
        all_rewards_mapping = {}  # (problem, response) -> reward
        for n in range(num_nodes):
            for r in range(dist.get_world_size()):
                tmp_file = args.output_path + f".tmp.{n}.{r}"
                if os.path.exists(tmp_file):
                    with jsonlines.open(tmp_file, mode="r") as reader:
                        data = reader.read()
                        for item in data["mappings"]:
                            all_rewards_mapping[(item["problem"], item["response"])] = item["reward"]
                    # os.remove(tmp_file)
                    
        print("ZHY DEBUG")
        print(len(all_rewards_mapping.values()))
        print(type(all_rewards_mapping))
        print(all_rewards_mapping)
        # 2. 读取原始数据并添加rewards
        original_dataset = []
        with jsonlines.open(args.dataset, mode="r") as reader:
            for item in reader:
                problem = item["problem"]
                rewards = []
                for response in item["generated_responses"]:
                    reward = all_rewards_mapping.get((problem, response))
                    if reward is not None:
                        rewards.append(reward)
                    else:
                        strategy.print(f"Warning: Missing reward for problem '{problem}' and response")
                item["rewards_list"] = rewards
                original_dataset.append(item)

        # 3. 计算统计信息
        all_rewards = list(all_rewards_mapping.values())
        rewards_tensor = torch.tensor(all_rewards)
        print(f"Reward mean: {rewards_tensor.mean().item()}, std: {rewards_tensor.std().item()}")

        # 4. 应用后处理（如果需要）
        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            original_dataset = processor(args, original_dataset)

        # 5. 保存最终结果
        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(original_dataset)
            
    # 所有节点等待最终输出文件生成完成
    while not os.path.exists(args.output_path):
        time.sleep(5)
        
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    if args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
