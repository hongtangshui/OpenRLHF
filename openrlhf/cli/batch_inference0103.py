import argparse
import os
from datetime import timedelta
import dataset
import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer


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

    # 配置 vLLM
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        # max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        swap_space=64
    )

    # 配置采样参数
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=True,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=False,
        n=args.best_of_n,
        stop=["<|im_end|>"]
    )

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

    # 生成完成
    completions = llm.generate(prompts, sampling_params)
    
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
            flat_item = item_without_responses.copy()  # 复制所有原始字段
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

    output_dataset = []
    with torch.no_grad():
        for _, input_ids, attention_masks, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = model(input_ids, attention_masks)
            for idx, (prompt, output, reward) in enumerate(zip(info["input"], info["output"], rewards)):
                # 保留所有原始字段，添加reward和original_index
                sample_data = {k: v[idx] if isinstance(v, list) else v for k, v in info.items()}
                sample_data.update({
                    "reward": reward.item(),
                    "original_index": start_idx + len(output_dataset)
                })
                output_dataset.append(sample_data)

            dist.barrier()
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
        
        # 按original_index排序，如果不存在则使用默认值
        existing_data.sort(key=lambda x: x.get("original_index", float('inf')))
        
        # 写回文件
        with open(file_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 获取独占锁
            try:
                # 写入jsonl格式
                for item in existing_data:
                    # 移除original_index字段，因为它只用于排序
                    if "original_index" in item:
                        del item["original_index"]
                    f.write(json.dumps(item) + '\n')
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁

    # 每个rank写入临时文件
    tmp_output_path = args.output_path + f".tmp.{node_rank}.{strategy.get_rank()}"
    with jsonlines.open(tmp_output_path, mode="w") as writer:
        writer.write_all(output_dataset)

    dist.barrier()

    # 只在每个节点的rank 0进程进行合并
    if strategy.is_rank_0():
        # 读取同一节点内的所有rank结果
        node_results = []
        world_size = dist.get_world_size()
        for rank in range(world_size):
            tmp_file = args.output_path + f".tmp.{node_rank}.{rank}"
            if os.path.exists(tmp_file):
                with jsonlines.open(tmp_file, mode="r") as reader:
                    node_results.extend([obj for obj in reader])
                os.remove(tmp_file)

        # 安全写入到最终输出文件
        safe_write_results(args.output_path, node_results)

    dist.barrier()

    # 在最后一个节点的rank 0进行后处理
    if node_rank == num_nodes - 1 and strategy.is_rank_0():
         # 读取 jsonl 文件
        flat_dataset = []
        with jsonlines.open(args.output_path, mode='r') as reader:
            for obj in reader:
                flat_dataset.append(obj)
        
        rewards = torch.tensor([obj["reward"] for obj in flat_dataset])
        print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        # 重组数据结构
        reorganized_dataset = {}
        for item in flat_dataset:
            problem = item["input"]
            if problem not in reorganized_dataset:
                # 创建基础结构，保留所有原始字段
                base_data = {k: v for k, v in item.items() 
                           if k not in ["input", "output", "reward", "original_index"]}
                base_data.update({
                    "problem": problem,
                    "generated_responses": [],
                    "rewards_list": []
                })
                reorganized_dataset[problem] = base_data
            reorganized_dataset[problem]["generated_responses"].append(item["output"])
            reorganized_dataset[problem]["rewards_list"].append(item["reward"])

        # 转换为列表格式
        final_dataset = list(reorganized_dataset.values())

        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            final_dataset = processor(args, final_dataset)

        # 写入最终结果
        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(final_dataset)

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
