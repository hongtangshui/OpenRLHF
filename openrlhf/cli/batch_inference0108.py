import argparse
import os
from datetime import timedelta
import dataset
import jsonlines
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
import time
from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer
from vllm import LLM, SamplingParams
import os
import fcntl
import json

# Move Empty class to global scope
class DummyStrategy:
    def __init__(self, args):
        self.args = args
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
    
    def is_rank_0(self):
        return True

def safe_write_results(file_path, results):
    """Safely write results to file with locking"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    existing_data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))
    
    existing_data.extend(results)
    existing_data.sort(key=lambda x: x["original_index"])
    
    with open(file_path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            for item in existing_data:
                f.write(json.dumps(item) + '\n')
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def run_vllm_instance(rank, args, start_idx, end_idx, gpu_ids):
    """Run a single vLLM instance"""
    # Configure GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    print(f"Process {rank} using GPUs {gpu_ids}, handling samples {start_idx} to {end_idx}")

    # Create dummy strategy for this process
    dummy_strategy = DummyStrategy(args)

    # Initialize vLLM
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=len(gpu_ids),  # Use all assigned GPUs
        trust_remote_code=True,
        seed=args.seed + rank,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=100000,
        max_num_seqs=args.max_num_seqs,
        swap_space=32,
    )

    # Setup tokenizer and sampling parameters
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=getattr(args, 'top_p', 1.0),
        temperature=getattr(args, 'temperature', 1.0),
        repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
        skip_special_tokens=True,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=False,
        n=args.best_of_n,
        stop=["<|im_end|>"]
    )

    # Load and process data
    prompts_data = blending_datasets(
        args.dataset,
        getattr(args, 'dataset_probs', None),
        dummy_strategy,
        args.seed,
        return_eval=False,
        max_count=getattr(args, 'max_samples', None),
        train_split=getattr(args, 'dataset_split', None),
    )
    
    prompts_data = prompts_data.select(range(start_idx, end_idx))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, 
                                  input_template=getattr(args, 'input_template', None))
    prompts = list(prompts_dataset)

    if getattr(args, 'enable_csft', False):
        prompts = [p + args.csft_prompt.strip() + " " for p in prompts]

    # Generate completions
    completions = llm.generate(prompts, sampling_params)
    
    # Prepare results
    output_dataset = []
    for i in range(len(prompts_data)):
        output_dataset.append({
            "problem": prompts_data[i]["problem"],
            "solution": prompts_data[i].get("solution", ""),
            "generated_responses": [completions[i].outputs[j].text 
                                  for j in range(len(completions[i].outputs))],
            "answer": prompts_data[i].get("answer", ""),
            "original_index": start_idx + i
        })

    # Write results
    safe_write_results(args.output_path, output_dataset)
    print(f"Process {rank}: Successfully wrote results")

def batch_generate_vllm(args):
    """Main function for multi-instance vLLM generation"""
    # Create dummy strategy
    dummy_strategy = DummyStrategy(args)
    
    # Get node information
    node_rank = int(os.environ.get("PET_NODE_RANK", "0"))
    num_nodes = int(os.environ.get("PET_NNODES", "1"))
    
    print(f"Node {node_rank}/{num_nodes} initializing")

    # Clean output file if this is the last node
    if os.path.exists(args.output_path) and node_rank == num_nodes - 1:
        os.remove(args.output_path)
    
    # Load dataset to get total samples
    full_dataset = blending_datasets(
        args.dataset,
        getattr(args, 'dataset_probs', None),
        dummy_strategy,
        args.seed,
        return_eval=False,
        max_count=getattr(args, 'max_samples', None),
        train_split=getattr(args, 'dataset_split', None),
    )

    # Calculate data ranges
    total_samples = len(full_dataset) if args.iter is None else args.rollout_batch_size
    samples_per_node = total_samples // num_nodes
    remainder = total_samples % num_nodes
    
    node_start_idx = node_rank * samples_per_node + min(node_rank, remainder)
    node_end_idx = node_start_idx + samples_per_node + (1 if node_rank < remainder else 0)
    
    if args.iter is not None:
        node_start_idx += args.iter * args.rollout_batch_size
        node_end_idx += args.iter * args.rollout_batch_size

    # Configure multi-instance settings
    num_instances = getattr(args, 'num_instances', 1)  # Default to 1 instance
    total_gpus = torch.cuda.device_count()
    gpus_per_instance = total_gpus // num_instances
    
    if gpus_per_instance == 0:
        raise ValueError(f"Not enough GPUs ({total_gpus}) to divide among {num_instances} instances")
    
    # Calculate samples per instance
    samples_per_instance = (node_end_idx - node_start_idx) // num_instances
    instance_remainder = (node_end_idx - node_start_idx) % num_instances

    # Start multiple processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for i in range(num_instances):
        # Calculate data range for this instance
        start_idx = node_start_idx + i * samples_per_instance + min(i, instance_remainder)
        end_idx = start_idx + samples_per_instance + (1 if i < instance_remainder else 0)
        
        # Assign GPUs to this instance
        gpu_ids = list(range(i * gpus_per_instance, (i + 1) * gpus_per_instance))
        
        p = mp.Process(
            target=run_vllm_instance,
            args=(i, args, start_idx, end_idx, gpu_ids)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Wait for all nodes to complete
    print(f"Node {node_rank}: Waiting for all nodes to complete...")
    for i in range(60):
        with open(args.output_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
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
    parser.add_argument("--num_instances", type=int, default=2)

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
