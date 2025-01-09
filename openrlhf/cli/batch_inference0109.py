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
from tensorrt_llm import LLM, SamplingParams
import fcntl
import subprocess
import multiprocessing

class DummyStrategy:
    def __init__(self, args):
        self.args = args
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
    
    def is_rank_0(self):
        return True

class NodeConfig:
    def __init__(self, node_rank=0, num_nodes=1):
        self.rank = int(os.environ.get("PET_NODE_RANK", node_rank))
        self.total = int(os.environ.get("PET_NNODES", num_nodes))
        
    def get_node_range(self, total_samples):
        samples_per_node = total_samples // self.total
        remainder = total_samples % self.total
        start = self.rank * samples_per_node + min(self.rank, remainder)
        end = start + samples_per_node + (1 if self.rank < remainder else 0)
        return start, end

class VLLMInstance:
    def __init__(self, args, gpu_ids, seed_offset=0):
        self.args = args
        self.strategy = DummyStrategy(args)
        self.gpu_ids = gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        
        self.tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/model/Qwen2.5-32B-Instruct")
        self.llm = LLM(
            tokenizer=self.tokenizer,
            model=args.pretrain,
            tensor_parallel_size=len(gpu_ids),
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            top_p=getattr(args, 'top_p', 1.0),
            temperature=getattr(args, 'temperature', 1.0),
            n=args.best_of_n,
            seed=args.seed + seed_offset,
        )

    def generate(self, prompts_data, start_idx):
        prompts_dataset = PromptDataset(
            prompts_data, 
            self.tokenizer,
            self.strategy,
            input_template=getattr(self.args, 'input_template', None)
        )
        prompts = list(prompts_dataset)
        
        if getattr(self.args, 'enable_csft', False):
            prompts = [p + self.args.csft_prompt.strip() + " " for p in prompts]
            
        completions = self.llm.generate(prompts, self.sampling_params)
        
        return [
            {
                "problem": prompts_data[i]["problem"],
                "solution": prompts_data[i].get("solution", ""),
                "generated_responses": [
                    completions[i].outputs[j].text 
                    for j in range(len(completions[i].outputs))
                ],
                "answer": prompts_data[i].get("answer", ""),
                "original_index": start_idx + i
            }
            for i in range(len(prompts_data))
        ]

def safe_write_results(file_path, results):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            existing = [json.loads(line) for line in f if line.strip()]
            f.seek(0)
            f.truncate()
            
            all_results = sorted(
                existing + results,
                key=lambda x: x["original_index"]
            )
            for item in all_results:
                f.write(json.dumps(item) + '\n')
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def process_data_chunk(rank, args, start_idx, end_idx, gpu_ids):
    instance = VLLMInstance(args, gpu_ids, seed_offset=rank)
    strategy = DummyStrategy(args)
    
    dataset = blending_datasets(
        args.dataset,
        getattr(args, 'dataset_probs', None),
        strategy,
        args.seed,
        return_eval=False,
        max_count=getattr(args, 'max_samples', None),
        train_split=getattr(args, 'dataset_split', None),
    )
    chunk_data = dataset.select(range(start_idx, end_idx))
    
    results = instance.generate(chunk_data, start_idx)
    safe_write_results(args.output_path, results)
    print(f"Process {rank}: Completed processing samples {start_idx} to {end_idx}")

def batch_generate_vllm(args):
    node_config = NodeConfig()
    strategy = DummyStrategy(args)
    
    parts = args.pretrain.split('/')
    model_name = parts[-3]
    version = parts[-2]
    ckpt_name = parts[-1]
    
    base_path = os.path.join('/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/trtllm')
    checkpoint_out = os.path.join(base_path, 'checkpoint', model_name, version, ckpt_name)
    engine_out = os.path.join(base_path, 'engine', model_name, version, ckpt_name)
    if os.path.exists(engine_out) and os.listdir(engine_out):
        args.pretrain = engine_out
    else:
        os.makedirs(checkpoint_out, exist_ok=True)
        os.makedirs(engine_out, exist_ok=True)

        cpu_count = max(1, multiprocessing.cpu_count() - 3)

        convert_cmd = [
            'python', 'openrlhf/utils/convert_checkpoint.py',
            '--model_dir', args.pretrain,
            '--output_dir', checkpoint_out,
            '--dtype', 'float16',
            '--tp_size', '4',
            '--workers', str(cpu_count)
        ]
        
        process = subprocess.run(convert_cmd)
        if process.returncode != 0:
            print(f"Node Checkpoint conversion failed")
            return None

        build_cmd = [
            'trtllm-build',
            '--checkpoint_dir', checkpoint_out,
            '--output_dir', engine_out,
            '--gemm_plugin', 'auto',
            '--workers', str(cpu_count)
        ]
        
        process = subprocess.run(build_cmd)
        if process.returncode != 0:
            print(f"Engine build failed")
            return None
        
        args.pretrain = engine_out
        
        print(f"Successfully converted checkpoint: {args.pretrain}")
    
    # Initialize or clean output file
    if os.path.exists(args.output_path) and node_config.rank == node_config.total - 1:
        os.remove(args.output_path)
    
    # Load dataset and calculate ranges
    dataset = blending_datasets(
        args.dataset,
        getattr(args, 'dataset_probs', None),
        strategy,
        args.seed,
        return_eval=False,
        max_count=getattr(args, 'max_samples', None),
        train_split=getattr(args, 'dataset_split', None),
    )
    
    total_samples = len(dataset) if args.iter is None else args.rollout_batch_size
    node_start, node_end = node_config.get_node_range(total_samples)
    
    if args.iter is not None:
        offset = args.iter * args.rollout_batch_size
        node_start += offset
        node_end += offset
    
    # Configure multi-instance processing
    num_instances = getattr(args, 'num_instances', 1)
    total_gpus = torch.cuda.device_count()
    gpus_per_instance = total_gpus // num_instances
    
    if gpus_per_instance == 0:
        raise ValueError(f"Not enough GPUs ({total_gpus}) for {num_instances} instances")
    
    # Start instances
    processes = []
    samples_per_instance = (node_end - node_start) // num_instances
    instance_remainder = (node_end - node_start) % num_instances
    
    for i in range(num_instances):
        start_idx = node_start + i * samples_per_instance + min(i, instance_remainder)
        end_idx = start_idx + samples_per_instance + (1 if i < instance_remainder else 0)
        gpu_ids = list(range(i * gpus_per_instance, (i + 1) * gpus_per_instance))
        
        p = mp.Process(
            target=process_data_chunk,
            args=(i, args, start_idx, end_idx, gpu_ids)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    # Wait for all nodes to complete
    if node_config.rank == 0:
        for _ in range(60):  # Wait up to 10 minutes
            with open(args.output_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    completed = sum(1 for line in f if line.strip())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            if completed == total_samples:
                break
                
            time.sleep(10)
            
        print("All nodes have completed their work")
        

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
    
    # Initialize distributed setup
    node_rank = int(os.environ.get("PET_NODE_RANK", "0"))
    num_nodes = int(os.environ.get("PET_NNODES", "1"))
    
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))
    
    print(f"Node {node_rank}/{num_nodes} initializing model")

    # Setup model and tokenizer
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )
    tokenizer = get_tokenizer(
        args.pretrain, 
        model,
        "left",
        strategy,
        use_fast=not args.disable_fast_tokenizer
    )
    model = strategy.prepare(model)
    model.eval()

    # Load and preprocess data
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Flatten dataset
    flat_data = []
    for item in dataset:
        base_item = {k: v for k, v in item.items() if k != 'generated_responses'}
        base_item['input'] = item.get('problem', item.get('question', ''))
        for response in item['generated_responses']:
            flat_item = base_item.copy()
            flat_item['output'] = response
            flat_data.append(flat_item)

    # Split data for current node
    total_samples = len(flat_data)
    samples_per_node = total_samples // num_nodes
    remainder = total_samples % num_nodes
    start_idx = node_rank * samples_per_node + min(node_rank, remainder)
    end_idx = start_idx + samples_per_node + (1 if node_rank < remainder else 0)
    
    print(f"Node {node_rank} processing samples {start_idx} to {end_idx}")
    node_data = dataset.from_list(flat_data[start_idx:end_idx])

    # Create dataloader
    processed_dataset = SFTDataset(
        node_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=False,
        input_template=args.input_template
    )
    dataloader = strategy.setup_dataloader(
        processed_dataset,
        args.micro_batch_size,
        True,
        False,
        processed_dataset.collate_fn,
        drop_last=False
    )

    # Compute rewards
    rewards_mapping = {}
    pbar = tqdm(dataloader, desc="Rewarding", disable=not strategy.is_rank_0())

    with torch.no_grad():
        for _, input_ids, attention_masks, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = model(input_ids, attention_masks)
            
            for prompt, response, reward in zip(info["input"], info["output"], rewards):
                rewards_mapping[(prompt, response)] = reward.item()

    # Write temporary results
    tmp_path = f"{args.output_path}.tmp.{node_rank}.{strategy.get_rank()}"
    with jsonlines.open(tmp_path, mode="w") as writer:
        writer.write({
            "mappings": [
                {"problem": prob, "response": resp, "reward": reward}
                for (prob, resp), reward in rewards_mapping.items()
            ]
        })

    dist.barrier()

    # Merge results on last node
    if node_rank == num_nodes - 1 and strategy.is_rank_0():
        # Wait for all temp files
        all_rewards = {}
        while True:
            all_done = True
            for n in range(num_nodes):
                for r in range(dist.get_world_size()):
                    if not os.path.exists(f"{args.output_path}.tmp.{n}.{r}"):
                        all_done = False
                        break
            if all_done:
                break
            time.sleep(1)

        # Collect all rewards
        for n in range(num_nodes):
            for r in range(dist.get_world_size()):
                tmp_file = f"{args.output_path}.tmp.{n}.{r}"
                if os.path.exists(tmp_file):
                    with jsonlines.open(tmp_file) as reader:
                        for item in reader.read()["mappings"]:
                            all_rewards[(item["problem"], item["response"])] = item["reward"]

        # Process original dataset
        final_dataset = []
        with jsonlines.open(args.dataset) as reader:
            for item in reader:
                problem = item["problem"]
                rewards = []
                for response in item["generated_responses"]:
                    reward = all_rewards.get((problem, response))
                    if reward is not None:
                        rewards.append(reward)
                    else:
                        strategy.print(f"Warning: Missing reward for problem '{problem}' and response")
                item["rewards_list"] = rewards
                final_dataset.append(item)

        # Calculate statistics
        rewards_tensor = torch.tensor(list(all_rewards.values()))
        print(f"Reward stats - mean: {rewards_tensor.mean().item()}, std: {rewards_tensor.std().item()}")

        # Post-processing if needed
        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            final_dataset = processor(args, final_dataset)

        # Save final results
        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(final_dataset)

    # Wait for final output
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
