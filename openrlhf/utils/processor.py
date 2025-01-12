import torch
from tqdm import tqdm


def reward_normalization(objs):
    rewards = [float(obj["reward"]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


# Conditional SFT
# See https://arxiv.org/abs/2308.12050
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def conditional_sft_processor(args, objs):
    if "reward_template" not in args or args.reward_template is None:
        reward_template = DEFAULT_REWARD_PROMPT
    else:
        reward_template = args.reward_template
    assert "{input}" in reward_template
    assert "{reward}" in reward_template

    if args.normalize_reward:
        reward_normalization(objs)

    for obj in tqdm(objs, desc="Conditional SFT process..."):
        input = obj["input"]
        reward = "{:.2f}".format(float(obj["reward"]))
        input = reward_template.replace("{reward}", reward).replace("{input}", input)
        obj["input"] = input

    return objs


# Rejection Sampling
# See https://arxiv.org/abs/2307.09288
def rejection_sampling_processor(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {"output": output, "reward": reward}
        elif reward > out[input]["reward"]:
            out[input]["reward"] = reward
            out[input]["output"] = output

    return [{"input": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]

def iterative_dpo_processor0103(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Iterative DPO process...."):
        problem = obj["problem"]
        responses = obj["generated_responses"]
        rewards = obj["rewards_list"]
        correctness = obj["answers_correctness"]
        
        if problem not in out:
            # 初始化
            out[problem] = {
                "output": responses[0],
                "chosen": None,
                "chosen_reward": float('-inf'),  # 初始化为负无穷
                "rejected": None,
                "rejected_reward": float('inf'),  # 初始化为正无穷
                "has_correct": False,    # 标记是否有正确答案
                "has_incorrect": False,   # 标记是否有错误答案
            }
        
        # 遍历所有response、reward和correctness
        for response, reward, is_correct in zip(responses, rewards, correctness):
            if is_correct:
                # 在正确答案中找最高reward的
                out[problem]["has_correct"] = True
                if reward > out[problem]["chosen_reward"]:
                    out[problem]["chosen_reward"] = reward
                    out[problem]["chosen"] = response
            else:
                # 在错误答案中找最低reward的
                out[problem]["has_incorrect"] = True
                if reward < out[problem]["rejected_reward"]:
                    out[problem]["rejected_reward"] = reward
                    out[problem]["rejected"] = response
        
        # 如果没有正确答案，就用最高reward的作为chosen
        if not out[problem]["has_correct"]:
            max_reward = max(rewards)
            max_idx = rewards.index(max_reward)
            out[problem]["chosen"] = responses[max_idx]
            out[problem]["chosen_reward"] = max_reward
            
        # 如果没有错误答案，就用最低reward的作为rejected
        if not out[problem]["has_incorrect"]:
            min_reward = min(rewards)
            min_idx = rewards.index(min_reward)
            out[problem]["rejected"] = responses[min_idx]
            out[problem]["rejected_reward"] = min_reward

    return [
        {
            "prompt": k,
            "chosen": v["chosen"],
            "chosen_reward": v["chosen_reward"],
            "rejected": v["rejected"],
            "rejected_reward": v["rejected_reward"],
        }
        for k, v in out.items()
    ]

def iterative_dpo_processor(args, objs, num_pairs=1):
    """
    处理 DPO 数据，为每个 problem 选择指定数量的 chosen-rejected pairs
    要求 chosen 和 rejected 都必须包含 \boxed{}
    
    Args:
        args: 参数配置
        objs: 原始数据对象列表
        num_pairs: 每个 problem 需要的 pairs 数量，默认为1
    """
    def has_boxed(response):
        """检查回答是否包含 \boxed{}"""
        return "\\boxed{" in response and "}" in response[response.index("\\boxed{"):]
    
    out = {}
    for obj in tqdm(objs, desc="Iterative DPO process...."):
        problem = obj["problem"]
        responses = obj["generated_responses"]
        rewards = obj["rewards_list"]
        correctness = obj["answers_correctness"]
        
        if problem not in out:
            out[problem] = {
                "pairs": [],
                "has_correct": False,
                "has_incorrect": False,
            }
        
        # 将所有包含 \boxed{} 的回答按照正确与否分类
        correct_responses = []
        incorrect_responses = []
        for response, reward, is_correct in zip(responses, rewards, correctness):
            if not has_boxed(response):  # 跳过不包含 \boxed{} 的回答
                continue
            if is_correct:
                out[problem]["has_correct"] = True
                correct_responses.append((response, reward))
            else:
                out[problem]["has_incorrect"] = True
                incorrect_responses.append((response, reward))
        
        # 按照 reward 排序
        correct_responses.sort(key=lambda x: x[1], reverse=True)
        incorrect_responses.sort(key=lambda x: x[1])
        
        pairs_added = 0
        
        # 如果同时存在正确和错误答案
        if out[problem]["has_correct"] and out[problem]["has_incorrect"]:
            # 优先使用正确答案作为 chosen，错误答案作为 rejected
            for i in range(min(num_pairs, len(correct_responses), len(incorrect_responses))):
                out[problem]["pairs"].append({
                    "chosen": correct_responses[i][0],
                    "chosen_reward": correct_responses[i][1],
                    "rejected": incorrect_responses[i][0],
                    "rejected_reward": incorrect_responses[i][1]
                })
                pairs_added += 1
        
        # 如果还需要更多pairs且还有剩余回答
        if pairs_added < num_pairs:
            # 将所有包含 \boxed{} 的回答按照reward排序
            all_responses = [(r, rw) for r, rw in zip(responses, rewards) if has_boxed(r)]
            all_responses.sort(key=lambda x: x[1], reverse=True)
            
            # 继续添加pairs直到达到要求数量
            for i in range(pairs_added, num_pairs):
                if i + 1 < len(all_responses):  # 确保还有足够的回答可以配对
                    out[problem]["pairs"].append({
                        "chosen": all_responses[i][0],
                        "chosen_reward": all_responses[i][1],
                        "rejected": all_responses[-i-1][0],
                        "rejected_reward": all_responses[-i-1][1]
                    })

    # 转换输出格式
    result = []
    for problem, data in out.items():
        for pair in data["pairs"]:
            result.append({
                "prompt": problem,
                "chosen": pair["chosen"],
                "chosen_reward": pair["chosen_reward"],
                "rejected": pair["rejected"],
                "rejected_reward": pair["rejected_reward"]
            })
    
    return result

PROCESSORS = {
    "rs": rejection_sampling_processor,
    "csft": conditional_sft_processor,
    "iter_dpo": iterative_dpo_processor,
}

def get_processor(name):
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")
