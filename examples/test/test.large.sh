#!/bin/bash

# 14b, longcot

cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/OpenRLHF
git checkout ppo-local-0122
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/env/openrlhf/bin/activate
export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=disable
export WANDB_DIR=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/mwandb

ROLLOUT_BS=16
N_SAMPLES_PER_PROMPT=1
TEMPERATURE=0.7
NUM_EPISODES=1
KL_COEF=0.001
BS=16
EP=1
LR=5e-7
EVAL_STEPS=1

DATASET_NAME=am.36k

TRIAL_NAME=rl.reinforce.14b_test_001

DATA_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/data/training_data/rl_prompt/$DATASET_NAME
POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-14b-instruct/ds.distill.v1.2.lr5e-6/checkpoint-240
SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-14b-instruct/$TRIAL_NAME
SAMPLES_SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/data/output/rl/$TRIAL_NAME

# start rm
if [ "$PET_NODE_RANK" -eq 0 ]; then
    python -m openrlhf.cli.serve_rm \
        --mode rule \
        --data_path $DATA_PATH \
        --port 5000 \
        --host $MASTER_ADDR &
fi

set -x
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265
if [ "$PET_NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    echo "Starting Ray worker node on $(hostname)"
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 10s

# start rl
if [ "$PET_NODE_RANK" -eq 0 ]; then
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/OpenRLHF"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 2 \
    --eval_steps $EVAL_STEPS \
    --save_steps 1 \
    --pretrain $POLICY_MODEL_PATH \
    --remote_rm_url http://$MASTER_ADDR:5000/get_reward \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH \
    --micro_train_batch_size 1 \
    --train_batch_size $BS \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size $ROLLOUT_BS \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $EP \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len 2048 \
    --generate_max_len 14336 \
    --advantage_estimator reinforce \
    --samples_save_path $SAMPLES_SAVE_PATH \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --init_kl_coef $KL_COEF \
    --prompt_data $DATA_PATH \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --flash_attn \
    --save_hf_ckpt \
    --max_ckpt_num 1000 \
    --max_ckpt_mem 2147483647 \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --temperature $TEMPERATURE 
fi