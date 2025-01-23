#!/bin/bash
# wandb_run_name rl.grpo.0122.0334_qwen.7b.v1_am.36k_rbs256.n4.t0.7es1.kl0.000.clip0.5_bs128.ep1.lr5e-7

# Environment Setup
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/env/openrlhf/bin/activate
export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/mwandb

# Git Setup
cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/OpenRLHF
git checkout ppo-local-0122

# Dataset Configuration
DATASET_NAME=am.36k
DATA_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/data/training_data/rl_prompt/$DATASET_NAME

# Model Paths
POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/model/qwen2.5-7b-instruct/policy.7b.v1

# Training Hyperparameters
ROLLOUT_BS=256
N_SAMPLES_PER_PROMPT=4
TEMPERATURE=0.7
NUM_EPISODES=1
KL_COEF=0.000
BS=128
EP=1
LR=5e-7
EVAL_STEPS=1
LR_WARMUP_RATIO=$(python3 -c "print('{:.6f}'.format(10 / (36000.0 * ${N_SAMPLES_PER_PROMPT} / ${BS})))")

# Trial Configuration
TIMESTAMP=$(TZ='UTC-8' date "+%m%d.%H%M")
TRIAL_NAME="rl.grpo.${TIMESTAMP}_qwen.7b.v1_${DATASET_NAME}_rbs${ROLLOUT_BS}.n${N_SAMPLES_PER_PROMPT}.t${TEMPERATURE}es${NUM_EPISODES}.kl${KL_COEF}.clip0.5_bs${BS}.ep${EP}.lr${LR}"

# Output Paths
SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/model/qwen2.5-7b-instruct/$TRIAL_NAME
SAMPLES_SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/data/output/rl/$TRIAL_NAME

# Ray Configuration
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265

# Start Reward Model Server
if [ "$PET_NODE_RANK" -eq 0 ]; then
    python -m openrlhf.cli.serve_rm \
        --mode rule \
        --data_path $DATA_PATH \
        --port 5000 \
        --host $MASTER_ADDR &
fi

# Ray Cluster Setup
set -x
if [ "$PET_NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    echo "Starting Ray worker node on $(hostname)"
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 10s

# Start Training
if [ "$PET_NODE_RANK" -eq 0 ]; then
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/OpenRLHF"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --pretrain $POLICY_MODEL_PATH \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH \
    --remote_rm_url http://$MASTER_ADDR:5000/get_reward \
    --save_hf_ckpt \
    --max_ckpt_num 1000 \
    --max_ckpt_mem 2147483647 \
    --micro_train_batch_size 1 \
    --train_batch_size $BS \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size $ROLLOUT_BS \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $EP \
    --num_episodes $NUM_EPISODES \
    --eval_steps $EVAL_STEPS \
    --save_steps 1 \
    --prompt_max_len 2048 \
    --generate_max_len 14336 \
    --prompt_data $DATA_PATH \
    --input_key context_messages \
    --samples_save_path $SAMPLES_SAVE_PATH \
    --max_samples 100000 \
    --advantage_estimator rloo \
    --actor_learning_rate $LR \
    --init_kl_coef $KL_COEF \
    --eps_clip 0.5 \
    --value_clip 0.2 \
    --l2 0.1 \
    --lr_warmup_ratio $LR_WARMUP_RATIO \
    --lambd 0.95 \
    --gamma 1.0 \
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --vllm_sync_backend nccl \
    --temperature $TEMPERATURE \
    --packing_samples \
    --normalize_reward \
    --apply_chat_template \
    --use_wandb "1badc41f0d258400b42ad079d39f9d58376dabf0" \
    --wandb_project Wiles \
    --wandb_group rl.grpo \
    --wandb_run_name $TRIAL_NAME   
fi