#!/bin/bash

# 14b, longcot

cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/OpenRLHF
git checkout ppo-local-0114
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/env/openrlhf/bin/activate
export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=disabled
export WANDB_DIR=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/mwandb

# 启用bash的调试输出
set -x

# 环境变量诊断
echo "=== Environment Diagnostics ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "PET_NODE_RANK: $PET_NODE_RANK"
echo "PET_NNODES: $PET_NNODES"
echo "Current hostname: $(hostname)"
echo "Current IP: $(hostname -I)"
echo "=========================="

# 网络连接测试
echo "=== Network Diagnostics ==="
if [ ! -z "$MASTER_ADDR" ]; then
    echo "Testing connection to MASTER_ADDR..."
    ping -c 3 $MASTER_ADDR || echo "Cannot ping MASTER_ADDR: $MASTER_ADDR"
    echo "Trying to resolve MASTER_ADDR..."
    nslookup $MASTER_ADDR || echo "Cannot resolve MASTER_ADDR: $MASTER_ADDR"
fi
echo "=========================="

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

TRIAL_NAME=rl.reinforce_test_001

DATA_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/data/training_data/rl_prompt/$DATASET_NAME
POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/model/Qwen2.5-7B-Instruct
SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/model/qwen2.5-7b-instruct/$TRIAL_NAME
SAMPLES_SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/wiles/data/output/rl/$TRIAL_NAME

# 创建一个临时文件来标记进程状态
LOCK_FILE="/tmp/rm_server_running.lock"
touch $LOCK_FILE

# 捕获SIGTERM信号，清理锁文件
cleanup() {
    echo "Received cleanup signal"
    rm -f $LOCK_FILE
    exit 0
}
trap cleanup SIGTERM

if [ "$PET_NODE_RANK" -eq "$((PET_NNODES-1))" ]; then
    echo "Starting serve_rm on host: $MASTER_ADDR"
    echo "Current working directory: $(pwd)"
    
    # 使用python -c来测试import
    echo "Testing Python imports..."
    python -c "from openrlhf.cli.serve_rm import main; print('Import successful')" || echo "Import failed"
    
    # 启动serve_rm并将输出重定向到日志文件
    python -m openrlhf.cli.serve_rm \
        --reward_pretrain /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwq.32b_unified.1225.greedy_ep1.bs256.lr5e-6 \
        --mode hybrid \
        --data_path $DATA_PATH \
        --port 5000 \
        --bf16 \
        --flash_attn \
        --max_len 16384 \
        --batch_size 16 \
        --host $MASTER_ADDR 2>&1 | tee /tmp/serve_rm.log &
    
    RM_PID=$!
    echo "serve_rm started with PID: $RM_PID"
    
    # 等待服务启动
    sleep 5
    
    # 检查服务是否正在运行
    echo "Checking if service is running..."
    if ps -p $RM_PID > /dev/null; then
        echo "Service process is running"
        # 测试服务可访问性
        echo "Testing service accessibility..."
        curl -v http://$MASTER_ADDR:5000/health 2>&1 || echo "Cannot connect to service"
    else
        echo "Service process is not running"
        echo "=== serve_rm log contents ==="
        cat /tmp/serve_rm.log
        echo "==========================="
    fi
    
    # 等待直到锁文件被删除
    while [ -f $LOCK_FILE ]; do
        # 检查serve_rm进程是否还在运行
        if ! kill -0 $RM_PID 2>/dev/null; then
            echo "RM服务异常退出"
            echo "=== Final serve_rm log contents ==="
            cat /tmp/serve_rm.log
            echo "==========================="
            rm -f $LOCK_FILE
            exit 1
        fi
        sleep 10
    done
fi

if [ "$PET_NODE_RANK" -lt "$((PET_NNODES-1))" ]; then
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
            --ref_num_gpus_per_node 4 \
            --actor_num_nodes 1 \
            --actor_num_gpus_per_node 4 \
            --vllm_num_engines 4 \
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
            --generate_max_len 4096 \
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
            --disable_ds_ckpt \
            --max_ckpt_num 1000 \
            --max_ckpt_mem 2147483647 \
            --vllm_sync_backend nccl \
            --gradient_checkpointing \
            --temperature $TEMPERATURE

        # Ray任务完成后，删除锁文件
        rm -f $LOCK_FILE
    fi
fi