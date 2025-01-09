
cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/OpenRLHF
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/env/openrlhf/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/data



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


sleep 30s
# path
POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-14b-instruct/ds.distill.v1.2.lr5e-6/checkpoint-240
REWARD_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwen.7b.ins_olympaids23k.rs.1223.1pair_ep1.bs256.lr9e-6
SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-7b-instruct/rl_test
if [ "$PET_NODE_RANK" -eq 0 ]; then
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/OpenRLHF"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 4 \
    --pretrain $POLICY_MODEL_PATH \
    --reward_pretrain $REWARD_MODEL_PATH \
    --save_path $SAVE_PATH \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 16384 \
    --advantage_estimator reinforce \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --prompt_data /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/rl_data/test_aime \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --adam_offload \
    --packing_samples \
    --normalize_reward \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing
fi