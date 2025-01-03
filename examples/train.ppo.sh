export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_GDR_LEVEL=4
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=186
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=22 
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0
export RAY_NETWORKING_INTERFACE=bond0
ulimit -n 65536


	
cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/OpenRLHF
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/env/openrlhf/bin/activate
export LD_LIBRARY_PATH=/mnt/bn/seedllm3-lixuefeng-2/miniconda3/envs/openrlhf/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

set -x
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
# ray start
if [ "$PET_NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head  --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 --num-gpus 8
else
    echo "Starting Ray worker node on $(hostname)"
    # worker节点需要一直--block 避免执行完成，pod退出
    ray start --address="$MASTER_ADDR:6379" --num-gpus 8 --block
fi


sleep 30s
# path

if [ "$PET_NODE_RANK" -eq 0 ]; then
echo "submit job now"
RAY_ADDRESS="http://127.0.0.1:8265" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/OpenRLHF"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 4 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --pretrain /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-7b-instruct/ds.distill.v1.2.lr5e-6/checkpoint-216 \
    --reward_pretrain /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwen.7b.ins_olympaids23k.rs.1223.1pair_ep1.bs256.lr9e-6 \
    --save_path /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-7b-instruct/rl_test \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/rl_data/test_aime \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing
fi 