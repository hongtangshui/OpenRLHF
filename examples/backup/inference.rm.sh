# source /mnt/bn/seedllm3-lixuefeng-2/env/llama_factory/bin/activate
# export LD_LIBRARY_PATH=/mnt/bn/seedllm3-lixuefeng-2/miniconda3/envs/openrlhf/lib/python3.11//site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

MODEL_PATH="/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwen.7b.ins_olympaids23k.rs.1223.1pair_ep1.bs256.lr9e-6"
GENERATE_OUTPUT="/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/eval/samples/analysis/qwen2.5-32b-instruct/ds.distill.v3.2.lr5e-6/checkpoint-630/aime/aime800_qwen-instruct_t1.0_tp1.0_ml16384_k256_s0_e869.jsonl"
RM_OUTPUT="/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/eval/samples/analysis/qwen2.5-32b-instruct/ds.distill.v3.2.lr5e-6/checkpoint-630/aime/rm0102_aime800_qwen-instruct_t1.0_tp1.0_ml16384_k256_s0_e869.jsonl"


echo $GENERATE_OUTPUT
read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference0103
   --eval_task rm \
   --pretrain $MODEL_PATH \
   --bf16 \
   --max_len 16384 \
   --dataset $GENERATE_OUTPUT  \
   --dataset_probs 1.0 \
   --zero_stage 2 \
   --post_processor null \
   --micro_batch_size 2 \
   --flash_attn \
   --output_path $RM_OUTPUT
EOF
echo $get_rewards_commands
deepspeed --module $get_rewards_commands