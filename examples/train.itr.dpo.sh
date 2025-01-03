set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

BASE_DIR="/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/data/openrlhf/chckpoint/qwen2.5-32b-instruct/ds.distill.v3.2.lr5e-6/iter.dpo.r5.s32.1227"
mkdir -p $BASE_DIR
ITER_LOG_PATH="${BASE_DIR}/iter.log"

TRAINING_ITERS=2
ROLLOUT_BATCH_SIZE=16

POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-32b-instruct/ds.distill.v3.2.lr5e-6/checkpoint-630
REF_MODEL_PATH=$POLICY_MODEL_PATH

iter=0
if [ -f $ITER_LOG_PATH ]; then
   iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
   echo "Iter: $iter"
   
   # Create iteration-specific directories and paths
   ITER_DIR="$BASE_DIR/iter_${iter}"
   mkdir -p $ITER_DIR
   
   GENERATE_OUTPUT="$ITER_DIR/generate.jsonl"
   RM_OUTPUT="$ITER_DIR/rm.jsonl"
   MODEL_OUTPUT_PATH="$ITER_DIR/checkpoint"
   
   # Use latest model if past first iteration
   if ((iter > 0)); then
      POLICY_MODEL_PATH="$BASE_DIR/iter_$((iter-1))/checkpoint"
   fi

   read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --max_new_tokens 16384 \
   --prompt_max_len 2048 \
   --dataset /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/OpenRLHF/dataset/dpo/aime32.jsonl \
   --input_key problem \
   --apply_chat_template \
   --temperature 1.0 \
   --tp_size 8 \
   --best_of_n 32 \
   --enable_prefix_caching \
   --max_num_seqs 1024 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT
EOF
   echo $generate_commands
   python -m $generate_commands
   checkSuccess "GENERATE"

   read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task rm \
   --pretrain /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwen.7b.ins_olympaids23k.rs.1223.1pair_ep1.bs256.lr9e-6 \
   --bf16 \
   --max_len 16384 \
   --dataset $GENERATE_OUTPUT  \
   --dataset_probs 1.0 \
   --zero_stage 2 \
   --post_processor iter_dpo \
   --micro_batch_size 2 \
   --tp_size 8 \
   --flash_attn \
   --output_path $RM_OUTPUT
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 16384 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --prompt_key problem \
   --apply_chat_template \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain $POLICY_MODEL_PATH \
   --ref_pretrain $REF_MODEL_PATH \
   --save_path $MODEL_OUTPUT_PATH \
   --zero_stage 3 \
   --max_epochs 3 \
   --bf16 \
   --learning_rate 5e-6 \
   --gradient_checkpointing
EOF
   echo $dpo_commands
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   # Create a symbolic link to the latest checkpoint
   ln -sf $MODEL_OUTPUT_PATH "$BASE_DIR/latest_checkpoint"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done