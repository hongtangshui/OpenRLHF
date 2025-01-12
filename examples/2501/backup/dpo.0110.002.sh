set -x
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

TRAINING_ITERS=3
ROLLOUT_BATCH_SIZE=288
BEST_OF_N=16
TOP=1
EPOCH=5
TRAIN_BATCH_SIZE=64
LEARNING_RATE=5e-6

TRIAL_NAME="dpo_qwen.32b.ins_aime800.$(date +%m%d)_iter$TRAINING_ITERS.rbs$ROLLOUT_BATCH_SIZE.s$BEST_OF_N.t$TOP.ep$EPOCH.bs$TRAIN_BATCH_SIZE.lr$LEARNING_RATE"
BASE_DIR="/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/data/openrlhf/chckpoint-2501/$TRIAL_NAME"

POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-32b-instruct/ds.distill.v3.2.lr5e-6/checkpoint-630

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

clean_gpu_cache() {
    if python -c "import torch" &>/dev/null; then
        python -c "import torch; torch.cuda.empty_cache()"
    fi
}

mkdir -p $BASE_DIR
ITER_LOG_PATH="${BASE_DIR}/iter.log"

# 计算每个epoch的steps数
NUM_STEPS_PER_EPOCH=$(( (ROLLOUT_BATCH_SIZE + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))

REF_MODEL_PATH=$POLICY_MODEL_PATH

iter=0
if [ -f $ITER_LOG_PATH ]; then
   iter=$(cat $ITER_LOG_PATH)
fi

# Clear BASE_DIR if iter is 0
if [ "$iter" -eq 0 ] && [ -n "$BASE_DIR" ] && [ "$BASE_DIR" != "/" ]; then
    rm -rf "${BASE_DIR:?}"/*
fi

while (($iter < $TRAINING_ITERS)); do
   echo "Iter: $iter"
   
   # Create iteration-specific directories and paths
   ITER_DIR="$BASE_DIR/iter_${iter}"
   mkdir -p $ITER_DIR
   
   GENERATE_OUTPUT="$ITER_DIR/data/generate.jsonl"
   RM_OUTPUT="$ITER_DIR/data/rm.jsonl"
   MODEL_OUTPUT_PATH="$ITER_DIR"
   
   # Use latest model if past first iteration
   if ((iter > 0)); then
      POLICY_MODEL_PATH="$BASE_DIR/iter_$((iter-1))"
   fi

   read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference0109
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --max_new_tokens 16384 \
   --prompt_max_len 2048 \
   --dataset /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/data/openrlhf/dataset/dpo/aime800.jsonl \
   --input_key problem \
   --temperature 0.6 \
   --apply_chat_template \
   --tp_size 8 \
   --best_of_n $BEST_OF_N \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT.tmp
EOF
   echo $generate_commands
   python -m $generate_commands
   checkSuccess "GENERATE"

   clean_gpu_cache

   if [ "${PET_NODE_RANK}" = "0" ]; then
      read -r -d '' eval_commands <<EOF
openrlhf.cli.math_eval 
   --input_file $GENERATE_OUTPUT.tmp \
   --output_file $GENERATE_OUTPUT
EOF
      echo $eval_commands
      python -m $eval_commands
      checkSuccess "EVAL"
   else
      # Wait for node 0 to complete evaluation
      while [ ! -f "$GENERATE_OUTPUT" ]; do
         sleep 10
      done
   fi

   read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference0109
   --eval_task rm \
   --pretrain /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/analysis/download/xuefengli/rm_7b_beta0/qwen.7b.ins_olympaids23k.rs.1223.1pair_ep1.bs256.lr9e-6 \
   --bf16 \
   --max_len 16384 \
   --dataset $GENERATE_OUTPUT  \
   --dataset_probs 1.0 \
   --zero_stage 2 \
   --post_processor iter_dpo \
   --micro_batch_size 4 \
   --tp_size 8 \
   --flash_attn \
   --output_path $RM_OUTPUT
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   clean_gpu_cache

   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 16384 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --prompt_key prompt \
   --apply_chat_template \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --micro_train_batch_size 1 \
   --pretrain $POLICY_MODEL_PATH \
   --ref_pretrain $REF_MODEL_PATH \
   --save_path $MODEL_OUTPUT_PATH \
   --ckpt_path $MODEL_OUTPUT_PATH \
   --zero_stage 3 \
   --max_epochs $EPOCH \
   --bf16 \
   --save_steps $NUM_STEPS_PER_EPOCH \
   --learning_rate 5e-6 \
   --adam_offload \
   --ref_offload \
   --flash_attn \
   --packing_samples \
   --iter $iter \
   --ring_attn_size 1 \
   --ring_head_stride 1 \
   --gradient_checkpointing \
   --use_wandb 1badc41f0d258400b42ad079d39f9d58376dabf0 \
   --wandb_project Wiles \
   --wandb_group dpo \
   --wandb_run_name $TRIAL_NAME"_iter"$iter


EOF
   echo $dpo_commands
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   clean_gpu_cache

   # Create a symbolic link to the latest checkpoint
   ln -sf $MODEL_OUTPUT_PATH "$BASE_DIR/latest_checkpoint"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done