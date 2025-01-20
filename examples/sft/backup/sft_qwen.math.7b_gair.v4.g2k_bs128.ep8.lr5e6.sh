set -x
sudo chown -R tiger /mnt/bn/seedllm3-lixuefeng-2/code/o1/OpenRLHF
sudo chown -R tiger /mnt/bn/seedllm3-lixuefeng-2/code/o1/data/
sudo chown -R tiger /mnt/bn/seedllm3-lixuefeng-3/ckpts

master_addr=${master_addr:=$ARNOLD_WORKER_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
nproc_per_node="${nproc_per_node:=$ARNOLD_WORKER_GPU}"
nnodes="${nnodes:=$ARNOLD_WORKER_NUM}"
node_rank="${node_rank:=$ARNOLD_ID}"


cd /mnt/bn/seedllm3-lixuefeng-2/code/o1/OpenRLHF
set -x

export WANDB_MODE=offline
export WANDB_DIR=/mnt/bn/seedllm3-lixuefeng-2/code/o1/data/wandb
source /mnt/bn/seedllm3-lixuefeng-2/miniconda3/bin/activate
conda activate openrlhf
export LD_LIBRARY_PATH=/mnt/bn/seedllm3-lixuefeng-2/miniconda3/envs/openrlhf/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH


DATA_NAME=gair.v4.g2k
BS=128
EP=8
LR=5e-6
LR_SCHEDULER=cosine_with_min_lr

TRIAL_NAME=sft_qwen.math.7b_${DATA_NAME}_bs${BS}.ep${EP}.lr${LR}


MODEL_PATH=/mnt/bn/seedllm3-lixuefeng-3/ckpts/Qwen/Qwen2.5-Math-7B
SAVE_PATH=/mnt/bn/seedllm3-lixuefeng-3/ckpts/openrlhf/sft/$TRIAL_NAME/
DATA_PATH=/mnt/bn/seedllm3-lixuefeng-2/code/o1/data/sft_data/training/${DATA_NAME}.jsonl

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 16384 \
   --dataset $DATA_PATH \
   --input_key prompt \
   --output_key response \
   --train_batch_size $BS \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --max_samples 50000000 \
   --pretrain $MODEL_PATH \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH \
   --disable_ds_ckpt \
   --max_ckpt_num 100 \
   --save_hf_ckpt \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $EP \
   --bf16 \
   --flash_attn \
   --learning_rate $LR \
   --lr_scheduler $LR_SCHEDULER \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb 1badc41f0d258400b42ad079d39f9d58376dabf0 \
   --wandb_project Wiles \
   --wandb_group sft \
   --wandb_run_name $TRIAL_NAME 
EOF

torchrun --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank \
    --master_addr $master_addr --master_port $master_port -m ${training_commands}
