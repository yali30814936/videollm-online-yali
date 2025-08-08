export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 4"
    nnodes=1
fi

${launcher} train.py --deepspeed configs/deepspeed/zero2.json \
    --live_version live1+ \
    --vision_drop_strategy mod_0.2 \
    --train_datasets ego4d_refined_narration_stream_train \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy steps \
    --save_steps 1000 \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --connector_type mlp \
    --finetune_modules connector \
    --report_to tensorboard \
    --output_dir outputs/ego4d_fast1+ \
