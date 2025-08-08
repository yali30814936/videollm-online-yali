export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 4"
    nnodes=1
fi

${launcher} evaluate.py \
    --live_version live1+ \
    --vision_drop_strategy mod_0.2 \
    --eval_datasets ego4d_refined_narration_stream_val \
    --resume_from_checkpoint outputs/ego4d_fast1+ \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 1 \
    --gradient_checkpointing True \
    --prediction_loss_only False \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --connector_type mlp \
    --finetune_modules connector \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory False \
