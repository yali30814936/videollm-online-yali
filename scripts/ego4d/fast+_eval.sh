export TOKENIZERS_PARALLELISM=false
export EGO4D_NARR_TOPN=100
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 1"
    nnodes=1
fi

${launcher} evaluate.py \
    --live_version live1+ \
    --eval_datasets ego4d_refined_narration_stream_val \
    --resume_from_checkpoint outputs/ego4d_fast1+_mod0.4/checkpoint-6436 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 1 \
    --gradient_checkpointing True \
    --prediction_loss_only False \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --connector_type mlp \
    --finetune_modules connector \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory False \
    --use_conversation_eval True \
    --use_emllm False \

    # --vision_drop_strategy mod_0.2 \
    # --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
# --live_version live1+ --vision_drop_strategy mod_0.2 --eval_datasets ego4d_refined_narration_stream_val --resume_from_checkpoint outputs/ego4d_fast1+_mod0.4/checkpoint-6436 --per_device_eval_batch_size 1 --eval_accumulation_steps 1 --gradient_checkpointing True --prediction_loss_only False --dataloader_num_workers 2 --bf16 True --tf32 True --connector_type mlp --finetune_modules connector --attn_implementation flash_attention_2 --gradient_checkpointing True --ddp_find_unused_parameters False --dataloader_pin_memory False
# --live_version live1+ --eval_datasets ego4d_refined_narration_stream_val --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus --per_device_eval_batch_size 1 --eval_accumulation_steps 1 --gradient_checkpointing True --prediction_loss_only False --dataloader_num_workers 2 --bf16 True --tf32 True --connector_type mlp --finetune_modules connector --attn_implementation flash_attention_2 --gradient_checkpointing True --ddp_find_unused_parameters False --dataloader_pin_memory False