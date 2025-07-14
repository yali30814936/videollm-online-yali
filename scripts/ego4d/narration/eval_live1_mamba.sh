#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# 設置 GPU 數量（評估通常不需要多 GPU，但可以根據需要調整）
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 1 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    launcher="torchrun --nproc_per_node 1"
fi

${launcher} evaluate.py \
    --live_version live1 \
    --eval_datasets ego4d_refined_narration_stream_val \
    --resume_from_checkpoint outputs/ego4d_narration_train/live1_mamba/checkpoint-xxxx \
    --per_device_eval_batch_size 1 \
    --bf16 True \
    --tf32 True \
    --connector_type mamba \
    --finetune_modules connector \
    --output_dir outputs/ego4d_narration_train/live1_mamba_eval \
    --dataloader_num_workers 16 \
    --prediction_loss_only False \
    --attn_implementation flash_attention_2
