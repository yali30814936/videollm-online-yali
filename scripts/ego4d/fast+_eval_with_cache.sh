#!/bin/bash

export TOKENIZERS_PARALLELISM=false
# 設定檢查點和緩存目錄
CHECKPOINT=outputs/ego4d_fast1+_nomod
CACHE_DIR=prediction_cache/nomod_llama3.1_3x3_4290_long2_clean

torchrun --nproc_per_node 1 \
    evaluate.py \
    --live_version live1+ \
    --resume_from_checkpoint $CHECKPOINT \
    --eval_datasets ego4d_refined_narration_stream_val \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --connector_type mlp \
    --finetune_modules connector \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory False \
    --use_conversation_eval True \
    --prediction_cache_dir $CACHE_DIR \
    --skip_inference True \
    # --n_max 4096 \
    # --n_min 4096 \
    # --attn_implementation flash_attention_2 \
    # --use_infcache True \
    # --attn_implementation sdpa \
