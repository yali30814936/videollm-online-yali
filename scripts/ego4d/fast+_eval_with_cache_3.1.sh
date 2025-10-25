#!/bin/bash

export TOKENIZERS_PARALLELISM=false
# 設定檢查點和緩存目錄
# CHECKPOINT="outputs/ego4d_narration+goalstep_livechat+robustness/live1+"
# CACHE_DIR="./prediction_cache/ego4d_narration+goalstep_livechat+robustness/live1+"
# CHECKPOINT="outputs/ego4d_fast1+_mod0.4/checkpoint-6436 --vision_drop_strategy mod_0.4"
# CACHE_DIR="./prediction_cache/ego4d_fast1+_mod0.4/checkpoint-6436"
CHECKPOINT=outputs/ego4d_nomod_llama3.1_3x3/checkpoint-4290
CACHE_DIR=./prediction_cache/nomod_llama3.1_3x3_4290_long2_clean
# CHECKPOINT=outputs/ego4d_fast1+_nomod
# CACHE_DIR=./prediction_cache/nomod_long_tbfifo

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
    --attn_implementation flash_attention_2 \
    --llm_pretrained meta-llama/Llama-3.1-8B-Instruct \
    --skip_inference True \
    # --n_max 8192 \
    # --n_min 8192 \
    # --use_infcache True \
    # --attn_implementation flash_attention_2 \
    # --skip_inference True \
    # --n_max 4096 \
    # --n_min 4096 \
    # --attn_implementation sdpa \
    # --n_max 2048 \
