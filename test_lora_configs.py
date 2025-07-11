#!/usr/bin/env python3
"""
測試不同的 finetune_modules 配置
"""

import torch
from models.arguments_live import LiveOneTrainingArguments
from models.live_llama.modeling_live_llama import LiveLlamaForCausalLM
from models.live_llama.configuration_live_llama import LiveLlamaConfig
from peft import LoraConfig, get_peft_model


def test_different_modules_to_save():
    """測試不同的 modules_to_save 配置"""
    print("Testing different modules_to_save configurations...")
    
    # 創建基礎模型
    args = LiveOneTrainingArguments(connector_type='mlp')
    config = LiveLlamaConfig.from_pretrained(
        args.llm_pretrained,
        vision_pretrained=args.vision_pretrained,
        frame_resolution=args.frame_resolution,
        frame_token_cls=args.frame_token_cls,
        frame_num_tokens=args.frame_num_tokens,
        connector_type=args.connector_type,
        stream_loss_weight=args.stream_loss_weight,
    )
    
    model = LiveLlamaForCausalLM.from_pretrained(
        args.llm_pretrained,
        config=config,
        torch_dtype='auto',
        attn_implementation=args.attn_implementation
    )
    
    print(f"Base model connector type: {type(model.connector)}")
    
    # 測試 1: 空的 modules_to_save
    print("\n1. Testing with empty modules_to_save:")
    try:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=[],  # 空列表
            inference_mode=False,
        )
        peft_model = get_peft_model(model, lora_config)
        print("  ✓ Success with empty modules_to_save")
        peft_model.print_trainable_parameters()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 重新創建模型（因為前面的可能已經被修改）
    model = LiveLlamaForCausalLM.from_pretrained(
        args.llm_pretrained,
        config=config,
        torch_dtype='auto',
        attn_implementation=args.attn_implementation
    )
    
    # 測試 2: 只 finetune connector，不使用 modules_to_save
    print("\n2. Testing manual connector finetuning:")
    try:
        # 凍結所有參數
        for param in model.parameters():
            param.requires_grad = False
        
        # 只解凍 connector
        for param in model.connector.parameters():
            param.requires_grad = True
        
        # 應用 LoRA（不包含 connector）
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=[],  # 空列表
            inference_mode=False,
        )
        peft_model = get_peft_model(model, lora_config)
        
        # 手動統計可訓練參數
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in peft_model.parameters())
        print(f"  ✓ Manual connector finetuning successful")
        print(f"  trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_mamba_lora():
    """測試 Mamba 是否有同樣的問題"""
    print("\n\n3. Testing Mamba with LoRA:")
    try:
        args = LiveOneTrainingArguments(connector_type='mamba')
        config = LiveLlamaConfig.from_pretrained(
            args.llm_pretrained,
            vision_pretrained=args.vision_pretrained,
            frame_resolution=args.frame_resolution,
            frame_token_cls=args.frame_token_cls,
            frame_num_tokens=args.frame_num_tokens,
            connector_type=args.connector_type,
            stream_loss_weight=args.stream_loss_weight,
        )
        
        model = LiveLlamaForCausalLM.from_pretrained(
            args.llm_pretrained,
            config=config,
            torch_dtype='auto',
            attn_implementation=args.attn_implementation
        )
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=['connector'],
            inference_mode=False,
        )
        peft_model = get_peft_model(model, lora_config)
        print("  ✓ Mamba with LoRA successful")
        peft_model.print_trainable_parameters()
        
    except Exception as e:
        print(f"  ✗ Mamba with LoRA failed: {e}")


if __name__ == "__main__":
    test_different_modules_to_save()
    test_mamba_lora()
