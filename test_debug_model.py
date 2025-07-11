#!/usr/bin/env python3
"""
詳細測試腳本：逐步調試模型初始化問題
"""

import torch
from models.arguments_live import LiveTrainingArguments, LiveOneTrainingArguments
from models.live_llama.modeling_live_llama import LiveLlamaForCausalLM
from models.live_llama.configuration_live_llama import LiveLlamaConfig
from models.connectors import build_connector
from dataclasses import asdict


def test_connector_creation():
    """測試連接器創建"""
    print("Testing connector creation...")
    
    input_dim = 1024   # SigLIP vision_hidden_size
    hidden_dim = 4096  # Llama hidden_size
    output_dim = 4096  # Llama hidden_size
    
    # 測試 MLP 連接器
    print("\n1. Testing MLP Connector creation:")
    try:
        mlp_connector = build_connector('mlp', input_dim, hidden_dim, output_dim)
        print(f"  ✓ MLP connector created: {type(mlp_connector)}")
        print(f"  Parameters: {sum(p.numel() for p in mlp_connector.parameters()):,}")
    except Exception as e:
        print(f"  ✗ MLP connector creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 測試 Mamba 連接器
    print("\n2. Testing Mamba Connector creation:")
    try:
        mamba_connector = build_connector('mamba', input_dim, hidden_dim, output_dim)
        print(f"  ✓ Mamba connector created: {type(mamba_connector)}")
        print(f"  Parameters: {sum(p.numel() for p in mamba_connector.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Mamba connector creation failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_creation_step_by_step():
    """逐步測試模型創建"""
    print("\n\nTesting model creation step by step...")
    
    args = LiveOneTrainingArguments(connector_type='mlp')
    
    print(f"1. Arguments: {args.connector_type}")
    
    # 步驟 1: 創建配置
    print("2. Creating config...")
    try:
        config = LiveLlamaConfig.from_pretrained(
            args.llm_pretrained,
            vision_pretrained=args.vision_pretrained,
            frame_resolution=args.frame_resolution,
            frame_token_cls=args.frame_token_cls,
            frame_num_tokens=args.frame_num_tokens,
            connector_type=args.connector_type,
            stream_loss_weight=args.stream_loss_weight,
        )
        print(f"  ✓ Config created, connector_type: {config.connector_type}")
    except Exception as e:
        print(f"  ✗ Config creation failed: {e}")
        return
    
    # 步驟 2: 創建基礎模型（不加 LoRA）
    print("3. Creating base model...")
    try:
        model = LiveLlamaForCausalLM.from_pretrained(
            args.llm_pretrained,
            config=config,
            torch_dtype='auto',
            attn_implementation=args.attn_implementation
        )
        print(f"  ✓ Base model created")
        print(f"  Connector type: {type(model.connector)}")
    except Exception as e:
        print(f"  ✗ Base model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步驟 3: 檢查 connector 屬性
    print("4. Checking connector attribute...")
    if hasattr(model, 'connector'):
        print(f"  ✓ Model has connector: {type(model.connector)}")
    else:
        print(f"  ✗ Model missing connector")
        return
    
    # 步驟 4: 測試 LoRA 配置
    print("5. Testing LoRA configuration...")
    try:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=args.finetune_modules,
            inference_mode=False,
        )
        print(f"  LoRA config modules_to_save: {lora_config.modules_to_save}")
        
        # 檢查 modules_to_save 是否存在於模型中
        for module_name in lora_config.modules_to_save:
            if hasattr(model, module_name):
                print(f"  ✓ Module '{module_name}' found in model")
            else:
                print(f"  ✗ Module '{module_name}' NOT found in model")
                print(f"    Available modules: {list(dict(model.named_children()).keys())}")
        
        # 應用 LoRA
        peft_model = get_peft_model(model, lora_config)
        print(f"  ✓ LoRA applied successfully")
        peft_model.print_trainable_parameters()
        
    except Exception as e:
        print(f"  ✗ LoRA configuration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_connector_creation()
    test_model_creation_step_by_step()
