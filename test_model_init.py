#!/usr/bin/env python3
"""
測試腳本：驗證模型初始化和 LoRA 配置
"""

import torch
from models.arguments_live import LiveTrainingArguments, LiveOneTrainingArguments
from models.live_llama import build_live_llama
from dataclasses import asdict


def test_model_initialization():
    """測試模型初始化"""
    print("Testing model initialization with different connector types...")
    
    # 測試 MLP 連接器
    print("\n1. Testing MLP Connector:")
    try:
        args_mlp = LiveOneTrainingArguments(connector_type='mlp')
        print(f"  Arguments: connector_type={args_mlp.connector_type}, finetune_modules={args_mlp.finetune_modules}")
        
        model_mlp, tokenizer_mlp = build_live_llama(is_training=True, **asdict(args_mlp))
        print(f"  ✓ MLP model initialized successfully")
        print(f"  Connector type: {type(model_mlp.base_model.model.connector).__name__}")
        
        # 檢查可訓練參數
        model_mlp.print_trainable_parameters()
        
    except Exception as e:
        print(f"  ✗ MLP model initialization failed: {e}")
    
    # 測試 Mamba 連接器
    print("\n2. Testing Mamba Connector:")
    try:
        args_mamba = LiveOneTrainingArguments(connector_type='mamba')
        print(f"  Arguments: connector_type={args_mamba.connector_type}, finetune_modules={args_mamba.finetune_modules}")
        
        model_mamba, tokenizer_mamba = build_live_llama(is_training=True, **asdict(args_mamba))
        print(f"  ✓ Mamba model initialized successfully")
        print(f"  Connector type: {type(model_mamba.base_model.model.connector).__name__}")
        
        # 檢查可訓練參數
        model_mamba.print_trainable_parameters()
        
    except Exception as e:
        print(f"  ✗ Mamba model initialization failed: {e}")
        import traceback
        traceback.print_exc()


def test_connector_modules():
    """測試連接器模組的可見性"""
    print("\n3. Testing connector module visibility:")
    
    try:
        args = LiveOneTrainingArguments(connector_type='mamba')
        model, tokenizer = build_live_llama(is_training=True, **asdict(args))
        
        # 檢查模組結構
        print(f"  Base model type: {type(model.base_model)}")
        print(f"  Model type: {type(model.base_model.model)}")
        
        # 檢查 connector 是否存在
        if hasattr(model.base_model.model, 'connector'):
            print(f"  ✓ connector module found: {type(model.base_model.model.connector)}")
        else:
            print(f"  ✗ connector module not found")
            
        # 列出所有子模組
        print(f"  Available modules:")
        for name, module in model.base_model.model.named_children():
            print(f"    - {name}: {type(module)}")
            
    except Exception as e:
        print(f"  ✗ Module visibility test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_initialization()
    test_connector_modules()
