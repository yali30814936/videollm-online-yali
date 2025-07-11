#!/usr/bin/env python3
"""
簡單的訓練測試腳本
"""

import torch
from models.arguments_live import LiveOneTrainingArguments
from models.live_llama import build_live_llama
from dataclasses import asdict


def test_training_setup():
    """測試訓練設置"""
    print("Testing training setup with different connector types...")
    
    # 測試數據
    batch_size = 1
    seq_len = 10
    num_frames = 5
    
    # 測試 MLP 連接器
    print("\n1. Testing MLP Connector training setup:")
    try:
        args_mlp = LiveOneTrainingArguments(
            connector_type='mlp',
            output_dir='outputs/test_mlp'
        )
        model_mlp, tokenizer_mlp = build_live_llama(is_training=True, **asdict(args_mlp))
        
        # 創建模擬數據
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        frames = torch.randn(batch_size, num_frames, 3, 384, 384)
        labels = input_ids.clone()
        
        # 前向傳播測試
        with torch.no_grad():
            outputs = model_mlp(input_ids=input_ids, frames=frames, labels=labels)
        
        print(f"  ✓ MLP forward pass successful")
        print(f"  Loss: {outputs.loss}")
        print(f"  Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"  ✗ MLP training setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 測試 Mamba 連接器
    print("\n2. Testing Mamba Connector training setup:")
    try:
        args_mamba = LiveOneTrainingArguments(
            connector_type='mamba',
            output_dir='outputs/test_mamba'
        )
        model_mamba, tokenizer_mamba = build_live_llama(is_training=True, **asdict(args_mamba))
        
        # 創建模擬數據
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        frames = torch.randn(batch_size, num_frames, 3, 384, 384)
        labels = input_ids.clone()
        
        # 前向傳播測試
        with torch.no_grad():
            outputs = model_mamba(input_ids=input_ids, frames=frames, labels=labels)
        
        print(f"  ✓ Mamba forward pass successful")
        print(f"  Loss: {outputs.loss}")
        print(f"  Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"  ✗ Mamba training setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_training_setup()
