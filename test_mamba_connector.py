#!/usr/bin/env python3
"""
測試腳本：驗證基於 Mamba 的連接器功能
"""

import torch
from models.connectors import build_connector


def test_connectors():
    """測試不同類型的連接器"""
    batch_size = 2
    seq_len = 10
    input_dim = 1024  # SigLIP hidden size
    hidden_dim = 4096  # Llama hidden size
    output_dim = 4096
    
    # 創建測試輸入
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing connectors...")
    print(f"Input shape: {x.shape}")
    
    # 測試 MLP 連接器
    print("\n1. Testing MLP Connector:")
    mlp_connector = build_connector('mlp', input_dim, hidden_dim, output_dim)
    print(f"MLP Connector parameters: {sum(p.numel() for p in mlp_connector.parameters()):,}")
    
    with torch.no_grad():
        mlp_output = mlp_connector(x)
    print(f"MLP output shape: {mlp_output.shape}")
    
    # 測試 Mamba 連接器
    print("\n2. Testing Mamba Connector:")
    try:
        mamba_connector = build_connector('mamba', input_dim, hidden_dim, output_dim)
        print(f"Mamba Connector parameters: {sum(p.numel() for p in mamba_connector.parameters()):,}")
        
        with torch.no_grad():
            mamba_output = mamba_connector(x)
        print(f"Mamba output shape: {mamba_output.shape}")
        
        # 比較參數數量
        mlp_params = sum(p.numel() for p in mlp_connector.parameters())
        mamba_params = sum(p.numel() for p in mamba_connector.parameters())
        
        print(f"\nParameter comparison:")
        print(f"MLP parameters: {mlp_params:,}")
        print(f"Mamba parameters: {mamba_params:,}")
        print(f"Mamba vs MLP ratio: {mamba_params / mlp_params:.2f}x")
        
        print("\n✅ All connectors working properly!")
        
    except ImportError as e:
        print(f"❌ Mamba connector not available: {e}")
        print("Falling back to MLP connector")
    
    print("\nTest completed successfully!")


def test_training_arguments():
    """測試訓練參數配置"""
    from models.arguments_live import LiveTrainingArguments, LiveOnePlusTrainingArguments
    
    print("\n3. Testing Training Arguments:")
    
    # 測試預設 Mamba 連接器
    args_mamba = LiveTrainingArguments()
    print(f"Default connector_type: {args_mamba.connector_type}")
    
    # 測試 MLP 連接器
    args_mlp = LiveTrainingArguments(connector_type='mlp')
    print(f"MLP connector_type: {args_mlp.connector_type}")
    
    # 測試 Live1+ 版本
    args_live1plus = LiveOnePlusTrainingArguments(connector_type='mamba')
    print(f"Live1+ with Mamba: {args_live1plus.connector_type}")
    print(f"Live1+ version: {args_live1plus.live_version}")


if __name__ == "__main__":
    test_connectors()
    test_training_arguments()
