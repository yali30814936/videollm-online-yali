import torch
import torch.nn as nn
from torch import Tensor
import math
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from functools import partial


class MLPConnector(nn.Module):
    """
    Traditional MLP-based connector (original implementation).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.connector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.connector(x)


def build_connector(connector_type: str, input_dim: int, hidden_dim: int, output_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to build different types of connectors.
    
    Args:
        connector_type: Type of connector ('mlp' or 'mamba')
        input_dim: Input dimension (vision_hidden_size)
        hidden_dim: Hidden dimension (usually same as output_dim)
        output_dim: Output dimension (model hidden_size)
        **kwargs: Additional arguments for specific connector types
    
    Returns:
        Connector module
    """
    if connector_type == 'mlp':
        return MLPConnector(input_dim, hidden_dim, output_dim)
    elif connector_type == 'mamba':
        return MambaConnector(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}. Supported types: 'mlp', 'mamba'")


class MambaConnector(nn.Module):
    """
    Mamba-based connector inspired by Video_Mamba_Seq but simplified for embedding processing.
    Based on the borrowed code structure but without PreNet, PostNet, ClsNet and pytorch_lightning.
    
    Integrated improvements:
    - Stable weight initialization for SSM layers
    - Layer normalization and dropout for stability
    - Numerical stability checks
    - Better error handling
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_ssm: int = 2,
        use_stable_init: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_ssm = n_ssm
        self.use_stable_init = use_stable_init
        
        # Input projection with normalization and dropout (改進：添加穩定性)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),  # 添加輸入標準化
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # 使用 GELU 而非 LeakyReLU，更穩定
            nn.Dropout(dropout)
        )
        
        # Mamba SSM layers (基於借來的代碼設計)
        self.ssms = nn.ModuleList([
            create_block(hidden_dim, d_intermediate=0, layer_idx=i) 
            for i in range(n_ssm)
        ])
        
        # Layer normalization (基於借來的代碼)
        self.norm_fn = nn.LayerNorm(hidden_dim)
        
        # Output projection with normalization and dropout (改進：添加穩定性)
        self.output_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights using improved method
        if use_stable_init:
            self._stable_init_weights()
        else:
            self.apply(partial(_init_weights, n_layer=n_ssm))
    
    def _stable_init_weights(self):
        """穩定的權重初始化（修復 SSM 權重過大問題）- 生產環境加強版"""
        # 標準初始化
        self.apply(partial(_init_weights, n_layer=self.n_ssm))
        
        # 對 SSM 層進行超保守處理（生產環境加強）
        for i, ssm in enumerate(self.ssms):
            # 進一步降低 A_log 的初始值（原來範數 >300，現在 <0.1）
            if hasattr(ssm.mixer, 'A_log'):
                with torch.no_grad():
                    # 使用極小的初始化範圍
                    ssm.mixer.A_log.data = torch.randn_like(ssm.mixer.A_log.data) * 0.001
                    # 確保在極小範圍內
                    ssm.mixer.A_log.data.clamp_(-0.01, 0.01)
            
            # 進一步降低 dt_proj 的 bias（原來範數 >180，現在 <0.01）
            if hasattr(ssm.mixer, 'dt_proj') and hasattr(ssm.mixer.dt_proj, 'bias'):
                if ssm.mixer.dt_proj.bias is not None:
                    with torch.no_grad():
                        ssm.mixer.dt_proj.bias.data = torch.randn_like(ssm.mixer.dt_proj.bias.data) * 0.0001
                        # 限制在極小範圍
                        ssm.mixer.dt_proj.bias.data.clamp_(-0.001, 0.001)
            
            # 對 D 參數進行特殊處理（如果存在）
            if hasattr(ssm.mixer, 'D'):
                with torch.no_grad():
                    ssm.mixer.D.data = torch.ones_like(ssm.mixer.D.data) * 0.1
            
            # 對其他線性層使用更保守的初始化
            for name, module in ssm.named_modules():
                if isinstance(module, nn.Linear):
                    # 使用更小的Xavier初始化
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # 對輸入和輸出投影使用保守初始化
        for module in [self.input_proj, self.output_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass based on Video_Mamba_Seq design but simplified.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, output_dim]
        """
        # Input projection with normalization (基於改進的 PreNet)
        x = self.input_proj(x)
        
        # Mamba SSM processing (基於借來的代碼結構)
        hidden_states = x
        residual = None
        
        # Apply Mamba layers with residual connections
        for i, ssm in enumerate(self.ssms):
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=None
            )
            
            # 溫和的數值檢查（只在異常時警告，不拋出異常）
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                # 嘗試修復而非直接拋出錯誤
                hidden_states = torch.where(
                    torch.isnan(hidden_states) | torch.isinf(hidden_states),
                    torch.zeros_like(hidden_states),
                    hidden_states
                )
                print(f"⚠️  SSM 層 {i} 檢測到異常值，已自動修復")
        
        # Final residual connection and normalization (基於借來的代碼)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))
        
        # Output projection (基於改進的 PostNet)
        output = self.output_proj(hidden_states)
        
        return output
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate inference cache for each SSM layer (基於借來的代碼)
        """
        return {
            i: ssm.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, ssm in enumerate(self.ssms)
        }
    
    def enable_production_mode(self):
        """啟用生產環境模式，加強數值穩定性"""
        self._production_mode = True
        
        # 重新初始化權重為更保守的值
        self._stable_init_weights()
        
        # 將所有 SSM 層設為 float32（避免 bf16 問題）
        for ssm in self.ssms:
            ssm.float()
        
        # 啟用梯度裁剪
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        
        print(f"[INFO] MambaConnector 已啟用生產環境模式")
    
    def get_parameter_stats(self):
        """獲取參數統計，用於調試"""
        stats = {}
        
        for i, ssm in enumerate(self.ssms):
            layer_stats = {}
            
            if hasattr(ssm.mixer, 'A_log'):
                layer_stats['A_log_norm'] = ssm.mixer.A_log.data.norm().item()
                layer_stats['A_log_range'] = (
                    ssm.mixer.A_log.data.min().item(),
                    ssm.mixer.A_log.data.max().item()
                )
            
            if hasattr(ssm.mixer, 'dt_proj') and hasattr(ssm.mixer.dt_proj, 'bias'):
                if ssm.mixer.dt_proj.bias is not None:
                    layer_stats['dt_bias_norm'] = ssm.mixer.dt_proj.bias.data.norm().item()
                    layer_stats['dt_bias_range'] = (
                        ssm.mixer.dt_proj.bias.data.min().item(),
                        ssm.mixer.dt_proj.bias.data.max().item()
                    )
            
            stats[f'ssm_layer_{i}'] = layer_stats
        
        return stats
