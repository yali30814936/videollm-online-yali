import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
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
    Mamba-based connector for video feature processing.
    Simplified version without PyTorch Lightning and classification components.
    Handles SigLIP class tokens to LLaMA3 token conversion.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_ssm: int = 1):
        super().__init__()
        self.pre_net = PreNet(input_dim, hidden_dim)
        self.ssms = nn.ModuleList(
            [create_block(hidden_dim, d_intermediate=0, layer_idx=i, ssm_cfg={"layer":"Mamba2"}) for i in range(n_ssm)]
        )
        self.norm_fn = nn.LayerNorm(hidden_dim)
        self.post_net = PostNet(hidden_dim, output_dim)
        
        # Initialize weights
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
        Args:
            x: Input tensor from SigLIP class tokens
               - Shape: (b, t, d) for class tokens per frame
        Returns:
            Output tensor for LLaMA3: (b, t, output_dim)
        """
        # Preprocessing: input_dim -> hidden_dim
        x = self.pre_net(x)
        
        # Pass through Mamba blocks for temporal modeling
        hidden_states = x
        residual = None
        
        for ssm in self.ssms:
            hidden_states, residual = ssm(hidden_states, residual)
        
        # Final normalization
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))
        
        x = self.post_net(x)
        
        return x


class PreNet(nn.Module):
    """Preprocessing network to transform input to model dimension."""
    
    def __init__(self, d_input: int, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_input, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x


class PostNet(nn.Module):
    """Postprocessing network to transform model output to target dimension."""
    
    def __init__(self, d_model: int, d_output: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_output)

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x
