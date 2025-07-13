# import torch
import torch.nn as nn
from torch import Tensor
# import einops
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
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_ssm: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_ssm = n_ssm
        
        # Input projection with non-linearity (替代 PreNet，包含 LeakyReLU)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Mamba SSM layers (基於借來的代碼設計)
        self.ssms = nn.ModuleList([
            create_block(hidden_dim, d_intermediate=0, layer_idx=i) 
            for i in range(n_ssm)
        ])
        
        # Layer normalization (基於借來的代碼)
        self.norm_fn = nn.LayerNorm(hidden_dim)
        
        # Output projection with non-linearity (替代 PostNet，包含 LeakyReLU)
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights using the borrowed code's method
        self.apply(partial(_init_weights, n_layer=n_ssm))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass based on Video_Mamba_Seq design but simplified.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, output_dim]
        """
        # Input projection with LeakyReLU (基於 PreNet)
        x = self.input_proj(x)
        
        # Mamba SSM processing (基於借來的代碼結構)
        hidden_states = x
        residual = None
        
        # Apply Mamba layers with residual connections
        for ssm in self.ssms:
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=None
            )
        
        # Final residual connection and normalization (基於借來的代碼)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))
        
        # Output projection with LeakyReLU (基於 PostNet)
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
