import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None
    print("Warning: mamba-ssm not available, falling back to MLP connector")


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
        return MLPConnector(input_dim, output_dim, output_dim)
    elif connector_type == 'mamba':
        if not MAMBA_AVAILABLE:
            print("Warning: mamba-ssm not available, falling back to MLP connector")
            return MLPConnector(input_dim, output_dim, output_dim)
        
        # Check if CUDA is available for Mamba
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, Mamba requires CUDA. Falling back to MLP connector")
            return MLPConnector(input_dim, output_dim, output_dim)
            
        return MambaConnector(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}. Supported types: 'mlp', 'mamba'")


class MambaConnector(nn.Module):
    """
    High-performance connector using official Mamba implementation.
    Provides optimized state space modeling with CUDA kernels.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        num_layers: int = 1
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required for MambaConnector. Please install it with: pip install mamba-ssm")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection to Mamba's expected dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba layers
        if not MAMBA_AVAILABLE or Mamba is None:
            raise ImportError("mamba-ssm is required for MambaConnector")
            
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bias=True,
                conv_bias=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms for residual connections
        if num_layers > 1:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using official Mamba implementation.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, output_dim]
        """
        # Check if CUDA is available and move to GPU if needed
        if torch.cuda.is_available() and not x.is_cuda:
            device = torch.cuda.current_device()
            x = x.to(device)
            self.to(device)
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Apply Mamba layers with residual connections
        for i, mamba_layer in enumerate(self.mamba_layers):
            residual = x
            
            # Apply Mamba layer
            x = mamba_layer(x)
            
            # Add residual connection for deeper networks
            if self.num_layers > 1:
                x = self.layer_norms[i](x + residual)
                x = self.dropout(x)
            
        # Final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return x
