# import torch
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
