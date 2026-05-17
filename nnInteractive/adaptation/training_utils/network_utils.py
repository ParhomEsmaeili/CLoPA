from torch import nn
import torch.nn.functional as F

class ConditioningNetwork(nn.Module):
    """Generates FILM parameters (gamma, beta) from conditioning input."""
    def __init__(self, conditioning_channels: int, feature_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feature_channels)
        )
    
    def forward(self, conditioning):
        """
        Args:
            conditioning: (B, C_cond) or (B, C_cond, 1, 1, 1) pooled tensor
        Returns:
            gamma, beta: each (B, C_feat, 1, 1, 1)
        """
        # Flatten to (B, C_cond)
        batch_size = conditioning.shape[0]
        if conditioning.dim() > 2:
            conditioning = conditioning.view(batch_size, -1)
        
        params = self.mlp(conditioning)  # (B, 2*C_feat)
        gamma_beta = params.view(batch_size, -1, 2)  # (B, C_feat, 2)
        gamma = gamma_beta[..., 0]  # (B, C_feat)
        beta = gamma_beta[..., 1]   # (B, C_feat)
        
        return gamma, beta


class FILMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation (FILM) to features."""
    def __init__(self, feature_channels: int, conditioning_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.conditioning_net = ConditioningNetwork(conditioning_channels, feature_channels, hidden_dim)
    
    def forward(self, features, conditioning):
        """
        Args:
            features: (B, C, *spatial) feature maps
            conditioning: (B, C_cond, *spatial) or (B, C_cond) conditioning input
        Returns:
            modulated: (B, C, *spatial)
        """
        batch_size = features.shape[0]
        feat_channels = features.shape[1]
        spatial_dims = features.shape[2:]
        
        # Global average pool conditioning
        if conditioning.dim() > 2:
            cond_pooled = F.adaptive_avg_pool3d(conditioning, 1) if len(spatial_dims) == 3 else \
                          F.adaptive_avg_pool2d(conditioning, 1)
        else:
            cond_pooled = conditioning.unsqueeze(-1).unsqueeze(-1)
            if len(spatial_dims) == 3:
                cond_pooled = cond_pooled.unsqueeze(-1)
        
        # Generate gamma, beta
        gamma, beta = self.conditioning_net(cond_pooled)  # (B, C_feat)
        
        # Reshape for broadcasting: (B, C_feat) → (B, C_feat, 1, 1, 1) or (B, C_feat, 1, 1)
        for _ in spatial_dims:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        # Apply FILM: output = gamma * features + beta
        modulated = gamma * features + beta
        return modulated