import torch
import torch.nn as nn
import math

class SeasonalEncoder(nn.Module):
    """
    Encodes seasonal metadata (Day of Year, Solar Angle, etc.) into a continuous embedding.
    Phase 5.3: Seasonal Encoding
    """
    def __init__(self, metadata_dim=4, embed_dim=64):
        super(SeasonalEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, metadata):
        # metadata: [Batch, 4] -> e.g. [DayOfYear_sin, DayOfYear_cos, SolarZenith, SolarAzimuth]
        return self.fc(metadata)

class TemporalAttentionTransformer(nn.Module):
    """
    Phase 5.2: Temporal Transformer
    Fuses multi-date inputs (T-4, T-3, T-2, T-1, Current) to identify persistent structures.
    """
    def __init__(self, feature_dim=64, num_heads=4, num_layers=2):
        super(TemporalAttentionTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encoding for time steps (0 to 4)
        self.time_embed = nn.Embedding(5, feature_dim)

    def forward(self, temporal_features):
        """
        temporal_features: [Batch, Time=5, Channels, H, W]
        """
        B, T, C, H, W = temporal_features.size()
        
        # Flatten spatial dimensions
        # [B, T, C, N] -> [B, N, T, C] -> [B*N, T, C]
        N = H * W
        feat_flat = temporal_features.view(B, T, C, N).permute(0, 3, 1, 2).reshape(B * N, T, C)
        
        # Add time positional embedding
        time_idx = torch.arange(T, device=temporal_features.device).unsqueeze(0).expand(B * N, T)
        t_emb = self.time_embed(time_idx)
        feat_flat = feat_flat + t_emb
        
        # Apply Temporal Self-Attention
        out = self.transformer(feat_flat) # [B*N, T, C]
        
        # Typically we pool across time or just take the latest feature (T-0)
        # Let's take the current frame enriched by temporal context (last element if T is chronological)
        out_current = out[:, -1, :] # [B*N, C]
        
        # Reshape back to spatial
        out_spatial = out_current.view(B, N, C).permute(0, 2, 1).view(B, C, H, W)
        return out_spatial
