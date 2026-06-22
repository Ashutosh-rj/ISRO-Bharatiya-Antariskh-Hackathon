import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Conditional GAN architecture.
    Evaluates whether N x N patches in an image are real or fake.
    """
    def __init__(self, in_channels=7, ndf=64, n_layers=3): # 4(opt+mask) + 3(target)
        super(PatchGANDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer Fusion (Phase 4.3).
    Replaces basic cross-attention with a multi-head transformer approach.
    Fuses Optical, SAR, and Sentinel-2 features.
    """
    def __init__(self, channels, attention_heads=4):
        super().__init__()
        self.attention_heads = attention_heads
        self.embed_dim = channels
        
        # Multi-head attention for Opt <-> SAR
        self.mha_sar = nn.MultiheadAttention(embed_dim=channels, num_heads=attention_heads, batch_first=True)
        # Multi-head attention for Opt <-> S2
        self.mha_s2 = nn.MultiheadAttention(embed_dim=channels, num_heads=attention_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, opt_feat, sar_feat, s2_feat=None):
        B, C, H, W = opt_feat.size()
        N = H * W
        
        # Flatten spatial dims
        opt_flat = opt_feat.view(B, C, N).permute(0, 2, 1) # B, N, C
        sar_flat = sar_feat.view(B, C, N).permute(0, 2, 1)
        
        # Cross-attention: Opt queries SAR
        attn_sar, _ = self.mha_sar(query=opt_flat, key=sar_flat, value=sar_flat)
        out = self.norm1(opt_flat + attn_sar)
        
        # Cross-attention: Opt queries S2 (if available)
        if s2_feat is not None:
            s2_flat = s2_feat.view(B, C, N).permute(0, 2, 1)
            attn_s2, _ = self.mha_s2(query=out, key=s2_flat, value=s2_flat)
            out = self.norm2(out + attn_s2)
        
        # FFN
        ffn_out = self.ffn(out)
        out = self.norm3(out + ffn_out)
        
        # Reshape back to spatial
        return out.permute(0, 2, 1).view(B, C, H, W)

class SARFusionUNet(nn.Module):
    """
    Dual-Encoder U-Net architecture.
    Encoder A: Optical + Mask (4ch)
    Encoder B: SAR VV+VH (2ch)
    """
    def __init__(self, optical_channels=4, sar_channels=2, s2_channels=4, out_channels=3, base_filters=64, attention_heads=4, dropout=0.2):
        super().__init__()
        
        # Optical Encoder (LISS-IV)
        self.opt_enc1 = ConvBlock(optical_channels, base_filters, dropout=dropout)
        self.opt_enc1 = ConvBlock(optical_channels, base_filters, dropout=dropout)
        self.opt_pool1 = nn.MaxPool2d(2)
        self.opt_enc2 = ConvBlock(base_filters, base_filters*2, dropout=dropout)
        self.opt_pool2 = nn.MaxPool2d(2)
        self.opt_enc3 = ConvBlock(base_filters*2, base_filters*4, dropout=dropout)
        self.opt_pool3 = nn.MaxPool2d(2)
        
        # SAR Encoder
        self.sar_enc1 = ConvBlock(sar_channels, base_filters, dropout=dropout)
        self.sar_pool1 = nn.MaxPool2d(2)
        self.sar_enc2 = ConvBlock(base_filters, base_filters*2, dropout=dropout)
        self.sar_pool2 = nn.MaxPool2d(2)
        self.sar_enc3 = ConvBlock(base_filters*2, base_filters*4, dropout=dropout)
        self.sar_pool3 = nn.MaxPool2d(2)
        
        # Sentinel-2 Encoder
        self.s2_enc1 = ConvBlock(s2_channels, base_filters, dropout=dropout)
        self.s2_pool1 = nn.MaxPool2d(2)
        self.s2_enc2 = ConvBlock(base_filters, base_filters*2, dropout=dropout)
        self.s2_pool2 = nn.MaxPool2d(2)
        self.s2_enc3 = ConvBlock(base_filters*2, base_filters*4, dropout=dropout)
        self.s2_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck & Fusion
        self.opt_bottleneck = ConvBlock(base_filters*4, base_filters*8, dropout=dropout)
        self.sar_bottleneck = ConvBlock(base_filters*4, base_filters*8, dropout=dropout)
        self.s2_bottleneck = ConvBlock(base_filters*4, base_filters*8, dropout=dropout)
        
        self.fusion = CrossModalTransformer(base_filters*8, attention_heads=attention_heads)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters*8, base_filters*4, dropout=dropout) # + skip connection
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters*4, base_filters*2, dropout=dropout)
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters*2, base_filters, dropout=dropout)
        
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, optical, mask, sar, s2=None):
        opt_input = torch.cat([optical, mask], dim=1)
        
        # Encoding
        opt_e1 = self.opt_enc1(opt_input)
        opt_e2 = self.opt_enc2(self.opt_pool1(opt_e1))
        opt_e3 = self.opt_enc3(self.opt_pool2(opt_e2))
        opt_b = self.opt_bottleneck(self.opt_pool3(opt_e3))
        
        sar_e1 = self.sar_enc1(sar)
        sar_e2 = self.sar_enc2(self.sar_pool1(sar_e1))
        sar_e3 = self.sar_enc3(self.sar_pool2(sar_e2))
        sar_b = self.sar_bottleneck(self.sar_pool3(sar_e3))
        
        s2_b = None
        if s2 is not None:
            s2_e1 = self.s2_enc1(s2)
            s2_e2 = self.s2_enc2(self.s2_pool1(s2_e1))
            s2_e3 = self.s2_enc3(self.s2_pool2(s2_e2))
            s2_b = self.s2_bottleneck(self.s2_pool3(s2_e3))
        
        # Fusion at bottleneck
        fused = self.fusion(opt_b, sar_b, s2_b)
        attention_map = None # Extracting map from multihead attention is complex, returning None for now
        
        # Decoding
        d3 = self.up3(fused)
        d3 = torch.cat([d3, opt_e3], dim=1) # skip connection from optical
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, opt_e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, opt_e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.sigmoid(self.final_conv(d1))
        
        # Blend output
        reconstructed = optical * (1 - mask) + out * mask
        
        # In training, we might need the raw out and attention.
        # But we also need backward compatibility.
        if self.training or not torch.jit.is_tracing():
            return reconstructed, attention_map
        return reconstructed
