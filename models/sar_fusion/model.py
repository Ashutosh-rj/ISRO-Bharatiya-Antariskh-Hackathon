import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CrossAttentionFusion(nn.Module):
    """Lightweight cross-attention for fusing SAR features into Optical features"""
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, opt_feat, sar_feat):
        B, C, H, W = opt_feat.size()
        
        # Query from Optical
        proj_query = self.query_conv(opt_feat).view(B, -1, H * W).permute(0, 2, 1) # B, N, C'
        # Key from SAR
        proj_key = self.key_conv(sar_feat).view(B, -1, H * W) # B, C', N
        
        # Attention map
        energy = torch.bmm(proj_query, proj_key) # B, N, N
        attention = F.softmax(energy, dim=-1)
        
        # Value from SAR
        proj_value = self.value_conv(sar_feat).view(B, -1, H * W) # B, C, N
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection
        return opt_feat + self.gamma * out

class SARFusionUNet(nn.Module):
    """
    Dual-Encoder U-Net architecture.
    Encoder A: Optical + Mask (4ch)
    Encoder B: SAR VV+VH (2ch)
    """
    def __init__(self, optical_channels=4, sar_channels=2, out_channels=3, base_filters=64):
        super().__init__()
        
        # Optical Encoder
        self.opt_enc1 = ConvBlock(optical_channels, base_filters)
        self.opt_pool1 = nn.MaxPool2d(2)
        self.opt_enc2 = ConvBlock(base_filters, base_filters*2)
        self.opt_pool2 = nn.MaxPool2d(2)
        self.opt_enc3 = ConvBlock(base_filters*2, base_filters*4)
        self.opt_pool3 = nn.MaxPool2d(2)
        
        # SAR Encoder
        self.sar_enc1 = ConvBlock(sar_channels, base_filters)
        self.sar_pool1 = nn.MaxPool2d(2)
        self.sar_enc2 = ConvBlock(base_filters, base_filters*2)
        self.sar_pool2 = nn.MaxPool2d(2)
        self.sar_enc3 = ConvBlock(base_filters*2, base_filters*4)
        self.sar_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck & Fusion
        self.opt_bottleneck = ConvBlock(base_filters*4, base_filters*8)
        self.sar_bottleneck = ConvBlock(base_filters*4, base_filters*8)
        self.fusion = CrossAttentionFusion(base_filters*8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters*8, base_filters*4) # + skip connection
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters*4, base_filters*2)
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters*2, base_filters)
        
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, optical, mask, sar):
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
        
        # Fusion at bottleneck
        fused = self.fusion(opt_b, sar_b)
        
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
        return optical * (1 - mask) + out * mask
