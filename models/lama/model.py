import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class FourierUnit(nn.Module):
    """
    Fourier Unit (FU) computes 2D Fast Fourier Transform, applies a convolution 
    in the frequency domain, and transforms back to spatial domain.
    Core innovation of Large Mask (LaMa) Inpainting.
    """
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        # Frequency domain convolution (using 1x1 conv on real/imag parts)
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2, 
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        
        # CPU-compatible FFT
        # rfft2 returns complex tensor, we view it as real/imag pairs
        ffted = torch.fft.rfft2(x, norm="backward")
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        
        # Concat real and imag parts along channel dim
        ffted_concat = torch.cat([ffted_real, ffted_imag], dim=1)
        
        # Apply 1x1 Conv in frequency domain
        ffted_conv = self.conv_layer(ffted_concat)
        ffted_conv = self.relu(ffted_conv)
        
        # Split back into real and imag
        real_part, imag_part = torch.chunk(ffted_conv, 2, dim=1)
        
        # Convert back to complex tensor
        ffted_complex = torch.complex(real_part, imag_part)
        
        # Inverse FFT to spatial domain
        output = torch.fft.irfft2(ffted_complex, s=(h, w), norm="backward")
        return output

class SpectralTransform(nn.Module):
    """
    Fast Fourier Convolution (FFC) applies both spatial and spectral convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpectralTransform, self).__init__()
        
        # Spatial branch (standard 3x3 conv)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        
        # Spectral branch (Fourier Unit)
        self.fu = FourierUnit(in_channels, out_channels // 2, groups)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Spatial branch
        x1 = self.relu(self.bn1(self.conv1(x)))
        
        # Spectral branch (global receptive field)
        x2 = self.relu(self.bn2(self.fu(x)))
        
        # Combine
        return torch.cat([x1, x2], dim=1)

class LaMaGenerator(nn.Module):
    """
    Large Mask Inpainting Generator using Fast Fourier Convolutions.
    Adapted for 4-channel input (RGB + Mask) and CPU-friendly execution.
    """
    def __init__(self, in_channels=4, out_channels=3, ngf=64, n_blocks=6):
        super(LaMaGenerator, self).__init__()
        
        # Initial Convolution
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        # FFC Residual Blocks (Bottleneck)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(SpectralTransform(ngf * 4, ngf * 4))
        self.ffc_blocks = nn.Sequential(*blocks)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Output layer
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh() # Output scaled to [-1, 1]
        )

    def forward(self, image, mask):
        """
        image: (B, 3, H, W)
        mask: (B, 1, H, W)
        """
        # Concat image and mask
        x = torch.cat([image, mask], dim=1)
        
        # Encode
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Transform (Global Receptive Field via FFT)
        x = self.ffc_blocks(x)
        
        # Decode
        x = self.up1(x)
        x = self.up2(x)
        x = self.out_conv(x)
        
        # Convert [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        
        # Blend: keep original unmasked pixels, use predicted for masked pixels
        # mask is 1 for clouds (to be replaced)
        output = image * (1 - mask) + x * mask
        return output
