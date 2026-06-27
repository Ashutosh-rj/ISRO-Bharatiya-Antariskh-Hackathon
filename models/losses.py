import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper loss.
    Measures the spectral angle between the reconstructed and true pixel vectors.
    Particularly useful for multispectral remote sensing data (e.g. NIR fidelity).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        """
        pred, target: (B, C, H, W)
        mask: (B, 1, H, W) binary mask where 1 means clouded pixel (requires reconstruction)
        """
        # Select strictly the masked pixel vectors to prevent unmasked zero regions from inflating cos_theta
        mask_bool = mask.squeeze(1) > 0 # (B, H, W)
        if not mask_bool.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_pixels = pred.permute(0, 2, 3, 1)[mask_bool] # (N_valid, C)
        target_pixels = target.permute(0, 2, 3, 1)[mask_bool] # (N_valid, C)
        
        # Compute dot product across channels (dim=1)
        dot = torch.sum(pred_pixels * target_pixels, dim=1)
        
        # Compute magnitudes safely (sqrt(sum(x^2) + eps^2))
        pred_norm = torch.sqrt(torch.sum(pred_pixels**2, dim=1) + self.eps**2)
        target_norm = torch.sqrt(torch.sum(target_pixels**2, dim=1) + self.eps**2)
        
        # Calculate angle with strict float32 numerical bounds (-0.9999, 0.9999) to prevent acos backward NaN
        cos_theta = (dot / (pred_norm * target_norm)).clamp(-0.9999, 0.9999)
        sam = torch.acos(cos_theta)
        
        return sam.mean()

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features via the LPIPS library.
    
    APPROXIMATION DISCLAIMER:
    LISS-IV multispectral radiometry captures Green (520-590nm), Red (620-680nm), and NIR (770-860nm).
    Because VGG networks are trained exclusively on natural-image RGB statistics (ImageNet), mapping
    Green -> R, Red -> G, and NIR -> B creates a pseudo-RGB transfer approximation. While effective for
    spatial edge and texture regularity matching, this introduces a known spectral domain shift with no
    absolute radiometric fidelity guarantee.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.active = False
        try:
            self.loss_fn = lpips.LPIPS(net='vgg').to(device)
            for param in self.loss_fn.parameters():
                param.requires_grad = False
            self.active = True
        except Exception as e:
            print(f"Warning: Could not load LPIPS VGG weights (likely offline). Perceptual loss disabled. Error: {e}")

    def forward(self, pred, target):
        if not self.active:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        # LPIPS expects inputs in [-1, 1], our models output [0, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        
        # Band adaptation for LISS-IV (Green, Red, NIR) -> pseudo-RGB for VGG feature extractor
        pred_pseudo_rgb = torch.cat([
            pred_scaled[:, 0:1, :, :], # Green -> R
            pred_scaled[:, 1:2, :, :], # Red -> G
            pred_scaled[:, 2:3, :, :]  # NIR -> B
        ], dim=1)
        
        target_pseudo_rgb = torch.cat([
            target_scaled[:, 0:1, :, :],
            target_scaled[:, 1:2, :, :],
            target_scaled[:, 2:3, :, :]
        ], dim=1)
        
        # Downsample to 64x64 for 16x CPU speedup during VGG feature extraction
        pred_64 = F.interpolate(pred_pseudo_rgb, size=(64, 64), mode='bilinear', align_corners=False)
        target_64 = F.interpolate(target_pseudo_rgb, size=(64, 64), mode='bilinear', align_corners=False)
        
        loss = self.loss_fn(pred_64, target_64)
        return loss.mean()

class GANLoss(nn.Module):
    """
    Adversarial loss for Conditional GAN.
    Uses BCEWithLogitsLoss for numerical stability.
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class CombinedLoss(nn.Module):
    """
    Multi-component loss: L1 (Pixel) + SAM (Spectral) + Perceptual (Feature) + GAN (Adversarial)
    """
    def __init__(self, l1_weight=10.0, sam_weight=2.0, perceptual_weight=1.0, adv_weight=0.1, device='cpu'):
        super().__init__()
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        
        self.l1 = nn.L1Loss()
        if sam_weight > 0:
            self.sam = SAMLoss()
        if perceptual_weight > 0:
            self.perceptual = PerceptualLoss(device=device)

    def forward(self, pred, target, mask, pred_fake_logits=None):
        # L1 only on reconstructed regions
        loss_l1 = self.l1(pred * mask, target * mask)
        total_loss = self.l1_weight * loss_l1
        
        metrics = {'L1': loss_l1.item()}
        
        if self.sam_weight > 0:
            loss_sam = self.sam(pred, target, mask)
            total_loss += self.sam_weight * loss_sam
            metrics['SAM'] = loss_sam.item()
            
        if self.perceptual_weight > 0:
            loss_perc = self.perceptual(pred, target)
            total_loss += self.perceptual_weight * loss_perc
            metrics['Perceptual'] = loss_perc.item()
            
        if self.adv_weight > 0 and pred_fake_logits is not None:
            # Generator wants discriminator to think fake is real (1.0)
            target_tensor = torch.ones_like(pred_fake_logits)
            loss_adv = F.binary_cross_entropy_with_logits(pred_fake_logits, target_tensor)
            total_loss += self.adv_weight * loss_adv
            metrics['Adv'] = loss_adv.item()
            
        return total_loss, metrics
