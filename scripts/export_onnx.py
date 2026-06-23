import torch
import os
import sys

# Ensure models are in path
sys.path.append(os.path.abspath('.'))

from models.lama.model import LaMaGenerator
from models.sar_fusion.model import SARFusionUNet

def export_lama():
    print("Exporting LaMa to ONNX...")
    model = LaMaGenerator()
    pth_path = 'models/lama/weights/lama_big.pth'
    if not os.path.exists(pth_path):
        print(f"Skipping LaMa, {pth_path} not found.")
        return
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    image = torch.randn(1, 3, 512, 512)
    mask = torch.randn(1, 1, 512, 512)
    
    torch.onnx.export(
        model, 
        (image, mask), 
        'models/lama/weights/lama_big.onnx',
        input_names=['image', 'mask'],
        output_names=['output'],
        dynamic_axes={'image': {2: 'height', 3: 'width'}, 'mask': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    print("LaMa ONNX export successful.")

def export_sar():
    print("Exporting SAR-Fusion to ONNX...")
    model = SARFusionUNet(base_filters=64)
    pth_path = 'models/sar_fusion/weights/sar_fusion_final.pth'
    if not os.path.exists(pth_path):
        print(f"Skipping SAR-Fusion, {pth_path} not found.")
        return
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    optical = torch.randn(1, 3, 512, 512)
    mask = torch.randn(1, 1, 512, 512)
    sar = torch.randn(1, 2, 512, 512)
    
    torch.onnx.export(
        model, 
        (optical, mask, sar), 
        'models/sar_fusion/weights/sar_fusion.onnx',
        input_names=['optical', 'mask', 'sar'],
        output_names=['output'],
        dynamic_axes={
            'optical': {2: 'height', 3: 'width'}, 
            'mask': {2: 'height', 3: 'width'}, 
            'sar': {2: 'height', 3: 'width'}, 
            'output': {2: 'height', 3: 'width'}
        }
    )
    print("SAR-Fusion ONNX export successful.")

if __name__ == "__main__":
    export_lama()
    export_sar()
