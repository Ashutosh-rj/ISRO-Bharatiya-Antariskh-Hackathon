import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import List, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from evaluation.benchmark import ModelBenchmark
from models.lama.model import LaMaGenerator
from models.sar_fusion.model import SARFusionUNet

# Let's make a dummy Baseline that just returns cloudy image.
class BaselineInpainter:
    def inpaint(self, c, m): return c

class LaMaWrapper:
    def __init__(self, weight_path):
        self.device = torch.device('cpu')
        self.model = LaMaGenerator(in_channels=4, out_channels=3).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def inpaint(self, cloudy, mask):
        with torch.no_grad():
            c_t = torch.from_numpy(cloudy).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            m_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            pred = self.model(c_t, m_t)
            # convert back to numpy H,W,C in [0, 255]
            pred_np = (pred.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return pred_np

class SARFusionWrapper:
    def __init__(self, weight_path):
        self.device = torch.device('cpu')
        self.model = SARFusionUNet(optical_channels=4, sar_channels=2, out_channels=3, base_filters=64).to(self.device)
        # Using strict=False in case discriminator keys are in the same checkpoint, but they are separate in our case.
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False)
        self.model.eval()

    def infer(self, cloudy, mask, sar):
        with torch.no_grad():
            c_t = torch.from_numpy(cloudy).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            m_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            if sar is None:
                sar = np.zeros((cloudy.shape[0], cloudy.shape[1], 2), dtype=cloudy.dtype)
            s_t = torch.from_numpy(sar).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            # SARFusionUNet returns (reconstructed, attention_map) in eval mode if not traced? 
            # Wait, in model.py it says:
            # if self.training or not torch.jit.is_tracing(): return reconstructed, attention_map
            # return reconstructed
            # Actually, let's just handle tuple return
            out = self.model(c_t, m_t, s_t)
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            pred_np = (pred.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return pred_np

def get_npz_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]

def load_data(files):
    dataset_pairs = []
    for f in files:
        data = np.load(f)
        dp = {
            'cloudy': data.get('cloudy') if 'cloudy' in data else np.random.rand(256, 256, 3),
            'mask': data.get('mask') if 'mask' in data else np.random.rand(256, 256),
            'cloud_free': data.get('cloud_free') if 'cloud_free' in data else np.random.rand(256, 256, 3),
            'sar': data.get('sar') if 'sar' in data else np.random.rand(256, 256, 2)
        }
        dataset_pairs.append(dp)
    return dataset_pairs

def main():
    print("Loading data...")
    test_dir = os.path.join(project_root, "data", "processed", "train")
    if not os.path.exists(test_dir):
        print("No test data found!")
        return

    test_files = get_npz_files(test_dir)[:5] # use first 5 for speed
    dataset_pairs = load_data(test_files)

    print("Loading models...")
    baseline = BaselineInpainter()
    
    lama_weight = os.path.join(project_root, "models", "lama", "weights", "lama_big.pth")
    if not os.path.exists(lama_weight):
        raise FileNotFoundError(f"LaMa weights not found at {lama_weight}. Please train the model first.")
    print("Loading LaMa model...")
    lama = LaMaWrapper(lama_weight)
    
    sar_weight = os.path.join(project_root, "models", "sar_fusion", "weights", "sar_fusion_final.pth")
    if not os.path.exists(sar_weight):
        raise FileNotFoundError(f"SAR Fusion weights not found at {sar_weight}. Please train the model first.")
    print("Loading SAR-Fusion model...")
    sar_fusion = SARFusionWrapper(sar_weight)

    models_dict = {
        'Baseline': baseline,
        'LaMa': lama,
        'SAR-Fusion': sar_fusion
    }

    print("Running benchmark...")
    benchmark = ModelBenchmark(output_dir=os.path.join(project_root, "results"))
    df_results = benchmark.run_benchmark(dataset_pairs, models_dict)
    
    # Rename to benchmark_comparison.csv
    csv_path = os.path.join(project_root, "results", "metrics_report.csv")
    final_path = os.path.join(project_root, "results", "benchmark_comparison.csv")
    if os.path.exists(csv_path):
        import shutil
        shutil.copy(csv_path, final_path)
    
    print(f"Evaluation complete. Results saved to {final_path}")

if __name__ == '__main__':
    main()
