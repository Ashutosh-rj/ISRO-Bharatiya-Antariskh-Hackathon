import os
import json
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class ModelBenchmark:
    """
    Framework to run standard benchmark across Baseline, LaMa, and SAR-Fusion models.
    """
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.metrics_calc = MetricsCalculator()
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_benchmark(self, dataset_pairs: List[Dict[str, np.ndarray]], models_dict: Dict) -> pd.DataFrame:
        """
        Runs benchmark across data pairs categorized into cloud-coverage buckets.
        """
        all_sample_metrics = []
        
        for i, data in enumerate(dataset_pairs):
            cloudy = data['cloudy']
            mask = data['mask']
            target = data['cloud_free']
            sar = data.get('sar')
            
            # Determine cloud coverage percentage bucket
            coverage = (mask > 0).mean() * 100.0
            if coverage < 20.0:
                bucket = "Thin (<20%)"
            elif coverage < 50.0:
                bucket = "Moderate (20-50%)"
            else:
                bucket = "Thick (>50%)"
                
            for model_name, model in models_dict.items():
                if model_name == 'SAR-Fusion':
                    pred = model.infer(cloudy, mask, sar)
                elif hasattr(model, 'inpaint'):
                    pred = model.inpaint(cloudy, mask)
                elif hasattr(model, '__call__'):
                    # PyTorch / nn.Module callable
                    c_t = torch.from_numpy(cloudy).float().permute(2,0,1).unsqueeze(0)/255.0
                    m_t = torch.from_numpy((mask>0).astype(np.float32)).permute(2,0,1)
                    if m_t.ndim == 3 and m_t.shape[0] != 1:
                        m_t = m_t[:1]
                    m_t = m_t.unsqueeze(0)
                    with torch.no_grad():
                        out = model(c_t, m_t)
                    pred = np.clip(out[0].permute(1,2,0).numpy() * 255.0, 0, 255).astype(np.uint8)
                else:
                    pred = cloudy

                metrics = self.metrics_calc.evaluate(pred, target)
                metrics['Model'] = model_name
                metrics['Bucket'] = bucket
                metrics['Sample'] = i
                all_sample_metrics.append(metrics)
                
        df_all = pd.DataFrame(all_sample_metrics)
        
        # Aggregate per Model and Bucket
        bucket_summary = df_all.groupby(['Model', 'Bucket']).mean(numeric_only=True).reset_index()
        bucket_summary.drop(columns=['Sample'], inplace=True, errors='ignore')
        
        # Aggregate Overall per Model
        overall_summary = df_all.groupby(['Model']).mean(numeric_only=True).reset_index()
        overall_summary['Bucket'] = 'Overall (0-100%)'
        overall_summary.drop(columns=['Sample'], inplace=True, errors='ignore')
        
        df_report = pd.concat([overall_summary, bucket_summary], ignore_index=True)
        self._save_results(df_report)
        
        return df_report
        
    def _save_results(self, df: pd.DataFrame):
        csv_path = os.path.join(self.output_dir, "metrics_report.csv")
        json_path = os.path.join(self.output_dir, "metrics_report.json")
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=4)
        logger.info(f"Saved benchmark results to {csv_path}")
