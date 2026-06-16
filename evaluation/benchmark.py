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
        Runs benchmark on a list of data pairs.
        dataset_pairs: list of dicts with 'cloudy', 'mask', 'cloud_free', 'sar' (optional)
        models_dict: {'Baseline': baseline_model_obj, 'LaMa': lama_obj, ...}
        """
        results = []
        
        for model_name, model in models_dict.items():
            logger.info(f"Benchmarking model: {model_name}")
            
            model_metrics = []
            for i, data in enumerate(dataset_pairs):
                cloudy = data['cloudy']
                mask = data['mask']
                target = data['cloud_free']
                
                # Inference
                if model_name == 'SAR-Fusion':
                    pred = model.infer(cloudy, mask, data.get('sar'))
                else:
                    pred = model.inpaint(cloudy, mask) # adapt to actual inference call
                
                # Evaluate
                metrics = self.metrics_calc.evaluate(pred, target)
                metrics['Model'] = model_name
                metrics['Sample'] = i
                model_metrics.append(metrics)
                
            # Aggregate for model
            df_model = pd.DataFrame(model_metrics)
            avg_metrics = df_model.drop(columns=['Sample', 'Model']).mean().to_dict()
            avg_metrics['Model'] = model_name
            results.append(avg_metrics)
            
        df_results = pd.DataFrame(results)
        self._save_results(df_results)
        
        return df_results
        
    def _save_results(self, df: pd.DataFrame):
        csv_path = os.path.join(self.output_dir, "metrics_report.csv")
        json_path = os.path.join(self.output_dir, "metrics_report.json")
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=4)
        logger.info(f"Saved benchmark results to {csv_path}")
