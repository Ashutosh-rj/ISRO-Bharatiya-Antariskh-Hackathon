import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

def plot_metrics_comparison_matplotlib(csv_path: str, output_path: str):
    """Generates a static bar chart comparison of models."""
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.barplot(data=df, x='Model', y='PSNR', ax=axes[0], palette='viridis')
    axes[0].set_title('PSNR (Higher is better)')
    
    sns.barplot(data=df, x='Model', y='SSIM', ax=axes[1], palette='viridis')
    axes[1].set_title('SSIM (Higher is better)')
    
    if 'LPIPS' in df.columns and not df['LPIPS'].isnull().all():
        sns.barplot(data=df, x='Model', y='LPIPS', ax=axes[2], palette='viridis')
        axes[2].set_title('LPIPS (Lower is better)')
    else:
        sns.barplot(data=df, x='Model', y='RMSE_Total', ax=axes[2], palette='viridis')
        axes[2].set_title('RMSE (Lower is better)')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_interactive_metrics_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a Plotly interactive chart for the Streamlit app."""
    # Melt dataframe for plotly express
    metrics_to_plot = ['PSNR', 'SSIM', 'RMSE_Total', 'SAM']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    df_melt = df.melt(id_vars=['Model'], value_vars=available_metrics, 
                      var_name='Metric', value_name='Score')
                      
    fig = px.bar(df_melt, x='Metric', y='Score', color='Model', barmode='group',
                 title="Model Performance Comparison",
                 color_discrete_sequence=['#A0A0A0', '#003087', '#FF6B35']) # Baseline, LaMa, SAR
                 
    fig.update_layout(template='plotly_dark')
    return fig

def plot_before_after_grid(images: List[np.ndarray], titles: List[str], output_path: str):
    """Plot a grid of images (e.g. Input, Mask, Baseline, LaMa, SAR, Target)"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        img = images[i]
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
