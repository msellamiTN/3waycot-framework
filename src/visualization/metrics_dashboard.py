"""
Comparative metrics dashboard for 3WayCoT framework.

This module provides tools to visualize and compare different metrics
across multiple decision-making strategies and scenarios.
"""

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class MetricsDashboard:
    """Interactive dashboard for comparing decision metrics across scenarios."""
    
    def __init__(self, metrics_data: List[Dict[str, Any]]):
        """Initialize with metrics data from multiple runs.
        
        Args:
            metrics_data: List of dictionaries containing metrics from different runs
        """
        self.metrics_data = metrics_data
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert metrics data to a pandas DataFrame for analysis."""
        return pd.DataFrame(self.metrics_data)
    
    def compare_metrics(self, metrics_to_compare: List[str], 
                       group_by: str = 'strategy',
                       output_file: Optional[str] = None,
                       show: bool = True) -> go.Figure:
        """Generate a comparison of multiple metrics across different groups.
        
        Args:
            metrics_to_compare: List of metric names to compare
            group_by: Column to group results by (e.g., 'strategy', 'use_case')
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        if not metrics_to_compare:
            metrics_to_compare = [col for col in self.df.columns 
                               if col not in [group_by, 'run_id', 'timestamp']]
        
        fig = make_subplots(
            rows=len(metrics_to_compare), 
            cols=1,
            subplot_titles=metrics_to_compare
        )
        
        for i, metric in enumerate(metrics_to_compare, 1):
            if metric not in self.df.columns:
                continue
                
            # Group data and calculate statistics
            grouped = self.df.groupby(group_by)[metric].agg(['mean', 'std']).reset_index()
            
            # Add bar chart for each metric
            fig.add_trace(
                go.Bar(
                    x=grouped[group_by],
                    y=grouped['mean'],
                    error_y=dict(
                        type='data',
                        array=grouped['std'],
                        visible=True
                    ),
                    name=f"{metric} (mean ± std)",
                    text=grouped['mean'].round(3),
                    textposition='auto',
                ),
                row=i, col=1
            )
            
            fig.update_yaxes(title_text=metric, row=i, col=1)
        
        fig.update_layout(
            title_text=f"Metrics Comparison by {group_by.capitalize()}",
            height=300 * len(metrics_to_compare),
            showlegend=False,
            hovermode='closest',
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
    
    def correlation_heatmap(self, metrics: Optional[List[str]] = None,
                           output_file: Optional[str] = None,
                           show: bool = True) -> go.Figure:
        """Generate a heatmap showing correlations between different metrics.
        
        Args:
            metrics: List of metrics to include in the correlation analysis
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        if metrics is None:
            metrics = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                     if col not in ['run_id']]
        
        corr = self.df[metrics].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size":10}
        ))
        
        fig.update_layout(
            title='Metrics Correlation Heatmap',
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            height=600,
            width=800,
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
