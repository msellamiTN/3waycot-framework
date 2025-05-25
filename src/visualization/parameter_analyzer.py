"""
Parameter sensitivity analysis for 3WayCoT framework.

This module provides tools to analyze how different parameter settings
affect the decision-making process and outcomes.
"""

from typing import Dict, List, Any, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

class ParameterAnalyzer:
    """Analyze the impact of different parameters on decision outcomes."""
    
    def __init__(self, experiment_data: List[Dict[str, Any]]):
        """Initialize with experimental data containing parameter variations.
        
        Args:
            experiment_data: List of experiment records with parameter settings and outcomes
        """
        self.experiment_data = experiment_data
        self.df = pd.DataFrame(experiment_data)
    
    def parameter_sensitivity_analysis(self, 
                                     target_metric: str,
                                     parameters: Optional[List[str]] = None,
                                     output_file: Optional[str] = None,
                                     show: bool = True) -> go.Figure:
        """Perform sensitivity analysis of parameters on a target metric.
        
        Args:
            target_metric: Name of the metric to analyze
            parameters: List of parameter names to include in analysis
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        if target_metric not in self.df.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in data")
            
        if parameters is None:
            # Exclude non-parameter columns
            exclude_cols = [target_metric, 'run_id', 'timestamp', 'strategy', 'use_case']
            parameters = [col for col in self.df.columns 
                        if col not in exclude_cols and self.df[col].nunique() > 1]
        
        # Prepare data for analysis
        X = pd.get_dummies(self.df[parameters])
        y = self.df[target_metric]
        
        # Train a random forest model to estimate feature importance
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create a DataFrame with importance scores
        importance_df = pd.DataFrame({
            'parameter': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        # Create the visualization
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=importance_df['importance'],
                y=importance_df['parameter'],
                error_x=dict(
                    type='data',
                    array=importance_df['std'],
                    visible=True
                ),
                orientation='h',
                name='Parameter Importance'
            )
        )
        
        fig.update_layout(
            title=f"Parameter Sensitivity for {target_metric}",
            xaxis_title="Importance (permutation)",
            yaxis_title="Parameter",
            showlegend=False,
            height=400 + 20 * len(parameters)
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
    
    def parameter_impact_heatmap(self, 
                               metrics: List[str],
                               parameters: Optional[List[str]] = None,
                               output_file: Optional[str] = None,
                               show: bool = True) -> go.Figure:
        """Visualize the impact of parameters on multiple metrics using a heatmap.
        
        Args:
            metrics: List of metric names to analyze
            parameters: List of parameter names to include in analysis
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        valid_metrics = [m for m in metrics if m in self.df.columns]
        if not valid_metrics:
            raise ValueError("No valid metrics found in the provided list")
            
        if parameters is None:
            # Exclude non-parameter columns
            exclude_cols = valid_metrics + ['run_id', 'timestamp', 'strategy', 'use_case']
            parameters = [col for col in self.df.columns 
                        if col not in exclude_cols and self.df[col].nunique() > 1]
        
        # Calculate correlation between parameters and metrics
        corr_data = []
        for param in parameters:
            for metric in valid_metrics:
                # Use point-biserial correlation for binary parameters
                if self.df[param].nunique() == 2:
                    from scipy.stats import pointbiserialr
                    corr, _ = pointbiserialr(self.df[param], self.df[metric])
                else:
                    corr = self.df[param].corr(self.df[metric])
                corr_data.append({'parameter': param, 'metric': metric, 'correlation': corr})
        
        corr_df = pd.DataFrame(corr_data)
        
        # Pivot for heatmap
        heatmap_data = corr_df.pivot(index='parameter', columns='metric', values='correlation')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Metric", y="Parameter", color="Correlation"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title="Parameter Impact on Metrics"
        )
        
        # Add annotations
        for i, row in enumerate(heatmap_data.index):
            for j, col in enumerate(heatmap_data.columns):
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{heatmap_data.loc[row, col]:.2f}",
                    showarrow=False,
                    font=dict(color='black' if abs(heatmap_data.loc[row, col]) < 0.5 else 'white')
                )
        
        fig.update_layout(
            xaxis_title="Metrics",
            yaxis_title="Parameters",
            height=200 + 30 * len(parameters),
            width=200 + 100 * len(valid_metrics)
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
    
    def parameter_interaction_analysis(self,
                                     param1: str,
                                     param2: str,
                                     target_metric: str,
                                     output_file: Optional[str] = None,
                                     show: bool = True) -> go.Figure:
        """Analyze the interaction effect between two parameters on a target metric.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            target_metric: Name of the target metric
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        if target_metric not in self.df.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in data")
            
        for param in [param1, param2]:
            if param not in self.df.columns:
                raise ValueError(f"Parameter '{param}' not found in data")
        
        # Create pivot table for heatmap
        pivot_data = self.df.pivot_table(
            values=target_metric,
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x=param2, y=param1, color=target_metric),
            x=pivot_data.columns,
            y=pivot_data.index,
            aspect="auto",
            color_continuous_scale='Viridis',
            title=f"Interaction Effect: {param1} x {param2} on {target_metric}"
        )
        
        # Add annotations
        for i, row in enumerate(pivot_data.index):
            for j, col in enumerate(pivot_data.columns):
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{pivot_data.loc[row, col]:.2f}",
                    showarrow=False,
                    font=dict(color='white' if pivot_data.loc[row, col] < pivot_data.values.mean() else 'black')
                )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
