"""
Confidence impact analysis for 3WayCoT framework.

This module provides tools to analyze how confidence values influence
the decision-making process and final outcomes.
"""

from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class ConfidenceAnalyzer:
    """Analyze the impact of confidence values on decision outcomes."""
    
    def __init__(self, decision_data: List[Dict[str, Any]]):
        """Initialize with decision data containing confidence values.
        
        Args:
            decision_data: List of decision records with confidence metrics
        """
        self.decision_data = decision_data
        self.df = pd.DataFrame(decision_data)
    
    def confidence_distribution(self, 
                              output_file: Optional[str] = None,
                              show: bool = True) -> go.Figure:
        """Visualize the distribution of confidence values.
        
        Args:
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        fig = px.histogram(
            self.df, 
            x='confidence',
            nbins=20,
            title='Distribution of Confidence Values',
            labels={'confidence': 'Confidence Score'},
            marginal='box',
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            showlegend=False
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
    
    def confidence_decision_relationship(self, 
                                        decision_field: str = 'final_decision',
                                        output_file: Optional[str] = None,
                                        show: bool = True) -> go.Figure:
        """Visualize the relationship between confidence and final decisions.
        
        Args:
            decision_field: Name of the field containing decision categories
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        if decision_field not in self.df.columns:
            raise ValueError(f"Decision field '{decision_field}' not found in data")
            
        fig = px.violin(
            self.df,
            x=decision_field,
            y='confidence',
            box=True,
            points="all",
            title=f'Confidence Distribution by {decision_field.capitalize()}',
            labels={'confidence': 'Confidence Score'}
        )
        
        fig.update_layout(
            xaxis_title="Decision Category",
            yaxis_title="Confidence Score",
            showlegend=False
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
    
    def confidence_impact_on_metrics(self, 
                                    metrics: List[str],
                                    decision_field: str = 'final_decision',
                                    output_file: Optional[str] = None,
                                    show: bool = True) -> go.Figure:
        """Analyze how confidence impacts various performance metrics.
        
        Args:
            metrics: List of metric names to analyze
            decision_field: Name of the field containing decision categories
            output_file: Path to save the visualization (HTML)
            show: Whether to display the visualization
            
        Returns:
            plotly Figure object
        """
        valid_metrics = [m for m in metrics if m in self.df.columns]
        if not valid_metrics:
            raise ValueError("No valid metrics found in the provided list")
            
        # Create subplots
        fig = make_subplots(
            rows=len(valid_metrics), 
            cols=1,
            subplot_titles=[f"Confidence vs {m.capitalize()}" for m in valid_metrics]
        )
        
        for i, metric in enumerate(valid_metrics, 1):
            # Scatter plot with trendline
            scatter = px.scatter(
                self.df,
                x='confidence',
                y=metric,
                color=decision_field,
                trendline="lowess",
                title=f"Confidence vs {metric.capitalize()}",
                labels={'confidence': 'Confidence Score'}
            )
            
            # Add traces to subplot
            for trace in scatter.data:
                fig.add_trace(trace, row=i, col=1)
                
            # Update y-axis labels
            fig.update_yaxes(title_text=metric, row=i, col=1)
        
        # Update layout
        fig.update_layout(
            height=300 * len(valid_metrics),
            showlegend=True,
            title_text="Impact of Confidence on Performance Metrics"
        )
        
        if output_file:
            fig.write_html(output_file)
            
        if show:
            fig.show()
            
        return fig
