#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Visualization Tool for 3WayCoT Framework

This script creates visualizations showing how metrics vary with different parameter values.
It runs the 3WayCoT framework with systematically varied parameters and collects the results.
"""

import os
import sys
import json
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import gradio as gr

# Add the project directory to the path so we can import from the main module
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from main import ThreeWayCoTApp, ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('3WayCoT.ParameterVisualization')

class ParameterVisualizer:
    """Tool to visualize how 3WayCoT metrics vary with different parameter values."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the parameter visualizer with configuration."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.results_dir = Path(self.config.get('paths.results', 'results'))
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Get parameter ranges from config
        viz_config = self.config.get('visualization.parameter_analysis', {})
        self.parameter_ranges = viz_config.get('parameter_ranges', {})
        self.parameters = viz_config.get('parameters', [])
        self.metrics_to_track = viz_config.get('metrics', [])
        
        # Create storage for results
        self.results = []
        
    def generate_parameter_combinations(self, num_combinations: int = 10) -> List[Dict[str, float]]:
        """Generate a list of parameter combinations to test."""
        param_combinations = []
        
        # For each parameter, generate values within its range
        param_values = {}
        for param in self.parameters:
            if param in self.parameter_ranges:
                param_range = self.parameter_ranges[param]
                min_val = param_range.get('min', 0.0)
                max_val = param_range.get('max', 1.0)
                step = param_range.get('step', 0.1)
                
                # Generate values
                values = np.arange(min_val, max_val + step/2, step)
                param_values[param] = values
        
        # If number of combinations is limited, sample from the full space
        full_combinations = list(itertools.product(*[param_values[p] for p in self.parameters]))
        if num_combinations < len(full_combinations):
            # Sample randomly without replacement
            indices = np.random.choice(len(full_combinations), num_combinations, replace=False)
            selected_combinations = [full_combinations[i] for i in indices]
        else:
            selected_combinations = full_combinations
        
        # Convert to dictionaries
        for combo in selected_combinations:
            param_dict = {param: value for param, value in zip(self.parameters, combo)}
            param_combinations.append(param_dict)
            
        return param_combinations
    
    def run_with_parameters(self, prompt: str, context: str, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Run the 3WayCoT framework with specific parameters and return results."""
        # Create a modified config with the specified parameters
        modified_config = self.config.copy() if hasattr(self.config, 'copy') else dict(self.config)
        
        # Update the framework parameters
        for param, value in parameters.items():
            if param in ['alpha', 'beta', 'tau', 'confidence_weight', 'max_steps']:
                # Convert max_steps to int if needed
                if param == 'max_steps' and isinstance(value, float):
                    value = int(value)
                    
                # Update in the config
                if 'framework' in modified_config:
                    modified_config['framework'][param] = value
        
        # Initialize the app with the modified config
        app = ThreeWayCoTApp(config_dict=modified_config)
        
        # Process the prompt
        result = app.process_prompt(prompt, context)
        
        # Add the parameters to the result for tracking
        result['parameters'] = parameters
        
        return result
    
    def run_parameter_exploration(self, prompt: str, context: str = "", 
                                num_combinations: int = 10) -> List[Dict[str, Any]]:
        """Run the framework with different parameter combinations and collect results."""
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(num_combinations)
        logger.info(f"Generated {len(param_combinations)} parameter combinations to test")
        
        # Run the framework with each combination
        results = []
        for i, params in enumerate(param_combinations):
            logger.info(f"Running combination {i+1}/{len(param_combinations)}: {params}")
            try:
                result = self.run_with_parameters(prompt, context, params)
                results.append(result)
                logger.info(f"Completed run with parameters: {params}")
            except Exception as e:
                logger.error(f"Error running with parameters {params}: {e}")
        
        # Store the results
        self.results = results
        return results
    
    def extract_metrics(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract key metrics from results for visualization."""
        data = []
        
        for result in results:
            # Get parameters
            params = result.get('parameters', {})
            
            # Get metrics from the result
            metrics = result.get('metrics', {})
            decision_metrics = result.get('analysis', {}).get('decisions', {}).get('summary', {})
            confidence_metrics = result.get('analysis', {}).get('confidence_metrics', {}).get('confidence_metrics', {})
            
            # Extract key metrics
            row = {
                # Parameters
                'alpha': params.get('alpha', 0),
                'beta': params.get('beta', 0),
                'tau': params.get('tau', 0),
                'confidence_weight': params.get('confidence_weight', 0),
                'max_steps': params.get('max_steps', 0),
                
                # Metrics
                'acceptance_rate': metrics.get('acceptance_rate', 0),
                'rejection_rate': metrics.get('rejection_rate', 0),
                'abstention_rate': metrics.get('abstention_rate', 0),
                'average_confidence': metrics.get('average_confidence', 0),
                
                # Decision metrics
                'accept_count': decision_metrics.get('accept_count', 0),
                'reject_count': decision_metrics.get('reject_count', 0),
                'abstain_count': decision_metrics.get('abstain_count', 0),
                'decision_consistency': decision_metrics.get('decision_consistency', 0),
                'confidence_alignment': decision_metrics.get('confidence_alignment', 0),
                
                # Confidence metrics
                'confidence_avg': confidence_metrics.get('average', 0),
                'confidence_max': confidence_metrics.get('max', 0),
                'confidence_min': confidence_metrics.get('min', 0),
                'confidence_range': confidence_metrics.get('range', 0),
                'confidence_variance': confidence_metrics.get('variance', 0),
                'high_confidence_ratio': confidence_metrics.get('high_ratio', 0),
                'low_confidence_ratio': confidence_metrics.get('low_ratio', 0),
                'confidence_trend': confidence_metrics.get('trend', 0),
            }
            
            data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    def create_heatmap(self, df: pd.DataFrame, x_param: str, y_param: str, 
                     metric: str, title: str = None) -> go.Figure:
        """Create a heatmap showing how a metric varies with two parameters."""
        # Pivot the data to create a 2D grid
        pivot_df = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            colorbar=dict(title=metric),
        ))
        
        # Set layout
        fig.update_layout(
            title=title or f"{metric} by {x_param} and {y_param}",
            xaxis_title=x_param,
            yaxis_title=y_param,
            width=800,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_3d_surface(self, df: pd.DataFrame, x_param: str, y_param: str, 
                        z_metric: str, title: str = None) -> go.Figure:
        """Create a 3D surface plot showing how a metric varies with two parameters."""
        # Pivot the data to create a 2D grid
        pivot_df = df.pivot_table(index=y_param, columns=x_param, values=z_metric, aggfunc='mean')
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            colorbar=dict(title=z_metric),
        )])
        
        # Set layout
        fig.update_layout(
            title=title or f"3D Surface of {z_metric} by {x_param} and {y_param}",
            scene=dict(
                xaxis_title=x_param,
                yaxis_title=y_param,
                zaxis_title=z_metric,
            ),
            width=800,
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def create_line_plot(self, df: pd.DataFrame, x_param: str, y_metrics: List[str], 
                        title: str = None) -> go.Figure:
        """Create a line plot showing how metrics vary with a parameter."""
        # Group by the parameter and calculate mean for each metric
        grouped_df = df.groupby(x_param)[y_metrics].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add a line for each metric
        for metric in y_metrics:
            fig.add_trace(go.Scatter(
                x=grouped_df[x_param],
                y=grouped_df[metric],
                mode='lines+markers',
                name=metric,
            ))
        
        # Set layout
        fig.update_layout(
            title=title or f"Metrics by {x_param}",
            xaxis_title=x_param,
            yaxis_title="Metric Value",
            legend_title="Metrics",
            width=800,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_parameter_parallel_plot(self, df: pd.DataFrame, params: List[str], 
                                     metrics: List[str], title: str = None) -> go.Figure:
        """Create a parallel coordinates plot showing relationships between parameters and metrics."""
        # Select columns for the plot
        columns = params + metrics
        
        # Create figure
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df['average_confidence'], colorscale='Viridis', showscale=True),
                dimensions=[
                    dict(range=[df[col].min(), df[col].max()], 
                         label=col, 
                         values=df[col]) 
                    for col in columns
                ]
            )
        )
        
        # Set layout
        fig.update_layout(
            title=title or "Parameter and Metric Relationships",
            width=1000,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> gr.Blocks:
        """Create an interactive Gradio dashboard for parameter visualization."""
        with gr.Blocks(title="3WayCoT Parameter Visualization") as dashboard:
            gr.Markdown("# 3WayCoT Framework Parameter Visualization")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(label="Prompt", lines=2, placeholder="Enter a prompt to analyze...")
                    context_input = gr.Textbox(label="Context (optional)", lines=2, placeholder="Enter any context...")
                    
                    with gr.Row():
                        num_combinations = gr.Slider(label="Number of Parameter Combinations", minimum=5, maximum=50, value=10, step=5)
                        run_button = gr.Button("Run Parameter Exploration")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Parameter Configuration"):
                            param_config = gr.DataFrame(
                                headers=["Parameter", "Min", "Max", "Step", "Include"],
                                datatype=["str", "number", "number", "number", "bool"],
                                label="Parameter Configuration"
                            )
                            
                            update_config_button = gr.Button("Update Configuration")
            
            with gr.Tabs():
                with gr.TabItem("Heatmaps"):
                    with gr.Row():
                        x_param = gr.Dropdown(label="X Parameter", choices=self.parameters)
                        y_param = gr.Dropdown(label="Y Parameter", choices=self.parameters)
                        metric = gr.Dropdown(label="Metric", choices=[
                            "acceptance_rate", "rejection_rate", "abstention_rate", 
                            "average_confidence", "decision_consistency", "confidence_alignment"
                        ])
                        
                    heatmap_plot = gr.Plot(label="Parameter Heatmap")
                    plot_heatmap_button = gr.Button("Generate Heatmap")
                    
                with gr.TabItem("3D Surfaces"):
                    with gr.Row():
                        x_param_3d = gr.Dropdown(label="X Parameter", choices=self.parameters)
                        y_param_3d = gr.Dropdown(label="Y Parameter", choices=self.parameters)
                        z_metric = gr.Dropdown(label="Z Metric", choices=[
                            "acceptance_rate", "rejection_rate", "abstention_rate", 
                            "average_confidence", "decision_consistency", "confidence_alignment"
                        ])
                        
                    surface_plot = gr.Plot(label="3D Parameter Surface")
                    plot_surface_button = gr.Button("Generate 3D Surface")
                    
                with gr.TabItem("Line Plots"):
                    with gr.Row():
                        line_x_param = gr.Dropdown(label="X Parameter", choices=self.parameters)
                        line_metrics = gr.CheckboxGroup(label="Metrics", choices=[
                            "acceptance_rate", "rejection_rate", "abstention_rate", 
                            "average_confidence", "decision_consistency", "confidence_alignment"
                        ])
                        
                    line_plot = gr.Plot(label="Parameter Line Plot")
                    plot_line_button = gr.Button("Generate Line Plot")
                    
                with gr.TabItem("Relationships"):
                    relationship_plot = gr.Plot(label="Parameter-Metric Relationships")
                    plot_relationship_button = gr.Button("Generate Relationship Plot")
                    
                with gr.TabItem("Raw Data"):
                    results_df = gr.DataFrame(label="Results Data")
                    refresh_data_button = gr.Button("Refresh Data")
            
            # Define parameter configurations from config.yml
            param_config_data = []
            for param in self.parameters:
                if param in self.parameter_ranges:
                    range_info = self.parameter_ranges[param]
                    param_config_data.append([
                        param, 
                        range_info.get('min', 0), 
                        range_info.get('max', 1), 
                        range_info.get('step', 0.1),
                        True
                    ])
            
            # Set initial values
            param_config.value = param_config_data
            x_param.value = self.parameters[0] if self.parameters else None
            y_param.value = self.parameters[1] if len(self.parameters) > 1 else None
            metric.value = "average_confidence"
            x_param_3d.value = self.parameters[0] if self.parameters else None
            y_param_3d.value = self.parameters[1] if len(self.parameters) > 1 else None
            z_metric.value = "average_confidence"
            line_x_param.value = self.parameters[0] if self.parameters else None
            line_metrics.value = ["acceptance_rate", "average_confidence"]
            
            # Define dashboard functionality
            def run_exploration(prompt, context, num_combos):
                if not prompt:
                    return gr.DataFrame.update(value=[["No results - please enter a prompt"]])
                
                self.run_parameter_exploration(prompt, context, int(num_combos))
                df = self.extract_metrics(self.results)
                return gr.DataFrame.update(value=df.to_dict('records'))
            
            def update_param_config(config_data):
                # Update parameter configuration based on user input
                for row in config_data:
                    param, min_val, max_val, step, include = row
                    if include and param in self.parameter_ranges:
                        self.parameter_ranges[param] = {
                            'min': float(min_val),
                            'max': float(max_val),
                            'step': float(step)
                        }
                
                # Update parameters list based on included parameters
                self.parameters = [row[0] for row in config_data if row[4]]
                
                return gr.Dropdown.update(choices=self.parameters), gr.Dropdown.update(choices=self.parameters), \
                       gr.Dropdown.update(choices=self.parameters), gr.Dropdown.update(choices=self.parameters), \
                       gr.Dropdown.update(choices=self.parameters)
            
            def generate_heatmap(x, y, m):
                if not self.results:
                    return gr.Plot.update(value=None)
                
                df = self.extract_metrics(self.results)
                fig = self.create_heatmap(df, x, y, m)
                return gr.Plot.update(value=fig)
            
            def generate_surface(x, y, z):
                if not self.results:
                    return gr.Plot.update(value=None)
                
                df = self.extract_metrics(self.results)
                fig = self.create_3d_surface(df, x, y, z)
                return gr.Plot.update(value=fig)
            
            def generate_line_plot(x, metrics_list):
                if not self.results:
                    return gr.Plot.update(value=None)
                
                df = self.extract_metrics(self.results)
                fig = self.create_line_plot(df, x, metrics_list)
                return gr.Plot.update(value=fig)
            
            def generate_relationship_plot():
                if not self.results:
                    return gr.Plot.update(value=None)
                
                df = self.extract_metrics(self.results)
                fig = self.create_parameter_parallel_plot(
                    df, 
                    self.parameters, 
                    ["acceptance_rate", "rejection_rate", "abstention_rate", "average_confidence", 
                     "decision_consistency", "confidence_alignment"]
                )
                return gr.Plot.update(value=fig)
            
            def refresh_results_data():
                if not self.results:
                    return gr.DataFrame.update(value=[["No results available"]])
                
                df = self.extract_metrics(self.results)
                return gr.DataFrame.update(value=df.to_dict('records'))
            
            # Connect event handlers
            run_button.click(run_exploration, inputs=[prompt_input, context_input, num_combinations], outputs=[results_df])
            update_config_button.click(update_param_config, inputs=[param_config], 
                                    outputs=[x_param, y_param, x_param_3d, y_param_3d, line_x_param])
            plot_heatmap_button.click(generate_heatmap, inputs=[x_param, y_param, metric], outputs=[heatmap_plot])
            plot_surface_button.click(generate_surface, inputs=[x_param_3d, y_param_3d, z_metric], outputs=[surface_plot])
            plot_line_button.click(generate_line_plot, inputs=[line_x_param, line_metrics], outputs=[line_plot])
            plot_relationship_button.click(generate_relationship_plot, inputs=[], outputs=[relationship_plot])
            refresh_data_button.click(refresh_results_data, inputs=[], outputs=[results_df])
            
            # Also update plots when running exploration
            run_button.click(generate_heatmap, inputs=[x_param, y_param, metric], outputs=[heatmap_plot])
            run_button.click(generate_surface, inputs=[x_param_3d, y_param_3d, z_metric], outputs=[surface_plot])
            run_button.click(generate_line_plot, inputs=[line_x_param, line_metrics], outputs=[line_plot])
            run_button.click(generate_relationship_plot, inputs=[], outputs=[relationship_plot])
            
        return dashboard
    
    def launch_dashboard(self, share: bool = False):
        """Launch the Gradio dashboard."""
        dashboard = self.create_interactive_dashboard()
        dashboard.launch(share=share)
        
def main():
    """Main entry point for the parameter visualization tool."""
    parser = argparse.ArgumentParser(description="3WayCoT Parameter Visualization Tool")
    
    # Input parameters
    parser.add_argument("--prompt", type=str, 
                      help="Input prompt to analyze")
    parser.add_argument("--context", type=str, default="", 
                      help="Additional context for the prompt")
    parser.add_argument("--config", type=str, 
                      help="Path to custom config file")
    parser.add_argument("--combinations", type=int, default=10,
                      help="Number of parameter combinations to test")
    parser.add_argument("--output", type=str, default="results/parameter_analysis.json",
                      help="Output file path for results")
    parser.add_argument("--dashboard", action="store_true",
                      help="Launch interactive dashboard")
    parser.add_argument("--share", action="store_true",
                      help="Share the dashboard publicly")
    
    args = parser.parse_args()
    
    try:
        # Initialize the parameter visualizer
        visualizer = ParameterVisualizer(args.config)
        
        # Either run with command line args or launch dashboard
        if args.dashboard:
            logger.info("Launching interactive dashboard")
            visualizer.launch_dashboard(share=args.share)
        elif args.prompt:
            # Run parameter exploration
            results = visualizer.run_parameter_exploration(
                args.prompt, args.context, args.combinations
            )
            
            # Extract metrics and save results
            df = visualizer.extract_metrics(results)
            
            # Save to CSV and JSON
            output_path = Path(args.output)
            csv_path = output_path.with_suffix('.csv')
            
            df.to_csv(csv_path, index=False)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved results to {output_path} and {csv_path}")
            
            # Create and save visualizations
            output_dir = output_path.parent / "visualizations"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create basic visualizations
            for param in visualizer.parameters:
                metrics = ["acceptance_rate", "rejection_rate", "abstention_rate", "average_confidence"]
                fig = visualizer.create_line_plot(df, param, metrics)
                fig.write_html(output_dir / f"{param}_metrics.html")
            
            # Create relationship visualization
            fig = visualizer.create_parameter_parallel_plot(
                df, 
                visualizer.parameters, 
                ["acceptance_rate", "rejection_rate", "abstention_rate", "average_confidence"]
            )
            fig.write_html(output_dir / "parameter_relationships.html")
            
            logger.info(f"Saved visualizations to {output_dir}")
        else:
            logger.info("No prompt provided. Use --prompt or --dashboard")
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error in parameter visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
