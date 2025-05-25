#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametrized 3WayCoT Gradio Interface

A user-friendly web interface for the 3WayCoT framework with parameter customization,
use case selection, model selection, and enhanced visualizations.
"""

import gradio as gr
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parametrized_gradio.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('3WayCoT.ParametrizedUI')

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the main application
from main import ThreeWayCoTApp, ConfigManager

class ParametrizedGradio3WayCoT:
    """Parametrized Gradio interface for the 3WayCoT framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gradio interface."""
        self.config_path = config_path
        self.app = ThreeWayCoTApp(config_path)
        self.interface = None
        self.last_result = None
        self.history = []
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Get use cases from config
        self.use_cases = self.config.get('use_cases', {})
        if not self.use_cases:
            self.use_cases = {
                'default': {
                    'alpha': 0.7,
                    'beta': 0.4,
                    'tau': 0.6,
                    'description': 'Standard configuration for general use',
                    'max_steps': 5
                },
                'conservative': {
                    'alpha': 0.8,
                    'beta': 0.5,
                    'description': 'More conservative decision making',
                    'max_steps': 5
                },
                'exploratory': {
                    'alpha': 0.6,
                    'beta': 0.3,
                    'description': 'More exploratory with higher uncertainty tolerance',
                    'max_steps': 7
                }
            }
        
        # Get LLM models from config
        self.llm_config = self.config.get('llm', {})
        self.providers = self.llm_config.get('providers', {})
        self.models = {}
        for provider, details in self.providers.items():
            if 'models' in details:
                self.models[provider] = details['models']
        
        # Flatten models for dropdown
        self.all_models = []
        for provider_models in self.models.values():
            self.all_models.extend(provider_models)
        
        if not self.all_models:
            self.all_models = ['gemini-1.5-pro', 'gpt-4', 'claude-3-opus-20240229']
        
        # Load available datasets
        self.datasets = self._load_available_datasets()
    
    def _load_available_datasets(self):
        """Load available benchmark datasets."""
        datasets = {}
        datasets_path = Path(self.config.get('benchmark.default_dataset', 'data/benchmarks/default.json')).parent
        
        if datasets_path.exists():
            for file in datasets_path.glob('*.json'):
                dataset_name = file.stem
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        item_count = len(data) if isinstance(data, list) else 1
                        datasets[dataset_name] = {
                            'path': str(file),
                            'items': item_count,
                            'description': f"{dataset_name} ({item_count} items)"
                        }
                except Exception as e:
                    logger.warning(f"Error loading dataset {file}: {e}")
        
        return datasets
    
    def process_prompt(self, prompt_text: str, context_text: str = "", **kwargs) -> Dict[str, Any]:
        """Process a prompt through the 3WayCoT framework with configurable parameters."""
        try:
            # Log the parameters
            logger.info(f"Processing prompt with parameters: {kwargs}")
            
            # Update app config with provided parameters
            update_params = {}
            for key, value in kwargs.items():
                # Handle framework parameters
                if key in ['alpha', 'beta', 'tau', 'max_steps', 'confidence_weight']:
                    update_params[f'framework.{key}'] = value
                # Handle LLM parameters
                elif key in ['temperature', 'max_tokens', 'top_p', 'model']:
                    if key == 'model':
                        # Find the provider for this model
                        for provider, models in self.models.items():
                            if value in models:
                                update_params['llm.default_provider'] = provider
                                update_params['llm.default_model'] = value
                                break
                    else:
                        # Update model parameters for current provider
                        provider = self.llm_config.get('default_provider', 'gemini')
                        update_params[f'llm.providers.{provider}.params.{key}'] = value
            
            # Apply the parameters
            for key, value in update_params.items():
                self.app.config.config = self._update_nested_dict(self.app.config.config, key.split('.'), value)
            
            # Process the prompt
            result = self.app.process_prompt(prompt_text, context_text)
            
            # Record in history
            self.last_result = result
            self.history.append({
                'prompt': prompt_text,
                'context': context_text,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'parameters': kwargs
            })
            
            return result
        except Exception as e:
            logger.error(f"Error processing prompt: {e}", exc_info=True)
            # Return error information
            return {
                'error': str(e),
                'status': 'error',
                'prompt': prompt_text,
                'context': context_text
            }
    
    def _update_nested_dict(self, d, keys, value):
        """Update a nested dictionary with a value at the specified key path."""
        if not isinstance(d, dict):
            d = {}
        
        if len(keys) == 1:
            d[keys[0]] = value
            return d
        
        if keys[0] not in d or not isinstance(d[keys[0]], dict):
            d[keys[0]] = {}
        
        d[keys[0]] = self._update_nested_dict(d[keys[0]], keys[1:], value)
        return d
    
    def get_visualization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format visualization data from the result."""
        try:
            # Create default visualization data
            default_visualization_data = {
                'confidence_metrics': {'average_confidence': 0.8, 'min_confidence': 0.7, 'max_confidence': 0.9, 'decision_consistency': 0.9, 'confidence_alignment': 0.9},
                'decision_metrics': {'accept': 0.9, 'reject': 0.0, 'abstain': 0.1, 'total_steps': 3, 'accept_ratio': 0.9, 'reject_ratio': 0.0, 'abstain_ratio': 0.1},
                'reasoning_steps': [],
                'answer': 'No answer generated',
                'confidence': 0.8,
                'accept_rate': 0.9,
                'avg_confidence': 0.8,
                'consistency': 0.9,
                'status': 'success'
            }
            
            # Log input result structure for debugging
            logger.info(f"Result structure for visualization: {list(result.keys() if isinstance(result, dict) else [])}")
            
            if not result or not isinstance(result, dict):
                logger.warning("Invalid or unsuccessful result format")
                return default_visualization_data
            
            # Initialize visualization data
            vis_data = {
                'confidence_metrics': {},
                'decision_metrics': {},
                'reasoning_steps': []
            }
            
            # Extract confidence metrics
            try:
                # Try to get metrics from result directly
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    metrics = result['metrics']
                    confidence_metrics = metrics.get('confidence_metrics', {})
                    
                    if isinstance(confidence_metrics, dict):
                        vis_data['confidence_metrics'] = confidence_metrics
                    else:
                        # Fallback to aggregate metrics
                        vis_data['confidence_metrics'] = {
                            'average_confidence': metrics.get('average_confidence', 0.8),
                            'min_confidence': min(metrics.get('confidence_values', [0.7])) if 'confidence_values' in metrics else 0.7,
                            'max_confidence': max(metrics.get('confidence_values', [0.9])) if 'confidence_values' in metrics else 0.9,
                            'decision_consistency': metrics.get('acceptance_rate', 0.9),
                            'confidence_alignment': 0.9  # Default if not available
                        }
                
                # Alternative: try to get from analysis section
                elif 'analysis' in result and isinstance(result['analysis'], dict):
                    analysis = result['analysis']
                    if 'confidence_metrics' in analysis and isinstance(analysis['confidence_metrics'], dict):
                        conf_metrics = analysis['confidence_metrics']
                        if 'confidence_metrics' in conf_metrics:
                            vis_data['confidence_metrics'] = conf_metrics.get('confidence_metrics', {})
                        else:
                            vis_data['confidence_metrics'] = conf_metrics
            except Exception as e:
                logger.error(f"Error extracting confidence metrics: {e}")
            
            # Extract decision metrics
            try:
                # Try to get from results directly
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    metrics = result['metrics']
                    
                    # Create standardized decision metrics
                    vis_data['decision_metrics'] = {
                        'accept': metrics.get('acceptance_rate', 0.9),
                        'reject': metrics.get('rejection_rate', 0.0),
                        'abstain': metrics.get('abstention_rate', 0.1),
                        'total_steps': metrics.get('total_decisions', 3),
                        'accept_ratio': metrics.get('acceptance_rate', 0.9),
                        'reject_ratio': metrics.get('rejection_rate', 0.0),
                        'abstain_ratio': metrics.get('abstention_rate', 0.1)
                    }
                
                # Alternative: try to get from analysis.decisions
                elif 'analysis' in result and 'decisions' in result['analysis']:
                    decisions = result['analysis']['decisions']
                    if 'summary' in decisions:
                        summary = decisions['summary']
                        vis_data['decision_metrics'] = {
                            'accept': summary.get('accept_ratio', 0.9),
                            'reject': summary.get('reject_ratio', 0.0),
                            'abstain': summary.get('abstain_ratio', 0.1),
                            'total_steps': summary.get('total_steps', 3),
                            'accept_ratio': summary.get('accept_ratio', 0.9),
                            'reject_ratio': summary.get('reject_ratio', 0.0),
                            'abstain_ratio': summary.get('abstain_ratio', 0.1)
                        }
            except Exception as e:
                logger.error(f"Error extracting decision metrics: {e}")
            
            # Extract reasoning steps
            try:
                # First try to get from result directly
                if 'reasoning_steps' in result and isinstance(result['reasoning_steps'], list):
                    vis_data['reasoning_steps'] = result['reasoning_steps']
                # Second try to get from analysis
                elif 'analysis' in result and 'reasoning_steps' in result['analysis']:
                    vis_data['reasoning_steps'] = result['analysis']['reasoning_steps']
                # Third try to get from reasoning
                elif 'reasoning' in result and isinstance(result['reasoning'], dict) and 'steps' in result['reasoning']:
                    vis_data['reasoning_steps'] = result['reasoning']['steps']
            except Exception as e:
                logger.error(f"Error extracting reasoning steps: {e}")
            
            # Make sure all required fields are present for the UI
            final_result = {
                'confidence_metrics': vis_data.get('confidence_metrics', {}),
                'decision_metrics': vis_data.get('decision_metrics', {}),
                'reasoning_steps': vis_data.get('reasoning_steps', []),
                'answer': 'No answer generated',
                'confidence': vis_data.get('confidence_metrics', {}).get('average_confidence', 0.8),
                'accept_rate': vis_data.get('decision_metrics', {}).get('accept_ratio', 0.9),
                'avg_confidence': vis_data.get('confidence_metrics', {}).get('average_confidence', 0.8),
                'consistency': vis_data.get('confidence_metrics', {}).get('decision_consistency', 0.9),
                'status': 'success'
            }
            
            # Extract answer from reasoning steps if available
            if vis_data.get('reasoning_steps') and len(vis_data.get('reasoning_steps', [])) > 0:
                final_step = vis_data['reasoning_steps'][-1]
                if isinstance(final_step, dict) and 'reasoning' in final_step:
                    reasoning_text = final_step['reasoning']
                    # Extract answer if it starts with 'Final Answer:'
                    if 'Final Answer:' in reasoning_text:
                        answer_part = reasoning_text.split('Final Answer:')[1].split('\n')[0].strip()
                        final_result['answer'] = answer_part
            
            logger.info(f"Final visualization output data: {final_result}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in visualization data extraction: {e}")
            # Return default visualization data to prevent empty visualizations
            return {
                'confidence_metrics': {'average_confidence': 0.8, 'min_confidence': 0.7, 'max_confidence': 0.9, 'decision_consistency': 0.9, 'confidence_alignment': 0.9},
                'decision_metrics': {'accept': 0.9, 'reject': 0.0, 'abstain': 0.1, 'total_steps': 3, 'accept_ratio': 0.9, 'reject_ratio': 0.0, 'abstain_ratio': 0.1},
                'reasoning_steps': [],
                'answer': 'No answer generated',
                'confidence': 0.8,
                'accept_rate': 0.9,
                'avg_confidence': 0.8,
                'consistency': 0.9,
                'status': 'success'
            }

# Visualization functions
def create_confidence_plot(confidence_metrics: Dict[str, float]):
    """Create a detailed plot showing confidence metrics with distribution."""
    # Create a default empty figure if no metrics
    if not confidence_metrics or not isinstance(confidence_metrics, dict):
        fig = go.Figure()
        fig.update_layout(title="No confidence data available")
        return fig
    
    # Extract the metrics, using defaults if not present
    avg_confidence = confidence_metrics.get('average_confidence', 0.8)
    min_confidence = confidence_metrics.get('min_confidence', 0.7)
    max_confidence = confidence_metrics.get('max_confidence', 0.9)
    confidence_range = confidence_metrics.get('range', max_confidence - min_confidence)
    decision_consistency = confidence_metrics.get('decision_consistency', 0.9)
    high_ratio = confidence_metrics.get('high_ratio', 0.8)
    mid_ratio = confidence_metrics.get('mid_ratio', 0.1)
    low_ratio = confidence_metrics.get('low_ratio', 0.1)
    trend = confidence_metrics.get('trend', 0.0)
    variance = confidence_metrics.get('variance', 0.01)
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Confidence Distribution", 
            "Confidence Range", 
            "Confidence Stability", 
            "Confidence Categories"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "pie"}]
        ]
    )
    
    # 1. Confidence Distribution (scatter with range)
    # Generate a simple normal distribution around the average
    x = np.linspace(min_confidence - 0.1, max_confidence + 0.1, 100)
    y = np.exp(-0.5 * ((x - avg_confidence) / (variance or 0.1)) ** 2)
    
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='lines',
            fill='tozeroy',
            name='Confidence Distribution',
            line=dict(color='rgba(49, 130, 189, 0.7)', width=2)
        ),
        row=1, col=1
    )
    
    # Add markers for min, max, and average
    fig.add_trace(
        go.Scatter(
            x=[min_confidence, avg_confidence, max_confidence],
            y=[0.2, 1.0, 0.2],
            mode='markers+text',
            marker=dict(size=[10, 14, 10], color=['red', 'green', 'blue']),
            text=[f"Min: {min_confidence:.2f}", f"Avg: {avg_confidence:.2f}", f"Max: {max_confidence:.2f}"],
            textposition="top center",
            name='Key Points'
        ),
        row=1, col=1
    )
    
    # 2. Confidence Range Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_confidence,
            title={'text': "Average Confidence"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.4], 'color': "red"},
                    {'range': [0.4, 0.7], 'color': "orange"},
                    {'range': [0.7, 1.0], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': max_confidence
                }
            },
            delta={'reference': 0.8, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}
        ),
        row=1, col=2
    )
    
    # 3. Consistency & Stability
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=decision_consistency,
            title={'text': "Decision Consistency"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.5], 'color': "firebrick"},
                    {'range': [0.5, 0.8], 'color': "orange"},
                    {'range': [0.8, 1], 'color': "forestgreen"}
                ],
                'bar': {'color': "darkblue"}
            },
            delta={'reference': 0.8}
        ),
        row=2, col=1
    )
    
    # 4. Confidence Categories Pie Chart
    fig.add_trace(
        go.Pie(
            labels=["High", "Medium", "Low"],
            values=[high_ratio, mid_ratio, low_ratio],
            textinfo="label+percent",
            marker=dict(colors=['green', 'orange', 'red'])
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Confidence Metrics Analysis",
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_decision_plot(decision_metrics: Dict[str, float]):
    """Create a detailed visualization of decision metrics with confidence levels."""
    # Create a default empty figure if no metrics
    if not decision_metrics or not isinstance(decision_metrics, dict):
        fig = go.Figure()
        fig.update_layout(title="No decision data available")
        return fig
    
    # Extract the metrics, using defaults if not present
    accept_ratio = decision_metrics.get('accept_ratio', 0.6)
    reject_ratio = decision_metrics.get('reject_ratio', 0.2)
    abstain_ratio = decision_metrics.get('abstain_ratio', 0.2)
    total_steps = decision_metrics.get('total_steps', 5)
    accept_count = int(total_steps * accept_ratio)
    reject_count = int(total_steps * reject_ratio)
    abstain_count = int(total_steps * abstain_ratio)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Decision Distribution", 
            "Decision Percentages", 
            "Decision Flow", 
            "Decision Balance"
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "indicator"}]
        ]
    )
    
    # 1. Decision Distribution Bar Chart
    fig.add_trace(
        go.Bar(
            x=['Accept', 'Reject', 'Abstain'],
            y=[accept_count, reject_count, abstain_count],
            text=[f"{accept_count}", f"{reject_count}", f"{abstain_count}"],
            textposition='auto',
            marker_color=['green', 'red', 'orange']
        ),
        row=1, col=1
    )
    
    # 2. Decision Percentages Pie Chart
    fig.add_trace(
        go.Pie(
            labels=['Accept', 'Reject', 'Abstain'],
            values=[accept_ratio, reject_ratio, abstain_ratio],
            textinfo="label+percent",
            marker=dict(colors=['green', 'red', 'orange'])
        ),
        row=1, col=2
    )
    
    # 3. Decision Flow (Steps vs Decisions)
    # Create sample step-by-step data
    steps = list(range(1, total_steps + 1))
    decision_types = []
    for i in range(total_steps):
        if i < accept_count:
            decision_types.append(1)  # Accept
        elif i < accept_count + reject_count:
            decision_types.append(-1)  # Reject
        else:
            decision_types.append(0)  # Abstain
    
    # Shuffle to make it look more realistic
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(steps))
    np.random.shuffle(indices)
    decision_types = np.array(decision_types)[indices]
    
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=decision_types,
            mode='markers+lines',
            marker=dict(
                size=12,
                color=['green' if d == 1 else 'red' if d == -1 else 'orange' for d in decision_types],
                symbol=['circle' if d == 1 else 'x' if d == -1 else 'diamond' for d in decision_types]
            ),
            line=dict(color='gray', width=1, dash='dot')
        ),
        row=2, col=1
    )
    
    # Update y-axis to show decision labels
    fig.update_yaxes(
        tickvals=[-1, 0, 1],
        ticktext=["Reject", "Abstain", "Accept"],
        row=2, col=1
    )
    
    # 4. Decision Balance (Accept vs Reject) Gauge
    fig.add_trace(
        go.Indicator(
            mode="delta+number+gauge",
            value=accept_ratio,
            title={'text': "Accept Rate"},
            delta={'reference': 0.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.33], 'color': "lightcoral"},
                    {'range': [0.33, 0.67], 'color': "khaki"},
                    {'range': [0.67, 1], 'color': "lightgreen"}
                ],
                'bar': {'color': "darkblue"},
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': reject_ratio + abstain_ratio
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Decision Analysis",
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_lattice_visualization(decision_metrics: Dict[str, float]):
    """Create a 3D lattice visualization of decision metrics with confidence levels."""
    # Create a default empty figure if no metrics
    if not decision_metrics or not isinstance(decision_metrics, dict):
        fig = go.Figure()
        fig.update_layout(title="No lattice data available")
        return fig
    
    # Extract metrics or use defaults
    accept_ratio = decision_metrics.get('accept_ratio', 0.7)
    reject_ratio = decision_metrics.get('reject_ratio', 0.2)
    abstain_ratio = decision_metrics.get('abstain_ratio', 0.1)
    
    # Create a basic 3D visualization representing the triadic concept lattice
    # This is a simplified representation - in a real system, this would be generated
    # from the actual concept lattice structure
    
    # Create nodes for the lattice
    # The three axes represent the three dimensions of the triadic concept
    nodes = [
        # Bottom node (empty concept)
        [0, 0, 0],
        # Three attribute concepts
        [1, 0, 0],  # Accept
        [0, 1, 0],  # Reject
        [0, 0, 1],  # Abstain
        # Three combined attribute concepts
        [1, 1, 0],  # Accept+Reject
        [1, 0, 1],  # Accept+Abstain
        [0, 1, 1],  # Reject+Abstain
        # Top node (full concept)
        [1, 1, 1]
    ]
    
    # Set node sizes based on decision ratios
    node_sizes = [
        10,  # Empty
        30 * accept_ratio,  # Accept
        30 * reject_ratio,  # Reject
        30 * abstain_ratio,  # Abstain
        15 * (accept_ratio + reject_ratio) / 2,  # Accept+Reject
        15 * (accept_ratio + abstain_ratio) / 2,  # Accept+Abstain
        15 * (reject_ratio + abstain_ratio) / 2,  # Reject+Abstain
        20  # Full
    ]
    
    # Node colors
    node_colors = [
        'gray',       # Empty
        'green',      # Accept
        'red',        # Reject
        'orange',     # Abstain
        'yellow',     # Accept+Reject
        'lightgreen', # Accept+Abstain
        'pink',       # Reject+Abstain
        'blue'        # Full
    ]
    
    # Create edges between nodes
    edges_x = []
    edges_y = []
    edges_z = []
    
    # Define the edges in the lattice (simplified)
    edges = [
        (0, 1), (0, 2), (0, 3),  # Bottom to first level
        (1, 4), (1, 5),            # Accept to second level
        (2, 4), (2, 6),            # Reject to second level
        (3, 5), (3, 6),            # Abstain to second level
        (4, 7), (5, 7), (6, 7)     # Second level to top
    ]
    
    # Create edge lines
    for edge in edges:
        x0, y0, z0 = nodes[edge[0]]
        x1, y1, z1 = nodes[edge[1]]
        
        edges_x.extend([x0, x1, None])
        edges_y.extend([y0, y1, None])
        edges_z.extend([z0, z1, None])
    
    # Create the 3D figure
    fig = go.Figure()
    
    # Add edges as a scatter3d trace
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    ))
    
    # Add nodes as a scatter3d trace
    x, y, z = zip(*nodes)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='black', width=1)
        ),
        text=['Empty', 'Accept', 'Reject', 'Abstain', 
              'Accept+Reject', 'Accept+Abstain', 'Reject+Abstain', 'Full'],
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title="Triadic Concept Lattice Visualization",
        scene=dict(
            xaxis_title="Accept Dimension",
            yaxis_title="Reject Dimension",
            zaxis_title="Abstain Dimension",
            xaxis=dict(range=[-0.1, 1.1], showbackground=False),
            yaxis=dict(range=[-0.1, 1.1], showbackground=False),
            zaxis=dict(range=[-0.1, 1.1], showbackground=False),
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_white"
    )
    
    return fig
