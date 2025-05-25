#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WayCoT Gradio Interface

A user-friendly web interface for the 3WayCoT framework using Gradio.
This provides a more interactive experience for exploring the framework's capabilities.
"""

import gradio as gr
import logging
import streamlit as st
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('3waycot_gradio.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the main application
from main import ThreeWayCoTApp, ConfigManager

class Gradio3WayCoT:
    """Gradio interface for the 3WayCoT framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gradio interface."""
        self.config_path = config_path
        self.app = ThreeWayCoTApp(config_path)
        self.interface = None
        self.last_result = None
        self.history = []
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load configuration from the config file and set up defaults."""
        try:
            # Get framework defaults
            framework_config = self.config.config.get('framework', {})
            llm_config = self.config.config.get('llm', {})
            use_cases = self.config.config.get('use_cases', {})
            
            # Set default parameters from config
            self.default_params = {
                'alpha': framework_config.get('alpha', 0.7),
                'beta': framework_config.get('beta', 0.6),
                'tau': framework_config.get('tau', 0.5),
                'max_assumptions': framework_config.get('max_assumptions', 5),
                'max_steps': framework_config.get('max_steps', 10),
                'similarity_threshold': framework_config.get('similarity_threshold', 0.65),
                'confidence_weight': framework_config.get('confidence_weight', 0.7)
            }
            
            # Get LLM parameters
            default_provider = llm_config.get('default_provider', 'gemini')
            provider_config = llm_config.get('providers', {}).get(default_provider, {})
            llm_params = provider_config.get('params', {})
            
            # Add LLM parameters to defaults
            self.default_params.update({
                'temperature': llm_params.get('temperature', 0.7),
                'max_tokens': llm_params.get('max_tokens', 2000),
                'top_p': llm_params.get('top_p', 1.0),
                'model': llm_config.get('default_model', 'gpt-4'),
                'provider': default_provider
            })
            
            # Load use cases
            self.use_cases = {}
            for name, config in use_cases.items():
                if isinstance(config, dict):
                    self.use_cases[name] = {
                        'alpha': config.get('alpha', self.default_params['alpha']),
                        'beta': config.get('beta', self.default_params['beta']),
                        'tau': config.get('tau', self.default_params['tau']),
                        'temperature': config.get('temperature', self.default_params['temperature']),
                        'max_tokens': config.get('max_tokens', self.default_params['max_tokens']),
                        'description': config.get('description', '')
                    }
            
            # Add default use cases if none found
            if not self.use_cases:
                self.use_cases = {
                    'conservative': {
                        'alpha': 0.8,
                        'beta': 0.5,
                        'tau': 0.6,
                        'temperature': 0.5,
                        'max_tokens': 1000,
                        'description': 'More conservative decision making with higher confidence thresholds'
                    },
                    'balanced': {
                        'alpha': 0.7,
                        'beta': 0.6,
                        'tau': 0.5,
                        'temperature': 0.7,
                        'max_tokens': 2000,
                        'description': 'Balanced approach for general use cases'
                    },
                    'exploratory': {
                        'alpha': 0.6,
                        'beta': 0.7,
                        'tau': 0.4,
                        'temperature': 0.9,
                        'max_tokens': 3000,
                        'description': 'More exploratory with higher uncertainty tolerance'
                    }
                }
            
            # Load benchmark datasets
            self.benchmark_datasets = self._load_benchmark_datasets()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _load_benchmark_datasets(self) -> Dict[str, Any]:
        """Load available benchmark datasets."""
        try:
            benchmark_config = self.config.config.get('benchmark', {})
            datasets_dir = Path(benchmark_config.get('datasets_dir', 'data/benchmarks'))
            
            if not datasets_dir.exists():
                datasets_dir.mkdir(parents=True, exist_ok=True)
                return {}
            
            datasets = {}
            for file_path in datasets_dir.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'name' in data:
                            datasets[file_path.stem] = {
                                'name': data.get('name', file_path.stem),
                                'path': str(file_path),
                                'description': data.get('description', ''),
                                'num_examples': len(data.get('examples', [])),
                                'metrics': data.get('metrics', [])
                            }
                except Exception as e:
                    self.logger.warning(f"Error loading benchmark {file_path}: {str(e)}")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark datasets: {str(e)}")
            return {}
        
    def get_visualization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format visualization data from the result.
        
        Args:
            result: The result dictionary from process_prompt
            
        Returns:
            Dictionary with formatted visualization data
        """
        try:
            logger = logging.getLogger('3WayCoT.GradioApp')
            logger.info("=================================================================================")
            logger.info(f"Visualization input data type: {type(result)}")
            logger.info(f"Visualization input data keys: {result.keys() if isinstance(result, dict) else 'Not a dictionary'}")
        
            # Debug the structure more deeply
            if isinstance(result, dict):
                for key, value in result.items():
                    logger.info(f"Key: {key}, Type: {type(value)}")
                    if isinstance(value, dict):
                        logger.info(f"  Subkeys for {key}: {value.keys()}")
            logger.info(f"Full visualization input data: {result}")
            
            # Check if result is valid
            if not result:
                logger.warning("Empty result provided to visualization")
                return {
                    'confidence_metrics': {'average_confidence': 0.8, 'min_confidence': 0.7, 'max_confidence': 0.9, 'decision_consistency': 0.9, 'confidence_alignment': 0.9},
                    'decision_metrics': {'accept': 0.9, 'reject': 0.0, 'abstain': 0.1, 'total_steps': 3, 'accept_ratio': 0.9, 'reject_ratio': 0.0, 'abstain_ratio': 0.1},
                    'reasoning_steps': []
                }
            
            # Handle different potential data structures
            result_data = result
            if isinstance(result, dict) and 'result' in result:
                result_data = result['result']
                
            # Initialize visualization data with default values to prevent empty visualizations
            vis_data = {
                'confidence_metrics': {
                    'average_confidence': 0.8, 
                    'min_confidence': 0.7, 
                    'max_confidence': 0.9,
                    'decision_consistency': 0.9,
                    'confidence_alignment': 0.9
                },
                'decision_metrics': {
                    'accept': 0.9,
                    'reject': 0.0,
                    'abstain': 0.1,
                    'total_steps': 3,
                    'accept_ratio': 0.9,
                    'reject_ratio': 0.0,
                    'abstain_ratio': 0.1
                },
                'reasoning_steps': []
            }
        
            # Extract confidence metrics from potential nested structures
            if 'decisions' in result_data:
                try:
                    decisions_data = result_data['decisions']
                    if isinstance(decisions_data, dict) and 'summary' in decisions_data:
                        summary = decisions_data['summary']
                        metadata = decisions_data.get('metadata', {})
                        confidence_metrics = metadata.get('confidence_metrics', {})
                        confidence_distribution = metadata.get('confidence_distribution', {})
                        
                        # Try to extract from either the summary or the metadata
                        vis_data['confidence_metrics'].update({
                            'average_confidence': summary.get('avg_confidence_accept', 
                                                 confidence_distribution.get('avg_confidence', 0.8)),
                            'min_confidence': confidence_distribution.get('min_confidence', 0.7),
                            'max_confidence': confidence_distribution.get('max_confidence', 0.9),
                            'decision_consistency': summary.get('decision_consistency', 0.9),
                            'confidence_alignment': summary.get('confidence_alignment', 0.9)
                        })
                    
                    # Extract decision metrics
                    if isinstance(decisions_data, dict):
                        # Try to get decisions from either the decisions key or directly from the array
                        decisions = decisions_data.get('decisions', [])
                        if not decisions and isinstance(decisions_data, list):
                            decisions = decisions_data
                            
                        # If we have a summary, use it for metrics
                        if 'summary' in decisions_data:
                            summary = decisions_data['summary']
                            vis_data['decision_metrics'].update({
                                'accept': summary.get('accept_ratio', 0.9),
                                'reject': summary.get('reject_ratio', 0.0),
                                'abstain': summary.get('abstain_ratio', 0.1),
                                'total_steps': summary.get('total_steps', 3),
                                'accept_ratio': summary.get('accept_ratio', 0.9),
                                'reject_ratio': summary.get('reject_ratio', 0.0),
                                'abstain_ratio': summary.get('abstain_ratio', 0.1)
                            })
                        else:  # Otherwise calculate from the decisions
                            accept_count = sum(1 for d in decisions if d.get('decision') == 'ACCEPT')
                            reject_count = sum(1 for d in decisions if d.get('decision') == 'REJECT')
                            abstain_count = sum(1 for d in decisions if d.get('decision') == 'ABSTAIN')
                            total = len(decisions) if decisions else 1
                            
                            vis_data['decision_metrics'].update({
                                'accept': accept_count / total if total > 0 else 0.9,
                                'reject': reject_count / total if total > 0 else 0.0,
                                'abstain': abstain_count / total if total > 0 else 0.1,
                                'total_steps': total,
                                'accept_ratio': accept_count / total if total > 0 else 0.9,
                                'reject_ratio': reject_count / total if total > 0 else 0.0,
                                'abstain_ratio': abstain_count / total if total > 0 else 0.1
                            })
                except Exception as e:
                    logger.error(f"Error extracting decision metrics: {e}")
            
            # Extract reasoning steps from multiple possible locations
            try:
                if 'reasoning_steps' in result_data:
                    vis_data['reasoning_steps'] = result_data['reasoning_steps']
                elif isinstance(result_data, dict) and 'result' in result_data and 'reasoning_steps' in result_data['result']:
                    vis_data['reasoning_steps'] = result_data['result']['reasoning_steps']
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
    
    def process_prompt(self, prompt_text: str, context_text: str = "", **kwargs) -> Dict[str, Any]:
        """Process a prompt through the 3WayCoT framework with configurable parameters.
        
        Args:
            prompt_text: The input prompt to process
            context_text: Optional context for the prompt
            **kwargs: Additional parameters to update in the app
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Update app parameters if any are provided
            if kwargs:
                self.logger.info(f"Updating parameters: {kwargs}")
                self.app.update_parameters(**kwargs)
            
            # Process the prompt using the 3WayCoT app
            result = self.app.process_prompt(prompt_text, context_text)
            self.last_result = result
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt_text,
                'context': context_text,
                'result': result,
                'params': kwargs or {}
            })
            
            # Get visualization data
            vis_data = self.get_visualization_data({'result': result})
            
            return {
                'status': 'success',
                'result': result,
                'visualization_data': vis_data
            }
        except Exception as e:
            self.logger.error("ERROR PROCESSING REQUEST", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_visualization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualization with detailed metrics and decisions."""
        logger = logging.getLogger('3WayCoT.GradioApp')
        logger.info("\n" + "="*80)
        logger.info("ENTERING get_visualization_data")
        logger.info("="*80)
        
        if not result or 'status' not in result or result['status'] != 'success':
            logger.warning("❌ Invalid or unsuccessful result format")
            return {}
            
        result_data = result.get('result', {})
        logger.info(f"🔍 Result data keys: {list(result_data.keys())}")
        
        # Log analysis results if available
        if 'analysis_results' in result_data:
            analysis = result_data['analysis_results']
            logger.info(f"📊 Analysis results keys: {list(analysis.keys())}")
            if 'confidence' in analysis:
                logger.info(f"   - Confidence from analysis: {analysis['confidence']}")
        
        # Log decisions if available
        if 'decisions' in result_data and isinstance(result_data['decisions'], dict):
            decisions = result_data['decisions']
            logger.info(f"🎯 Decisions keys: {list(decisions.keys())}")
            
            # Log metadata if available
            if 'metadata' in decisions:
                metadata = decisions['metadata']
                logger.info(f"   📋 Metadata keys: {list(metadata.keys())}")
                
                # Log confidence distribution if available
                if 'confidence_distribution' in metadata:
                    dist = metadata['confidence_distribution']
                    logger.info(f"   📊 Confidence distribution: {dist}")
            
            # Log summary if available
            if 'summary' in decisions and isinstance(decisions['summary'], dict):
                summary = decisions['summary']
                logger.info("📝 Decision summary:")
                for k, v in summary.items():
                    if 'confidence' in k.lower():
                        logger.info(f"   - {k}: {v}")
        
        logger.info("-"*80)
        
        # Extract decisions if available
        decisions = result_data.get('decisions', {})
        decision_list = decisions.get('decisions', []) if isinstance(decisions, dict) else []
        decision_summary = decisions.get('summary', {}) if isinstance(decisions, dict) else {}
        
        # Prepare confidence metrics from analysis_results or decisions
        confidence_metrics = {}
        
        # First, try to get confidence from analysis_results
        if 'analysis_results' in result_data and isinstance(result_data['analysis_results'], dict):
            analysis = result_data['analysis_results']
            confidence_metrics.update({
                'confidence': float(analysis.get('confidence', 0.5)),
                'clarity': float(analysis.get('clarity', 0.5)),
                'consistency': float(analysis.get('consistency', 0.5))
            })
            logger.info(f"Added metrics from analysis_results: {list(confidence_metrics.keys())}")
        
        # Then try to get from decisions metadata
        if 'decisions' in result_data and isinstance(result_data['decisions'], dict):
            decisions = result_data['decisions']
            metadata = decisions.get('metadata', {})
            
            # Get confidence distribution if available
            if 'confidence_distribution' in metadata:
                dist = metadata['confidence_distribution']
                confidence_metrics.update({
                    'min_confidence': float(dist.get('min_confidence', 0.0)),
                    'max_confidence': float(dist.get('max_confidence', 1.0)),
                    'avg_confidence': float(dist.get('avg_confidence', 0.5)),
                    'high_confidence_ratio': float(dist.get('distribution', {}).get('high', 0) / max(1, dist.get('count', 1))),
                    'medium_confidence_ratio': float(dist.get('distribution', {}).get('medium', 0) / max(1, dist.get('count', 1))),
                    'low_confidence_ratio': float(dist.get('distribution', {}).get('low', 0) / max(1, dist.get('count', 1)))
                })
                logger.info(f"Added metrics from confidence_distribution: {list(confidence_metrics.keys())}")
            
            # Get from decision summary if available
            if 'summary' in decisions and isinstance(decisions['summary'], dict):
                summary = decisions['summary']
                confidence_metrics.update({
                    'accept_confidence': float(summary.get('avg_confidence_accept', 0.5)),
                    'reject_confidence': float(summary.get('avg_confidence_reject', 0.5)),
                    'abstain_confidence': float(summary.get('avg_confidence_abstain', 0.5)),
                    'decision_consistency': float(summary.get('decision_consistency', 1.0)),
                    'confidence_alignment': float(summary.get('confidence_alignment', 1.0))
                })
                logger.info(f"Added metrics from decision summary: {list(confidence_metrics.keys())}")
        
        # Finally, try top-level result data
        if 'confidence' in result_data:
            confidence_metrics['confidence'] = float(result_data['confidence'])
            logger.info(f"Updated confidence from top-level result: {confidence_metrics['confidence']}")
            
        # Ensure we have at least some confidence value
        if 'confidence' not in confidence_metrics and 'avg_confidence' in confidence_metrics:
            confidence_metrics['confidence'] = confidence_metrics['avg_confidence']
        elif 'confidence' not in confidence_metrics:
            confidence_metrics['confidence'] = 0.5  # Last resort default
            
        logger.info(f"Final confidence metrics: {confidence_metrics}")
        
        # Extract confidence distribution if available in metadata
        if 'metadata' in decisions and 'confidence_distribution' in decisions['metadata']:
            dist = decisions['metadata']['confidence_distribution']
            confidence_metrics.update({
                'min_confidence': dist.get('min_confidence', 0.0),
                'max_confidence': dist.get('max_confidence', 1.0),
                'avg_confidence': dist.get('avg_confidence', 0.7),
                'high_confidence_ratio': dist.get('distribution', {}).get('high', 0) / max(1, dist.get('count', 1)),
                'medium_confidence_ratio': dist.get('distribution', {}).get('medium', 0) / max(1, dist.get('count', 1)),
                'low_confidence_ratio': dist.get('distribution', {}).get('low', 0) / max(1, dist.get('count', 1))
            })
            
        # Prepare decision metrics
        decision_metrics = {
            'accept': decision_summary.get('accept_ratio', 0.0),
            'reject': decision_summary.get('reject_ratio', 0.0),
            'abstain': decision_summary.get('abstain_ratio', 0.0),
            'accept_confidence': decision_summary.get('avg_confidence_accept', 0.7),
            'reject_confidence': decision_summary.get('avg_confidence_reject', 0.0),
            'abstain_confidence': decision_summary.get('avg_confidence_abstain', 0.0)
        }
        
        # Prepare reasoning steps with more robust extraction
        reasoning_steps = []
        
        # Try to get reasoning steps from different possible locations in the result
        steps_data = []
        
        # Case 1: Directly in result_data
        if 'reasoning_steps' in result_data and isinstance(result_data['reasoning_steps'], list):
            steps_data = result_data['reasoning_steps']
            logger.info(f"Found {len(steps_data)} reasoning steps in result_data['reasoning_steps']")
        # Case 2: In analysis_results
        elif 'analysis_results' in result_data and 'reasoning_steps' in result_data['analysis_results']:
            steps_data = result_data['analysis_results']['reasoning_steps']
            logger.info(f"Found {len(steps_data)} reasoning steps in analysis_results")
        # Case 3: In the root of the result
        elif 'steps' in result_data and isinstance(result_data['steps'], list):
            steps_data = result_data['steps']
            logger.info(f"Found {len(steps_data)} steps in result_data")
            
        # Process the steps if we found any
        if steps_data:
            for i, step in enumerate(steps_data):
                if isinstance(step, dict):
                    reasoning_steps.append({
                        'step': i + 1,
                        'content': step.get('content', 
                                         step.get('thought', 
                                                step.get('reasoning', 
                                                      str(step)))),
                        'confidence': float(step.get('confidence', 0.5))
                    })
                else:
                    # Handle case where step is just a string
                    reasoning_steps.append({
                        'step': i + 1,
                        'content': str(step),
                        'confidence': 0.5
                    })
            
            logger.info(f"Processed {len(reasoning_steps)} reasoning steps")
        else:
            logger.warning("No reasoning steps found in the result data")
        
        # Prepare decision metrics - extract from decisions if available
        decision_metrics = {
            'accept': 0.0,
            'reject': 0.0,
            'abstain': 0.0,
            'accept_confidence': 0.0,
            'reject_confidence': 0.0,
            'abstain_confidence': 0.0,
            'decision_consistency': 0.0,
            'confidence_alignment': 0.0
        }
        
        if 'decisions' in result_data and isinstance(result_data['decisions'], dict):
            decisions = result_data['decisions']
            # Get base decision metrics
            decision_metrics.update({
                'accept': float(decisions.get('accept_degree', 0.0)),
                'reject': float(decisions.get('reject_degree', 0.0)),
                'abstain': float(decisions.get('abstain_degree', 0.0))
            })
            
            # Get confidence metrics if available in decisions
            if 'metadata' in decisions and 'confidence_metrics' in decisions['metadata']:
                conf_metrics = decisions['metadata']['confidence_metrics']
                decision_metrics.update({
                    'accept_confidence': float(conf_metrics.get('accept_confidence', 0.0)),
                    'reject_confidence': float(conf_metrics.get('reject_confidence', 0.0)),
                    'abstain_confidence': float(conf_metrics.get('abstain_confidence', 0.0)),
                    'decision_consistency': float(conf_metrics.get('decision_consistency', 0.0)),
                    'confidence_alignment': float(conf_metrics.get('confidence_alignment', 0.0))
                })
        
        # Determine final decision
        final_decision = 'Unknown'
        if 'summary' in result_data and result_data['summary']:
            final_decision = result_data['summary']
        elif 'final_decision' in result_data:
            final_decision = result_data['final_decision']
        
        return {
            'confidence_metrics': confidence_metrics,
            'reasoning_steps': reasoning_steps,
            'decision_metrics': decision_metrics,
            'final_decision': final_decision
        }

def create_confidence_plot(confidence_metrics: Dict[str, float]) -> go.Figure:
    """Create a detailed plot showing confidence metrics with distribution."""
    if not confidence_metrics:
        return go.Figure(data=[go.Bar(x=[], y=[])])
    
    # Categorize metrics for better organization
    base_metrics = {k: v for k, v in confidence_metrics.items() 
                   if k in ['confidence', 'clarity', 'consistency']}
    dist_metrics = {k: v for k, v in confidence_metrics.items() 
                   if k in ['min_confidence', 'max_confidence', 'avg_confidence']}
    ratio_metrics = {k: v for k, v in confidence_metrics.items() 
                    if k.endswith('_ratio')}
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Base Confidence Metrics',
            'Confidence Distribution',
            'Confidence Level Ratios',
            'Decision Consistency'
        ),
        specs=[[{"type": "bar"}, {"type": "box"}],
              [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Base metrics
    if base_metrics:
        fig.add_trace(
            go.Bar(
                x=list(base_metrics.keys()),
                y=list(base_metrics.values()),
                text=[f"{v:.2f}" for v in base_metrics.values()],
                textposition='auto',
                marker_color='rgba(55, 128, 191, 0.7)'
            ),
            row=1, col=1
        )
    
    # Distribution metrics
    if dist_metrics:
        fig.add_trace(
            go.Box(
                y=[confidence_metrics.get('min_confidence', 0.6), 
                   confidence_metrics.get('avg_confidence', 0.7), 
                   confidence_metrics.get('max_confidence', 0.8)],
                name='Confidence Range',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker_color='rgb(7,40,89)',
                line_color='rgb(7,40,89)'
            ),
            row=1, col=2
        )
    
    # Ratio metrics
    if ratio_metrics:
        fig.add_trace(
            go.Bar(
                x=[k.replace('_ratio', '').replace('_', ' ').title() 
                   for k in ratio_metrics.keys()],
                y=list(ratio_metrics.values()),
                text=[f"{v*100:.1f}%" for v in ratio_metrics.values()],
                textposition='auto',
                marker_color='rgba(50, 171, 96, 0.7)'
            ),
            row=2, col=1
        )
    
    # Decision consistency gauge
    consistency = confidence_metrics.get('decision_consistency', 1.0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=consistency * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Decision Consistency (%)'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': consistency * 100}
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white"
    )
    
    # Update y-axis ranges
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    
    return fig

def create_lattice_visualization(decision_metrics: Dict[str, float]) -> go.Figure:
    """Create a 3D lattice visualization of decision metrics with confidence levels."""
    if not decision_metrics or not all(k in decision_metrics for k in ['accept', 'reject', 'abstain']):
        return go.Figure()
        
    # Extract metrics
    accept = decision_metrics.get('accept', 0)
    reject = decision_metrics.get('reject', 0)
    abstain = decision_metrics.get('abstain', 0)
    
    # Create 3D lattice points
    x = [accept, -reject, 0, 0, 0, 0]
    y = [0, 0, accept, -reject, 0, 0]
    z = [0, 0, 0, 0, accept, -reject]
    
    # Create figure
    fig = go.Figure()
    
    # Add decision points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+text',
        marker=dict(
            size=10,
            color=['green', 'red', 'green', 'red', 'blue', 'blue'],
            opacity=0.8
        ),
        text=['Accept', 'Reject', 'Accept', 'Reject', 'Abstain', 'Abstain'],
        textposition='top center'
    ))
    
    # Add decision boundaries
    fig.add_trace(go.Mesh3d(
        x=[1, -1, 0, 0],
        y=[0, 0, 1, -1],
        z=[0, 0, 0, 0],
        opacity=0.2,
        color='lightgray',
        showscale=False
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Accept/Reject',
            yaxis_title='Accept/Reject',
            zaxis_title='Abstain',
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[-1.5, 1.5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title='Decision Space Lattice Visualization',
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_decision_plot(decision_metrics: Dict[str, float]) -> go.Figure:
    """Create a detailed visualization of decision metrics with confidence levels."""
    if not decision_metrics:
        return go.Figure(data=[go.Bar(x=[], y=[])])
        
    # Ensure all required metrics exist
    for k in ['accept', 'reject', 'abstain']:
        if k not in decision_metrics:
            decision_metrics[k] = 0.0
    for k in ['accept_confidence', 'reject_confidence', 'abstain_confidence']:
        if k not in decision_metrics:
            decision_metrics[k] = 0.0
    
    # Extract base decision metrics
    base_metrics = {k: v for k, v in decision_metrics.items() 
                   if k in ['accept', 'reject', 'abstain']}
    
    # Extract confidence metrics if available
    conf_metrics = {k: v for k, v in decision_metrics.items()
                   if k.endswith('_confidence')}
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "bar"}]],
        subplot_titles=(
            'Decision Metrics Radar',
            'Decision Confidence Comparison'
        )
    )
    
    # Radar chart for base metrics
    if base_metrics:
        categories = list(base_metrics.keys())
        values = [base_metrics[k] for k in categories]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=[c.upper() for c in categories] + [categories[0].upper()],
                fill='toself',
                name='Decision Metrics',
                line=dict(color='#3498db', width=2),
                marker=dict(size=8, symbol='circle')
            ),
            row=1, col=1
        )
    
    # Bar chart for confidence comparison
    if conf_metrics:
        conf_categories = [k.replace('_confidence', '').title() 
                         for k in conf_metrics.keys()]
        conf_values = list(conf_metrics.values())
        
        fig.add_trace(
            go.Bar(
                x=conf_categories,
                y=conf_values,
                text=[f"{v:.1%}" for v in conf_values],
                textposition='auto',
                marker_color=['#2ecc71', '#e74c3c', '#f39c12'],
                opacity=0.8,
                width=0.6,
                textfont=dict(size=12, color='black')
            ),
            row=1, col=2
        )
    
    # Update radar chart layout
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickfont=dict(size=10),
            gridcolor='lightgray',
            linewidth=2,
            linecolor='gray',
            gridwidth=1
        ),
        angularaxis=dict(
            rotation=90,
            direction='clockwise',
            gridcolor='lightgray',
            linecolor='gray',
            linewidth=2
        ),
        bgcolor='rgba(245, 246, 249, 0.5)'
    )
    
    # Update bar chart layout
    fig.update_yaxes(
        title_text="Confidence Level",
        range=[0, 1.1],
        row=1, col=2
    )
    
    # Update overall layout
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        title=dict(
            text="Decision Analysis",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        )
    )
    
    # Add annotations for metrics if available
    if base_metrics and conf_metrics:
        annotations = []
        for i, (metric, value) in enumerate(base_metrics.items()):
            conf_value = decision_metrics.get(f"{metric}_confidence", 0)
            annotations.append(dict(
                x=1.05 if i % 2 == 0 else 1.4,
                y=0.9 - (i * 0.15),
                xref="paper",
                yref="paper",
                text=f"{metric.upper()}: {value:.2f} (Confidence: {conf_value:.0%})",
                showarrow=False,
                font=dict(size=11)
            ))
        
        fig.update_layout(annotations=annotations)
    
    return fig

def create_reasoning_steps_table(reasoning_steps: List[Dict]) -> pd.DataFrame:
    """Create a detailed table of reasoning steps with additional metrics.
    
    Args:
        reasoning_steps: List of dictionaries containing reasoning step data
        
    Returns:
        pd.DataFrame: DataFrame with detailed step information
    """
    if not reasoning_steps or not isinstance(reasoning_steps, list):
        return pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision", "Metrics"])
    
    processed_steps = []
    
    for i, step in enumerate(reasoning_steps, 1):
        if not isinstance(step, dict):
            continue
            
        try:
            # Extract basic step information
            step_num = int(step.get('step', i))
            content = str(step.get('content', step.get('thought', '')) or ''
                        ).replace('\n', '<br>')  # Preserve line breaks
            
            # Extract confidence and related metrics
            confidence = float(step.get('confidence', 0.0))
            decision = step.get('decision', 'UNKNOWN')
            
            # Extract metrics if available
            metrics = step.get('metrics', {})
            similarity = metrics.get('similarity_score', 0.0)
            uncertainty = metrics.get('uncertainty_score', 0.0)
            
            # Create metrics text with tooltips
            metrics_text = (
                f"Similarity: {similarity:.2f} • "
                f"Uncertainty: {uncertainty:.2f}"
            )
            
            # Get membership scores if available
            membership = step.get('membership_scores', {})
            if membership:
                metrics_text += (
                    f"<br>Membership: "
                    f"A:{membership.get('accept', 0):.2f} • "
                    f"R:{membership.get('reject', 0):.2f} • "
                    f"U:{membership.get('abstain', 0):.2f}"
                )
            
            # Color code confidence
            confidence_color = ""
            if confidence >= 0.7:
                confidence_color = "color: #27ae60"  # Green
            elif confidence >= 0.4:
                confidence_color = "color: #f39c12"  # Orange
            else:
                confidence_color = "color: #e74c3c"  # Red
            
            # Create step dictionary with styling information
            processed_step = {
                'Step': step_num,
                'Content': content,
                'Confidence': f"{confidence:.1%}",
                'Confidence_Value': confidence,  # For sorting
                'Decision': decision,
                'Metrics': metrics_text,
                'confidence_style': confidence_color,
                'decision_style': (
                    'color: #27ae60' if decision == 'ACCEPT' else
                    'color: #e74c3c' if decision == 'REJECT' else
                    'color: #f39c12'  # For ABSTAIN or other
                )
            }
            
            processed_steps.append(processed_step)
            
        except (ValueError, TypeError, AttributeError) as e:
            logging.warning(f"Skipping invalid reasoning step {i}: {e}")
            continue
    
    if not processed_steps:
        return pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision", "Metrics"])
    
    try:
        # Create DataFrame and sort by step number
        columns = ["Step", "Content", "Confidence", "Decision", "Metrics"]
        df = pd.DataFrame(processed_steps)
        
        # Sort by step number if available
        if 'Step' in df.columns:
            df = df.sort_values('Step')
        
        # Apply styling
        def style_row(row):
            styles = [''] * len(row)
            if 'confidence_style' in row.index:
                styles[columns.index('Confidence')] = row['confidence_style']
            if 'decision_style' in row.index:
                styles[columns.index('Decision')] = row['decision_style']
            return styles
        
        # Apply styling and clean up
        styled_df = df[columns].style.apply(style_row, axis=1)
        
        # Set table properties
        styled_df = styled_df.set_properties(**{
            'white-space': 'pre-wrap',
            'text-align': 'left',
            'vertical-align': 'top',
            'padding': '8px',
            'border': '1px solid #e0e0e0'
        })
        
        # Format table headers
        styled_df = styled_df.set_table_styles([{
            'selector': 'th',
            'props': [
                ('background-color', '#f8f9fa'),
                ('color', '#2c3e50'),
                ('font-weight', 'bold'),
                ('padding', '10px'),
                ('text-align', 'left'),
                ('border-bottom', '2px solid #dee2e6')
            ]
        }, {
            'selector': 'tr:hover',
            'props': [('background-color', '#f1f9fe')]
        }])
        
        return styled_df
        
    except Exception as e:
        logging.error(f"Error creating reasoning steps table: {e}")
        return pd.DataFrame()


def create_interface():
    """Create and return the Gradio interface with enhanced visualizations and parameter controls."""
    # Set up logger
    logger = logging.getLogger("3WayCoT.UI")
    
    # Initialize the application
    app = ThreeWayCoTApp()
    gradio_app = Gradio3WayCoT()
    
    # Get configuration data
    config_manager = ConfigManager()
    config = config_manager.config
    
    # Get use cases from config
    use_cases = config.get('use_cases', {})
    use_case_names = list(use_cases.keys())
    if not use_case_names:
        use_case_names = ['default', 'conservative', 'exploratory']
    
    # Get LLM models from config
    llm_config = config.get('llm', {})
    providers = llm_config.get('providers', {})
    
    # Build model options dict
    model_options = {}
    for provider, details in providers.items():
        if 'models' in details:
            model_options[provider] = details['models']
    
    # Get all model names for dropdown
    all_models = []
    for provider_models in model_options.values():
        all_models.extend(provider_models)
    
    if not all_models:
        all_models = ['gemini-1.5-pro', 'gpt-4', 'claude-3-opus-20240229']
    
    # Get datasets path
    datasets_path = Path(config.get('benchmark.default_dataset', 'data/benchmarks/default.json')).parent
    
    # Load available datasets
    datasets = {}
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
    
    # Set up the theme
    theme = gr.themes.Default().set(
        body_background_fill="#f7f9fb",
        block_background_fill="#ffffff",
        button_primary_background_fill="#1f77b4",
    )
    # Create the interface
    with gr.Blocks(title="3WayCoT Framework", theme=gr.themes.Soft()) as interface:
        # Add header
        gr.Markdown("""
        # 3WayCoT Framework
        Interactive interface for the Three-Way Chain of Thought framework with configurable parameters.
        """)
        
        # Create tabs for different sections
        with gr.Tabs():
            # Main tab for prompt input and results
            with gr.Tab("Prompt & Results"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Input section
                        with gr.Group():
                            gr.Markdown("### Input")
                            prompt_input = gr.Textbox(
                                lines=5, 
                                placeholder="Enter your prompt here...", 
                                label="Prompt"
                            )
                            context_input = gr.Textbox(
                                lines=3, 
                                placeholder="Enter any context information here (optional)", 
                                label="Context (optional)"
                            )
                            
                            # Parameter selection tabs
                            with gr.Tabs():
                                with gr.TabItem("Quick Settings"):
                                    # Use case selection
                                    use_case_dropdown = gr.Dropdown(
                                        choices=use_case_names,
                                        value="default" if "default" in use_case_names else use_case_names[0],
                                        label="Use Case",
                                        info="Select a predefined parameter configuration"
                                    )
                                    use_case_description = gr.Markdown("""
                                    **Default**: Standard configuration for general use
                                    """)
                                    
                                    # Model selection
                                    model_dropdown = gr.Dropdown(
                                        choices=all_models,
                                        value=llm_config.get('default_model', all_models[0]),
                                        label="Model",
                                        info="Select the language model to use"
                                    )
                                    
                                with gr.TabItem("Advanced Parameters"):
                                    with gr.Row():
                                        with gr.Column():
                                            # Thresholds
                                            alpha_slider = gr.Slider(
                                                minimum=0.5, maximum=0.9, step=0.05, 
                                                value=config.get('framework.alpha', 0.7),
                                                label="Alpha Threshold",
                                                info="Higher values lead to more conservative decisions"
                                            )
                                            beta_slider = gr.Slider(
                                                minimum=0.3, maximum=0.7, step=0.05, 
                                                value=config.get('framework.beta', 0.4),
                                                label="Beta Threshold",
                                                info="Lower values increase rejection rate"
                                            )
                                            tau_slider = gr.Slider(
                                                minimum=0.3, maximum=0.8, step=0.05, 
                                                value=config.get('framework.tau', 0.6),
                                                label="Tau Threshold",
                                                info="Confidence value threshold"
                                            )
                                        
                                        with gr.Column():
                                            # LLM parameters
                                            temp_slider = gr.Slider(
                                                minimum=0.1, maximum=1.0, step=0.1, 
                                                value=0.7,
                                                label="Temperature",
                                                info="Controls randomness of the model output"
                                            )
                                            max_tokens_slider = gr.Slider(
                                                minimum=500, maximum=4000, step=500, 
                                                value=2000,
                                                label="Max Tokens",
                                                info="Maximum output length"
                                            )
                                            top_p_slider = gr.Slider(
                                                minimum=0.1, maximum=1.0, step=0.1, 
                                                value=1.0,
                                                label="Top P",
                                                info="Controls diversity of the model output"
                                            )
                                            
                                with gr.TabItem("Dataset"):
                                    # Dataset selection
                                    dataset_keys = list(datasets.keys())
                                    dataset_dropdown = gr.Dropdown(
                                        choices=dataset_keys,
                                        value=dataset_keys[0] if dataset_keys else None,
                                        label="Dataset",
                                        info="Select a dataset for batch processing",
                                        interactive=len(dataset_keys) > 0
                                    )
                                    dataset_info = gr.Markdown("""
                                    No datasets available. Add JSON files to the data/benchmarks directory.
                                    """ if not dataset_keys else f"""**{dataset_keys[0]}**: {datasets[dataset_keys[0]].get('items', 0)} items""")
                                    
                                    run_dataset_btn = gr.Button(
                                        "Run Dataset", 
                                        interactive=len(dataset_keys) > 0,
                                        variant="secondary"
                                    )
                            
                            # Control buttons
                            with gr.Row():
                                process_btn = gr.Button("Process", variant="primary")
                                clear_btn = gr.Button("Clear")
                                view_history_btn = gr.Button("View History")
                        
                        # Results section
                        with gr.Group() as results_box:
                            gr.Markdown("### Results")
                            results_output = gr.JSON(label="Processing Results")
                            
                            # Confidence visualization
                            with gr.Row():
                                confidence_plot = gr.Plot(label="Confidence Distribution")
                                decision_plot = gr.Plot(label="Decision Distribution")
                            
                            # Reasoning steps table
                            reasoning_steps_table = gr.Dataframe(
                                label="Reasoning Steps",
                                headers=["Step", "Content", "Confidence", "Decision"],
                                datatype=["number", "str", "number", "str"],
                                interactive=False
                            )
                    
                    # Parameters panel
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Decision Parameters")
                            
                            # Decision thresholds
                            alpha_slider = gr.Slider(
                                minimum=0.5, 
                                maximum=1.0, 
                                value=app.default_params['alpha'],
                                step=0.05,
                                label="Alpha (Acceptance Threshold)",
                                info="Higher values require more confidence to accept a step"
                            )
                            
                            beta_slider = gr.Slider(
                                minimum=0.0, 
                                maximum=0.5, 
                                value=app.default_params['beta'],
                                step=0.05,
                                label="Beta (Rejection Threshold)",
                                info="Lower values make it harder to reject a step"
                            )
                            
                            tau_slider = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=app.default_params['tau'],
                                step=0.1,
                                label="Tau (Similarity Threshold)",
                                info="Threshold for concept similarity"
                            )
                            
                            # LLM Parameters
                            gr.Markdown("### LLM Parameters")
                            
                            temp_slider = gr.Slider(
                                minimum=0.0, 
                                maximum=2.0, 
                                value=app.default_params['temperature'],
                                step=0.1,
                                label="Temperature",
                                info="Higher values make output more random"
                            )
                            
                            max_tokens_slider = gr.Slider(
                                minimum=100, 
                                maximum=4000, 
                                value=app.default_params['max_tokens'],
                                step=100,
                                label="Max Tokens",
                                info="Maximum number of tokens to generate"
                            )
                            
                            top_p_slider = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=app.default_params['top_p'],
                                step=0.1,
                                label="Top-p (Nucleus Sampling)",
                                info="Controls diversity of generated text"
                            )
                            
                            # Use Case Selection
                            gr.Markdown("### Use Case Presets")
                            use_case_dropdown = gr.Dropdown(
                                choices=list(app.use_cases.keys()),
                                value='balanced' if 'balanced' in app.use_cases else list(app.use_cases.keys())[0],
                                label="Select Use Case",
                                interactive=True
                            )
                            
                            # Display use case description
                            use_case_desc = gr.Markdown(
                                value=app.use_cases.get('balanced', {}).get('description', '') 
                                if 'balanced' in app.use_cases 
                                else list(app.use_cases.values())[0].get('description', '') if app.use_cases else ''
                            )
                            
                            # Update description when use case changes
                            def update_use_case_desc(use_case):
                                return app.use_cases.get(use_case, {}).get('description', '')
                                
                            use_case_dropdown.change(
                                fn=update_use_case_desc,
                                inputs=use_case_dropdown,
                                outputs=use_case_desc
                            )
                            
                            # Benchmark Dataset Selection
                            if app.benchmark_datasets:
                                gr.Markdown("### Benchmark Datasets")
                                benchmark_dropdown = gr.Dropdown(
                                    choices=[(v['name'], k) for k, v in app.benchmark_datasets.items()],
                                    label="Select Benchmark Dataset",
                                    interactive=True
                                )
                                
                                # Display benchmark info
                                benchmark_info = gr.JSON(
                                    label="Benchmark Info",
                                    visible=bool(app.benchmark_datasets)
                                )
                                
                                def update_benchmark_info(dataset_key):
                                    if not dataset_key or dataset_key not in app.benchmark_datasets:
                                        return {}
                                    dataset = app.benchmark_datasets[dataset_key]
                                    return {
                                        'name': dataset['name'],
                                        'examples': dataset['num_examples'],
                                        'metrics': dataset['metrics'],
                                        'description': dataset['description']
                                    }
                                    
                                benchmark_dropdown.change(
                                    fn=update_benchmark_info,
                                    inputs=benchmark_dropdown,
                                    outputs=benchmark_info
                                )
            
            # History tab
            with gr.Tab("History"):
                history_table = gr.Dataframe(
                    label="Prompt History",
                    headers=["Timestamp", "Prompt", "Context", "Status", "Result"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    type="pandas"
                )
                view_history_btn = gr.Button("Load History")
        # Event handlers
        def process_prompt(prompt, context, alpha, beta, tau, temp, max_tokens_slider, top_p):
            """Process the prompt with the given parameters and update visualizations."""
            try:
                # Ensure parameters are of correct type
                alpha = float(alpha) if alpha is not None else 0.7
                beta = float(beta) if beta is not None else 0.6
                tau = float(tau) if tau is not None else 0.4
                temperature = float(temp) if temp is not None else 0.7
                max_tokens = int(max_tokens_slider) if max_tokens_slider is not None else 1000
                top_p = float(top_p) if top_p is not None else 1.0
                
                # Log the parameters being used
                logger.info(f"Processing with params - alpha: {alpha}, beta: {beta}, tau: {tau}, "
                          f"temp: {temperature}, max_tokens: {max_tokens}, top_p: {top_p}")
                
                # Process the prompt and get results
                result = app.process_prompt(
                    prompt_text=prompt,
                    context_text=context,
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                # Get visualization data
                vis_data = result.get('visualization_data', {}) if result else {}
                
                # Create visualizations
                confidence_plot = None
                decision_plot = None
                reasoning_table = pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
                
                # Format the final answer
                final_answer = "No answer generated"
                if result and 'result' in result:
                    if isinstance(result['result'], dict):
                        final_answer = result['result'].get('final_answer', final_answer)
                        if isinstance(final_answer, dict):
                            final_answer = final_answer.get('text', str(final_answer))
                    else:
                        final_answer = str(result['result'])
                
                try:
                    if 'confidence_metrics' in vis_data and vis_data['confidence_metrics']:
                        confidence_plot = create_confidence_plot(vis_data['confidence_metrics'])
                    
                    if 'decision_metrics' in vis_data and vis_data['decision_metrics']:
                        decision_plot = create_decision_plot(vis_data['decision_metrics'])
                    
                    if 'reasoning_steps' in vis_data and vis_data['reasoning_steps']:
                        reasoning_table = create_reasoning_steps_table(vis_data['reasoning_steps'])
                except Exception as e:
                    logger.error(f"Error creating visualizations: {str(e)}")
                
                return [
                    final_answer,
                    confidence_plot,
                    decision_plot,
                    reasoning_table
                ]
                
            except Exception as e:
                error_msg = f"Error processing prompt: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [
                    error_msg,
                    None,
                    None,
                    pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
                ]
        def load_history():
            """Load and display the prompt history."""
            history = []
            for item in app.history:
                # Extract the result, handling both dictionary and string results
                result = item.get('result', {})
                if isinstance(result, dict):
                    result_str = result.get('answer', str(result))
                else:
                    result_str = str(result)[:200] + ('...' if len(str(result)) > 200 else '')
                
                history.append([
                    item.get('timestamp', 'N/A'),
                    item.get('prompt', '')[:100] + ('...' if len(item.get('prompt', '')) > 100 else ''),
                    item.get('context', '')[:50] + ('...' if len(item.get('context', '')) > 50 else '') if item.get('context') else '',
                    'Success' if isinstance(result, dict) and result.get('status') == 'success' else 'Error',
                    result_str
                ])
            return history
        
        def apply_use_case(use_case_name):
            """Apply a use case configuration."""
            use_case = app.use_cases.get(use_case_name, app.use_cases[list(app.use_cases.keys())[0]] if app.use_cases else {})
            return [
                use_case.get('alpha', app.default_params['alpha']),
                use_case.get('beta', app.default_params['beta']),
                use_case.get('tau', app.default_params['tau']),
                use_case.get('temperature', app.default_params['temperature']),
                use_case.get('max_tokens', app.default_params.get('max_tokens', 2000)),
                app.default_params.get('top_p', 1.0)
            ]
        
        # Connect UI elements to event handlers
        submit_btn.click(
            fn=process_prompt,
            inputs=[
                prompt_input,
                context_input,
                alpha_slider,
                beta_slider,
                tau_slider,
                temp_slider,
                max_tokens_slider,
                top_p_slider
            ],
            outputs=[
                results_output,
                confidence_plot,
                decision_plot,
                reasoning_steps_table
            ]
        )
        
        clear_btn.click(
            fn=lambda: [
                "",  # Clear results_output
                None,  # Clear confidence_plot
                None,  # Clear decision_plot
                pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])  # Clear reasoning_steps_table
            ],
            outputs=[
                results_output,
                confidence_plot,
                decision_plot,
                reasoning_steps_table
            ]
        )
        
        def load_history():
            """Load and format the history of prompts and results."""
            history = []
            for item in app.history:
                history.append([
                    item['timestamp'],
                    item['prompt'][:100] + ('...' if len(item['prompt']) > 100 else ''),
                    item['context'][:50] + ('...' if len(item['context']) > 50 else '') if item['context'] else '',
                    'Success' if item['result'].get('status') == 'success' else 'Error',
                    json.dumps(item['result'], indent=2)[:200] + ('...' if len(json.dumps(item['result'], indent=2)) > 200 else '')
                ])
            return history
            
        view_history_btn.click(
            fn=load_history,
            outputs=history_table
        )
        
        # Connect use case selection
        def apply_use_case(use_case_name):
            use_case = app.use_cases.get(use_case_name, {})
            return [
                use_case.get('alpha', app.default_params['alpha']),
                use_case.get('beta', app.default_params['beta']),
                use_case.get('tau', app.default_params['tau']),
                use_case.get('temperature', app.default_params['temperature']),
                use_case.get('max_tokens', app.default_params.get('max_tokens', 2000)),
                use_case.get('top_p', app.default_params.get('top_p', 1.0))
            ]
            
        use_case_dropdown.change(
            fn=apply_use_case,
            inputs=use_case_dropdown,
            outputs=[
                alpha_slider,
                beta_slider,
                tau_slider,
                temp_slider,
                max_tokens_slider,
                top_p_slider
            ]
        )
        
        # Set initial state
        interface.load(load_history, outputs=history_table)
    
    # Define custom CSS for better styling
    custom_css = """
    /* Smooth Scroll and Selection */
    html {
        scroll-behavior: smooth;
    }
    
    ::selection {
        background: rgba(52, 152, 219, 0.3);
        color: #2c3e50;
    }
    
    /* Animation for loading state */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
        opacity: 0.7;
    }
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    /* General Styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #2c3e50;
    }
    
    /* Input Section */
    .input-section {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Results Section */
    .results-section {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Metrics Display */
    .metrics-display {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .metrics-display .label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-bottom: 0.25rem;
    }
    .metrics-display .value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Decision Output */
    .decision-output {
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .decision-accept {
        background-color: #e8f5e9;
        color: #27ae60;
        border-left: 4px solid #27ae60;
    }
    .decision-reject {
        background-color: #ffebee;
        color: #e74c3c;
        border-left: 4px solid #e74c3c;
    }
    .decision-abstain {
        background-color: #fff8e1;
        color: #f39c12;
        border-left: 4px solid #f39c12;
    }
    
    /* Tabs */
    .tab-nav {
        margin-bottom: 1rem;
    }
    .tab-nav button {
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        border: 1px solid #e0e0e0;
        background: #f8f9fa;
        border-radius: 4px;
        cursor: pointer;
    }
    .tab-nav button.active {
        background: #3498db;
        color: white;
        border-color: #3498db;
    }
    
    /* Buttons */
    .btn-primary {
        background: #3498db !important;
        border: none !important;
        color: white !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    .btn-secondary {
        background: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        color: #2c3e50 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metrics-row {
            flex-direction: column;
        }
        .metrics-display {
            width: 100%;
            margin: 0.25rem 0;
        }
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            spacing_size="sm",
            radius_size="md"
        ),
        css=custom_css
    ) as demo:
        # Header
        gr.Markdown(
            """
            # 🧠 3WayCoT Framework
            Interactive interface for the 3WayCoT framework with Chain-of-Thought reasoning 
            and Three-Way Decision making.
            """
        )
        
        with gr.Row(equal_height=False):
            # Left column - Input and Results
            with gr.Column(scale=2):
                # Input Section
                with gr.Group(elem_classes=["input-section"]):
                    gr.Markdown("### 📝 Input")
                    prompt = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Type your question or instruction here...",
                        lines=3,
                        max_lines=5,
                        container=True
                    )
                    context = gr.Textbox(
                        label="Additional context (optional)",
                        placeholder="Provide any additional context here...",
                        lines=2,
                        max_lines=3,
                        container=True
                    )
                    with gr.Row():
                        clear_btn = gr.Button(
                            "Clear", 
                            variant="secondary",
                            elem_classes=["btn-secondary"]
                        )
                        submit_btn = gr.Button(
                            "Process", 
                            variant="primary",
                            elem_classes=["btn-primary"]
                        )
                
                # Results Section
                with gr.Group(elem_classes=["results-section"]):
                    gr.Markdown("### 📊 Results")
                    
                    # Decision and Confidence Score
                    with gr.Row():
                        with gr.Column(scale=1):
                            decision_output = gr.HTML(
                                label="Final Decision",
                                value="<div class='decision-output decision-pending'>No decision yet</div>",
                                visible=True,
                                elem_classes=["decision-container"]
                            )
                        with gr.Column(scale=1):
                            confidence_score = gr.HTML(
                                label="Confidence Score",
                                value="<div class='metrics-display'><div class='label'>Confidence</div><div class='value'>0%</div></div>",
                                visible=True,
                                elem_classes=["confidence-container"]
                            )
                    
                    # Metrics summary
                    with gr.Row(elem_classes=["metrics-row"]):
                        with gr.Column(min_width=150):
                            accept_rate = gr.HTML(
                                value="<div class='metrics-display'><div class='label'>Acceptance Rate</div><div class='value'>0%</div></div>",
                                visible=True
                            )
                        with gr.Column(min_width=150):
                            avg_confidence = gr.HTML(
                                value="<div class='metrics-display'><div class='label'>Avg. Confidence</div><div class='value'>0%</div></div>",
                                visible=True
                            )
                        with gr.Column(min_width=150):
                            consistency = gr.HTML(
                                value="<div class='metrics-display'><div class='label'>Consistency</div><div class='value'>0%</div></div>",
                                visible=True
                            )
                    
                    # Detailed reasoning output
                    with gr.Group():
                        gr.Markdown("### 📝 Detailed Reasoning")
                        reasoning_output = gr.Markdown(
                            value="*Processing will begin when you submit your prompt...*",
                            visible=True,
                            elem_classes=["reasoning-output"]
                        )
            
            # Right column - Visualizations
            with gr.Column(scale=3):
                with gr.Tabs(elem_classes=["tab-nav"]) as tabs:
                    # Confidence Analysis Tab
                    with gr.Tab("📊 Confidence Analysis"):
                        confidence_plot = gr.Plot(
                            label="Confidence Metrics",
                            show_label=True,
                            container=True,
                            elem_classes=["visualization-plot"]
                        )
                    
                    # Decision Analysis Tab
                    with gr.Tab("🎯 Decision Analysis"):
                        with gr.Tabs():
                            with gr.Tab("2D View"):
                                decision_plot = gr.Plot(
                                    label="Decision Metrics",
                                    show_label=True,
                                    container=True,
                                    elem_classes=["visualization-plot"]
                                )
                            with gr.Tab("3D Lattice View"):
                                lattice_plot = gr.Plot(
                                    label="Decision Space Lattice",
                                    show_label=True,
                                    container=True,
                                    elem_classes=["visualization-plot"]
                                )
                    
                    # Reasoning Steps Tab
                    with gr.Tab("🔍 Reasoning Steps"):
                        reasoning_table = gr.HTML(
                            value="<div class='empty-state'>Reasoning steps will appear here after processing.</div>",
                            show_label=False,
                            elem_classes=["reasoning-steps-container"]
                        )
                    
                    # Raw Output Tab
                    with gr.Tab("📝 Raw Output"):
                        raw_output = gr.JSON(
                            value={"status": "No data available yet. Process a prompt to see the raw output."},
                            show_label=False,
                            container=True,
                            elem_classes=["raw-output-container"],
                            visible=True
                        )

        # Define the processing function
        def process(prompt: str, context: str) -> Dict[str, Any]:
            """Process the prompt and return the results with enhanced visualizations."""
            # Initialize app if not already done
            if not hasattr(process, 'app'):
                process.app = Gradio3WayCoT()
                
            # Process the prompt
            result = process.app.process_prompt(
                prompt_text=prompt,
                context_text=context,
                include_reasoning_steps=True,
                include_confidence=True
            )
            
            # Get visualization data
            vis_data = process.app.get_visualization_data(result)
            
            # Create visualizations
            confidence_plot_fig = create_confidence_plot(vis_data.get('confidence_metrics', {}))
            decision_plot_fig = create_decision_plot(vis_data.get('decision_metrics', {}))
            lattice_plot_fig = create_lattice_visualization(vis_data.get('decision_metrics', {}))
            
            # Create reasoning steps table
            reasoning_df = create_reasoning_steps_table(vis_data.get('reasoning_steps', []))
            reasoning_html = reasoning_df.to_html(classes='reasoning-steps-table', index=False, escape=False) \
                if not reasoning_df.empty else "<div class='empty-state'>No reasoning steps available.</div>"
            
            # Prepare metrics
            metrics = {
                'decision': vis_data.get('final_decision', 'Unknown'),
                'confidence': vis_data.get('confidence_metrics', {}).get('confidence', 0),
                'accept_rate': vis_data.get('decision_metrics', {}).get('accept', 0),
                'avg_confidence': vis_data.get('confidence_metrics', {}).get('avg_confidence', 0),
                'consistency': vis_data.get('decision_metrics', {}).get('decision_consistency', 0)
            }
            
            # Format metrics for display
            formatted_metrics = {
                'decision': f"<div class='decision-output decision-{metrics['decision'].lower()}'>{metrics['decision']}</div>",
                'confidence': f"<div class='metrics-display'><div class='label'>Confidence</div><div class='value'>{metrics['confidence']*100:.1f}%</div></div>",
                'accept_rate': f"<div class='metrics-display'><div class='label'>Acceptance Rate</div><div class='value'>{metrics['accept_rate']*100:.1f}%</div></div>",
                'avg_confidence': f"<div class='metrics-display'><div class='label'>Avg. Confidence</div><div class='value'>{metrics['avg_confidence']*100:.1f}%</div></div>",
                'consistency': f"<div class='metrics-display'><div class='label'>Consistency</div><div class='value'>{metrics['consistency']*100:.1f}%</div>"
            }
            
            return {
                'decision_output': formatted_metrics['decision'],
                'confidence_score': formatted_metrics['confidence'],
                'accept_rate': formatted_metrics['accept_rate'],
                'avg_confidence': formatted_metrics['avg_confidence'],
                'consistency': formatted_metrics['consistency'],
                'reasoning_output': reasoning_html,
                'confidence_plot': confidence_plot_fig,
                'decision_plot': decision_plot_fig,
                'lattice_plot': lattice_plot_fig,
                'raw_output': result
            }
            import logging
            logger = logging.getLogger('3WayCoT.GradioApp')
            if not prompt.strip():
                raise gr.Error("❌ Please enter a prompt to process.")

            try:
                # Process the prompt using the 3WayCoT framework
                logger.info("Starting to process prompt...")
                result = app.process_prompt(prompt, context)
                logger.info("Successfully processed prompt")
                confidence_plot_fig = create_confidence_plot(confidence_metrics)
                decision_plot_fig = create_decision_plot(decision_metrics)

                # Create reasoning steps table
                reasoning_df = create_reasoning_steps_table(reasoning_steps)
                
                # Check if DataFrame is empty and handle Styler object
                if hasattr(reasoning_df, 'data') and not reasoning_df.data.empty:
                    reasoning_html = reasoning_df.to_html(classes='reasoning-steps-table', index=False, escape=False)
                elif hasattr(reasoning_df, 'empty') and not reasoning_df.empty:
                    reasoning_html = reasoning_df.to_html(classes='reasoning-steps-table', index=False, escape=False)
                else:
                    reasoning_html = "<div class='empty-state'>No reasoning steps available.</div>"
                
                # Wrap in styled container
                reasoning_html = f"""
                <style>
                    .reasoning-steps-container {{
                        max-height: 400px;
                        overflow-y: auto;
                        margin: 1rem 0;
                        border: 1px solid #e2e8f0;
                        border-radius: 0.5rem;
                        padding: 1rem;
                    }}
                    .reasoning-steps-table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .reasoning-steps-table th, .reasoning-steps-table td {{
                        padding: 0.75rem;
                        text-align: left;
                        border-bottom: 1px solid #e2e8f0;
                    }}
                    .reasoning-steps-table th {{
                        background-color: #f8fafc;
                        font-weight: 600;
                        text-transform: uppercase;
                        font-size: 0.75rem;
                        letter-spacing: 0.05em;
                        color: #64748b;
                    }}
                    .reasoning-steps-table tr:nth-child(even) {{
                        background-color: #f9fafc;
                    }}
                    .reasoning-steps-table tr:hover {{
                        background-color: #f1f5f9;
                    }}
                    .reasoning-steps-table .step-content {{
                        white-space: pre-wrap;
                        line-height: 1.5;
                    }}
                    .empty-state {{
                        color: #64748b;
                        font-style: italic;
                        text-align: center;
                        padding: 1rem;
                    }}
                </style>
                <div class="reasoning-steps-container">
                    {reasoning_html}
                </div>
                """

                # Format the final decision with styling
                final_decision = vis_data.get('final_decision', 'No decision made')
                decision_style = ""
                if final_decision.upper() == 'ACCEPT':
                    decision_style = "color: #27ae60; font-weight: bold;"
                elif final_decision.upper() == 'REJECT':
                    decision_style = "color: #e74c3c; font-weight: bold;"
                else:  # ABSTAIN or other
                    decision_style = "color: #f39c12; font-weight: bold;"

                # Prepare the final output
                output = {
                    'decision': f"<div style='{decision_style}'>{final_decision}</div>",
                    'confidence_score': f"{confidence_metrics.get('avg_confidence', 0):.1%}",
                    'reasoning': vis_data.get('reasoning', 'No reasoning available'),
                    'confidence_plot': confidence_plot_fig,
                    'decision_plot': decision_plot_fig,
                    'reasoning_table': reasoning_html,
                    'raw_output': result,
                    'metrics': {
                        'accept_rate': f"{decision_metrics.get('accept_ratio', 0):.1%}",
                        'avg_confidence': f"{confidence_metrics.get('avg_confidence', 0):.1%}",
                        'consistency': f"{confidence_metrics.get('decision_consistency', 0):.1%}"
                    }
                }

                return output

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                logging.exception(error_msg)
                return {
                    'decision': "❌ Error",
                    'confidence_score': "0%",
                    'reasoning': error_msg,
                    'metrics': {
                        'accept_rate': "0%",
                        'avg_confidence': "0%",
                        'consistency': "0%"
                    },
                    'error': str(e)
                }

        def format_decision_html(decision, confidence):
            """Format the decision with appropriate styling."""
            decision_lower = str(decision).lower()
            if 'accept' in decision_lower:
                decision_class = 'decision-accept'
                icon = '✅'
            elif 'reject' in decision_lower:
                decision_class = 'decision-reject'
                icon = '❌'
            else:  # abstain or unknown
                decision_class = 'decision-abstain'
                icon = '❓'
            
            return f"""
            <div class='decision-output {decision_class}'>
                <div class='decision-content'>
                    <span class='decision-icon'>{icon}</span>
                    <span class='decision-text'>{decision}</span>
                    <span class='confidence-badge'>{confidence} confidence</span>
                </div>
            </div>
            """
        
        def format_metrics_html(label, value):
            """Format a metric value with its label."""
            return f"""
            <div class='metrics-display'>
                <div class='label'>{label}</div>
                <div class='value'>{value}</div>
            </div>
            """
        
        def process_and_display(prompt_text, context_text):
            """Process the input and update all UI components."""
            # Set up logger for this function
            logger = logging.getLogger('3WayCoT.GradioApp.process')
            
            try:
                # Call the main processing function
                logger.info("\n" + "="*80)
                logger.info("STARTING PROCESS_AND_DISPLAY")
                logger.info("="*80)
                
                # Get current parameter values from the UI
                params = {
                    'alpha': alpha_slider,
                    'beta': beta_slider,
                    'tau': tau_slider,
                    'temperature': temp_slider,
                    'max_tokens': max_tokens_slider,
                    'top_p': top_p_slider,
                    'model': model_dropdown,
                    'provider': provider_dropdown
                }
                
                # Process the prompt using the 3WayCoT framework
                result = app.process_prompt(
                    prompt_text,
                    context=context_text,
                    **{k: v for k, v in params.items() if k in ['temperature', 'max_tokens', 'top_p']}
                )
                
                # Initialize default values for all return values
                decision_output = "<div class='decision-output'>No decision generated</div>"
                confidence_score = format_metrics_html('Confidence Score', 'N/A')
                reasoning_output = "<div class='reasoning-steps'>No reasoning steps generated</div>"
                accept_rate = format_metrics_html('Accept Rate', 'N/A')
                avg_confidence = format_metrics_html('Average Confidence', 'N/A')
                consistency = format_metrics_html('Consistency', 'N/A')
                reasoning_table = pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
                
                # Create empty figures for plots
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No data available",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    annotations=[dict(
                        text="No visualization data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )]
                )
                confidence_plot = empty_fig
                decision_plot = empty_fig
                lattice_plot = None
                
                # Check if the result is an error
                if result.get('status') == 'error':
                    error_msg = result.get('error', 'Unknown error occurred')
                    logger.error(f"Error processing prompt: {error_msg}")
                    decision_output = f"<div class='decision-output decision-error'>{error_msg}</div>"
                    return [
                        decision_output, confidence_score, reasoning_output, 
                        accept_rate, avg_confidence, consistency,
                        confidence_plot, decision_plot, lattice_plot,
                        reasoning_table, result
                    ]
                
                # Check if we have a valid result with reasoning steps
                if not result.get('result') or not isinstance(result['result'], dict):
                    error_msg = "No valid result returned from the LLM. The model may not have generated any reasoning steps."
                    logger.error(error_msg)
                    decision_output = f"<div class='decision-output decision-error'>{error_msg}</div>"
                    return [
                        decision_output, confidence_score, reasoning_output, 
                        accept_rate, avg_confidence, consistency,
                        confidence_plot, decision_plot, lattice_plot,
                        reasoning_table, result
                    ]
                
                try:
                    # Extract the result data
                    result_data = result['result']
                    
                    # Get reasoning steps or default to empty list
                    reasoning_steps = result_data.get('reasoning_steps', [])
                    if not isinstance(reasoning_steps, list):
                        reasoning_steps = []
                    
                    # Get final answer or use a default message
                    final_answer = result_data.get('final_answer', 'No answer generated')
                    decision_output = f"<div class='decision-output'>{final_answer}</div>"
                    
                    # Initialize metrics
                    confidence_metrics = {}
                    decision_metrics = {}
                    
                    # Extract visualization data if available
                    if 'visualization' in result_data:
                        visualization_data = result_data['visualization']
                        if isinstance(visualization_data, dict):
                            confidence_metrics = visualization_data.get('confidence_metrics', {})
                            decision_metrics = visualization_data.get('decision_metrics', {})
                    
                    # Format confidence score if available
                    if confidence_metrics and 'average_confidence' in confidence_metrics:
                        confidence_score = format_metrics_html(
                            'Confidence Score', 
                            f"{confidence_metrics['average_confidence'] * 100:.1f}%"
                        )
                    
                    # Format reasoning steps
                    if reasoning_steps:
                        reasoning_output = "<div class='reasoning-steps'>"
                        for i, step in enumerate(reasoning_steps, 1):
                            # Handle different possible step formats
                            if isinstance(step, dict):
                                reasoning_text = step.get('reasoning', '').strip()
                                if not reasoning_text:
                                    reasoning_text = step.get('content', '').strip()
                                assumptions = step.get('assumptions', [])
                                if not isinstance(assumptions, list):
                                    assumptions = [assumptions] if assumptions else []
                                
                                reasoning_output += f"<div class='reasoning-step'><strong>Step {i}:</strong> {reasoning_text}"
                                if assumptions:
                                    reasoning_output += "<div class='assumptions'><strong>Assumptions:</strong><ul>"
                                    for assumption in assumptions:
                                        if isinstance(assumption, str):
                                            reasoning_output += f"<li>{assumption}</li>"
                                        elif isinstance(assumption, dict) and 'content' in assumption:
                                            reasoning_output += f"<li>{assumption['content']}</li>"
                                    reasoning_output += "</ul></div>"
                                reasoning_output += "</div>"
                            elif isinstance(step, str):
                                reasoning_output += f"<div class='reasoning-step'><strong>Step {i}:</strong> {step}</div>"
                        reasoning_output += "</div>"
                    
                    # Prepare metrics for display
                    if decision_metrics:
                        accept_rate = format_metrics_html(
                            'Accept Rate', 
                            f"{decision_metrics.get('accept_rate', 0) * 100:.1f}%"
                        )
                    
                    if confidence_metrics:
                        avg_confidence = format_metrics_html(
                            'Average Confidence', 
                            f"{confidence_metrics.get('average_confidence', 0) * 100:.1f}%"
                        )
                        
                        consistency = format_metrics_html(
                            'Consistency', 
                            f"{confidence_metrics.get('consistency', 0) * 100:.1f}%"
                        )
                    
                    # Create visualizations if we have data
                    if confidence_metrics:
                        try:
                            confidence_plot = create_confidence_plot(confidence_metrics)
                        except Exception as e:
                            logger.error(f"Error creating confidence plot: {e}")
                    
                    if decision_metrics:
                        try:
                            decision_plot = create_decision_plot(decision_metrics)
                        except Exception as e:
                            logger.error(f"Error creating decision plot: {e}")
                    
                    # Create lattice visualization if we have enough data
                    if len(reasoning_steps) >= 2:
                        try:
                            lattice_plot = create_lattice_visualization(reasoning_steps)
                        except Exception as e:
                            logger.error(f"Error creating lattice visualization: {e}")
                    
                    # Create a table of reasoning steps for better display
                    reasoning_table_data = []
                    for i, step in enumerate(reasoning_steps, 1):
                        if isinstance(step, dict):
                            reasoning_text = step.get('reasoning', step.get('content', '')).replace('\n', ' ').strip()
                            confidence = step.get('confidence', 0)
                            decision = step.get('decision', 'N/A')
                            reasoning_table_data.append([
                                f"Step {i}",
                                reasoning_text[:200] + ('...' if len(reasoning_text) > 200 else ''),
                                f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else 'N/A',
                                decision
                            ])
                        elif isinstance(step, str):
                            reasoning_table_data.append([
                                f"Step {i}",
                                step[:200] + ('...' if len(step) > 200 else ''),
                                'N/A',
                                'N/A'
                            ])
                    
                    if reasoning_table_data:
                        reasoning_table = pd.DataFrame(
                            reasoning_table_data,
                            columns=["Step", "Content", "Confidence", "Decision"]
                        )
                    
                except Exception as e:
                    error_msg = f"Error processing results: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    decision_output = f"<div class='decision-output decision-error'>{error_msg}</div>"
                
                # Debug: Log available metrics with more details
                logger.info("\n" + "="*80)
                logger.info("PROCESSING CONFIDENCE SCORE")
                logger.info("="*80)
                logger.info(f"🔍 Available metrics keys: {list(metrics.keys())}")
                
                # Log all confidence-related metrics
                conf_metrics = {k: v for k, v in metrics.items() if any(term in k.lower() for term in ['conf', 'certain', 'score'])}
                logger.info(f"🔍 Confidence-related metrics: {conf_metrics}")
                
                # Try to get confidence score from various possible locations with priority
                confidence_sources = [
                    ('direct_confidence', metrics.get('confidence')),
                    ('avg_confidence', metrics.get('avg_confidence', metrics.get('average_confidence'))),
                    ('accept_confidence', metrics.get('accept_confidence')),
                    ('confidence_metrics', metrics.get('confidence_metrics', {}).get('confidence')),
                    ('any_confidence_key', next((v for k, v in metrics.items() if 'conf' in k.lower() or 'score' in k.lower()), None))
                ]
                
                # Log all potential sources
                logger.info("🔍 Potential confidence sources:")
                for name, value in confidence_sources:
                    logger.info(f"   - {name}: {value} (type: {type(value).__name__ if value is not None else 'None'})")
                
                # Get the first non-None value
                confidence_score_value = None
                source = "default"
                for name, value in confidence_sources:
                    if value is not None:
                        confidence_score_value = value
                        source = name
                        break
                
                # If still no confidence score found, try to calculate from decision metrics
                if confidence_score_value is None and 'decisions' in result:
                    try:
                        decisions = result['decisions'].get('decisions', [])
                        if decisions:
                            confidences = [d.get('confidence', 0) for d in decisions if isinstance(d, dict) and 'confidence' in d]
                            if confidences:
                                confidence_score_value = sum(confidences) / len(confidences)
                                source = 'average_from_decisions'
                                logger.info(f"Calculated average confidence from {len(confidences)} decisions: {confidence_score_value}")
                    except Exception as e:
                        logger.warning(f"Error calculating confidence from decisions: {e}")
                
                if confidence_score_value is None:
                    confidence_score_value = 0.5
                    logger.warning("⚠️  No confidence score found, using default 0.5")
                else:
                    logger.info(f"✅ Using confidence score from {source}: {confidence_score_value}")
                
                # Ensure it's a float between 0 and 1
                try:
                    if isinstance(confidence_score_value, str):
                        # Remove percentage sign if present
                        if '%' in confidence_score_value:
                            confidence_score_value = confidence_score_value.replace('%', '').strip()
                        # Convert to float and handle percentage
                        num = float(confidence_score_value)
                        if num > 1.0:  # If it's a percentage (e.g., 50%)
                            num = num / 100.0
                        confidence_score_value = num
                    
                    confidence_score_value = float(confidence_score_value)
                    confidence_score_value = max(0.0, min(1.0, confidence_score_value))  # Clamp between 0 and 1
                    logger.info(f"🔢 Processed confidence score: {confidence_score_value:.4f}")
                except (ValueError, TypeError) as e:
                    logger.error(f"❌ Failed to parse confidence score '{confidence_score_value}': {e}")
                    confidence_score_value = 0.5  # Fallback if conversion fails
                
                # Format for display (as percentage)
                confidence_display = format_percentage(confidence_score_value)
                logger.info(f"📊 Formatted confidence display: {confidence_display}")
                logger.info("-"*80)
                
                # Update the confidence score with the processed value
                confidence_score = format_metrics_html('Confidence Score', f"{confidence_display}%")
                
                return [
                    decision_output,  # decision_output
                    confidence_score,  # confidence_score
                    reasoning_output,  # reasoning_output
                    accept_rate,  # accept_rate
                    avg_confidence,  # avg_confidence
                    consistency,  # consistency
                    confidence_plot,  # confidence_plot
                    decision_plot,  # decision_plot
                    lattice_plot,  # lattice_plot
                    reasoning_table,  # reasoning_table
                    result  # raw_output
                ]
                
                decision_html = f"""
                <div class='decision-output {decision_class}'>
                    <div class='decision-content'>
                        <span class='decision-icon'>{decision_icon}</span>
                        <span class='decision-text'>{final_decision}</span>
                        <span class='confidence-badge'>{confidence_text} confidence</span>
                    </div>
                </div>
                """
                
                # Prepare metrics HTML with safe formatting
                accept_rate = metrics.get('accept_ratio', metrics.get('accept_rate', 0))
                consistency = metrics.get('decision_consistency', metrics.get('consistency', 0))
                
                accept_rate_html = format_metrics_html('Acceptance Rate', format_percentage(accept_rate))
                avg_conf_html = format_metrics_html('Avg. Confidence', format_percentage(metrics.get('avg_confidence', 0)))
                consistency_html = format_metrics_html('Consistency', format_percentage(consistency))
                
                # Prepare the reasoning output
                reasoning = result.get('reasoning', 'No reasoning available')
                if not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                
                # Prepare the raw output
                raw_output = result.get('raw_output', {})
                if not isinstance(raw_output, (dict, list)):
                    raw_output = {'output': str(raw_output)}
                
                # Create reasoning steps table
                reasoning_steps = result.get('decisions', [])
                reasoning_df = create_reasoning_steps_table(reasoning_steps)
                
                # Check if DataFrame is empty and handle Styler object
                if hasattr(reasoning_df, 'data') and not reasoning_df.data.empty:
                    reasoning_html = reasoning_df.to_html(classes='reasoning-steps-table', index=False, escape=False)
                elif hasattr(reasoning_df, 'empty') and not reasoning_df.empty:
                    reasoning_html = reasoning_df.to_html(classes='reasoning-steps-table', index=False, escape=False)
                else:
                    reasoning_html = "<div class='empty-state'>No reasoning steps available.</div>"
                
                # Wrap the reasoning steps table in a styled container
                reasoning_table_html = f"""
                <style>
                    .reasoning-steps-container {{
                        max-height: 400px;
                        overflow-y: auto;
                        margin: 1rem 0;
                        border: 1px solid #e2e8f0;
                        border-radius: 0.5rem;
                        padding: 1rem;
                    }}
                    .reasoning-steps-table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .reasoning-steps-table th, .reasoning-steps-table td {{
                        padding: 0.75rem;
                        text-align: left;
                        border-bottom: 1px solid #e2e8f0;
                    }}
                    .reasoning-steps-table th {{
                        background-color: #f8fafc;
                        font-weight: 600;
                        text-transform: uppercase;
                        font-size: 0.75rem;
                        letter-spacing: 0.05em;
                        color: #64748b;
                    }}
                    .reasoning-steps-table tr:nth-child(even) {{
                        background-color: #f9fafc;
                    }}
                    .reasoning-steps-table tr:hover {{
                        background-color: #f1f5f9;
                    }}
                    .reasoning-steps-table .step-content {{
                        white-space: pre-wrap;
                        line-height: 1.5;
                    }}
                    .empty-state {{
                        color: #64748b;
                        font-style: italic;
                        text-align: center;
                        padding: 1rem;
                    }}
                </style>
                <div class="reasoning-steps-container">
                    {reasoning_html}
                </div>
                """
                
                # Return all outputs as a list in the correct order
                return [
                    decision_html,  # decision_output
                    format_metrics_html('Confidence', confidence_display),  # confidence_score
                    reasoning or "No reasoning available",  # reasoning_output
                    accept_rate_html,  # accept_rate
                    avg_conf_html,  # avg_confidence
                    consistency_html,  # consistency
                    confidence_plot_fig,  # confidence_plot
                    decision_plot_fig,  # decision_plot
                    lattice_plot_fig,  # lattice_plot
                    reasoning_table_html,  # reasoning_table
                    raw_output  # raw_output
                ]
                
            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                logger.error(error_msg, exc_info=True)
                error_html = f"""
                <div class='decision-output decision-error'>
                    <div class='error-message'>{error_msg}</div>
                </div>
                """
                empty_metric = format_metrics_html('', '0%')
                return [
                    error_html,  # decision_output
                    format_metrics_html('Confidence', '0%'),  # confidence_score
                    "An error occurred while processing your request. Please try again.",  # reasoning_output
                    empty_metric,  # accept_rate
                    empty_metric,  # avg_confidence
                    empty_metric,  # consistency
                    None,  # confidence_plot
                    None,  # decision_plot
                    "<div class='error-message'>No data available due to processing error.</div>",  # reasoning_table
                    {"error": str(e), "status": "error"}  # raw_output
                ]
        
        def clear_inputs():
            """Reset all inputs and outputs to their initial state."""
            empty_decision = "<div class='decision-output decision-pending'>No decision yet</div>"
            empty_metric = format_metrics_html('', '0%')
            
            # Create empty figure for plots
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[dict(
                    text="No data to display",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            
            # Create empty DataFrame for reasoning table
            empty_df = pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
            
            return [
                "",  # prompt
                "",  # context
                empty_decision,  # decision_output
                empty_metric,  # confidence_score
                "*Processing will begin when you submit your prompt...*",  # reasoning_output
                empty_metric,  # accept_rate
                empty_metric,  # avg_confidence
                empty_metric,  # consistency
                empty_fig,  # confidence_plot
                empty_fig,  # decision_plot
                None,  # lattice_plot (not currently generated)
                empty_df,  # reasoning_table
                {"status": "No data available. Process a prompt to see results."}  # raw_output
            ]
            
        # Connect the submit button
        def process_and_display(prompt, context, alpha, beta, tau, temp, max_tokens_slider, top_p):
            """Process the prompt and update all UI elements."""
            try:
                # Process the prompt - this returns a list: [final_answer, confidence_plot, decision_plot, reasoning_table]
                result = process_prompt(prompt, context, alpha, beta, tau, temp, max_tokens_slider, top_p)
                
                # Extract results from the list
                final_answer = result[0] if len(result) > 0 else 'No answer generated'
                confidence_plot = result[1] if len(result) > 1 else None
                decision_plot = result[2] if len(result) > 2 else None
                reasoning_table = result[3] if len(result) > 3 else pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
                
                # Set default values for metrics based on the plots if available
                confidence = 0.8  # Default confidence value
                avg_conf = 0.8    # Default average confidence
                accept_rate = 0.9 # Default accept rate
                consistency = 0.9 # Default consistency
                
                # Try to extract metrics from the visualization data if available
                try:
                    if confidence_plot and hasattr(confidence_plot, 'data'):
                        # Extract confidence from plot data if available
                        for trace in confidence_plot.data:
                            if hasattr(trace, 'y') and len(trace.y) > 0:
                                confidence = float(trace.y[0])
                                break
                    
                    if decision_plot and hasattr(decision_plot, 'data'):
                        # Extract decision metrics if available
                        for trace in decision_plot.data:
                            if hasattr(trace, 'name') and 'Accept' in str(trace.name):
                                accept_rate = float(trace.y[0]) if hasattr(trace, 'y') and len(trace.y) > 0 else accept_rate
                            if hasattr(trace, 'name') and 'Confidence' in str(trace.name):
                                avg_conf = float(trace.y[0]) if hasattr(trace, 'y') and len(trace.y) > 0 else avg_conf
                except Exception as e:
                    logger.warning(f"Could not extract metrics from plots: {str(e)}")
                
                # Create decision HTML with appropriate styling
                decision_class = 'accept' if confidence >= 0.7 else 'reject' if confidence <= 0.4 else 'abstain'
                decision_html = f"<div class='decision-output decision-{decision_class}'>{final_answer}</div>"
                
                # Create a clean final answer for the reasoning output
                clean_answer = final_answer
                if isinstance(clean_answer, dict):
                    clean_answer = clean_answer.get('text', str(clean_answer))
                clean_answer = str(clean_answer).strip()
                
                # Prepare the return values in the correct order for the Gradio interface
                # Extract reasoning steps from the result if available
                reasoning_steps = []
                if len(result) > 3 and hasattr(result[3], 'to_dict'):
                    try:
                        reasoning_steps = result[3].to_dict('records')
                    except Exception as e:
                        logger.warning(f"Could not convert reasoning table to dict: {str(e)}")
                        reasoning_steps = []
                
                return [
                    decision_html,  # decision_output
                    format_metrics_html('Confidence', f"{confidence*100:.1f}%"),  # confidence_score
                    clean_answer,  # reasoning_output
                    format_metrics_html('Accept Rate', f"{accept_rate*100:.1f}%"),  # accept_rate
                    format_metrics_html('Avg. Confidence', f"{avg_conf*100:.1f}%"),  # avg_confidence
                    format_metrics_html('Consistency', f"{consistency*100:.1f}%"),  # consistency
                    confidence_plot,  # confidence_plot
                    decision_plot,  # decision_plot
                    None,  # lattice_plot (not currently generated)
                    reasoning_table,  # reasoning_table
                    {
                        "status": "success", 
                        "answer": clean_answer,
                        "confidence": confidence,
                        "accept_rate": accept_rate,
                        "avg_confidence": avg_conf,
                        "consistency": consistency,
                        "reasoning_steps": reasoning_steps
                    }  # raw_output
                ]
                
            except Exception as e:
                error_msg = f"Error in process_and_display: {str(e)}"
                logger.error(error_msg, exc_info=True)
                error_html = f"<div class='decision-output decision-error'>{error_msg}</div>"
                empty_metric = format_metrics_html('Error', 'N/A')
                
                # Create an empty figure for the plots
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error occurred",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    annotations=[dict(
                        text="Error occurred while generating visualization",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )]
                )
                
                # Create an empty DataFrame for the reasoning steps
                empty_reasoning_table = pd.DataFrame(columns=["Step", "Content", "Confidence", "Decision"])
                
                # Prepare the raw output with all required fields
                raw_output = {
                    "error": str(e),
                    "status": "error",
                    "confidence": 0.0,
                    "accept_rate": 0.0,
                    "avg_confidence": 0.0,
                    "consistency": 0.0,
                    "reasoning_steps": [],
                    "final_answer": error_msg,
                    "decisions": {
                        "decisions": [],
                        "summary": {},
                        "metadata": {}
                    }
                }
                
                return [
                    error_html,  # decision_output
                    empty_metric,  # confidence_score
                    error_msg,  # reasoning_output
                    empty_metric,  # accept_rate
                    empty_metric,  # avg_confidence
                    empty_metric,  # consistency
                    empty_fig,  # confidence_plot
                    empty_fig,  # decision_plot
                    None,  # lattice_plot
                    empty_reasoning_table,  # reasoning_table
                    raw_output  # raw_output
                ]
        
        # Create empty figure for placeholders
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text="No data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
        
        # Get default values from config or use sensible defaults
        default_params = getattr(app, 'config', {}).get('default_parameters', {
            'alpha': 0.7,
            'beta': 0.6,
            'tau': 0.4,
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 1.0
        })
        
        # Connect the submit button
        submit_btn.click(
            fn=process_and_display,
            inputs=[
                prompt_input,
                context_input,
                alpha_slider,
                beta_slider,
                tau_slider,
                temp_slider,
                max_tokens_slider,
                top_p_slider
            ],
            outputs=[
                decision_output,
                confidence_score,
                reasoning_output,
                accept_rate,
                avg_confidence,
                consistency,
                confidence_plot,
                decision_plot,
                lattice_plot,
                reasoning_table,
                raw_output
            ]
        )
        
        # Connect the clear button
        def clear_inputs():
            """Clear all input fields."""
            try:
                # Clear all input fields
                return [
                    "",  # prompt_input
                    "",  # context_input
                    "",  # decision_output
                    format_metrics_html('Confidence', '0%'),  # confidence_score
                    "",  # reasoning_output
                    format_metrics_html('Accept Rate', '0%'),  # accept_rate
                    format_metrics_html('Avg. Confidence', '0%'),  # avg_confidence
                    format_metrics_html('Consistency', '0%'),  # consistency
                    empty_fig,  # confidence_plot
                    empty_fig,  # decision_plot
                    empty_fig,  # lattice_plot
                    "<div class='empty-state'>No reasoning steps available.</div>",  # reasoning_table
                    {"status": "No data available. Process a prompt to see results."}  # raw_output
                ]
            except Exception as e:
                logger.error(f"Error in clear_inputs: {str(e)}", exc_info=True)
                # Return a minimal set of defaults if something goes wrong
                return [""] * 11  # Adjust the number based on the number of outputs
        
        # Connect the clear button
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                prompt_input,
                context_input,
                decision_output,
                confidence_score,
                reasoning_output,
                accept_rate,
                avg_confidence,
                consistency,
                confidence_plot,
                decision_plot,
                lattice_plot,
                reasoning_table,
                raw_output
            ]
        )
        
        # View history button
        view_history_btn.click(
            fn=load_history,
            inputs=[],
            outputs=history_table
        )
        
        # Add some examples
        gr.Examples(
            examples=[
                
                ["A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?", ""],
                ["What are the ethical implications of artificial intelligence?", ""],
                ["How does the three-way decision framework improve upon traditional binary classification?", ""]
            ],
            inputs=[prompt, context],
            label="Example Prompts"
        )
    
    return demo

def main():
    """Main function to run the Gradio interface."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('3waycot_ui.log')
        ]
    )
    logger = logging.getLogger("3WayCoT.UI")
    
    try:
        # Initialize the application
        logger.info("Starting 3WayCoT UI...")
        demo = create_interface()
        
        # Launch the interface
        logger.info("Launching Gradio interface...")
        
        # Try multiple ports if the default one is in use
        base_port = 7860
        max_attempts = 10
        
        for attempt in range(max_attempts):
            current_port = base_port + attempt
            try:
                logger.info(f"Attempting to launch on port {current_port}...")
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=current_port,
                    share=True,
                    debug=True,
                   
                )
                break  # If successful, exit the loop
            except OSError as e:
                if "address already in use" in str(e).lower() or "cannot find empty port" in str(e).lower():
                    if attempt == max_attempts - 1:  # Last attempt
                        logger.error(f"Failed to find an available port after {max_attempts} attempts")
                        raise
                    logger.warning(f"Port {current_port} is in use, trying next port...")
                    continue
                raise  # Re-raise other OSError exceptions
    except Exception as e:
        logger.error(f"Failed to launch the application: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('3waycot_ui.log')
        ]
    )
    
    # Get logger for main
    logger = logging.getLogger("3WayCoT.Main")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
