# app.py
import gradio as gr
import json
import yaml
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import visualization modules
from src.visualization import (
    LatticeVisualizer,
    MetricsDashboard,
    ConfidenceAnalyzer,
    ParameterAnalyzer
)
from main import ThreeWayCoTApp

# Load configuration
CONFIG_PATH = "src/config/config.yml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize the 3WayCoT app with config
app = ThreeWayCoTApp(config_path=CONFIG_PATH)

# Create output directories
os.makedirs("results/visualizations", exist_ok=True)

def run_3waycot(prompt: str, use_case: str, visualization_type: str = "decision_distribution") -> Dict[str, Any]:
    """Run the 3WayCoT pipeline and generate visualizations.
    
    Args:
        prompt: Input prompt for the 3WayCoT framework
        use_case: Which use case configuration to use
        visualization_type: Type of visualization to generate
        
    Returns:
        Dictionary containing visualization results and metadata
    """
    try:
        # Apply use case configuration if specified
        if use_case and use_case in config.get('framework', {}).get('use_cases', {}):
            use_case_config = config['framework']['use_cases'][use_case]
            app.update_parameters(**use_case_config)
            
        # Run the 3WayCoT pipeline
        results = app.process_prompt(
            prompt=prompt,
            output_file="results/latest_results.json"
        )
        
        # Load the results
        with open("results/latest_results.json", "r") as f:
            results_data = json.load(f)
        
        # Get visualization config
        vis_config = config.get('visualization', {})
        
        # Generate visualizations based on type
        if visualization_type == "decision_distribution":
            return create_decision_distribution(results_data, vis_config)
        elif visualization_type == "confidence_analysis":
            return create_confidence_analysis(results_data, vis_config)
        elif visualization_type == "parameter_impact":
            return create_parameter_impact_analysis(results_data, vis_config)
        elif visualization_type == "lattice":
            return create_lattice_visualization(results_data, vis_config)
        else:
            return {"error": f"Visualization type '{visualization_type}' not supported"}
            
    except Exception as e:
        return {"error": f"Error generating visualization: {str(e)}"}

def create_decision_distribution(results_data: Dict[str, Any], vis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a visualization of decision distribution.
    
    Args:
        results_data: Results from 3WayCoT processing
        vis_config: Visualization configuration
        
    Returns:
        Dictionary containing visualization and metadata
    """
    try:
        # Extract decision counts
        decisions = results_data.get("decisions", [])
        if not decisions:
            return {"error": "No decision data available"}
        
        # Count decision types
        decision_counts = {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0}
        for decision in decisions:
            if isinstance(decision, dict):
                decision_type = decision.get("decision", "").upper()
                if decision_type in decision_counts:
                    decision_counts[decision_type] += 1
        
        # Get configuration
        colors = vis_config.get('colors', {}).get('decision_distribution', 
                 ['#2ecc71', '#e74c3c', '#f39c12'])  # Green, Red, Orange
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(decision_counts.keys()),
            values=list(decision_counts.values()),
            marker_colors=colors,
            textinfo='label+percent',
            hoverinfo='label+value+percent',
            hole=0.3
        )])
        
        fig.update_layout(
            title={
                'text': "Decision Distribution",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Save the figure
        output_path = "results/visualizations/decision_distribution.html"
        fig.write_html(output_path)
        
        return {
            "visualization": fig.to_dict(),
            "type": "plotly",
            "output_path": output_path,
            "stats": {
                "total_decisions": sum(decision_counts.values()),
                "accept_rate": decision_counts["ACCEPT"] / max(1, sum(decision_counts.values())),
                "reject_rate": decision_counts["REJECT"] / max(1, sum(decision_counts.values())),
                "abstain_rate": decision_counts["ABSTAIN"] / max(1, sum(decision_counts.values()))
            }
        }
        
    except Exception as e:
        return {"error": f"Error creating decision distribution: {str(e)}"}

def create_confidence_analysis(results_data: Dict[str, Any], vis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create confidence analysis visualizations.
    
    Args:
        results_data: Results from 3WayCoT processing
        vis_config: Visualization configuration
        
    Returns:
        Dictionary containing visualizations and metadata
    """
    try:
        # Initialize ConfidenceAnalyzer
        decisions = results_data.get("decisions", [])
        if not decisions:
            return {"error": "No decision data available for confidence analysis"}
            
        # Prepare data for confidence analysis
        confidence_data = []
        for decision in decisions:
            if isinstance(decision, dict):
                confidence_data.append({
                    "decision": decision.get("decision", "UNKNOWN").upper(),
                    "confidence": float(decision.get("confidence", 0.0)),
                    "step": len(confidence_data) + 1
                })
        
        if not confidence_data:
            return {"error": "No confidence data available for analysis"}
            
        # Create confidence analyzer
        analyzer = ConfidenceAnalyzer(confidence_data)
        
        # Generate visualizations
        output = {
            "visualizations": {},
            "stats": {}
        }
        
        # 1. Confidence distribution
        dist_fig = analyzer.confidence_distribution(show=False)
        dist_path = "results/visualizations/confidence_distribution.html"
        dist_fig.write_html(dist_path)
        output["visualizations"]["distribution"] = {
            "figure": dist_fig.to_dict(),
            "path": dist_path,
            "type": "plotly"
        }
        
        # 2. Confidence by decision
        rel_fig = analyzer.confidence_decision_relationship(show=False)
        rel_path = "results/visualizations/confidence_by_decision.html"
        rel_fig.write_html(rel_path)
        output["visualizations"]["by_decision"] = {
            "figure": rel_fig.to_dict(),
            "path": rel_path,
            "type": "plotly"
        }
        
        # 3. Confidence flow over steps
        df = pd.DataFrame(confidence_data)
        flow_fig = px.line(
            df, 
            x="step", 
            y="confidence",
            color="decision",
            title="Confidence Flow by Decision Type",
            markers=True,
            color_discrete_map=vis_config.get('colors', {}).get('decision_types', {
                'ACCEPT': '#2ecc71',
                'REJECT': '#e74c3c',
                'ABSTAIN': '#f39c12'
            })
        )
        
        flow_fig.update_layout(
            xaxis_title="Decision Step",
            yaxis_title="Confidence Score",
            yaxis_range=[0, 1.1],
            legend_title="Decision Type"
        )
        
        flow_path = "results/visualizations/confidence_flow.html"
        flow_fig.write_html(flow_path)
        output["visualizations"]["flow"] = {
            "figure": flow_fig.to_dict(),
            "path": flow_path,
            "type": "plotly"
        }
        
        # Add statistics
        stats = df.groupby("decision")["confidence"].agg(['mean', 'std', 'count']).to_dict('index')
        output["stats"] = {
            "overall_mean": df["confidence"].mean(),
            "overall_std": df["confidence"].std(),
            "by_decision": stats
        }
        
        return output
        
    except Exception as e:
        return {"error": f"Error in confidence analysis: {str(e)}"}

def create_parameter_impact_analysis(results_data: Dict[str, Any], vis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create parameter impact analysis visualizations.
    
    Args:
        results_data: Results from 3WayCoT processing
        vis_config: Visualization configuration
        
    Returns:
        Dictionary containing visualizations and metadata
    """
    try:
        # Check if we have parameter variation data
        if "parameter_variations" not in results_data or not results_data["parameter_variations"]:
            return {"error": "No parameter variation data available for analysis"}
            
        # Prepare data for parameter analysis
        param_data = results_data["parameter_variations"]
        
        # Initialize ParameterAnalyzer
        analyzer = ParameterAnalyzer(param_data)
        
        # Get metrics to analyze
        metrics = vis_config.get('parameter_analysis', {}).get('metrics', 
                  ['accuracy', 'precision', 'recall', 'f1_score'])
        
        # Get parameters to analyze
        parameters = vis_config.get('parameter_analysis', {}).get('parameters')
        
        # Generate visualizations
        output = {
            "visualizations": {},
            "stats": {}
        }
        
        # 1. Parameter sensitivity analysis
        for metric in metrics:
            try:
                sens_fig = analyzer.parameter_sensitivity_analysis(
                    target_metric=metric,
                    parameters=parameters,
                    show=False
                )
                
                sens_path = f"results/visualizations/parameter_sensitivity_{metric}.html"
                sens_fig.write_html(sens_path)
                
                output["visualizations"][f"sensitivity_{metric}"] = {
                    "figure": sens_fig.to_dict(),
                    "path": sens_path,
                    "type": "plotly",
                    "metric": metric
                }
            except Exception as e:
                print(f"Could not generate sensitivity analysis for {metric}: {str(e)}")
        
        # 2. Parameter impact heatmap
        try:
            heatmap_fig = analyzer.parameter_impact_heatmap(
                metrics=metrics[:5],  # Limit to 5 metrics for readability
                parameters=parameters,
                show=False
            )
            
            heatmap_path = "results/visualizations/parameter_impact_heatmap.html"
            heatmap_fig.write_html(heatmap_path)
            
            output["visualizations"]["impact_heatmap"] = {
                "figure": heatmap_fig.to_dict(),
                "path": heatmap_path,
                "type": "plotly"
            }
        except Exception as e:
            print(f"Could not generate parameter impact heatmap: {str(e)}")
        
        # 3. Parameter interaction analysis (if we have at least 2 parameters)
        if parameters and len(parameters) >= 2:
            try:
                # Analyze interaction between first two parameters
                inter_fig = analyzer.parameter_interaction_analysis(
                    param1=parameters[0],
                    param2=parameters[1],
                    target_metric=metrics[0],
                    show=False
                )
                
                inter_path = f"results/visualizations/parameter_interaction_{parameters[0]}_vs_{parameters[1]}.html"
                inter_fig.write_html(inter_path)
                
                output["visualizations"]["parameter_interaction"] = {
                    "figure": inter_fig.to_dict(),
                    "path": inter_path,
                    "type": "plotly",
                    "parameters": [parameters[0], parameters[1]],
                    "metric": metrics[0]
                }
            except Exception as e:
                print(f"Could not generate parameter interaction analysis: {str(e)}")
        
        return output
        
    except Exception as e:
        return {"error": f"Error in parameter impact analysis: {str(e)}"}

def update_visualization_type(use_case: str) -> Dict:
    """Update available visualization options based on use case."""
    base_visualizations = ["decision_distribution", "confidence_analysis"]
    
    if use_case == "parameter_exploration":
        return {"choices": base_visualizations + ["parameter_impact", "lattice"]}
    return {"choices": base_visualizations}

def format_visualization_output(result: Dict) -> go.Figure:
    """Format the visualization output for Gradio."""
    if "error" in result:
        return f"Error: {result['error']}"
    
    # If we have a single visualization
    if "visualization" in result:
        return go.Figure(result["visualization"])
    
    # If we have multiple visualizations, show the first one
    if "visualizations" in result and result["visualizations"]:
        first_key = next(iter(result["visualizations"]))
        return go.Figure(result["visualizations"][first_key]["figure"])
    
    return "No visualization data available"

# Create the Gradio interface
with gr.Blocks(title="3WayCoT Interactive Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 3WayCoT Framework Interactive Visualizer
    Explore and analyze the decision-making process with interactive visualizations.
    """)
    
    with gr.Row():
        # Left panel - Inputs
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                gr.Markdown("### Input Parameters")
                prompt = gr.Textbox(
                    label="Enter your prompt", 
                    lines=3, 
                    placeholder="Type your reasoning task here...",
                    value="Analyze the impact of climate change on urban areas."
                )
                
                use_case = gr.Dropdown(
                    label="Select use case",
                    choices=["default", "conservative", "exploratory", "parameter_exploration"],
                    value="default",
                    interactive=True
                )
                
                visualization_type = gr.Dropdown(
                    label="Visualization type",
                    choices=["decision_distribution", "confidence_analysis"],
                    value="decision_distribution",
                    interactive=True
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    max_steps = gr.Slider(
                        minimum=1, 
                        maximum=10, 
                        value=5,
                        step=1,
                        label="Max Reasoning Steps"
                    )
                    confidence_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Confidence Weight"
                    )
            
            submit_btn = gr.Button("Run Analysis", variant="primary")
            
            # Stats panel
            with gr.Group(visible=False) as stats_panel:
                gr.Markdown("### Analysis Statistics")
                stats_output = gr.JSON(label="Statistics")
        
        # Right panel - Visualizations
        with gr.Column(scale=2):
            gr.Markdown("### Visualization")
            visualization_output = gr.Plot(
                label="Analysis Results",
                show_label=False,
                container=False
            )
            
            # Tabs for multiple visualizations
            with gr.Tabs() as visualization_tabs:
                with gr.Tab("Primary"):
                    primary_viz = gr.Plot(label="Primary Visualization")
                with gr.Tab("Secondary"):
                    secondary_viz = gr.Plot(label="Secondary Visualization")
                with gr.Tab("Statistics"):
                    stats_display = gr.JSON(label="Detailed Statistics")
    
    # Update visualization options based on use case
    def update_vis_dropdown(use_case_name):
        vis_types = ["decision_distribution", "confidence_analysis"]
        if use_case_name == "parameter_exploration":
            vis_types.append("parameter_impact")
        return gr.Dropdown.update(choices=vis_types)
    
    use_case.change(
        fn=update_vis_dropdown,
        inputs=use_case,
        outputs=visualization_type
    )
    
    # Handle form submission
    def process_analysis(prompt, use_case, visualization_type, max_steps, confidence_weight):
        # Update config based on UI inputs
        config['framework']['max_steps'] = max_steps
        config['framework']['confidence_weight'] = confidence_weight
        
        # Save updated config
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f)
        
        # Run analysis
        result = run_3waycot(prompt, use_case, visualization_type)
        
        # Format output
        stats = {}
        visualizations = {}
        
        if "stats" in result:
            stats = result["stats"]
        
        if "visualizations" in result:
            visualizations = result["visualizations"]
        elif "visualization" in result:
            visualizations = {"primary": {"figure": result["visualization"]}}
        
        # Prepare visualization figures
        viz_figures = []
        for viz in visualizations.values():
            if isinstance(viz, dict) and "figure" in viz:
                viz_figures.append(go.Figure(viz["figure"]))
        
        # Ensure we have at least one figure
        if not viz_figures:
            return [
                gr.update(visible=True),  # stats_panel
                {},                       # stats_output
                None,                     # visualization_output
                None,                     # primary_viz
                None,                     # secondary_viz
                {}                        # stats_display
            ]
        
        # Return updates for all components
        return [
            gr.update(visible=bool(stats)),  # stats_panel
            stats,                          # stats_output
            viz_figures[0],                # visualization_output
            viz_figures[0] if len(viz_figures) > 0 else None,  # primary_viz
            viz_figures[1] if len(viz_figures) > 1 else None,  # secondary_viz
            stats                          # stats_display
        ]
    
    submit_btn.click(
        fn=process_analysis,
        inputs=[prompt, use_case, visualization_type, max_steps, confidence_weight],
        outputs=[
            stats_panel,
            stats_output,
            visualization_output,
            primary_viz,
            secondary_viz,
            stats_display
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )