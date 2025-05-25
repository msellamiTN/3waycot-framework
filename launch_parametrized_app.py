#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher script for the Parametrized 3WayCoT Framework

This script initializes and launches the parametrized Gradio interface.
"""

import gradio as gr
import logging
from parametrized_gradio_app import ParametrizedGradio3WayCoT
from parametrized_gradio_app import create_confidence_plot, create_decision_plot, create_lattice_visualization
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parametrized_gradio.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('3WayCoT.Launcher')

def create_interface():
    """Create and return the Gradio interface with parameter controls and enhanced visualizations."""
    # Initialize the Gradio app
    app = ParametrizedGradio3WayCoT()
    
    # Get use case names
    use_case_names = list(app.use_cases.keys())
    
    # Create the interface
    with gr.Blocks(title="Parametrized 3WayCoT Framework", theme=gr.themes.Soft()) as interface:
        # Add header
        gr.Markdown("""
        # Parametrized 3WayCoT Framework
        
        Analyze prompts using Three-Way Decisions with customizable parameters.
        """)
        
        # Main layout
        with gr.Row():
            # Left column: Input and Parameters
            with gr.Column(scale=1):
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
                
                # Parameter controls
                with gr.Group():
                    gr.Markdown("### Parameters")
                    
                    # Use case selection
                    with gr.Row():
                        use_case_dropdown = gr.Dropdown(
                            choices=use_case_names,
                            value="default" if "default" in use_case_names else use_case_names[0],
                            label="Use Case"
                        )
                        apply_use_case_btn = gr.Button("Apply Use Case")
                    
                    use_case_description = gr.Markdown("")
                    
                    # Parameter tabs
                    with gr.Tabs():
                        with gr.TabItem("Decision Parameters"):
                            alpha_slider = gr.Slider(
                                minimum=0.5, maximum=0.9, step=0.05, 
                                value=app.config.get('framework.alpha', 0.7),
                                label="Alpha Threshold",
                                info="Higher values lead to more conservative decisions"
                            )
                            beta_slider = gr.Slider(
                                minimum=0.3, maximum=0.7, step=0.05, 
                                value=app.config.get('framework.beta', 0.4),
                                label="Beta Threshold",
                                info="Lower values increase rejection rate"
                            )
                            tau_slider = gr.Slider(
                                minimum=0.3, maximum=0.8, step=0.05, 
                                value=app.config.get('framework.tau', 0.6),
                                label="Tau Threshold",
                                info="Confidence value threshold"
                            )
                            
                            # Confidence weight slider (specifically addressing user's needs)
                            confidence_weight_slider = gr.Slider(
                                minimum=0.3, maximum=1.0, step=0.05, 
                                value=app.config.get('framework.confidence_weight', 0.7),
                                label="Confidence Weight",
                                info="Weight given to confidence in the decision-making process"
                            )
                        
                        with gr.TabItem("Model Parameters"):
                            model_dropdown = gr.Dropdown(
                                choices=app.all_models,
                                value=app.llm_config.get('default_model', app.all_models[0]),
                                label="Model",
                                info="Select the language model to use"
                            )
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
                
                # Control buttons
                with gr.Row():
                    process_btn = gr.Button("Process", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            # Right column: Results and Visualizations
            with gr.Column(scale=2):
                # Main output
                with gr.Group():
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column(scale=1):
                            decision_output = gr.Markdown("", label="Decision")
                        with gr.Column(scale=1):
                            confidence_score = gr.Markdown("", label="Confidence")
                    
                    reasoning_output = gr.Markdown("", label="Reasoning")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            accept_rate = gr.Markdown("", label="Accept Rate")
                        with gr.Column(scale=1):
                            avg_confidence = gr.Markdown("", label="Average Confidence")
                        with gr.Column(scale=1):
                            consistency = gr.Markdown("", label="Decision Consistency")
                
                # Visualizations
                with gr.Tabs(selected=0):
                    with gr.TabItem("Confidence Analysis"):
                        confidence_plot = gr.Plot(label="Confidence Metrics")
                    
                    with gr.TabItem("Decision Analysis"):
                        decision_plot = gr.Plot(label="Decision Metrics")
                    
                    with gr.TabItem("Triadic Lattice"):
                        lattice_plot = gr.Plot(label="Concept Lattice")
                    
                    with gr.TabItem("Raw Output"):
                        raw_output = gr.JSON(label="Raw Results")
        
        # Define event handlers
        def update_use_case_desc(use_case):
            """Update the use case description."""
            if use_case in app.use_cases:
                uc = app.use_cases[use_case]
                return f"**{use_case.title()}**: {uc.get('description', '')}\\n\\nAlpha: {uc.get('alpha', 0.7)}, Beta: {uc.get('beta', 0.4)}, Tau: {uc.get('tau', 0.6)}"
            return ""
        
        def apply_use_case(use_case_name):
            """Apply a use case configuration."""
            if use_case_name in app.use_cases:
                uc = app.use_cases[use_case_name]
                return [
                    gr.Slider.update(value=uc.get('alpha', 0.7)),
                    gr.Slider.update(value=uc.get('beta', 0.4)),
                    gr.Slider.update(value=uc.get('tau', 0.6)),
                    gr.Slider.update(value=uc.get('confidence_weight', 0.7)),
                    gr.Slider.update(value=uc.get('temperature', 0.7)),
                    update_use_case_desc(use_case_name)
                ]
            return [gr.Slider.update(), gr.Slider.update(), gr.Slider.update(), 
                   gr.Slider.update(), gr.Slider.update(), ""]
                   
        def process_prompt(prompt, context, alpha, beta, tau, confidence_weight, temp, max_tokens, top_p, model):
            """Process the prompt with the given parameters."""
            if not prompt or prompt.strip() == "":
                gr.Warning("Please enter a prompt first")
                return {
                    'error': "No prompt provided",
                    'status': 'error'
                }
            
            # Prepare parameters
            parameters = {
                'alpha': alpha,
                'beta': beta,
                'tau': tau,
                'confidence_weight': confidence_weight,
                'temperature': temp,
                'max_tokens': int(max_tokens),
                'top_p': top_p,
                'model': model
            }
            
            # Process the prompt
            result = app.process_prompt(prompt, context, **parameters)
            return result
            
        def format_decision_html(accept_ratio):
            """Format the decision with appropriate styling."""
            if accept_ratio >= 0.7:
                color = "green"
                decision = "ACCEPT"
            elif accept_ratio <= 0.3:
                color = "red"
                decision = "REJECT"
            else:
                color = "orange"
                decision = "ABSTAIN"
            
            return f"<div style='text-align: center; color: {color}; font-size: 28px; font-weight: bold;'>{decision}</div>"
        
        def format_metrics_html(label, value):
            """Format a metric value with its label."""
            return f"<div style='text-align: center;'><span style='font-size: 16px;'>{label}</span><br><span style='font-size: 24px; font-weight: bold;'>{value:.2f}</span></div>"
        
        def process_and_display(prompt, context, alpha, beta, tau, confidence_weight, temp, max_tokens, top_p, model):
            """Process the prompt and update all UI elements."""
            # Check if prompt is empty
            if not prompt or prompt.strip() == "":
                gr.Warning("Please enter a prompt first")
                return (
                    "No prompt provided", 
                    "N/A", 
                    "Please enter a prompt to analyze", 
                    "N/A", "N/A", "N/A",
                    None, None, None, None
                )
            
            try:
                # Process the prompt
                result = process_prompt(prompt, context, alpha, beta, tau, confidence_weight, temp, max_tokens, top_p, model)
                
                # Get visualization data
                vis_data = app.get_visualization_data(result)
                
                # Extract key components
                confidence_metrics = vis_data.get('confidence_metrics', {})
                decision_metrics = vis_data.get('decision_metrics', {})
                reasoning_steps = vis_data.get('reasoning_steps', [])
                
                # Format outputs for display
                decision_html = format_decision_html(decision_metrics.get('accept_ratio', 0.5))
                confidence_html = f"<span style='font-size: 24px;'>{vis_data.get('confidence', 0.8):.2f}</span>"
                
                # Get the reasoning text
                reasoning_text = "No reasoning steps available"
                if reasoning_steps and len(reasoning_steps) > 0:
                    final_step = reasoning_steps[-1]
                    if isinstance(final_step, dict) and 'reasoning' in final_step:
                        reasoning_text = final_step['reasoning']
                
                # Format metrics
                accept_rate_html = format_metrics_html("Accept Rate", decision_metrics.get('accept_ratio', 0.0))
                avg_confidence_html = format_metrics_html("Avg. Confidence", confidence_metrics.get('average_confidence', 0.0))
                consistency_html = format_metrics_html("Consistency", confidence_metrics.get('decision_consistency', 0.0))
                
                # Create visualizations
                confidence_fig = create_confidence_plot(confidence_metrics)
                decision_fig = create_decision_plot(decision_metrics)
                lattice_fig = create_lattice_visualization(decision_metrics)
                
                return (
                    decision_html,
                    confidence_html,
                    reasoning_text,
                    accept_rate_html,
                    avg_confidence_html,
                    consistency_html,
                    confidence_fig,
                    decision_fig,
                    lattice_fig,
                    result
                )
            except Exception as e:
                logger.error(f"Error in process_and_display: {e}", exc_info=True)
                return (
                    "Error processing prompt", 
                    "N/A", 
                    f"An error occurred: {str(e)}", 
                    "N/A", "N/A", "N/A",
                    None, None, None, None
                )
        
        def clear_inputs():
            """Reset all inputs to their default values."""
            return (
                "", "",  # prompt, context
                "", "", "", "", "", "",  # outputs
                None, None, None, None  # visualizations
            )
        
        # Connect event handlers
        use_case_dropdown.change(
            fn=update_use_case_desc,
            inputs=[use_case_dropdown],
            outputs=[use_case_description]
        )
        
        apply_use_case_btn.click(
            fn=apply_use_case,
            inputs=[use_case_dropdown],
            outputs=[
                alpha_slider,
                beta_slider,
                tau_slider,
                confidence_weight_slider,
                temp_slider,
                use_case_description
            ]
        )
        
        process_btn.click(
            fn=process_and_display,
            inputs=[
                prompt_input,
                context_input,
                alpha_slider,
                beta_slider,
                tau_slider,
                confidence_weight_slider,
                temp_slider,
                max_tokens_slider,
                top_p_slider,
                model_dropdown
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
                raw_output
            ]
        )
        
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
                raw_output
            ]
        )
        
        # Add examples focused on triadic decision making and confidence metrics
        gr.Examples([
            ["Explain the main advantages of Triadic Fuzzy Concept Analysis over traditional Fuzzy Concept Analysis.", ""],
            ["How does the confidence weighting in Three-Way Decision making affect boundary region decisions?", ""],
            ["What are the key factors that determine high confidence vs. low confidence in reasoning steps?", ""],
            ["How does varying alpha and beta thresholds affect decision consistency in a Three-Way Decision framework?", ""]
        ], inputs=[prompt_input, context_input])
    
    return interface

def main():
    """Main function to run the Parametrized 3WayCoT application."""
    # Create the interface
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
