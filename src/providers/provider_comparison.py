#!/usr/bin/env python
"""
Provider Comparison Example for 3WayCoT Framework.

This example demonstrates how to use multiple LLM providers and compare their results.
"""

import os
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Fix import paths
import fix_imports

# Import from the framework
from src.core.threeway_cot import ThreeWayCoT
from src.utils.visualization import Visualizer
from src.models.inverted_tfca import InvertedTriadicFuzzyAnalysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("3WayCoT.Example")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Provider Comparison for 3WayCoT Framework")
    parser.add_argument(
        "--providers", 
        nargs="+", 
        default=["gemini-pro", "gemini-1.5-pro"], 
        help="LLM providers to compare (default: gemini-pro and gemini-1.5-pro)"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is the diagnosis for a patient with fever, cough, and shortness of breath?",
        help="Query to process with 3WayCoT"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--temperature1", 
        type=float, 
        default=0.3,
        help="Temperature for the first Gemini model"
    )
    parser.add_argument(
        "--temperature2", 
        type=float, 
        default=0.7,
        help="Temperature for the second Gemini model"
    )
    parser.add_argument(
        "--max-assumptions",
        type=int,
        default=None,
        help="Maximum number of assumptions/conditions to include in the context (default: no limit)"
    )
    parser.add_argument(
        "--inverted-tfca",
        action="store_true",
        help="Use inverted TFCA for analysis (default: False)"
    )
    return parser.parse_args()

def run_with_provider(provider, query, temperature=0.3, max_assumptions=None, inverted_tfca=False):
    """
    Run the 3WayCoT framework with a specific provider and configuration.
    
    Args:
        provider: The LLM provider to use (e.g., 'gemini-pro')
        query: The input query to process
        temperature: Temperature parameter for generation (default: 0.3)
        max_assumptions: Maximum number of assumptions/conditions to include (default: None)
        inverted_tfca: Whether to use inverted TFCA for analysis (default: False)
        
    Returns:
        Dictionary containing the processing results
    """
    logger.info(f"Running 3WayCoT with provider: {provider} (temperature: {temperature}, max_assumptions: {max_assumptions}, inverted_tfca: {inverted_tfca})")
    
    # Determine provider and model from the provider string
    if '-' in provider:
        provider_parts = provider.split('-')
        llm_provider = provider_parts[0]  # e.g., 'gemini'
        llm_model = '-'.join(provider_parts[1:])  # e.g., 'pro' or '1.5-pro'
    else:
        llm_provider = provider
        llm_model = None
    
    # Initialize the model with the specified provider and model
    model = ThreeWayCoT(
        llm_provider=llm_provider,
        llm_model=llm_model,
        alpha=0.7,  # Acceptance threshold
        beta=0.6,   # Rejection threshold
        tau=0.5,    # Similarity threshold
        theta_abs=0.3,  # Abstention threshold
        max_assumptions=max_assumptions    
    )
    
    # Set temperature in the cot_generator if it has the attribute
    if hasattr(model.cot_generator, 'temperature'):
        model.cot_generator.temperature = temperature
    
    # Process the query
    result = model.process(query)
    
    # If inverted TFCA is enabled, perform additional analysis
    if inverted_tfca and 'reasoning_steps' in result:
        logger.info("Performing inverted TFCA analysis...")
        try:
            inverted_analyzer = InvertedTriadicFuzzyAnalysis()
            inverted_result = inverted_analyzer.analyze_reasoning(
                result['reasoning_steps'], 
                tau=0.5  # Using the same similarity threshold as the main model
            )
            
            # Add inverted TFCA results to the output
            result['inverted_tfca'] = {
                'concepts': inverted_result.get('concepts', []),
                'lattice': inverted_result.get('lattice', {}),
                'analysis': inverted_result.get('analysis', {})
            }
            logger.info(f"Inverted TFCA analysis completed. Found {len(result['inverted_tfca']['concepts'])} concepts.")
            
        except Exception as e:
            logger.error(f"Error during inverted TFCA analysis: {e}")
            result['inverted_tfca'] = {'error': str(e)}
    
    # Convert the result to include the full details needed for comparison
    full_results = {
        'answer': results.get('answer', ''),
        'uncertainty_bounds': results.get('uncertainty_bounds', (0, 0)),
        'reasoning_steps': results.get('reasoning_steps', []),
        'decisions': results.get('decisions', {}),
        'acceptance_intervals': results.get('acceptance_intervals', {}),
        'rejection_intervals': results.get('rejection_intervals', {}),
        'concept_lattice': results.get('concept_lattice', [])
    }
    
    return full_results

def main():
    """Run the provider comparison example."""
    args = parse_args()
    
    print("\n3WayCoT Framework Provider Comparison")
    print("=======================================\n")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Dictionary to store results for each provider
    all_results = {}
    
    # Process query with each provider
    provider_configs = [
        (args.providers[0], args.temperature1, f"{args.providers[0]} (T={args.temperature1})"),
        (args.providers[1] if len(args.providers) > 1 else args.providers[0], 
         args.temperature2, 
         f"{args.providers[1] if len(args.providers) > 1 else args.providers[0]} (T={args.temperature2})")
    ]
    
    for provider, temperature, display_name in provider_configs:
        # Skip if this is a duplicate provider (when only one provider is specified)
        if provider == provider_configs[0][0] and provider != provider_configs[1][0] and len(provider_configs) > 1:
            continue
        try:
            print(f"\nProcessing with {display_name}...")
            results = run_with_provider(
                provider, 
                args.query, 
                temperature, 
                max_assumptions=args.max_assumptions,
                inverted_tfca=args.inverted_tfca
            )
            all_results[provider] = results
            
            # Print individual provider results
            print(f"\n{provider.capitalize()} Results:")
            print(f"Steps Generated: {len(results['reasoning_steps'])}")
            print(f"Accept: {sum(1 for d in results['decisions'].values() if d == 'accept')}")
            print(f"Reject: {sum(1 for d in results['decisions'].values() if d == 'reject')}")
            print(f"Abstain: {sum(1 for d in results['decisions'].values() if d == 'abstain')}")
            print(f"Final Answer: {results['answer']}")
            
            # Save individual uncertainty plot
            uncertainty_values = {int(step['step_num']): step.get('uncertainty', 0) 
                                 for step in results['reasoning_steps']}
            
            uncertainty_plot_path = output_dir / f"{provider}_uncertainty_{timestamp}.png"
            visualizer.plot_uncertainty_distribution(
                uncertainty_values, 
                provider=provider,
                save_path=str(uncertainty_plot_path)
            )
            
            # Save decision space plot
            decision_data = {}
            for step in results['reasoning_steps']:
                step_num = step['step_num']
                step_data = {
                    'accept': step.get('acceptance_interval', (0, 0)),
                    'reject': step.get('rejection_interval', (0, 0))
                }
                decision_data[step_num] = step_data
                
            decision_plot_path = output_dir / f"{provider}_decision_space_{timestamp}.png"
            visualizer.plot_decision_space(
                decision_data,
                save_path=str(decision_plot_path)
            )
            
            # Save lattice plot if concept_lattice is available
            if 'concept_lattice' in results:
                lattice_plot_path = output_dir / f"{provider}_lattice_{timestamp}.png"
                visualizer.plot_concept_lattice(
                    results['concept_lattice'],
                    save_path=str(lattice_plot_path)
                )
                print(f"Concept lattice visualization saved to '{lattice_plot_path}'")
            
        except Exception as e:
            logger.error(f"Error processing with provider {provider}: {e}")
            print(f"Failed to process with {provider}: {e}")
    
    # Prepare data for comparison visualizations
    provider_results = {}
    for provider, results in all_results.items():
        uncertainty_values = {int(step['step_num']): step.get('uncertainty', 0) 
                             for step in results['reasoning_steps']}
        
        decisions = {step['step_num']: results['decisions'].get(step['step_num'], 'unknown') 
                    for step in results['reasoning_steps']}
        
        provider_results[provider] = {
            "uncertainty": uncertainty_values,
            "decisions": decisions,
            "steps": results['reasoning_steps']
        }
    
    # Generate comparison visualizations
    if len(provider_results) > 1:
        print("\nGenerating comparison visualizations...")
        
        # Uncertainty comparison
        uncertainty_comparison_path = output_dir / f"uncertainty_comparison_{timestamp}.png"
        visualizer.compare_providers(
            provider_results,
            comparison_type="uncertainty",
            save_path=str(uncertainty_comparison_path)
        )
        
        # Decision comparison
        decision_comparison_path = output_dir / f"decision_comparison_{timestamp}.png"
        visualizer.compare_providers(
            provider_results,
            comparison_type="decisions",
            save_path=str(decision_comparison_path)
        )
        
        # Step count comparison
        step_comparison_path = output_dir / f"step_comparison_{timestamp}.png"
        visualizer.compare_providers(
            provider_results,
            comparison_type="steps",
            save_path=str(step_comparison_path)
        )
        
        # Lattice comparison
        lattice_comparison_path = output_dir / f"lattice_comparison_{timestamp}.png"
        visualizer.compare_lattices(
            provider_results,
            save_path=str(lattice_comparison_path)
        )
        print(f"Lattice comparison visualization saved to '{lattice_comparison_path}'")
        
        print(f"\nResults saved to {output_dir}/")
    else:
        print("\nNot enough providers for comparison visualization")

if __name__ == "__main__":
    main()
