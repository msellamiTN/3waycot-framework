#!/usr/bin/env python3
"""
Simple test script for Triadic FCA lattice metrics
"""

import sys
import json
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the necessary components
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis

def main():
    # Create a simple example for testing
    print("Creating test data...")
    
    # Create example reasoning steps
    reasoning_steps = [
        {
            "step_num": 1,
            "reasoning": "First reasoning step about medical treatment",
            "assumptions": ["The treatment is effective", "Side effects are minimal"]
        },
        {
            "step_num": 2, 
            "reasoning": "Second reasoning step about long-term effects",
            "assumptions": ["Long-term data is limited", "Safety monitoring is essential"]
        },
        {
            "step_num": 3,
            "reasoning": "Third reasoning step with ethical considerations",
            "assumptions": ["Patient consent is informed", "Benefits outweigh risks"]
        }
    ]
    
    # Initialize the TFCA analyzer
    tfca = TriadicFuzzyFCAAnalysis()
    
    # Process the reasoning steps
    print("\nAnalyzing reasoning steps...")
    analysis_results = tfca.analyze_reasoning(reasoning_steps, tau=0.5)
    
    # Extract the lattice analysis
    lattice_analysis = analysis_results.get("lattice_analysis", {})
    
    # Print the results
    print("\n==== DETAILED LATTICE ANALYSIS METRICS ====")
    print(f"Number of concepts: {analysis_results.get('concepts', []).__len__()}")
    
    # Display top-level metrics
    for key, value in lattice_analysis.items():
        if isinstance(value, (int, float)):
            print(f"\n{key}: {value}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float, str)):
                    print(f"  {subkey}: {subvalue}")
                else:
                    print(f"  {subkey}: {type(subvalue).__name__}")
    
    # Save the results
    output_file = "detailed_lattice_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nSaved detailed lattice metrics to {output_file}")

if __name__ == "__main__":
    main()
