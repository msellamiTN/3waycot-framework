#!/usr/bin/env python3
"""
Test script for detailed Triadic FCA lattice metrics
"""

import os
import json
import logging
import sys
from pathlib import Path
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the necessary components
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TriadicFCA.Metrics")

def create_example_data():
    """Create a small example triadic context for testing"""
    # Create a simple set of reasoning steps with assumptions
    reasoning_steps = [
        {
            "step_num": 1,
            "reasoning": "Step 1 reasoning about medical treatment",
            "assumptions": [
                "Assumption 1.1: The treatment is effective",
                "Assumption 1.2: Side effects are minimal"
            ]
        },
        {
            "step_num": 2,
            "reasoning": "Step 2 analyzing the long-term effects",
            "assumptions": [
                "Assumption 2.1: Long-term data is limited",
                "Assumption 2.2: Safety monitoring is essential"
            ]
        },
        {
            "step_num": 3,
            "reasoning": "Step 3 ethical considerations",
            "assumptions": [
                "Assumption 3.1: Patient consent is informed",
                "Assumption 3.2: Benefits outweigh risks"
            ]
        }
    ]
    
    logger.info(f"Created example data with {len(reasoning_steps)} reasoning steps")
    return reasoning_steps

def test_enhanced_metrics(reasoning_steps, tau=0.5):
    """Test the enhanced triadic FCA metrics"""
    logger.info(f"Testing enhanced metrics with {len(reasoning_steps)} reasoning steps")
    
    # Create the Triadic FCA analyzer
    tfca = TriadicFuzzyFCAAnalysis()
    
    # Analyze reasoning steps
    analysis_results = tfca.analyze_reasoning(reasoning_steps, tau)
    
    # Extract and display the lattice analysis
    lattice_analysis = analysis_results.get("lattice_analysis", {})
    logger.info(f"Lattice analysis contains {len(lattice_analysis)} metrics")
    
    for metric_name, metric_value in lattice_analysis.items():
        if isinstance(metric_value, dict):
            logger.info(f"{metric_name}: [complex metric with {len(metric_value)} sub-metrics]")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    # Save the results to a dedicated file
    output_file = "enhanced_lattice_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.info(f"Saved detailed lattice metrics to {output_file}")
    return analysis_results

if __name__ == "__main__":
    # Create example data
    reasoning_steps = create_example_data()
    
    # Test enhanced metrics
    results = test_enhanced_metrics(reasoning_steps)
    
    # Display key counts
    logger.info(f"Number of concepts: {len(results.get('concepts', []))}")
    logger.info(f"Number of triadic connections: {len(results.get('triadic_connections', []))}")
    
    # Display the full lattice analysis structure
    logger.info("\nDetailed Lattice Analysis:")
    lattice_analysis = results.get("lattice_analysis", {})
    for metric_name, metric_value in lattice_analysis.items():
        if isinstance(metric_value, dict):
            logger.info(f"  {metric_name}:")
            for sub_name, sub_value in metric_value.items():
                if isinstance(sub_value, dict):
                    logger.info(f"    {sub_name}: {json.dumps(sub_value, indent=2)}")
                else:
                    logger.info(f"    {sub_name}: {sub_value}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")
