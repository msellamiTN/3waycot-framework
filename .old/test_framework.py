#!/usr/bin/env python3
"""
Test script for the enhanced 3WayCoT framework.

This script tests the enhanced confidence extraction and decision-making components
to verify that our changes work correctly.
"""

import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("3WayCoT.Test")

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

# Import the framework components
from src.core.confidence_extractor import ConfidenceExtractor
from src.core.three_way_decision import ThreeWayDecisionMaker

def test_confidence_extraction():
    """
    Test the enhanced confidence extraction functionality.
    """
    logger.info("Testing confidence extraction...")
    extractor = ConfidenceExtractor()
    
    # Test cases with varying confidence expressions
    test_cases = [
        "I am 90% confident that this approach will work.",
        "This seems like the right solution (confidence: high).",
        "I'm not very sure about this step, maybe 30% confident.",
        "This is definitely the correct approach.",
        "I'm somewhat confident that this will work.",
        "I'm fairly uncertain about this approach.",
        "This should work with about 75% confidence.",
        "It's hard to say if this is right, maybe 50-50.",
        "With low confidence, I think this might work.",
        "I have high confidence in this calculation."
    ]
    
    results = []
    for i, text in enumerate(test_cases):
        confidence, method = extractor.extract_confidence(text)
        results.append({
            "text": text,
            "confidence": confidence,
            "method": method
        })
        logger.info(f"Case {i+1}: Confidence = {confidence:.2f}, Method = {method}")
    
    # Test confidence distribution analysis
    steps = [{'content': text, 'confidence': conf, 'method': method} 
             for text, (conf, method) in zip(test_cases, 
                                          [extractor.extract_confidence(t) for t in test_cases])]
    
    distribution = extractor.analyze_confidence_distribution(steps)
    logger.info(f"Confidence distribution: {distribution}")
    
    return results, distribution

def test_decision_making(confidence_results):
    """
    Test the enhanced decision-making with the extracted confidence values.
    """
    logger.info("Testing three-way decision making...")
    
    # Create steps with confidence values
    steps = []
    for i, result in enumerate(confidence_results):
        steps.append({
            "step_num": i + 1,
            "content": result["text"],
            "confidence": result["confidence"],
            "similarity_score": 0.7,  # Example similarity score
            "assumption_coverage": 0.8  # Example coverage score
        })
    
    # Initialize the decision maker with appropriate thresholds
    decision_maker = ThreeWayDecisionMaker(alpha=0.7, beta=0.3, tau=0.5)
    
    # Create analysis results structure
    analysis = {
        "reasoning_steps": steps,
        "confidence_metrics": {
            "average": sum(step["confidence"] for step in steps) / len(steps),
            "max": max(step["confidence"] for step in steps),
            "min": min(step["confidence"] for step in steps),
            "variance": 0.05  # Example variance
        }
    }
    
    # Create uncertainty analysis with confidence distribution
    uncertainty_analysis = {
        "confidence_distribution": distribution,
        "step_uncertainties": [{"confidence": step["confidence"]} for step in steps]
    }
    
    # Make decisions
    decisions = decision_maker.make_decisions(
        analysis=analysis,
        uncertainty_analysis=uncertainty_analysis
    )
    
    # Log the decisions
    logger.info(f"Decision summary: {decisions['summary']}")
    for i, decision in enumerate(decisions['decisions']):
        logger.info(f"Step {i+1}: {decision['decision']} (Confidence: {decision['confidence']:.2f})")
    
    return decisions

def save_results(confidence_results, distribution, decisions, output_path):
    """
    Save the test results to a JSON file.
    """
    results = {
        "confidence_extraction": confidence_results,
        "confidence_distribution": distribution,
        "decisions": decisions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Run the tests
    confidence_results, distribution = test_confidence_extraction()
    decisions = test_decision_making(confidence_results)
    
    # Save the results
    output_path = Path(__file__).parent / "results" / "enhanced_test_output.json"
    os.makedirs(output_path.parent, exist_ok=True)
    save_results(confidence_results, distribution, decisions, output_path)
    
    logger.info("Tests completed successfully!")
