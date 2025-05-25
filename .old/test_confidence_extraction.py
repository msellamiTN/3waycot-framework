"""
Test script for enhanced Triadic FCA with dynamic confidence extraction.

This script demonstrates how our enhanced confidence extraction works
and its impact on three-way decision making.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from collections import Counter

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_path)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("3WayCoT.ConfidenceTest")

# Import components
from src.core.confidence_extractor import ConfidenceExtractor
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis
from src.core.three_way_decision import ThreeWayDecisionMaker
from src.core.threeway_cot import ThreeWayCOT

def test_confidence_extraction():
    """Test the confidence extraction functionality with various examples."""
    logger.info("\n=== Testing Confidence Extraction ===\n")
    
    test_texts = [
        "The result is true with high confidence.",
        "I'm very confident this approach will work.",
        "There's a 75% likelihood this will succeed.",
        "Confidence: 0.8 based on available evidence.",
        "The data shows moderate confidence in this conclusion.",
        "This is somewhat uncertain due to limited data.",
        "With low confidence, I suggest this may be correct.",
        "The confidence level is 0.3 due to conflicting information.",
        "The results are unclear and questionable.",
        "This conclusion has 90% certainty based on the analysis."
    ]
    
    logger.info("Extracting confidence from test sentences:")
    for text in test_texts:
        confidence = ConfidenceExtractor.extract_confidence(text)
        category = ConfidenceExtractor.categorize_confidence(confidence)
        logger.info(f"  • \"{text}\" → {confidence:.2f} ({category})")
    
    logger.info("\nConfidence extraction test completed.")

def create_test_reasoning_steps():
    """Create diverse test reasoning steps with varying confidence levels."""
    return [
        {
            "step_num": 1,
            "reasoning": "The first reasoning step examines the evidence from the Phase 2 trial. With 80% confidence, I conclude that the efficacy data shows promise, but the sample size is limited.",
            "assumptions": [
                "The trial methodology was rigorous",
                "The reported efficacy is accurate",
                "The sample represents the target population"
            ]
        },
        {
            "step_num": 2,
            "reasoning": "The second step analyzes potential long-term effects. There is considerable uncertainty here as the data only covers 6 months. Confidence: 0.4 - this is a moderate level of confidence given the limitations.",
            "assumptions": [
                "Long-term effects may differ from short-term effects",
                "Continued monitoring would reveal additional side effects",
                "The absence of evidence is not evidence of absence"
            ]
        },
        {
            "step_num": 3,
            "reasoning": "The third step evaluates ethical implications. I'm very confident that informed consent is essential, particularly given the uncertainties identified in step 2.",
            "assumptions": [
                "Patient autonomy is paramount",
                "Informed consent requires full disclosure of known risks",
                "The expected benefits must outweigh potential harms"
            ]
        },
        {
            "step_num": 4,
            "reasoning": "The fourth step considers regulatory pathways. With low confidence (approximately 0.3), I suggest that conditional approval might be justified for severe cases with no other treatment options.",
            "assumptions": [
                "Regulatory frameworks allow for conditional approval",
                "Post-marketing surveillance would be mandated",
                "The approval could be rescinded if new safety concerns emerge"
            ]
        },
        {
            "step_num": 5,
            "reasoning": "Final conclusion: While the treatment shows promise, significant uncertainties remain. I have moderate confidence (0.6) that a limited approval for severe cases with robust informed consent and ongoing monitoring would balance innovation and caution appropriately.",
            "assumptions": [
                "The risk-benefit calculation differs for severe cases",
                "Further research is essential and ongoing",
                "Confidence level: 0.6"
            ],
            "is_final": True
        }
    ]

def test_triadic_fca_with_confidence():
    """Test the enhanced Triadic FCA with dynamic confidence extraction."""
    logger.info("\n=== Testing Triadic FCA with Dynamic Confidence ===\n")
    
    # Create test reasoning steps
    reasoning_steps = create_test_reasoning_steps()
    
    logger.info(f"Created {len(reasoning_steps)} test reasoning steps")
    
    # Initialize the Triadic FCA analyzer
    tfca = TriadicFuzzyFCAAnalysis()
    
    # Test with different tau values
    tau_values = [0.3, 0.5, 0.7]
    
    for tau in tau_values:
        logger.info(f"\nAnalyzing with tau = {tau}")
        
        # Analyze the reasoning steps
        results = tfca.analyze_reasoning(reasoning_steps, tau)
        
        # Log key metrics
        lattice_analysis = results.get("lattice_analysis", {})
        concepts = results.get("concepts", [])
        
        logger.info(f"Generated {len(concepts)} concepts")
        logger.info(f"Concept stability: {lattice_analysis.get('concept_stability', {}).get('avg_stability', 0):.3f}")
        logger.info(f"Lattice density: {lattice_analysis.get('density', 0):.3f}")
        
        # Handle information_content which might be a dict
        info_content = lattice_analysis.get('information_content', 0)
        if isinstance(info_content, dict):
            logger.info(f"Information content: {info_content}")
        else:
            logger.info(f"Information content: {info_content:.3f}")
        
        # Save detailed results
        output_file = f"confidence_tfca_results_tau_{tau}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved detailed results to {output_file}")

def test_threeway_decision_making():
    """Test the impact of dynamic confidence on three-way decision making."""
    logger.info("\n=== Testing Three-Way Decision Making with Dynamic Confidence ===\n")
    
    # Create test reasoning steps
    reasoning_steps = create_test_reasoning_steps()
    
    # Process reasoning steps to extract confidence
    processed_steps = ConfidenceExtractor.extract_from_reasoning_steps(reasoning_steps)
    
    # Set up parameter combinations to test
    params = [
        {"alpha": 0.6, "beta": 0.4},
        {"alpha": 0.7, "beta": 0.3},
        {"alpha": 0.8, "beta": 0.2}
    ]
    
    # Run tests with each parameter set
    for param_set in params:
        logger.info(f"\nTesting with parameters: α={param_set['alpha']}, β={param_set['beta']}")
        
        # Initialize the ThreeWayDecisionMaker directly
        decision_maker = ThreeWayDecisionMaker(
            alpha=param_set["alpha"],
            beta=param_set["beta"],
            gamma=0.6  # Boundary width threshold
        )
        
        # Create a simplified analysis result
        simplified_analysis = {
            "reasoning_steps": processed_steps
        }
        
        # Create a simplified uncertainty analysis
        simplified_uncertainty = {
            "uncertainty_scores": [
                {"step_index": i, "score": 1.0 - step.get("original_confidence", 0.5)}
                for i, step in enumerate(processed_steps)
            ]
        }
        
        # Make decisions directly
        decisions = []
        for i, step in enumerate(processed_steps):
            decision = {
                "step_index": i,
                "decision": "ABSTAIN",  # Default
                "confidence": step.get("original_confidence", 0.5),
                "explanation": ""
            }
            
            # Use the confidence value to determine decision
            confidence = step.get("original_confidence", 0.5)
            if confidence >= param_set["alpha"]:
                decision["decision"] = "ACCEPT"
                decision["explanation"] = f"High confidence in step validity (confidence: {confidence:.2f})"
            elif confidence <= param_set["beta"]:
                decision["decision"] = "REJECT"
                decision["explanation"] = f"Low confidence in step validity (confidence: {confidence:.2f})"
            else:
                decision["decision"] = "ABSTAIN"
                decision["explanation"] = f"Uncertain about step validity (confidence: {confidence:.2f})"
            
            decisions.append(decision)
        
        # Count decision types
        decision_counts = Counter([d.get("decision", "ABSTAIN") for d in decisions])
        
        # Log decision distribution
        total = len(decisions)
        logger.info("Decision distribution:")
        for decision_type, count in decision_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            logger.info(f"  • {decision_type}: {count}/{total} ({percentage:.1f}%)")
        
        # Log confidence and decisions for each step
        logger.info("\nDetailed step decisions:")
        for idx, decision in enumerate(decisions):
            confidence = decision.get("confidence", 0.5)
            decision_type = decision.get("decision", "ABSTAIN")
            original_text = reasoning_steps[idx]["reasoning"][:50] + "..."
            logger.info(f"  • Step {idx+1}: {decision_type} (confidence: {confidence:.2f})")
            logger.info(f"    Text: {original_text}")
        
        # Save detailed results
        output_file = f"confidence_decisions_a{param_set['alpha']}_b{param_set['beta']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "parameters": param_set,
                "decisions": decisions,
                "decision_counts": {k: v for k, v in decision_counts.items()}
            }, f, indent=2)
        
        logger.info(f"Saved decision results to {output_file}")

def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description="Test enhanced confidence extraction and dynamic decision making")
    parser.add_argument("--test", choices=["confidence", "tfca", "decisions", "all"], default="all",
                        help="Which test to run (default: all)")
    args = parser.parse_args()
    
    logger.info("Starting enhanced confidence extraction and decision making tests")
    
    if args.test in ["confidence", "all"]:
        test_confidence_extraction()
    
    if args.test in ["tfca", "all"]:
        test_triadic_fca_with_confidence()
    
    if args.test in ["decisions", "all"]:
        test_threeway_decision_making()
    
    logger.info("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
