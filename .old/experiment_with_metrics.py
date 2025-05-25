#!/usr/bin/env python3
"""
Experiment with different metrics in the Triadic FCA framework.

This script allows you to vary the weights of different metrics and see how
they affect the three-way decision distribution (ACCEPT, REJECT, ABSTAIN).
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import necessary components
from src.core.confidence_extractor import ConfidenceExtractor
from src.core.three_way_decision import ThreeWayDecisionMaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("3WayCoT.MetricsExperiment")

class MetricsExperiment:
    """Class to experiment with different metrics in the Triadic FCA framework."""
    
    def __init__(self, input_file, output_dir=None):
        """
        Initialize the experiment.
        
        Args:
            input_file: Path to the input JSON file
            output_dir: Directory to save output files (defaults to current directory)
        """
        self.input_file = input_file
        self.output_dir = output_dir or "."
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_data()
        
        # Extract steps for processing
        self.reasoning_steps = self.data.get("reasoning_steps", [])
        if not self.reasoning_steps:
            logger.error(f"No reasoning steps found in {input_file}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(self.reasoning_steps)} reasoning steps from {input_file}")
        
        # Default metric weights
        self.default_weights = {
            "confidence": 0.6,
            "concept_similarity": 0.15,
            "concept_stability": 0.1,
            "uncertainty": 0.05,
            "connectivity": 0.05,
            "density": 0.025,
            "coverage": 0.025
        }
    
    def _load_data(self):
        """Load data from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)
    
    def _inject_confidence_values(self, confidence_distribution="varied"):
        """
        Inject confidence values into the reasoning steps.
        
        Args:
            confidence_distribution: Type of confidence distribution to inject
                "varied" - Random values with natural variation
                "high" - Mostly high values 
                "low" - Mostly low values
                "medium" - Mostly medium values
                "extreme" - Very high and very low values
                
        Returns:
            List of steps with injected confidence values
        """
        # Process steps to extract/inject confidence
        processed_steps = []
        
        logger.info(f"Injecting {confidence_distribution} confidence distribution")
        
        for i, step in enumerate(self.reasoning_steps):
            # Create a copy of the step
            processed_step = dict(step)
            
            # Calculate confidence based on the selected distribution
            if confidence_distribution == "varied":
                # Random values between 0.2 and 0.9
                confidence = random.uniform(0.2, 0.9)
            elif confidence_distribution == "high":
                # High values between 0.7 and 0.95
                confidence = random.uniform(0.7, 0.95)
            elif confidence_distribution == "low":
                # Low values between 0.2 and 0.45
                confidence = random.uniform(0.2, 0.45)
            elif confidence_distribution == "medium":
                # Medium values between 0.45 and 0.7
                confidence = random.uniform(0.45, 0.7)
            elif confidence_distribution == "extreme":
                # Either very high or very low
                confidence = random.choice([
                    random.uniform(0.85, 0.98),
                    random.uniform(0.02, 0.15)
                ])
            else:
                # Default to medium confidence
                confidence = 0.5
            
            # Round to 2 decimal places
            confidence = round(confidence, 2)
            
            # Inject the confidence value
            processed_step["original_confidence"] = confidence
            
            # Add a confidence expression to the reasoning text
            reasoning_text = processed_step.get("reasoning", "")
            if "confidence" not in reasoning_text.lower():
                confidence_expr = f"Confidence: {confidence}"
                # Add to the beginning of the text
                processed_step["reasoning"] = f"{reasoning_text}\n{confidence_expr}"
            
            # Add to the processed steps
            processed_steps.append(processed_step)
            
            logger.info(f"Step {i+1}: Injected confidence {confidence:.2f}")
        
        return processed_steps
    
    def _calculate_memberships(self, weights, confidence, concept_similarity, concept_stability, 
                              uncertainty, connectivity, density, coverage):
        """
        Calculate membership degrees for the three regions using custom weights.
        
        Args:
            weights: Dictionary of weights for each metric
            Other parameters: The actual metric values
            
        Returns:
            Dictionary of membership degrees for the three regions
        """
        # Calculate membership degrees for the three regions
        accept_score = (
            weights["confidence"] * confidence + 
            weights["concept_similarity"] * concept_similarity + 
            weights["concept_stability"] * concept_stability + 
            weights["density"] * density
        )
        
        reject_score = (
            weights["confidence"] * (1.0 - confidence) + 
            weights["uncertainty"] * uncertainty + 
            weights["concept_similarity"] * (1.0 - concept_similarity) + 
            weights["coverage"] * (1.0 - coverage)
        )
        
        abstain_score = (
            weights["confidence"] * abs(0.5 - confidence) + 
            weights["density"] * (1.0 - density) + 
            weights["connectivity"] * (1.0 - connectivity) + 
            weights["uncertainty"] * uncertainty
        )
        
        # Normalize the scores
        total = accept_score + reject_score + abstain_score
        if total > 0:
            accept_score /= total
            reject_score /= total
            abstain_score /= total
        
        return {
            "accept": accept_score,
            "reject": reject_score,
            "abstain": abstain_score
        }
    
    def _make_decisions(self, steps, weights, alpha, beta):
        """
        Make three-way decisions for the given steps.
        
        Args:
            steps: List of reasoning steps
            weights: Dictionary of weights for each metric
            alpha: Lower bound for positive region
            beta: Upper bound for negative region
            
        Returns:
            List of decisions, one for each step
        """
        # Initialize decision maker
        decision_maker = ThreeWayDecisionMaker(alpha=alpha, beta=beta)
        
        # Make decisions for each step
        decisions = []
        decision_types = Counter()
        
        for i, step in enumerate(steps):
            # Extract confidence
            confidence = step.get("original_confidence", 0.5)
            
            # Get membership degrees using custom weights
            memberships = self._calculate_memberships(
                weights,
                confidence,
                0.5,  # Default concept_similarity
                0.5,  # Default concept_stability
                0.5,  # Default uncertainty
                0.5,  # Default connectivity
                0.5,  # Default density
                0.5   # Default coverage
            )
            
            # Determine decision based on membership degrees
            decision = self._determine_decision(memberships, confidence, alpha, beta)
            
            # Count decision type
            decision_types[decision["decision"]] += 1
            
            # Add decision to list
            decisions.append(decision)
            
            logger.info(f"Step {i+1}: {decision['decision']} (confidence: {confidence:.2f})")
        
        # Log decision distribution
        logger.info(f"Decision distribution: {dict(decision_types)}")
        
        return decisions, decision_types
    
    def _determine_decision(self, memberships, confidence, alpha, beta):
        """
        Determine the three-way decision based on membership degrees and thresholds.
        
        Args:
            memberships: Dictionary of membership degrees
            confidence: Confidence value
            alpha: Lower bound for positive region
            beta: Upper bound for negative region
            
        Returns:
            Decision dictionary
        """
        # Extract membership scores
        accept_score = memberships["accept"]
        reject_score = memberships["reject"]
        abstain_score = memberships["abstain"]
        
        # Find the highest score
        scores = {
            "ACCEPT": accept_score,
            "REJECT": reject_score,
            "ABSTAIN": abstain_score
        }
        max_decision = max(scores, key=scores.get)
        max_score = scores[max_decision]
        
        # Determine decision
        if confidence >= alpha:
            decision = "ACCEPT"
            explanation = f"High confidence in step validity (confidence={confidence:.2f})"
        elif confidence <= beta:
            decision = "REJECT"
            explanation = f"Low confidence in step validity (confidence={confidence:.2f})"
        elif max_decision == "ACCEPT" and accept_score >= alpha:
            decision = "ACCEPT"
            explanation = f"High membership in positive region (accept={accept_score:.2f})"
        elif max_decision == "REJECT" and reject_score >= (1.0 - beta):
            decision = "REJECT"
            explanation = f"High membership in negative region (reject={reject_score:.2f})"
        else:
            decision = "ABSTAIN"
            explanation = f"Uncertain about step validity (accept={accept_score:.2f}, reject={reject_score:.2f}, abstain={abstain_score:.2f})"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "explanation": explanation,
            "memberships": memberships
        }
    
    def run_experiment(self, metric_configs, thresholds, confidence_distributions=None):
        """
        Run the experiment with different metric configurations and thresholds.
        
        Args:
            metric_configs: List of dictionaries with metric weights
            thresholds: List of dictionaries with alpha and beta values
            confidence_distributions: List of confidence distributions to test
            
        Returns:
            Dictionary of experiment results
        """
        results = {}
        
        # Use default confidence distribution if none provided
        if not confidence_distributions:
            confidence_distributions = ["varied"]
        
        # Run experiments for each configuration
        for conf_dist in confidence_distributions:
            # Inject confidence values
            steps = self._inject_confidence_values(conf_dist)
            
            conf_results = {}
            
            # Test each metric configuration
            for i, metrics in enumerate(metric_configs):
                metric_results = {}
                
                # Test each threshold configuration
                for j, thresh in enumerate(thresholds):
                    alpha = thresh["alpha"]
                    beta = thresh["beta"]
                    
                    # Make decisions
                    decisions, counts = self._make_decisions(steps, metrics, alpha, beta)
                    
                    # Store results
                    metric_results[f"threshold_{j+1}"] = {
                        "alpha": alpha,
                        "beta": beta,
                        "decisions": decisions,
                        "counts": dict(counts)
                    }
                
                # Store metric configuration results
                conf_results[f"metrics_{i+1}"] = {
                    "weights": metrics,
                    "thresholds": metric_results
                }
            
            # Store confidence distribution results
            results[conf_dist] = conf_results
        
        return results
    
    def visualize_results(self, results):
        """
        Create visualizations of the experiment results.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            List of generated visualization files
        """
        visualization_files = []
        
        # Create visualizations for each confidence distribution
        for conf_dist, conf_results in results.items():
            # Create a bar chart for each metric configuration
            for metric_key, metric_results in conf_results.items():
                # Extract thresholds and counts
                thresholds = []
                accepts = []
                rejects = []
                abstains = []
                
                for thresh_key, thresh_results in metric_results["thresholds"].items():
                    thresholds.append(f"α={thresh_results['alpha']}, β={thresh_results['beta']}")
                    
                    counts = thresh_results["counts"]
                    accepts.append(counts.get("ACCEPT", 0))
                    rejects.append(counts.get("REJECT", 0))
                    abstains.append(counts.get("ABSTAIN", 0))
                
                # Create the visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Set the width of the bars
                width = 0.25
                
                # Set the positions of the bars on the x-axis
                r1 = np.arange(len(thresholds))
                r2 = [x + width for x in r1]
                r3 = [x + width for x in r2]
                
                # Create the bars
                ax.bar(r1, accepts, width, label='ACCEPT', color='green')
                ax.bar(r2, rejects, width, label='REJECT', color='red')
                ax.bar(r3, abstains, width, label='ABSTAIN', color='blue')
                
                # Add labels, title and legend
                ax.set_xlabel('Threshold Configuration')
                ax.set_ylabel('Count')
                ax.set_title(f'Decision Distribution - {conf_dist.capitalize()} Confidence, {metric_key}')
                ax.set_xticks([r + width for r in range(len(thresholds))])
                ax.set_xticklabels(thresholds)
                ax.legend()
                
                # Save the figure
                filename = f"decision_dist_{conf_dist}_{metric_key}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath)
                plt.close(fig)
                
                visualization_files.append(filepath)
                logger.info(f"Created visualization: {filepath}")
        
        return visualization_files
    
    def save_results(self, results):
        """
        Save the experiment results to a file.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            Path to the saved file
        """
        # Create output filename
        output_file = os.path.join(self.output_dir, "metrics_experiment_results.json")
        
        # Save the results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved experiment results to {output_file}")
        return output_file

def generate_metric_configs():
    """Generate a set of metric configurations to test."""
    configs = []
    
    # Default configuration
    configs.append({
        "confidence": 0.6,
        "concept_similarity": 0.15,
        "concept_stability": 0.1,
        "uncertainty": 0.05,
        "connectivity": 0.05,
        "density": 0.025,
        "coverage": 0.025
    })
    
    # High confidence weight
    configs.append({
        "confidence": 0.8,
        "concept_similarity": 0.08,
        "concept_stability": 0.05,
        "uncertainty": 0.03,
        "connectivity": 0.02,
        "density": 0.01,
        "coverage": 0.01
    })
    
    # High similarity weight
    configs.append({
        "confidence": 0.4,
        "concept_similarity": 0.35,
        "concept_stability": 0.1,
        "uncertainty": 0.05,
        "connectivity": 0.05,
        "density": 0.025,
        "coverage": 0.025
    })
    
    # High uncertainty weight
    configs.append({
        "confidence": 0.4,
        "concept_similarity": 0.15,
        "concept_stability": 0.1,
        "uncertainty": 0.25,
        "connectivity": 0.05,
        "density": 0.025,
        "coverage": 0.025
    })
    
    # Equal weights
    configs.append({
        "confidence": 0.143,
        "concept_similarity": 0.143,
        "concept_stability": 0.143,
        "uncertainty": 0.143,
        "connectivity": 0.143,
        "density": 0.143,
        "coverage": 0.142
    })
    
    return configs

def generate_threshold_configs():
    """Generate a set of threshold configurations to test."""
    configs = []
    
    # Default thresholds
    configs.append({"alpha": 0.7, "beta": 0.3})
    
    # High accept threshold
    configs.append({"alpha": 0.8, "beta": 0.3})
    
    # Low accept threshold
    configs.append({"alpha": 0.6, "beta": 0.3})
    
    # High reject threshold
    configs.append({"alpha": 0.7, "beta": 0.4})
    
    # Low reject threshold
    configs.append({"alpha": 0.7, "beta": 0.2})
    
    # Small uncertain region
    configs.append({"alpha": 0.65, "beta": 0.55})
    
    # Large uncertain region
    configs.append({"alpha": 0.85, "beta": 0.15})
    
    return configs

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Experiment with different metrics in the Triadic FCA framework")
    parser.add_argument("--input", default="enhanced_metrics_results.json", help="Input JSON file")
    parser.add_argument("--output", default="metric_experiments", help="Output directory")
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MetricsExperiment(args.input, args.output)
    
    # Generate configurations
    metric_configs = generate_metric_configs()
    threshold_configs = generate_threshold_configs()
    confidence_distributions = ["varied", "high", "low", "medium", "extreme"]
    
    # Run experiment
    logger.info("Running experiment with different metric configurations and thresholds")
    results = experiment.run_experiment(metric_configs, threshold_configs, confidence_distributions)
    
    # Save results
    experiment.save_results(results)
    
    # Visualize results
    experiment.visualize_results(results)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
