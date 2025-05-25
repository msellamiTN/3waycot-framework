#!/usr/bin/env python3
"""
Metrics Weights Experiment for Triadic FCA Analysis

This script allows you to experiment with different combinations of metric weights
to see how they affect the three-way decision distribution in the framework.
It provides visualizations and detailed analysis of the impact of each weight.
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import necessary components from the framework
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis, ConfidenceExtractor
from src.core.three_way_decision import ThreeWayDecisionMaker
from src.core.threeway_cot import ThreeWayCOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger("3WayCoT.MetricsExperiment")

class MetricsWeightsExperiment:
    """Run experiments with different metric weight combinations for three-way decisions."""
    
    def __init__(self, input_file: str, output_dir: str = None):
        """
        Initialize the experiment.
        
        Args:
            input_file: Path to the input file containing reasoning steps
            output_dir: Directory to save output files
        """
        self.input_file = input_file
        self.output_dir = output_dir or "experiment_results"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the input data
        self.data = self._load_data()
        
        # Extract reasoning steps
        self.reasoning_steps = self.data.get("reasoning_steps", [])
        if not self.reasoning_steps:
            logger.error(f"No reasoning steps found in {input_file}")
            sys.exit(1)
        
        # Initialize the confidence extractor
        self.confidence_extractor = ConfidenceExtractor()
        
        logger.info(f"Loaded {len(self.reasoning_steps)} reasoning steps from {input_file}")
        
        # Base weights template (starting point)
        self.base_weights = {
            "confidence": 0.6,         # Direct confidence from step
            "similarity": 0.15,        # Concept similarity score
            "coverage": 0.05,          # Assumption coverage
            "stability": 0.05,         # Concept stability
            "connectivity": 0.05,      # Graph connectivity
            "density": 0.05,           # Graph density
            "uncertainty": 0.05,       # Uncertainty score
        }
    
    def _load_data(self) -> Dict:
        """Load data from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)
    
    def _create_weight_variations(self, base_metric: str, variations: List[float]) -> List[Dict[str, float]]:
        """
        Create weight variations for a specific metric.
        
        Args:
            base_metric: The metric to vary
            variations: List of weights to test for this metric
            
        Returns:
            List of weight dictionaries with different values for the base metric
        """
        weight_configs = []
        
        for weight in variations:
            # Start with base weights
            weights = dict(self.base_weights)
            
            # Calculate the weight difference
            diff = weights[base_metric] - weight
            
            # Distribute the difference equally among other metrics
            other_metrics = [k for k in weights.keys() if k != base_metric]
            adjustment = diff / len(other_metrics)
            
            # Update the base metric weight
            weights[base_metric] = weight
            
            # Adjust other metrics to ensure sum = 1.0
            for metric in other_metrics:
                weights[metric] += adjustment
                
                # Ensure no negative weights
                weights[metric] = max(0.01, weights[metric])
            
            # Normalize to ensure weights sum to 1.0
            total = sum(weights.values())
            for k in weights:
                weights[k] /= total
                
            weight_configs.append(weights)
        
        return weight_configs
    
    def _calculate_decision_metrics(self, step: Dict, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate decision metrics for a step using the given weights.
        
        Args:
            step: The reasoning step to analyze
            weights: Dictionary of metric weights
            
        Returns:
            Dictionary of calculated metrics
        """
        # Extract confidence from the step using the confidence extractor
        confidence = step.get("original_confidence", 0.5)
        
        # Create placeholder values for other metrics (in a real system these would come from analysis)
        similarity = random.uniform(0.3, 0.8)
        coverage = random.uniform(0.3, 0.8)
        stability = random.uniform(0.3, 0.8)
        connectivity = random.uniform(0.3, 0.8)
        density = random.uniform(0.3, 0.8)
        uncertainty = random.uniform(0.3, 0.8)
        
        # Calculate membership degrees for each decision region
        accept_degree = (
            weights["confidence"] * confidence +
            weights["similarity"] * similarity +
            weights["coverage"] * coverage +
            weights["stability"] * stability
        )
        
        reject_degree = (
            weights["confidence"] * (1.0 - confidence) +
            weights["similarity"] * (1.0 - similarity) +
            weights["uncertainty"] * uncertainty +
            weights["connectivity"] * (1.0 - connectivity)
        )
        
        abstain_degree = (
            weights["confidence"] * abs(0.5 - confidence) +
            weights["density"] * (1.0 - density) +
            weights["uncertainty"] * uncertainty +
            weights["stability"] * (1.0 - stability)
        )
        
        # Normalize degrees to sum to 1.0
        total = accept_degree + reject_degree + abstain_degree
        if total > 0:
            accept_degree /= total
            reject_degree /= total
            abstain_degree /= total
        
        return {
            "confidence": confidence,
            "similarity": similarity,
            "coverage": coverage,
            "stability": stability,
            "connectivity": connectivity,
            "density": density,
            "uncertainty": uncertainty,
            "membership_degrees": {
                "accept": accept_degree,
                "reject": reject_degree,
                "abstain": abstain_degree
            }
        }
    
    def _make_decision(self, metrics: Dict[str, Any], alpha: float, beta: float) -> str:
        """
        Make a three-way decision based on the calculated metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            alpha: Lower threshold for positive region
            beta: Upper threshold for negative region
            
        Returns:
            Decision ("ACCEPT", "REJECT", or "ABSTAIN")
        """
        # Extract confidence and membership degrees
        confidence = metrics["confidence"]
        membership = metrics["membership_degrees"]
        
        # Get the membership degrees
        accept_degree = membership["accept"]
        reject_degree = membership["reject"]
        abstain_degree = membership["abstain"]
        
        # Direct confidence-based decision rules
        if confidence >= alpha:
            return "ACCEPT"
        elif confidence <= beta:
            return "REJECT"
        
        # Membership-based decision rules
        max_degree = max(accept_degree, reject_degree, abstain_degree)
        
        if max_degree == accept_degree and accept_degree >= alpha:
            return "ACCEPT"
        elif max_degree == reject_degree and reject_degree >= (1.0 - beta):
            return "REJECT"
        else:
            return "ABSTAIN"
    
    def run_single_experiment(self, weights: Dict[str, float], alpha: float, beta: float) -> Dict[str, Any]:
        """
        Run a single experiment with the given weights and thresholds.
        
        Args:
            weights: Dictionary of metric weights
            alpha: Lower threshold for positive region
            beta: Upper threshold for negative region
            
        Returns:
            Dictionary of experiment results
        """
        decisions = []
        decision_counts = {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0}
        
        for i, step in enumerate(self.reasoning_steps):
            # Calculate metrics for this step
            metrics = self._calculate_decision_metrics(step, weights)
            
            # Make decision based on calculated metrics
            decision = self._make_decision(metrics, alpha, beta)
            
            # Record decision
            decisions.append({
                "step_index": i,
                "decision": decision,
                "confidence": metrics["confidence"],
                "membership_degrees": metrics["membership_degrees"]
            })
            
            # Update decision counts
            decision_counts[decision] += 1
        
        return {
            "weights": weights,
            "alpha": alpha,
            "beta": beta,
            "decisions": decisions,
            "decision_counts": decision_counts
        }
    
    def run_weight_experiments(self, metric_to_vary: str, variations: List[float], 
                              thresholds: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Run experiments by varying weights for a specific metric.
        
        Args:
            metric_to_vary: The metric to vary weights for
            variations: List of weight values to test
            thresholds: List of alpha/beta threshold configurations
            
        Returns:
            Dictionary of experiment results
        """
        weight_configs = self._create_weight_variations(metric_to_vary, variations)
        
        logger.info(f"Running experiments with {len(weight_configs)} weight configurations "
                   f"and {len(thresholds)} threshold configurations")
        
        results = {}
        
        # For each weight configuration
        for i, weights in enumerate(weight_configs):
            weight_key = f"weights_{i+1}_{metric_to_vary}_{variations[i]:.2f}"
            threshold_results = {}
            
            # For each threshold configuration
            for j, thresh in enumerate(thresholds):
                alpha = thresh["alpha"]
                beta = thresh["beta"]
                
                # Run experiment
                experiment_result = self.run_single_experiment(weights, alpha, beta)
                
                # Store result
                threshold_key = f"threshold_{j+1}_alpha_{alpha:.2f}_beta_{beta:.2f}"
                threshold_results[threshold_key] = experiment_result
            
            # Store results for this weight configuration
            results[weight_key] = {
                "metric_varied": metric_to_vary,
                "weight_value": variations[i],
                "weights": weights,
                "threshold_results": threshold_results
            }
        
        return results
    
    def run_comprehensive_experiments(self) -> Dict[str, Any]:
        """
        Run a comprehensive set of experiments varying weights for all metrics.
        
        Returns:
            Dictionary of all experiment results
        """
        metrics_to_vary = list(self.base_weights.keys())
        all_results = {}
        
        # Define weight variations and thresholds
        variations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        thresholds = [
            {"alpha": 0.6, "beta": 0.4},
            {"alpha": 0.7, "beta": 0.3},
            {"alpha": 0.8, "beta": 0.2},
            {"alpha": 0.75, "beta": 0.25},
            {"alpha": 0.65, "beta": 0.35}
        ]
        
        # Run experiments for each metric
        for metric in metrics_to_vary:
            logger.info(f"Varying weights for metric: {metric}")
            metric_results = self.run_weight_experiments(metric, variations, thresholds)
            all_results[f"metric_{metric}"] = metric_results
        
        return all_results
    
    def visualize_results(self, results: Dict[str, Any]) -> List[str]:
        """
        Visualize experiment results.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            List of paths to generated visualization files
        """
        visualization_files = []
        
        # For each metric
        for metric_key, metric_results in results.items():
            metric_name = metric_key.replace("metric_", "")
            
            # For each threshold configuration
            for thresh_idx in range(1, 6):  # Assuming 5 threshold configurations
                # Find matching threshold keys - they include the threshold index but also alpha/beta values
                matching_thresh_keys = []
                example_thresh_info = None
                
                # Loop through the first weight configuration to find all threshold keys
                if len(metric_results) > 0:
                    first_weight_key = list(metric_results.keys())[0]
                    first_weight_results = metric_results[first_weight_key]
                    for thresh_key in first_weight_results["threshold_results"].keys():
                        if thresh_key.startswith(f"threshold_{thresh_idx}_"):
                            matching_thresh_keys.append(thresh_key)
                            example_thresh_info = first_weight_results["threshold_results"][thresh_key]
                
                # Skip if no matching threshold keys found
                if not matching_thresh_keys or not example_thresh_info:
                    continue
                    
                # Use the first matching threshold key
                thresh_key = matching_thresh_keys[0]
                
                # Prepare data for visualization
                weights = []
                accept_counts = []
                reject_counts = []
                abstain_counts = []
                
                # Extract data for each weight configuration
                for weight_key, weight_results in metric_results.items():
                    if thresh_key in weight_results["threshold_results"]:
                        weights.append(weight_results["weight_value"])
                        counts = weight_results["threshold_results"][thresh_key]["decision_counts"]
                        accept_counts.append(counts["ACCEPT"])
                        reject_counts.append(counts["REJECT"])
                        abstain_counts.append(counts["ABSTAIN"])
                
                # Sort data by weight value
                sorted_indices = np.argsort(weights)
                weights = [weights[i] for i in sorted_indices]
                accept_counts = [accept_counts[i] for i in sorted_indices]
                reject_counts = [reject_counts[i] for i in sorted_indices]
                abstain_counts = [abstain_counts[i] for i in sorted_indices]
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                width = 0.25
                
                x = np.arange(len(weights))
                ax.bar(x - width, accept_counts, width, label='ACCEPT', color='green')
                ax.bar(x, reject_counts, width, label='REJECT', color='red')
                ax.bar(x + width, abstain_counts, width, label='ABSTAIN', color='blue')
                
                # Add labels and title
                # Use the example_thresh_info we saved earlier
                alpha = example_thresh_info["alpha"]
                beta = example_thresh_info["beta"]
                
                ax.set_xlabel(f'{metric_name} Weight')
                ax.set_ylabel('Count')
                ax.set_title(f'Decision Distribution by {metric_name} Weight (α={alpha:.2f}, β={beta:.2f})')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{w:.2f}' for w in weights])
                ax.legend()
                
                # Save figure
                filename = f"decision_dist_{metric_name}_{thresh_key}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath)
                plt.close(fig)
                
                visualization_files.append(filepath)
                logger.info(f"Created visualization: {filepath}")
        
        return visualization_files
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """
        Save experiment results to a file.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            Path to the output file
        """
        output_file = os.path.join(self.output_dir, "weight_experiment_results.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved experiment results to {output_file}")
        return output_file
    
    def run_and_visualize(self) -> Tuple[str, List[str]]:
        """
        Run experiments and visualize results.
        
        Returns:
            Tuple of (results_file_path, visualization_file_paths)
        """
        # Run experiments
        results = self.run_comprehensive_experiments()
        
        # Save results
        results_file = self.save_results(results)
        
        # Visualize results
        visualization_files = self.visualize_results(results)
        
        return results_file, visualization_files

def add_varied_confidence(steps: List[Dict]) -> List[Dict]:
    """
    Add varied confidence values to reasoning steps.
    
    Args:
        steps: List of reasoning steps
        
    Returns:
        List of steps with varied confidence values
    """
    confidence_patterns = [
        "With {confidence:.2f} confidence,",
        "I'm {confidence_pct}% confident that",
        "Confidence level: {confidence:.2f}",
        "This step has {confidence_level} certainty ({confidence:.2f}).",
        "I believe with {confidence:.2f} confidence that"
    ]
    
    confidence_levels = {
        0.1: "very low",
        0.2: "very low",
        0.3: "low",
        0.4: "somewhat low",
        0.5: "moderate",
        0.6: "somewhat high",
        0.7: "high",
        0.8: "very high",
        0.9: "very high",
    }
    
    processed_steps = []
    
    for step in steps:
        processed_step = dict(step)
        
        # Generate a random confidence value
        confidence = round(random.uniform(0.1, 0.9), 2)
        confidence_pct = int(confidence * 100)
        confidence_level = confidence_levels[round(confidence, 1)]
        
        # Format a confidence statement
        pattern = random.choice(confidence_patterns)
        confidence_statement = pattern.format(
            confidence=confidence,
            confidence_pct=confidence_pct,
            confidence_level=confidence_level
        )
        
        # Add confidence to the step
        processed_step["original_confidence"] = confidence
        
        # Add confidence statement to reasoning text
        reasoning = processed_step.get("reasoning", "")
        processed_step["reasoning"] = f"{confidence_statement} {reasoning}"
        
        processed_steps.append(processed_step)
    
    return processed_steps

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run experiments with varying metric weights")
    parser.add_argument("--input", default="enhanced_metrics_results.json",
                        help="Input JSON file with reasoning steps")
    parser.add_argument("--output", default="weight_experiments",
                        help="Directory to save output files")
    parser.add_argument("--add_confidence", action="store_true",
                        help="Add varied confidence values to reasoning steps")
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MetricsWeightsExperiment(args.input, args.output)
    
    # Add varied confidence if requested
    if args.add_confidence:
        logger.info("Adding varied confidence values to reasoning steps")
        experiment.reasoning_steps = add_varied_confidence(experiment.reasoning_steps)
        
        # Save the steps with added confidence
        with open(os.path.join(args.output, "steps_with_confidence.json"), 'w') as f:
            json.dump({"reasoning_steps": experiment.reasoning_steps}, f, indent=2)
    
    # Run experiments and visualize results
    results_file, visualization_files = experiment.run_and_visualize()
    
    logger.info(f"Experiment completed successfully")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Created {len(visualization_files)} visualizations")

if __name__ == "__main__":
    main()
