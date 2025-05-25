#!/usr/bin/env python3
"""
Benchmark Metrics Experiment for Three-Way Decision Framework

This script runs benchmark tests with varying metric weights to analyze
how different metrics affect the three-way decision distribution.
It uses the actual framework components and benchmark data.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import necessary components
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis
from src.core.three_way_decision import ThreeWayDecisionMaker
from src.core.threeway_cot import ThreeWayCOT
from src.core.uncertainty_resolver import UncertaintyResolver

# Simple Config class implementation
class Config:
    """Simple configuration class for the experiment."""
    
    def __init__(self):
        """Initialize with default configuration."""
        self.config = {
            "default_provider": "gemini",
            "default_model": ""
        }
        
        # Try to load configuration from file
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config.update(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("3WayCoT.BenchmarkExperiment")

class MetricWeightExperiment:
    """Class to run benchmark experiments with varying metric weights."""
    
    def __init__(self, benchmark_file: str, output_dir: str = None):
        """
        Initialize the experiment.
        
        Args:
            benchmark_file: Path to the benchmark JSON file
            output_dir: Directory to save output files
        """
        self.benchmark_file = benchmark_file
        self.output_dir = output_dir or "benchmark_metrics"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load benchmark data
        self.benchmark_data = self._load_benchmark_data()
        
        # Load configuration
        self.config = Config()
        
        # Default weight sets
        self.default_weights = {
            "confidence": 0.6,
            "similarity": 0.15,
            "coverage": 0.05,
            "concept_stability": 0.05,
            "connectivity": 0.05,
            "density": 0.05,
            "uncertainty": 0.05
        }
        
        logger.info(f"Loaded benchmark data with {len(self.benchmark_data)} items")
    
    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark data from file."""
        try:
            with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            sys.exit(1)
    
    def _apply_weights_to_decision_maker(self, decision_maker: ThreeWayDecisionMaker, 
                                        weights: Dict[str, float]):
        """
        Apply custom weights to the decision maker.
        
        Args:
            decision_maker: ThreeWayDecisionMaker instance
            weights: Dictionary of weights for different metrics
        """
        # Store the original method for later restoration
        original_method = decision_maker._calculate_decision_metrics
        
        # Define a new method that applies custom weights
        def weighted_calculate_metrics(step, analysis, uncertainty, step_index):
            # Call the original method to get base metrics
            metrics = original_method(step, analysis, uncertainty, step_index)
            
            # Extract membership degrees if available
            membership_degrees = metrics.get("membership_degrees", {})
            
            # Recalculate membership degrees with custom weights if available
            if membership_degrees and "accept" in membership_degrees and "reject" in membership_degrees:
                # Extract confidence
                confidence = step.get("original_confidence", 0.5)
                
                # Extract other metrics
                similarity = metrics.get("similarity_score", 0.5)
                coverage = metrics.get("assumption_coverage", 0.5)
                concept_stability = uncertainty.get("lattice_metrics", {}).get("concept_stability", 0.5)
                connectivity = uncertainty.get("lattice_metrics", {}).get("connectivity", 0.5)
                density = uncertainty.get("lattice_metrics", {}).get("density", 0.5)
                uncertainty_score = metrics.get("uncertainty_score", 0.5)
                
                # Calculate weighted scores for each decision region
                accept_score = (
                    weights["confidence"] * confidence +
                    weights["similarity"] * similarity +
                    weights["coverage"] * coverage +
                    weights["concept_stability"] * concept_stability
                )
                
                reject_score = (
                    weights["confidence"] * (1.0 - confidence) +
                    weights["similarity"] * (1.0 - similarity) +
                    weights["uncertainty"] * uncertainty_score +
                    weights["connectivity"] * (1.0 - connectivity)
                )
                
                abstain_score = (
                    weights["confidence"] * abs(0.5 - confidence) +
                    weights["density"] * (1.0 - density) +
                    weights["uncertainty"] * uncertainty_score +
                    weights["concept_stability"] * (1.0 - concept_stability)
                )
                
                # Normalize scores
                total = accept_score + reject_score + abstain_score
                if total > 0:
                    accept_score /= total
                    reject_score /= total
                    abstain_score /= total
                
                # Update membership degrees
                metrics["membership_degrees"] = {
                    "accept": accept_score,
                    "reject": reject_score,
                    "abstain": abstain_score
                }
            
            return metrics
        
        # Replace the method with our custom one
        decision_maker._calculate_decision_metrics = weighted_calculate_metrics.__get__(decision_maker, type(decision_maker))
        
        return original_method
    
    def _restore_decision_maker(self, decision_maker: ThreeWayDecisionMaker, original_method):
        """Restore the original method to the decision maker."""
        decision_maker._calculate_decision_metrics = original_method
    
    def _run_benchmark_item(self, item: Dict, weights: Dict[str, float], 
                           alpha: float, beta: float, tau: float) -> Dict:
        """
        Run a single benchmark item with the given weights and thresholds.
        
        Args:
            item: Benchmark item
            weights: Dictionary of metric weights
            alpha: Lower threshold for positive region
            beta: Upper threshold for negative region
            tau: Relevance threshold
            
        Returns:
            Dictionary of results
        """
        # Extract prompt
        prompt = item.get("prompt", "")
        if not prompt:
            logger.error(f"No prompt found in benchmark item")
            return None
        
        # Create UncertaintyResolver
        resolver = UncertaintyResolver(
            provider=self.config.get("default_provider", "gemini"),
            model=self.config.get("default_model", ""),
            relevance_threshold=tau,
            validity_threshold=0.6
        )
        
        # Create DecisionMaker with custom thresholds
        decision_maker = ThreeWayDecisionMaker(alpha=alpha, beta=beta)
        
        # Apply custom weights
        original_method = self._apply_weights_to_decision_maker(decision_maker, weights)
        
        try:
            # Generate reasoning steps
            reasoning_result = resolver.generate_reasoning(prompt)
            reasoning_steps = reasoning_result.get("steps", [])
            
            if not reasoning_steps:
                logger.error(f"No reasoning steps generated for prompt: {prompt[:50]}...")
                return None
            
            # Process reasoning using ThreeWayCOT
            three_way_cot = ThreeWayCOT(decision_maker=decision_maker)
            results = three_way_cot.process_reasoning(reasoning_steps)
            
            # Extract decisions
            decisions = results.get("decisions", [])
            
            # Count decision types
            decision_counts = {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0}
            for decision in decisions:
                decision_type = decision.get("decision", "ABSTAIN")
                decision_counts[decision_type] += 1
            
            # Calculate percentages
            total_steps = len(decisions)
            decision_percentages = {
                k: (v / total_steps) * 100 if total_steps > 0 else 0
                for k, v in decision_counts.items()
            }
            
            return {
                "prompt": prompt,
                "total_steps": total_steps,
                "decisions": decisions,
                "decision_counts": decision_counts,
                "decision_percentages": decision_percentages
            }
        
        finally:
            # Restore original method
            self._restore_decision_maker(decision_maker, original_method)
    
    def generate_weight_variations(self, metric: str, values: List[float]) -> List[Dict[str, float]]:
        """
        Generate weight variations for a specific metric.
        
        Args:
            metric: Metric to vary
            values: List of weight values to test
            
        Returns:
            List of weight dictionaries
        """
        weight_variations = []
        
        for value in values:
            # Start with default weights
            weights = dict(self.default_weights)
            
            # Calculate weight difference
            diff = weights[metric] - value
            
            # Distribute difference among other metrics
            other_metrics = [k for k in weights.keys() if k != metric]
            adjustment = diff / len(other_metrics)
            
            # Set new weight for target metric
            weights[metric] = value
            
            # Adjust other metrics
            for other_metric in other_metrics:
                weights[other_metric] += adjustment
                
                # Ensure no negative weights
                weights[other_metric] = max(0.01, weights[other_metric])
            
            # Normalize to ensure sum is 1.0
            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    weights[k] /= total
            
            weight_variations.append(weights)
        
        return weight_variations
    
    def run_metric_experiment(self, metric: str, values: List[float], 
                             thresholds: List[Dict[str, float]]) -> Dict:
        """
        Run an experiment by varying weights for a specific metric.
        
        Args:
            metric: Metric to vary
            values: List of weight values to test
            thresholds: List of threshold configurations
            
        Returns:
            Dictionary of results
        """
        weight_variations = self.generate_weight_variations(metric, values)
        
        results = {}
        
        logger.info(f"Running experiment for metric '{metric}' with {len(weight_variations)} variations "
                   f"and {len(thresholds)} threshold configurations")
        
        # For each benchmark item
        for i, item in enumerate(self.benchmark_data):
            item_key = f"item_{i+1}"
            item_results = {}
            
            # For each weight variation
            for j, weights in enumerate(weight_variations):
                weight_key = f"weight_{j+1}_{metric}_{values[j]:.2f}"
                threshold_results = {}
                
                # For each threshold configuration
                for k, thresh in enumerate(thresholds):
                    alpha = thresh["alpha"]
                    beta = thresh["beta"]
                    tau = thresh.get("tau", 0.5)
                    
                    # Run benchmark
                    result = self._run_benchmark_item(item, weights, alpha, beta, tau)
                    
                    if result:
                        threshold_key = f"threshold_{k+1}_alpha_{alpha:.2f}_beta_{beta:.2f}"
                        threshold_results[threshold_key] = result
                
                item_results[weight_key] = {
                    "weights": weights,
                    "threshold_results": threshold_results
                }
            
            results[item_key] = item_results
        
        return results
    
    def run_comprehensive_experiment(self, metrics_to_vary: List[str] = None) -> Dict:
        """
        Run a comprehensive experiment for multiple metrics.
        
        Args:
            metrics_to_vary: List of metrics to vary (default: all metrics)
            
        Returns:
            Dictionary of results
        """
        if not metrics_to_vary:
            metrics_to_vary = list(self.default_weights.keys())
        
        # Define weight values and thresholds
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [
            {"alpha": 0.6, "beta": 0.4, "tau": 0.5},
            {"alpha": 0.7, "beta": 0.3, "tau": 0.5},
            {"alpha": 0.8, "beta": 0.2, "tau": 0.5}
        ]
        
        all_results = {}
        
        # Run experiment for each metric
        for metric in metrics_to_vary:
            logger.info(f"Running experiment for metric: {metric}")
            metric_results = self.run_metric_experiment(metric, values, thresholds)
            all_results[metric] = metric_results
        
        return all_results
    
    def visualize_results(self, results: Dict) -> List[str]:
        """
        Visualize experiment results.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            List of paths to generated visualization files
        """
        visualization_files = []
        
        # For each metric
        for metric, metric_results in results.items():
            # For each item
            for item_key, item_results in metric_results.items():
                # Extract weights and thresholds
                weights = []
                threshold_configs = set()
                
                # First pass to collect all weights and thresholds
                for weight_key, weight_results in item_results.items():
                    # Extract weight value from key
                    parts = weight_key.split('_')
                    if len(parts) >= 4:
                        weight_value = float(parts[3])
                        weights.append(weight_value)
                    
                    # Collect threshold configurations
                    for thresh_key in weight_results.get("threshold_results", {}).keys():
                        threshold_configs.add(thresh_key)
                
                # Sort weights
                weights = sorted(weights)
                
                # For each threshold configuration
                for thresh_key in sorted(threshold_configs):
                    # Extract alpha and beta from threshold key
                    thresh_parts = thresh_key.split('_')
                    alpha = float(thresh_parts[3])
                    beta = float(thresh_parts[5])
                    
                    # Prepare data for visualization
                    weight_values = []
                    accept_counts = []
                    reject_counts = []
                    abstain_counts = []
                    accept_pcts = []
                    reject_pcts = []
                    abstain_pcts = []
                    
                    # For each weight
                    for weight in weights:
                        # Find matching weight key
                        matching_key = None
                        for key in item_results.keys():
                            if f"{metric}_{weight:.2f}" in key:
                                matching_key = key
                                break
                        
                        if matching_key and thresh_key in item_results[matching_key]["threshold_results"]:
                            # Extract decision counts
                            result = item_results[matching_key]["threshold_results"][thresh_key]
                            counts = result.get("decision_counts", {})
                            
                            # Add to lists
                            weight_values.append(weight)
                            accept_counts.append(counts.get("ACCEPT", 0))
                            reject_counts.append(counts.get("REJECT", 0))
                            abstain_counts.append(counts.get("ABSTAIN", 0))
                            
                            # Calculate percentages
                            pcts = result.get("decision_percentages", {})
                            accept_pcts.append(pcts.get("ACCEPT", 0))
                            reject_pcts.append(pcts.get("REJECT", 0))
                            abstain_pcts.append(pcts.get("ABSTAIN", 0))
                    
                    # Skip if no data
                    if not weight_values:
                        continue
                    
                    # Create count visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    width = 0.25
                    
                    # Prepare x-axis
                    x = np.arange(len(weight_values))
                    
                    # Create bars
                    ax.bar(x - width, accept_counts, width, label='ACCEPT', color='green')
                    ax.bar(x, reject_counts, width, label='REJECT', color='red')
                    ax.bar(x + width, abstain_counts, width, label='ABSTAIN', color='blue')
                    
                    # Set labels and title
                    ax.set_xlabel(f'{metric} Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Decision Counts by {metric} Weight (α={alpha:.2f}, β={beta:.2f})')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{w:.2f}' for w in weight_values])
                    ax.legend()
                    
                    # Save figure
                    count_filename = f"decision_counts_{item_key}_{metric}_alpha{alpha:.2f}_beta{beta:.2f}.png"
                    count_filepath = os.path.join(self.output_dir, count_filename)
                    plt.savefig(count_filepath)
                    plt.close(fig)
                    
                    visualization_files.append(count_filepath)
                    
                    # Create percentage visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create bars
                    ax.bar(x - width, accept_pcts, width, label='ACCEPT', color='green')
                    ax.bar(x, reject_pcts, width, label='REJECT', color='red')
                    ax.bar(x + width, abstain_pcts, width, label='ABSTAIN', color='blue')
                    
                    # Set labels and title
                    ax.set_xlabel(f'{metric} Weight')
                    ax.set_ylabel('Percentage')
                    ax.set_title(f'Decision Percentages by {metric} Weight (α={alpha:.2f}, β={beta:.2f})')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{w:.2f}' for w in weight_values])
                    ax.set_ylim(0, 100)
                    ax.legend()
                    
                    # Save figure
                    pct_filename = f"decision_percentages_{item_key}_{metric}_alpha{alpha:.2f}_beta{beta:.2f}.png"
                    pct_filepath = os.path.join(self.output_dir, pct_filename)
                    plt.savefig(pct_filepath)
                    plt.close(fig)
                    
                    visualization_files.append(pct_filepath)
                    
                    logger.info(f"Created visualizations for {item_key}, {metric}, α={alpha:.2f}, β={beta:.2f}")
        
        return visualization_files
    
    def save_results(self, results: Dict) -> str:
        """
        Save experiment results to a file.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            Path to the saved file
        """
        output_file = os.path.join(self.output_dir, "benchmark_metrics_results.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
        return output_file
    
    def run_and_visualize(self, metrics_to_vary: List[str] = None) -> Tuple[str, List[str]]:
        """
        Run the experiment and visualize results.
        
        Args:
            metrics_to_vary: List of metrics to vary
            
        Returns:
            Tuple of (results_file_path, visualization_file_paths)
        """
        # Run experiment
        results = self.run_comprehensive_experiment(metrics_to_vary)
        
        # Save results
        results_file = self.save_results(results)
        
        # Visualize results
        visualization_files = self.visualize_results(results)
        
        return results_file, visualization_files

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run benchmark metrics experiment")
    parser.add_argument("--benchmark", default="benchmark_items.json", 
                        help="Benchmark file path")
    parser.add_argument("--output", default="benchmark_metrics_results", 
                        help="Output directory")
    parser.add_argument("--metrics", nargs="+", 
                        help="Metrics to vary (default: all metrics)")
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MetricWeightExperiment(args.benchmark, args.output)
    
    # Run experiment
    results_file, visualization_files = experiment.run_and_visualize(args.metrics)
    
    logger.info(f"Experiment completed successfully")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Created {len(visualization_files)} visualizations")

if __name__ == "__main__":
    main()
