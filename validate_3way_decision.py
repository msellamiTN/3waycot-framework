#!/usr/bin/env python3
"""
Three-Way Decision Validation Framework

This script provides a comprehensive framework for validating the Three-Way Decision 
mechanism using controlled test cases with known expected outcomes. It evaluates 
multiple parameter combinations to determine optimal settings for confidence weighting,
decision thresholds, and membership degrees.

The validation framework:
1. Tests various alpha/beta threshold values across different confidence weights
2. Evaluates membership degree calculations in the three-way decision process
3. Generates visualizations showing parameter impacts on decision accuracy
4. Identifies optimal parameter combinations for maximum decision accuracy

Example usage:
    # Basic validation with default parameters
    python validate_3way_decision.py
    
    # Test with specific confidence weight
    python validate_3way_decision.py --confidence-weight 0.8
    
    # Test with custom threshold values
    python validate_3way_decision.py --alpha-values 0.65,0.7,0.75 --beta-values 0.35,0.4,0.45
    
    # Comprehensive parameter sweep
    python validate_3way_decision.py --confidence-weight 0.6,0.7,0.8 --alpha-values 0.6,0.7,0.8 \
        --beta-values 0.3,0.4,0.5 --tau-values 0.4,0.5,0.6

Author: Triadic Fuzzy Analysis Research Team
Version: 1.0.0
Date: May 2025
"""

import os
import sys
import json
import logging
import argparse
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import necessary components
from src.core.three_way_decision import ThreeWayDecisionMaker
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("3WayCoT.Validation")

def parse_args():
    """Parse command line arguments for the validation framework
    
    Returns:
        argparse.Namespace: Parsed command line arguments with validation parameters
    """
    parser = argparse.ArgumentParser(
        description="Validate Three-Way Decision Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python validate_3way_decision.py --confidence-weight 0.8\n"
            "  python validate_3way_decision.py --alpha-values 0.6,0.7,0.8 --beta-values 0.3,0.4,0.5\n"
        )
    )
    
    # Dataset options
    data_group = parser.add_argument_group('Dataset Options')
    data_group.add_argument(
        "--dataset", 
        type=str, 
        default="benchmarks/datasets/triadic_validation_benchmark.json",
        help="Path to validation dataset with expected decisions"
    )
    data_group.add_argument(
        "--output", 
        type=str, 
        default="validation_results",
        help="Output directory for results and visualizations"
    )
    
    # Parameter sweep options
    param_group = parser.add_argument_group('Parameter Sweep Options')
    param_group.add_argument(
        "--confidence-weight", 
        type=str, 
        default="0.6",
        help="Comma-separated list of confidence weights to test (e.g., 0.6,0.7,0.8)"
    )
    param_group.add_argument(
        "--alpha-values", 
        type=str, 
        default="0.6,0.65,0.7,0.75,0.8",
        help="Comma-separated list of alpha threshold values to test (acceptance threshold)"
    )
    param_group.add_argument(
        "--beta-values", 
        type=str, 
        default="0.3,0.35,0.4,0.45,0.5",
        help="Comma-separated list of beta threshold values to test (rejection threshold)"
    )
    param_group.add_argument(
        "--tau-values", 
        type=str, 
        default="0.4,0.5,0.6",
        help="Comma-separated list of tau threshold values to test (fuzzy membership threshold)"
    )
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        "--disable-visualizations",
        action="store_true",
        help="Disable generation of visualization files (faster execution)"
    )
    viz_group.add_argument(
        "--plot-format", 
        type=str, 
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for visualization plots"
    )
    
    return parser.parse_args()

def load_validation_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load validation dataset from file
    
    This function loads the validation dataset containing test cases with expected
    decisions for validating the Three-Way Decision framework.
    
    Args:
        dataset_path (str): Path to the validation dataset JSON file
        
    Returns:
        List[Dict[str, Any]]: List of validation items with expected decisions
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        ValueError: If the file has invalid JSON format
    """
    # Check if file exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    # Load and parse the dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded validation dataset: {data.get('name')} with {len(data.get('items', []))} items")
        return data.get('items', [])
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []

def validate_decision(
    item: Dict[str, Any], 
    alpha: float, 
    beta: float, 
    tau: float, 
    confidence_weight: float
) -> Dict[str, Any]:
    """
    Validate a single benchmark item with the given parameters
    
    This function is the core decision validation unit that evaluates a single test case
    using the specified parameter set. It constructs a ThreeWayDecisionMaker instance with
    the provided parameters and tests if it produces the expected decision for the item.
    
    Args:
        item (Dict[str, Any]): Validation item with expected decision
        alpha (float): Alpha threshold value (acceptance threshold)
        beta (float): Beta threshold value (rejection threshold)
        tau (float): Tau fuzzy membership threshold value
        confidence_weight (float): Weight to assign to confidence in decision making
        
    Returns:
        Dict[str, Any]: Decision result with details about the decision process
        
    Raises:
        ValueError: If required item fields are missing
    """
    # Validate the item has required fields
    required_fields = ['id', 'expected_decision', 'confidence_level', 'membership_degrees']
    for field in required_fields:
        if field not in item:
            raise ValueError(f"Validation item is missing required field: {field}")
    # Set up ThreeWayDecisionMaker with specific parameters
    decision_maker = ThreeWayDecisionMaker(
        alpha=alpha,
        beta=beta, 
        gamma=0.6  # Default threshold for boundary region width
    )
    
    # Original method to store
    original_method = decision_maker._calculate_decision_metrics
    
    # Create a wrapper method that applies the confidence weight
    def weighted_calculate_metrics(step, analysis, uncertainty, step_index):
        # Call the original method to get the base metrics
        metrics = original_method(step, analysis, uncertainty, step_index)
        
        # Get confidence value
        confidence = item.get('confidence_level', 0.5)
        
        # Adjust the metrics based on confidence weight
        accept_degree = metrics.get('accept_degree', 0.5) * (1 - confidence_weight) + confidence * confidence_weight
        reject_degree = metrics.get('reject_degree', 0.5) * (1 - confidence_weight) + (1 - confidence) * confidence_weight
        
        # Normalize to ensure they sum to 1.0
        total = accept_degree + reject_degree
        if total > 0:
            accept_degree = accept_degree / total
            reject_degree = reject_degree / total
            
        # Update metrics
        metrics['accept_degree'] = accept_degree
        metrics['reject_degree'] = reject_degree
        metrics['abstain_degree'] = 1 - max(accept_degree, reject_degree)
        
        return metrics
    
    # Apply the modified method
    decision_maker._calculate_decision_metrics = weighted_calculate_metrics
    
    # Create a dummy analysis result
    analysis = {
        "concepts": [],
        "reasoning_steps": []
    }
    
    # Get the membership degrees from the benchmark item
    membership_degrees = item.get('membership_degrees', {})
    accept_degree = (membership_degrees.get('accept', {}).get('lower', 0.5) + 
                    membership_degrees.get('accept', {}).get('upper', 0.5)) / 2
    reject_degree = (membership_degrees.get('reject', {}).get('lower', 0.5) + 
                    membership_degrees.get('reject', {}).get('upper', 0.5)) / 2
    
    # Create an uncertainty object
    uncertainty = {
        "score": 0.2,  # Default uncertainty score
        "membership_degrees": membership_degrees
    }
    
    # Create a step with the item's characteristics
    step = {
        "step_num": 1,
        "reasoning": item.get('prompt', ''),
        "assumptions": [],
        "original_confidence": item.get('confidence_level', 0.5)
    }
    
    # Make the decision
    metrics = decision_maker._calculate_decision_metrics(step, analysis, uncertainty, 0)
    
    # Ensure metrics has the expected format
    if 'uncertainty_score' not in metrics:
        metrics['uncertainty_score'] = 0.5
    
    # Get confidence value for the decision
    confidence = item.get('confidence_level', 0.5)
    
    # Call _make_decision with the correct parameters
    decision_result = decision_maker._make_decision(metrics, confidence, uncertainty)
    decision = decision_result[0]  # Extract the decision from the tuple
    
    # Create a result object
    result = {
        "item_id": item.get('id'),
        "prompt": item.get('prompt'),
        "expected_decision": item.get('expected_decision'),
        "actual_decision": decision,
        "is_correct": decision == item.get('expected_decision'),
        "confidence_level": item.get('confidence_level'),
        "membership_degrees": membership_degrees,
        "metrics": metrics,
        "parameters": {
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "confidence_weight": confidence_weight
        }
    }
    
    # Restore the original method
    decision_maker._calculate_decision_metrics = original_method
    
    return result

def run_validation(
    dataset: List[Dict[str, Any]],
    alphas: List[float],
    betas: List[float],
    taus: List[float],
    confidence_weights: List[float]
) -> Dict[str, Any]:
    """
    Run validation across multiple parameter combinations
    
    This function is the core of the validation framework, testing each parameter
    combination against the validation dataset and recording accuracy metrics.
    It systematically evaluates how different threshold values and confidence
    weights affect the three-way decision making process.
    
    Args:
        dataset (List[Dict[str, Any]]): List of validation items with expected decisions
        alphas (List[float]): List of alpha threshold values to test
        betas (List[float]): List of beta threshold values to test
        taus (List[float]): List of tau fuzzy membership threshold values to test
        confidence_weights (List[float]): List of confidence weight values to test
    
    Returns:
        Dict[str, Any]: Dictionary containing validation results, parameter scores,
                        and the best parameter combination found
    
    Raises:
        ValueError: If invalid parameter combinations are detected
    """
    # Validate parameters
    if not dataset:
        raise ValueError("Dataset is empty. Cannot run validation.")
        
    if not all(0 <= a <= 1 for a in alphas):
        raise ValueError("Alpha values must be between 0 and 1")
        
    if not all(0 <= b <= 1 for b in betas):
        raise ValueError("Beta values must be between 0 and 1")
        
    if not all(0 <= t <= 1 for t in taus):
        raise ValueError("Tau values must be between 0 and 1")
        
    if not all(0 <= c <= 1 for c in confidence_weights):
        raise ValueError("Confidence weights must be between 0 and 1")
    
    # Initialize result storage
    results = []
    param_scores = {}
    
    total_combinations = len(alphas) * len(betas) * len(taus) * len(confidence_weights)
    logger.info(f"Testing {total_combinations} parameter combinations on {len(dataset)} benchmark items")
    
    # Track best combination
    best_combination = None
    best_accuracy = 0.0
    
    # Process each parameter combination
    for alpha in alphas:
        for beta in betas:
            for tau in taus:
                for confidence_weight in confidence_weights:
                    # Skip invalid combinations where alpha < beta
                    if alpha <= beta:
                        continue
                    
                    key = f"a{alpha:.2f}_b{beta:.2f}_t{tau:.2f}_c{confidence_weight:.2f}"
                    item_results = []
                    correct_count = 0
                    
                    # Test each benchmark item
                    for item in dataset:
                        result = validate_decision(
                            item, 
                            alpha=alpha, 
                            beta=beta, 
                            tau=tau, 
                            confidence_weight=confidence_weight
                        )
                        item_results.append(result)
                        if result.get('is_correct'):
                            correct_count += 1
                    
                    # Calculate accuracy for this combination
                    accuracy = correct_count / len(dataset) if dataset else 0
                    
                    # Track the best combination
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_combination = {
                            "alpha": alpha,
                            "beta": beta,
                            "tau": tau,
                            "confidence_weight": confidence_weight,
                            "accuracy": accuracy
                        }
                    
                    # Store summary for this parameter combination
                    param_scores[key] = {
                        "alpha": alpha,
                        "beta": beta,
                        "tau": tau,
                        "confidence_weight": confidence_weight,
                        "accuracy": accuracy,
                        "correct_count": correct_count,
                        "total_items": len(dataset)
                    }
                    
                    # Store detailed results
                    results.append({
                        "params": {
                            "alpha": alpha,
                            "beta": beta,
                            "tau": tau,
                            "confidence_weight": confidence_weight
                        },
                        "accuracy": accuracy,
                        "correct_count": correct_count,
                        "total_items": len(dataset),
                        "item_results": item_results
                    })
                    
                    logger.info(f"Combination {key}: Accuracy {accuracy:.2f} ({correct_count}/{len(dataset)})")
    
    logger.info(f"Best combination: alpha={best_combination['alpha']:.2f}, "
                f"beta={best_combination['beta']:.2f}, "
                f"tau={best_combination['tau']:.2f}, "
                f"confidence_weight={best_combination['confidence_weight']:.2f} "
                f"with accuracy {best_combination['accuracy']:.2f}")
    
    return {
        "results": results,
        "param_scores": param_scores,
        "best_combination": best_combination
    }

def generate_visualizations(results: Dict[str, Any], output_dir: str, plot_format: str = 'png') -> List[str]:
    """
    Generate visualizations from validation results
    
    This function creates multiple types of visualizations to help understand the impact
    of different parameter combinations on decision accuracy:
    
    1. Line charts showing how accuracy varies with confidence weight for each alpha/beta pair
    2. Heatmaps showing how accuracy varies across alpha and beta values
    3. Summary CSV files for further analysis
    
    Args:
        results (Dict[str, Any]): The validation results dictionary
        output_dir (str): Directory where visualization files will be saved
        plot_format (str, optional): File format for visualization outputs. Defaults to 'png'.
    
    Returns:
        List[str]: Paths to generated visualization files
    
    Raises:
        ValueError: If results dictionary has unexpected format
        OSError: If there's an issue with file operations
    """
    # Validate inputs
    if not isinstance(results, dict) or 'param_scores' not in results:
        raise ValueError("Results dictionary has invalid format. Expected 'param_scores' key.")
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory: {e}") from e
    # Set up visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    visualization_files = []
    
    # Extract data for visualization
    param_scores = results.get('param_scores', {})
    best_combination = results.get('best_combination', {})
    
    # Group by alpha/beta
    alpha_beta_data = {}
    for key, score in param_scores.items():
        alpha = score['alpha']
        beta = score['beta']
        conf_weight = score['confidence_weight']
        tau = score['tau']
        
        alpha_beta_key = f"a{alpha:.2f}_b{beta:.2f}"
        if alpha_beta_key not in alpha_beta_data:
            alpha_beta_data[alpha_beta_key] = []
        
        alpha_beta_data[alpha_beta_key].append({
            'confidence_weight': conf_weight,
            'tau': tau,
            'accuracy': score['accuracy']
        })
    
    # Create accuracy vs confidence weight plots for each alpha/beta pair
    for alpha_beta_key, points in alpha_beta_data.items():
        # Group by tau
        tau_groups = {}
        for point in points:
            tau = point['tau']
            if tau not in tau_groups:
                tau_groups[tau] = []
            tau_groups[tau].append(point)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for tau, tau_points in tau_groups.items():
            # Sort by confidence weight
            tau_points.sort(key=lambda x: x['confidence_weight'])
            
            # Extract data
            conf_weights = [p['confidence_weight'] for p in tau_points]
            accuracies = [p['accuracy'] for p in tau_points]
            
            # Plot line
            plt.plot(conf_weights, accuracies, marker='o', label=f'tau={tau:.2f}')
        
        parts = alpha_beta_key.replace('a', '').split('_b')  # Split by '_b' to get alpha and beta parts
        alpha = parts[0]
        beta = parts[1] if len(parts) > 1 else '0.0'
        plt.title(f'Accuracy vs Confidence Weight (α={alpha}, β={beta})')
        plt.xlabel('Confidence Weight')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        filename = os.path.join(viz_dir, f'accuracy_vs_confidence_{alpha_beta_key}.{plot_format}')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_files.append(filename)
    
    # Create a heat map of alpha vs beta for best confidence weight and tau
    best_conf = best_combination.get('confidence_weight')
    best_tau = best_combination.get('tau')
    
    # Group by alpha/beta for the best confidence weight and tau
    alpha_values = sorted(set(score['alpha'] for score in param_scores.values()))
    beta_values = sorted(set(score['beta'] for score in param_scores.values()))
    
    # Create data for heatmap
    heatmap_data = np.zeros((len(alpha_values), len(beta_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # Skip invalid combinations
            if alpha <= beta:
                heatmap_data[i, j] = np.nan
                continue
                
            # Find scores for this alpha/beta
            scores = [
                score['accuracy'] for key, score in param_scores.items()
                if (abs(score['alpha'] - alpha) < 0.001 and 
                    abs(score['beta'] - beta) < 0.001 and
                    abs(score['confidence_weight'] - best_conf) < 0.001 and
                    abs(score['tau'] - best_tau) < 0.001)
            ]
            
            if scores:
                heatmap_data[i, j] = max(scores)
            else:
                heatmap_data[i, j] = np.nan
    
    # Add additional matplotlib configuration
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.tight_layout()
    plt.colorbar(label='Accuracy')
    
    # Set labels
    plt.title(f'Accuracy by α/β (Confidence Weight={best_conf:.2f}, τ={best_tau:.2f})')
    plt.xlabel('β Values')
    plt.ylabel('α Values')
    
    plt.xticks(range(len(beta_values)), [f'{b:.2f}' for b in beta_values])
    plt.yticks(range(len(alpha_values)), [f'{a:.2f}' for a in alpha_values])
    
    # Add text annotations
    for i in range(len(alpha_values)):
        for j in range(len(beta_values)):
            if not np.isnan(heatmap_data[i, j]):
                plt.text(j, i, f'{heatmap_data[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if heatmap_data[i, j] < 0.7 else 'black')
    
    # Save figure
    filename = os.path.join(viz_dir, f'alpha_beta_heatmap_conf{best_conf:.2f}_tau{best_tau:.2f}.{plot_format}')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(filename)
    
    # Create a summary CSV file
    csv_file = os.path.join(viz_dir, 'validation_results_summary.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Alpha', 'Beta', 'Tau', 'Confidence_Weight', 'Accuracy', 'Correct', 'Total'])
        
        for key, score in param_scores.items():
            writer.writerow([
                score['alpha'],
                score['beta'],
                score['tau'],
                score['confidence_weight'],
                score['accuracy'],
                score['correct_count'],
                score['total_items']
            ])
    
    visualization_files.append(csv_file)
    
    return visualization_files

def main():
    """Main validation function that orchestrates the entire validation process
    
    This function drives the validation framework by:
    1. Parsing command line arguments
    2. Setting up parameter combinations to test
    3. Loading the validation dataset
    4. Running the validation experiments
    5. Generating visualizations
    6. Reporting results
    
    Returns:
        None
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Parse parameter lists
        try:
            alphas = [float(a) for a in args.alpha_values.split(',')]
            betas = [float(b) for b in args.beta_values.split(',')]
            taus = [float(t) for t in args.tau_values.split(',')]
            confidence_weights = [float(c) for c in args.confidence_weight.split(',')]
            
            # Validate the parameter ranges
            for alpha in alphas:
                if not 0 <= alpha <= 1:
                    raise ValueError(f"Alpha value {alpha} outside valid range [0,1]")
            for beta in betas:
                if not 0 <= beta <= 1:
                    raise ValueError(f"Beta value {beta} outside valid range [0,1]")
            for tau in taus:
                if not 0 <= tau <= 1:
                    raise ValueError(f"Tau value {tau} outside valid range [0,1]")
            for conf in confidence_weights:
                if not 0 <= conf <= 1:
                    raise ValueError(f"Confidence weight {conf} outside valid range [0,1]")
        except ValueError as e:
            logger.error(f"Parameter parsing error: {e}")
            print(f"\nERROR: Invalid parameter value. {e}\n")
            return 1
        
        # Set up output directory with timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output, f"validation_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging to file
        log_file = os.path.join(output_dir, "validation.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Log validation parameters
        logger.info(f"Starting validation with parameters:")
        logger.info(f"  Alpha values: {alphas}")
        logger.info(f"  Beta values: {betas}")
        logger.info(f"  Tau values: {taus}")
        logger.info(f"  Confidence weights: {confidence_weights}")
        logger.info(f"  Output directory: {output_dir}")
        
        # Load validation dataset
        dataset = load_validation_dataset(args.dataset)
        if not dataset:
            logger.error("No validation data found. Exiting.")
            print("\nERROR: Failed to load validation dataset. See log for details.\n")
            return 1
        
        # Save validation configuration
        config = {
            "timestamp": timestamp,
            "parameters": {
                "alphas": alphas,
                "betas": betas,
                "taus": taus,
                "confidence_weights": confidence_weights
            },
            "dataset": {
                "path": args.dataset,
                "item_count": len(dataset)
            }
        }
        
        config_file = os.path.join(output_dir, "validation_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Run validation
        start_time = time.time()
        results = run_validation(
            dataset=dataset,
            alphas=alphas,
            betas=betas,
            taus=taus,
            confidence_weights=confidence_weights
        )
        validation_time = time.time() - start_time
        
        # Generate visualizations if not disabled
        visualization_files = []
        if not args.disable_visualizations:
            try:
                visualization_files = generate_visualizations(
                    results, 
                    output_dir,
                    plot_format=args.plot_format
                )
                logger.info(f"Generated {len(visualization_files)} visualization files")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                print(f"\nWARNING: Error generating visualizations. See log for details.\n")
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'detailed_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Add execution metrics to results
        execution_metrics = {
            "validation_time_seconds": validation_time,
            "parameter_combinations_tested": len(results.get('param_scores', {})),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metrics_file = os.path.join(output_dir, 'execution_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(execution_metrics, f, indent=2)
        
        logger.info(f"Validation complete in {validation_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        # Print a summary of the best combination
        best = results.get('best_combination', {})
        print("\n" + "=" * 50)
        print("===== THREE-WAY DECISION VALIDATION SUMMARY =====")
        print("=" * 50)
        print(f"Best parameter combination:")
        print(f"  Alpha (acceptance threshold): {best.get('alpha', 0):.2f}")
        print(f"  Beta (rejection threshold): {best.get('beta', 0):.2f}")
        print(f"  Tau (fuzzy membership threshold): {best.get('tau', 0):.2f}")
        print(f"  Confidence Weight: {best.get('confidence_weight', 0):.2f}")
        print(f"  Accuracy: {best.get('accuracy', 0):.2f}")
        print("\nValidation statistics:")
        print(f"  Parameter combinations tested: {execution_metrics['parameter_combinations_tested']}")
        print(f"  Execution time: {validation_time:.2f} seconds")
        print(f"  Results directory: {output_dir}")
        print("=" * 50)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.\n")
        logger.info("Validation interrupted by user")
        return 130
    
    except Exception as e:
        logger.exception(f"Unexpected error during validation: {e}")
        print(f"\nERROR: Unexpected error occurred. See log for details.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
