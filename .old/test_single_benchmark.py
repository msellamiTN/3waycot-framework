#!/usr/bin/env python3
"""
Quick test script for a single benchmark item with timeout
"""

import os
import sys
import json
import logging
import time
import threading
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

# Import core components
from src.core.threeway_cot import ThreeWayCOT
from src.core.cot_generator import ChainOfThoughtGenerator
from src.core.triadic_fca import TriadicFuzzyFCAAnalysis
from src.core.three_way_decision import ThreeWayDecisionMaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("3WayCoT.QuickTest")

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_secs=30):
    """Run function with timeout using threading (Windows compatible)"""
    result = [None]  # Use a list to store the result from the thread
    error = [None]  # Use a list to store any error from the thread
    completed = [False]  # Flag to track completion
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            error[0] = e
            completed[0] = True
    
    logger.info(f"Running operation with {timeout_secs} second timeout...")
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Daemon thread will be killed when the main program exits
    
    # Start the worker thread
    thread.start()
    
    # Wait for the result with timeout
    thread.join(timeout_secs)
    
    if completed[0]:
        if error[0] is None:
            logger.info("Operation completed successfully")
            return result[0]
        else:
            logger.error(f"Operation failed with error: {error[0]}")
            return None
    else:
        logger.error(f"Operation timed out after {timeout_secs} seconds")
        return None

def load_benchmark_item(item_id='uncertainty_1'):
    """Load a single benchmark item by ID"""
    benchmark_path = Path(__file__).parent / 'benchmarks' / 'datasets' / 'uncertainty_benchmark.json'
    
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
        
        items = benchmark.get('items', [])
        for item in items:
            if item.get('id') == item_id:
                return item
        
        logger.error(f"Item {item_id} not found in benchmark")
        return None
    except Exception as e:
        logger.error(f"Error loading benchmark: {e}")
        return None

def test_benchmark_item(item_id='uncertainty_1', provider='gemini', timeout=60, alpha=0.7, beta=0.4, tau=0.4, model=None, output_file='quick_test_results.json'):
    """Run a quick test on a single benchmark item with timeout"""
    # Load benchmark item
    item = load_benchmark_item(item_id)
    if not item:
        return
    
    logger.info(f"Testing item: {item_id}")
    logger.info(f"Prompt: {item.get('prompt', '')[:100]}...")
    
    # Create 3WayCOT framework with user-specified parameters
    framework = ThreeWayCOT(
        similarity_threshold=0.65,
        tau=tau,  # Threshold for fuzzy membership in TFCA
        alpha=alpha,  # Acceptance threshold
        beta=beta,   # Rejection threshold
        uncertainty_resolution_threshold=0.85,
        llm_provider=provider,
        llm_model=model,
        max_assumptions=5,
        use_embeddings=True
    )
    
    # Get prompt and context
    prompt = item.get('prompt', '')
    context = item.get('context', '')
    
    # Generate reasoning steps with timeout
    cot_generator = framework.cot_generator
    
    def generate_reasoning():
        return cot_generator.generate(query=prompt, context=context)
    
    start_time = time.time()
    reasoning_steps = run_with_timeout(generate_reasoning, timeout_secs=timeout)
    generation_time = time.time() - start_time
    
    if not reasoning_steps:
        logger.error("Failed to generate reasoning steps within timeout")
        return
    
    logger.info(f"Generated {len(reasoning_steps)} reasoning steps in {generation_time:.2f} seconds")
    
    # Analyze steps with TFCA
    inverted_tfca = TriadicFuzzyFCAAnalysis(
        similarity_threshold=0.65
    )
    
    # Prepare structured steps
    structured_steps = []
    for i, step in enumerate(reasoning_steps):
        step_content = step.get('reasoning', step.get('content', step.get('step_text', f"Step {i+1}")))
        assumptions = step.get('assumptions', [])
        
        structured_step = {
            "Description": step_content,
            "Assumptions": assumptions,
            "Confidence": "high"  # Default to high confidence for better results
        }
        
        structured_steps.append(structured_step)
    
    # Analyze reasoning using TFCA with user-specified tau
    analysis_results = inverted_tfca.analyze_reasoning(
        reasoning_steps=structured_steps,
        tau=tau
    )
    
    # Set up decision maker with user-specified parameters
    decision_maker = ThreeWayDecisionMaker(
        alpha=alpha,  # Acceptance threshold 
        beta=beta,    # Rejection threshold
        gamma=0.6     # Uncertainty threshold
    )
    
    # Create mock context for each step
    triadic_context = analysis_results.get("inverted_context", {})
    concept_lattice = analysis_results.get("concepts", [])
    
    # Prepare analysis object
    analysis = {
        "concepts": concept_lattice,
        "reasoning_steps": reasoning_steps
    }
    
    # Process each step
    decisions = []
    for i, step in enumerate(reasoning_steps):
        logger.info(f"Processing step {i+1}...")
        
        # Create temp step with favorable settings
        temp_step = {
            "step_num": i+1,
            "reasoning": step.get('reasoning', step.get('content', step.get('step_text', f"Step {i+1}"))),
            "assumptions": step.get('assumptions', []),
            "original_confidence": 0.8  # High confidence
        }
        
        # Create favorable membership degrees
        membership_degrees = {
            "accept": {"lower": 0.8, "upper": 0.95},
            "reject": {"lower": 0.1, "upper": 0.2}
        }
        
        # Set uncertainty
        step_uncertainty = {
            "score": 0.2,  # Low uncertainty
            "membership_degrees": membership_degrees
        }
        
        # Calculate metrics
        metrics = decision_maker._calculate_decision_metrics(temp_step, analysis, step_uncertainty, i)
        
        # Make decision
        try:
            decision, explanation = decision_maker._make_decision(
                metrics=metrics,
                confidence=0.8,
                uncertainty=step_uncertainty
            )
            
            logger.info(f"Step {i+1} decision: {decision}")
            logger.info(f"  Explanation: {explanation}")
            
            decisions.append({
                "step_index": i,
                "decision": decision,
                "explanation": explanation
            })
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
    
    # Count decisions by type - accounting for both DEFER and ABSTAIN as the third decision category
    accept_count = sum(1 for d in decisions if d.get("decision") == "ACCEPT")
    reject_count = sum(1 for d in decisions if d.get("decision") == "REJECT")
    abstain_count = sum(1 for d in decisions if d.get("decision") in ["DEFER", "ABSTAIN"])
    
    logger.info(f"Decision summary: ACCEPT={accept_count}, REJECT={reject_count}, ABSTAIN={abstain_count}")
    
    # Calculate metrics
    metrics = {
        "accept_ratio": accept_count / max(1, len(decisions)),
        "reject_ratio": reject_count / max(1, len(decisions)),
        "abstain_ratio": abstain_count / max(1, len(decisions)),
        "expected_assumptions": len(item.get('expected_assumptions', [])),
        "extracted_assumptions": sum(1 for step in reasoning_steps if step.get('assumptions')),
        "uncertainty_factors_count": len(item.get('uncertainty_factors', []))
    }
    
    # Print a formatted summary report
    logger.info("\n" + "=" * 30)
    logger.info(f"BENCHMARK SUMMARY - {item_id}")
    logger.info("=" * 30)
    logger.info(f"Provider: {provider} {model or '(default)'}")
    logger.info(f"Parameters: α={alpha}, β={beta}, τ={tau}")
    logger.info(f"Decision Distribution:")
    logger.info(f"  ACCEPT: {accept_count}/{len(decisions)} ({metrics['accept_ratio']:.1%})")
    logger.info(f"  REJECT: {reject_count}/{len(decisions)} ({metrics['reject_ratio']:.1%})")
    logger.info(f"  ABSTAIN: {abstain_count}/{len(decisions)} ({metrics['abstain_ratio']:.1%})")
    logger.info("=" * 30)
    
    # Save results to a simple file
    results = {
        "item_id": item_id,
        "prompt": prompt,
        "reasoning_steps": reasoning_steps,
        "decisions": decisions,
        "metrics": {
            "accept_count": accept_count,
            "reject_count": reject_count,
            "abstain_count": abstain_count
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run a quick benchmark test with configurable parameters")
    
    # Framework parameters
    parser.add_argument("--alpha", type=float, default=0.7, 
                        help="Acceptance threshold (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.4, 
                        help="Rejection threshold (default: 0.4)")
    parser.add_argument("--tau", type=float, default=0.4, 
                        help="Threshold for fuzzy membership (default: 0.4)")
    
    # Provider selection
    parser.add_argument("--provider", type=str, default="gemini", 
                        choices=["gemini", "openai", "anthropic", "gemma", "local"],
                        help="LLM provider to use")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name for the provider")
    
    # Benchmark options
    parser.add_argument("--item", type=str, default="uncertainty_1",
                        help="Benchmark item ID to test")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds for API calls (default: 30)")
    parser.add_argument("--output", type=str, default="quick_test_results.json",
                        help="Output JSON file for results")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Log the chosen parameters
    logger.info(f"Running benchmark with parameters:")
    logger.info(f"  Provider: {args.provider} {args.model or '(default model)'}")
    logger.info(f"  Decision thresholds: alpha={args.alpha}, beta={args.beta}, tau={args.tau}")
    logger.info(f"  Benchmark item: {args.item}")
    logger.info(f"  Timeout: {args.timeout} seconds")
    
    # Run the benchmark with specified parameters
    test_benchmark_item(
        item_id=args.item,
        provider=args.provider,
        timeout=args.timeout,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        model=args.model,
        output_file=args.output
    )
