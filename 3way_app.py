#!/usr/bin/env python3
"""
3WayCoT Framework - Main Entry Point

A command-line interface for the Three-Way Chain of Thought framework.
Supports interactive mode, file input, and benchmark testing.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("3waycot.log")
    ]
)
logger = logging.getLogger("3WayCoT")

# Import framework components
try:
    from src.core.cot_generator import ChainOfThoughtGenerator
    from src.core.three_way_decision import ThreeWayDecisionMaker
    from src.core.knowledge_base import KnowledgeBase
    from src.core.threeway_cot import ThreeWayCOT
    from src.core.triadic_fca import TriadicFuzzyFCAAnalysis as InvertedTriadicFuzzyAnalysis
    from src.core.uncertainty_resolver import UncertaintyResolver
except ImportError as e:
    logger.error(f"Failed to import framework components: {e}")
    logger.error("Please ensure all dependencies are installed and the src directory is in your PYTHONPATH")
    sys.exit(1)


class ThreeWayCoTApp:
    """
    Main application class for the 3WayCoT framework.
    
    This class serves as the main entry point for the application, handling
    initialization, configuration, and execution of the 3WayCoT pipeline.
    """
    
    def __init__(self, 
                 knowledge_base_path: Optional[str] = None,
                 similarity_threshold: float = 0.65,
                 tau: float = 0.4,
                 alpha: float = 0.7,
                 beta: float = 0.6,
                 max_assumptions: Optional[int] = None,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 use_embeddings: bool = True,
                 verbose: bool = False):
        """
        Initialize the 3WayCoT application.
        
        Args:
            knowledge_base_path: Path to the knowledge base file
            similarity_threshold: Threshold for similarity-based matching
            tau: Threshold for fuzzy membership in TFCA
            alpha: Acceptance threshold for three-way decisions
            beta: Rejection threshold for three-way decisions
            max_assumptions: Maximum number of assumptions per step
            llm_provider: LLM provider (openai, gemini, anthropic, etc.)
            llm_model: Model name for the LLM
            use_embeddings: Whether to use vector embeddings
            verbose: Enable verbose logging
        """
        self.logger = logging.getLogger("3WayCoT.App")
        self.verbose = verbose
        
        # Initialize the core 3WayCoT framework
        self.framework = ThreeWayCOT(
            similarity_threshold=similarity_threshold,
            tau=tau,
            alpha=alpha,
            beta=beta,
            knowledge_base_path=knowledge_base_path,
            use_embeddings=use_embeddings,
            max_assumptions=max_assumptions,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        # Initialize other components
        self.knowledge_base = self.framework.knowledge_base
        self.tfca_analyzer = self.framework.tfca_analyzer
        self.context_constructor = self.framework.context_constructor
        
        self.logger.info("3WayCoT application initialized")
    
    def process_prompt(self, 
                      prompt: str, 
                      context: str = "",
                      max_steps: int = 10) -> Dict[str, Any]:
        """
        Process a single prompt through the 3WayCoT pipeline.
        
        Args:
            prompt: The input prompt to process
            context: Additional context for the prompt
            max_steps: Maximum number of reasoning steps to generate
            
        Returns:
            Dictionary containing the processed results
        """
        self.logger.info(f"Processing prompt: {prompt[:100]}...")
        
        try:
            # Generate initial chain of thought
            reasoning_steps = self.framework.generate_chain_of_thought(
                prompt=prompt,
                context=context,
                max_steps=max_steps
            )
            
            # Analyze the reasoning steps
            analysis_results = self.framework.analyze_reasoning(
                reasoning_steps=reasoning_steps,
                tau=self.framework.tau
            )
            
            # Make three-way decisions
            decision_results = self.framework.make_decisions(
                reasoning_steps=reasoning_steps,
                analysis_results=analysis_results
            )
            
            # Resolve uncertainties if needed
            if any(step.get('decision') == 'ABSTAIN' for step in decision_results):
                self.logger.info("Resolving uncertainties...")
                decision_results = self.framework.resolve_uncertainties(
                    reasoning_steps=decision_results,
                    analysis_results=analysis_results
                )
            
            # Prepare results
            results = {
                'prompt': prompt,
                'context': context,
                'reasoning_steps': reasoning_steps,
                'analysis_results': analysis_results,
                'decisions': decision_results,
                'summary': self._generate_summary(decision_results)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            raise
    
    def process_dataset(self, dataset_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a dataset of prompts through the 3WayCoT pipeline.
        
        Args:
            dataset_path: Path to the dataset file (JSON)
            output_path: Optional path to save results
            
        Returns:
            List of processed results
        """
        self.logger.info(f"Processing dataset: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            results = []
            for item in dataset:
                try:
                    result = self.process_prompt(
                        prompt=item['prompt'],
                        context=item.get('context', ''),
                        max_steps=item.get('max_steps', 10)
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item: {str(e)}")
                    continue
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}", exc_info=True)
            raise
    
    def _generate_summary(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the decisions."""
        accepted = sum(1 for d in decisions if d.get('decision') == 'ACCEPT')
        rejected = sum(1 for d in decisions if d.get('decision') == 'REJECT')
        abstained = sum(1 for d in decisions if d.get('decision') == 'ABSTAIN')
        
        return {
            'total_steps': len(decisions),
            'accepted': accepted,
            'rejected': rejected,
            'abstained': abstained,
            'acceptance_rate': accepted / len(decisions) if decisions else 0,
            'rejection_rate': rejected / len(decisions) if decisions else 0,
            'abstention_rate': abstained / len(decisions) if decisions else 0
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to a file.
        
        Args:
            results: Results to save
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="3WayCoT Framework")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--context", type=str, default="", help="Additional context")
    parser.add_argument("--knowledge-base", type=str, help="Path to knowledge base")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    args = parser.parse_args()
    
    # Initialize the application
    app = ThreeWayCoTApp(
        knowledge_base_path=args.knowledge_base,
        verbose=True
    )
    
    # Process the prompt
    if args.prompt:
        results = app.process_prompt(
            prompt=args.prompt,
            context=args.context
        )
        app.save_results(results, args.output)
    else:
        print("Please provide a prompt using --prompt")