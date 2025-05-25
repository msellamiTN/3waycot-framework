#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WayCoT Framework - Main Entry Point

This script provides the command-line interface for the 3WayCoT framework,
integrating Chain-of-Thought reasoning with Three-Way Decision making
and Triadic Fuzzy Concept Analysis.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import codecs
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml

# Configure sys.stdout to handle unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config_path = Path(__file__).parent / "src" / "config" / "config.yml"
        
        # Try user-specified config path first, then default
        if config_path is None:
            config_path = default_config_path
        else:
            config_path = Path(config_path)
        
        # Initialize with empty config
        config = {}
            
        try:
            # Check if file exists
            if not config_path.exists():
                print(f"Warning: Config file not found at {config_path}. Using default configuration.")
                if config_path != default_config_path and default_config_path.exists():
                    print(f"Falling back to default config at {default_config_path}")
                    config_path = default_config_path
                else:
                    return {}
            
            # Load and parse the YAML file
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read().strip()
                if not config_content:
                    print(f"Warning: Config file at {config_path} is empty. Using default configuration.")
                    return {}
                    
                loaded_config = yaml.safe_load(config_content)
                if loaded_config is None:
                    print(f"Warning: Config file at {config_path} has invalid YAML. Using default configuration.")
                    return {}
                    
                config = loaded_config
                print(f"Successfully loaded configuration from {config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in config file {config_path}: {e}")
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            
        # Add timestamp to config for tracking
        if isinstance(config, dict):
            if 'app' not in config:
                config['app'] = {}
            config['app']['start_time'] = self._get_timestamp()
            
        return config
        
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get('app', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # Create log directory if it doesn't exist
        log_file = log_config.get('log_file', '3waycot.log')
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[]
        )
        
        # Create handlers
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Get the root logger and add handlers
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Set logger for this class
        self.logger = logging.getLogger("3WayCoT")
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class ThreeWayCoTApp:
    """Main application class for the 3WayCoT framework."""
    
    def _calculate_enhanced_metrics(self, reasoning_steps: List[Dict]) -> Dict[str, Any]:
        """
        Calculate enhanced metrics from reasoning steps to improve decision making.
        
        This method extracts and analyzes confidence values from reasoning steps,
        providing additional metrics to enhance the decision-making process.
        
        Args:
            reasoning_steps: List of preprocessed reasoning steps with confidence values
            
        Returns:
            Dictionary of enhanced metrics for decision making
        """
        # Initialize metrics dictionary
        metrics = {
            'confidence_metrics': {},
            'membership_degrees': {},
            'confidence_calibration': {}
        }
        
        # Extract confidence values from steps
        confidence_values = [step.get('confidence', 0.5) for step in reasoning_steps]
        
        if not confidence_values:
            return metrics
        
        # Calculate statistical metrics
        avg_confidence = sum(confidence_values) / len(confidence_values)
        max_confidence = max(confidence_values)
        min_confidence = min(confidence_values)
        confidence_range = max_confidence - min_confidence
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidence_values) / len(confidence_values)
        
        # Calculate distribution metrics
        high_confidence_ratio = sum(1 for c in confidence_values if c >= 0.7) / len(confidence_values)
        low_confidence_ratio = sum(1 for c in confidence_values if c <= 0.3) / len(confidence_values)
        mid_confidence_ratio = 1.0 - high_confidence_ratio - low_confidence_ratio
        
        # Calculate trend metrics (increasing or decreasing confidence)
        confidence_trend = 0.0
        if len(confidence_values) > 1:
            # Positive means increasing confidence, negative means decreasing
            confidence_trend = sum(confidence_values[i] - confidence_values[i-1] for i in range(1, len(confidence_values)))
        
        # Store confidence metrics
        metrics['confidence_metrics'] = {
            'average': avg_confidence,
            'max': max_confidence,
            'min': min_confidence,
            'range': confidence_range,
            'variance': confidence_variance,
            'high_ratio': high_confidence_ratio,
            'low_ratio': low_confidence_ratio,
            'mid_ratio': mid_confidence_ratio,
            'trend': confidence_trend
        }
        
        # Calculate membership degrees based on confidence patterns
        # This helps the decision maker determine the appropriate decision region
        accept_degree = 0.0
        reject_degree = 0.0
        abstain_degree = 0.0
        
        # Consider confidence distribution and patterns
        if high_confidence_ratio > 0.7:
            # Mostly high confidence suggests acceptance
            accept_degree = 0.6 + 0.4 * high_confidence_ratio
            reject_degree = 0.2 * low_confidence_ratio
            abstain_degree = 1.0 - accept_degree - reject_degree
        elif low_confidence_ratio > 0.7:
            # Mostly low confidence suggests rejection
            reject_degree = 0.6 + 0.4 * low_confidence_ratio
            accept_degree = 0.2 * high_confidence_ratio
            abstain_degree = 1.0 - accept_degree - reject_degree
        elif confidence_variance > 0.1:
            # High variance suggests uncertainty
            abstain_degree = 0.5 + 0.5 * min(1.0, confidence_variance * 5)
            accept_degree = 0.5 * high_confidence_ratio
            reject_degree = 0.5 * low_confidence_ratio
            
            # Normalize to sum to 1.0
            total = accept_degree + reject_degree + abstain_degree
            if total > 0:
                accept_degree /= total
                reject_degree /= total
                abstain_degree /= total
        else:
            # Use average confidence to determine membership
            if avg_confidence >= 0.7:
                accept_degree = 0.7 + 0.3 * (avg_confidence - 0.7) / 0.3  # Scale from 0.7 to 1.0
                reject_degree = 0.1
                abstain_degree = 0.2
            elif avg_confidence <= 0.3:
                reject_degree = 0.7 + 0.3 * (0.3 - avg_confidence) / 0.3  # Scale from 0.3 to 0.0
                accept_degree = 0.1
                abstain_degree = 0.2
            else:
                # In the middle range, scale abstain appropriately
                abstain_degree = 0.5 + 0.5 * (1.0 - abs(avg_confidence - 0.5) * 2)
                accept_degree = (avg_confidence - 0.3) / 0.4 * (1.0 - abstain_degree)  # Scale based on position in 0.3-0.7 range
                reject_degree = 1.0 - accept_degree - abstain_degree
        
        # Store membership degrees
        metrics['membership_degrees'] = {
            'accept': accept_degree,
            'reject': reject_degree,
            'abstain': abstain_degree
        }
        
        # Calculate confidence calibration factors
        # These adjust for potential over/under-confidence in the reasoning
        if confidence_trend > 0.3:
            # Increasing confidence pattern - might indicate growing certainty
            metrics['confidence_calibration'] = {
                'accept_factor': 1.2,
                'reject_factor': 0.8,
                'abstain_factor': 0.9
            }
        elif confidence_trend < -0.3:
            # Decreasing confidence pattern - might indicate growing uncertainty
            metrics['confidence_calibration'] = {
                'accept_factor': 0.8,
                'reject_factor': 1.2,
                'abstain_factor': 1.1
            }
        else:
            # Stable confidence pattern
            metrics['confidence_calibration'] = {
                'accept_factor': 1.0,
                'reject_factor': 1.0,
                'abstain_factor': 1.0
            }
        
        return metrics
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path."""
        self.config = ConfigManager(config_path)
        self.logger = logging.getLogger("3WayCoT.App")
        
        # Ensure required directories exist
        self._setup_directories()
        
        # Initialize core components
        self._init_components()
        
        self.logger.info("3WayCoT application initialized")

    def _setup_directories(self):
        """Create required directories if they don't exist."""
        dirs = [
            self.config.get('knowledge_base.path', 'data/'),
            self.config.get('paths.results', 'results/'),
            self.config.get('paths.logs', 'logs/')
        ]
        for d in dirs:
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def _load_llm_config(self):
        """Load the LLM provider configuration from config.json.
        
        This method reads the LLM configuration from config.json to ensure compatibility
        with the metrics experimentation frameworks.
        
        Returns:
            Dictionary with provider and model information
        """
        config_path = Path(__file__).parent / 'config.json'
        default_config = {
            'provider': 'openai',
            'model': 'gpt-4',
            'temperature': 0.7
        }
        
        if not config_path.exists():
            self.logger.warning(f"config.json not found at {config_path}, using default configuration")
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Find which provider is configured (marked as is_configured=True)
            provider = None
            for provider_name, provider_config in config_data.items():
                if provider_config.get('is_configured', False):
                    provider = provider_name
                    model = provider_config.get('model')
                    temperature = provider_config.get('temperature', 0.7)
                    self.logger.info(f"Found configured LLM provider: {provider} with model: {model}")
                    return {
                        'provider': provider,
                        'model': model,
                        'temperature': temperature
                    }
            
            # If no provider is configured, use the first one as default
            if config_data:
                provider = next(iter(config_data.keys()))
                provider_config = config_data[provider]
                return {
                    'provider': provider,
                    'model': provider_config.get('model'),
                    'temperature': provider_config.get('temperature', 0.7)
                }
                
            return default_config
                
        except Exception as e:
            self.logger.error(f"Error loading LLM config from {config_path}: {e}")
            return default_config
    
    def _init_components(self):
        """Initialize framework components."""
        from src.core.threeway_cot import ThreeWayCOT
        from src.core.knowledge_base import KnowledgeBase
        from src.core.confidence_extractor import ConfidenceExtractor
        
        # Initialize knowledge base
        kb_path = self.config.get('knowledge_base.path')
        # Store both the path and the instance for different components
        self.kb_path = kb_path  # Store the path for components that need it directly
        try:
            self.knowledge_base = KnowledgeBase(kb_path) if kb_path else None
        except Exception as e:
            self.logger.warning(f"Could not load knowledge base from {kb_path}: {e}")
            self.knowledge_base = None
        
        # Get configuration parameters with defaults
        alpha = self.config.get('framework.alpha', 0.7)
        beta = self.config.get('framework.beta', 0.6)
        tau = self.config.get('framework.tau', 0.4)
        similarity_threshold = self.config.get('framework.similarity_threshold', 0.65)
        max_assumptions = self.config.get('framework.max_assumptions', 5)
        uncertainty_threshold = self.config.get('uncertainty.resolution_threshold', 0.85)
        
        # Get LLM configuration from config.json
        llm_config = self._load_llm_config()
        llm_provider = llm_config.get('provider', 'openai')
        llm_model = llm_config.get('model', 'gpt-4')
        
        # Validate three-way decision thresholds, similar to test_single_benchmark.py
        if alpha + beta <= 1.0:
            self.logger.warning(f"Alpha ({alpha}) + beta ({beta}) should be > 1. Adjusting to defaults.")
            alpha = 0.7  # Default from specifications
            beta = 0.6   # Default from specifications
        
        # Initialize Confidence Extractor for enhanced metrics support
        self.confidence_extractor = ConfidenceExtractor()
            
        # Initialize 3WayCOT framework with proper parameters
        self.framework = ThreeWayCOT(
            alpha=alpha,
            beta=beta,
            tau=tau,
            similarity_threshold=similarity_threshold,
            knowledge_base_path=kb_path,  # Use path, not the object
            max_assumptions=max_assumptions,
            llm_provider=llm_provider,
            llm_model=llm_model,
            uncertainty_resolution_threshold=uncertainty_threshold,
            use_embeddings=self.config.get('framework.use_embeddings', True)
        )
        
        # Log the initialized components
        self.logger.info(f"Initialized 3WayCoT with parameters: α={alpha}, β={beta}, τ={tau}")
        self.logger.info(f"Using LLM provider: {llm_provider} model: {llm_model}")

    def process_prompt(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """Process a single prompt through the 3WayCoT pipeline.
        
        This enhanced method integrates with the metrics experimentation frameworks
        by properly handling confidence values throughout the pipeline.
        
        Args:
            prompt: The input prompt to process
            context: Optional additional context
            
        Returns:
            Dictionary with processing results and metrics
        """
        if prompt is None:
            self.logger.warning("Received None prompt, setting to empty string")
            prompt = ""
        self.logger.info(f"Processing prompt: {prompt[:100] if prompt else ''}...")
        
        try:
            # Record start time for performance tracking
            import time
            start_time = time.time()
            
            # Generate reasoning steps and extract uncertainties
            reasoning_steps = self.framework.cot_generator.generate(
                query=prompt,
                context=context
            )
            self.logger.info(f"Generated reasoning steps :{reasoning_steps}")
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {len(reasoning_steps)} reasoning steps in {generation_time:.2f} seconds")
            
            # Enhanced confidence extraction - critical for metrics experimentation
            steps_with_enhanced_confidence = self._enhance_confidence_values(reasoning_steps)
            
            # Preprocess assumptions for proper handling in the triadic analysis
            preprocessed_steps = self._preprocess_assumptions(steps_with_enhanced_confidence)
            
            # Instead of using the high-level process_reasoning method which has compatibility issues,
            # we'll directly use the core analysis methods that are known to work
            try:
                # Run the triadic fuzzy concept analysis on the preprocessed steps
                tfca_results = self.framework.tfca_analyzer.analyze_reasoning(
                    reasoning_steps=preprocessed_steps,
                    tau=self.framework.tau
                )
                
                # ThreeWayDecisionMaker requires 0 <= beta < alpha <= 1.0
                # Convert three-way beta to proper form for ThreeWayDecisionMaker
                # In three-way decisions, (1-beta) is the rejection threshold
                # So we need to use a proper beta that's less than alpha
                alpha = self.framework.alpha  # Typically 0.7
                decision_beta = min(0.4, alpha - 0.1)  # Ensure beta < alpha
                
                self.logger.info(f"Initializing ThreeWayDecisionMaker with alpha={alpha}, beta={decision_beta}")
                
                # Import here to avoid circular imports
                from src.core.three_way_decision import ThreeWayDecisionMaker
                
                decision_maker = ThreeWayDecisionMaker(
                    alpha=alpha,
                    beta=decision_beta,
                    tau=0.6  # Standard boundary width threshold
                )
                
                # Apply the decision maker to get classification results
                # Include reasoning_steps in the analysis parameter as expected by ThreeWayDecisionMaker
                tfca_results['reasoning_steps'] = preprocessed_steps
                
                # ENHANCEMENT: Add metrics from confidence extraction to the analysis
                # This ensures the decision maker has access to all confidence-related metrics
                metrics_results = self._calculate_enhanced_metrics(preprocessed_steps)
                tfca_results['confidence_metrics'] = metrics_results
                
                # Log the confidence metrics for debugging
                self.logger.info(f"Enhanced confidence metrics: {metrics_results}")
                
                # Include confidence distribution analysis in uncertainty analysis
                confidence_distribution = self.confidence_extractor.analyze_confidence_distribution(preprocessed_steps)
                uncertainty_analysis = {
                    'confidence_distribution': confidence_distribution,
                    'step_uncertainties': [{'confidence': step.get('confidence', 0.7)} for step in preprocessed_steps]
                }
                
                decisions = decision_maker.make_decisions(
                    analysis=tfca_results,
                    uncertainty_analysis=uncertainty_analysis
                )
                logging.info(f"Decisions: {decisions}")
                # Construct the analysis results dictionary with the necessary components
                analysis_results = {
                    "decisions": decisions,
                    "reasoning_steps": preprocessed_steps,
                    "tfca_results": tfca_results,
                    "metrics": {
                        "confidence_metrics": self._calculate_metrics(decisions),
                        "lattice_metrics": tfca_results.get("lattice_analysis", {})
                    }
                }
                
                # Extract the actual decision items to process
                decision_items = []
                if isinstance(decisions, dict) and 'decisions' in decisions:
                    # If decisions is a dictionary with a 'decisions' key, use that list
                    decision_items = decisions['decisions']
                elif isinstance(decisions, list):
                    # If decisions is already a list, use it directly
                    decision_items = decisions
                else:
                    self.logger.warning(f"Cannot process decisions of type {type(decisions)}")
                    
                # Add concept information to each decision for lattice integration
                for i, decision in enumerate(decision_items):
                    concept_info = {}
                    for concept in tfca_results.get('concepts', []):
                        # Find concepts related to this decision's step
                        intent_indices = [intent[0] for intent in concept.get('intent', []) 
                                         if isinstance(intent, list) and len(intent) > 0]
                        if i in intent_indices:
                            concept_info = {
                                'concept_id': concept.get('id', -1),
                                'concept_type': concept.get('type', ''),
                                'concept_intent': concept.get('intent', []),
                                'concept_extent': concept.get('extent', []),
                                'concept_modus': concept.get('modus', [])
                            }
                            break
                    
                    # Attach concept information to the decision
                    if isinstance(decision, dict):
                        # If it's already a dictionary, just add the concept info
                        decision['concept_info'] = concept_info
                    elif isinstance(decision, str):
                        # If it's a string decision, convert it to a dictionary format
                        # Handle both list and dictionary formats of decisions
                        if isinstance(decisions, list):
                            # Create a new dictionary with the decision string as the 'decision' field
                            decision_dict = {
                                'decision': decision,
                                'step_index': i,  # Use the current index from the enumeration
                                'confidence': 0.7,  # Default confidence
                                'concept_info': concept_info
                            }
                            # Replace the string decision with the dictionary in the decision_items list
                            decision_items[i] = decision_dict
                            self.logger.info(f"Converted string decision '{decision}' to dictionary format")
                        else:
                            # We're already using decision_items, so we don't need to find the decision in the original structure
                            # Just convert the current item and update decision_items
                            decision_dict = {
                                'decision': decision,
                                'step_index': i,  # Use the current index from the enumeration
                                'confidence': 0.7,  # Default confidence
                                'concept_info': concept_info
                            }
                            # Replace the string decision with the dictionary in the decision_items list
                            decision_items[i] = decision_dict
                            self.logger.info(f"Converted string decision '{decision}' to dictionary format")
                    else:
                        # For other types, log a warning and skip
                        self.logger.warning(f"Cannot attach concept info to decision of type {type(decision)}")
                        
                # Update the original decisions structure with our modified decision_items
                if isinstance(decisions, dict) and 'decisions' in decisions:
                    # If decisions is a dictionary with a 'decisions' key, update that list
                    decisions['decisions'] = decision_items
                elif isinstance(decisions, list):
                    # If decisions is a list, replace it with the modified decision_items
                    # This is a bit tricky as we need to update the original reference
                    for i in range(min(len(decisions), len(decision_items))):
                        decisions[i] = decision_items[i]
                    # Handle case where decision_items has more items than the original list
                    if len(decision_items) > len(decisions):
                        for i in range(len(decisions), len(decision_items)):
                            decisions.append(decision_items[i])
                    # Handle case where the original list had more items than decision_items
                    if len(decisions) > len(decision_items):
                        for i in range(len(decision_items), len(decisions)):
                            decisions.pop()
                
                # Log that we've completed attaching concept info
                self.logger.debug("Completed attaching concept information to decisions")

                
            except Exception as e:
                self.logger.error(f"Error during reasoning analysis: {e}", exc_info=True)
                raise
            
            # Get decisions from analysis
            decisions = analysis_results.get('decisions', [])
            
            # Ensure all decisions have proper confidence values for metrics calculation
            self._normalize_decision_confidence(decisions)
            
            # Log decision summary
            
            # Extract the decisions list if we have a dictionary with 'decisions' key
            if isinstance(decisions, dict) and 'decisions' in decisions:
                decision_list = decisions['decisions']
                # Use summary if available for accurate counts
                if 'summary' in decisions:
                    summary = decisions['summary']
                    accept_count = summary.get('accept_count', 0)
                    reject_count = summary.get('reject_count', 0)
                    abstain_count = summary.get('abstain_count', 0)
                    self.logger.info(f"Using decision counts from summary: ACCEPT={accept_count}, REJECT={reject_count}, ABSTAIN={abstain_count}")
                else:
                    # Count manually if no summary
                    accept_count = sum(1 for d in decision_list if isinstance(d, dict) and d.get('decision') == 'ACCEPT')
                    reject_count = sum(1 for d in decision_list if isinstance(d, dict) and d.get('decision') == 'REJECT')
                    abstain_count = sum(1 for d in decision_list if isinstance(d, dict) and d.get('decision') == 'ABSTAIN')
            else:
                # Handle when decisions is already a list
                decision_list = decisions if isinstance(decisions, list) else []
                accept_count = 0
                reject_count = 0
                abstain_count = 0
                
                for d in decision_list:
                    if isinstance(d, dict):
                        decision_value = d.get('decision')
                    else:
                        # If it's a string, assume it's the decision value itself
                        decision_value = d
                        
                    if decision_value == 'ACCEPT':
                        accept_count += 1
                    elif decision_value == 'REJECT':
                        reject_count += 1
                    elif decision_value == 'ABSTAIN':
                        abstain_count += 1
                    
            self.logger.info(f"Decision summary: ACCEPT={accept_count}, REJECT={reject_count}, ABSTAIN={abstain_count}")
            
            # Resolve uncertainties if present
            if abstain_count > 0:
                self.logger.info(f"Resolving {abstain_count} uncertain decisions")
                decisions = self._resolve_uncertainties(decisions, analysis_results)
            
            # Calculate metrics with enhanced confidence values
            metrics = self._calculate_metrics(decisions)
            
            # Prepare and return results
            return self._prepare_results(
                prompt=prompt,
                context=context,
                reasoning_steps=steps_with_enhanced_confidence,
                decisions=decisions,
                analysis_results=analysis_results,
                metrics=metrics
            )
        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}", exc_info=True)
            raise

    def _preprocess_assumptions(self, reasoning_steps):
        """Preprocess reasoning steps to ensure proper assumption handling.
        
        This method ensures that assumptions are properly formatted as whole entities
        rather than individual characters, which prevents issues in the concept lattice.
        
        Args:
            reasoning_steps: List of reasoning steps with assumptions
            
        Returns:
            Preprocessed reasoning steps with properly formatted assumptions
        """
        preprocessed_steps = []
        
        for i, step in enumerate(reasoning_steps):
            # Create a copy of the step
            processed_step = dict(step)
            
            # Extract assumptions if they exist
            raw_assumptions = step.get('extracted_assumptions', [])
            if not raw_assumptions and 'assumptions' in step and step['assumptions']:
                # Try to parse assumptions from the assumptions field
                if isinstance(step['assumptions'], str):
                    raw_assumptions = [a.strip() for a in step['assumptions'].split('\n') if a.strip()]
                elif isinstance(step['assumptions'], list):
                    raw_assumptions = step['assumptions']
            
            # If still no assumptions, try to extract from reasoning text
            if not raw_assumptions and 'reasoning' in step:
                reasoning_text = step['reasoning']
                assumption_section = reasoning_text.split('Assumptions:', 1)
                if len(assumption_section) > 1:
                    assumption_lines = assumption_section[1].strip().split('\n')
                    raw_assumptions = [line.strip() for line in assumption_lines if line.strip()]
            
            # Ensure assumptions are properly indexed and not split into characters
            indexed_assumptions = []
            for j, assumption in enumerate(raw_assumptions):
                # Only use the assumption if it's a non-empty string
                if assumption and isinstance(assumption, str):
                    # Add an index prefix to ensure unique identifiers
                    indexed_assumption = f"assumption_{i}_{j}: {assumption}"
                    indexed_assumptions.append(indexed_assumption)
            
            # Update the step with properly formatted assumptions
            processed_step['assumptions'] = indexed_assumptions
            
            # Ensure we have standard fields needed by the analyzer
            if 'Assumptions' not in processed_step:
                processed_step['Assumptions'] = processed_step['assumptions']
            
            if 'Description' not in processed_step:
                processed_step['Description'] = f"Step {i+1}"
                
            # Make sure confidence is properly categorized
            confidence = processed_step.get('confidence', 0.5)
            if 'confidence_category' not in processed_step:
                if confidence >= 0.75:
                    processed_step['confidence_category'] = 'high'
                elif confidence >= 0.4:
                    processed_step['confidence_category'] = 'medium'
                else:
                    processed_step['confidence_category'] = 'low'
            
            preprocessed_steps.append(processed_step)
            
        return preprocessed_steps

    def _prepare_results(self, prompt: str, context: str, reasoning_steps: List[Dict], 
                        decisions: List[Dict], analysis_results: Dict, metrics: Dict = None) -> Dict[str, Any]:
        """Prepare results dictionary with all processing outputs.
        
        Args:
            prompt: The original prompt
            context: Optional context provided with the prompt
            reasoning_steps: Enhanced reasoning steps with confidence values
            decisions: List of decisions, possibly with resolved uncertainties
            analysis_results: Results from the framework analysis
            metrics: Pre-calculated metrics or None (will be calculated if None)
            
        Returns:
            Dictionary with complete processing results
        """
        # Calculate metrics if not provided
        if metrics is None:
            metrics = self._calculate_metrics(decisions)
            
        return {
            'prompt': prompt,
            'context': context,
            'reasoning_steps': reasoning_steps,
            'decisions': decisions,
            'analysis_results': analysis_results,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }

    def _enhance_confidence_values(self, reasoning_steps):
        """Extract and enhance confidence values for reasoning steps.
        
        This method extracts confidence values from reasoning step text using
        the confidence extractor and enhances them with additional metadata
        needed for metrics experimentation.
        
        Args:
            reasoning_steps: List of reasoning steps
            
        Returns:
            Enhanced reasoning steps with confidence values
        """
        enhanced_steps = []
        
        for i, step in enumerate(reasoning_steps):
            # Create a copy of the step
            enhanced_step = dict(step)
            
            # Extract confidence using the enhanced confidence extractor
            # The extract_confidence method now returns a tuple (confidence_value, method)
            confidence_tuple = self.confidence_extractor.extract_confidence(step.get('reasoning', ''))
            confidence_value = confidence_tuple[0]  # Extract the numeric value
            confidence_method = confidence_tuple[1]  # Extract the method
            
            # Map confidence value to a level description
            if confidence_value >= 0.75:
                confidence_level = 'high'
            elif confidence_value >= 0.4:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
                
            # Store extracted confidence and method in the step for use in decision making
            enhanced_step['confidence'] = confidence_value
            enhanced_step['confidence_method'] = confidence_method
            enhanced_step['confidence_level'] = confidence_level
            
            # Add metadata needed for metrics experimentation
            enhanced_step['step_index'] = i
            enhanced_step['metrics'] = {
                'confidence': confidence_value,
                'uncertainty': 1.0 - confidence_value,  # Simple inverse for uncertainty
                'stability': 0.5 + (confidence_value - 0.5) * 0.8,  # Stability correlates with confidence
                'connectivity': 0.6,  # Default values for other metrics
                'density': 0.6,
                'coverage': 0.6
            }
            
            # Log extracted confidence
            self.logger.info(f"Step {i+1}: Extracted confidence {confidence_value:.2f} ({confidence_level})")
            enhanced_steps.append(enhanced_step)
            
        return enhanced_steps
    
    def _normalize_decision_confidence(self, decisions):
        """Normalize confidence values in decisions to ensure consistency.
        
        This method ensures all decisions have normalized confidence values
        for consistent metrics calculation.
        
        Args:
            decisions: List of decision dictionaries or decision object with 'decisions' key
            
        Returns:
            None (modifies decisions in place)
        """
        if not decisions:
            return
            
        # Handle the case where decisions is a dictionary with a 'decisions' key (from enhanced ThreeWayDecisionMaker)
        if isinstance(decisions, dict) and 'decisions' in decisions:
            decision_list = decisions['decisions']
        # Handle case when decisions is already a list
        elif isinstance(decisions, list):
            decision_list = decisions
        # Handle unexpected input type
        else:
            self.logger.warning(f"Cannot normalize confidence for decision object of type {type(decisions)}")
            return
            
        # Ensure all decisions have a confidence value
        for i, decision in enumerate(decision_list):
            # Skip non-dictionary items
            if not isinstance(decision, dict):
                self.logger.warning(f"Skipping non-dictionary decision at index {i} of type {type(decision)}")
                continue
                
            # If no confidence, use decision-based defaults
            if 'confidence' not in decision:
                if decision.get('decision') == 'ACCEPT':
                    decision['confidence'] = 0.8
                elif decision.get('decision') == 'REJECT':
                    decision['confidence'] = 0.2
                else:  # ABSTAIN
                    decision['confidence'] = 0.5
                    
            # Ensure confidence is within [0,1] range
            decision['confidence'] = max(0.0, min(1.0, decision['confidence']))
            
            # Add membership degrees if not present (for metrics calculation)
            if 'membership_degrees' not in decision:
                decision['membership_degrees'] = {
                    'accept': decision['confidence'] if decision.get('decision') == 'ACCEPT' else 0.3,
                    'reject': decision['confidence'] if decision.get('decision') == 'REJECT' else 0.3,
                    'abstain': decision['confidence'] if decision.get('decision') == 'ABSTAIN' else 0.4
                }
                
    def _apply_use_case(self, use_case: str):
        """Apply predefined parameter configurations based on the use case.
        
        This method configures all threshold parameters like alpha, beta, tau,
        max_assumptions_per_step, and max_steps based on the specified use case
        as defined in the configuration file.
        
        Args:
            use_case: The use case preset ('conservative', 'exploratory', etc.)
        """
        self.logger.info(f"Applying use case preset: {use_case}")
        
        # Get use case parameters from config file
        use_cases_config = self.config.get('framework.use_cases', {})
        
        # Get the specific use case configuration or use default if not found
        if use_case in use_cases_config:
            use_case_params = use_cases_config[use_case]
            self.logger.info(f"Found configuration for use case '{use_case}' in config file")
        else:
            self.logger.warning(f"Use case '{use_case}' not found in config. Using default parameters.")
            use_case_params = use_cases_config.get('default', {})
        
        # Get default framework parameters as fallback
        default_params = {
            'alpha': self.config.get('framework.alpha', 0.7),
            'beta': self.config.get('framework.beta', 0.4),
            'tau': self.config.get('framework.tau', 0.6),
            'max_assumptions': self.config.get('framework.max_assumptions', 5),
            'max_steps': self.config.get('framework.max_steps', 5),
            'confidence_weight': self.config.get('framework.confidence_weight', 0.7)
        }
        
        # Get metrics weights from config if available
        metrics_weights = self.config.get('framework.metrics_weights', {
            'similarity': 0.3,
            'coverage': 0.2,
            'consistency': 0.2,
            'confidence': 0.3
        })
        
        # Merge parameters: default framework params < use case params
        params = {**default_params, **use_case_params}
        params['metrics_weights'] = metrics_weights
        
        # Apply the parameters to the framework components
        self.logger.info(f"Setting framework parameters: alpha={params['alpha']}, beta={params['beta']}, tau={params['tau']}")
        
        # Set the parameters on the framework
        self.framework.alpha = params['alpha']
        self.framework.beta = params['beta']
        self.framework.tau = params['tau']
        
        # Set max assumptions per step in the Chain of Thought generator
        if hasattr(self.framework, 'cot_generator') and self.framework.cot_generator:
            self.framework.cot_generator.max_assumptions = params['max_assumptions']
            self.framework.cot_generator.max_steps = params['max_steps']
            self.logger.info(f"Set max_assumptions={params['max_assumptions']}, max_steps={params['max_steps']}")
        
        # Apply metric weights if the decision maker supports it
        if hasattr(self.framework, 'decision_maker') and self.framework.decision_maker:
            if hasattr(self.framework.decision_maker, 'set_weights'):
                self.framework.decision_maker.set_weights(params['metrics_weights'])
                self.logger.info(f"Applied metrics weights: {params['metrics_weights']}")
            
            # Set confidence weight if supported
            if hasattr(self.framework.decision_maker, 'confidence_weight'):
                self.framework.decision_maker.confidence_weight = params['confidence_weight']
                self.logger.info(f"Set confidence_weight={params['confidence_weight']}")
        
        # Store the parameters in the config for reference
        self.config.config['framework'] = {
            **self.config.config.get('framework', {}),
            **params
        }
    
    def _calculate_metrics(self, decisions):
        """
        Calculate metrics from the decisions.
        
        This method calculates various metrics based on the decisions and their confidence values,
        which are critical for the metrics experimentation frameworks.
        
        Args:
            decisions: List of decision dictionaries or dictionary with 'decisions' key
            
        Returns:
            Dictionary with calculated metrics
        """
        # Initialize default metrics
        default_metrics = {
            'acceptance_rate': 0,
            'rejection_rate': 0,
            'abstention_rate': 0,
            'average_confidence': 0,
            'confidence_by_decision': {
                'ACCEPT': 0,
                'REJECT': 0,
                'ABSTAIN': 0
            },
            'total_decisions': 0,
            'decision_counts': {
                'ACCEPT': 0,
                'REJECT': 0,
                'ABSTAIN': 0
            }
        }
        
        if not decisions:
            self.logger.warning("No decisions provided for metrics calculation")
            return default_metrics
            
        # Initialize decision list and handle different input formats
        decision_list = []
        
        # Handle case when decisions is a dictionary with a 'decisions' key (from enhanced ThreeWayDecisionMaker)
        if isinstance(decisions, dict):
            if 'decisions' in decisions:
                decision_list = decisions['decisions']
                # If we have a summary in the decisions, use it for more accurate counts
                if 'summary' in decisions:
                    summary = decisions['summary']
                    accepted = summary.get('accept_count', 0)
                    rejected = summary.get('reject_count', 0)
                    abstained = summary.get('abstain_count', 0)
                    self.logger.info(f"Using decision counts from summary: ACCEPT={accepted}, REJECT={rejected}, ABSTAIN={abstained}")
            else:
                self.logger.warning("Decisions dictionary does not contain 'decisions' key")
                return default_metrics
        # Handle case when decisions is already a list
        elif isinstance(decisions, list):
            decision_list = decisions
        # Handle unexpected input type
        else:
            self.logger.warning(f"Unexpected decision format: {type(decisions)}. Expected dict with 'decisions' key or list.")
            return default_metrics
            
        if not isinstance(decision_list, list):
            self.logger.warning(f"Expected decision_list to be a list, got {type(decision_list)}")
            return default_metrics
        
        # Initialize counters
        total = len(decision_list)
        if total == 0:
            self.logger.warning("No decisions in decision_list for metrics calculation")
            return default_metrics
            
        # Initialize decision counts
        accepted = 0
        rejected = 0
        abstained = 0
        confidences = []
        accept_confidences = []
        reject_confidences = []
        abstain_confidences = []
        
        # Process each decision
        for d in decision_list:
            if not isinstance(d, dict):
                self.logger.warning(f"Skipping non-dictionary decision: {d}")
                continue
                
            # Get decision and confidence
            decision = d.get('decision')
            confidence = float(d.get('confidence', 0))
            
            # Update counters
            if decision == 'ACCEPT':
                accepted += 1
                accept_confidences.append(confidence)
            elif decision == 'REJECT':
                rejected += 1
                reject_confidences.append(confidence)
            elif decision == 'ABSTAIN':
                abstained += 1
                abstain_confidences.append(confidence)
            else:
                self.logger.warning(f"Unknown decision type: {decision}")
                
            confidences.append(confidence)
        
        # Calculate metrics
        total_decisions = accepted + rejected + abstained
        if total_decisions == 0:
            self.logger.warning("No valid decisions found in decision_list")
            return default_metrics
            
        # Calculate rates
        acceptance_rate = accepted / total_decisions if total_decisions > 0 else 0
        rejection_rate = rejected / total_decisions if total_decisions > 0 else 0
        abstention_rate = abstained / total_decisions if total_decisions > 0 else 0
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate confidence by decision type
        confidence_by_decision = {
            'ACCEPT': sum(accept_confidences) / len(accept_confidences) if accept_confidences else 0,
            'REJECT': sum(reject_confidences) / len(reject_confidences) if reject_confidences else 0,
            'ABSTAIN': sum(abstain_confidences) / len(abstain_confidences) if abstain_confidences else 0
        }
        
        # Prepare the metrics dictionary
        metrics = {
            'acceptance_rate': acceptance_rate,
            'rejection_rate': rejection_rate,
            'abstention_rate': abstention_rate,
            'average_confidence': avg_confidence,
            'confidence_by_decision': confidence_by_decision,
            'total_decisions': total_decisions,
            'decision_counts': {
                'ACCEPT': accepted,
                'REJECT': rejected,
                'ABSTAIN': abstained
            },
            # Add additional metrics for debugging
            'total_confidence': sum(confidences),
            'confidence_values': confidences
        }
        
        # Log detailed metrics information
        self.logger.info(
            f"Metrics calculated - "
            f"Acceptance: {acceptance_rate:.2f} ({accepted}/{total_decisions}), "
            f"Rejection: {rejection_rate:.2f} ({rejected}/{total_decisions}), "
            f"Abstention: {abstention_rate:.2f} ({abstained}/{total_decisions}), "
            f"Avg Confidence: {avg_confidence:.2f}"
        )
        
        # Log confidence by decision type
        for decision_type, confidence in confidence_by_decision.items():
            self.logger.debug(f"  {decision_type} confidence: {confidence:.2f}")
            
        return metrics
        
    def _safe_get_dependencies(self, step_index, triadic_context):
        """Safely get dependencies for a step, handling missing method errors.
        
        Args:
            step_index: Index of the step to get dependencies for
            triadic_context: The triadic context to extract dependencies from
            
        Returns:
            List of dependencies or empty list if not available
        """
        try:
            # Check if the tfca_analyzer has the get_dependencies method
            if hasattr(self.framework.tfca_analyzer, 'get_dependencies'):
                return self.framework.tfca_analyzer.get_dependencies(
                    step_index=step_index,
                    triadic_context=triadic_context
                )
            # If not, try to extract dependencies from the triadic context directly
            elif isinstance(triadic_context, dict) and 'dependencies' in triadic_context:
                dependencies_dict = triadic_context.get('dependencies', {})
                return dependencies_dict.get(str(step_index), [])
            else:
                # Return an empty list as fallback
                return []
        except Exception as e:
            self.logger.warning(f"Error getting dependencies for step {step_index}: {e}")
            return []
            
    def _resolve_uncertainties(self, decisions, analysis_results):
        """Resolve uncertain decisions using the uncertainty resolver.
        
        This method attempts to resolve decisions classified as ABSTAIN by
        using the framework's uncertainty resolver.
        
        Args:
            decisions: List of decision dictionaries
            analysis_results: Analysis results from the framework
            
        Returns:
            List of decisions with uncertainties resolved where possible
        """
        # The uncertainty resolver should already be initialized in the framework
        # but we'll check just to be safe
        if not hasattr(self.framework, 'uncertainty_resolver') or not self.framework.uncertainty_resolver:
            from src.core.uncertainty_resolver import UncertaintyResolver
            
            # Get default uncertainty parameters from config
            relevance_threshold = self.config.get('uncertainty.relevance_threshold', 0.7)
            validity_threshold = self.config.get('uncertainty.validity_threshold', 0.6)
            confidence_threshold = self.config.get('uncertainty.confidence_threshold', 0.7)
            
            # Create UncertaintyResolver with the CoT generator and thresholds
            self.framework.uncertainty_resolver = UncertaintyResolver(
                cot_generator=self.framework.cot_generator,  # Make sure we use the same generator
                knowledge_base=self.knowledge_base,
                relevance_threshold=relevance_threshold,
                validity_threshold=validity_threshold,
                confidence_threshold=confidence_threshold
            )
            self.logger.info(f"Initialized UncertaintyResolver with relevance_threshold={relevance_threshold}, "
                          f"validity_threshold={validity_threshold}, confidence_threshold={confidence_threshold}")
        
        # Extract uncertain steps from decisions
        uncertain_steps = []
        
        # Handle different decision object formats
        if isinstance(decisions, dict) and 'decisions' in decisions:
            decision_list = decisions['decisions']
        elif isinstance(decisions, list):
            decision_list = decisions
        else:
            self.logger.warning(f"Cannot resolve uncertainties for decision object of type {type(decisions)}")
            return decisions
            
        for i, decision in enumerate(decision_list):
            # Skip non-dictionary items
            if not isinstance(decision, dict):
                self.logger.warning(f"Skipping non-dictionary decision at index {i} of type {type(decision)}")
                continue
                
            if decision.get('decision') == 'ABSTAIN':
                # Get the reasoning step for this decision
                step_idx = decision.get('step_index', i)
                if step_idx < len(analysis_results.get('reasoning_steps', [])):
                    # Get the step from the analysis results
                    step = analysis_results.get('reasoning_steps', [])[step_idx]
                    step['uncertainty'] = {
                        'resolved': False,
                        'resolution_attempts': 0
                    }
                    uncertain_steps.append(step)
        
        # Resolve uncertainties if there are any
        resolved_steps = []
        if uncertain_steps:
            # Get all reasoning steps for context
            all_steps = analysis_results.get('reasoning_steps', [])
            
            # Try to resolve uncertainties only if we have the uncertainty resolver
            if hasattr(self.framework, 'uncertainty_resolver') and self.framework.uncertainty_resolver:
                try:
                    # For each uncertain step, try to resolve it
                    for step in uncertain_steps:
                        try:
                            # Get dependencies from the triadic context if possible
                            dependencies = self._safe_get_dependencies(
                                step_index=step.get('step_index', 0),
                                triadic_context=analysis_results.get('triadic_context', {})
                            )
                            
                            # Resolve the uncertain step
                            resolved_step = self.framework.uncertainty_resolver.resolve(
                                uncertain_step=step,
                                all_steps=all_steps,
                                triadic_context=analysis_results.get('triadic_context', {})
                            )
                            
                            if resolved_step:
                                resolved_steps.append(resolved_step)
                        except Exception as step_error:
                            self.logger.warning(f"Error resolving step {step.get('step_index', 0)}: {step_error}")
                            # Skip this step and continue with others
                            continue
                except Exception as e:
                    self.logger.warning(f"Skipping uncertainty resolution due to error: {e}")
            
        # Update decisions with resolved information
        for i, decision in enumerate(decision_list):
            if isinstance(decision, dict) and decision.get('decision') == 'ABSTAIN':
                step_idx = decision.get('step_index', i)
                # Find the matching resolved step
                for resolved_step in resolved_steps:
                    if resolved_step.get('step_num') == step_idx + 1 or resolved_step.get('step_index') == step_idx:
                        # If resolution was successful, update the decision
                        if resolved_step.get('uncertainty', {}).get('resolved', False):
                            # Use the resolved confidence for the decision
                            confidence = resolved_step.get('confidence', 0.5)
                            
                            # Determine decision based on confidence
                            if confidence >= self.framework.alpha:
                                new_decision = 'ACCEPT'
                            elif confidence <= 1 - self.framework.beta:
                                new_decision = 'REJECT'
                            else:
                                new_decision = 'ABSTAIN'
                                
                            if new_decision != 'ABSTAIN':
                                self.logger.info(f"Resolved uncertainty: {decision.get('decision')} -> {new_decision}")
                                decision['decision'] = new_decision
                                decision['confidence'] = confidence
                                decision['explanation'] = f"Resolved using {resolved_step.get('uncertainty', {}).get('resolution_strategy', 'unknown')}"
        
        return decisions

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save results to a file."""
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
            
    def _apply_use_case(self, use_case: str) -> None:
        """Apply a predefined use case configuration.
        
        Args:
            use_case: Name of the use case to apply (e.g., 'conservative', 'exploratory')
        """
        use_case_config = self.config.get(f'use_cases.{use_case}', {})
        if not use_case_config:
            self.logger.warning(f"Use case '{use_case}' not found in configuration")
            return
        
        self.logger.info(f"Applying '{use_case}' use case configuration")
        self.update_parameters(**use_case_config)
    
    def update_parameters(self, **kwargs) -> None:
        """Update framework parameters dynamically.
        
        Args:
            **kwargs: Parameters to update with their new values
        """
        if not kwargs:
            self.logger.warning("No parameters provided to update")
            return
            
        self.logger.info(f"Updating parameters: {', '.join(kwargs.keys())}")
        
        # Update framework parameters
        if 'alpha' in kwargs:
            self.framework.alpha = float(kwargs['alpha'])
            self.logger.debug(f"Set alpha to {self.framework.alpha}")
            
        if 'beta' in kwargs:
            self.framework.beta = float(kwargs['beta'])
            self.logger.debug(f"Set beta to {self.framework.beta}")
            
        if 'tau' in kwargs:
            self.framework.tau = float(kwargs['tau'])
            self.logger.debug(f"Set tau to {self.framework.tau}")
            
        if 'similarity_threshold' in kwargs:
            self.framework.similarity_threshold = float(kwargs['similarity_threshold'])
            self.logger.debug(f"Set similarity threshold to {self.framework.similarity_threshold}")
            
        if 'max_steps' in kwargs and hasattr(self.framework, 'cot_generator'):
            self.framework.cot_generator.max_steps = int(kwargs['max_steps'])
            self.logger.debug(f"Set max steps to {self.framework.cot_generator.max_steps}")
            
        if 'max_assumptions' in kwargs and hasattr(self.framework, 'cot_generator'):
            self.framework.cot_generator.max_assumptions = int(kwargs['max_assumptions'])
            self.logger.debug(f"Set max assumptions to {self.framework.cot_generator.max_assumptions}")
            
        if 'confidence_weight' in kwargs and hasattr(self.framework, 'decision_maker'):
            self.framework.decision_maker.confidence_weight = float(kwargs['confidence_weight'])
            self.logger.debug(f"Set confidence weight to {self.framework.decision_maker.confidence_weight}")
    
def process_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Process a dataset of prompts and contexts.
        
        Args:
            dataset_path: Path to the dataset file (JSON format)
            
        Returns:
            Dictionary with combined results
        """
        self.logger.info(f"Processing dataset from {dataset_path}")
        
        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            if not isinstance(dataset, list):
                raise ValueError("Dataset must be a list of prompt objects")
                
            results = []
            for i, item in enumerate(dataset):
                self.logger.info(f"Processing item {i+1}/{len(dataset)}")
                
                # Extract prompt and context
                prompt = item.get('prompt', '')
                context = item.get('context', '')
                
                if not prompt:
                    self.logger.warning(f"Skipping item {i+1}: No prompt provided")
                    continue
                    
                # Process the item
                item_result = self.process_prompt(prompt, context)
                
                # Add metadata
                item_result['item_id'] = item.get('id', str(i))
                if 'metadata' in item:
                    item_result['metadata'] = item['metadata']
                    
                results.append(item_result)
                
            # Create combined results
            combined_results = {
                'dataset': os.path.basename(dataset_path),
                'timestamp': self.config.get('app.start_time', ''),
                'item_count': len(results),
                'items': results,
                'summary': self._calculate_dataset_metrics(results)
            }
            
            return combined_results
                
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            raise
    
def _calculate_dataset_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics for a dataset of results.
        
        Args:
            results: List of result items from processing prompts
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {
                'acceptance_rate': 0,
                'rejection_rate': 0,
                'abstention_rate': 0,
                'average_confidence': 0,
                'item_count': 0
            }
            
        # Extract metrics from each result
        all_metrics = [r.get('metrics', {}) for r in results if 'metrics' in r]
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in ['acceptance_rate', 'rejection_rate', 'abstention_rate', 'average_confidence']:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0
            
        # Count decisions across all items
        all_decisions = []
        for r in results:
            if 'decisions' in r:
                all_decisions.extend(r['decisions'])
                
        total_decisions = len(all_decisions) if all_decisions else 1
        decision_counts = {
            'accept': sum(1 for d in all_decisions if d.get('decision') == 'ACCEPT'),
            'reject': sum(1 for d in all_decisions if d.get('decision') == 'REJECT'),
            'abstain': sum(1 for d in all_decisions if d.get('decision') == 'ABSTAIN')
        }
        
        # Return combined metrics
        return {
            **avg_metrics,
            'total_decisions': total_decisions,
            'decision_counts': decision_counts,
            'item_count': len(results)
        }

def main():
    """Main entry point for the 3WayCoT application."""
    parser = argparse.ArgumentParser(description="3WayCoT Framework")
    
    # Input parameters
    parser.add_argument("--prompt", type=str, help="Input prompt to analyze")
    parser.add_argument("--context", type=str, default="", 
                       help="Additional context for the prompt")
    parser.add_argument("--dataset", type=str, 
                       help="Path to dataset file (JSON format)")
    parser.add_argument("--config", type=str, 
                       help="Path to custom config file")
    parser.add_argument("--output", type=str, default="results/output.json",
                       help="Output file path")
    parser.add_argument("--use-case", type=str, 
                       choices=['default', 'conservative', 'exploratory'],
                       default='default',
                       help="Use case preset to apply")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode for detailed error messages")
    
    args = parser.parse_args()
    app = None  # Initialize app variable to None for error handling scope
    
    try:
        # Initialize application with config
        app = ThreeWayCoTApp(args.config)
        
        # Apply use case preset if specified
        if args.use_case != 'default':
            app._apply_use_case(args.use_case)
        
        # Process input
        if args.prompt:
            result = app.process_prompt(args.prompt, args.context)
            app.save_results(result, args.output)
            print(f"Results saved to {args.output}")
        elif args.dataset:
            results = app.process_dataset(args.dataset)
            app.save_results(results, args.output)
            print(f"Processed dataset. Results saved to {args.output}")
        else:
            parser.print_help()
            print("\nError: Either --prompt or --dataset must be provided")
            return 1
            
    except Exception as e:
        logging.error(f"Application error: {e}")
        # Check if debug mode is enabled either via app config or command line
        debug_mode = False
        if app is not None and app.config is not None:
            debug_mode = app.config.get('app.debug', False)
        debug_mode = debug_mode or args.debug  # Command-line flag overrides config
        
        if debug_mode:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())