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
    self.logger.info(f"Processing prompt: {prompt[:100]}...")
    
    try:
        # Record start time for performance tracking
        import time
        start_time = time.time()
        
        # Generate reasoning steps and extract uncertainties
        reasoning_steps = self.framework.cot_generator.generate(
            query=prompt,
            context=context
        )
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generated {len(reasoning_steps)} reasoning steps in {generation_time:.2f} seconds")
        
        # Enhanced confidence extraction - critical for metrics experimentation
        steps_with_enhanced_confidence = self._enhance_confidence_values(reasoning_steps)
        
        # Preprocess assumptions for proper handling in the triadic analysis
        preprocessed_steps = self._preprocess_assumptions(steps_with_enhanced_confidence)
        
        # Instead of using the high-level process_reasoning method which might have compatibility issues,
        # we'll directly use the core analysis methods that are known to work
        try:
            # Run the triadic fuzzy concept analysis on the preprocessed steps
            tfca_results = self.framework.triadic_analyzer.analyze_data(
                reasoning_steps=preprocessed_steps,
                similarity_threshold=self.framework.similarity_threshold
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
                gamma=0.6  # Standard boundary width threshold
            )
            
            # Apply the decision maker to get classification results
            decisions = decision_maker.make_decisions(
                analysis=tfca_results,
                uncertainty_analysis={"reasoning_steps": preprocessed_steps}
            )
            
            # Integrate decisions with concept information for better metrics experimentation
            for i, decision in enumerate(decisions):
                # Find the corresponding step and enhance with confidence metrics
                step_idx = decision.get('step_index', i)
                if step_idx < len(preprocessed_steps):
                    step = preprocessed_steps[step_idx]
                    
                    # Copy confidence metrics from step to decision for metrics experimentation
                    if 'metrics' in step:
                        decision['metrics'] = step['metrics']
                    elif 'confidence' in step:
                        # If no metrics, at least ensure confidence is included
                        confidence = step.get('confidence', 0.5)
                        decision['metrics'] = {
                            'confidence': confidence,
                            'uncertainty': 1.0 - confidence,
                            'stability': 0.5 + (confidence - 0.5) * 0.8,
                            'connectivity': 0.6,
                            'density': 0.6,
                            'coverage': 0.6
                        }
                
                # Add concept information to each decision for lattice integration
                concept_info = {}
                for concept in tfca_results.get('concepts', []):
                    # Find concepts related to this decision's step
                    intent_indices = [intent[0] for intent in concept.get('intent', []) 
                                     if isinstance(intent, list) and len(intent) > 0]
                    if step_idx in intent_indices:
                        concept_info = {
                            'concept_id': concept.get('id', -1),
                            'concept_type': concept.get('type', ''),
                            'concept_intent': concept.get('intent', []),
                            'concept_extent': concept.get('extent', []),
                            'concept_modus': concept.get('modus', [])
                        }
                        break
                
                # Attach concept information to the decision
                decision['concept_info'] = concept_info
            
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
        except Exception as e:
            self.logger.error(f"Error during reasoning analysis: {e}", exc_info=True)
            raise
        
        # Get decisions from analysis
        decisions = analysis_results.get('decisions', [])
        
        # Ensure all decisions have proper confidence values for metrics calculation
        self._normalize_decision_confidence(decisions)
        
        # Log decision summary
        accept_count = sum(1 for d in decisions if d.get('decision') == 'ACCEPT')
        reject_count = sum(1 for d in decisions if d.get('decision') == 'REJECT')
        abstain_count = sum(1 for d in decisions if d.get('decision') == 'ABSTAIN')
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
            reasoning_steps=preprocessed_steps,
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
        
        # Add metrics information to support metrics experimentation framework
        if 'metrics' not in processed_step:
            confidence = processed_step.get('confidence', 0.5)
            processed_step['metrics'] = {
                'confidence': confidence,
                'uncertainty': 1.0 - confidence,
                'stability': 0.5 + (confidence - 0.5) * 0.8,  # Higher confidence = higher stability
                'connectivity': 0.6,  # Default value, will be updated by concept analysis
                'density': 0.6,       # Default value, will be updated by concept analysis
                'coverage': 0.6       # Default value, will be updated by concept analysis
            }
        
        preprocessed_steps.append(processed_step)
    
    self.logger.info(f"Preprocessed {len(preprocessed_steps)} reasoning steps with formatted assumptions")
    return preprocessed_steps
