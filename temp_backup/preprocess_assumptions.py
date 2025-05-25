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
        
    self.logger.info(f"Preprocessed {len(preprocessed_steps)} reasoning steps with formatted assumptions")
    return preprocessed_steps
