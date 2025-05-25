"""
Three-Way Decision Maker Component.

This module implements the Three-Way Decision Maker for the 3WayCoT framework,
which classifies reasoning steps as accept, reject, or abstain.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

class ThreeWayDecisionMaker:
    """
    Makes three-way decisions for reasoning steps.
    
    This class implements the Three-Way Decision Maker as described in Algorithm 4
    of the 3WayCoT framework specifications.
    """
    
    def __init__(
        self, 
        alpha: float = 0.7,  # Acceptance threshold
        beta: float = 0.6,   # Rejection threshold
        tau: float = 0.5     # Similarity threshold
    ):
        """
        Initialize the Three-Way Decision Maker.
        
        Args:
            alpha: Acceptance threshold
            beta: Rejection threshold
            tau: Similarity threshold for concept formation
        """
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.logger = logging.getLogger("3WayCoT.DecisionMaker")
        
        # Verify that thresholds are valid
        if alpha + beta <= 1.0:
            self.logger.warning(
                f"Warning: alpha ({alpha}) + beta ({beta}) <= 1.0. " +
                "This may lead to overlapping acceptance and rejection regions."
            )
    
    def calculate_membership(
        self, 
        step: Dict[str, Any], 
        concept_lattice: List[Dict[str, Any]]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate acceptance and rejection intervals for a reasoning step.
        
        Args:
            step: The reasoning step to evaluate
            concept_lattice: The concept lattice from the Triadic Context Constructor
            
        Returns:
            Tuple of (acceptance_interval, rejection_interval), where each interval is (lower, upper)
        """
        self.logger.info(f"Calculating membership intervals for step: {step.get('step_num', '?')}")
        
        # Find the step's index in original list (needed to find corresponding concepts)
        step_idx = step.get("step_num", 0) - 1  # Adjust for 0-indexing
        
        # Extract the accept and reject attributes from the lattice
        accept_attributes = []
        reject_attributes = []
        
        for concept in concept_lattice:
            # Skip if this concept doesn't contain the step
            if step_idx not in concept.get("A", []):
                continue
                
            # Process all attributes in this concept
            for attr_idx in concept.get("C", []):
                # Get the attribute from the first concept that has it defined
                for c in concept_lattice:
                    if hasattr(c, "M") and attr_idx < len(c.get("M", [])):
                        attr = c.get("M")[attr_idx]
                        if attr.get("type") == "accept":
                            accept_attributes.append(attr)
                        elif attr.get("type") == "reject":
                            reject_attributes.append(attr)
                        break
        
        # Calculate acceptance interval
        accept_lower, accept_upper = self._calculate_interval(step, accept_attributes, concept_lattice)
        
        # Calculate rejection interval
        reject_lower, reject_upper = self._calculate_interval(step, reject_attributes, concept_lattice)
        
        self.logger.info(
            f"Calculated intervals for step {step.get('step_num', '?')}: " +
            f"Accept: [{accept_lower:.2f}, {accept_upper:.2f}], " +
            f"Reject: [{reject_lower:.2f}, {reject_upper:.2f}]"
        )
        
        return (accept_lower, accept_upper), (reject_lower, reject_upper)
    
    def _calculate_interval(
        self, 
        step: Dict[str, Any], 
        attributes: List[Dict[str, Any]], 
        concept_lattice: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Calculate the interval (lower, upper) for a set of attributes.
        
        Args:
            step: The reasoning step
            attributes: List of attributes (either accept or reject type)
            concept_lattice: The concept lattice
            
        Returns:
            Tuple of (lower, upper) bounds for the interval
        """
        if not attributes:
            # If no attributes of this type, default to low confidence
            return 0.0, 0.2
        
        # Find the step's index
        step_idx = step.get("step_num", 0) - 1
        
        # Initialize with extreme values
        lower_bound = 1.0
        upper_bound = 0.0
        
        # For all concepts containing this step and at least one of these attributes
        for concept in concept_lattice:
            # Skip if this concept doesn't contain the step
            if step_idx not in concept.get("A", []):
                continue
                
            # Check if any of the attributes are in this concept
            attr_indices = concept.get("C", [])
            if not any(attr.get("id") in [a.get("id") for a in attributes] for attr in attr_indices):
                continue
                
            # Find the incidence values for each attribute under each condition
            for attr in attributes:
                attr_idx = next((i for i, a in enumerate(concept.get("M", [])) 
                              if a.get("id") == attr.get("id")), None)
                if attr_idx is None:
                    continue
                    
                for b_idx in concept.get("D", []):
                    # Get the incidence value I(step_idx, attr_idx, b_idx)
                    # This requires accessing the original triadic context
                    # As a simplification, use model confidence as proxy
                    incidence = self._get_incidence_value(step, attr, b_idx, concept_lattice)
                    
                    # Update bounds
                    lower_bound = min(lower_bound, incidence)
                    upper_bound = max(upper_bound, incidence)
        
        # If no matching concepts were found, use default values
        if lower_bound > upper_bound:
            if any(attr.get("type") == "accept" for attr in attributes):
                return 0.4, 0.6  # Default for accept
            else:
                return 0.2, 0.4  # Default for reject
        
        return lower_bound, upper_bound
    
    def _get_incidence_value(
        self, 
        step: Dict[str, Any], 
        attribute: Dict[str, Any], 
        condition_idx: int,
        concept_lattice: List[Dict[str, Any]]
    ) -> float:
        """
        Retrieve or estimate the incidence value I(g, m, b).
        
        In a full implementation, this would access the original triadic context.
        As a simplification, this uses heuristics based on the step and attribute type.
        
        Args:
            step: The reasoning step
            attribute: The attribute
            condition_idx: Index of the condition
            concept_lattice: The concept lattice
            
        Returns:
            Estimated incidence value in [0,1]
        """
        # If step has confidence, use it as a base
        base_confidence = step.get("confidence", 0.5)
        
        # Adjust based on attribute type
        if attribute.get("type") == "accept":
            # For accept attributes, use confidence directly
            return base_confidence
        else:
            # For reject attributes, use inverse of confidence
            return 1.0 - base_confidence
    
    def decide(
        self, 
        acceptance_interval: Tuple[float, float], 
        rejection_interval: Tuple[float, float]
    ) -> str:
        """
        Make a three-way decision based on acceptance and rejection intervals.
        
        Args:
            acceptance_interval: (lower, upper) bounds for acceptance
            rejection_interval: (lower, upper) bounds for rejection
            
        Returns:
            Decision: "accept", "reject", or "abstain"
        """
        accept_lower, accept_upper = acceptance_interval
        reject_lower, reject_upper = rejection_interval
        
        # Apply three-way decision rule
        if accept_lower >= self.alpha:
            return "accept"
        elif reject_lower >= self.beta:
            return "reject"
        else:
            return "abstain"
    
    def calculate_uncertainty(
        self, 
        acceptance_interval: Tuple[float, float], 
        rejection_interval: Tuple[float, float]
    ) -> float:
        """
        Calculate the uncertainty measure for a reasoning step.
        
        Args:
            acceptance_interval: (lower, upper) bounds for acceptance
            rejection_interval: (lower, upper) bounds for rejection
            
        Returns:
            Uncertainty measure in [0,2]
        """
        accept_lower, accept_upper = acceptance_interval
        reject_lower, reject_upper = rejection_interval
        
        # Uncertainty is the sum of interval widths
        uncertainty = (accept_upper - accept_lower) + (reject_upper - reject_lower)
        
        return uncertainty
