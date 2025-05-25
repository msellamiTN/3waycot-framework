"""
Three-Way Decision Maker for 3WayCoT Framework.

This module implements the Three-Way Decision Making process that classifies
reasoning steps into three regions: Positive, Negative, and Boundary (uncertain)
based on the analysis results from the Inverted TFCA and Uncertainty Resolver.
"""

import logging
import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

class ThreeWayDecisionMaker:
    """
    Implements the Three-Way Decision Making process for reasoning steps.
    
    The Three-Way Decision Making process classifies each reasoning step into one of three regions:
    1. Positive Region (Accept): High confidence in the step's validity
    2. Negative Region (Reject): High confidence in the step's invalidity
    3. Boundary Region (Defer): Uncertainty about the step's validity
    
    The classification is based on:
    - Confidence scores from the reasoning steps
    - Similarity measures from the Inverted TFCA
    - Uncertainty resolution results
    """
    
    def __init__(self, 
                 alpha: float = 0.7,  # Acceptance threshold (aligned with specifications)
                 beta: float = 0.6,   # Rejection threshold (aligned with specifications)
                 tau: float = 0.5     # Similarity threshold (aligned with specifications)
                ):
        """
        Initialize the Three-Way Decision Maker with specification-aligned thresholds.
        
        Args:
            alpha: Acceptance threshold (0.7 in specifications)
            beta: Rejection threshold (0.6 in specifications)
            tau: Similarity threshold for concept formation (0.5 in specifications)
        """
        self.alpha = alpha
        self.beta = beta
        self.tau = tau  # Renamed from gamma to tau for consistency with specifications
        self.logger = logging.getLogger("3WayCoT.ThreeWayDecisionMaker")
        
        # Validate thresholds with stricter constraints aligned with specifications
        if not (0 <= self.beta < self.alpha <= 1.0):
            raise ValueError("Thresholds must satisfy: 0 <= beta < alpha <= 1.0")
        if not (0 <= self.tau <= 1.0):
            raise ValueError("Tau (similarity threshold) must be between 0 and 1")
            
        # Additional check to ensure alpha + beta > 1 as per mathematical property
        if not (self.alpha + self.beta > 1.0):
            self.logger.warning(f"Warning: alpha ({alpha}) + beta ({beta}) <= 1.0. This violates the mathematical property required for sound three-way decisions.")
            # Auto-adjust to ensure compliance with mathematical property
            old_alpha, old_beta = self.alpha, self.beta
            self.alpha = max(self.alpha, 1.0 - self.beta + 0.05)  # Ensure alpha + beta > 1 with margin
            self.logger.warning(f"Auto-adjusted alpha from {old_alpha} to {self.alpha} to ensure alpha + beta > 1")

    
    def make_decisions(self, analysis: Dict[str, Any], uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make three-way decisions for each reasoning step.
        
        This enhanced version integrates confidence metrics more effectively
        and ensures proper propagation of information through the system.
        
        Args:
            analysis: Analysis results from Triadic FCA
            uncertainty_analysis: Uncertainty analysis results
            
        Returns:
            Dictionary with decisions for each reasoning step
        """
        self.logger.info("Making three-way decisions for reasoning steps using enhanced algorithm...")
        
        # Initialize results dictionary with enhanced metadata
        results = {
            'decisions': [],
            'summary': {},
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'thresholds': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'tau': self.tau
                },
                'algorithm_version': '2.0',  # Enhanced version indicator
                'confidence_weighted': True   # Flag indicating confidence weighting is active
            }
        }
        
        # Get reasoning steps from analysis
        reasoning_steps = analysis.get('reasoning_steps', [])
        if not reasoning_steps:
            self.logger.warning("No reasoning steps found in analysis. Cannot make decisions.")
            return results
        
        # ENHANCEMENT: Extract global confidence metrics if available
        confidence_metrics = analysis.get('confidence_metrics', {})
        if confidence_metrics:
            self.logger.info(f"Using enhanced confidence metrics: {confidence_metrics}")
            results['metadata']['confidence_metrics'] = confidence_metrics
        
        # Get confidence distribution from uncertainty analysis
        confidence_distribution = uncertainty_analysis.get('confidence_distribution', {})
        if confidence_distribution:
            self.logger.info(f"Using confidence distribution: {confidence_distribution}")
            results['metadata']['confidence_distribution'] = confidence_distribution
        
        # Get step-specific uncertainties
        step_uncertainties = uncertainty_analysis.get('step_uncertainties', [])
        
        # Make decision for each reasoning step
        decisions = []
        accept_count = 0
        reject_count = 0
        abstain_count = 0
        
        # ENHANCEMENT: Track average confidence by decision type for analysis
        confidence_by_decision = {
            'ACCEPT': [],
            'REJECT': [],
            'ABSTAIN': []
        }
        
        for i, step in enumerate(reasoning_steps):
            # Extract enhanced metrics for the step
            metrics = {
                'similarity_score': step.get('similarity_score', 0.7),  # Improved defaults
                'assumption_coverage': step.get('assumption_coverage', 0.7),
                'uncertainty_score': step.get('uncertainty_score', 0.3)
            }
            
            # ENHANCEMENT: Add stability metrics if available
            if 'stability' in step:
                metrics['stability'] = step['stability']
            if 'connectivity' in step:
                metrics['connectivity'] = step['connectivity']
            if 'density' in step:
                metrics['density'] = step['density']
            
            # ENHANCEMENT: Add membership degrees from confidence metrics if available
            if confidence_metrics and 'membership_degrees' in confidence_metrics:
                metrics['membership_degrees'] = confidence_metrics['membership_degrees']
            
            # ENHANCEMENT: Add confidence calibration if available
            if confidence_metrics and 'confidence_calibration' in confidence_metrics:
                metrics['confidence_calibration'] = confidence_metrics['confidence_calibration']
            
            # Get step-specific uncertainty information
            step_uncertainty = step_uncertainties[i] if i < len(step_uncertainties) else {}
            
            # Extract confidence from step with better default
            confidence = step.get('confidence', 0.7)  # Default to higher confidence when not specified
            
            # ENHANCEMENT: Log confidence value for debugging
            self.logger.debug(f"Step {i} confidence: {confidence}")
            
            # Make the decision with enhanced metrics and uncertainty information
            decision, explanation = self._make_decision(
                metrics=metrics,
                confidence=confidence,
                uncertainty={
                    'lattice_metrics': uncertainty_analysis.get('lattice_metrics', {}),
                    'membership_degrees': step.get('membership_degrees', {}),
                    'step_uncertainty': step_uncertainty
                }
            )
            
            # Update decision counts
            if decision == "ACCEPT":
                accept_count += 1
                confidence_by_decision['ACCEPT'].append(confidence)
            elif decision == "REJECT":
                reject_count += 1
                confidence_by_decision['REJECT'].append(confidence)
            else:  # ABSTAIN
                abstain_count += 1
                confidence_by_decision['ABSTAIN'].append(confidence)
            
            # Add enhanced decision information to results
            decisions.append({
                'step_index': i,
                'step_content': step.get('content', ''),
                'decision': decision,
                'confidence': confidence,
                'explanation': explanation,
                'metrics': metrics,
                'membership_scores': {  # ENHANCEMENT: Include membership scores in output
                    'accept': self._calculate_accept_membership(confidence),
                    'reject': self._calculate_reject_membership(confidence),
                    'abstain': self._calculate_abstain_membership(confidence)
                }
            })
        
        # Calculate enhanced summary statistics
        total_steps = len(reasoning_steps)
        summary = {
            'total_steps': total_steps,
            'accept_count': accept_count,
            'reject_count': reject_count,
            'abstain_count': abstain_count,
            'accept_ratio': accept_count / total_steps if total_steps > 0 else 0.0,
            'reject_ratio': reject_count / total_steps if total_steps > 0 else 0.0,
            'abstain_ratio': abstain_count / total_steps if total_steps > 0 else 0.0,
            # ENHANCEMENT: Average confidence by decision type
            'avg_confidence_accept': sum(confidence_by_decision['ACCEPT']) / len(confidence_by_decision['ACCEPT']) if confidence_by_decision['ACCEPT'] else 0.0,
            'avg_confidence_reject': sum(confidence_by_decision['REJECT']) / len(confidence_by_decision['REJECT']) if confidence_by_decision['REJECT'] else 0.0,
            'avg_confidence_abstain': sum(confidence_by_decision['ABSTAIN']) / len(confidence_by_decision['ABSTAIN']) if confidence_by_decision['ABSTAIN'] else 0.0,
            # ENHANCEMENT: Decision quality metrics
            'decision_consistency': self._calculate_decision_consistency(decisions),
            'confidence_alignment': self._calculate_confidence_alignment(decisions)
        }
        
        # Update results dictionary
        results['decisions'] = decisions
        results['summary'] = summary
        
        self.logger.info(f"Enhanced decision summary: {summary}")
        return results
        
    def _calculate_decision_consistency(self, decisions: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency of decisions based on transitions between consecutive steps.
        
        Args:
            decisions: List of decision dictionaries
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if len(decisions) <= 1:
            return 1.0  # Single decision is always consistent
            
        # Count consistent transitions (same decision type in consecutive steps)
        consistent_transitions = 0
        for i in range(1, len(decisions)):
            if decisions[i]['decision'] == decisions[i-1]['decision']:
                consistent_transitions += 1
                
        # Calculate consistency ratio
        return consistent_transitions / (len(decisions) - 1) if len(decisions) > 1 else 1.0
    
    def _calculate_confidence_alignment(self, decisions: List[Dict[str, Any]]) -> float:
        """
        Calculate how well the decisions align with confidence values.
        
        Args:
            decisions: List of decision dictionaries
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not decisions:
            return 0.0
            
        # Check alignment of each decision with its confidence value
        alignment_scores = []
        for decision in decisions:
            confidence = decision.get('confidence', 0.5)
            decision_type = decision.get('decision', 'ABSTAIN')
            
            # Calculate alignment score based on decision type and confidence
            if decision_type == 'ACCEPT' and confidence >= self.alpha:
                # High confidence ACCEPT - good alignment
                alignment_scores.append(min(1.0, confidence / self.alpha))
            elif decision_type == 'REJECT' and confidence <= self.beta:
                # Low confidence REJECT - good alignment
                alignment_scores.append(min(1.0, (1.0 - confidence) / (1.0 - self.beta)))
            elif decision_type == 'ABSTAIN' and self.beta < confidence < self.alpha:
                # Mid confidence ABSTAIN - good alignment
                # Calculate how centered the confidence is in the abstain region
                abstain_center = (self.alpha + self.beta) / 2
                distance_from_center = abs(confidence - abstain_center)
                abstain_range = (self.alpha - self.beta) / 2
                alignment_scores.append(1.0 - (distance_from_center / abstain_range if abstain_range > 0 else 0.0))
            else:
                # Decision doesn't align well with confidence
                alignment_scores.append(0.2)  # Some minimum alignment
                
        # Return average alignment score
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def decide_step(self, 
                    step: Dict[str, Any], 
                    analysis: Dict[str, Any], 
                    uncertainty_analysis: Dict[str, Any],
                    step_index: int) -> Dict[str, Any]:
        """
        Make a three-way decision for a single reasoning step.
        
        Args:
            step: The reasoning step to evaluate
            analysis: Results from the Inverted TFCA analysis
            uncertainty_analysis: Results from the Uncertainty Resolver
            step_index: Index of the step in the analysis
            
        Returns:
            Decision dictionary with 'decision', 'confidence', and 'explanation'
        """
        # Extract dynamic confidence from the step
        # Use the 'confidence' key that is populated from the CoT generator
        confidence = step.get('confidence', step.get('original_confidence', 0.7))  # Get dynamically extracted confidence
        assumptions = step.get('assumptions', [])
        
        # Get uncertainty information if available
        uncertainty = uncertainty_analysis.get('step_uncertainties', [{}] * (step_index + 1))[step_index]
        
        # Calculate decision metrics
        metrics = self._calculate_decision_metrics(step, analysis, uncertainty, step_index)
        
        # Make the three-way decision
        decision, explanation = self._make_decision(metrics, confidence, uncertainty)
        
        return {
            'step_num': step.get('step_num', step_index + 1),
            'decision': decision,
            'confidence': confidence,
            'metrics': metrics,
            'explanation': explanation,
            'assumptions': assumptions,
            'uncertainty': uncertainty
        }
    
    def _calculate_decision_metrics(self, 
                                  step: Dict[str, Any], 
                                  analysis: Dict[str, Any], 
                                  uncertainty: Dict[str, Any],
                                  step_index: int) -> Dict[str, float]:
        """
        Calculate metrics used for the three-way decision.
        
        Args:
            step: The reasoning step
            analysis: Results from the Inverted TFCA analysis
            uncertainty: Uncertainty information for the step
            step_index: Index of the step in the analysis
            
        Returns:
            Dictionary of metrics used for decision making
        """
        # Extract relevant metrics from analysis
        metrics = {}
        
        # Extract confidence from the step - CRITICAL COMPONENT
        # This confidence value now has higher weight in decision calculations
        confidence = step.get('original_confidence', 0.5)
        metrics['confidence'] = confidence
        
        # Log the confidence value being used
        self.logger.info(f"Step {step_index+1}: Using confidence value: {confidence:.2f}")
        
        # Get concept mappings for this step
        concepts = analysis.get('concepts', [])
        concept_mappings = []
        
        for i, concept in enumerate(concepts):
            # If this concept includes this step in its intent (attributes)
            intent = concept.get('C', [])
            
            # Fix for the intent format - handle both tuple/list and integer formats
            contains_step = False
            for attr in intent:
                if isinstance(attr, (list, tuple)) and len(attr) > 0 and attr[0] == step_index:
                    contains_step = True
                    break
                elif isinstance(attr, int) and attr == step_index:
                    contains_step = True
                    break
            
            if contains_step:
                mapping = {
                    'concept_index': i,
                    'similarity': concept.get('similarity', 0.5),
                    'stability': concept.get('stability', 0.5)
                }
                concept_mappings.append(mapping)
        
        # Get the best concept mapping (if any)
        if concept_mappings:
            best_mapping = max(concept_mappings, key=lambda x: x['similarity'])
            metrics['concept_similarity'] = best_mapping['similarity']
            metrics['concept_stability'] = best_mapping.get('stability', 0.5)
        else:
            metrics['concept_similarity'] = 0.5
            metrics['concept_stability'] = 0.5
            
        # Extract uncertainty information
        uncertainty_score = 0.5  # Default midpoint
        
        if uncertainty:
            # Extract from uncertainty resolution results
            if isinstance(uncertainty, dict):
                step_uncertainty = next(
                    (u for u in uncertainty.get('uncertainty_scores', []) if u.get('step_index') == step_index), 
                    None
                )
                if step_uncertainty:
                    uncertainty_score = step_uncertainty.get('score', 0.5)
                    
        metrics['uncertainty'] = uncertainty_score
        
        # Calculate membership degrees for the three regions
        membership_degrees = {}
        lattice_analysis = analysis.get('lattice_analysis', {})
        
        # Get metrics from lattice analysis if available
        connectivity = lattice_analysis.get('connectivity', {}).get('connectivity_ratio', 0.5)
        density = lattice_analysis.get('density', 0.5)
        coverage = lattice_analysis.get('concept_distribution', {}).get('coverage', 0.5)
        
        # Store additional metrics
        metrics['connectivity'] = connectivity
        metrics['density'] = density
        metrics['coverage'] = coverage
        
        # Get membership degrees for the three regions - this is the core of the three-way decision process
        # ENHANCED: Give more weight to confidence value in membership calculations
        membership_result = self._calculate_memberships(
            confidence,  # Use the confidence value directly with higher weight
            metrics['concept_similarity'],
            metrics['concept_stability'],
            metrics['uncertainty'],
            metrics['connectivity'],
            metrics['density'],
            metrics['coverage']
        )
        
        metrics['membership_degrees'] = membership_result
        
        return metrics
    
    def _calculate_memberships(self, 
                              confidence: float, 
                              concept_similarity: float, 
                              concept_stability: float, 
                              uncertainty: float, 
                              connectivity: float, 
                              density: float, 
                              coverage: float) -> Dict[str, float]:
        """
        Calculate membership degrees for the three regions.
        
        Args:
            confidence: Confidence score of the step
            concept_similarity: Similarity to other concepts
            concept_stability: Stability of the concept
            uncertainty: Uncertainty score
            connectivity: Connectivity ratio
            density: Density of the concept
            coverage: Coverage of the concept
            
        Returns:
            Dictionary of membership degrees for the three regions
        """
        # Calculate membership degrees for the three regions
        membership_degrees = {}
        
        # Calculate membership degrees for the three regions
        # ENHANCED: Give more weight to confidence value in membership calculations
        membership_degrees['accept'] = 0.6 * confidence + 0.2 * concept_similarity + 0.1 * concept_stability + 0.1 * density
        membership_degrees['reject'] = 0.6 * (1.0 - confidence) + 0.2 * uncertainty + 0.1 * (1.0 - concept_similarity) + 0.1 * (1.0 - coverage)
        membership_degrees['abstain'] = 0.6 * abs(0.5 - confidence) + 0.2 * (1.0 - density) + 0.1 * (1.0 - connectivity) + 0.1 * uncertainty
        
        return membership_degrees
    
    def _make_decision(self, 
                       metrics: Dict[str, float], 
                       confidence: float, 
                       uncertainty: Dict[str, Any]) -> Tuple[str, str]:
        """
        Make the three-way decision based on the calculated metrics.
        
        This implements an enhanced three-way decision process where:
        - ACCEPT: High confidence in the step's validity
        - ABSTAIN: Uncertain about the step's validity (requires more information)
        - REJECT: High confidence in the step's invalidity
        
        The enhanced algorithm prioritizes confidence values in accordance with
        the mathematical properties of three-way decisions defined in the specifications.
        
        Args:
            metrics: Calculated decision metrics
            confidence: Confidence score of the step
            uncertainty: Uncertainty information
            
        Returns:
            Tuple of (decision, explanation)
        """
        # Extract standard metrics with improved defaults
        similarity = metrics.get('similarity_score', 0.7)  # Default to higher similarity when not available
        coverage = metrics.get('assumption_coverage', 0.7)  # Default to higher coverage when not available
        uncertainty_score = metrics.get('uncertainty_score', 0.3)  # Default to lower uncertainty when not available
        
        # Extract enhanced metrics if available
        stability = metrics.get('stability', 0.7)  # Stability of reasoning (new metric)
        connectivity = metrics.get('connectivity', 0.6)  # Connectivity between concepts (new metric)
        density = metrics.get('density', 0.6)  # Density of relationships (new metric)
        
        # Get triadic membership degrees - enhanced for more accurate decision-making
        membership_degrees = uncertainty.get('membership_degrees', {})
        
        # Initialize scores for each decision region
        accept_score = 0.0
        reject_score = 0.0
        abstain_score = 0.0
        
        # ENHANCEMENT: Direct confidence-based calculation first
        # Calculate the primary membership degrees directly from confidence
        # This ensures confidence has the highest impact on decision-making
        confidence_accept = self._calculate_accept_membership(confidence)
        confidence_reject = self._calculate_reject_membership(confidence)
        confidence_abstain = self._calculate_abstain_membership(confidence)
        
        # Calculate enhanced membership degrees with a weighted approach
        if 'membership_degrees' in metrics:
            # Use membership degrees from metrics with higher confidence weight
            membership = metrics['membership_degrees']
            accept_score = 0.7 * confidence_accept + 0.3 * membership.get('accept', 0.0)
            reject_score = 0.7 * confidence_reject + 0.3 * membership.get('reject', 0.0)
            abstain_score = 0.7 * confidence_abstain + 0.3 * membership.get('abstain', 0.0)
        elif membership_degrees and 'accept' in membership_degrees and 'reject' in membership_degrees:
            # Use available membership degrees with confidence prioritization
            if 'abstain' in membership_degrees:
                # Full triadic membership
                accept_score = 0.6 * confidence_accept + 0.4 * membership_degrees['accept'].get('degree', 0.0)
                reject_score = 0.6 * confidence_reject + 0.4 * membership_degrees['reject'].get('degree', 0.0)
                abstain_score = 0.6 * confidence_abstain + 0.4 * membership_degrees['abstain'].get('degree', 0.0)
            else:
                # Binary membership with bounds
                accept_lower = membership_degrees['accept'].get('lower', 0.0)
                accept_upper = membership_degrees['accept'].get('upper', 0.0)
                reject_lower = membership_degrees['reject'].get('lower', 0.0)
                reject_upper = membership_degrees['reject'].get('upper', 0.0)
                
                # Calculate with confidence prioritization
                accept_score = 0.6 * confidence_accept + 0.4 * ((accept_lower + accept_upper) / 2)
                reject_score = 0.6 * confidence_reject + 0.4 * ((reject_lower + reject_upper) / 2)
                
                # Abstain score incorporates boundary region size
                boundary_size = (accept_upper - accept_lower) + (reject_upper - reject_lower)
                abstain_score = 0.5 * confidence_abstain + 0.5 * boundary_size
        else:
            # Direct confidence-based approach with additional metrics
            accept_score = 0.6 * confidence_accept + 0.2 * similarity + 0.1 * coverage + 0.1 * stability
            reject_score = 0.6 * confidence_reject + 0.2 * uncertainty_score + 0.1 * (1.0 - similarity) + 0.1 * (1.0 - coverage)
            abstain_score = 0.6 * confidence_abstain + 0.2 * (1.0 - density) + 0.1 * (1.0 - connectivity) + 0.1 * uncertainty_score
        
        # ENHANCEMENT: Add confidence calibration to ensure scores reflect uncertainty properly
        # This accounts for potential overconfidence or underconfidence in the reasoning
        if 'confidence_calibration' in metrics:
            calibration = metrics['confidence_calibration']
            accept_score *= calibration.get('accept_factor', 1.0)
            reject_score *= calibration.get('reject_factor', 1.0)
            abstain_score *= calibration.get('abstain_factor', 1.0)
            
        # Log scores for debugging before normalization
        self.logger.debug(f"Pre-normalization scores: accept={accept_score:.3f}, reject={reject_score:.3f}, abstain={abstain_score:.3f}")
            
        # Normalize scores to sum to 1.0
        total = accept_score + reject_score + abstain_score
        if total > 0:
            accept_score /= total
            reject_score /= total
            abstain_score /= total
        
        # Log final normalized scores
        self.logger.info(f"Membership degrees: accept={accept_score:.3f}, reject={reject_score:.3f}, abstain={abstain_score:.3f}, confidence={confidence:.3f}")
            
        # Make the three-way decision
        decision = "ABSTAIN"  # Default decision
        explanation = ""
        
        # ENHANCEMENT: Strict confidence-based decision rules first
        # This ensures confidence has the highest priority in the decision-making process
        if confidence >= self.alpha:
            # High confidence directly leads to ACCEPT
            decision = "ACCEPT"
            explanation = f"High confidence in step validity (confidence={confidence:.3f}, accept={accept_score:.3f}, reject={reject_score:.3f})"
        elif confidence <= self.beta:
            # Low confidence directly leads to REJECT
            decision = "REJECT"
            explanation = f"Low confidence in step validity (confidence={confidence:.3f}, reject={reject_score:.3f}, accept={accept_score:.3f})"
        else:
            # If confidence alone doesn't decide, use the membership scores
            max_score = max(accept_score, reject_score, abstain_score)
            
            if max_score == accept_score and accept_score >= self.alpha:
                decision = "ACCEPT"
                explanation = f"High membership in positive region (accept={accept_score:.3f}, confidence={confidence:.3f})"
            elif max_score == reject_score and reject_score >= (1.0 - self.beta):
                decision = "REJECT"
                explanation = f"High membership in negative region (reject={reject_score:.3f}, confidence={confidence:.3f})"
            else:
                decision = "ABSTAIN"
                explanation = f"Uncertain about step validity (abstain={abstain_score:.3f}, confidence={confidence:.3f})"
                
        return decision, explanation
        
    def _calculate_accept_membership(self, confidence: float) -> float:
        """
        Calculate accept membership based on confidence value.
        Uses a sigmoidal function centered around the alpha threshold.
        
        Args:
            confidence: Confidence value [0,1]
            
        Returns:
            Accept membership degree [0,1]
        """
        # Sigmoidal function centered at alpha with steepness 10
        if confidence >= self.alpha:
            return 0.5 + 0.5 * min(1.0, (confidence - self.alpha) * 5)
        else:
            # Sharp drop below alpha
            return max(0.0, 0.5 * (confidence / self.alpha))
    
    def _calculate_reject_membership(self, confidence: float) -> float:
        """
        Calculate reject membership based on confidence value.
        Uses an inverse sigmoidal function centered around the beta threshold.
        
        Args:
            confidence: Confidence value [0,1]
            
        Returns:
            Reject membership degree [0,1]
        """
        # Inverse sigmoidal function centered at beta
        if confidence <= self.beta:
            return 0.5 + 0.5 * min(1.0, (self.beta - confidence) * 5)
        else:
            # Sharp drop above beta
            return max(0.0, 0.5 * (1.0 - (confidence - self.beta) / (1.0 - self.beta)))
    
    def _calculate_abstain_membership(self, confidence: float) -> float:
        """
        Calculate abstain membership based on confidence value.
        Highest when confidence is in the boundary region between beta and alpha.
        
        Args:
            confidence: Confidence value [0,1]
            
        Returns:
            Abstain membership degree [0,1]
        """
        # Triangular function peaking in the middle of the boundary region
        boundary_mid = (self.alpha + self.beta) / 2.0
        boundary_width = self.alpha - self.beta
        
        if boundary_width <= 0:
            # If boundary region doesn't exist (alpha <= beta), use distance from extremes
            return 1.0 - 2.0 * abs(confidence - 0.5)
        
        # Distance from the middle of the boundary region, normalized
        distance = abs(confidence - boundary_mid) / (boundary_width / 2.0)
        
        # Triangular function: 1 at middle, 0 at boundaries
        return max(0.0, 1.0 - distance)
