"""
Uncertainty Resolution Module.

This module implements the Uncertainty Resolution Module for the 3WayCoT framework,
which addresses reasoning steps with high uncertainty through retrieval and backtracking.
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional, Set

class UncertaintyResolver:
    """
    Resolves uncertainty in reasoning steps using retrieval and backtracking.
    
    This class implements the Uncertainty Resolution Module as described in Algorithm 5
    of the 3WayCoT framework specifications.
    """
    
    def __init__(
        self, 
        knowledge_base,
        cot_generator,
        relevance_threshold: float = 0.7,
        validity_threshold: float = 0.6
    ):
        """
        Initialize the Uncertainty Resolution Module.
        
        Args:
            knowledge_base: Knowledge base for retrieving evidence
            cot_generator: Chain-of-Thought Generator for reformulating steps
            relevance_threshold: Threshold for relevant evidence
            validity_threshold: Threshold for assumption validity
        """
        self.knowledge_base = knowledge_base
        self.cot_generator = cot_generator
        self.relevance_threshold = relevance_threshold
        self.validity_threshold = validity_threshold
        self.logger = logging.getLogger("3WayCoT.UncertaintyResolver")
        
        # Log the initialization
        self.logger.info(f"Initialized UncertaintyResolver with relevance_threshold={relevance_threshold}, "
                        f"validity_threshold={validity_threshold}")
        if cot_generator:
            self.logger.info(f"Using CoT Generator: {cot_generator.__class__.__name__}")
        else:
            self.logger.warning("No CoT Generator provided. Some uncertainty resolution features may be limited.")
    
    def resolve(
        self, 
        uncertain_step: Dict[str, Any], 
        all_steps: List[Dict[str, Any]], 
        triadic_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve uncertainty in a reasoning step using multiple strategies.
        
        The resolution process follows these steps:
        1. First, try to reformulate the step using the CoT generator
        2. If that fails, try retrieval-augmented generation (RAG)
        3. If RAG fails, try dependency-based backtracking
        4. If all else fails, mark the step as explicitly uncertain
        
        Args:
            uncertain_step: The uncertain reasoning step
            all_steps: All reasoning steps in the chain
            triadic_context: The triadic context
            
        Returns:
            Resolved step or original step if no resolution was possible
        """
        step_num = uncertain_step.get('step_num', '?')
        self.logger.info(f"Resolving uncertainty for step: {step_num}")
        
        # Track resolution attempts
        resolution_attempts = uncertain_step.get("uncertainty", {}).get("resolution_attempts", 0)
        if resolution_attempts >= 3:  # Prevent infinite loops
            self.logger.warning(f"Step {step_num} has reached maximum resolution attempts, marking as uncertain")
            return self.mark_as_uncertain(uncertain_step)
            
        # Strategy 0: Try to reformulate the step using CoT
        # Convert step_num to int for comparison if it's a string
        try:
            current_step_num = int(step_num) if isinstance(step_num, str) and step_num.isdigit() else step_num
            previous_steps = [s for s in all_steps if int(s.get('step_num', -1)) < current_step_num]
            reformulated_step = self.reformulate_with_cot(uncertain_step, previous_steps)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error comparing step numbers: {e}")
            previous_steps = []
            reformulated_step = None
        
        # If reformulation produced a different result, use it
        if reformulated_step and reformulated_step.get('reasoning') != uncertain_step.get('reasoning'):
            self.logger.info("Successfully reformulated step using CoT")
            reformulated_step["uncertainty"] = {
                "resolved": True,
                "resolution_attempts": resolution_attempts + 1,
                "resolution_strategy": "cot_reformulation"
            }
            return reformulated_step
        
        # Strategy 1: Retrieval-Augmented Generation (RAG)
        resolved_step = self.rag_retriever(uncertain_step)
        if resolved_step:
            self.logger.info("Resolved step using RAG retrieval")
            resolved_step["uncertainty"] = {
                "resolved": True,
                "resolution_attempts": resolution_attempts + 1,
                "resolution_strategy": "rag_retrieval"
            }
            return resolved_step
        
        # Strategy 2: Dependency-based backtracking
        dependencies = self.dependency_analysis(uncertain_step, all_steps)
        if dependencies:
            self.logger.info(f"Found {len(dependencies)} dependencies for step {step_num}")
            
            # Try to resolve using backtracking
            resolved_steps = self.backtrack(dependencies, uncertain_step, triadic_context)
            if resolved_steps and len(resolved_steps) > 0:
                self.logger.info("Resolved step using dependency backtracking")
                resolved_step = resolved_steps[-1]
                resolved_step["uncertainty"] = {
                    "resolved": True,
                    "resolution_attempts": resolution_attempts + 1,
                    "resolution_strategy": "dependency_backtracking"
                }
                return resolved_step
        
        # If no resolution possible, mark the step as explicitly uncertain
        self.logger.info("Could not resolve uncertainty, marking step as explicitly uncertain")
        return self.mark_as_uncertain(uncertain_step)
    
    def rag_retriever(self, uncertain_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to resolve uncertainty using Retrieval-Augmented Generation.
        
        Args:
            uncertain_step: The uncertain reasoning step
            
        Returns:
            Resolved step or None if no resolution was possible
        """
        # Formulate a query from the uncertain step
        query = self.formulate_query(uncertain_step)
        
        # Check if knowledge base exists and can be queried
        if not self.knowledge_base or not hasattr(self.knowledge_base, "retrieve_evidence"):
            self.logger.warning("Knowledge base not available or missing retrieve_evidence method")
            return None
        
        try:
            # Retrieve evidence from knowledge base
            evidence = self.knowledge_base.retrieve_evidence(query)
            
            # Check if evidence is relevant
            relevance_score = self.compute_relevance(evidence, uncertain_step)
            
            if relevance_score >= self.relevance_threshold:
                # Reformulate step with evidence
                resolved_step = self.reformulate_step(uncertain_step, evidence)
                return resolved_step
            else:
                self.logger.info(f"Retrieved evidence not relevant enough (score: {relevance_score:.2f})")
                return None
        except Exception as e:
            self.logger.warning(f"Error in RAG retrieval: {e}")
            return None
    
    def formulate_query(self, step: Dict[str, Any]) -> str:
        """
        Formulate a search query from an uncertain reasoning step.
        
        Args:
            step: The uncertain reasoning step
            
        Returns:
            Search query string
        """
        # Extract key terms from the step
        reasoning = step.get("reasoning", "")
        assumptions = step.get("assumptions", "")
        
        # Remove common stop words and punctuation
        query_text = f"{reasoning} {assumptions}"
        query_text = re.sub(r'[^\w\s]', ' ', query_text)
        
        # Extract key noun phrases and entities
        # In a full implementation, this would use NLP techniques
        query_terms = [term for term in query_text.split() if len(term) > 3]
        
        # Limit query length
        query = " ".join(query_terms[:10])
        
        return query
    
    def compute_relevance(self, evidence: Any, step: Dict[str, Any]) -> float:
        """
        Compute the relevance of retrieved evidence to the uncertain step.
        
        Args:
            evidence: The retrieved evidence
            step: The uncertain reasoning step
            
        Returns:
            Relevance score in [0,1]
        """
        # In a full implementation, this would use semantic similarity
        # For simplicity, return a moderate score
        return 0.65
    
    def reformulate_step(self, step: Dict[str, Any], evidence: Any) -> Dict[str, Any]:
        """
        Reformulate a reasoning step using retrieved evidence.
        
        Args:
            step: The uncertain reasoning step
            evidence: The retrieved evidence
            
        Returns:
            Reformulated step
        """
        # Create a copy of the original step
        resolved_step = step.copy()
        
        # Add evidence to the reasoning
        evidence_str = str(evidence)[:100]  # Truncate for brevity
        original_reasoning = step.get("reasoning", "")
        
        resolved_step["reasoning"] = f"{original_reasoning} [Enhanced with evidence: {evidence_str}]"
        
        # Update confidence
        resolved_step["confidence"] = min(0.8, step.get("confidence", 0.5) + 0.2)
        
        # Update assumptions to reflect the new evidence
        original_assumptions = step.get("assumptions", "")
        resolved_step["assumptions"] = f"{original_assumptions} [Supported by external evidence]"
        
        # Update extracted assumptions
        if "extracted_assumptions" in step:
            resolved_step["extracted_assumptions"] = step["extracted_assumptions"] + ["Assumption supported by evidence"]
        
        # Update annotated reasoning
        resolved_step["annotated_reasoning"] = f"{resolved_step['reasoning']}\n[Assumptions: {resolved_step['assumptions']}]"
        
        return resolved_step
    
    def dependency_analysis(
        self, 
        uncertain_step: Dict[str, Any], 
        all_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze dependencies between reasoning steps.
        
        Args:
            uncertain_step: The uncertain reasoning step
            all_steps: All reasoning steps in the chain
            
        Returns:
            List of steps that the uncertain step depends on
        """
        # Get the step number (if available)
        step_num = uncertain_step.get("step_num", 0)
        
        # Simple heuristic: steps depend on all previous steps
        dependencies = [step for step in all_steps if step.get("step_num", 0) < step_num]
        
        # Sort by step number (most recent first)
        dependencies.sort(key=lambda x: x.get("step_num", 0), reverse=True)
        
        return dependencies
    
    def backtrack(
        self, 
        dependencies: List[Dict[str, Any]], 
        uncertain_step: Dict[str, Any], 
        triadic_context: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Attempt to resolve uncertainty by backtracking to earlier steps.
        
        Args:
            dependencies: Steps that the uncertain step depends on
            uncertain_step: The uncertain reasoning step
            triadic_context: The triadic context
            
        Returns:
            List of revised steps or None if no resolution was possible
        """
        for dep in dependencies:
            # Extract assumptions from the dependency
            assumptions = dep.get("extracted_assumptions", [])
            
            for assumption in assumptions:
                # Validate the assumption
                validity = self.validate_assumption(assumption, triadic_context)
                
                if validity < self.validity_threshold:
                    # Revise the step with a corrected assumption
                    revised_dep = self.revise_step(dep, assumption)
                    
                    # Propagate the revision to the uncertain step
                    revised_uncertain = self.propagate_revision(revised_dep, uncertain_step)
                    
                    return [revised_dep, revised_uncertain]
        
        return None
    
    def validate_assumption(self, assumption: str, triadic_context: Dict[str, Any]) -> float:
        """
        Validate an assumption against the triadic context.
        
        Args:
            assumption: The assumption to validate
            triadic_context: The triadic context
            
        Returns:
            Validity score in [0,1]
        """
        # In a full implementation, this would check the assumption against a knowledge base
        # For simplicity, return a moderate validity
        return 0.55
    
    def revise_step(self, step, assumption):
        """
        Revise a step by incorporating a new assumption.
        
        Args:
            step: The reasoning step to revise
            assumption: The assumption to incorporate
            
        Returns:
            Revised reasoning step with updated confidence
        """
        revised_step = dict(step)  # Create a copy to avoid modifying the original
        
        # Add the new assumption to the step
        original_assumptions = step.get("assumptions", "")
        
        # Handle different assumption formats
        if isinstance(original_assumptions, list):
            # If assumptions is a list, append the new assumption
            if assumption not in original_assumptions:
                revised_step["assumptions"] = original_assumptions + [assumption]
        elif isinstance(original_assumptions, str):
            # If assumptions is a string, append the new assumption with proper formatting
            if original_assumptions:
                revised_step["assumptions"] = original_assumptions.replace(
                    "Assumptions:", f"Assumptions:\n- {assumption}"
                )
            else:
                revised_step["assumptions"] = f"Assumptions:\n- {assumption}"
        else:
            # Default case - create a new list with the assumption
            revised_step["assumptions"] = [assumption]
        
        # Update the reasoning to reflect the corrected assumption
        original_reasoning = step.get("reasoning", "")
        revised_step["reasoning"] = f"{original_reasoning} [Revised]"
        
        # Handle different assumption formats
        original_assumptions = step.get("assumptions", "")
        if isinstance(original_assumptions, list):
            # If assumptions is a list, update it properly
            revised_assumptions = []
            for assump in original_assumptions:
                if assump == assumption:
                    revised_assumptions.append(f"Corrected: {assumption}")
                else:
                    revised_assumptions.append(assump)
            
            # Add the assumption if it's not already in the list
            if assumption not in original_assumptions:
                revised_assumptions.append(assumption)
                
            revised_step["assumptions"] = revised_assumptions
        elif isinstance(original_assumptions, str):
            # If assumptions is a string, update it properly
            if original_assumptions:
                if assumption in original_assumptions:
                    revised_step["assumptions"] = original_assumptions.replace(
                        assumption, f"Corrected: {assumption}"
                    )
                else:
                    revised_step["assumptions"] = f"{original_assumptions}\n- {assumption}"
            else:
                revised_step["assumptions"] = f"Assumptions:\n- {assumption}"
        else:
            # Default case - create a new list with the assumption
            revised_step["assumptions"] = [assumption]
        
        # Update annotated reasoning
        revised_step["annotated_reasoning"] = f"{revised_step['reasoning']}\n[Assumptions: {revised_step['assumptions']}]"
        
        return revised_step
    
    def propagate_revision(self, revised_dependency: Dict[str, Any], uncertain_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagate dependency revision to the uncertain step.
        
        Args:
            revised_dependency: The revised dependency
            uncertain_step: The uncertain step
            
        Returns:
            Updated uncertain step
        """
        # Create a copy of the uncertain step
        propagated_step = uncertain_step.copy()
        
        # Update the step to reflect the revised dependency
        dependency_num = revised_dependency.get("step_num", 0)
        original_reasoning = uncertain_step.get("reasoning", "")
        
        propagated_step["reasoning"] = f"{original_reasoning} [Updated based on revision to Step {dependency_num}]"
        
        # Increase confidence
        propagated_step["confidence"] = min(0.75, uncertain_step.get("confidence", 0.5) + 0.15)
        
        # Update assumptions to reflect the dependency revision
        original_assumptions = uncertain_step.get("assumptions", "")
        propagated_step["assumptions"] = f"{original_assumptions} [Updated with revised assumptions from Step {dependency_num}]"
        
        # Update annotated reasoning
        propagated_step["annotated_reasoning"] = f"{propagated_step['reasoning']}\n[Assumptions: {propagated_step['assumptions']}]"
        
        return propagated_step
    
    def reformulate_with_cot(self, step: Dict[str, Any], context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reformulate a reasoning step using the Chain-of-Thought Generator.
        
        Args:
            step: The step to reformulate
            context: Optional context from previous steps
            
        Returns:
            Reformulated step or original step if reformulation fails
        """
        if not self.cot_generator:
            self.logger.warning("No CoT Generator available for reformulation")
            return step
            
        try:
            # Extract the reasoning text and any assumptions
            reasoning = step.get('reasoning', '')
            assumptions = step.get('assumptions', [])
            
            # Create a context string from previous steps if provided
            context_str = ""
            if context:
                context_str = "\n".join([
                    f"Step {s.get('step_num', i+1)}: {s.get('reasoning', '')}" 
                    for i, s in enumerate(context)
                ])
            
            # Check which parameters the CoT generator accepts
            import inspect
            cot_params = inspect.signature(self.cot_generator.generate).parameters
            generate_kwargs = {
                'query': reasoning,
                'context': context_str,
            }
            
            # Only add parameters that are supported by the CoT generator
            if 'max_steps' in cot_params:
                generate_kwargs['max_steps'] = 1  # Only reformulate this single step
            
            if 'max_assumptions' in cot_params:
                generate_kwargs['max_assumptions'] = getattr(self.cot_generator, 'max_assumptions', None)
            
            # Use the CoT generator with compatible parameters
            reformulated = self.cot_generator.generate(**generate_kwargs)
            
            if reformulated and 'steps' in reformulated and len(reformulated['steps']) > 0:
                # Take the first (and only) reformulated step
                new_step = reformulated['steps'][0]
                self.logger.info("Successfully reformulated reasoning step")
                return new_step
                
        except Exception as e:
            self.logger.error(f"Error during CoT reformulation: {str(e)}", exc_info=True)
            
        return step
        
    def mark_as_uncertain(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mark a step as explicitly uncertain.
        
        Args:
            step: The step to mark
            
        Returns:
            The marked step
        """
        marked_step = step.copy()
        marked_step['status'] = 'uncertain'
        marked_step['resolved'] = False
        
        # Update annotated reasoning
        marked_step["annotated_reasoning"] = f"{marked_step['reasoning']}\n[Assumptions: {marked_step['assumptions']}]"
        
        # Add uncertainty metadata
        marked_step["uncertainty"] = {
            "resolved": False,
            "resolution_attempts": step.get("uncertainty", {}).get("resolution_attempts", 0) + 1,
            "resolution_strategy": "marked_as_uncertain"
        }
        
        self.logger.warning(f"Marked step {step.get('step_num', '?')} as explicitly uncertain")
        return marked_step
