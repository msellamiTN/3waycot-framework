"""
Inverted Triadic Fuzzy Formal Concept Analysis for Chain of Thought Reasoning

This module extends the original TFCA by inverting the roles of objects and attributes
to better handle scenarios where there are many assumptions but fewer reasoning steps.
"""

from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

# Import the new confidence extractor
from .confidence_extractor import ConfidenceExtractor

# Import from the original TFCA module or recreate essentials
class SimilarityMetrics:
    """Implements similarity metrics as in original TFCA."""
    
    @staticmethod
    def string_similarity(str1: str, str2: str) -> float:
        """Calculate string similarity using sequence matcher."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        from scipy.spatial.distance import cosine
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return 1 - cosine(vec1, vec2)

class TriadicFuzzyFCAAnalysis:
    """
    Class that inverts the traditional TFCA approach by using:
    - G (objects) = assumptions and conditions combined
    - M (attributes) = reasoning steps
    - B (conditions) = confidence levels
    
    This inversion is beneficial when there are many assumptions but fewer reasoning steps.
    
    The class implements Triadic Fuzzy Formal Concept Analysis with Galois connections
    to establish relationships between formal concepts in the triadic context.
    """
    
    def __init__(self, knowledge_base=None, similarity_threshold: float = 0.7, use_embeddings: bool = False):
        """Initialize analyzer with optional reference to knowledge base.
        
        Args:
            knowledge_base: Optional knowledge base for lookups
            similarity_threshold: Threshold for fuzzy similarity matching
            use_embeddings: Whether to use embeddings for similarity calculation
        """
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self._similarity_cache = {}
    
    def analyze_reasoning(self, reasoning_steps: List[Dict[str, Any]], tau: float = 0.5) -> Dict[str, Any]:
        """
        Analyze reasoning steps using inverted triadic context structure.
        
        Args:
            reasoning_steps: List of reasoning steps with assumptions
            tau: Threshold for fuzzy membership
            
        Returns:
            Analysis results including inverted context and concepts
        """
        # First, process reasoning steps to extract confidence values
        processed_steps = ConfidenceExtractor.extract_from_reasoning_steps(reasoning_steps)
        
        # Extract all unique assumptions as objects (G)
        all_assumptions = set()
        for step in processed_steps:
            assumptions = step.get("Assumptions", step.get("assumptions", []))
            all_assumptions.update(assumptions)
        
        G = list(all_assumptions)  # Objects = assumptions
        
        # Steps become attributes (M)
        M = [(i, step.get("Description", f"Step {i+1}")) for i, step in enumerate(processed_steps)]
        
        # Define B (conditions = confidence levels)
        B = ["high", "medium", "low"]
        
        # Initialize Y (fuzzy ternary relation)
        # Y[g_idx][m_idx][b_idx] = membership of assumption g in step m under confidence b
        Y = [[[0.0 for _ in range(len(B))] for _ in range(len(M))] for _ in range(len(G))]
        
        # Populate Y with fuzzy membership values
        for g_idx, assumption in enumerate(G):
            for m_idx, (step_idx, _) in enumerate(M):
                step = processed_steps[step_idx]
                # Use the extracted confidence category
                step_confidence = step.get("confidence_category", "medium").lower()
                step_assumptions = step.get("Assumptions", step.get("assumptions", []))
                
                # Map confidence level to an index in B
                confidence_idx = B.index(step_confidence) if step_confidence in B else B.index("medium")
                
                # Check if this assumption is part of this step
                if assumption in step_assumptions:
                    Y[g_idx][m_idx][confidence_idx] = 1.0
                else:
                    # Check for similar assumptions (semantic matching)
                    for step_assumption in step_assumptions:
                        similarity = self._compute_similarity(assumption, step_assumption)
                        if similarity >= self.similarity_threshold:
                            Y[g_idx][m_idx][confidence_idx] = similarity
                            break
        
        # Generate inverted context representation
        inverted_context = {
            "G": G,  # Objects = assumptions
            "M": M,  # Attributes = steps
            "B": B,  # Conditions = confidence levels
            "Y": Y   # Fuzzy ternary relation
        }
        
        # Generate concepts using a simplified approach for the inverted context
        concepts = self.generate_inverted_concepts(G, M, B, Y, tau)
        
        # Build Galois lattice structure with triadic connections
        lattice_result = self.build_concept_lattice(concepts, Y, tau)
        
        # Create a detailed lattice analysis report with enhanced metrics
        lattice_analysis = {
            # Basic concept statistics
            "concept_count": len(concepts),
            "max_extent_size": max([len(c.get('A', [])) for c in concepts]) if concepts else 0,
            "max_intent_size": max([len(c.get('C', [])) for c in concepts]) if concepts else 0,
            "max_modus_size": max([len(c.get('D', [])) for c in concepts]) if concepts else 0,
            "avg_extent_size": sum([len(c.get('A', [])) for c in concepts]) / max(1, len(concepts)),
            "avg_intent_size": sum([len(c.get('C', [])) for c in concepts]) / max(1, len(concepts)),
            "avg_modus_size": sum([len(c.get('D', [])) for c in concepts]) / max(1, len(concepts)),
            
            # Graph structure metrics
            "density": self._calculate_lattice_density(lattice_result['lattice_structure']),
            "connectivity": self._calculate_lattice_connectivity(lattice_result['lattice_structure']),
            
            # Stability and quality metrics
            "concept_stability": self._calculate_concept_stability(concepts, Y, tau),
            "triadic_quality": self._calculate_triadic_quality(concepts, Y),
            
            # Information-theoretic measures
            "information_content": self._calculate_information_content(concepts),
            
            # Distribution metrics
            "concept_distribution": self._analyze_concept_distribution(concepts),
            
            # Triadic-specific metrics
            "triadic_cohesion": self._calculate_triadic_cohesion(concepts, Y, tau)
        }
        
        return {
            "original_steps": reasoning_steps,
            "inverted_context": inverted_context,
            "concepts": lattice_result['concepts'],  # Updated concepts with triadic connections
            "lattice": lattice_result['lattice_structure'],  # Triadic concept lattice structure
            "lattice_analysis": lattice_analysis,  # Detailed lattice metrics and analysis
            "triadic_connections": [
                {
                    "source_concept": conn['source'],
                    "target_concept": conn['target'],
                    "similarity": conn['similarity'],
                    "connection_type": conn['connection_type']
                } for conn in lattice_result['lattice_structure'].get('triadic_connections', [])
            ],
            "concept_visualization": self._generate_concept_visualization(concepts),
            "tau_threshold": tau
        }
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings with caching."""
        # Create cache key (sort to ensure consistency)
        key = tuple(sorted([text1, text2]))
        
        if key in self._similarity_cache:
            return self._similarity_cache[key]
        
        similarity = SimilarityMetrics.string_similarity(text1, text2)
        self._similarity_cache[key] = similarity
        return similarity
    
    def generate_inverted_concepts(self, G, M, B, Y, tau: float) -> List[Dict[str, Any]]:
        """
        Generate triadic formal concepts for the triadic context using proper triadic derivation operators.
        
        A triadic formal concept is a triple (A, B, C) where:
        - A is a subset of objects (assumptions)
        - B is a subset of attributes (steps)
        - C is a subset of conditions (confidence levels)
        - The triple is maximal with respect to component-wise set inclusion
        
        Args:
            G: Assumptions as objects
            M: Steps as attributes
            B: Confidence levels as conditions
            tau: Threshold for fuzzy membership
            Y: Fuzzy ternary relation
            
        Returns:
            List of triadic formal concepts
        """
        concepts = []
        
        # Add top concept (all assumptions, no steps, no conditions)
        top_concept = {
            "intent": "Top Concept",
            "is_top": True,
            "is_bottom": False,
            "A": list(range(len(G))),  # All assumptions
            "C": [],                   # No steps
            "D": [],                   # No conditions
            "A_objects": [G[i] for i in range(len(G))],
            "C_attributes": [],
            "D_conditions": [],
            "children": [],
            "parents": []
        }
        concepts.append(top_concept)
        
        # Add bottom concept (no assumptions, all steps, all conditions)
        bottom_concept = {
            "intent": "Bottom Concept",
            "is_top": False,
            "is_bottom": True,
            "A": [],                   # No assumptions
            "C": list(range(len(M))),  # All steps
            "D": list(range(len(B))),  # All conditions
            "A_objects": [],
            "C_attributes": [M[i] for i in range(len(M))],
            "D_conditions": [B[i] for i in range(len(B))],
            "children": [],
            "parents": []
        }
        concepts.append(bottom_concept)
        
        # Generate concepts using the triadic derivation operators
        # First, generate concepts for each individual condition (confidence level)
        for b_idx, confidence in enumerate(B):
            # For each step, find related assumptions
            for m_idx, (step_idx, step_desc) in enumerate(M):
                # Find assumptions with membership >= tau for this step and confidence
                A_indices = [g_idx for g_idx in range(len(G)) 
                           if Y[g_idx][m_idx][b_idx] >= tau]
                
                if A_indices:
                    # Use triadic derivation to complete the concept
                    X, Y_derive, Z = self._triadic_closure(A_indices, [m_idx], [b_idx], Y, tau)
                    
                    # Create the concept
                    concept = {
                        "intent": f"Concept for Step {step_idx+1} with {confidence} confidence",
                        "is_top": False,
                        "is_bottom": False,
                        "A": X,  # Derivation may have modified the extent
                        "C": Y_derive,  # Derivation may have added more attributes
                        "D": Z,  # Derivation may have added more conditions
                        "A_objects": [G[i] for i in X if i < len(G)],
                        "C_attributes": [M[i] for i in Y_derive if i < len(M)],
                        "D_conditions": [B[i] for i in Z if i < len(B)],
                        "children": [],
                        "parents": []
                    }
                    
                    # Check if concept is unique
                    if not any(self._is_same_concept(concept, c) for c in concepts):
                        concepts.append(concept)
        
        # Generate multi-attribute concepts - find intersections
        for b_idx, confidence in enumerate(B):
            # Try pairs of steps
            for m1_idx in range(len(M)):
                for m2_idx in range(m1_idx + 1, len(M)):
                    # Find common assumptions
                    A_indices = [
                        g_idx for g_idx in range(len(G))
                        if Y[g_idx][m1_idx][b_idx] >= tau and Y[g_idx][m2_idx][b_idx] >= tau
                    ]
                    
                    if A_indices:
                        # Use triadic derivation to complete this concept
                        X, Y_derive, Z = self._triadic_closure(A_indices, [m1_idx, m2_idx], [b_idx], Y, tau)
                        
                        step1_idx = M[m1_idx][0]
                        step2_idx = M[m2_idx][0]
                        concept = {
                            "intent": f"Concept linking Steps {step1_idx+1} and {step2_idx+1} with {confidence}",
                            "is_top": False,
                            "is_bottom": False,
                            "A": X,
                            "C": Y_derive,
                            "D": Z,
                            "A_objects": [G[i] for i in X if i < len(G)],
                            "C_attributes": [M[i] for i in Y_derive if i < len(M)],
                            "D_conditions": [B[i] for i in Z if i < len(B)],
                            "children": [],
                            "parents": []
                        }
                        
                        # Check if concept is unique
                        if not any(self._is_same_concept(concept, c) for c in concepts):
                            concepts.append(concept)
        
        # Generate multi-condition concepts
        for b1_idx in range(len(B)):
            for b2_idx in range(b1_idx + 1, len(B)):
                for m_idx, (step_idx, _) in enumerate(M):
                    # Find assumptions satisfying both conditions
                    A_indices = [
                        g_idx for g_idx in range(len(G))
                        if Y[g_idx][m_idx][b1_idx] >= tau and Y[g_idx][m_idx][b2_idx] >= tau
                    ]
                    
                    if A_indices:
                        # Use triadic derivation to complete this concept
                        X, Y_derive, Z = self._triadic_closure(A_indices, [m_idx], [b1_idx, b2_idx], Y, tau)
                        
                        concept = {
                            "intent": f"Step {step_idx+1} with mixed confidence",
                            "is_top": False,
                            "is_bottom": False,
                            "A": X,
                            "C": Y_derive,
                            "D": Z,
                            "A_objects": [G[i] for i in X if i < len(G)],
                            "C_attributes": [M[i] for i in Y_derive if i < len(M)],
                            "D_conditions": [B[i] for i in Z if i < len(B)],
                            "children": [],
                            "parents": []
                        }
                        
                        # Check if concept is unique
                        if not any(self._is_same_concept(concept, c) for c in concepts):
                            concepts.append(concept)
        
        return concepts
    
    def _is_same_concept(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> bool:
        """Check if two concepts are essentially the same."""
        return (set(concept1['A']) == set(concept2['A']) and
                set(concept1['C']) == set(concept2['C']) and
                set(concept1['D']) == set(concept2['D']))
                
    def _is_subconcept(self, child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
        """Check if child is a subconcept of parent."""
        # Child has fewer or same objects as parent
        A_child = set(child['A'])
        A_parent = set(parent['A'])
        
        # Child has more or same attributes as parent
        C_child = set(child['C'])
        C_parent = set(parent['C'])
        
        # Child concept is more specific (or equal to) parent concept
        return (A_child.issubset(A_parent) and C_parent.issubset(C_child))
    
    def _calculate_triadic_derivation(self, X: List[int], Y: List[int], Z: List[int], 
                                Y_matrix: List[List[List[float]]], tau: float = 0.5) -> List[int]:
        """
        Calculate triadic derivation for a triple (X, Y, Z) in a fuzzy triadic context.
        
        For triadic contexts, there are several derivation operators:
        1. X→YZ that maps from objects to attribute-condition pairs
        2. Y→XZ that maps from attributes to object-condition pairs
        3. Z→XY that maps from conditions to object-attribute pairs
        
        This method implements these operators, taking into account the fuzzy nature of the context.
        
        Args:
            X: List of object indices
            Y: List of attribute indices
            Z: List of condition indices
            Y_matrix: The fuzzy ternary relation (membership degrees)
            tau: Threshold for fuzzy membership
            
        Returns:
            Derived set of indices based on the input
        """
        G_len = len(Y_matrix)  # Number of objects
        M_len = len(Y_matrix[0]) if G_len > 0 else 0  # Number of attributes
        B_len = len(Y_matrix[0][0]) if M_len > 0 else 0  # Number of conditions
        
        if not X:  # X is empty, derive objects
            result = []
            for g in range(G_len):
                add_g = True
                for m in Y:
                    for b in Z:
                        if Y_matrix[g][m][b] < tau:
                            pass
                if add_g:
                    result.append(g)
            return result
            
        elif not Y:  # Y is empty, derive attributes
            result = []
            for m in range(M_len):
                add_m = True
                for g in X:
                    for b in Z:
                        if Y_matrix[g][m][b] < tau:
                            add_m = False
                            break
                    if not add_m:
                        break
                if add_m:
                    result.append(m)
            return result
            
        elif not Z:  # Z is empty, derive conditions
            result = []
            for b in range(B_len):
                add_b = True
                for g in X:
                    for m in Y:
                        if Y_matrix[g][m][b] < tau:
                            add_b = False
                            break
                    if not add_b:
                        break
                if add_b:
                    result.append(b)
            return result
            
        return []  # Default empty result
    
    def _triadic_closure(self, X: List[int], Y: List[int], Z: List[int], 
                          Y_matrix: List[List[List[float]]], tau: float = 0.5) -> Tuple[List[int], List[int], List[int]]:
        """
        Calculate the triadic closure of a triple (X, Y, Z).
        
        In triadic contexts, the closure operators are combinations of derivation operators:
        * For objects: X → (X′Y′Z′)′YZ′ → (X′′Y′Z′)′YZ′ → ...
        * For attributes: Y → (X′Y′Z′)′XZ′ → (X′Y′′Z′)′XZ′ → ...
        * For conditions: Z → (X′Y′Z′)′XY′ → (X′Y′Z′′)′XY′ → ...
        
        This function iteratively applies these closure operators until a fixed point is reached.
        
        Args:
            X: List of object indices
            Y: List of attribute indices
            Z: List of condition indices
            Y_matrix: The fuzzy ternary relation
            tau: Threshold for fuzzy membership
            
        Returns:
            The closed triple (X̃, Ỹ, Z̃)
        """
        X_new, Y_new, Z_new = X.copy(), Y.copy(), Z.copy()
        
        # Iteratively apply derivation operators until fixed point
        changed = True
        while changed:
            changed = False
            
            # Compute X closure
            Z_derive = self._calculate_triadic_derivation(X_new, Y_new, [], Y_matrix, tau)
            Y_derive = self._calculate_triadic_derivation(X_new, [], Z_derive, Y_matrix, tau)
            X_derive = self._calculate_triadic_derivation([], Y_derive, Z_derive, Y_matrix, tau)
            
            # Compute Y closure
            Z_derive2 = self._calculate_triadic_derivation(X_new, Y_new, [], Y_matrix, tau)
            X_derive2 = self._calculate_triadic_derivation([], Y_new, Z_derive2, Y_matrix, tau)
            Y_derive2 = self._calculate_triadic_derivation(X_derive2, [], Z_derive2, Y_matrix, tau)
            
            # Compute Z closure
            Y_derive3 = self._calculate_triadic_derivation(X_new, [], Z_new, Y_matrix, tau)
            X_derive3 = self._calculate_triadic_derivation([], Y_derive3, Z_new, Y_matrix, tau)
            Z_derive3 = self._calculate_triadic_derivation(X_derive3, Y_derive3, [], Y_matrix, tau)
            
            # Check if any set has changed
            if set(X_derive) != set(X_new) or set(Y_derive2) != set(Y_new) or set(Z_derive3) != set(Z_new):
                changed = True
                X_new, Y_new, Z_new = X_derive, Y_derive2, Z_derive3
        
        return X_new, Y_new, Z_new
    
    def _calculate_concept_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two triadic concepts based on their extents, intents, and modus.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Similarity score between 0 (completely different) and 1 (identical)
        """
        # Extract concept components
        A1 = set(concept1.get('A', []))
        A2 = set(concept2.get('A', []))
        C1 = set(concept1.get('C', []))
        C2 = set(concept2.get('C', []))
        D1 = set(concept1.get('D', []))
        D2 = set(concept2.get('D', []))
        
        # Calculate Jaccard similarity for each component
        a_similarity = len(A1.intersection(A2)) / max(1, len(A1.union(A2)))
        c_similarity = len(C1.intersection(C2)) / max(1, len(C1.union(C2)))
        d_similarity = len(D1.intersection(D2)) / max(1, len(D1.union(D2)))
        
        # Weight the similarities (can be adjusted based on importance)
        weighted_similarity = (0.4 * a_similarity + 0.4 * c_similarity + 0.2 * d_similarity)
        
        return weighted_similarity
    
    def build_concept_lattice(self, concepts: List[Dict[str, Any]], Y_matrix: List[List[List[float]]] = None, tau: float = 0.5) -> Dict[str, Any]:
        """
        Build concept lattice specific to Triadic Fuzzy Formal Concept Analysis.
        
        This method creates a proper triadic concept lattice by:
        1. Establishing concept ordering using triadic closures
        2. Calculating triadic concept similarity
        3. Building a navigable lattice structure with proper connections
        
        Args:
            concepts: List of triadic concepts (with extents A, intents C, and modi D)
            Y_matrix: The fuzzy ternary relation (needed for derivation operations)
            tau: Threshold for fuzzy membership
            
        Returns:
            Dictionary containing the triadic concept lattice with proper connections
        """
        # Sort concepts by size of extent (A) in descending order and intent (C) in ascending order
        concepts.sort(key=lambda c: (-len(c['A']), len(c['C']), len(c['D'])))
        
        # Initialize parent-child relationships
        for i, concept in enumerate(concepts):
            concepts[i]['children'] = []
            concepts[i]['parents'] = []
            concepts[i]['triadic_connections'] = []
        
        # For each pair of concepts, determine if they have a hierarchical relationship
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i == j:
                    continue
                    
                # Check if concept1 is a subconcept of concept2 using proper triadic subconcept relation
                if self._is_subconcept(concept1, concept2):
                    # Check if this is a direct subconcept (no concepts in between)
                    is_direct = True
                    for k, middle in enumerate(concepts):
                        if i != k and j != k and self._is_subconcept(concept1, middle) and self._is_subconcept(middle, concept2):
                            is_direct = False
                            break
                    
                    if is_direct:
                        # Add direct parent-child relationship
                        if j not in concept1['parents']:
                            concept1['parents'].append(j)
                        if i not in concept2['children']:
                            concept2['children'].append(i)
                        
                        # Calculate the similarity between the concepts
                        similarity = self._calculate_concept_similarity(concept1, concept2)
                        
                        # Add triadic connection between concepts
                        connection = {
                            'source': i,
                            'target': j,
                            'similarity': similarity,
                            'type': 'subconcept'
                        }
                        
                        # Store connection in the concept
                        concept1['triadic_connections'].append({
                            'target_concept': j,
                            'connection_type': 'parent',
                            'similarity': similarity
                        })
                        concept2['triadic_connections'].append({
                            'target_concept': i,
                            'connection_type': 'child',
                            'similarity': similarity
                        })
        
        # Create the triadic concept lattice structure
        triadic_lattice = {
            'nodes': [
                {
                    'id': i,
                    'label': f"Concept {i}",
                    'type': 'triadic_concept',
                    'is_top': concept.get('is_top', False),
                    'is_bottom': concept.get('is_bottom', False),
                    'extent_size': len(concept.get('A', [])),
                    'intent_size': len(concept.get('C', [])),
                    'modus_size': len(concept.get('D', [])),
                    'extent': concept.get('A_objects', []),
                    'intent': concept.get('C_attributes', []),
                    'modus': concept.get('D_conditions', [])
                } for i, concept in enumerate(concepts)
            ],
            'edges': [],
            'triadic_connections': []
        }
        
        # Add edges to the lattice structure
        for i, concept in enumerate(concepts):
            # Add parent-child edges
            for parent_idx in concept.get('parents', []):
                triadic_lattice['edges'].append({
                    'source': i,
                    'target': parent_idx,
                    'type': 'subconcept'
                })
            
            # Add triadic connections
            for conn in concept.get('triadic_connections', []):
                if conn['connection_type'] == 'parent':
                    triadic_lattice['triadic_connections'].append({
                        'source': i,
                        'target': conn['target_concept'],
                        'similarity': conn['similarity'],
                        'connection_type': 'triadic'
                    })
        
        return {
            'concepts': concepts,
            'lattice_structure': triadic_lattice
        }
    
    def _calculate_lattice_density(self, lattice_structure: Dict[str, Any]) -> float:
        """
        Calculate the density of the lattice, which measures how connected the concepts are.
        
        Density = number of edges / maximum possible edges in a DAG with n nodes
        For a DAG, max edges = n(n-1)/2
        
        Args:
            lattice_structure: The lattice structure with nodes and edges
            
        Returns:
            Density value between 0 and 1
        """
        nodes = lattice_structure.get('nodes', [])
        edges = lattice_structure.get('edges', [])
        
        n = len(nodes)
        if n <= 1:  # No density for 0 or 1 node
            return 0.0
        
        max_edges = n * (n - 1) / 2  # Maximum possible edges in a DAG
        return len(edges) / max_edges if max_edges > 0 else 0.0
    
    def _calculate_lattice_connectivity(self, lattice_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate connectivity metrics for the lattice.
        
        Args:
            lattice_structure: The lattice structure with nodes and edges
            
        Returns:
            Dictionary with connectivity metrics
        """
        nodes = lattice_structure.get('nodes', [])
        edges = lattice_structure.get('edges', [])
        
        # Create an adjacency list representation of the graph
        graph = {node['id']: [] for node in nodes}
        for edge in edges:
            graph[edge['source']].append(edge['target'])
        
        # Calculate basic connectivity metrics
        avg_degree = sum(len(neighbors) for neighbors in graph.values()) / len(graph) if graph else 0
        max_degree = max(len(neighbors) for neighbors in graph.values()) if graph else 0
        isolated_nodes = sum(1 for neighbors in graph.values() if len(neighbors) == 0)
        
        return {
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'isolated_nodes': isolated_nodes,
            'connectivity_ratio': 1.0 - (isolated_nodes / len(nodes)) if nodes else 0.0
        }
    
    def _calculate_concept_stability(self, concepts: List[Dict[str, Any]], Y: List[List[List[float]]], tau: float) -> Dict[str, Any]:
        """
        Calculate stability metrics for concepts in the lattice.
        
        Concept stability measures how robust a concept is to changes in the underlying data.
        Higher stability means the concept is more likely to persist even if parts of the data change.
        
        Args:
            concepts: List of triadic concepts
            Y: The fuzzy ternary relation
            tau: Threshold for fuzzy membership
            
        Returns:
            Dictionary with stability metrics for concepts
        """
        if not concepts:
            return {'avg_stability': 0.0, 'max_stability': 0.0, 'min_stability': 0.0}
        
        # Calculate stability for each concept
        stabilities = []
        for concept in concepts:
            # Skip top and bottom concepts
            if concept.get('is_top', False) or concept.get('is_bottom', False):
                continue
                
            # Calculate concept stability based on extent size, intent size, and connectivity
            extent_size = len(concept.get('A', []))
            intent_size = len(concept.get('C', []))
            modus_size = len(concept.get('D', []))
            
            # Stability increases with extent size and decreases with intent size
            # This reflects that concepts with large extents and small intents are more stable
            if extent_size > 0 and intent_size > 0:
                stability = (extent_size / (extent_size + intent_size + modus_size)) * extent_size
                stabilities.append(stability)
        
        if not stabilities:
            return {'avg_stability': 0.0, 'max_stability': 0.0, 'min_stability': 0.0}
            
        return {
            'avg_stability': sum(stabilities) / len(stabilities),
            'max_stability': max(stabilities),
            'min_stability': min(stabilities),
            'stability_distribution': {
                'high': sum(1 for s in stabilities if s > 0.7) / len(stabilities),
                'medium': sum(1 for s in stabilities if 0.3 <= s <= 0.7) / len(stabilities),
                'low': sum(1 for s in stabilities if s < 0.3) / len(stabilities)
            }
        }
    
    def _calculate_triadic_quality(self, concepts: List[Dict[str, Any]], Y: List[List[List[float]]]) -> Dict[str, Any]:
        """
        Calculate the overall quality of the triadic concept formation.
        
        Quality is measured by how well the concepts cover the original data and how
        meaningful the discovered triadic relationships are.
        
        Args:
            concepts: List of triadic concepts
            Y: The fuzzy ternary relation
            
        Returns:
            Dictionary with quality metrics
        """
        if not concepts or not Y:
            return {'coverage': 0.0, 'precision': 0.0, 'overall_quality': 0.0}
        
        # Calculate coverage: how much of the original data is covered by concepts
        total_cells = len(Y) * len(Y[0]) * len(Y[0][0]) if Y and Y[0] and Y[0][0] else 0
        covered_cells = 0
        
        # Calculate how many cells from Y are covered by at least one concept
        covered_positions = set()
        for concept in concepts:
            A = concept.get('A', [])
            C = concept.get('C', [])
            D = concept.get('D', [])
            
            for i in A:
                for j in C:
                    for k in D:
                        if i < len(Y) and j < len(Y[0]) and k < len(Y[0][0]):
                            covered_positions.add((i, j, k))
        
        coverage = len(covered_positions) / max(1, total_cells)
        
        # Calculate precision: how well-formed the concepts are
        # Higher precision means the concepts have fewer false positives
        precision = 0.0
        for concept in concepts:
            A = concept.get('A', [])
            C = concept.get('C', [])
            D = concept.get('D', [])
            
            # Smaller concepts tend to be more precise
            concept_size = len(A) * len(C) * len(D)
            if concept_size > 0:
                # Calculate how well this concept matches the original data
                matching_cells = 0
                for i in A:
                    for j in C:
                        for k in D:
                            if i < len(Y) and j < len(Y[0]) and k < len(Y[0][0]) and Y[i][j][k] > 0.5:
                                matching_cells += 1
                
                concept_precision = matching_cells / concept_size
                precision += concept_precision
        
        avg_precision = precision / max(1, len(concepts))
        
        # Calculate F1-like overall quality (harmonic mean of coverage and precision)
        overall_quality = 0.0
        if coverage + avg_precision > 0:
            overall_quality = 2 * coverage * avg_precision / (coverage + avg_precision)
        
        return {
            'coverage': coverage,
            'precision': avg_precision,
            'overall_quality': overall_quality
        }
    
    def _calculate_information_content(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the information content of the concept lattice.
        
        Information content measures how much information is captured by the conceptual structure.
        Higher values indicate a more informative and less redundant concept lattice.
        
        Args:
            concepts: List of triadic concepts
            
        Returns:
            Dictionary with information content metrics
        """
        if not concepts:
            return {'entropy': 0.0, 'redundancy': 0.0, 'information_density': 0.0}
        
        # Calculate entropy based on concept size distribution
        total_elements = sum(len(c.get('A', [])) + len(c.get('C', [])) + len(c.get('D', [])) for c in concepts)
        
        # Skip calculation if no elements
        if total_elements == 0:
            return {'entropy': 0.0, 'redundancy': 0.0, 'information_density': 0.0}
        
        # Calculate probabilities for each concept
        probabilities = []
        for concept in concepts:
            concept_size = len(concept.get('A', [])) + len(concept.get('C', [])) + len(concept.get('D', []))
            probability = concept_size / total_elements
            if probability > 0:
                probabilities.append(probability)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for p in probabilities:
            entropy -= p * np.log2(p)
        
        # Calculate redundancy (overlap between concepts)
        # Higher redundancy means more overlap between concepts
        redundancy = 0.0
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                
                A1 = set(concept1.get('A', []))
                C1 = set(concept1.get('C', []))
                D1 = set(concept1.get('D', []))
                
                A2 = set(concept2.get('A', []))
                C2 = set(concept2.get('C', []))
                D2 = set(concept2.get('D', []))
                
                # Calculate Jaccard similarity for each dimension
                jaccard_A = len(A1.intersection(A2)) / max(1, len(A1.union(A2)))
                jaccard_C = len(C1.intersection(C2)) / max(1, len(C1.union(C2)))
                jaccard_D = len(D1.intersection(D2)) / max(1, len(D1.union(D2)))
                
                # Average similarity across all dimensions
                avg_similarity = (jaccard_A + jaccard_C + jaccard_D) / 3
                redundancy += avg_similarity
        
        # Normalize redundancy by the number of possible pairs
        num_pairs = max(1, len(concepts) * (len(concepts) - 1) / 2)
        normalized_redundancy = redundancy / num_pairs
        
        # Calculate information density: entropy per concept
        information_density = entropy / len(concepts)
        
        return {
            'entropy': entropy,
            'redundancy': normalized_redundancy,
            'information_density': information_density
        }
    
    def _analyze_concept_distribution(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of concepts by their properties.
        
        This helps understand the structure and balance of the concept lattice.
        
        Args:
            concepts: List of triadic concepts
            
        Returns:
            Dictionary with distribution metrics
        """
        if not concepts:
            return {'size_distribution': {}, 'balance_ratio': 0.0}
        
        # Create size categories
        tiny = 0  # 1-3 elements
        small = 0  # 4-10 elements
        medium = 0  # 11-30 elements
        large = 0  # 31+ elements
        
        # Count concepts by their total size
        for concept in concepts:
            size = len(concept.get('A', [])) + len(concept.get('C', [])) + len(concept.get('D', []))
            
            if size <= 3:
                tiny += 1
            elif size <= 10:
                small += 1
            elif size <= 30:
                medium += 1
            else:
                large += 1
        
        # Analyze balance between dimensions (A, C, D)
        dimension_sizes = {
            'A': [len(c.get('A', [])) for c in concepts],
            'C': [len(c.get('C', [])) for c in concepts],
            'D': [len(c.get('D', [])) for c in concepts]
        }
        
        # Calculate average size for each dimension
        avg_sizes = {
            'A': sum(dimension_sizes['A']) / len(concepts),
            'C': sum(dimension_sizes['C']) / len(concepts),
            'D': sum(dimension_sizes['D']) / len(concepts)
        }
        
        # Calculate balance ratio (how evenly distributed the sizes are)
        max_avg = max(avg_sizes.values())
        min_avg = min(avg_sizes.values())
        balance_ratio = min_avg / max_avg if max_avg > 0 else 0.0
        
        return {
            'size_distribution': {
                'tiny': tiny / len(concepts),
                'small': small / len(concepts),
                'medium': medium / len(concepts),
                'large': large / len(concepts)
            },
            'balance_ratio': balance_ratio,
            'dimension_averages': avg_sizes
        }
    
    def _calculate_triadic_cohesion(self, concepts: List[Dict[str, Any]], Y: List[List[List[float]]], tau: float) -> Dict[str, Any]:
        """
        Calculate the triadic cohesion metrics for the concept lattice.
        
        Triadic cohesion measures how strongly the three dimensions (A, C, D) are interconnected
        through the triadic concepts.
        
        Args:
            concepts: List of triadic concepts
            Y: The fuzzy ternary relation
            tau: Threshold for fuzzy membership
            
        Returns:
            Dictionary with cohesion metrics
        """
        if not concepts or not Y:
            return {'avg_cohesion': 0.0, 'max_cohesion': 0.0}
        
        cohesion_scores = []
        for concept in concepts:
            A = concept.get('A', [])
            C = concept.get('C', [])
            D = concept.get('D', [])
            
            # Skip concepts with empty dimensions
            if not A or not C or not D:
                continue
            
            # Calculate internal density of connections
            total_possible = len(A) * len(C) * len(D)
            actual_connections = 0
            
            for i in A:
                for j in C:
                    for k in D:
                        if i < len(Y) and j < len(Y[0]) and k < len(Y[0][0]):
                            if Y[i][j][k] >= tau:
                                actual_connections += 1
            
            # Calculate cohesion for this concept
            cohesion = actual_connections / max(1, total_possible)
            cohesion_scores.append(cohesion)
        
        if not cohesion_scores:
            return {'avg_cohesion': 0.0, 'max_cohesion': 0.0}
        
        return {
            'avg_cohesion': sum(cohesion_scores) / len(cohesion_scores),
            'max_cohesion': max(cohesion_scores),
            'min_cohesion': min(cohesion_scores),
            'cohesion_std': np.std(cohesion_scores) if len(cohesion_scores) > 1 else 0.0
        }
    
    def _generate_concept_visualization(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a simplified visualization structure for the concepts.
        
        Args:
            concepts: List of triadic concepts
            
        Returns:
            Dictionary with visualization data for concepts
        """
        # Create a simplified representation for visualization
        nodes = []
        links = []
        
        # Create nodes for each concept
        for i, concept in enumerate(concepts):
            # Node attributes that would be useful for visualization
            node_type = "top" if concept.get('is_top', False) else \
                       "bottom" if concept.get('is_bottom', False) else "regular"
            
            nodes.append({
                'id': i,
                'label': concept.get('intent', f"Concept {i}"),
                'type': node_type,
                'extent_size': len(concept.get('A', [])),
                'intent_size': len(concept.get('C', [])),
                'modus_size': len(concept.get('D', [])),
            })
            
            # Create links based on parent-child relationships
            for child_idx in concept.get('children', []):
                links.append({
                    'source': i,
                    'target': child_idx,
                    'type': 'subconcept'
                })
        
        return {
            'nodes': nodes,
            'links': links,
            'layout': 'hierarchical',  # Suggest a hierarchical layout for the lattice
            'node_count': len(nodes),
            'link_count': len(links)
        }
    
    def build(self, reasoning_steps: List[Dict[str, Any]], tau: float = 0.5) -> Dict[str, Any]:
        """
        Compatibility method for ThreeWayCOT framework integration.
        Acts as a wrapper around analyze_reasoning.
        
        Args:
            reasoning_steps: List of reasoning steps with assumptions
            tau: Threshold for fuzzy membership
            
        Returns:
            Analysis results with context, concepts, and full lattice structure
        """
        return self.analyze_reasoning(reasoning_steps, tau)
    
    def _is_subconcept(self, child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
        """
        Check if child is a subconcept of parent.
        
        In the inverted setting:
        - child.A ⊇ parent.A (child has more assumptions)
        - child.C ⊆ parent.C (child has fewer steps)
        - child.D ⊆ parent.D (child has fewer conditions)
        
        Args:
            child: Child concept
            parent: Parent concept
            
        Returns:
            True if child is subconcept of parent
        """
        return (set(child['A']).issuperset(set(parent['A'])) and 
                set(child['C']).issubset(set(parent['C'])) and
                set(child['D']).issubset(set(parent['D'])))

def apply_inverted_tfca_to_cot(cot_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply Inverted Triadic Fuzzy Concept Analysis to a chain of thought reasoning.
    
    Args:
        cot_steps: List of reasoning steps in the chain of thought
        
    Returns:
        Analysis results with inverted roles of objects and attributes
    """
    analyzer = TriadicFuzzyAnalysis()
    return analyzer.analyze_reasoning(cot_steps)

# Example usage
if __name__ == "__main__":
    # Enhanced example with more reasoning steps and assumptions
    reasoning_steps = [
        {
            "Description": "Initial problem understanding",
            "Assumptions": [
                "Problem is well-defined", 
                "All variables are known", 
                "Linear relationship exists",
                "Input data is complete",
                "Problem scope is clear"
            ],
            "Confidence": "high",
            "Context": "initial_analysis"
        },
        {
            "Description": "Data collection and preprocessing",
            "Assumptions": [
                "Data is representative", 
                "No significant outliers", 
                "Missing values are handled",
                "Data is properly normalized",
                "Variables are properly encoded"
            ],
            "Confidence": "medium",
            "Context": "data_processing"
        },
        {
            "Description": "Mathematical model formulation",
            "Assumptions": [
                "Linear relationship exists", 
                "Variables are independent", 
                "Normal distribution of errors",
                "Homoscedasticity holds",
                "No multicollinearity"
            ],
            "Confidence": "medium",
            "Context": "modeling"
        },
        {
            "Description": "Solution approach",
            "Assumptions": [
                "Problem is well-defined", 
                "Solution exists", 
                "Unique solution",
                "Algorithm will converge",
                "Computational resources are sufficient"
            ],
            "Confidence": "high",
            "Context": "solution_design"
        },
        {
            "Description": "Implementation and testing",
            "Assumptions": [
                "Code is bug-free",
                "Test cases are comprehensive",
                "Performance metrics are appropriate",
                "Edge cases are handled",
                "Results are reproducible"
            ],
            "Confidence": "medium",
            "Context": "implementation"
        },
        {
            "Description": "Validation and deployment",
            "Assumptions": [
                "Validation data is representative",
                "Model generalizes well",
                "Deployment environment is stable",
                "Monitoring is in place",
                "Maintenance plan exists"
            ],
            "Confidence": "low",
            "Context": "deployment"
        }
    ]
    
    # Apply inverted TFCA with different tau values to see the effect
    for tau in [0.3, 0.5, 0.7]:
        print(f"\n{'='*50}")
        print(f"INVERTED TFCA ANALYSIS (τ = {tau})")
        print(f"{'='*50}")
        
        analyzer = InvertedTriadicFuzzyAnalysis(similarity_threshold=tau)
        results = analyzer.analyze_reasoning(reasoning_steps, tau=tau)
        
        # Print summary
        print(f"\nGenerated {len(results['concepts'])} concepts")
        print(f"Number of assumptions (G): {len(results['inverted_context']['G'])}")
        print(f"Number of steps (M): {len(results['inverted_context']['M'])}")
        print(f"Number of conditions (B): {len(results['inverted_context']['B'])}")
        
        # Print most interesting concepts (non-extreme ones)
        print("\nMost informative concepts:")
        concepts = sorted(
            [c for c in results['concepts'] if not c.get('is_top', False) and not c.get('is_bottom', False)],
            key=lambda x: len(x['A_objects']) * len(x['C_attributes']),
            reverse=True
        )
        
        for i, concept in enumerate(concepts[:5]):  # Show top 5 most interesting concepts
            print(f"\nConcept {i+1}:")
            print(f"  Intent: {concept['intent']}")
            print(f"  Assumptions ({len(concept['A_objects'])}): {concept['A_objects'][:3]}{'...' if len(concept['A_objects']) > 3 else ''}")
            steps = [f"Step {s[0]+1}: {s[1]}" for s in concept['C_attributes']]
            print(f"  Steps ({len(steps)}): {steps}")
            print(f"  Conditions: {concept['D_conditions']}")
        
        # Print top and bottom concepts if they exist
        top_concepts = [c for c in results['concepts'] if c.get('is_top', False)]
        bottom_concepts = [c for c in results['concepts'] if c.get('is_bottom', False)]
        
        if top_concepts:
            print("\nTop Concept:")
            print(f"  Assumptions: {len(top_concepts[0]['A_objects'])}")
            print(f"  Steps: {len(top_concepts[0]['C_attributes'])}")
            
        if bottom_concepts:
            print("\nBottom Concept:")
            print(f"  Assumptions: {len(bottom_concepts[0]['A_objects'])}")
            print(f"  Steps: {len(bottom_concepts[0]['C_attributes'])}")
