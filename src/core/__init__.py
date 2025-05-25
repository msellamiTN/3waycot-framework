"""
3WayCoT Core Components

This package contains the core components of the 3WayCoT framework,
including the main ThreeWayCOT class and supporting modules.
"""

from .threeway_cot import ThreeWayCOT
from .cot_generator import ChainOfThoughtGenerator
from .three_way_decision import ThreeWayDecisionMaker
from .triadic_fca import TriadicFuzzyFCAAnalysis
from .knowledge_base import KnowledgeBase
from .uncertainty_resolver import UncertaintyResolver

__all__ = [ 
    'ThreeWayCOT',
    'ChainOfThoughtGenerator',
    'ThreeWayDecisionMaker',
    'TriadicFuzzyFCAAnalysis',
    'KnowledgeBase',
    'UncertaintyResolver'
]
