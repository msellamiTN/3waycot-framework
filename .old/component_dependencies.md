# 3WayCoT Framework Component Dependencies

## Core Components Overview

### ThreeWayCOT (src/core/threeway_cot.py)
- **Primary framework class** that orchestrates the entire process
- Initializes with parameters:
  - `alpha`: Acceptance threshold (default: 0.7)
  - `beta`: Rejection threshold (default: 0.6)
  - `tau`: Threshold for fuzzy membership (default: 0.4)
  - `similarity_threshold`: For similarity matching (default: 0.65)
  - `knowledge_base_path`: Path to knowledge base
  - `max_assumptions`: Maximum assumptions to include per step
  - `llm_provider`: LLM provider name
  - `llm_model`: LLM model name
- **Dependencies**:
  - KnowledgeBase
  - TriadicFuzzyFCAAnalysis (both for analysis and context construction)
  - ChainOfThoughtGenerator
  - UncertaintyResolver
  - ThreeWayDecisionMaker (created internally during processing)

### ThreeWayDecisionMaker (src/core/three_way_decision.py)
- **Makes three-way decisions** based on the TFCA analysis
- Initializes with parameters:
  - `alpha`: Acceptance threshold (must satisfy: 0 <= beta < alpha <= 1.0)
  - `beta`: Rejection threshold (must satisfy: 0 <= beta < alpha <= 1.0)
  - `gamma`: Boundary width threshold (typically 0.6)
- **Key methods**:
  - `make_decisions()`: Takes analysis and uncertainty_analysis as input
  - `decide_step()`: Makes decision for a single reasoning step
  - `_calculate_decision_metrics()`: Calculates metrics for decision making
  - `_make_decision()`: Applies decision rules based on thresholds

### TriadicFuzzyFCAAnalysis (src/core/triadic_fca.py)
- **Performs Triadic Fuzzy Concept Analysis** on reasoning steps
- Initializes with parameters:
  - `knowledge_base`: Optional knowledge base reference
  - `similarity_threshold`: Threshold for fuzzy similarity matching (default: 0.7)
  - `use_embeddings`: Whether to use embeddings for similarity (default: False)
- **Key methods**:
  - `analyze_reasoning()`: Primary method that analyzes reasoning steps
  - `build()`: Builds a triadic context from reasoning steps
- **Note**: Does **NOT** have `integrate_with_triadic_context()` method that ThreeWayCOT tries to call

### ConfidenceExtractor (src/core/confidence_extractor.py)
- **Extracts confidence values** from reasoning text
- **Key methods**:
  - `extract_confidence()`: Returns a float confidence value between 0-1
  - `extract_from_reasoning_steps()`: Processes multiple reasoning steps

### UncertaintyResolver (src/core/uncertainty_resolver.py)
- **Resolves uncertain decisions**
- Initializes with:
  - `cot_generator`: Reference to the CoT generator
  - `knowledge_base`: Knowledge base for lookups
  - `relevance_threshold`, `validity_threshold`, `confidence_threshold`: Various thresholds
- **Key methods**:
  - `resolve()`: Takes uncertain_step, all_steps, and triadic_context

## Key Parameter Requirements

### ThreeWayDecisionMaker Thresholds
- Must satisfy: 0 <= beta < alpha <= 1.0
- Typical values: alpha = 0.7, beta = 0.6 **DO NOT WORK**
- Correct values should be: alpha = 0.7, beta = 0.3 or lower

### Process Flow
1. **ChainOfThoughtGenerator** generates reasoning steps
2. **ConfidenceExtractor** extracts confidence from text
3. **TriadicFuzzyFCAAnalysis** analyzes the reasoning steps
4. **ThreeWayDecisionMaker** classifies decisions based on thresholds
5. **UncertaintyResolver** resolves any uncertain (ABSTAIN) decisions

## Common Issues

### Missing Methods
- TriadicFuzzyFCAAnalysis does not have `integrate_with_triadic_context()`
- ThreeWayCOT assumes it exists in `process_reasoning()`

### Threshold Confusion
- Three-way decision making uses alpha and beta differently than traditional TFCA
- In three-way decisions: alpha defines acceptance region, (1-beta) defines rejection region
- Must have beta < alpha for there to be a non-empty abstention region

### Integration Issues
- ThreeWayCOT.process_reasoning() uses methods that don't exist in core classes
- Need to directly use methods known to exist: analyze_reasoning() and make_decisions()
