"""
Enhanced Confidence Extraction Module for the 3WayCoT Framework.

This module provides advanced methods for extracting confidence values
from natural language text, with improved pattern matching and more
robust confidence estimation for various expression formats.
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("3WayCoT.ConfidenceExtractor")

class ConfidenceExtractor:
    """
    Provides enhanced methods to extract confidence values from text using advanced pattern matching.

    This class uses multiple strategies to detect both explicit numeric confidence values
    and natural language expressions of confidence, with improved pattern matching and
    more reliable confidence estimation for various formats.
    """
    
    @staticmethod
    def extract_confidence(text: str) -> Tuple[float, str]:
        """
        Extract confidence value from text using advanced pattern matching.
        
        This enhanced method detects various natural language expressions of confidence
        and converts them to a standardized numeric value between 0 and 1. It also returns
        the pattern or method used to extract the confidence for better traceability.
        
        Args:
            text: The text to extract confidence from
            
        Returns:
            Tuple of (confidence_value, extraction_method)
            - confidence_value: Float between 0 and 1
            - extraction_method: String describing how confidence was determined
        """
        # Default moderate confidence with fallback method
        default_confidence = 0.7  # Changed from 0.5 to align with framework specifications
        confidence = default_confidence
        method = "default"
        
        # First, check for structured confidence format
        # Look for patterns like:
        # confidence: 0.85
        # method: expert judgment based on clinical evidence
        structured_pattern = re.search(r'confidence:\s*(0\.\d+|\d+\.\d+|\d+)[\r\n\s]+method:\s*([^\n\r]+)', text.lower())
        if structured_pattern:
            try:
                confidence_value = float(structured_pattern.group(1))
                method_text = structured_pattern.group(2).strip()
                
                # Normalize confidence to [0,1] range
                if confidence_value > 1.0:
                    if confidence_value <= 10.0:
                        # If between 1 and 10, assume it's on a 0-10 scale
                        confidence_value = confidence_value / 10.0
                    else:
                        # If greater than 10, assume it's a percentage
                        confidence_value = min(1.0, confidence_value / 100.0)
                
                logger.debug(f"Extracted structured confidence: {confidence_value} with method: {method_text}")
                return float(confidence_value), f"structured-{method_text}"
            except (ValueError, IndexError):
                # Continue to other patterns if structured extraction fails
                logger.debug("Structured confidence extraction failed, falling back to patterns")
        
        # Normalize text - convert to lowercase and handle potential Unicode issues
        normalized_text = text.lower().strip()
        
        # Explicit numeric confidence patterns - expanded to catch more variations
        confidence_patterns = [
            # Direct numeric specification
            (r"confidence:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-labeled"),  # confidence: X.Y
            (r"confidence[\s]+level:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-level"),  # confidence level: X.Y
            (r"confidence[\s]+score:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-score"),  # confidence score: X.Y
            (r"with[\s]+(0\.\d+|\d+\.\d+|\d+)[\s]*confidence", "explicit-with"),  # with X.Y confidence
            (r"(0\.\d+|\d+\.\d+|\d+)[\s]*confidence", "explicit-numeric"),  # X.Y confidence
            (r"(0\.\d+|\d+\.\d+|\d+)[\s]*certainty", "explicit-certainty"),  # X.Y certainty
            (r"certainty:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-certainty-labeled"),  # certainty: X.Y
            (r"probability:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-probability"),  # probability: X.Y
            (r"likelihood:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-likelihood"),  # likelihood: X.Y
            (r"score:?[\s]*(0\.\d+|\d+\.\d+|\d+)", "explicit-score"),  # score: X.Y
            # More formats to capture confidence expressions
            (r"i[\s']*m[\s]*(0\.\d+|\d+\.\d+|\d+)[\s]*sure", "explicit-sure"),  # I'm X.Y sure
            (r"i[\s']*m[\s]*(0\.\d+|\d+\.\d+|\d+)[\s]*confident", "explicit-confident")  # I'm X.Y confident
        ]
        
        for pattern, method_name in confidence_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                try:
                    confidence = float(match.group(1))
                    # Normalize confidence to [0,1] range
                    if confidence > 1.0:
                        if confidence <= 10.0:
                            # If between 1 and 10, assume it's on a 0-10 scale
                            confidence = confidence / 10.0
                        else:
                            # If greater than 10, assume it's a percentage
                            confidence = min(1.0, confidence / 100.0)
                    
                    logger.debug(f"Extracted explicit confidence: {confidence} from pattern '{pattern}'")
                    method = method_name
                    return float(confidence), method
                except (ValueError, IndexError):
                    # Continue to next pattern if conversion fails
                    pass
        
        # Natural language confidence expressions - expanded with more variations and assigned base values
        high_confidence_phrases = [
            (r"\bhigh confidence\b", 0.85), 
            (r"\bvery confident\b", 0.9), 
            (r"\bstrong(ly)? believe\b", 0.8), 
            (r"\bcertain\b", 0.95), 
            (r"\bdefinitely\b", 0.95), 
            (r"\bwithout doubt\b", 0.95), 
            (r"\bconvinced\b", 0.85),
            (r"\bextremely likely\b", 0.9), 
            (r"\bvery high probability\b", 0.9),
            (r"\bvery sure\b", 0.9),
            (r"\babsolutely\b", 0.95),
            (r"\bno doubt\b", 0.95),
            (r"\bvery high confidence\b", 0.95),
            (r"\bhighly confident\b", 0.9),
            (r"\bconfident\b", 0.8)
        ]
        
        medium_confidence_phrases = [
            (r"\bmoderate confidence\b", 0.6), 
            (r"\breasonably confident\b", 0.65), 
            (r"\bfairly certain\b", 0.65), 
            (r"\bprobably\b", 0.6), 
            (r"\blikely\b", 0.65), 
            (r"\bplausible\b", 0.55), 
            (r"\breasonable\b", 0.6),
            (r"\bmiddle ground\b", 0.5), 
            (r"\bmedium certainty\b", 0.5),
            (r"\bmoderate certainty\b", 0.55),
            (r"\bsomewhat confident\b", 0.55),
            (r"\bcautiously optimistic\b", 0.6),
            (r"\bgood chance\b", 0.65)
        ]
        
        low_confidence_phrases = [
            (r"\blow confidence\b", 0.25), 
            (r"\bnot (very )?confident\b", 0.2), 
            (r"\buncertain\b", 0.3), 
            (r"\bunclear\b", 0.25), 
            (r"\bquestionable\b", 0.3), 
            (r"\bdoubtful\b", 0.2), 
            (r"\bimprobable\b", 0.15),
            (r"\bunlikely\b", 0.2), 
            (r"\bsomewhat\b", 0.4), 
            (r"\bpossibly\b", 0.4), 
            (r"\bperhaps\b", 0.4),
            (r"\bslim chance\b", 0.2),
            (r"\btentative\b", 0.3),
            (r"\bdoubt\b", 0.2),
            (r"\bskeptical\b", 0.25),
            (r"\bunsure\b", 0.3)
        ]
        
        # Check for high confidence expressions
        for phrase, base_value in high_confidence_phrases:
            if re.search(phrase, normalized_text):
                # Add small random variation for more natural distribution
                variation = random.uniform(-0.05, 0.05)
                confidence = min(0.95, max(0.75, base_value + variation))  # Keep within reasonable range
                logger.debug(f"Extracted high confidence: {confidence:.2f} from phrase matching '{phrase}'")
                return confidence, f"phrase-high-{phrase}"
                
        # Check for medium confidence expressions
        for phrase, base_value in medium_confidence_phrases:
            if re.search(phrase, normalized_text):
                variation = random.uniform(-0.05, 0.05)
                confidence = min(0.7, max(0.4, base_value + variation))  # Keep within reasonable range
                logger.debug(f"Extracted medium confidence: {confidence:.2f} from phrase matching '{phrase}'")
                return confidence, f"phrase-medium-{phrase}"
                
        # Check for low confidence expressions
        for phrase, base_value in low_confidence_phrases:
            if re.search(phrase, normalized_text):
                variation = random.uniform(-0.05, 0.05)
                confidence = min(0.4, max(0.1, base_value + variation))  # Keep within reasonable range
                logger.debug(f"Extracted low confidence: {confidence:.2f} from phrase matching '{phrase}'")
                return confidence, f"phrase-low-{phrase}"
        
        # Advanced percentage pattern matching
        percentage_patterns = [
            (r"(\d+)[\s]*%[\s]*(?:confidence|certain|sure|likelihood)", "percentage-explicit"),  # 90% confidence
            (r"(?:confidence|certain|sure|likelihood)[\s]*(?:of|is|at)[\s]*(\d+)[\s]*%", "percentage-inverted"),  # confidence of 90%
            (r"(\d+)[\s]*percent[\s]*(?:confidence|certain|sure|likelihood)", "percentage-word"),  # 90 percent confidence
            (r"(?:confidence|certain|sure|likelihood)[\s]*(?:of|is|at)[\s]*(\d+)[\s]*percent", "percentage-word-inverted"),  # confidence of 90 percent
            (r"(\d+)[\s]*%", "percentage-only")  # Just a percentage number like 90%
        ]
        
        for pattern, method_name in percentage_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                try:
                    percentage = float(match.group(1))
                    confidence = min(1.0, percentage / 100.0)
                    logger.debug(f"Extracted percentage confidence: {confidence:.2f} from '{match.group(0)}'")
                    return float(confidence), method_name
                except (ValueError, IndexError):
                    pass
        
        # Check for numerical scales (1-5, 1-10)
        scale_patterns = [
            (r"([1-5])[\s]*/[\s]*5", "scale-1-5"),  # 4/5 scale
            (r"([1-9]|10)[\s]*/[\s]*10", "scale-1-10"),  # 8/10 scale
            (r"rating[\s]*:?[\s]*([1-5])[\s]*/[\s]*5", "rating-1-5"),  # rating: 4/5
            (r"rating[\s]*:?[\s]*([1-9]|10)[\s]*/[\s]*10", "rating-1-10")  # rating: 8/10
        ]
        
        for pattern, method_name in scale_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                try:
                    value = float(match.group(1))
                    if "/5" in match.group(0) or "scale-1-5" in method_name or "rating-1-5" in method_name:
                        confidence = value / 5.0
                    else:  # 10-point scale
                        confidence = value / 10.0
                    logger.debug(f"Extracted scale confidence: {confidence:.2f} from '{match.group(0)}'")
                    return float(confidence), method_name
                except (ValueError, IndexError):
                    pass
        
        # Content-based confidence estimation as a last resort
        if confidence == default_confidence:
            # Look for hedging language which indicates lower confidence
            hedging_phrases = ["might", "may", "could", "possibly", "perhaps", "i think", "potentially"]
            certainty_phrases = ["certainly", "definitely", "clearly", "obviously", "undoubtedly"]
            
            hedge_count = sum(1 for phrase in hedging_phrases if phrase in normalized_text)
            certainty_count = sum(1 for phrase in certainty_phrases if phrase in normalized_text)
            
            # Adjust confidence based on hedging and certainty language
            if hedge_count > certainty_count:
                confidence = max(0.3, default_confidence - (0.1 * (hedge_count - certainty_count)))
                method = "content-hedging"
            elif certainty_count > hedge_count:
                confidence = min(0.9, default_confidence + (0.1 * (certainty_count - hedge_count)))
                method = "content-certainty"
        
        logger.debug(f"No explicit confidence found, using {method} confidence: {confidence}")
        return float(confidence), method
    
    @staticmethod
    def categorize_confidence(confidence_value: float) -> str:
        """
        Categorize a numeric confidence value into qualitative levels.
        
        Args:
            confidence_value: A confidence value between 0 and 1
            
        Returns:
            A qualitative confidence level: "high", "medium", or "low"
        """
        if confidence_value >= 0.7:
            return "high"
        elif confidence_value <= 0.4:  # Changed from 0.3 to 0.4 for better distribution
            return "low"
        else:
            return "medium"
    
    @staticmethod
    def extract_from_reasoning_steps(reasoning_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process reasoning steps to extract and add confidence values.
        
        Args:
            reasoning_steps: A list of reasoning step dictionaries
            
        Returns:
            The reasoning steps with added confidence values and extraction metadata
        """
        processed_steps = []
        
        for i, step in enumerate(reasoning_steps):
            # Get the reasoning text
            reasoning_text = step.get('reasoning', '')
            
            # Extract confidence with method information
            confidence, extraction_method = ConfidenceExtractor.extract_confidence(reasoning_text)
            
            # Add confidence to the step with additional metadata
            processed_step = dict(step)
            processed_step['confidence'] = confidence
            processed_step['confidence_category'] = ConfidenceExtractor.categorize_confidence(confidence)
            processed_step['confidence_extraction_method'] = extraction_method
            
            # For backward compatibility
            processed_step['original_confidence'] = confidence
            
            # Log extracted confidence
            logger.info(f"Step {i+1}: Extracted confidence {confidence:.2f} ({processed_step['confidence_category']}) using method: {extraction_method}")
            
            processed_steps.append(processed_step)
            
        return processed_steps
    
    @staticmethod
    def analyze_confidence_distribution(reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of confidence values across reasoning steps.
        
        Args:
            reasoning_steps: A list of reasoning step dictionaries with confidence values
            
        Returns:
            Dictionary with confidence distribution metrics
        """
        # Extract confidence values
        confidence_values = [step.get('confidence', step.get('original_confidence', 0.7)) for step in reasoning_steps if 'confidence' in step or 'original_confidence' in step]
        
        if not confidence_values:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        # Calculate statistics
        avg_confidence = sum(confidence_values) / len(confidence_values)
        min_confidence = min(confidence_values)
        max_confidence = max(confidence_values)
        
        # Categorize each confidence value
        categories = [ConfidenceExtractor.categorize_confidence(c) for c in confidence_values]
        distribution = {
            "high": categories.count("high"),
            "medium": categories.count("medium"),
            "low": categories.count("low")
        }
        
        return {
            "count": len(confidence_values),
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "distribution": distribution
        }
