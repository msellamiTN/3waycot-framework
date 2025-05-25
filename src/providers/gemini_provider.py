"""Gemini provider implementation for the 3WayCoT framework.

This module provides integration with Google's Gemini models through the Google Generative AI API.
It implements the ProviderBase interface and supports all Gemini models with configurable
safety settings and generation parameters.

Example:
    >>> from src.providers.gemini_provider import GeminiProvider
    >>> import asyncio
    >>>
    >>> async def example():
    ...     provider = GeminiProvider()
    ...     response = await provider.generate("Hello, world!")
    ...     print(response)
    ...
    >>> asyncio.run(example())
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import google.generativeai as genai

from ..core.cot_generator import ProviderBase

logger = logging.getLogger(__name__)

class GeminiProvider(ProviderBase):
    """Provider for Google's Gemini models.
    
    This provider supports all Gemini models available through the Google Generative AI API,
    including text and multimodal capabilities. It handles API communication, error handling,
    and response processing.
    
    Attributes:
        model_name (str): The name of the Gemini model being used.
        api_key (str): The Google API key for authentication.
        safety_settings (list): List of safety settings for content generation.
        generation_config (dict): Configuration for text generation parameters.
        model: The loaded Gemini model instance.
    
    Example:
        >>> provider = GeminiProvider(model="gemini-1.5-pro")
        >>> response = await provider.generate("Explain quantum computing")
    """
    
    def __init__(
        self, 
        model: str = "gemini-pro",
        api_key: Optional[str] = None,
        safety_settings: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Gemini provider with the specified configuration.
        
        Args:
            model (str, optional): The model to use. Defaults to "gemini-pro".
                Other options include 'gemini-1.5-pro' and 'gemini-2.0-flash'.
            api_key (str, optional): Google API key. If not provided, will use
                the GOOGLE_API_KEY environment variable.
            safety_settings (list, optional): Custom safety settings for content
                generation. If not provided, default safety settings will be used.
                Each setting should be a dictionary with 'category' and 'threshold' keys.
            generation_config (dict, optional): Custom configuration for generation
                parameters. If not provided, default values will be used.
                
        Raises:
            ValueError: If the API key is not provided and not found in environment variables.
            RuntimeError: If there is an error initializing the Gemini model.
                
        Example:
            >>> # Initialize with custom settings
            >>> safety_settings = [{
            ...     "category": "HARM_CATEGORY_HARASSMENT",
            ...     "threshold": "BLOCK_ONLY_HIGH"
            ... }]
            >>> provider = GeminiProvider(
            ...     model="gemini-1.5-pro",
            ...     safety_settings=safety_settings,
            ...     generation_config={"temperature": 0.5}
            ... )
        """
        self.model_name = model
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            logger.warning("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
        
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        
        # Default safety settings if not provided
        self.safety_settings = safety_settings or [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Default generation config if not provided
        self.generation_config = generation_config or {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            logger.info(f"Initialized Gemini provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using the Gemini API.
        
        This method sends a text generation request to the Gemini API and returns
        the generated response. It handles API communication, error handling, and
        response processing.
        
        Args:
            prompt (str): The prompt to generate text from.
            max_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to 1000. The maximum allowed is 8192 for most models.
            temperature (float, optional): Controls randomness in generation.
                Lower values make output more deterministic. Must be between 0 and 1.
                Defaults to 0.7.
            **kwargs: Additional parameters to pass to the API, including:
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter
                - candidate_count (int): Number of responses to generate
                
        Returns:
            str: The generated text response.
            
        Raises:
            ValueError: If the API key is not set, the request is invalid, or
                the response cannot be processed.
                
        Example:
            >>> response = await provider.generate(
            ...     "Write a haiku about artificial intelligence",
            ...     temperature=0.9,
            ...     max_tokens=100
            ... )
        """
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        try:
            # Update generation config with provided parameters
            gen_config = {
                **self.generation_config,
                "max_output_tokens": max_tokens,
                "temperature": max(0.0, min(1.0, temperature)),
                **{k: v for k, v in kwargs.items() if k in ["top_p", "top_k", "candidate_count"]}
            }
            
            # Create a new model instance with updated config
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Generate content
            response = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: model.generate_content(prompt)
            )
            
            # Check for safety issues
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                if response.prompt_feedback.block_reason:
                    raise ValueError(
                        f"Content blocked due to: {response.prompt_feedback.block_reason}. "
                        f"Safety ratings: {response.prompt_feedback.safety_ratings}"
                    )
            
            # Extract the generated text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected response format from Gemini API")
                
        except Exception as e:
            error_msg = f"Gemini API request failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    async def batch_generate(
        self, 
        prompts: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of prompts to generate text from.
            max_tokens: Maximum number of tokens to generate per prompt.
            temperature: Controls randomness (0.0 to 1.0).
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            List of generated texts.
        """
        tasks = [
            self.generate(prompt, max_tokens, temperature, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

# Example usage
async def example_usage():
    """Example showing how to use the Gemini provider."""
    # Initialize the provider
    provider = GeminiProvider()
    
    # Generate a response
    response = await provider.generate(
        "Explain the concept of Chain of Thought reasoning",
        temperature=0.7,
        max_tokens=500
    )
    
    print("Generated response:")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
