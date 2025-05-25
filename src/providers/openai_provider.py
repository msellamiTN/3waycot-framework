"""OpenAI provider implementation for the 3WayCoT framework."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import httpx

from ..core.cot_generator import ProviderBase

logger = logging.getLogger(__name__)

class OpenAIProvider(ProviderBase):
    """Provider for OpenAI's GPT models.
    
    This provider supports all OpenAI chat completion models.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """Initialize the OpenAI provider.
        
        Args:
            model: The model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            api_key: Optional API key. If not provided, will use OPENAI_API_KEY env var.
        """
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to generate text from.
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness (0.0 to 2.0).
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text.
            
        Raises:
            ValueError: If the API key is not set or the request fails.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": max(0.0, min(2.0, temperature)),
            **kwargs
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" not in result or not result["choices"]:
                    raise ValueError("Invalid response format from OpenAI API")
                
                return result["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenAI API request failed with status {e.response.status_code}"
            if e.response.text:
                try:
                    error_data = e.response.json()
                    error_msg = f"{error_msg}: {error_data.get('error', {}).get('message', 'Unknown error')}"
                except json.JSONDecodeError:
                    error_msg = f"{error_msg}: {e.response.text}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except (httpx.RequestError, json.JSONDecodeError) as e:
            error_msg = f"OpenAI API request failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
