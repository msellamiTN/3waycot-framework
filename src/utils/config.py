"""
Configuration module for the 3WayCoT framework.

This module handles loading configuration settings for the LLM providers
and other framework components.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("3WayCoT.Config")

class Config:
    """
    Configuration manager for the 3WayCoT framework.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a config file. If not provided,
                       looks for config in default locations.
        """
        self.config = {}
        self.config_path = config_path
        
        # Try to load config from specified path
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            # Try to load from default locations
            default_locations = [
                Path.cwd() / "config.json",
                Path.home() / ".3waycot" / "config.json",
                Path(__file__).parent.parent.parent / "config.json"
            ]
            
            for path in default_locations:
                if path.exists():
                    self._load_config(str(path))
                    break
        
        # Load API keys from environment variables if not in config
        self._load_from_env()
    
    def _load_config(self, path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to the configuration file
        """
        try:
            with open(path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {path}: {e}")
            self.config = {}
    
    def _load_from_env(self) -> None:
        """
        Load API keys from environment variables if not in config.
        """
        # OpenAI
        if not self.config.get("openai", {}).get("api_key"):
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                if "openai" not in self.config:
                    self.config["openai"] = {}
                self.config["openai"]["api_key"] = openai_key
                self.config["openai"]["is_configured"] = True
        
        # Gemini
        if not self.config.get("gemini", {}).get("api_key"):
            google_key = os.environ.get("GOOGLE_API_KEY")
            if google_key:
                if "gemini" not in self.config:
                    self.config["gemini"] = {}
                self.config["gemini"]["api_key"] = google_key
                self.config["gemini"]["is_configured"] = True
                self.config["gemini"]["model"] = "gemini-1.5-flash"
        
        # Anthropic
        if not self.config.get("anthropic", {}).get("api_key"):
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                if "anthropic" not in self.config:
                    self.config["anthropic"] = {}
                self.config["anthropic"]["api_key"] = anthropic_key
                self.config["anthropic"]["is_configured"] = True
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific LLM provider.
        
        Args:
            provider: Name of the provider (e.g., "openai", "gemini")
            
        Returns:
            Dictionary with provider configuration
        """
        # Get provider config with defaults
        provider_config = self.config.get(provider, {}).copy()
        
        # Set defaults based on provider
        if provider == "openai":
            provider_config.setdefault("model", "gpt-4")
            provider_config.setdefault("temperature", 0.7)
            provider_config.setdefault("max_tokens", 1000)
            provider_config.setdefault("is_configured", bool(provider_config.get("api_key")))
        
        elif provider == "gemini":
            provider_config.setdefault("model", "gemini-1.5-flash")
            provider_config.setdefault("temperature", 0.7)
            provider_config.setdefault("max_tokens", 1000)
            provider_config.setdefault("top_p", 1.0)
            provider_config.setdefault("is_configured", bool(provider_config.get("api_key")))
        
        elif provider == "anthropic":
            provider_config.setdefault("model", "claude-3-opus-20240229")
            provider_config.setdefault("temperature", 0.7)
            provider_config.setdefault("max_tokens", 1000)
            provider_config.setdefault("is_configured", bool(provider_config.get("api_key")))
        
        return provider_config
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration. If None, use the current config_path.
            
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.config_path
        
        if not save_path:
            logger.warning("No path provided to save configuration")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save config without API keys for security
            safe_config = self.config.copy()
            
            # Remove API keys from saved config
            for provider in ["openai", "gemini", "anthropic"]:
                if provider in safe_config and "api_key" in safe_config[provider]:
                    safe_config[provider]["api_key"] = "<REDACTED>"
            
            with open(save_path, 'w') as f:
                json.dump(safe_config, f, indent=2)
            
            logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
