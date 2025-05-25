"""
Knowledge Base Module for 3WayCoT Framework

This module implements a knowledge base that stores and retrieves information
for use in the reasoning process.
"""
from typing import Dict, List, Any, Optional
import json
import os

class KnowledgeBase:
    """
    A simple in-memory knowledge base for storing and retrieving information.
    
    The knowledge base can be initialized from a file and provides methods
    for querying and updating the stored knowledge.
    """
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            file_path: Optional path to a JSON file containing initial knowledge
        """
        self.knowledge: Dict[str, Any] = {}
        self.file_path = file_path
        
        if file_path and os.path.exists(file_path):
            self.load_from_file(file_path)
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load knowledge from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            self.file_path = file_path
        except Exception as e:
            print(f"Warning: Could not load knowledge base from {file_path}: {e}")
            self.knowledge = {}
    
    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """
        Save the current knowledge to a JSON file.
        
        Args:
            file_path: Path to save the file. Uses the original path if not provided.
        """
        save_path = file_path or self.file_path
        if not save_path:
            raise ValueError("No file path provided for saving the knowledge base")
            
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base to {save_path}: {e}")
    
    def query(self, key: str, default: Any = None) -> Any:
        """
        Query the knowledge base for a specific key.
        
        Args:
            key: The key to look up in the knowledge base
            default: Default value to return if key is not found
            
        Returns:
            The value associated with the key, or the default value if not found
        """
        return self.knowledge.get(key, default)
    
    def update(self, key: str, value: Any) -> None:
        """
        Update the knowledge base with a new key-value pair.
        
        Args:
            key: The key to update
            value: The new value to associate with the key
        """
        self.knowledge[key] = value
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for entries matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching entries with relevance scores
        """
        # Simple string matching implementation
        # In a real implementation, this could use more sophisticated search
        query = query.lower()
        results = []
        
        for key, value in self.knowledge.items():
            if query in key.lower() or (isinstance(value, str) and query in value.lower()):
                results.append({
                    'key': key,
                    'value': value,
                    'relevance': 1.0  # Simple binary relevance
                })
        
        return results
    
    def clear(self) -> None:
        """Clear all knowledge from the knowledge base."""
        self.knowledge = {}
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the knowledge base."""
        return key in self.knowledge
    
    def __getitem__(self, key: str) -> Any:
        """Get an item from the knowledge base using dict-like access."""
        return self.knowledge[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in the knowledge base using dict-like access."""
        self.knowledge[key] = value
    
    def __delitem__(self, key: str) -> None:
        """Delete an item from the knowledge base using dict-like access."""
        del self.knowledge[key]
    
    def __len__(self) -> int:
        """Get the number of items in the knowledge base."""
        return len(self.knowledge)
    
    def __str__(self) -> str:
        """Get a string representation of the knowledge base."""
        return f"KnowledgeBase with {len(self)} entries"
