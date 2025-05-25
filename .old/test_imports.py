"""Test script to verify imports and basic functionality."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Try importing the main components
try:
    from src.core import ThreeWayCOT, ChainOfThoughtGenerator
    from src.providers import GeminiProvider
    
    print("✅ Successfully imported core components")
    print(f"- ThreeWayCOT: {ThreeWayCOT.__module__}.{ThreeWayCOT.__name__}")
    print(f"- ChainOfThoughtGenerator: {ChainOfThoughtGenerator.__module__}.{ChainOfThoughtGenerator.__name__}")
    print(f"- GeminiProvider: {GeminiProvider.__module__}.{GeminiProvider.__name__}")
    
    # Test environment variables
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    print("\n✅ Environment variables loaded")
    print(f"- GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")
    print(f"- GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'Not set')}")
    
    # Test Gemini provider initialization
    try:
        provider = GeminiProvider(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("\n✅ Gemini provider initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize Gemini provider: {e}")
        raise
        
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nCurrent Python path:")
    for path in sys.path:
        print(f"- {path}")
    raise
