#!/usr/bin/env python3
"""
Simple test runner for RAG integration tests
Can be used locally or in CI/CD environments
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

async def main():
    """Run the integration tests"""
    try:
        # Import and run the integration tests
        from test_integration_rag import main as run_tests
        await run_tests()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have all dependencies installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Running RAG Integration Tests Locally")
    print("=" * 50)
    asyncio.run(main()) 