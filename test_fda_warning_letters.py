#!/usr/bin/env python3
"""
Test script for FDA Warning Letters collection
Tests the collection with its specific schema
"""

import asyncio
import os
import sys

# Add the current directory to Python path
sys.path.append('.')

from index import search_similar_documents, chat_with_gpt

async def test_fda_warning_letters():
    """Test the FDA warning letters collection"""
    print("🧪 Testing FDA Warning Letters Collection...")
    
    # Test 1: Test search function directly
    print("\n🔍 Test 1: Testing search_similar_documents function")
    try:
        sources = await search_similar_documents("FDA warning", limit=5, collection_name="fda_warning_letters")
        print(f"✅ Search function returned {len(sources)} sources")
        
        if sources:
            print("📰 Sources found:")
            for i, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                print(f"  {i}. Company: {metadata.get('company_name', 'No company')}")
                print(f"     Date: {metadata.get('letter_date', 'No date')}")
                print(f"     Type: {metadata.get('chunk_type', 'No type')}")
                print(f"     Violations: {metadata.get('violations', [])}")
        else:
            print("❌ No sources found - collection might be empty")
            
    except Exception as e:
        print(f"❌ Search function error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Test chat function if we have sources
    print("\n💬 Test 2: Testing chat_with_gpt function")
    try:
        if sources:
            response = await chat_with_gpt("What FDA warning letters do you have?", [], sources)
            print(f"✅ Chat function returned response: {response[:100]}...")
        else:
            print("⚠️  Skipping chat test - no sources available")
            
    except Exception as e:
        print(f"❌ Chat function error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Test with different query terms
    print("\n🔍 Test 3: Testing different query terms")
    test_queries = [
        "FDA warning",
        "violations",
        "company violations",
        "regulatory issues",
        "product recalls"
    ]
    
    for query in test_queries:
        try:
            print(f"\n   Testing query: '{query}'")
            sources = await search_similar_documents(query, limit=3, collection_name="fda_warning_letters")
            print(f"   Found {len(sources)} sources")
            
            if sources:
                for source in sources[:2]:  # Show first 2 sources
                    metadata = source.get('metadata', {})
                    company = metadata.get('company_name', 'Unknown')
                    date = metadata.get('letter_date', 'Unknown')
                    print(f"     - {company} ({date})")
            else:
                print("     No sources found")
                
        except Exception as e:
            print(f"     Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fda_warning_letters()) 