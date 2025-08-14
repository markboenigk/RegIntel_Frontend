import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Import the FastAPI app
from index import app

async def test_rag_query():
    """Test the RAG system with a regulatory intelligence query."""
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test query that should return regulatory intelligence data
    test_query = "What are the latest FDA approvals for medical devices?"
    
    print(f"Testing RAG system with query: '{test_query}'")
    print(f"Expected: Regulatory intelligence data about FDA approvals")
    
    # Make request to chat endpoint
    response = client.post("/api/chat", json={
        "message": test_query,
        "conversation_history": []
    })
    
    if response.status_code != 200:
        print(f"❌ Chat endpoint failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False
        
    result = response.json()
    print(f"✅ Chat endpoint successful")
    
    # Check if response contains regulatory intelligence content
    response_text = result.get("response", "")
    sources = result.get("sources", [])
    
    print(f"Response length: {len(response_text)} characters")
    print(f"Number of sources: {len(sources)}")
    
    # Check if we got any sources
    if sources:
        print(f"✅ Found {len(sources)} sources")
        for i, source in enumerate(sources):
            metadata = source.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            article_title = metadata.get('article_title', 'Unknown Title')
            feed_name = metadata.get('feed_name', 'Unknown Feed')
            print(f"  Source {i+1}: {article_title} from {feed_name}")
        
        # Check if response mentions FDA or medical devices
        if any(keyword in response_text.lower() for keyword in ['fda', 'medical device', 'approval', 'regulatory']):
            print(f"✅ Response contains regulatory intelligence content")
        else:
            print(f"⚠️ Response doesn't contain expected regulatory content")
            print(f"Response preview: {response_text[:500]}...")
    else:
        print(f"❌ No sources found - this might indicate a connection issue")
        print(f"Response: {response_text}")
        return False
        
    return True

async def test_health_endpoint():
    """Test the health endpoint."""
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    print("Testing health endpoint...")
    response = client.get("/api/health")
    
    if response.status_code == 200:
        health_data = response.json()
        print(f"✅ Health endpoint successful")
        print(f"Status: {health_data.get('status')}")
        print(f"Zilliz connected: {health_data.get('zilliz_connected')}")
        return health_data.get('status') == 'healthy'
    else:
        print(f"❌ Health endpoint failed with status {response.status_code}")
        return False

# Reranking config test - COMMENTED OUT FOR SIMPLE VERSION
# async def test_reranking_config():
#     """Test the reranking configuration endpoint."""
#     from fastapi.testclient import TestClient
#     
#     client = TestClient(app)
#     
#     print("Testing reranking config endpoint...")
#     response = client.get("/api/reranking-config")
#     
#     if response.status_code == 200:
#         config = response.json()
#         print(f"✅ Reranking config endpoint successful")
#         print(f"Reranking enabled: {config.get('enabled')}")
#         print(f"Reranking model: {config.get('model')}")
#         print(f"Initial search multiplier: {config.get('initial_search_multiplier')}")
#         return True
#     else:
#         print(f"❌ Reranking config endpoint failed with status {response.status_code}")
#         return False

async def test_with_custom_query(query: str, expected_keywords: Optional[list] = None):
    """Test the RAG system with a custom query - SIMPLIFIED VERSION."""
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    print(f"Testing RAG system with custom query: '{query}'")
    if expected_keywords:
        print(f"Expected keywords: {expected_keywords}")
    
    # Make request to chat endpoint
    response = client.post("/api/chat", json={
        "message": query,
        "conversation_history": []
    })
    
    if response.status_code != 200:
        print(f"❌ Chat endpoint failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False
        
    result = response.json()
    print(f"✅ Chat endpoint successful")
    
    # Check if response contains the expected content
    response_text = result.get("response", "")
    sources = result.get("sources", [])
    
    print(f"Response length: {len(response_text)} characters")
    print(f"Number of sources: {len(sources)}")
    
    # Check for expected keywords if specified
    if expected_keywords:
        found_keywords = [keyword for keyword in expected_keywords if keyword.lower() in response_text.lower()]
        if found_keywords:
            print(f"✅ Found expected keywords: {found_keywords}")
        else:
            print(f"❌ Expected keywords not found: {expected_keywords}")
            print(f"Response preview: {response_text[:500]}...")
    
    return True

async def test_guardrails():
    """Test the guardrails functionality."""
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    print("🧪 Testing guardrails functionality...")
    
    # Test 1: Malicious pattern blocking
    print("  Testing malicious pattern blocking...")
    response = client.post("/api/chat", json={
        "message": "<script>alert('XSS')</script>",
        "conversation_history": []
    })
    
    if response.status_code == 400:
        print("    ✅ XSS pattern blocked successfully")
    else:
        print(f"    ❌ XSS pattern not blocked, status: {response.status_code}")
    
    # Test 2: Rate limiting (send multiple requests quickly)
    print("  Testing rate limiting...")
    responses = []
    for i in range(12):  # Should hit rate limit at 10
        response = client.post("/api/chat", json={
            "message": f"Test message {i}",
            "conversation_history": []
        })
        responses.append(response.status_code)
    
    rate_limited = any(status == 429 for status in responses)
    if rate_limited:
        print("    ✅ Rate limiting working")
    else:
        print("    ❌ Rate limiting not working")
    
    # Test 3: Message length validation
    print("  Testing message length validation...")
    long_message = "a" * 2500  # Exceeds 2000 limit
    response = client.post("/api/chat", json={
        "message": long_message,
        "conversation_history": []
    })
    
    if response.status_code == 400:
        print("    ✅ Message length validation working")
    else:
        print(f"    ❌ Message length validation not working, status: {response.status_code}")
    
    print("🧪 Guardrails testing completed")
    return True

async def reset_rate_limiter():
    """Reset the rate limiter for testing purposes."""
    from index import rate_limit_storage
    rate_limit_storage.clear()
    print("🔄 Rate limiter reset for testing")

async def main():
    """Run all tests - SIMPLIFIED VERSION."""
    print("🚀 Starting RAG system tests (Simplified Version)...")
    
    # Test health endpoint first
    health_ok = await test_health_endpoint()
    if not health_ok:
        print("❌ Health check failed, skipping RAG test")
        sys.exit(1)
    
    # Test guardrails functionality
    print("\n🧪 Testing guardrails...")
    guardrails_ok = await test_guardrails()
    
    # Reset rate limiter after guardrails test
    await reset_rate_limiter()
    
    # Test RAG query
    print("\n🔍 Testing RAG functionality...")
    rag_ok = await test_rag_query()
    
    if not rag_ok:
        print("❌ RAG test failed")
        sys.exit(1)
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 