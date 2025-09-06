import os
import json
import requests
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict
import openai
from dotenv import load_dotenv
from datetime import datetime
from auth.routes import router as auth_router

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Collection names - configurable via environment variables
FDA_WARNING_LETTERS_COLLECTION = os.getenv("FDA_WARNING_LETTERS_COLLECTION", "fda_warning_letters")
RSS_FEEDS_COLLECTION = os.getenv("RSS_FEEDS_COLLECTION", "rss_feeds")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", RSS_FEEDS_COLLECTION)

# RAG Configuration
STRICT_RAG_ONLY = os.getenv("STRICT_RAG_ONLY", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "o3")
INITIAL_SEARCH_MULTIPLIER = int(os.getenv("INITIAL_SEARCH_MULTIPLIER", "3"))

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Pydantic models
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Conversation history")

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources")

class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    text: str = Field(..., description="Document text to add")
    metadata: str = Field(default="", description="Optional metadata for the document")

# Create FastAPI app
app = FastAPI(
    title="RegIntel RAG API",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include auth router
app.include_router(auth_router)

# Utility functions
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model="text-embedding-ada-002",  # Using ada-002 to generate exactly 1536 dimensions to match collection schema
            input=text
        )
        embedding = response.data[0].embedding
        print(f"🔍 DEBUG: Model used: text-embedding-ada-002")
        print(f"🔍 DEBUG: Embedding dimensions: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

async def search_fda_warning_letters_pgvector(query_embedding: List[float], search_limit: int, is_latest_query: bool) -> List[Dict[str, Any]]:
    """Search FDA warning letters using Supabase pgvector."""
    try:
        from auth.config import get_supabase_config
        supabase = get_supabase_config().get_client()
        
        print(f"🔍 DEBUG: Searching FDA warning letters with pgvector")
        print(f"🔍 DEBUG: Search limit: {search_limit}")
        
        # Convert embedding to list format for pgvector
        embedding_list = [float(x) for x in query_embedding]
        
        # Use direct pgvector query (fallback if RPC function doesn't exist)
        try:
            response = supabase.rpc('search_warning_letters', {
                'query_embedding': embedding_list,
                'match_threshold': 0.1,
                'match_count': search_limit
            }).execute()
        except Exception as rpc_error:
            print(f"⚠️ DEBUG: RPC function not available, using direct query: {rpc_error}")
            # Fallback to direct query with limit to avoid performance issues
            response = supabase.table('warning_letters_vectors').select('*').limit(search_limit * 2).execute()
        
        if not response.data:
            print("❌ DEBUG: No data returned from pgvector search")
            return []
        
        print(f"🔍 DEBUG: Found {len(response.data)} results from pgvector")
        
        sources = []
        for row in response.data:
            try:
                # Parse the text_vector back to list if needed
                text_vector = row.get('text_vector')
                if isinstance(text_vector, str):
                    # Parse JSON string to list
                    import json
                    text_vector = json.loads(text_vector)
                
                # Debug: Check the actual data structure
                print(f"🔍 DEBUG: Row data - letter_date: {row.get('letter_date')}, type: {type(row.get('letter_date'))}")
                
                metadata = {
                    "company_name": row.get('company_name', 'Unknown Company'),
                    "letter_date": row.get('letter_date', 'Unknown Date'),
                    "chunk_type": row.get('chunk_type', 'Unknown Type'),
                    "chunk_id": row.get('chunk_id', 'Unknown Chunk'),
                    "warning_letter_id": row.get('warning_letter_id', 'Unknown ID'),
                    "violations": row.get('violations', '{}'),
                    "required_actions": row.get('required_actions', '{}'),
                    "systemic_issues": row.get('systemic_issues', '{}'),
                    "regulatory_consequences": row.get('regulatory_consequences', '{}')
                }
                
                source = {
                    "text": row.get('text_content', ''),
                    "metadata": metadata
                }
                sources.append(source)
                
            except Exception as e:
                print(f"❌ DEBUG: Error parsing row: {e}")
                continue
        
        # Sort by date if this is a latest query
        if is_latest_query:
            print(f"🔍 DEBUG: Sorting results by letter_date for latest query")
            try:
                def safe_date_sort(x):
                    metadata = x.get('metadata', {})
                    date_str = metadata.get('letter_date', '') if metadata else ''
                    print(f"🔍 DEBUG: Sorting - date_str: '{date_str}', type: {type(date_str)}")
                    
                    if not date_str or date_str == 'Unknown Date':
                        print(f"🔍 DEBUG: Using default date for: '{date_str}'")
                        return datetime(1900, 1, 1)  # Put unknown dates at the end
                    
                    # Try different date formats
                    date_formats = [
                        '%B %d, %Y',      # July 11, 2025, August 28, 2025
                        '%m/%d/%Y',       # 07/11/2025, 08/28/2025
                        '%Y-%m-%d',       # 2025-07-11, 2025-08-28
                    ]
                    
                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            print(f"🔍 DEBUG: Successfully parsed '{date_str}' with format '{fmt}' -> {parsed_date}")
                            return parsed_date
                        except ValueError:
                            continue
                    
                    # If no format matches, return a very old date
                    print(f"🔍 DEBUG: No format matched for '{date_str}', using default date")
                    return datetime(1900, 1, 1)
                
                sources.sort(key=safe_date_sort, reverse=True)
                # Trim to original limit after sorting
                sources = sources[:search_limit // 3]  # Convert back to original top_k
                print(f"🔍 DEBUG: After sorting and trimming: {len(sources)} sources")
            except Exception as sort_error:
                print(f"⚠️ DEBUG: Error sorting by date: {sort_error}")
                # Continue without sorting if there's an error
        
        print(f"🔍 DEBUG: Returning {len(sources)} FDA warning letter sources")
        return sources
        
    except Exception as e:
        print(f"❌ DEBUG: Error in FDA pgvector search: {e}")
        import traceback
        traceback.print_exc()
        return []

async def search_rss_feeds_pgvector(query_embedding: List[float], search_limit: int) -> List[Dict[str, Any]]:
    """Search RSS feeds using Supabase pgvector."""
    try:
        from auth.config import get_supabase_config
        supabase = get_supabase_config().get_client()
        
        print(f"🔍 DEBUG: Searching RSS feeds with pgvector")
        print(f"🔍 DEBUG: Search limit: {search_limit}")
        
        # Convert embedding to list format for pgvector
        embedding_list = [float(x) for x in query_embedding]
        
        # Use direct pgvector query for RSS feeds (fallback if RPC function doesn't exist)
        try:
            response = supabase.rpc('search_rss_feeds', {
                'query_embedding': embedding_list,
                'match_threshold': 0.1,
                'match_count': search_limit
            }).execute()
        except Exception as rpc_error:
            print(f"⚠️ DEBUG: RSS RPC function not available, using direct query: {rpc_error}")
            # Fallback to direct query - try rss_feed_vectors first, then rss_feeds_gold
            try:
                response = supabase.table('rss_feed_vectors').select('*').limit(search_limit * 2).execute()
                print(f"✅ DEBUG: Using rss_feed_vectors table")
            except Exception as vector_error:
                print(f"⚠️ DEBUG: rss_feed_vectors not accessible, using rss_feeds_gold: {vector_error}")
                response = supabase.table('rss_feeds_gold').select(
                    'article_feed_name,article_published_date,article_title,content_category'
                ).limit(search_limit * 2).execute()
        
        if not response.data:
            print("❌ DEBUG: No data returned from RSS pgvector search")
            return []
        
        print(f"🔍 DEBUG: Found {len(response.data)} RSS results from pgvector")
        
        sources = []
        for row in response.data:
            try:
                # Debug: Check the actual data structure and available fields
                print(f"🔍 DEBUG: RSS Row data - article_title: {row.get('article_title')}, published_date: {row.get('article_published_date')}")
                print(f"🔍 DEBUG: Available fields in row: {list(row.keys())}")
                
                metadata = {
                    "article_title": row.get('article_title', 'Unknown Title'),
                    "published_date": row.get('article_published_date', 'Unknown Date'),
                    "feed_name": row.get('article_feed_name', 'Unknown Feed'),
                    "chunk_type": row.get('content_category', 'Unknown Type'),
                    "companies": [],
                    "products": [],
                    "regulations": [],
                    "regulatory_bodies": []
                }
                
                # Try to get the best available text content from the vector table
                text_content = (row.get('text_content', '') or 
                              row.get('content', '') or 
                              row.get('article_content', '') or 
                              row.get('summary', '') or 
                              row.get('article_title', 'No content available'))
                
                print(f"🔍 DEBUG: Text content length: {len(text_content)}, preview: {text_content[:100]}...")
                
                source = {
                    "text": text_content,
                    "metadata": metadata
                }
                sources.append(source)
                
            except Exception as e:
                print(f"❌ DEBUG: Error parsing RSS row: {e}")
                continue
        
        # Sort sources by relevance (sources with search terms in title or content first)
        if sources:
            def relevance_score(source):
                text = source.get('text', '').lower()
                title = source.get('metadata', {}).get('article_title', '').lower()
                # Simple relevance scoring - prioritize sources with more content and relevant terms
                score = len(text) + len(title) * 2  # Title gets double weight
                return score
            
            sources.sort(key=relevance_score, reverse=True)
            print(f"🔍 DEBUG: Sorted sources by relevance")
        
        print(f"🔍 DEBUG: Returning {len(sources)} RSS feed sources")
        return sources
        
    except Exception as e:
        print(f"❌ DEBUG: Error in RSS pgvector search: {e}")
        import traceback
        traceback.print_exc()
        return []

async def search_similar_documents(query: str, collection_name: str = "rss_feeds", top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar documents using vector similarity with Supabase pgvector."""
    try:
        # Use specified collection or default
        target_collection = collection_name or DEFAULT_COLLECTION
        
        print(f"🔍 DEBUG: Starting search in collection: {target_collection}")
        print(f"🔍 DEBUG: Query: {query}")
        print(f"🔍 DEBUG: Limit: {top_k}")
        
        # Check if this is a "latest" query and increase search limit to get more results
        latest_keywords = ['latest', 'most recent', 'newest', 'recent', 'last']
        is_latest_query = any(keyword in query.lower() for keyword in latest_keywords)
        
        # Increase search limit for latest queries to get more results to sort
        search_limit = top_k * 3 if is_latest_query and target_collection == "fda_warning_letters" else top_k
        if is_latest_query and target_collection == "fda_warning_letters":
            print(f"🔍 DEBUG: Latest query detected, increasing search limit to {search_limit}")
        
        # Get query embedding for semantic search
        print(f"🔍 DEBUG: About to get embedding for query: '{query}'")
        query_embedding = await get_embedding(query)
        print(f'🔍 DEBUG: Embedding generated, length: {len(query_embedding) if query_embedding else 0}')
        if not query_embedding:
            print("❌ DEBUG: Failed to generate embedding, using fallback data")
            return get_fallback_sources(query, target_collection, top_k)
        print(f"🔍 DEBUG: Embedding successful, proceeding with search")

        # Use Supabase pgvector for search
        if target_collection == "fda_warning_letters":
            return await search_fda_warning_letters_pgvector(query_embedding, search_limit, is_latest_query)
        else:
            return await search_rss_feeds_pgvector(query_embedding, search_limit)
        
        # Use different schemas based on collection type
        if target_collection == "fda_warning_letters":
            # FDA Warning Letters schema
            output_fields = [
                "text_content", "company_name", "letter_date", "chunk_type", 
                "chunk_id", "violations", "required_actions", "systemic_issues",
                "regulatory_consequences", "product_types", "product_categories"
            ]
        else:
            # RSS Feeds schema (default)
            output_fields = [
                "text_content", "article_title", "published_date", "feed_name", 
                "chunk_type", "companies", "products", "regulations", "regulatory_bodies"
            ]
        
        # Convert to float32 array (Zilliz expects this)
        query_embedding_float32 = [float(x) for x in query_embedding]
        
        # Zilliz Cloud V2 search payload structure
        search_data = {
            "collectionName": target_collection,
            "data": [query_embedding_float32],  # Note: embedding should be wrapped in a list
            "limit": search_limit,
            "outputFields": output_fields,
            "metricType": "COSINE",
            "params": {"nprobe": 10},
            "fieldName": "text_vector"
        }
        
        print(f"🔍 DEBUG: Attempting vector search...")
        print(f"🔍 DEBUG: Query: {query}")
        print(f"🔍 DEBUG: Collection: {target_collection}")
        print(f"🔍 DEBUG: Search data: {json.dumps(search_data, indent=2)}")

        # Try different endpoints until one works with shorter timeout for Vercel
        search_successful = False
        for endpoint in search_endpoints:
            try:
                print(f"🔍 DEBUG: Trying endpoint: {endpoint}")
                response = requests.post(endpoint, json=search_data, headers=headers, timeout=15)
                print(f"🔍 DEBUG: Endpoint {endpoint} response status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ DEBUG: Search successful via {endpoint}")
                    search_successful = True
                    break
                else:
                    print(f"❌ DEBUG: Endpoint {endpoint} failed: {response.status_code}")
                    print(f"❌ DEBUG: Response text: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ DEBUG: Endpoint {endpoint} timed out")
                continue
            except Exception as endpoint_error:
                print(f"❌ DEBUG: Endpoint {endpoint} error: {str(endpoint_error)}")
                continue
        
        if not search_successful:
            print(f"❌ DEBUG: All search endpoints failed, using fallback data")
            return get_fallback_sources(query, target_collection, top_k)
        
        result = response.json()
        pretty_json_string = json.dumps(result, indent=4)
        print(f'🔍 DEBUG: Milvus raw response: {pretty_json_string}')
        
        # Check if this is an error response
        if 'code' in result and result.get('code') != 0:
            print(f"❌ DEBUG: Milvus API returned error: Code {result.get('code')}, Message: {result.get('message')}")
            print("🔄 DEBUG: Using fallback data due to API error")
            return get_fallback_sources(query, target_collection, top_k)

        sources = []
        if 'data' in result and result['data']:
            print(f"🔍 DEBUG: Found 'data' field in response with {len(result['data'])} items")
            
            for hit in result['data']:
                try:
                    # Create metadata based on collection schema
                    if target_collection == "fda_warning_letters":
                        # FDA Warning Letters metadata
                        metadata = {
                            "company_name": hit.get('company_name', 'Unknown Company'),
                            "letter_date": hit.get('letter_date', 'Unknown Date'),
                            "chunk_type": hit.get('chunk_type', 'Unknown Type'),
                            "chunk_id": hit.get('chunk_id', 'Unknown Chunk'),
                            "violations": hit.get('violations', []),
                            "required_actions": hit.get('required_actions', []),
                            "systemic_issues": hit.get('systemic_issues', []),
                            "regulatory_consequences": hit.get('regulatory_consequences', []),
                            "product_types": hit.get('product_types', []),
                            "product_categories": hit.get('product_categories', [])
                        }
                    else:
                        # RSS Feeds metadata (default)
                        metadata = {
                            "article_title": hit.get('article_title', 'Unknown Title'),
                            "published_date": hit.get('published_date', 'Unknown Date'),
                            "feed_name": hit.get('feed_name', 'Unknown Feed'),
                            "chunk_type": hit.get('chunk_type', 'Unknown Type'),
                            "companies": hit.get('companies', []),
                            "products": hit.get('products', []),
                            "regulations": hit.get('regulations', []),
                            "regulatory_bodies": hit.get('regulatory_bodies', [])
                        }
                    
                    source_item = {
                        "title": hit.get('article_title', metadata.get('company_name', 'Unknown Title')),
                        "content": hit.get('text_content', ''),
                        "metadata": metadata,
                        "collection": target_collection
                    }
                    sources.append(source_item)
                    print(f"🔍 DEBUG: Added source: {source_item['title']}")
                
                except Exception as e:
                    print(f"❌ DEBUG: Error parsing hit: {e}")
                    continue
            
            # Check if this is a "latest" or "most recent" query and sort by date
            if is_latest_query and target_collection == "fda_warning_letters":
                print(f"🔍 DEBUG: Detected latest query, sorting by date")
                # Sort sources by letter_date in descending order (most recent first)
                def parse_date(date_str):
                    try:
                        # Handle different date formats
                        if not date_str or date_str == 'Unknown Date':
                            return datetime(1900, 1, 1)  # Put unknown dates at the end
                        
                        # Try different date formats
                        date_formats = [
                            '%B %d, %Y',      # July 11, 2025, August 28, 2025
                            '%m/%d/%Y',       # 07/11/2025, 08/28/2025
                            '%Y-%m-%d',       # 2025-07-11, 2025-08-28
                        ]
                        
                        for fmt in date_formats:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                        
                        # If no format matches, return a very old date
                        return datetime(1900, 1, 1)
                    except:
                        return datetime(1900, 1, 1)
                
                # Sort by date (most recent first)
                def safe_sort_key(x):
                    metadata = x.get('metadata', {})
                    letter_date = metadata.get('letter_date', '') if metadata else ''
                    return parse_date(letter_date)
                
                sources.sort(key=safe_sort_key, reverse=True)
                print(f"🔍 DEBUG: Sorted sources by date, most recent first")
                
                # Show the sorted dates for debugging
                sorted_dates = [safe_sort_key(x) for x in sources]
                print(f"🔍 DEBUG: Sorted dates: {[d.strftime('%Y-%m-%d') for d in sorted_dates[:5]]}")
            
            # Return all sources up to the limit
            sources = sources[:top_k]
            print(f"🔍 DEBUG: Returning {len(sources)} sources to LLM")
            
        else:
            print(f"❌ DEBUG: No 'data' field found in response or empty data")
            print("🔄 DEBUG: Using fallback data due to empty response")
            return get_fallback_sources(query, target_collection, top_k)
        
        print(f"🔍 DEBUG: Final sources count: {len(sources)}")
        if sources:
            pretty_json_string = json.dumps(sources, indent=4)
            print('🔍 DEBUG: Final sources:', pretty_json_string)
    
        return sources
        
    except Exception as e:
        print(f"❌ DEBUG: Error in search_similar_documents: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 DEBUG: Using fallback data due to exception")
        return get_fallback_sources(query, collection_name, top_k)

def get_fallback_sources(query: str, collection_name: str, top_k: int) -> List[Dict[str, Any]]:
    """Provide fallback sources when Milvus search fails."""
    print(f"🔄 DEBUG: Providing fallback sources for collection: {collection_name}")
    
    if collection_name == "rss_feeds":
        # Fallback RSS feed data
        return [
            {
                "title": f"Regulatory News: {query.title()} Update",
                "content": f"Latest regulatory news and updates related to {query}. This includes industry developments, policy changes, and compliance updates from regulatory bodies. The system is currently using fallback data while vector search is being configured.",
                "metadata": {
                    "article_title": f"Regulatory News: {query.title()} Update",
                    "published_date": "2025-08-18",
                    "feed_name": "Regulatory Intelligence Feed",
                    "chunk_type": "news_article",
                    "companies": [],
                    "products": [],
                    "regulations": [],
                    "regulatory_bodies": []
                },
                "collection": "rss_feeds"
            },
            {
                "title": f"Industry Compliance Report: {query.title()}",
                "content": f"Comprehensive report on {query} compliance requirements, industry standards, and best practices for regulatory adherence. This is fallback data while the vector database is being configured.",
                "metadata": {
                    "article_title": f"Industry Compliance Report: {query.title()}",
                    "published_date": "2025-08-18",
                    "feed_name": "Industry Reports",
                    "chunk_type": "compliance_report",
                    "companies": [],
                    "products": [],
                    "regulations": [],
                    "regulatory_bodies": []
                },
                "collection": "rss_feeds"
            }
        ]
    elif collection_name == "fda_warning_letters":
        # Fallback FDA warning letter data
        return [
            {
                "title": f"FDA Warning Letter: {query.title()} Compliance Issues",
                "content": f"FDA warning letter addressing {query} compliance violations. The letter outlines specific regulatory concerns and required corrective actions. This is fallback data while the vector database is being configured.",
                "metadata": {
                    "company_name": f"Company {query.title()}",
                    "letter_date": "2025-08-18",
                    "chunk_type": "warning_letter",
                    "chunk_id": f"wl_{query.lower()}",
                    "violations": ["Quality System", "Documentation"],
                    "required_actions": ["Corrective Action Plan", "Documentation Review"],
                    "systemic_issues": ["Quality Management System"],
                    "regulatory_consequences": ["Warning Letter", "Follow-up Inspection"],
                    "product_types": ["Medical Device"],
                    "product_categories": ["Class II"]
                },
                "collection": "fda_warning_letters"
            }
        ]
    else:
        # Default fallback
        return [
            {
                "title": f"Regulatory Document from {collection_name}",
                "content": f"This is a sample regulatory document related to: {query}. The system is currently using fallback data while vector search is being configured.",
                "metadata": {"collection": collection_name, "source": "fallback_data"},
                "collection": collection_name
            }
        ]

async def load_collection_if_needed(collection_name: str) -> bool:
    """Load a collection into memory if it's not already loaded."""
    try:
        print(f"🔄 DEBUG: Checking if collection '{collection_name}' needs to be loaded...")
        
        # Test Zilliz Cloud V2 API endpoints for collection description
        describe_endpoints = [
            f"{MILVUS_URI}/v2/vectordb/collections/describe",
            f"{MILVUS_URI}/v2/vectordb/collections/list",
            f"{MILVUS_URI}/v2/vectordb/entities/get",
            f"{MILVUS_URI}/v1/vectordb/collections/describe"
        ]
        
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        describe_data = {
            "collectionName": collection_name
        }
        
        # For list collections, we don't need collection name
        list_data = {}
        
        print(f"🔄 DEBUG: Checking collection status for: {collection_name}")
        
        # Try different endpoints until one works
        describe_successful = False
        for endpoint in describe_endpoints:
            try:
                print(f"🔄 DEBUG: Trying describe endpoint: {endpoint}")
                
                # Use appropriate data based on endpoint type
                if "list" in endpoint:
                    payload = list_data
                else:
                    payload = describe_data
                
                response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
                print(f"🔄 DEBUG: Endpoint {endpoint} response status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ DEBUG: Collection describe successful via {endpoint}")
                    describe_successful = True
                    break
                else:
                    print(f"❌ DEBUG: Endpoint {endpoint} failed: {response.status_code}")
                    print(f"❌ DEBUG: Response text: {response.text[:200]}")
                    
            except Exception as endpoint_error:
                print(f"❌ DEBUG: Endpoint {endpoint} error: {str(endpoint_error)}")
                continue
        
        if not describe_successful:
            print(f"❌ DEBUG: All describe endpoints failed")
            return False
        
        # Get the successful response from the loop
        collection_info = response.json()
        print(f"🔄 DEBUG: Collection info response: {json.dumps(collection_info, indent=2)}")
        
        load_state = collection_info.get('data', {}).get('load', 'Unknown')
        print(f"🔄 DEBUG: Collection '{collection_name}' load state: {load_state}")
        
        if load_state == "LoadStateNotLoad":
            print(f"🔄 DEBUG: Loading collection '{collection_name}'...")
            
            # Test Zilliz Cloud V2 load endpoints
            load_endpoints = [
                f"{MILVUS_URI}/v2/vectordb/collections/load",
                f"{MILVUS_URI}/v1/vectordb/collections/load",
                f"{MILVUS_URI}/v2/vectordb/collections/{collection_name}/load",
                f"{MILVUS_URI}/v1/vectordb/collections/{collection_name}/load"
            ]
            
            load_data = {
                "collectionName": collection_name
            }
            
            # Try different load endpoints
            load_successful = False
            for load_endpoint in load_endpoints:
                try:
                    print(f"🔄 DEBUG: Trying load endpoint: {load_endpoint}")
                    load_response = requests.post(load_endpoint, json=load_data, headers=headers, timeout=10)
                    print(f"🔄 DEBUG: Endpoint {load_endpoint} response status: {load_response.status_code}")
                    
                    if load_response.status_code == 200:
                        load_result = load_response.json()
                        if load_result.get('code') == 0:
                            print(f"✅ DEBUG: Collection '{collection_name}' loaded successfully via {load_endpoint}")
                            load_successful = True
                            break
                        else:
                            print(f"❌ DEBUG: Load failed with code: {load_result.get('code')}")
                    else:
                        print(f"❌ DEBUG: Endpoint {load_endpoint} failed: {load_response.status_code}")
                        print(f"❌ DEBUG: Response text: {load_response.text[:200]}")
                        
                except Exception as load_error:
                    print(f"❌ DEBUG: Endpoint {load_endpoint} error: {str(load_error)}")
                    continue
            
            if not load_successful:
                print(f"❌ DEBUG: All load endpoints failed")
                return False
                
            return True
        else:
            print(f"✅ DEBUG: Collection '{collection_name}' is already loaded")
            return True
            
    except Exception as e:
        print(f"❌ DEBUG: Error loading collection: {e}")
        return False

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources."""
    if not client:
        return "OpenAI client not available. Please check your API key configuration."
    
    try:
        # Build context from sources if available
        context = ""
        collection_type = "general"
        
        if sources:
            # Determine collection type from first source
            first_source = sources[0]
            collection_type = first_source.get('collection', 'general')
            
            # Build context with collection-specific information (top 3 sources for better coverage)
            context = f"\n\nRelevant sources from {collection_type.replace('_', ' ').title()}:\n"
            for i, source in enumerate(sources[:3], 1):  # Take top 3 sources instead of just 1
                metadata = source.get('metadata', {})
                # Determine title per collection
                if collection_type == "fda_warning_letters":
                    title = metadata.get('company_name', 'Unknown Company')
                else:
                    title = metadata.get('article_title', 'Unknown Title')
                # Use the actual content field and provide a longer excerpt
                content = source.get('content', '') or source.get('text', '')[:1200]
                print(f"🔍 DEBUG: Chat function - content length: {len(content)}, preview: {content[:200]}...")
                
                # Add collection-specific details
                if collection_type == "fda_warning_letters":
                    company = metadata.get('company_name', 'Unknown Company')
                    date = metadata.get('letter_date', 'Unknown Date')
                    context += f"{i}. {title} - Company: {company}, Date: {date}\n"
                    
                    # For FDA warning letters, use metadata fields instead of raw content
                    systemic_issues = metadata.get('systemic_issues', '[]')
                    regulatory_consequences = metadata.get('regulatory_consequences', '[]')
                    violations = metadata.get('violations', '[]')
                    required_actions = metadata.get('required_actions', '[]')
                    
                    # Clean up JSON strings and add to context
                    if systemic_issues != '[]':
                        context += f"   Systemic Issues: {systemic_issues}\n"
                    if regulatory_consequences != '[]':
                        context += f"   Regulatory Consequences: {regulatory_consequences}\n"
                    if violations != '[]':
                        context += f"   Violations: {violations}\n"
                    if required_actions != '[]':
                        context += f"   Required Actions: {required_actions}\n"
                    
                    # Add a brief excerpt from the actual content for context
                    if content and len(content) > 200:
                        # Find the actual warning letter content (skip HTML boilerplate)
                        warning_start = content.find('WARNING LETTER')
                        if warning_start > 0:
                            warning_content = content[warning_start:warning_start + 300]
                            context += f"   Content: {warning_content}...\n"
                        else:
                            context += f"   Content: {content[:300]}...\n"
                    context += "\n"
                else:
                    feed = metadata.get('feed_name', 'Unknown Feed')
                    date = metadata.get('published_date', 'Unknown Date')
                    context += f"{i}. {title} - Feed: {feed}, Date: {date}\n"
                    context += f"   {content}...\n\n"
        
        print(f"🔍 DEBUG: Final context being sent to AI: {context[:500]}...")
        
        # Build conversation messages
        messages = []
        for msg in conversation_history[-5:]:  # Keep last 5 messages for context
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message with context
        current_message = message
        if context:
            current_message = f"{message}\n\n{context}"
        
        messages.append({"role": "user", "content": current_message})
        
        # Add system message with strict grounding and brevity
        if collection_type == "rss_feeds":
            system_content = (
                "You are RegIntel, an AI assistant for regulatory intelligence. "
                "Answer strictly and only from the provided source excerpt. "
                "Do not introduce unrelated companies, news, or external knowledge. "
                "If the excerpt lacks the answer, say you don't have enough information. "
                "Keep the answer concise (3-5 sentences) and directly address the user query."
            )
        elif collection_type == "fda_warning_letters":
            system_content = (
                "You are RegIntel, an AI assistant for regulatory intelligence. "
                "Answer strictly and only from the provided source excerpt. "
                "For FDA warning letters, focus on: company name, violations, required actions, and regulatory consequences. "
                "Extract specific details from the text content provided. "
                "If the excerpt contains the answer, provide it clearly. "
                "Do not say 'the excerpt does not provide information' if the information is actually there. "
                "Keep the answer concise (3-5 sentences) and directly address the user query."
            )
        else:
            system_content = (
                "You are RegIntel, an AI assistant for regulatory intelligence. "
                "Answer strictly and only from the provided source excerpt. "
                "Do not add external knowledge. "
                "If the excerpt lacks the answer, say you don't have enough information. "
                "Keep the answer concise (3-5 sentences)."
            )
        
        system_message = {
            "role": "system", 
            "content": system_content
        }
        messages.insert(0, system_message)
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=400,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error chatting with GPT: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main chat interface."""
    try:
        # Try to get current user from auth middleware
        from auth.middleware import get_optional_user
        current_user = await get_optional_user(request)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": current_user
        })
    except Exception as e:
        print(f"Error getting user for index page: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": None
        })

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RegIntel API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG integration."""
    try:
        print(f"🔍 DEBUG: Main chat endpoint called")
        print(f"🔍 DEBUG: Request message: {request.message}")
        print(f"🔍 DEBUG: Conversation history length: {len(request.conversation_history)}")
        
        # Search for relevant documents
        sources = await search_similar_documents(request.message, DEFAULT_COLLECTION)
        print(f"🔍 DEBUG: Found {len(sources)} sources")
        
        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        print(f"🔍 DEBUG: Converted {len(history)} history messages")
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        print(f"🔍 DEBUG: Generated response: {response[:100]}...")
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ Internal error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/chat/{collection}", response_model=ChatResponse)
async def chat_with_collection(collection: str, request: ChatRequest):
    """Chat endpoint with RAG integration for a specific collection."""
    try:
        print(f"🔍 DEBUG: Chat endpoint called for collection: {collection}")
        print(f"🔍 DEBUG: Request message: {request.message}")
        print(f"🔍 DEBUG: Conversation history length: {len(request.conversation_history)}")
        
        # Add timeout protection for the entire chat process
        import asyncio
        
        # Create tasks for search and chat
        search_task = asyncio.create_task(search_similar_documents(request.message, collection))
        search_sources = await asyncio.wait_for(search_task, timeout=25.0)
        
        print(f"🔍 DEBUG: Found {len(search_sources)} sources")
        
        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        print(f"🔍 DEBUG: Converted {len(history)} history messages")
        
        # Get AI response with timeout
        chat_task = asyncio.create_task(chat_with_gpt(request.message, history, search_sources))
        response = await asyncio.wait_for(chat_task, timeout=20.0)
        
        print(f"🔍 DEBUG: Generated response: {response[:100]}...")
        
        return ChatResponse(
            response=response,
            sources=search_sources
        )
        
    except asyncio.TimeoutError:
        print("⏰ DEBUG: Chat endpoint timed out")
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        print(f"❌ Internal error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Vercel monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/collections")
async def get_collections():
    """Get available collections"""
    return {
        "collections": [
            {"id": "rss_feeds", "name": "Regulatory News", "description": "RSS feeds from regulatory sources"},
            {"id": "fda_warning_letters", "name": "FDA Warning Letters", "description": "FDA compliance documents"}
        ]
    }

@app.get("/api/rss-feeds/latest")
async def get_latest_rss_feeds(limit: int = 10):
    """Get the latest RSS feeds from Supabase rss_feeds_gold table."""
    try:
        print(f"📰 DEBUG: Fetching latest {limit} RSS feeds from Supabase")
        
        # Import Supabase client here to avoid circular imports
        from auth.config import get_supabase_config
        
        # Get Supabase client
        supabase_config = get_supabase_config()
        if not supabase_config or not supabase_config.is_client_available():
            error_msg = "Supabase client not available - check configuration and environment variables"
            print(f"❌ DEBUG: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "articles": []
            }
        
        supabase = supabase_config.get_client()
        
        # Query the rss_feeds_gold table for the most recent entries
        print(f"📰 DEBUG: About to query Supabase with limit={limit}")
        response = supabase.table('rss_feeds_gold').select(
            'article_feed_name,article_published_date,article_title,content_category'
        ).order('article_published_date', desc=True).limit(limit).execute()
        
        print(f"📰 DEBUG: Supabase response received")
        if hasattr(response, 'data'):
            articles = response.data
            print(f"📰 DEBUG: Response has 'data' attribute, length: {len(articles)}")
        else:
            articles = response.get('data', [])
            print(f"📰 DEBUG: Response uses .get('data'), length: {len(articles)}")
        
        print(f"📰 DEBUG: Found {len(articles)} RSS articles from Supabase")
        
        # Filter out articles with missing required data
        valid_articles = []
        for article in articles:
            if (article.get('article_feed_name') and 
                article.get('article_published_date') and
                article.get('article_title')):
                valid_articles.append(article)
        
        print(f"📰 DEBUG: {len(valid_articles)} articles have complete data")
        
        return {
            "success": True,
            "articles": valid_articles,
            "count": len(valid_articles),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error fetching RSS feeds: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "articles": []
        }

@app.get("/api/warning-letters/weekly-stats", response_model=dict)
async def get_weekly_warning_letter_stats():
    """Get weekly warning letter statistics for the last 4 weeks"""
    try:
        print(f"📊 DEBUG: Fetching weekly warning letter statistics...")
        
        # Get Supabase client
        from auth.config import get_supabase_config
        supabase = get_supabase_config().get_client()
        
        # Query for weekly statistics
        response = supabase.table('warning_letters_analytics').select('letter_date').execute()
        
        if not response.data:
            print(f"❌ DEBUG: No data found in warning_letters_analytics table")
            return {"success": False, "message": "No data found", "weekly_stats": []}
        
        print(f"📊 DEBUG: Found {len(response.data)} warning letters")
        
        # Process the data to get weekly counts
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        weekly_counts = defaultdict(int)
        
        for letter in response.data:
            letter_date = letter.get('letter_date')
            if letter_date:
                try:
                    # Parse the date and get the week number
                    date_obj = datetime.strptime(letter_date, '%Y-%m-%d')
                    week_num = date_obj.isocalendar()[1]  # ISO week number
                    year = date_obj.year
                    week_key = f"{year}-W{week_num:02d}"
                    weekly_counts[week_key] += 1
                except Exception as e:
                    print(f"⚠️ DEBUG: Error parsing date {letter_date}: {e}")
                    continue
        
        # Get the last 4 weeks
        current_date = datetime.now()
        last_4_weeks = []
        
        for i in range(4):
            week_date = current_date - timedelta(weeks=i)
            week_num = week_date.isocalendar()[1]
            year = week_date.year
            week_key = f"{year}-W{week_num:02d}"
            
            count = weekly_counts.get(week_key, 0)
            print(f"📊 DEBUG: Week {week_key} has count {count}")
            # Only include weeks with data (count > 0)
            if count > 0:
                print(f"📊 DEBUG: Including week {week_key} with count {count}")
                last_4_weeks.append({
                    "week": week_key,
                    "week_display": f"Week {week_num}, {year}",
                    "count": count
                })
            else:
                print(f"📊 DEBUG: Excluding week {week_key} with count {count}")
        
        # Sort by week chronologically (oldest first: Week 33, 34, 35)
        last_4_weeks.sort(key=lambda x: x["week"])
        
        print(f"📊 DEBUG: Weekly stats: {last_4_weeks}")
        
        return {
            "success": True,
            "weekly_stats": last_4_weeks,
            "total_weeks": len(last_4_weeks)
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error fetching weekly stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error fetching weekly stats: {str(e)}", "weekly_stats": []}

@app.get("/api/warning-letters/latest")
async def get_latest_warning_letters(limit: int = 10):
    """Get the most recent warning letters from Supabase warning_letter_analytics table."""
    try:
        print(f"🔍 DEBUG: Fetching latest {limit} warning letters from Supabase")
        
        # Import Supabase client here to avoid circular imports
        from auth.config import get_supabase_config
        
        # Get Supabase client
        supabase_config = get_supabase_config()
        if not supabase_config or not supabase_config.is_client_available():
            error_msg = "Supabase client not available - check configuration and environment variables"
            print(f"❌ DEBUG: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "warning_letters": [],
                "source_table": "error"
            }
        
        supabase = supabase_config.get_client()
        
        # Query the warning_letter_analytics table for the most recent entries
        print(f"🔍 DEBUG: About to query Supabase for unique warning letters")
        response = supabase.table('warning_letters_analytics').select(
            'letter_date,company_name,summary'
        ).order('letter_date', desc=True).execute()  # Get all rows, we'll deduplicate and limit in Python
        
        print(f"🔍 DEBUG: Supabase response received")
        if hasattr(response, 'data'):
            warning_letters = response.data
            print(f"🔍 DEBUG: Response has 'data' attribute, length: {len(warning_letters)}")
        else:
            warning_letters = response.get('data', [])
            print(f"🔍 DEBUG: Response uses .get('data'), length: {len(warning_letters)}")
        
        print(f"🔍 DEBUG: Found {len(warning_letters)} total warning letters from Supabase")
        
        # Deduplicate the warning letters based on company_name and letter_date combination
        # This ensures we get truly unique warning letters across all data
        seen_combinations = set()
        unique_letters = []
        
        for letter in warning_letters:
            company_name = letter.get('company_name', 'Unknown Company')
            letter_date = letter.get('letter_date', 'Unknown Date')
            
            # Create a unique key for deduplication
            unique_key = f"{company_name}_{letter_date}"
            
            if unique_key not in seen_combinations:
                seen_combinations.add(unique_key)
                unique_letters.append(letter)
        
        print(f"🔍 DEBUG: After deduplication: {len(unique_letters)} unique warning letters")
        
        # Sort by date (most recent first) and limit to the requested number
        unique_letters.sort(key=lambda x: x.get('letter_date', ''), reverse=True)
        limited_letters = unique_letters[:limit]
        
        print(f"🔍 DEBUG: After limiting: {len(limited_letters)} warning letters")
        
        # Transform the deduplicated and limited data to match the expected format
        transformed_letters = []
        for letter in limited_letters:
            company_name = letter.get('company_name', 'Unknown Company')
            letter_date = letter.get('letter_date', 'Unknown Date')
            summary = letter.get('summary', 'No Subject')
            
            transformed_letter = {
                "company_name": company_name,
                "letter_date": letter_date,
                "subject": summary
            }
            transformed_letters.append(transformed_letter)
        
        print(f"🔍 DEBUG: Transformed {len(transformed_letters)} warning letters")
        print(f"🔍 DEBUG: Sample transformed letter: {transformed_letters[0] if transformed_letters else 'None'}")
        
        # Return the exact format your frontend expects
        response_data = {
            "success": True,
            "count": len(transformed_letters),
            "warning_letters": transformed_letters,
            "source_table": "warning_letter_analytics"
        }
        
        print(f"🔍 DEBUG: Final API response: {response_data}")
        return response_data
        
    except Exception as e:
        print(f"❌ DEBUG: Error fetching warning letters from Supabase: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "warning_letters": [],
            "source_table": "error"
        }

@app.get("/api/debug/warning-letters")
async def debug_warning_letters():
    """Debug endpoint to test warning letters API directly"""
    try:
        print(f"🔍 DEBUG: Testing warning letters API directly")
        
        # Import Supabase client here to avoid circular imports
        from auth.config import get_supabase_config
        
        # Get Supabase client
        supabase_config = get_supabase_config()
        if not supabase_config or not supabase_config.is_client_available():
            error_msg = "Supabase client not available - check configuration and environment variables"
            print(f"❌ DEBUG: {error_msg}")
            return {
                "success": False,
                "supabase_connected": False,
                "table_exists": False,
                "error": error_msg,
                "response_type": "N/A",
                "data_type": "N/A",
                "data_length": 0
            }
        
        supabase = supabase_config.get_client()
        
        # Test the connection and table
        print(f"🔍 DEBUG: Testing Supabase connection...")
        
        # Try to get table info
        try:
            response = supabase.table('warning_letter_analytics').select('*').limit(1).execute()
            
            if hasattr(response, 'data'):
                data = response.data
            else:
                data = response.get('data', [])
            
            print(f"🔍 DEBUG: Supabase response: {response}")
            print(f"🔍 DEBUG: Data extracted: {data}")
            
            return {
                "success": True,
                "supabase_connected": True,
                "table_exists": True,
                "sample_data": data,
                "response_type": str(type(response)),
                "data_type": str(type(data)),
                "data_length": len(data) if data else 0
            }
            
        except Exception as table_error:
            print(f"❌ DEBUG: Table query failed: {table_error}")
            return {
                "success": False,
                "supabase_connected": True,
                "table_exists": False,
                "error": str(table_error),
                "response_type": "N/A",
                "data_type": "N/A",
                "data_length": 0
            }
            
    except Exception as e:
        print(f"❌ DEBUG: Error in debug endpoint: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "supabase_connected": False,
            "error": str(e),
            "response_type": "N/A",
            "data_type": "N/A",
            "data_length": 0
        }

@app.get("/api/debug/vector-tables")
async def debug_vector_tables():
    """Debug endpoint to check if vector tables exist and are accessible."""
    try:
        print(f"🔍 DEBUG: Checking vector tables accessibility...")
        
        # Import Supabase client here to avoid circular imports
        from auth.config import get_supabase_config
        
        # Get Supabase client
        supabase_config = get_supabase_config()
        if not supabase_config or not supabase_config.is_client_available():
            return {
                "success": False,
                "error": "Supabase client not available",
                "tables": {}
            }
        
        supabase = supabase_config.get_client()
        
        # Test each table
        tables_to_check = {
            "rss_feed_vectors": "RSS feeds vector table (singular)",
            "rss_feeds": "RSS feeds vector table (plural - fallback)",
            "rss_feeds_gold": "RSS feeds summary table", 
            "warning_letters_vectors": "FDA warning letters vector table",
            "warning_letters_analytics": "FDA warning letters summary table"
        }
        
        results = {}
        
        for table_name, description in tables_to_check.items():
            try:
                print(f"🔍 DEBUG: Testing table: {table_name}")
                
                # Try to query the table with a small limit
                response = supabase.table(table_name).select('*').limit(1).execute()
                
                if hasattr(response, 'data'):
                    data = response.data
                else:
                    data = response.get('data', [])
                
                # Get table info
                table_info = {
                    "exists": True,
                    "accessible": True,
                    "row_count": len(data),
                    "sample_data": data[0] if data else None,
                    "description": description
                }
                
                # Try to get column info by examining sample data
                if data:
                    table_info["columns"] = list(data[0].keys()) if data[0] else []
                else:
                    # Try a different approach to get column info
                    try:
                        # Query with specific fields to see what's available
                        test_response = supabase.table(table_name).select('*').limit(0).execute()
                        table_info["columns"] = "Unknown - no data in table"
                    except:
                        table_info["columns"] = "Unknown - cannot determine"
                
                results[table_name] = table_info
                print(f"✅ DEBUG: Table {table_name} is accessible")
                
            except Exception as table_error:
                print(f"❌ DEBUG: Table {table_name} error: {table_error}")
                results[table_name] = {
                    "exists": False,
                    "accessible": False,
                    "error": str(table_error),
                    "description": description,
                    "columns": []
                }
        
        # Check for pgvector extension
        pgvector_available = False
        try:
            # Try to query a vector column if any table has one
            for table_name in ["rss_feed_vectors", "rss_feeds", "warning_letters_vectors"]:
                if results.get(table_name, {}).get("accessible", False):
                    try:
                        # Try to query a vector column
                        response = supabase.table(table_name).select('text_vector').limit(1).execute()
                        pgvector_available = True
                        break
                    except:
                        continue
        except:
            pass
        
        return {
            "success": True,
            "tables": results,
            "pgvector_available": pgvector_available,
            "recommendations": {
                "vector_tables_missing": not (results.get("rss_feed_vectors", {}).get("accessible", False) and 
                                            results.get("warning_letters_vectors", {}).get("accessible", False)),
                "summary_tables_available": (results.get("rss_feeds_gold", {}).get("accessible", False) and 
                                           results.get("warning_letters_analytics", {}).get("accessible", False))
            }
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error checking vector tables: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "tables": {}
        }

# Auth endpoints are handled by auth/routes.py router

@app.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("auth/login.html", {"request": request})

@app.get("/auth/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serve the registration page"""
    return templates.TemplateResponse("auth/register.html", {"request": request})

@app.get("/api/debug/status")
async def debug_status():
    """Debug endpoint to check environment variables and connection status."""
    try:
        # Check environment variables
        env_status = {
            "milvus_uri": "✅ Set" if MILVUS_URI else "❌ Not set",
            "milvus_token": "✅ Set" if MILVUS_TOKEN else "❌ Not set",
            "openai_api_key": "✅ Set" if OPENAI_API_KEY else "❌ Not set",
            "default_collection": DEFAULT_COLLECTION,
            "fda_collection": FDA_WARNING_LETTERS_COLLECTION,
            "rss_collection": RSS_FEEDS_COLLECTION
        }
        
        # Check Milvus connection if credentials are available
        milvus_status = "Not configured"
        milvus_details = {}
        if MILVUS_URI and MILVUS_TOKEN:
            try:
                                # Test Zilliz Cloud V2 API endpoints - correct structure
                endpoints_to_test = [
                    # V2 entities endpoints (these should work)
                    "/v2/vectordb/entities/get",
                    "/v2/vectordb/entities/search",
                    "/v2/vectordb/entities/insert",
                    # V2 collections endpoints
                    "/v2/vectordb/collections/list",
                    "/v2/vectordb/collections/describe",
                    # V1 fallbacks
                    "/v1/vectordb/entities/get",
                    "/v1/vectordb/entities/search",
                    # Health and info
                    "/health",
                    "/"
                ]
                
                for endpoint in endpoints_to_test:
                    try:
                        response = requests.get(f"{MILVUS_URI}{endpoint}", 
                                             headers={"Authorization": f"Bearer {MILVUS_TOKEN}"})
                        milvus_details[f"endpoint_{endpoint}"] = {
                            "status": response.status_code,
                            "response": response.text[:200] if response.text else "No response body"
                        }
                        if response.status_code == 200:
                            milvus_status = f"✅ Connected via {endpoint} (Status: {response.status_code})"
                            break
                    except Exception as endpoint_error:
                        milvus_details[f"endpoint_{endpoint}"] = {
                            "status": "error",
                            "error": str(endpoint_error)
                        }
                
                if milvus_status == "Not configured":
                    milvus_status = f"❌ All endpoints failed - check API structure"
                    
            except Exception as e:
                milvus_status = f"❌ Connection error: {str(e)}"
        
        # Check OpenAI connection
        openai_status = "Not configured"
        if OPENAI_API_KEY:
            try:
                # Simple test - try to get an embedding
                test_embedding = await get_embedding("test")
                if test_embedding:
                    openai_status = f"✅ Connected (Embedding length: {len(test_embedding)})"
                else:
                    openai_status = "❌ Embedding generation failed"
            except Exception as e:
                openai_status = f"❌ OpenAI error: {str(e)}"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "environment_variables": env_status,
            "milvus_connection": milvus_status,
            "milvus_endpoint_tests": milvus_details,
            "openai_connection": openai_status,
            "debug_info": {
                "milvus_uri_length": len(MILVUS_URI) if MILVUS_URI else 0,
                "milvus_token_length": len(MILVUS_TOKEN) if MILVUS_TOKEN else 0,
                "openai_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 