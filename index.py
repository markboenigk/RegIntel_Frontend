import os
import mmh3
import json
import requests
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import openai
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import time


# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Security check - fail if critical credentials are missing
if not OPENAI_API_KEY:
    raise ValueError("❌ CRITICAL: OPENAI_API_KEY environment variable is required")
if not MILVUS_URI:
    raise ValueError("❌ CRITICAL: MILVUS_URI environment variable is required")
if not MILVUS_TOKEN:
    raise ValueError("❌ CRITICAL: MILVUS_TOKEN environment variable is required")

# Debug: Print environment variable status
print(f"🔧 DEBUG: Environment variables loaded:")
print(f"   OPENAI_API_KEY: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
print(f"   MILVUS_URI: {'✅ Set' if MILVUS_URI else '❌ Not set'}")
print(f"   MILVUS_TOKEN: {'✅ Set' if MILVUS_TOKEN else '❌ Not set'}")
if OPENAI_API_KEY:
    print(f"   OpenAI Key length: {len(OPENAI_API_KEY)} characters")
    print(f"   OpenAI Key starts with: {OPENAI_API_KEY[:20]}...")
if MILVUS_TOKEN:
    print(f"   Milvus Token length: {len(MILVUS_TOKEN)} characters")
    print(f"   Milvus Token starts with: {MILVUS_TOKEN[:20]}...")
    print(f"   Milvus Token ends with: ...{MILVUS_TOKEN[-20:]}")

# Collection names - configurable via environment variables
FDA_WARNING_LETTERS_COLLECTION = os.getenv("FDA_WARNING_LETTERS_COLLECTION", "fda_warning_letters")
RSS_FEEDS_COLLECTION = os.getenv("RSS_FEEDS_COLLECTION", "rss_feeds")

# Default collection for backward compatibility (you can remove this if not needed)
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", RSS_FEEDS_COLLECTION)

# RAG Configuration
STRICT_RAG_ONLY = os.getenv("STRICT_RAG_ONLY", "true").lower() == "true"  # Only use provided sources, no external knowledge
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"  # Disabled by default for simple version
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "o3")  # GPT model for prompt-based reranking
INITIAL_SEARCH_MULTIPLIER = int(os.getenv("INITIAL_SEARCH_MULTIPLIER", "3"))  # How many more results to fetch initially

# Input Validation Configuration
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))  # Maximum characters per message
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # Maximum conversation history length
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # Max requests per time window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Time window in seconds for rate limiting

# Debug: Print validation configuration
print(f"🔒 Input Validation Configuration:")
print(f"   MAX_MESSAGE_LENGTH: {MAX_MESSAGE_LENGTH}")
print(f"   MAX_CONVERSATION_HISTORY: {MAX_CONVERSATION_HISTORY}")
print(f"   RATE_LIMIT_REQUESTS: {RATE_LIMIT_REQUESTS}")
print(f"   RATE_LIMIT_WINDOW: {RATE_LIMIT_WINDOW} seconds")

# Rate limiting storage (in production, use Redis or similar)
rate_limit_storage = {}

# Input validation patterns
BLOCKED_PATTERNS = [
    r"system:", r"assistant:", r"user:",  # Role injection attempts
    r"ignore previous instructions", r"forget everything",  # Instruction injection
    r"you are now", r"act as", r"pretend to be",  # Role manipulation
    r"<script>", r"javascript:", r"onload=",  # XSS attempts
    r"SELECT.*FROM", r"INSERT.*INTO", r"DROP.*TABLE",  # SQL injection attempts
    r"rm -rf", r"del /s", r"format c:",  # System command attempts
]

# Medical/regulatory specific blocked patterns
MEDICAL_BLOCKED_PATTERNS = [
    r"prescribe", r"diagnose", r"treat", r"cure",  # Medical advice
    r"take this medication", r"stop taking", r"dosage",  # Medication advice
    r"you should", r"you need to", r"you must",  # Direct medical instructions
]

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Conversation history")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed.')
        return v.strip()
    
    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        if len(v) > MAX_CONVERSATION_HISTORY:
            raise ValueError(f'Conversation history too long. Maximum {MAX_CONVERSATION_HISTORY} messages allowed.')
        return v

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources")
    # reranking_info: Dict[str, Any] = Field(default={}, description="Reranking information") - COMMENTED OUT

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to add")
    metadata: str = Field(default="", description="Optional metadata for the document")

# RerankingConfig - COMMENTED OUT FOR SIMPLE VERSION
# class RerankingConfig(BaseModel):
#     enabled: bool = Field(..., description="Whether reranking is enabled")
#     model: str = Field(..., description="Reranking model to use")
#     initial_search_multiplier: int = Field(..., description="Multiplier for initial search results")


# Create FastAPI app
app = FastAPI(
    title="ChatGPT RAG API",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Utility functions
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

# Input Validation Functions
def validate_message_content(message: str) -> Dict[str, Any]:
    """
    Validate message content for malicious patterns and inappropriate content.
    Returns validation result with success status and any errors.
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check message length
    if len(message) > MAX_MESSAGE_LENGTH:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed.")
    
    # Check for blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            validation_result["valid"] = False
            validation_result["errors"].append(f"Message contains blocked pattern: {pattern}")
    
    # Check for medical advice patterns
    for pattern in MEDICAL_BLOCKED_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            validation_result["warnings"].append(f"Message may contain medical advice: {pattern}")
    
    # Check for excessive special characters (potential encoding attacks)
    special_char_ratio = len(re.findall(r'[^\w\s]', message)) / len(message) if message else 0
    if special_char_ratio > 0.3:  # More than 30% special characters
        validation_result["warnings"].append("Message contains unusually high number of special characters")
    
    # Check for repeated characters (potential spam)
    if len(message) > 10:
        for char in set(message):
            if message.count(char) > len(message) * 0.7:  # More than 70% of message is one character
                validation_result["warnings"].append("Message contains excessive repeated characters")
                break
    
    return validation_result

def validate_conversation_history(history: List[ChatMessage]) -> Dict[str, Any]:
    """
    Validate conversation history for length and content.
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check history length
    if len(history) > MAX_CONVERSATION_HISTORY:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Conversation history too long. Maximum {MAX_CONVERSATION_HISTORY} messages allowed.")
    
    # Validate each message in history
    for i, msg in enumerate(history):
        if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid message format at position {i}")
            continue
        
        # Check role validity
        if msg.role not in ['user', 'assistant', 'system']:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid role '{msg.role}' at position {i}")
        
        # Check content validity
        content_validation = validate_message_content(msg.content)
        if not content_validation["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend([f"History message {i}: {error}" for error in content_validation["errors"]])
        
        validation_result["warnings"].extend([f"History message {i}: {warning}" for warning in content_validation["warnings"]])
    
    return validation_result

def check_rate_limit(client_ip: str) -> Dict[str, Any]:
    """
    Check if client has exceeded rate limits.
    Returns rate limit status with remaining requests and reset time.
    """
    global rate_limit_storage
    current_time = time.time()
    
    print(f"🔒 DEBUG: Rate limit check for {client_ip}")
    print(f"🔒 DEBUG: Current storage: {rate_limit_storage}")
    
    # Clean up old entries
    rate_limit_storage = {ip: data for ip, data in rate_limit_storage.items() 
                         if current_time - data['timestamp'] < RATE_LIMIT_WINDOW}
    
    print(f"🔒 DEBUG: After cleanup: {rate_limit_storage}")
    
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = {
            'requests': 1,
            'timestamp': current_time
        }
        print(f"🔒 Rate limit: New client {client_ip}, requests: 1")
        print(f"🔒 DEBUG: Storage after new client: {rate_limit_storage}")
        return {
            "allowed": True,
            "remaining": RATE_LIMIT_REQUESTS - 1,
            "reset_time": current_time + RATE_LIMIT_WINDOW
        }
    
    client_data = rate_limit_storage[client_ip]
    print(f"🔒 DEBUG: Client data: {client_data}")
    
    # Check if window has expired
    if current_time - client_data['timestamp'] >= RATE_LIMIT_WINDOW:
        client_data['requests'] = 1
        client_data['timestamp'] = current_time
        print(f"🔒 Rate limit: Window expired for {client_ip}, reset to 1 request")
        return {
            "allowed": True,
            "remaining": RATE_LIMIT_REQUESTS - 1,
            "reset_time": current_time + RATE_LIMIT_WINDOW
        }
    
    # Check if limit exceeded
    if client_data['requests'] >= RATE_LIMIT_REQUESTS:
        print(f"🔒 Rate limit: EXCEEDED for {client_ip}, requests: {client_data['requests']}")
        return {
            "allowed": False,
            "remaining": 0,
            "reset_time": client_data['timestamp'] + RATE_LIMIT_WINDOW
        }
    
    # Increment request count
    client_data['requests'] += 1
    print(f"🔒 Rate limit: {client_ip} at {client_data['requests']}/{RATE_LIMIT_REQUESTS} requests")
    print(f"🔒 DEBUG: Final storage: {rate_limit_storage}")
    
    return {
        "allowed": True,
        "remaining": RATE_LIMIT_REQUESTS - client_data['requests'],
        "reset_time": client_data['timestamp'] + RATE_LIMIT_WINDOW
    }

# Reranking function - COMMENTED OUT FOR SIMPLE VERSION
# async def rerank_results(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
#     """Rerank search results using OpenAI's GPT model with prompt-based evaluation."""
#     if not client or not documents or not ENABLE_RERANKING:
#         return documents[:top_k]
#     
#     try:
#         print(f"Reranking {len(documents)} documents using prompt-based evaluation")
#         
#         # Prepare documents for reranking
#         doc_texts = []
#         for doc in documents:
#             text = doc.get('text', '')
#             metadata = doc.get('metadata', {})
#             # Create a formatted document string for reranking
#             if isinstance(metadata, str):
#                 try:
#                     metadata = json.loads(metadata)
#                 except:
#                     metadata = {}
#             
#             # Format document with metadata context for better semantic matching
#             channel_name = metadata.get('channel_name', 'Unknown')
#             video_title = metadata.get('video_title', 'Unknown Video')
#             doc_str = f"Channel: {channel_name}\nVideo: {video_title}\nContent: {text}"
#             doc_texts.append(doc_str)
#         
#         # Create evaluation prompt
#         evaluation_prompt = f"""
# You are an expert at evaluating the relevance of documents to a user query. 

# User Query: "{query}"

# Please evaluate each document below and assign a relevance score from 0.0 to 1.0, where:
# - 0.0 = Completely irrelevant
# - 0.5 = Somewhat relevant
# - 1.0 = Highly relevant

# Consider factors like:
# - Semantic similarity to the query
# - Whether the document directly addresses the query
# - Contextual relevance
# - Information completeness

# Documents to evaluate:

# """
#         
#         # Add each document to the prompt with a number
#         for i, doc_text in enumerate(doc_texts, 1):
#             evaluation_prompt += f"\nDocument {i}:\n{doc_text}\n"
#         
#         evaluation_prompt += f"""

# Please respond with ONLY a JSON array of scores, one for each document, in order.
# Example format: [0.8, 0.3, 0.9, 0.1, 0.7]

# Scores:"""
#         
#         # Get relevance scores from GPT
#         try:
#             response = await client.chat.completions.create(
#                 model=RERANKING_MODEL,
#                 messages=[
#                     {"role": "system", "content": "You are a precise evaluator. Respond only with the JSON array of scores."},
#                     {"role": "user", "content": evaluation_prompt}
#                 ],
#                 max_tokens=200,
#                 temperature=0.1  # Low temperature for consistent scoring
#             )
#             
#             # Parse the response to get scores
#             response_text = response.choices[0].message.content.strip()
#             
#             # Extract JSON array from response
#             import re
#             json_match = re.search(r'\[[0-9.,\s]+\]', response_text)
#             if json_match:
#                 scores_text = json_match.group()
#                 scores = [float(score.strip()) for score in scores_text.strip('[]').split(',')]
#             else:
#                 # Fallback: try to parse the entire response as JSON
#                 scores = json.loads(response_text)
#             
#             # Ensure we have the right number of scores
#             if len(scores) != len(documents):
#                 print(f"Warning: Expected {len(documents)} scores, got {len(scores)}")
#                 # Pad or truncate scores
#                 if len(scores) < len(documents):
#                     scores.extend([0.0] * (len(documents) - len(scores)))
#                 else:
#                     scores = scores[:len(documents)]
#             
#         except Exception as e:
#             print(f"Error getting GPT scores: {e}")
#             # Fallback to uniform scores
#             scores = [0.5] * len(documents)
#         
#         # Create list of (score, index) tuples and sort by score
#         scored_docs = [(scores[i], i) for i in range(len(documents))]
#         scored_docs.sort(key=lambda x: x[0], reverse=True)
#         
#         # Return reranked documents
#         reranked_docs = []
#         for score, idx in scored_docs[:top_k]:
#             doc = documents[idx]
#             # Add relevance score to metadata for debugging
#             if 'rerank_score' not in doc:
#                 doc['rerank_score'] = score
#             reranked_docs.append(doc)
#         
#         print(f"Reranking completed. Top relevance scores: {[f'{s:.3f}' for s, _ in scored_docs[:3]]}")
#         return reranked_docs
#         
#     except Exception as e:
#         print(f"Error in reranking: {e}")
#         return documents[:top_k]

async def get_embedding_with_model(text: str, model: str) -> List[float]:
    """Get embedding for text using specified OpenAI model."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding with {model}: {e}")
        return []

async def search_similar_documents(query: str, limit: int = 10, collection_name: str = None) -> List[Dict[str, Any]]:
    """Search for similar documents using Zilliz dedicated API over HTTP - SIMPLIFIED VERSION."""        
    try:
        # Use specified collection or default
        target_collection = collection_name or DEFAULT_COLLECTION
        
        print(f"🔍 DEBUG: Starting search in collection: {target_collection}")
        print(f"🔍 DEBUG: Query: {query}")
        print(f"🔍 DEBUG: Limit: {limit}")
        
        # Get query embedding for semantic search
        print(f"🔍 DEBUG: About to get embedding for query: '{query}'")
        query_embedding = await get_embedding(query)
        print(f'🔍 DEBUG: Embedding generated, length: {len(query_embedding) if query_embedding else 0}')
        if not query_embedding:
            print("❌ DEBUG: Failed to generate embedding")
            return []
        print(f"🔍 DEBUG: Embedding successful, proceeding with search")

        # Use query endpoint for text-based search (more reliable than vector search)
        search_url = f"{MILVUS_URI}/v2/vectordb/entities/query"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Query with filter to find relevant documents
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
        
        search_data = {
            "collectionName": target_collection,
            "filter": "",  # No filter for now, get all documents
            "limit": limit,
            "outputFields": output_fields
        }
        
        print(f"🔍 DEBUG: Search URL: {search_url}")
        print(f"🔍 DEBUG: Search data: {json.dumps(search_data, indent=2)}")

        response = requests.post(search_url, json=search_data, headers=headers)
        print(f"🔍 DEBUG: Milvus response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Zilliz API error: {response.status_code}")
            print(f"❌ DEBUG: Response text: {response.text}")
            return []
        
        result = response.json()
        pretty_json_string = json.dumps(result, indent=4)
        print(f'🔍 DEBUG: Milvus raw response: {pretty_json_string}')
        
        # Check if this is an error response
        if 'code' in result and 'message' in result:
            print(f"❌ DEBUG: Milvus API returned error: Code {result.get('code')}, Message: {result.get('message')}")
            print(f"❌ DEBUG: This suggests the Zilliz API format is incorrect")
            return []

        sources = []
        if 'data' in result:
            print(f"🔍 DEBUG: Found 'data' field in response with {len(result['data'])} items")
            
            # Let the RAG system work as intended - no manual filtering
            # The vector search already found relevant documents, let the LLM handle the rest
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
                        "text": hit.get('text_content', ''),
                        "metadata": metadata,
                        "collection": target_collection
                    }
                    sources.append(source_item)
                    print(f"🔍 DEBUG: Added source: {metadata.get('article_title', 'No title')}")
                
                except Exception as e:
                    print(f"❌ DEBUG: Error parsing hit: {e}")
                    continue
            
            # Return all sources up to the limit (let the LLM decide relevance)
            sources = sources[:limit]
            print(f"🔍 DEBUG: Returning {len(sources)} sources to LLM")
            
        else:
            print(f"❌ DEBUG: No 'data' field found in response. Available fields: {list(result.keys())}")
        
        print(f"🔍 DEBUG: Final sources count: {len(sources)}")
        if sources:
            pretty_json_string = json.dumps(sources, indent=4)
            print('🔍 DEBUG: Final sources:', pretty_json_string)
    
        return sources
        
    except Exception as e:
        print(f"❌ DEBUG: Error in search_similar_documents: {e}")
        import traceback
        traceback.print_exc()
        return []

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources - SIMPLIFIED VERSION."""
    if not client:
        return "OpenAI API key not configured."
    try:
        # Prepare system message - STRICT RAG ONLY
        system_message = """You are a RAG-based AI assistant that ONLY uses the provided sources to answer questions. 

CRITICAL RULES:
1. ONLY use information from the provided sources
2. DO NOT use any external knowledge or training data
3. If the sources don't contain enough information to answer a question, say "I can only answer based on the provided sources, and they don't contain enough information to answer this question."
4. Always cite specific sources when providing information
5. If asked about something not covered in the sources, explicitly state that you don't have that information in your sources

Your role is to analyze and present information from the provided regulatory intelligence sources only."""
        if sources:
            # Enhanced context with RSS feeds information
            context_parts = []
            for source in sources:
                try:
                    metadata = source.get('metadata', {})
                    article_title = metadata.get('article_title', 'Unknown Title')
                    published_date = metadata.get('published_date', 'Unknown Date')
                    feed_name = metadata.get('feed_name', 'Unknown Feed')
                    chunk_type = metadata.get('chunk_type', 'Unknown Type')
                    
                    print(f"Article: {article_title}, Date: {published_date}, Feed: {feed_name}")
                    
                    # Format the context with article information
                    context_parts.append(f"Source ({article_title} - {published_date} - {feed_name} - {chunk_type}): {source['text']}")
                except Exception as e :
                    print(e)
                    # Fallback if metadata parsing fails
                    context_parts.append(f"Source: {source['text']}")
            
            context = "\n\n".join(context_parts)
            system_message += f"\n\n<Sources>:\n{context}"
        

        print(system_message)
        # Prepare messages
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Limit to last 10 messages
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error chatting with GPT: {e}")
        return "I apologize, but I'm having trouble processing your request right now."

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client_request: Request):
    """Chat endpoint with RAG integration - SIMPLIFIED VERSION."""
    try:
        # Get client IP for rate limiting
        client_ip = client_request.client.host or "unknown"
        print(f"🌐 DEBUG: Client IP detected: {client_ip}")
        print(f"🌐 DEBUG: Client request headers: {dict(client_request.headers)}")
        
        # Check rate limiting
        rate_limit_status = check_rate_limit(client_ip)
        if not rate_limit_status["allowed"]:
            raise HTTPException(
                status_code=429, 
                detail={
                    "error": "Rate limit exceeded",
                    "reset_time": rate_limit_status["reset_time"],
                    "message": f"Too many requests. Try again in {int(rate_limit_status['reset_time'] - time.time())} seconds."
                }
            )
        
        # Additional content validation beyond Pydantic
        content_validation = validate_message_content(request.message)
        if not content_validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid message content",
                    "details": content_validation["errors"]
                }
            )
        
        # Validate conversation history
        history_validation = validate_conversation_history(request.conversation_history)
        if not history_validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid conversation history",
                    "details": history_validation["errors"]
                }
            )
        
        # Log validation warnings if any
        if content_validation["warnings"] or history_validation["warnings"]:
            print(f"⚠️ Validation warnings for IP {client_ip}: {content_validation['warnings'] + history_validation['warnings']}")
        
        # Search for relevant documents in rss_feeds collection (hardcoded for now)
        print(f"🔍 DEBUG: About to call search_similar_documents with query: '{request.message}'")
        print(f"🔍 DEBUG: Function search_similar_documents exists: {search_similar_documents is not None}")
        print(f"🔍 DEBUG: Function type: {type(search_similar_documents)}")
        
        sources = await search_similar_documents(request.message, collection_name="rss_feeds")
        
        print(f"🔍 DEBUG: search_similar_documents returned {len(sources)} sources")
        print(f"🔍 DEBUG: Sources: {sources}")

        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        
        # Validate that we have sources before proceeding
        if not sources:
            return ChatResponse(
                response="I cannot answer this question as I don't have any relevant sources in my knowledge base. Please try rephrasing your question or ask about a different topic that might be covered in the available regulatory intelligence sources.",
                sources=[]
            )
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        
        # SIMPLIFIED: No reranking information
        # reranking_info = { ... } - COMMENTED OUT
        
        return ChatResponse(
            response=response,
            sources=sources,
            # reranking_info=reranking_info - COMMENTED OUT
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/{collection}", response_model=ChatResponse)
async def chat_with_collection(collection: str, request: ChatRequest, client_request: Request):
    """Chat endpoint with RAG integration for a specific collection."""
    try:
        # Get client IP for rate limiting
        client_ip = client_request.client.host or "unknown"
        
        # Check rate limiting
        rate_limit_status = check_rate_limit(client_ip)
        if not rate_limit_status["allowed"]:
            raise HTTPException(
                status_code=429, 
                detail={
                    "error": "Rate limit exceeded",
                    "reset_time": rate_limit_status["reset_time"],
                    "message": f"Too many requests. Try again in {int(rate_limit_status['reset_time'] - time.time())} seconds."
                }
            )
        
        # Validate collection name
        valid_collections = [FDA_WARNING_LETTERS_COLLECTION, RSS_FEEDS_COLLECTION]
        if collection not in valid_collections:
            raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {valid_collections}")
        
        # Additional content validation beyond Pydantic
        content_validation = validate_message_content(request.message)
        if not content_validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid message content",
                    "details": content_validation["errors"]
                }
            )
        
        # Validate conversation history
        history_validation = validate_conversation_history(request.conversation_history)
        if not history_validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid conversation history",
                    "details": history_validation["errors"]
                }
            )
        
        # Log validation warnings if any
        if content_validation["warnings"] or history_validation["warnings"]:
            print(f"⚠️ Validation warnings for IP {client_ip}: {content_validation['warnings'] + history_validation['warnings']}")
        
        # Search for relevant documents in the specified collection
        sources = await search_similar_documents(request.message, collection_name=collection)
        
        print(f"Searching in collection: {collection}")
        print(sources)

        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        
        # Validate that we have sources before proceeding
        if not sources:
            return ChatResponse(
                response="I cannot answer this question as I don't have any relevant sources in my knowledge base. Please try rephrasing your question or ask about a different topic that might be covered in the available regulatory intelligence sources.",
                sources=[]
            )
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        
        return ChatResponse(
            response=response,
            sources=sources,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get current RAG configuration."""
    return {
        "strict_rag_only": STRICT_RAG_ONLY,
        "enable_reranking": ENABLE_RERANKING,
        "reranking_model": RERANKING_MODEL,
        "initial_search_multiplier": INITIAL_SEARCH_MULTIPLIER,
        "default_collection": DEFAULT_COLLECTION,
        "milvus_uri": MILVUS_URI[:50] + "..." if MILVUS_URI and len(MILVUS_URI) > 50 else MILVUS_URI,
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.get("/api/collections")
async def get_collections():
    """Get available collections."""
    return {
        "available_collections": [
            {
                "name": "rss_feeds", 
                "env_var": "RSS_FEEDS_COLLECTION",
                "current_value": RSS_FEEDS_COLLECTION,
                "description": "RSS feeds for regulatory intelligence and medical device news"
            },
            {
                "name": "fda_warning_letters",
                "env_var": "FDA_WARNING_LETTERS_COLLECTION",
                "current_value": FDA_WARNING_LETTERS_COLLECTION,
                "description": "FDA warning letters and regulatory documents"
            }
        ],
        "default_collection": DEFAULT_COLLECTION,
        "default_description": "RSS feeds collection (regulatory intelligence)"
    }

@app.get("/api/collections/{collection}/list")
async def list_collection_documents(collection: str, limit: int = 10):
    """List documents in a collection without vector search - for debugging."""
    try:
        # Validate collection name
        valid_collections = [FDA_WARNING_LETTERS_COLLECTION, RSS_FEEDS_COLLECTION]
        if collection not in valid_collections:
            raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {valid_collections}")
        
        # Simple query to list documents
        query_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Query without vector search, just to see what's there
        query_data = {
            "collectionName": collection,
            "filter": "",  # No filter, get all documents
            "limit": limit,
            "outputFields": ["id", "text_content", "article_title", "published_date", "feed_name", "chunk_type", "text_vector"]
        }
        
        print(f"🔍 DEBUG: Listing documents in {collection}")
        print(f"🔍 DEBUG: Query data: {json.dumps(query_data, indent=2)}")
        
        response = requests.post(query_url, json=query_data, headers=headers)
        print(f"🔍 DEBUG: List response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: List failed: {response.text}")
            return {"error": f"Failed to list documents: {response.status_code}"}
        
        result = response.json()
        print(f"🔍 DEBUG: List response: {json.dumps(result, indent=2)}")
        
        return {
            "collection": collection,
            "total_documents": len(result.get('data', [])),
            "documents": result.get('data', []),
            "embedding_status": {
                "total": len(result.get('data', [])),
                "with_embeddings": len([doc for doc in result.get('data', []) if doc.get('text_vector') and len(doc.get('text_vector', [])) > 0]),
                "without_embeddings": len([doc for doc in result.get('data', []) if not doc.get('text_vector') or len(doc.get('text_vector', [])) == 0])
            }
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error listing documents: {e}")
        return {"error": str(e)}

# Reranking config endpoint - COMMENTED OUT FOR SIMPLE VERSION
# @app.get("/api/reranking-config", response_model=RerankingConfig)
# async def get_reranking_config():
#     """Get current reranking configuration."""
#     return RerankingConfig(
#         enabled=ENABLE_RERANKING,
#         model=RERANKING_MODEL,
#         initial_search_multiplier=INITIAL_SEARCH_MULTIPLIER
#     )

@app.post("/api/add-document")
async def add_document(request: AddDocumentRequest):
    """Add a document to the RAG system using Zilliz dedicated API - adds to default collection."""
    try:
        # Generate Murmur3 hash of the text as primary key
        text_hash = mmh3.hash(request.text)
        
        print(text_hash)
        
        # Check if document already exists using Zilliz API
        query_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        query_data = {
            "collectionName": DEFAULT_COLLECTION,
            "filter": f"primary_key == {text_hash}",
            "outputFields": ["primary_key"]
        }

        response = requests.post(query_url, json=query_data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                return {"message": "Document already exists", "id": text_hash}
        
        # Get embedding
        embedding = await get_embedding(request.text)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        json_metadata = json.loads(request.metadata)
        print(json_metadata)
        print(text_hash)
        
        # Debug: Show Milvus configuration being used
        print(f"🔍 DEBUG: Using Milvus URI: {MILVUS_URI}")
        print(f"🔍 DEBUG: Using Milvus Token: {MILVUS_TOKEN[:20]}...{MILVUS_TOKEN[-20:] if MILVUS_TOKEN else 'None'}")

        # Insert into Zilliz using HTTP API
        insert_url = f"{MILVUS_URI}/v2/vectordb/entities/upsert"
        insert_data = {
            "collectionName": DEFAULT_COLLECTION,
            "data": {
                "id": text_hash,
                "text_content": request.text,  # Use text_content instead of text
                "text_vector": embedding,      # Use text_vector instead of vector
                "article_title": json_metadata.get('article_title', 'Unknown Title'),
                "published_date": json_metadata.get('published_date', 'Unknown Date'),
                "feed_name": json_metadata.get('feed_name', 'Unknown Feed'),
                "chunk_id": json_metadata.get('chunk_id', f'chunk_{text_hash}'),
                "chunk_type": json_metadata.get('chunk_type', 'article_section'),
                "chunk_index": json_metadata.get('chunk_index', 0),
                "total_chunks": json_metadata.get('total_chunks', 1),
                "text_length": len(request.text),
                "estimated_tokens": len(request.text.split()) // 4,  # Rough estimate
                "companies": json_metadata.get('companies', []),
                "products": json_metadata.get('products', []),
                "regulations": json_metadata.get('regulations', []),
                "regulatory_bodies": json_metadata.get('regulatory_bodies', []),
                "people": json_metadata.get('people', []),
                "locations": json_metadata.get('locations', []),
                "dates": json_metadata.get('dates', []),
                "summary": json_metadata.get('summary', ''),
                "article_tags": json_metadata.get('article_tags', []),
                "total_entities": json_metadata.get('total_entities', 0),
                "created_at": json_metadata.get('created_at', '2024-01-01T00:00:00Z'),
                "updated_at": json_metadata.get('updated_at', '2024-01-01T00:00:00Z')
            }
        }

        response = requests.post(insert_url, json=insert_data, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to insert document: {response.status_code}")
        
        return {"message": "Document added successfully", "id": text_hash}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-document/{collection}")
async def add_document_to_collection(collection: str, request: AddDocumentRequest):
    """Add a document to a specific collection in the RAG system."""
    try:
        # Validate collection name
        valid_collections = [FDA_WARNING_LETTERS_COLLECTION, RSS_FEEDS_COLLECTION]
        if collection not in valid_collections:
            raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {valid_collections}")
        
        # Generate Murmur3 hash of the text as primary key
        text_hash = mmh3.hash(request.text)
        
        print(f"Adding document to collection: {collection}")
        print(text_hash)
        
        # Check if document already exists using Zilliz API
        query_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        query_data = {
            "collectionName": collection,
            "filter": f"primary_key == {text_hash}",
            "outputFields": ["primary_key"]
        }

        response = requests.post(query_url, json=query_data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                return {"message": "Document already exists", "id": text_hash}
        
        # Get embedding
        embedding = await get_embedding(request.text)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        json_metadata = json.loads(request.metadata)
        print(json_metadata)
        print(text_hash)

        # Insert into Zilliz using HTTP API
        insert_url = f"{MILVUS_URI}/v2/vectordb/entities/upsert"
        insert_data = {
            "collectionName": collection,
            "data": {
                "id": text_hash,
                "text_content": request.text,  # Use text_content instead of text
                "text_vector": embedding,      # Use text_vector instead of vector
                "article_title": json_metadata.get('article_title', 'Unknown Title'),
                "published_date": json_metadata.get('published_date', 'Unknown Date'),
                "feed_name": json_metadata.get('feed_name', 'Unknown Feed'),
                "chunk_id": json_metadata.get('chunk_id', f'chunk_{text_hash}'),
                "chunk_type": json_metadata.get('chunk_type', 'article_section'),
                "chunk_index": json_metadata.get('chunk_index', 0),
                "total_chunks": json_metadata.get('total_chunks', 1),
                "text_length": len(request.text),
                "estimated_tokens": len(request.text.split()) // 4,  # Rough estimate
                "companies": json_metadata.get('companies', []),
                "products": json_metadata.get('products', []),
                "regulations": json_metadata.get('regulations', []),
                "regulatory_bodies": json_metadata.get('regulatory_bodies', []),
                "people": json_metadata.get('people', []),
                "locations": json_metadata.get('locations', []),
                "dates": json_metadata.get('dates', []),
                "summary": json_metadata.get('summary', ''),
                "article_tags": json_metadata.get('article_tags', []),
                "total_entities": json_metadata.get('total_entities', 0),
                "created_at": json_metadata.get('created_at', '2024-01-01T00:00:00Z'),
                "updated_at": json_metadata.get('updated_at', '2024-01-01T00:00:00Z')
            }
        }

        response = requests.post(insert_url, json=insert_data, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to insert document: {response.status_code}")
        
        return {"message": f"Document added successfully to {collection}", "id": text_hash}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/collections/{collection}/update-embeddings")
async def update_collection_embeddings(collection: str):
    """Update existing documents in a collection with embeddings - for documents that exist but lack text_vector."""
    try:
        # Validate collection name
        valid_collections = [FDA_WARNING_LETTERS_COLLECTION, RSS_FEEDS_COLLECTION]
        if collection not in valid_collections:
            raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {valid_collections}")
        
        print(f"🔄 DEBUG: Updating embeddings for collection: {collection}")
        
        # First, get all documents without embeddings
        query_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Get documents that need embeddings
        query_data = {
            "collectionName": collection,
            "filter": "",  # Get all documents
            "limit": 100,  # Adjust as needed
            "outputFields": ["id", "text_content"]
        }
        
        response = requests.post(query_url, json=query_data, headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to fetch documents: {response.status_code}"}
        
        result = response.json()
        documents = result.get('data', [])
        
        if not documents:
            return {"message": "No documents found to update"}
        
        print(f"🔄 DEBUG: Found {len(documents)} documents to update")
        
        updated_count = 0
        for doc in documents:
            try:
                doc_id = doc.get('id')
                text_content = doc.get('text_content', '')
                
                if not text_content:
                    continue
                
                # Generate embedding for the text content
                embedding = await get_embedding(text_content)
                if not embedding:
                    print(f"❌ DEBUG: Failed to generate embedding for document {doc_id}")
                    continue
                
                # Update the document with the embedding
                update_url = f"{MILVUS_URI}/v2/vectordb/entities/upsert"
                update_data = {
                    "collectionName": collection,
                    "data": {
                        "id": doc_id,
                        "text_vector": embedding
                    }
                }
                
                update_response = requests.post(update_url, json=update_data, headers=headers)
                if update_response.status_code == 200:
                    updated_count += 1
                    print(f"✅ DEBUG: Updated document {doc_id} with embedding")
                else:
                    print(f"❌ DEBUG: Failed to update document {doc_id}: {update_response.status_code}")
                    
            except Exception as e:
                print(f"❌ DEBUG: Error updating document {doc.get('id')}: {e}")
                continue
        
        return {
            "message": f"Updated {updated_count} out of {len(documents)} documents with embeddings",
            "total_documents": len(documents),
            "updated_count": updated_count
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error updating embeddings: {e}")
        return {"error": str(e)}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Zilliz API connection
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Simple health check using list collections endpoint
        health_url = f"{MILVUS_URI}/v2/vectordb/collections/list"
        response = requests.post(health_url, headers=headers)
        zilliz_connected = response.status_code == 200
        print(zilliz_connected)
        return {"status": "healthy", "zilliz_connected": zilliz_connected}
    except Exception as e:
        return {"status": "unhealthy", "zilliz_connected": False, "error": str(e)}

@app.get("/api/test-search")
async def test_search_get(query: str = "stryker", limit: int = 5):
    """Test endpoint to debug search functionality (GET)"""
    try:
        print(f"🧪 TEST SEARCH GET: Query='{query}', Limit={limit}")
        
        # Test the search function directly
        sources = await search_similar_documents(query, limit)
        
        print(f"🧪 TEST SEARCH GET: Found {len(sources)} sources")
        
        return {
            "query": query,
            "limit": limit,
            "sources_found": len(sources),
            "sources": sources,
            "debug_info": {
                "collection_used": DEFAULT_COLLECTION,
                "milvus_uri": MILVUS_URI[:50] + "..." if MILVUS_URI and len(MILVUS_URI) > 50 else MILVUS_URI,
                "openai_configured": bool(OPENAI_API_KEY)
            }
        }
        
    except Exception as e:
        print(f"❌ TEST SEARCH GET ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    """Test endpoint to debug search functionality"""
    try:
        print(f"🧪 TEST SEARCH: Query='{query}', Limit={limit}")
        
        # Test the search function directly
        sources = await search_similar_documents(query, limit)
        
        print(f"🧪 TEST SEARCH: Found {len(sources)} sources")
        
        return {
            "query": query,
            "limit": limit,
            "sources_found": len(sources),
            "sources": sources,
            "debug_info": {
                "collection_used": DEFAULT_COLLECTION,
                "milvus_uri": MILVUS_URI[:50] + "..." if MILVUS_URI and len(MILVUS_URI) > 50 else MILVUS_URI,
                "openai_configured": bool(OPENAI_API_KEY)
            }
        }
        
    except Exception as e:
        print(f"❌ TEST SEARCH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/debug/collection/{collection_name}")
async def debug_collection(collection_name: str):
    """Debug endpoint to see what's actually in a collection."""
    try:
        # Validate collection name
        valid_collections = [FDA_WARNING_LETTERS_COLLECTION, RSS_FEEDS_COLLECTION]
        if collection_name not in valid_collections:
            raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {valid_collections}")
        
        print(f"🔍 DEBUG: Checking collection: {collection_name}")
        
        # Try to get collection info
        info_url = f"{MILVUS_URI}/v2/vectordb/collections/describe"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        info_data = {
            "collectionName": collection_name
        }
        
        print(f"🔍 DEBUG: Collection info URL: {info_url}")
        print(f"🔍 DEBUG: Collection info data: {json.dumps(info_data, indent=2)}")
        
        response = requests.post(info_url, json=info_data, headers=headers)
        print(f"🔍 DEBUG: Collection info response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Failed to get collection info: {response.status_code}")
            print(f"❌ DEBUG: Response: {response.text}")
            return {"error": f"Failed to get collection info: {response.status_code}", "response": response.text}
        
        collection_info = response.json()
        print(f"🔍 DEBUG: Collection info: {json.dumps(collection_info, indent=2)}")
        
        # Try to get some sample data
        sample_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        sample_data = {
            "collectionName": collection_name,
            "data": [[0.0] * 1536],  # Dummy embedding
            "limit": 5,
            "outputFields": ["*"]  # Get all fields
        }
        
        print(f"🔍 DEBUG: Sample data URL: {sample_url}")
        print(f"🔍 DEBUG: Sample data request: {json.dumps(sample_data, indent=2)}")
        
        sample_response = requests.post(sample_url, json=sample_data, headers=headers)
        print(f"🔍 DEBUG: Sample data response status: {sample_response.status_code}")
        
        if sample_response.status_code != 200:
            print(f"❌ DEBUG: Failed to get sample data: {sample_response.status_code}")
            print(f"❌ DEBUG: Response: {sample_response.text}")
            return {
                "collection_info": collection_info,
                "error": f"Failed to get sample data: {sample_response.status_code}",
                "sample_response": sample_response.text
            }
        
        sample_result = sample_response.json()
        print(f"🔍 DEBUG: Sample data result: {json.dumps(sample_result, indent=2)}")
        
        return {
            "collection_info": collection_info,
            "sample_data": sample_result,
            "collection_name": collection_name,
            "milvus_uri": MILVUS_URI
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Error in debug_collection: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/validation/status")
async def get_validation_status():
    """Get current validation configuration and status."""
    return {
        "validation_config": {
            "max_message_length": MAX_MESSAGE_LENGTH,
            "max_conversation_history": MAX_CONVERSATION_HISTORY,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW
        },
        "blocked_patterns": {
            "general": BLOCKED_PATTERNS,
            "medical": MEDICAL_BLOCKED_PATTERNS
        },
        "rate_limit_status": {
            "active_connections": len(rate_limit_storage),
            "storage_type": "in_memory"
        },
        "validation_features": [
            "message_length_validation",
            "content_pattern_validation", 
            "conversation_history_validation",
            "rate_limiting",
            "medical_advice_detection",
            "injection_attempt_detection"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 