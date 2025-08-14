#!/usr/bin/env python3
"""
Milvus Client for RSS Feeds RAG (Environment Variables Version)

This module handles all Milvus operations including:
- Connection management using environment variables
- Data insertion
- Vector search
- Collection management
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from pymilvus import connections, Collection, utility
import openai

logger = logging.getLogger(__name__)

class MilvusClientEnv:
    """Handles all Milvus operations for RSS feeds using environment variables"""
    
    def __init__(self):
        self.collection_name = "rss_feeds"
        self.milvus_uri = os.environ.get('MILVUS_URI')
        self.milvus_token = os.environ.get('MILVUS_TOKEN')
        self.collection = None
        self.openai_client = None
        
        # Initialize OpenAI client if API key is available
        self._init_openai()
        
    def _init_openai(self):
        """Initialize OpenAI client for embeddings"""
        try:
            # Try to get OpenAI API key from environment
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            
            if not openai_api_key:
                logger.warning("OpenAI API key not found in environment variables")
                logger.warning("Text embedding will be disabled.")
                return
            
            # Initialize OpenAI client for version 1.99.9+
            try:
                # For OpenAI 1.99.9+, we create a client object
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI client initialized successfully (using 1.99.9+ API)")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                logger.warning("OpenAI client not available. Text embedding will be disabled.")
                self.openai_client = None
                return
            
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    def connect_to_milvus(self) -> bool:
        """Connect to Milvus cluster using environment variables"""
        try:
            if not self.milvus_uri or not self.milvus_token:
                logger.error("Missing required credentials: MILVUS_URI or MILVUS_TOKEN")
                return False
            
            logger.info("Connecting to Milvus cluster...")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=self.milvus_uri,
                token=self.milvus_token
            )
            
            logger.info(f"✅ Successfully connected to Milvus cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def get_collection(self) -> bool:
        """Get existing collection (for read operations)"""
        try:
            # Verify collection exists
            if not utility.has_collection(self.collection_name):
                logger.error(f"Collection '{self.collection_name}' does not exist")
                return False
            
            # Get collection
            self.collection = Collection(self.collection_name)
            
            # Load collection into memory
            self.collection.load()
            
            logger.info(f"✅ Successfully loaded collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to get collection: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using OpenAI"""
        if not self.openai_client:
            logger.warning("OpenAI client not available. Cannot generate embeddings.")
            return None
        
        try:
            # Use OpenAI 1.99.9+ API
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: {len(text)} chars -> {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def prepare_chunk_for_insertion(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a chunk for insertion into Milvus"""
        try:
            # Check if chunk already has a vector
            if 'text_vector' in chunk and chunk['text_vector']:
                # Use pre-generated vector
                embedding = chunk['text_vector']
                text_content = chunk.get('text_content', '')
                logger.debug(f"Using pre-generated vector for chunk: {chunk.get('chunk_id', 'unknown')}")
            else:
                # Generate embedding for text content
                text_content = chunk.get('text_content', '')
                if not text_content:
                    logger.warning(f"No text content found in chunk: {chunk.get('chunk_id', 'unknown')}")
                    return None
                
                embedding = self.generate_embedding(text_content)
                if not embedding:
                    logger.warning(f"Failed to generate embedding for chunk: {chunk.get('chunk_id', 'unknown')}")
                    return None
            
            # Prepare data for Milvus insertion
            milvus_data = {
                'text_vector': embedding,
                'article_id': chunk.get('article_id', ''),
                'article_title': chunk.get('article_title', ''),
                'published_date': chunk.get('published_date', ''),
                'feed_name': chunk.get('feed_name', ''),
                'author': chunk.get('author', ''),
                'article_link': chunk.get('article_link', ''),
                'chunk_id': chunk.get('chunk_id', ''),
                'chunk_type': chunk.get('chunk_type', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'text_content': text_content,
                'text_length': chunk.get('text_length', 0),
                'estimated_tokens': chunk.get('estimated_tokens', 0),
                'companies': json.dumps(chunk.get('metadata', {}).get('companies', [])),
                'products': json.dumps(chunk.get('metadata', {}).get('products', [])),
                'regulations': json.dumps(chunk.get('metadata', {}).get('regulations', [])),
                'regulatory_bodies': json.dumps(chunk.get('metadata', {}).get('regulatory_bodies', [])),
                'people': json.dumps(chunk.get('metadata', {}).get('people', [])),
                'locations': json.dumps(chunk.get('metadata', {}).get('locations', [])),
                'dates': json.dumps(chunk.get('metadata', {}).get('dates', [])),
                'summary': chunk.get('metadata', {}).get('summary', ''),
                'article_tags': json.dumps(chunk.get('article_tags', [])),
                'total_entities': chunk.get('metadata', {}).get('total_entities', 0),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.debug(f"Prepared chunk for insertion: {chunk.get('chunk_id', 'unknown')}")
            return milvus_data
            
        except Exception as e:
            logger.error(f"Failed to prepare chunk for insertion: {e}")
            return None
    
    def overwrite_collection(self, chunks: List[Dict[str, Any]]) -> bool:
        """Overwrite the entire collection with new data"""
        try:
            if not chunks:
                logger.warning("No chunks to insert")
                return False
            
            logger.info("🔄 Starting collection overwrite process...")
            
            # Step 1: Connect to Milvus cluster first
            if not self.connect_to_milvus():
                logger.error("Failed to connect to Milvus cluster")
                return False
            
            # Step 2: Drop existing collection if it exists
            if utility.has_collection(self.collection_name):
                logger.info(f"🗑️ Dropping existing collection '{self.collection_name}'...")
                utility.drop_collection(self.collection_name)
                logger.info("✅ Existing collection dropped")
            
            # Step 3: Recreate collection with current schema
            logger.info("🏗️ Recreating collection with current schema...")
            self.create_collection()
            
            # Step 4: Create indexes
            logger.info("🔍 Creating indexes...")
            self.create_indexes()
            
            # Step 5: Load collection
            logger.info("📚 Loading collection...")
            self.load_collection()
            
            # Step 6: Prepare and insert chunks
            logger.info("📝 Preparing chunks for insertion...")
            prepared_chunks = []
            for chunk in chunks:
                prepared_chunk = self.prepare_chunk_for_insertion(chunk)
                if prepared_chunk:
                    prepared_chunks.append(prepared_chunk)
                else:
                    logger.warning(f"Skipping chunk: {chunk.get('chunk_id', 'unknown')}")
            
            if not prepared_chunks:
                logger.error("No chunks prepared for insertion")
                return False
            
            # Step 7: Insert all chunks
            logger.info(f"🚀 Inserting {len(prepared_chunks)} chunks into fresh collection...")
            insert_result = self.collection.insert(prepared_chunks)
            
            # Step 8: Flush to ensure data is persisted
            logger.info("💾 Flushing collection to ensure data persistence...")
            self.collection.flush()
            
            logger.info(f"🎉 SUCCESS! Collection overwrite completed:")
            logger.info(f"   - Collection: {self.collection_name}")
            logger.info(f"   - Chunks inserted: {len(prepared_chunks)}")
            logger.info(f"   - Insert result: {insert_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during collection overwrite: {e}")
            return False

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert chunks into existing Milvus collection (legacy method)"""
        try:
            if not self.collection:
                logger.error("Not connected to Milvus collection")
                return False
            
            logger.info(f"Preparing {len(chunks)} chunks for insertion...")
            
            # Prepare all chunks
            prepared_chunks = []
            for chunk in chunks:
                prepared_chunk = self.prepare_chunk_for_insertion(chunk)
                if prepared_chunk:
                    prepared_chunks.append(prepared_chunk)
                else:
                    logger.warning(f"Skipping chunk: {chunk.get('chunk_id', 'unknown')}")
            
            if not prepared_chunks:
                logger.error("No chunks prepared for insertion")
                return False
            
            logger.info(f"Inserting {len(prepared_chunks)} chunks into Milvus...")
            
            # Insert chunks
            insert_result = self.collection.insert(prepared_chunks)
            
            # Flush to ensure data is persisted
            self.collection.flush()
            
            logger.info(f"✅ Successfully inserted {len(prepared_chunks)} chunks")
            logger.info(f"   Insert result: {insert_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            return False
    
    def search_similar_chunks(self, query_text: str, limit: int = 10, 
                            filter_expr: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            if not self.collection:
                logger.error("Not connected to Milvus collection")
                return []
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            search_result = self.collection.search(
                data=[query_embedding],
                anns_field="text_vector",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["*"]
            )
            
            # Process results
            results = []
            for hits in search_result:
                for hit in hits:
                    result = {
                        'score': hit.score,
                        'chunk_id': hit.entity.get('chunk_id'),
                        'article_title': hit.entity.get('article_title'),
                        'published_date': hit.entity.get('published_date'),
                        'feed_name': hit.entity.get('feed_name'),
                        'chunk_type': hit.entity.get('chunk_type'),
                        'text_content': hit.entity.get('text_content'),
                        'companies': json.loads(hit.entity.get('companies', '[]')),
                        'products': json.loads(hit.entity.get('products', '[]')),
                        'regulations': json.loads(hit.entity.get('regulations', '[]')),
                        'regulatory_bodies': json.loads(hit.entity.get('regulatory_bodies', '[]'))
                    }
                    results.append(result)
            
            logger.info(f"✅ Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                return {}
            
            stats = {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "indexes": self.collection.indexes,
                "partitions": self.collection.partitions
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def close_connection(self):
        """Close Milvus connection"""
        try:
            if self.collection:
                self.collection.release()
            
            connections.disconnect("default")
            logger.info("✅ Milvus connection closed")
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def create_collection(self) -> bool:
        """Create the Milvus collection with the defined schema"""
        try:
            from setup_milvus_collection import MilvusCollectionSetup
            
            # Create setup instance and use existing connection
            setup = MilvusCollectionSetup()
            
            # Set the credentials from environment variables
            setup.milvus_uri = self.milvus_uri
            setup.milvus_token = self.milvus_token
            
            # Create collection schema first
            schema = setup.create_collection_schema()
            if not schema:
                logger.error("Failed to create collection schema")
                return False
            
            # Create collection using our credentials and schema
            collection = setup.create_collection(schema)
            if not collection:
                logger.error("Failed to create collection")
                return False
            
            # Store the created collection
            self.collection = collection
            logger.info(f"✅ Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create indexes for the collection"""
        try:
            if not self.collection:
                logger.error("No collection available for indexing")
                return False
            
            # Create HNSW index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 500}
            }
            
            self.collection.create_index(
                field_name="text_vector",
                index_params=index_params
            )
            
            logger.info("✅ HNSW index created for text_vector field")
            logger.info("ℹ️  VARCHAR fields are automatically indexed for filtering")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False
    
    def load_collection(self) -> bool:
        """Load the collection into memory"""
        try:
            if not self.collection:
                logger.error("No collection available to load")
                return False
            
            self.collection.load()
            logger.info(f"✅ Collection '{self.collection_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            return False 