"""
Vector Store Manager Module

Provides unified vector storage abstraction with support for:
- OpenSearch (existing infrastructure)
- User document isolation
- Hybrid search (BM25 + vector)
- LlamaIndex integration
"""

import logging
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

import aiohttp
from pydantic import BaseModel

from .config import get_config
from .document_manager import Document, DocumentChunk, DocumentType, DocumentStatus

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Search result with relevance score."""
    id: str
    document_id: str
    user_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    
    # Source information
    source_type: str = "document"  # "document", "citation", "general"
    source_url: Optional[str] = None
    title: Optional[str] = None


class VectorStoreManager:
    """
    Unified vector store manager for document indexing and retrieval.
    
    Features:
    - OpenSearch backend
    - Hybrid search (BM25 + KNN vector)
    - User document isolation
    - Citation indexing
    - LlamaIndex-compatible interface
    """
    
    def __init__(
        self,
        config: Optional["RAGConfig"] = None,
        openai_client: Optional[Any] = None,
    ):
        self.config = config or get_config()
        self.os_config = self.config.opensearch
        self.openai_client = openai_client
        
        # OpenSearch connection settings
        self.endpoint = self.os_config.endpoint
        self.auth = (self.os_config.user, self.os_config.password)
        
        # Index names
        self.general_index = self.os_config.general_index
        self.user_docs_index = self.os_config.user_docs_index
        self.citations_index = self.os_config.citations_index
        
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenSearch requests."""
        return {"Content-Type": "application/json"}
    
    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenSearch."""
        url = f"{self.endpoint}/{path}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    auth=aiohttp.BasicAuth(self.auth[0], self.auth[1]),
                    json=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status >= 400:
                        text = await response.text()
                        logger.error(f"OpenSearch error: {response.status} - {text}")
                        raise Exception(f"OpenSearch error: {response.status}")
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"OpenSearch request failed: {e}")
                raise
    
    async def ensure_indices(self) -> None:
        """Create necessary indices if they don't exist."""
        # User documents index with KNN support
        user_docs_mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        },
                        "nori_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {
                        "type": "text",
                        "analyzer": "nori_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.os_config.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    },
                    "doc_type": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "source_url": {"type": "keyword"},
                }
            }
        }
        
        # Citations index
        citations_mapping = {
            "settings": {
                "index": {"knn": True},
                "analysis": {
                    "analyzer": {
                        "nori_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "paper_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "nori_analyzer"
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "nori_analyzer"
                    },
                    "authors": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "venue": {"type": "text"},
                    "doi": {"type": "keyword"},
                    "citation_key": {"type": "keyword"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.os_config.embedding_dimension,
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "nori_analyzer"
                    },
                    "created_at": {"type": "date"},
                }
            }
        }
        
        # Create indices (ignore if exists)
        for index_name, mapping in [
            (self.user_docs_index, user_docs_mapping),
            (self.citations_index, citations_mapping),
        ]:
            try:
                await self._make_request("PUT", index_name, mapping)
                logger.info(f"Created index: {index_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create index {index_name}: {e}")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text using OpenAI."""
        if not self.openai_client:
            logger.error("OpenAI client not configured for embeddings")
            return None
            
        try:
            # Support both sync and async OpenAI clients
            if hasattr(self.openai_client, 'embeddings'):
                response = self.openai_client.embeddings.create(
                    model=self.config.openai.embedding_model,
                    input=text,
                )
                return response.data[0].embedding
            else:
                # Async client
                response = await self.openai_client.embeddings.create(
                    model=self.config.openai.embedding_model,
                    input=text,
                )
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    async def index_document_chunks(
        self,
        document_id: str,
        user_id: str,
        chunks: List[DocumentChunk],
    ) -> int:
        """
        Index document chunks with embeddings.
        
        Args:
            document_id: Document ID
            user_id: User ID for isolation
            chunks: List of document chunks
            
        Returns:
            Number of successfully indexed chunks
        """
        indexed_count = 0
        
        for chunk in chunks:
            # Generate embedding
            embedding = await self.generate_embedding(chunk.content)
            
            doc_data = {
                "id": chunk.id,
                "document_id": document_id,
                "user_id": user_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "vector": embedding,
                "metadata": chunk.metadata,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            try:
                await self._make_request(
                    "POST",
                    f"{self.user_docs_index}/_doc/{chunk.id}",
                    doc_data
                )
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index chunk {chunk.id}: {e}")
                
        return indexed_count
    
    async def index_citation(
        self,
        user_id: str,
        citation_data: Dict[str, Any],
        content: str,
    ) -> bool:
        """
        Index a citation with embedding.
        
        Args:
            user_id: User ID
            citation_data: Citation metadata (paper info)
            content: Text content for embedding
            
        Returns:
            True if indexing was successful
        """
        embedding = await self.generate_embedding(content)
        
        doc_data = {
            "id": citation_data.get("paper_id"),
            "user_id": user_id,
            "paper_id": citation_data.get("paper_id"),
            "title": citation_data.get("title"),
            "abstract": citation_data.get("abstract"),
            "authors": citation_data.get("authors", []),
            "year": citation_data.get("year"),
            "venue": citation_data.get("venue"),
            "doi": citation_data.get("doi"),
            "citation_key": citation_data.get("citation_key"),
            "content": content,
            "vector": embedding,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        try:
            await self._make_request(
                "POST",
                f"{self.citations_index}/_doc/{citation_data.get('paper_id')}_{user_id}",
                doc_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index citation: {e}")
            return False
    
    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        user_id: Optional[str] = None,
        search_sources: List[str] = ["general", "user_docs", "citations"],
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector similarity.
        
        Args:
            query: Text query
            query_vector: Query embedding vector
            user_id: Optional user ID for user-specific documents
            search_sources: Sources to search ["general", "user_docs", "citations"]
            top_k: Number of results per source
            alpha: Weight for vector score (1-alpha for BM25)
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        results = []
        
        # Search general documents
        if "general" in search_sources:
            general_results = await self._search_general_index(
                query, query_vector, top_k
            )
            for r in general_results:
                r.source_type = "general"
                results.append(r)
        
        # Search user documents
        if "user_docs" in search_sources and user_id:
            user_results = await self._search_user_index(
                query, query_vector, user_id, top_k
            )
            for r in user_results:
                r.source_type = "document"
                results.append(r)
        
        # Search citations
        if "citations" in search_sources and user_id:
            citation_results = await self._search_citations_index(
                query, query_vector, user_id, top_k
            )
            for r in citation_results:
                r.source_type = "citation"
                results.append(r)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k * len(search_sources)]
    
    async def _search_general_index(
        self,
        query: str,
        query_vector: List[float],
        top_k: int,
    ) -> List[SearchResult]:
        """Search the general (existing) document index."""
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "merged_text": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "knn": {
                                "vector": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": 2.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["url", "merged_text", "tables", "title"]
        }
        
        try:
            # Use existing pipeline
            url = f"{self.general_index}/_search?search_pipeline=hybrid-pipeline-balanced"
            response = await self._make_request("POST", url, search_query)
            
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append(SearchResult(
                    id=hit.get("_id"),
                    document_id=hit.get("_id"),
                    user_id="",
                    content=source.get("merged_text", ""),
                    score=hit.get("_score", 0),
                    metadata={
                        "tables": source.get("tables"),
                    },
                    source_url=source.get("url"),
                    title=source.get("title"),
                ))
            return results
        except Exception as e:
            logger.error(f"General index search failed: {e}")
            return []
    
    async def _search_user_index(
        self,
        query: str,
        query_vector: List[float],
        user_id: str,
        top_k: int,
    ) -> List[SearchResult]:
        """Search user-specific document index."""
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {"user_id": user_id}
                        }
                    ],
                    "should": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "knn": {
                                "vector": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": 2.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["document_id", "content", "metadata", "title", "source_url"]
        }
        
        try:
            response = await self._make_request(
                "POST",
                f"{self.user_docs_index}/_search",
                search_query
            )
            
            results = []
            seen_docs = set()
            
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                doc_id = source.get("document_id")
                
                # Avoid duplicate documents (multiple chunks)
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                
                results.append(SearchResult(
                    id=hit.get("_id"),
                    document_id=doc_id,
                    user_id=user_id,
                    content=source.get("content", ""),
                    score=hit.get("_score", 0),
                    metadata=source.get("metadata", {}),
                    source_url=source.get("source_url"),
                    title=source.get("title"),
                ))
            return results
        except Exception as e:
            logger.error(f"User index search failed: {e}")
            return []
    
    async def _search_citations_index(
        self,
        query: str,
        query_vector: List[float],
        user_id: str,
        top_k: int,
    ) -> List[SearchResult]:
        """Search user's citations index."""
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {"user_id": user_id}
                        }
                    ],
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "abstract", "content"],
                                "boost": 1.0
                            }
                        },
                        {
                            "knn": {
                                "vector": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": 2.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["paper_id", "title", "abstract", "content", "authors", "year", "venue", "doi", "citation_key"]
        }
        
        try:
            response = await self._make_request(
                "POST",
                f"{self.citations_index}/_search",
                search_query
            )
            
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append(SearchResult(
                    id=hit.get("_id"),
                    document_id=source.get("paper_id"),
                    user_id=user_id,
                    content=source.get("content", "") or source.get("abstract", ""),
                    score=hit.get("_score", 0),
                    metadata={
                        "authors": source.get("authors", []),
                        "year": source.get("year"),
                        "venue": source.get("venue"),
                        "doi": source.get("doi"),
                        "citation_key": source.get("citation_key"),
                    },
                    title=source.get("title"),
                ))
            return results
        except Exception as e:
            logger.error(f"Citations index search failed: {e}")
            return []
    
    async def get_user_documents(
        self,
        user_id: str,
        doc_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """Get list of user's documents with filtering."""
        must_conditions = [{"term": {"user_id": user_id}}]
        
        if doc_type:
            must_conditions.append({"term": {"doc_type": doc_type.value}})
        if status:
            must_conditions.append({"term": {"status": status.value}})
        
        # Aggregate by document_id to get unique documents
        search_query = {
            "size": 0,
            "query": {
                "bool": {"must": must_conditions}
            },
            "aggs": {
                "documents": {
                    "terms": {
                        "field": "document_id",
                        "size": limit
                    },
                    "aggs": {
                        "doc": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["document_id", "title", "metadata", "doc_type", "status", "created_at", "source_url"]
                            }
                        }
                    }
                }
            }
        }
        
        try:
            response = await self._make_request(
                "POST",
                f"{self.user_docs_index}/_search",
                search_query
            )
            
            documents = []
            for bucket in response.get("aggregations", {}).get("documents", {}).get("buckets", []):
                hit = bucket.get("doc", {}).get("hits", {}).get("hits", [])
                if hit:
                    source = hit[0].get("_source", {})
                    documents.append(Document(
                        id=source.get("document_id"),
                        user_id=user_id,
                        title=source.get("title", "Untitled"),
                        doc_type=DocumentType(source.get("doc_type", "txt")),
                        status=DocumentStatus(source.get("status", "indexed")),
                        metadata=source.get("metadata", {}),
                        source_url=source.get("source_url"),
                        created_at=datetime.fromisoformat(source["created_at"]) if source.get("created_at") else None,
                    ))
            
            return documents
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []
    
    async def delete_document(
        self,
        user_id: str,
        document_id: str,
    ) -> bool:
        """Delete a document and all its chunks."""
        delete_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {"term": {"document_id": document_id}}
                    ]
                }
            }
        }
        
        try:
            await self._make_request(
                "POST",
                f"{self.user_docs_index}/_delete_by_query",
                delete_query
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def search_user_documents(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search within user's documents (simple interface)."""
        # Generate query embedding
        query_vector = await self.generate_embedding(query)
        
        if query_vector:
            results = await self._search_user_index(
                query, query_vector, user_id, top_k
            )
            return [r.model_dump() for r in results]
        return []
