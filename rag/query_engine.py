"""
Query Engine Module

Core RAG query engine that orchestrates retrieval, context assembly,
and response generation with proper citation support.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from .config import get_config
from .document_manager import DocumentManager
from .vector_store import VectorStoreManager, SearchResult
from .citation_engine import CitationEngine, Citation, CitationStyle
from .semantic_scholar import SemanticScholarClient, Paper

logger = logging.getLogger(__name__)


class QuerySource(str, Enum):
    """Sources to include in query."""
    GENERAL = "general"
    USER_DOCS = "user_docs"
    CITATIONS = "citations"
    ALL = "all"


@dataclass
class QueryResult:
    """
    Result of a RAG query.
    
    Contains the response, sources, and citations.
    """
    query: str
    response: str
    sources: List[Dict[str, Any]]
    citations: List[Citation]
    context_used: str
    retrieval_time: float
    generation_time: float
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query": self.query,
            "response": self.response,
            "sources": [
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "url": s.get("source_url"),
                    "score": s.get("score"),
                    "type": s.get("source_type"),
                }
                for s in self.sources
            ],
            "citations": [
                {
                    "key": c.citation_key,
                    "title": c.title,
                    "authors": c.authors,
                    "year": c.year,
                }
                for c in self.citations
            ],
            "timing": {
                "retrieval_ms": self.retrieval_time * 1000,
                "generation_ms": self.generation_time * 1000,
                "total_ms": self.total_time * 1000,
            },
        }


class QueryEngine:
    """
    Main RAG query engine.
    
    Features:
    - Multi-source retrieval (general docs, user docs, citations)
    - Hybrid search (BM25 + vector)
    - Context assembly with source tracking
    - Response generation with citation support
    - Query rewriting and expansion
    - Streaming responses
    """
    
    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        vector_store: Optional[VectorStoreManager] = None,
        document_manager: Optional[DocumentManager] = None,
        semantic_scholar: Optional[SemanticScholarClient] = None,
        config: Optional["RAGConfig"] = None,
    ):
        self.config = config or get_config()
        self.openai_client = openai_client
        self.vector_store = vector_store
        self.document_manager = document_manager
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.citation_engine = CitationEngine()
        
    async def query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        sources: QuerySource = QuerySource.ALL,
        top_k: int = 5,
        citation_style: CitationStyle = CitationStyle.APA,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> QueryResult:
        """
        Execute a RAG query.
        
        Args:
            user_query: User's question
            user_id: User ID for personalized results
            sources: Which sources to search
            top_k: Number of documents to retrieve per source
            citation_style: Citation style for references
            language: Response language
            conversation_history: Previous conversation turns
            stream: Whether to stream the response
            
        Returns:
            QueryResult with response and citations
        """
        start_time = time.perf_counter()
        
        # 1. Query rewriting
        rewritten_query = await self._rewrite_query(user_query, conversation_history)
        logger.info(f"Rewritten query: {rewritten_query}")
        
        # 2. Generate query embedding
        query_vector = await self.vector_store.generate_embedding(rewritten_query)
        
        # 3. Determine search sources
        search_sources = self._get_search_sources(sources)
        
        # 4. Hybrid search
        retrieval_start = time.perf_counter()
        search_results = await self.vector_store.hybrid_search(
            query=rewritten_query,
            query_vector=query_vector,
            user_id=user_id,
            search_sources=search_sources,
            top_k=top_k,
        )
        retrieval_time = time.perf_counter() - retrieval_start
        
        # 5. Assemble context with source tracking
        context, citations = self._assemble_context(search_results, citation_style)
        
        # 6. Generate response
        generation_start = time.perf_counter()
        if stream:
            response = await self._generate_response_stream(
                user_query, rewritten_query, context, language, conversation_history
            )
        else:
            response = await self._generate_response(
                user_query, rewritten_query, context, language, conversation_history
            )
        generation_time = time.perf_counter() - generation_start
        
        total_time = time.perf_counter() - start_time
        
        return QueryResult(
            query=user_query,
            response=response,
            sources=[s.model_dump() for s in search_results],
            citations=citations,
            context_used=context[:500] + "..." if len(context) > 500 else context,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
        )
    
    async def stream_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        sources: QuerySource = QuerySource.ALL,
        top_k: int = 5,
        citation_style: CitationStyle = CitationStyle.APA,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response for a RAG query.
        
        Yields response chunks as they're generated.
        """
        # 1. Query rewriting
        rewritten_query = await self._rewrite_query(user_query, conversation_history)
        
        # 2. Generate query embedding
        query_vector = await self.vector_store.generate_embedding(rewritten_query)
        
        # 3. Hybrid search
        search_sources = self._get_search_sources(sources)
        search_results = await self.vector_store.hybrid_search(
            query=rewritten_query,
            query_vector=query_vector,
            user_id=user_id,
            search_sources=search_sources,
            top_k=top_k,
        )
        
        # 4. Assemble context
        context, citations = self._assemble_context(search_results, citation_style)
        
        # 5. Stream response
        async for chunk in self._stream_response(
            user_query, rewritten_query, context, language, conversation_history
        ):
            yield chunk
    
    def _get_search_sources(self, source: QuerySource) -> List[str]:
        """Convert QuerySource enum to list of source names."""
        if source == QuerySource.ALL:
            return ["general", "user_docs", "citations"]
        elif source == QuerySource.GENERAL:
            return ["general"]
        elif source == QuerySource.USER_DOCS:
            return ["user_docs"]
        elif source == QuerySource.CITATIONS:
            return ["citations"]
        return ["general", "user_docs", "citations"]
    
    async def _rewrite_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Rewrite query for better retrieval.
        
        Expands abbreviations, resolves coreferences from conversation history.
        """
        if not history or len(history) == 0:
            return query
        
        # Simple query expansion based on recent context
        recent_queries = [h.get("query", "") for h in history[-2:]]
        
        rewrite_prompt = f"""Given the conversation history and current query, rewrite the query to be self-contained and optimized for document retrieval.

Conversation history:
{chr(10).join(recent_queries)}

Current query: {query}

Rewritten query (keep it concise, only output the rewritten query):"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.openai.chat_model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    def _assemble_context(
        self,
        search_results: List[SearchResult],
        citation_style: CitationStyle,
    ) -> tuple:
        """
        Assemble context from search results with proper source tracking.
        
        Returns:
            Tuple of (context_string, list_of_citations)
        """
        context_parts = []
        citations = []
        
        for idx, result in enumerate(search_results):
            # Build context entry
            entry_lines = [f"BEGIN OF CONTEXT {idx + 1}"]
            
            # Add source information
            if result.source_url:
                entry_lines.append(f"[URL]\n{result.source_url}")
            
            if result.title:
                entry_lines.append(f"[TITLE]\n{result.title}")
            
            # Add content
            entry_lines.append(f"[TEXT]\n{result.content}")
            
            # Add metadata
            if result.metadata:
                if result.metadata.get("authors"):
                    entry_lines.append(f"[AUTHORS]\n{', '.join(result.metadata['authors'])}")
                if result.metadata.get("year"):
                    entry_lines.append(f"[YEAR]\n{result.metadata['year']}")
                if result.metadata.get("venue"):
                    entry_lines.append(f"[VENUE]\n{result.metadata['venue']}")
            
            entry_lines.append(f"END OF CONTEXT {idx + 1}")
            
            context_parts.append("\n".join(entry_lines))
            
            # Create citation for this source
            citation = Citation(
                id=result.id,
                title=result.title or "Unknown",
                authors=result.metadata.get("authors", []),
                year=result.metadata.get("year"),
                venue=result.metadata.get("venue"),
                doi=result.metadata.get("doi"),
                url=result.source_url,
                source_type=result.source_type,
                document_id=result.document_id,
            )
            citations.append(citation)
        
        context = "\n\n--------------------\n\n".join(context_parts)
        return context, citations
    
    async def _generate_response(
        self,
        original_query: str,
        rewritten_query: str,
        context: str,
        language: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response using OpenAI."""
        system_prompt = self._build_system_prompt(language)
        
        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            for h in history[-3:]:
                messages.append({"role": "user", "content": h.get("query", "")})
                messages.append({"role": "assistant", "content": h.get("response", "")})
        
        # Add current query with context
        user_message = f"""Background information:
{context}

CURRENT QUERY:
Original user query: {original_query}

Respond in {language}. If the query is about academic or research topics, include relevant citations from the context using the format: "According to [Title] (Author, Year), ..." or similar."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.openai.chat_model,
                messages=messages,
                max_tokens=self.config.openai.max_tokens,
                temperature=self.config.openai.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error generating the response."
    
    async def _generate_response_stream(
        self,
        original_query: str,
        rewritten_query: str,
        context: str,
        language: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response (non-streaming wrapper)."""
        return await self._generate_response(
            original_query, rewritten_query, context, language, history
        )
    
    async def _stream_response(
        self,
        original_query: str,
        rewritten_query: str,
        context: str,
        language: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response using OpenAI."""
        system_prompt = self._build_system_prompt(language)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            for h in history[-3:]:
                messages.append({"role": "user", "content": h.get("query", "")})
                messages.append({"role": "assistant", "content": h.get("response", "")})
        
        user_message = f"""Background information:
{context}

CURRENT QUERY:
Original user query: {original_query}

Respond in {language}. Include relevant citations from the context."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = self.openai_client.chat.completions.create(
                model=self.config.openai.chat_model,
                messages=messages,
                max_tokens=self.config.openai.max_tokens,
                temperature=self.config.openai.temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield "I apologize, but I encountered an error generating the response."
    
    def _build_system_prompt(self, language: str) -> str:
        """Build the system prompt for response generation."""
        return f"""You are a highly reliable and knowledgeable research assistant. Your responses must strictly adhere to the provided documents and ensure factual accuracy through structured reasoning.

### RESPONSE GUIDELINES:
1. **Grounded in Evidence**:
   - All responses must be based only on facts explicitly present in the provided context.
   - Include source citations for all claims.
   - Never include unsupported or speculative details.

2. **Citation Format**:
   - Use inline citations: (Author, Year) or "According to [Title]..."
   - Always reference the source URL when available.
   - Format: "출처: [url]" for Korean, "Source: [url]" for English.

3. **Clarity and Relevance**:
   - Provide concise and user-friendly answers tailored to the query.
   - Focus solely on the user's query without unnecessary details.

4. **Unavailable Information**:
   - If the context lacks relevant information, clearly state:
     - Korean: "죄송합니다. 제공된 문서에서 관련 정보를 찾을 수 없습니다."
     - English: "I apologize. I couldn't find relevant information in the provided documents."

### RESPONSE LANGUAGE:
- Respond in {language}.
- Use appropriate date formats for the language."""
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[tuple] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Search for academic papers via Semantic Scholar.
        
        Args:
            query: Search query
            limit: Maximum results
            year_range: (start_year, end_year)
            venue: Publication venue filter
            fields_of_study: Research fields filter
            
        Returns:
            List of Paper objects
        """
        return await self.semantic_scholar.search_papers(
            query=query,
            limit=limit,
            year_range=year_range,
            venue=venue,
            fields_of_study=fields_of_study,
        )
    
    async def import_citation(
        self,
        user_id: str,
        paper_id: str,
    ) -> Optional[Citation]:
        """
        Import a paper as a citation for a user.
        
        Args:
            user_id: User ID
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Citation object if successful
        """
        paper = await self.semantic_scholar.get_paper(paper_id)
        
        if not paper:
            return None
        
        # Create citation document
        if self.document_manager:
            doc = await self.document_manager.create_citation_document(
                user_id=user_id,
                paper_id=paper_id,
                paper_data=paper.to_dict(),
            )
            
            # Index in vector store
            if self.vector_store:
                await self.vector_store.index_citation(
                    user_id=user_id,
                    citation_data=paper.to_dict(),
                    content=paper.abstract or paper.title,
                )
        
        return Citation.from_paper(paper, user_id)
