"""
Enhanced RAG API Endpoints

New API endpoints for the enhanced RAG pipeline with:
- Document upload and management
- Semantic Scholar citation import
- Multi-source hybrid search
- Citation-aware responses
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import (
    APIRouter, UploadFile, File, Form, HTTPException,
    Depends, BackgroundTasks, Query as QueryParam
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from openai import OpenAI

from .config import get_config, RAGConfig
from .document_manager import DocumentManager, Document, DocumentType, DocumentStatus
from .vector_store import VectorStoreManager
from .citation_engine import CitationEngine, Citation, CitationStyle
from .semantic_scholar import SemanticScholarClient, Paper
from .query_engine import QueryEngine, QuerySource, QueryResult

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

# Global instances (initialized on startup)
_config: Optional[RAGConfig] = None
_openai_client: Optional[OpenAI] = None
_vector_store: Optional[VectorStoreManager] = None
_document_manager: Optional[DocumentManager] = None
_semantic_scholar: Optional[SemanticScholarClient] = None
_query_engine: Optional[QueryEngine] = None
_citation_engine: Optional[CitationEngine] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        config = get_config()
        if not config.openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        _openai_client = OpenAI(api_key=config.openai.api_key)
    return _openai_client


def get_vector_store() -> VectorStoreManager:
    """Get or create vector store manager."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager(
            config=get_config(),
            openai_client=get_openai_client(),
        )
    return _vector_store


def get_document_manager() -> DocumentManager:
    """Get or create document manager."""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager(
            vector_store=get_vector_store(),
            config=get_config(),
        )
    return _document_manager


def get_semantic_scholar() -> SemanticScholarClient:
    """Get or create Semantic Scholar client."""
    global _semantic_scholar
    if _semantic_scholar is None:
        _semantic_scholar = SemanticScholarClient()
    return _semantic_scholar


def get_query_engine() -> QueryEngine:
    """Get or create query engine."""
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(
            openai_client=get_openai_client(),
            vector_store=get_vector_store(),
            document_manager=get_document_manager(),
            semantic_scholar=get_semantic_scholar(),
            config=get_config(),
        )
    return _query_engine


def get_citation_engine() -> CitationEngine:
    """Get or create citation engine."""
    global _citation_engine
    if _citation_engine is None:
        _citation_engine = CitationEngine()
    return _citation_engine


# ============ Request/Response Models ============

class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question or query")
    user_id: Optional[str] = Field(None, max_length=256, description="User ID for personalized results")
    sources: str = Field(default="all", pattern="^(all|general|user_docs|citations)$", description="Sources to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    citation_style: str = Field(default="apa", pattern="^(apa|mla|chicago|ieee|harvard|vancouver|bibtex)$", description="Citation format")
    language: str = Field(default="en", max_length=10, description="Response language code")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation turns")
    stream: bool = Field(default=False, description="Stream response tokens")


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    title: str
    status: str
    message: str


class PaperSearchResponse(BaseModel):
    """Response for paper search."""
    papers: List[Dict[str, Any]]
    total: int


class CitationImportRequest(BaseModel):
    """Request to import a citation."""
    paper_id: str
    user_id: str


class CitationImportResponse(BaseModel):
    """Response for citation import."""
    citation_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    citation_key: str


class DocumentListResponse(BaseModel):
    """Response for document listing."""
    documents: List[Dict[str, Any]]
    total: int


# ============ Health Check ============

@router.get("/health")
async def health_check():
    """Check RAG pipeline health."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "openai": _openai_client is not None,
            "vector_store": _vector_store is not None,
            "document_manager": _document_manager is not None,
            "semantic_scholar": _semantic_scholar is not None,
        }
    }


# ============ Query Endpoints ============

@router.post("/query")
async def query_rag(request: QueryRequest):
    """
    Execute a RAG query with multi-source retrieval.
    
    Searches across:
    - General knowledge base (existing documents)
    - User-uploaded documents
    - User's saved citations
    
    Returns response with sources and citations.
    
    Raises:
        400: Invalid request parameters
        500: Query processing error
    """
    try:
        engine = get_query_engine()
        
        # Map source string to enum
        source_map = {
            "all": QuerySource.ALL,
            "general": QuerySource.GENERAL,
            "user_docs": QuerySource.USER_DOCS,
            "citations": QuerySource.CITATIONS,
        }
        
        # Map citation style
        style_map = {
            "apa": CitationStyle.APA,
            "mla": CitationStyle.MLA,
            "chicago": CitationStyle.CHICAGO,
            "ieee": CitationStyle.IEEE,
            "harvard": CitationStyle.HARVARD,
            "vancouver": CitationStyle.VANCOUVER,
            "bibtex": CitationStyle.BIBTEX,
        }
        
        # Validate conversation history
        if request.conversation_history:
            # Limit history length
            request.conversation_history = request.conversation_history[-5:]
        
        result = await engine.query(
            user_query=request.query,
            user_id=request.user_id,
            sources=source_map[request.sources],
            top_k=request.top_k,
            citation_style=style_map[request.citation_style],
            language=request.language,
            conversation_history=request.conversation_history,
            stream=request.stream,
        )
        
        return result.to_dict()
        
    except ValueError as e:
        logger.warning(f"Query validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Stream response for a RAG query.
    
    Same as /query but streams the response tokens.
    """
    async def generate():
        engine = get_query_engine()
        async for chunk in engine.stream_query(
            user_query=request.query,
            user_id=request.user_id,
            language=request.language,
            conversation_history=request.conversation_history,
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


# ============ Document Management Endpoints ============

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    """
    Upload a document for a user.
    
    Supports: PDF, DOCX, TXT, MD, HTML
    
    Maximum file size: 50 MB
    
    The document will be:
    1. Stored in user-specific directory
    2. Processed and chunked
    3. Indexed in vector store for retrieval
    
    Raises:
        400: Invalid file type or file too large
        500: Processing error
    """
    try:
        # Validate user_id
        if not user_id or len(user_id) > 256:
            raise HTTPException(status_code=400, detail="Invalid user_id")
        
        doc_manager = get_document_manager()
        
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            import json
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        if title:
            # Sanitize title
            title = title[:500]  # Limit title length
            meta_dict["title"] = title
        
        # Upload document
        doc = await doc_manager.upload_document(
            user_id=user_id,
            file=file.file,
            filename=file.filename or "document.txt",
            metadata=meta_dict,
        )
        
        return DocumentUploadResponse(
            document_id=doc.id,
            title=doc.title,
            status=doc.status,
            message=f"Document uploaded and {'processed' if doc.status == 'indexed' else 'processing'}",
        )
        
    except ValueError as e:
        logger.warning(f"Document upload validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents/{user_id}", response_model=DocumentListResponse)
async def list_user_documents(
    user_id: str,
    doc_type: Optional[str] = QueryParam(None),
    limit: int = QueryParam(50, ge=1, le=200),
    offset: int = QueryParam(0, ge=0),
):
    """
    List documents for a user.
    
    Optionally filter by document type.
    """
    try:
        doc_manager = get_document_manager()
        
        type_filter = None
        if doc_type:
            try:
                type_filter = DocumentType(doc_type)
            except ValueError:
                pass
        
        documents = await doc_manager.get_user_documents(
            user_id=user_id,
            doc_type=type_filter,
            limit=limit,
            offset=offset,
        )
        
        return DocumentListResponse(
            documents=[d.model_dump() for d in documents],
            total=len(documents),
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{user_id}/{document_id}")
async def delete_document(user_id: str, document_id: str):
    """
    Delete a document and all its chunks.
    
    Only the owner can delete their documents.
    """
    try:
        doc_manager = get_document_manager()
        success = await doc_manager.delete_document(user_id, document_id)
        
        if success:
            return {"status": "deleted", "document_id": document_id}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/search/{user_id}")
async def search_user_documents(
    user_id: str,
    query: str = QueryParam(..., min_length=1),
    top_k: int = QueryParam(5, ge=1, le=20),
):
    """
    Search within a user's documents.
    
    Uses hybrid search (BM25 + vector) for best results.
    """
    try:
        doc_manager = get_document_manager()
        results = await doc_manager.search_user_documents(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
        
        return {
            "query": query,
            "results": results,
            "total": len(results),
        }
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Semantic Scholar Endpoints ============

@router.get("/papers/search", response_model=PaperSearchResponse)
async def search_papers(
    query: str = QueryParam(..., min_length=1),
    limit: int = QueryParam(10, ge=1, le=50),
    year_start: Optional[int] = QueryParam(None),
    year_end: Optional[int] = QueryParam(None),
    venue: Optional[str] = QueryParam(None),
    fields_of_study: Optional[str] = QueryParam(None),
):
    """
    Search for academic papers via Semantic Scholar.
    
    Useful for finding relevant citations and references.
    """
    try:
        engine = get_query_engine()
        
        year_range = None
        if year_start and year_end:
            year_range = (year_start, year_end)
        
        fields = fields_of_study.split(",") if fields_of_study else None
        
        papers = await engine.search_papers(
            query=query,
            limit=limit,
            year_range=year_range,
            venue=venue,
            fields_of_study=fields,
        )
        
        return PaperSearchResponse(
            papers=[p.to_dict() for p in papers],
            total=len(papers),
        )
        
    except Exception as e:
        logger.error(f"Paper search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """
    Get detailed information about a specific paper.
    
    Uses Semantic Scholar paper ID or DOI.
    """
    try:
        s2_client = get_semantic_scholar()
        paper = await s2_client.get_paper(paper_id)
        
        if paper:
            return paper.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Paper not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/citations/import", response_model=CitationImportResponse)
async def import_citation(request: CitationImportRequest):
    """
    Import a paper as a citation for a user.
    
    The citation will be:
    1. Saved to user's citation library
    2. Indexed for retrieval in RAG queries
    """
    try:
        engine = get_query_engine()
        citation = await engine.import_citation(
            user_id=request.user_id,
            paper_id=request.paper_id,
        )
        
        if citation:
            return CitationImportResponse(
                citation_id=citation.id,
                title=citation.title,
                authors=citation.authors,
                year=citation.year,
                citation_key=citation.citation_key,
            )
        else:
            raise HTTPException(status_code=404, detail="Paper not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/citations/{user_id}")
async def list_user_citations(
    user_id: str,
    limit: int = QueryParam(50, ge=1, le=200),
    offset: int = QueryParam(0, ge=0),
):
    """
    List citations saved by a user.
    """
    try:
        doc_manager = get_document_manager()
        documents = await doc_manager.get_user_documents(
            user_id=user_id,
            doc_type=DocumentType.SEMANTIC_SCHOLAR,
            limit=limit,
            offset=offset,
        )
        
        return {
            "citations": [d.model_dump() for d in documents],
            "total": len(documents),
        }
        
    except Exception as e:
        logger.error(f"Failed to list citations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Citation Formatting Endpoints ============

@router.post("/citations/format")
async def format_citation(
    paper_id: str = QueryParam(...),
    style: str = QueryParam("apa", pattern="^(apa|mla|chicago|ieee|harvard|vancouver|bibtex)$"),
):
    """
    Get a formatted citation for a paper.
    
    Supports multiple citation styles.
    """
    try:
        s2_client = get_semantic_scholar()
        citation_engine = get_citation_engine()
        
        paper = await s2_client.get_paper(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        citation = Citation.from_paper(paper)
        
        style_map = {
            "apa": CitationStyle.APA,
            "mla": CitationStyle.MLA,
            "chicago": CitationStyle.CHICAGO,
            "ieee": CitationStyle.IEEE,
            "harvard": CitationStyle.HARVARD,
            "vancouver": CitationStyle.VANCOUVER,
            "bibtex": CitationStyle.BIBTEX,
        }
        
        formatted = citation_engine.format_citation(
            citation,
            style=style_map[style],
        )
        
        return {
            "paper_id": paper_id,
            "style": style,
            "citation": formatted,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation formatting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Index Management ============

@router.post("/admin/init-indices")
async def initialize_indices():
    """
    Initialize OpenSearch indices for RAG pipeline.
    
    Creates:
    - User documents index
    - Citations index
    """
    try:
        vector_store = get_vector_store()
        await vector_store.ensure_indices()
        
        return {
            "status": "success",
            "message": "Indices initialized",
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
