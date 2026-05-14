"""
FastAPI router for the scientific RAG pipeline.
Mount this on the main app with: app.include_router(rag_router, prefix="/rag")
"""
import os
import logging
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import ScientificRAGPipeline

logger = logging.getLogger(__name__)

rag_router = APIRouter(prefix="/rag", tags=["Scientific RAG"])

# ---------------------------------------------------------------------------
# Singleton pipeline (initialised once at import time)
# ---------------------------------------------------------------------------

_pipeline: Optional[ScientificRAGPipeline] = None


def get_pipeline() -> ScientificRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ScientificRAGPipeline(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            citation_style=os.getenv("RAG_CITATION_STYLE", "apa"),
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    citation_style: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    context_chunks: int


class PaperSearchRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    limit: int = 5


class PaperIngestRequest(BaseModel):
    paper_id: str
    user_id: str = "anonymous"


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@rag_router.post("/query", response_model=QueryResponse, summary="Query the RAG pipeline")
async def query(
    req: QueryRequest,
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    """
    Answer a scientific/research question using hybrid retrieval
    over uploaded documents and indexed Semantic Scholar papers.
    """
    result = pipeline.query(
        question=req.question,
        user_id=req.user_id,
        citation_style=req.citation_style,
        stream=False,
    )
    return QueryResponse(**result)


@rag_router.post("/query/stream", summary="Streaming RAG query (SSE)")
async def query_stream(
    req: QueryRequest,
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    """Server-Sent Events streaming endpoint."""
    return StreamingResponse(
        pipeline.query_stream(
            question=req.question,
            user_id=req.user_id,
            citation_style=req.citation_style,
        ),
        media_type="text/event-stream",
    )


@rag_router.post("/documents/upload", summary="Upload a document for RAG indexing")
async def upload_document(
    user_id: str = Query(default="anonymous"),
    file: UploadFile = File(...),
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    """
    Upload PDF, DOCX, TXT, MD, or HTML. The file is chunked and indexed.
    Returns the document ID and chunk count.
    """
    suffix = os.path.splitext(file.filename or "")[1].lower()
    allowed = {".pdf", ".docx", ".txt", ".md", ".html"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {allowed}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest_file(tmp_path, user_id=user_id)
    except Exception as exc:
        logger.error("Document ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


@rag_router.get("/documents", summary="List indexed documents")
async def list_documents(
    user_id: Optional[str] = Query(default=None),
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    docs = pipeline.doc_manager.list_documents(user_id=user_id)
    return [
        DocumentListItem(
            doc_id=d.doc_id,
            filename=d.filename,
            file_type=d.file_type,
            chunks=len(d.chunks),
        )
        for d in docs
    ]


@rag_router.delete("/documents/{doc_id}", summary="Remove a document from the index")
async def delete_document(
    doc_id: str,
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    deleted = pipeline.doc_manager.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "doc_id": doc_id}


@rag_router.post("/papers/search", summary="Search Semantic Scholar and auto-index results")
async def search_papers(
    req: PaperSearchRequest,
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    """Search academic papers and add them to the retrieval index."""
    results = pipeline.search_and_index(
        query=req.query, user_id=req.user_id, limit=req.limit
    )
    return {"papers": results, "count": len(results)}


@rag_router.post("/papers/ingest", summary="Ingest a specific paper by Semantic Scholar ID")
async def ingest_paper(
    req: PaperIngestRequest,
    pipeline: ScientificRAGPipeline = Depends(get_pipeline),
):
    result = pipeline.ingest_paper(paper_id=req.paper_id, user_id=req.user_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Paper '{req.paper_id}' not found on Semantic Scholar."
        )
    return result
