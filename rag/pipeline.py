"""
Main entry point: ScientificRAGPipeline.

Ties together:
- DocumentManager  – user-uploaded files
- SemanticScholarClient – academic paper search
- HybridRetriever – BM25 + vector retrieval
- CitationManager – citation formatting
- OpenAI – answer generation

Usage::

    from rag import ScientificRAGPipeline
    pipeline = ScientificRAGPipeline(openai_api_key="...")
    answer = pipeline.query("What is BERT?", user_id="alice")
"""
import logging
import os
from typing import List, Optional, Dict, Any, Generator

from dotenv import load_dotenv

from .document_manager import DocumentManager
from .semantic_scholar import SemanticScholarClient
from .citation_manager import CitationManager
from .hybrid_retriever import HybridRetriever

load_dotenv()
logger = logging.getLogger(__name__)


class ScientificRAGPipeline:
    """
    Scientific/research-optimised RAG pipeline.

    Parameters
    ----------
    openai_api_key: str
        OpenAI API key for embeddings + generation. Falls back to env var.
    semantic_scholar_api_key: str, optional
        S2 key for higher rate limits.
    model: str
        Chat completion model.
    embed_model: str
        Embedding model for dense retrieval.
    top_k: int
        Number of chunks to retrieve per query.
    alpha: float
        BM25/vector blending (0 = BM25 only, 1 = vector only).
    citation_style: str
        Default citation format.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 5,
        alpha: float = 0.6,
        citation_style: str = "apa",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        from openai import OpenAI  # lazy import

        self._openai_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=self._openai_key)
        self._model = model
        self._embed_model = embed_model
        self.top_k = top_k
        self.citation_style = citation_style

        self.doc_manager = DocumentManager(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.s2_client = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.citation_manager = CitationManager()
        self.retriever = HybridRetriever(
            embed_fn=self._embed_texts,
            alpha=alpha,
        )

        # Cache: track which paper IDs we already indexed
        self._indexed_paper_ids: set = set()

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        user_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Parse and index a local file. Returns doc summary."""
        doc = self.doc_manager.add_document(
            file_path=file_path,
            user_id=user_id,
            extra_metadata=metadata,
        )
        texts = [c.text for c in doc.chunks]
        metas = [
            {**c.metadata, "doc_id": doc.doc_id, "user_id": user_id, "source": "user_doc"}
            for c in doc.chunks
        ]
        self.retriever.add_texts(texts, metas)
        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "chunks": len(doc.chunks),
        }

    def ingest_paper(
        self,
        paper_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a Semantic Scholar paper by ID and index its abstract + metadata.
        Returns paper summary or None if not found.
        """
        if paper_id in self._indexed_paper_ids:
            logger.info("Paper '%s' already indexed.", paper_id)
            return {"paper_id": paper_id, "status": "already_indexed"}

        paper = self.s2_client.get_paper(paper_id)
        if paper is None:
            return None

        text = SemanticScholarClient.paper_to_text(paper)
        meta = {
            "paper_id": paper_id,
            "source": "semantic_scholar",
            "user_id": user_id,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "url": paper.get("url"),
        }
        self.retriever.add_texts([text], [meta])
        self._indexed_paper_ids.add(paper_id)

        citation = self.citation_manager.format(paper, self.citation_style)
        return {
            "paper_id": paper_id,
            "title": paper.get("title"),
            "citation": citation,
        }

    # ------------------------------------------------------------------
    # Search Semantic Scholar + auto-index
    # ------------------------------------------------------------------

    def search_and_index(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar, auto-index results, return citation list.
        """
        papers = self.s2_client.search(query, limit=limit)
        results = []
        for paper in papers:
            pid = paper.get("paperId")
            if not pid:
                continue
            if pid not in self._indexed_paper_ids:
                text = SemanticScholarClient.paper_to_text(paper)
                meta = {
                    "paper_id": pid,
                    "source": "semantic_scholar",
                    "user_id": user_id,
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                }
                self.retriever.add_texts([text], [meta])
                self._indexed_paper_ids.add(pid)

            citation = self.citation_manager.format(paper, self.citation_style)
            results.append({"paper_id": pid, "title": paper.get("title"), "citation": citation})

        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        user_id: str,
        citation_style: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Answer a scientific question using retrieved context.
        Returns a dict with 'answer' and 'citations', or a generator if stream=True.
        """
        style = citation_style or self.citation_style
        chunks = self.retriever.retrieve(question, top_k=self.top_k, user_id=user_id)

        if not chunks:
            # Fall back: search Semantic Scholar on the fly
            logger.info("No local chunks found. Searching Semantic Scholar for '%s'.", question)
            self.search_and_index(question, user_id=user_id, limit=5)
            chunks = self.retriever.retrieve(question, top_k=self.top_k, user_id=user_id)

        context = self._build_context(chunks)
        citations = self._extract_citations(chunks, style)

        if stream:
            return self._stream_answer(question, context, citations)

        answer = self._generate_answer(question, context)
        return {
            "answer": answer,
            "citations": citations,
            "context_chunks": len(chunks),
        }

    def query_stream(
        self,
        question: str,
        user_id: str,
        citation_style: Optional[str] = None,
    ) -> Generator:
        """SSE-friendly streaming generator."""
        yield from self.query(
            question, user_id=user_id, citation_style=citation_style, stream=True
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self._client.embeddings.create(
                model=self._embed_model, input=texts
            )
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning("OpenAI embedding error: %s", exc)
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get("metadata", {}).get("source", "")
            title = chunk.get("metadata", {}).get("title") or chunk.get("metadata", {}).get("source_file", "")
            header = f"[{i}] Source: {title or src}"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _extract_citations(
        self, chunks: List[Dict[str, Any]], style: str
    ) -> List[str]:
        seen: set = set()
        citations: List[str] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if meta.get("source") == "semantic_scholar":
                pid = meta.get("paper_id", "")
                if pid in seen:
                    continue
                seen.add(pid)
                paper = self.s2_client.get_paper(pid)
                if paper:
                    citations.append(self.citation_manager.format(paper, style))
            else:
                ref = meta.get("source_file") or meta.get("doc_id", "")
                if ref and ref not in seen:
                    seen.add(ref)
                    citations.append(f"User document: {ref}")
        return citations

    def _generate_answer(self, question: str, context: str) -> str:
        system_prompt = (
            "You are a rigorous scientific research assistant. "
            "Answer questions strictly based on the provided context. "
            "Always cite the numbered source blocks [1], [2], ... where relevant. "
            "If the context does not contain enough information, say so clearly."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a precise, well-structured answer with citations."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI completion error: %s", exc)
            return f"Error generating answer: {exc}"

    def _stream_answer(
        self, question: str, context: str, citations: List[str]
    ) -> Generator:
        import json

        system_prompt = (
            "You are a rigorous scientific research assistant. "
            "Answer questions strictly based on the provided context. "
            "Cite numbered source blocks [1], [2], ... where relevant."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a precise, well-structured answer with citations."
        )
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'MESSAGE', 'content': delta})}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: {json.dumps({'type': 'ERROR', 'content': str(exc)})}\n\n"

        # Send citations at the end
        if citations:
            yield f"data: {json.dumps({'type': 'CITATIONS', 'content': citations})}\n\n"
        yield "data: [DONE]\n\n"
