"""
Enhanced RAG Pipeline for Scientific/Research Workflows

This module provides a comprehensive RAG (Retrieval-Augmented Generation) pipeline
optimized for research and scientific workflows, featuring:

- Unified document management (user uploads + Semantic Scholar citations)
- LlamaIndex integration for advanced retrieval
- Smart citation extraction and formatting
- Multi-source context assembly
- User-isolated document storage
"""

from .document_manager import DocumentManager, Document, DocumentType
from .citation_engine import CitationEngine, Citation, CitationStyle
from .semantic_scholar import SemanticScholarClient, Paper
from .vector_store import VectorStoreManager
from .query_engine import QueryEngine, QueryResult
from .config import RAGConfig

__all__ = [
    "DocumentManager",
    "Document",
    "DocumentType",
    "CitationEngine", 
    "Citation",
    "CitationStyle",
    "SemanticScholarClient",
    "Paper",
    "VectorStoreManager",
    "QueryEngine",
    "QueryResult",
    "RAGConfig",
]

__version__ = "2.0.0"
