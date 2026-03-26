"""
Document Manager Module

Provides unified document management for user-uploaded documents
and Semantic Scholar citations with user isolation.
"""

import os
import re
import uuid
import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, BinaryIO, Union
from pathlib import Path

from pydantic import BaseModel, Field
import aiofiles

from .config import get_config

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max file size
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    URL = "url"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseModel):
    """
    Unified document model for both uploaded files and citations.
    
    Attributes:
        id: Unique document identifier
        user_id: Owner user ID for isolation
        title: Document title
        content: Extracted text content
        doc_type: Type of document
        source_url: Original URL (if applicable)
        file_path: Local storage path (if uploaded file)
        metadata: Additional metadata (authors, year, venue, etc.)
        status: Processing status
        created_at: Creation timestamp
        updated_at: Last update timestamp
        chunks: List of document chunks with embeddings
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    content: str = ""
    doc_type: DocumentType
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Citation-specific fields
    citation_key: Optional[str] = None  # For academic citations
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """A chunk of a document for vector storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    user_id: str
    content: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentManager:
    """
    Unified document management system.
    
    Handles:
    - Document upload and storage
    - Document processing and chunking
    - User isolation
    - Integration with vector store
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        vector_store: Optional["VectorStoreManager"] = None,
        config: Optional["RAGConfig"] = None,
    ):
        self.config = config or get_config()
        self.storage_path = Path(storage_path or os.getenv("DOCUMENT_STORAGE_PATH", "./documents"))
        self.vector_store = vector_store
        self._ensure_storage_dirs()
        
    def _ensure_storage_dirs(self) -> None:
        """Create necessary storage directories."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "uploads").mkdir(exist_ok=True)
        (self.storage_path / "temp").mkdir(exist_ok=True)
        
    def _get_user_dir(self, user_id: str) -> Path:
        """Get user-specific storage directory."""
        user_dir = self.storage_path / "uploads" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _compute_file_hash(self, file_content: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(file_content).hexdigest()
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Removes:
        - Path separators
        - Null bytes
        - Parent directory references
        - Special characters
        """
        # Remove any path components
        filename = os.path.basename(filename)
        
        # Remove null bytes
        filename = filename.replace("\x00", "")
        
        # Remove parent directory references
        filename = filename.replace("..", "")
        
        # Keep only alphanumeric, dots, dashes, underscores
        filename = re.sub(r'[^\w\-.]', '_', filename)
        
        # Ensure filename is not empty
        if not filename or filename.startswith('.'):
            filename = f"document_{uuid.uuid4().hex[:8]}"
            
        # Limit filename length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:190] + ext
            
        return filename
    
    def _validate_file(self, filename: str, file_size: int) -> None:
        """
        Validate file before processing.
        
        Raises:
            ValueError: If file validation fails
        """
        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed ({MAX_FILE_SIZE / 1024 / 1024} MB)")
        
        if file_size == 0:
            raise ValueError("Empty file uploaded")
        
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type '{ext}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    async def upload_document(
        self,
        user_id: str,
        file: BinaryIO,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Upload and process a document for a user.
        
        Args:
            user_id: User ID for document isolation
            file: Binary file object
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Document object with processing status
            
        Raises:
            ValueError: If file validation fails
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        
        # Determine document type from extension
        ext = Path(safe_filename).suffix.lower().lstrip(".")
        try:
            doc_type = DocumentType(ext)
        except ValueError:
            doc_type = DocumentType.TXT
            
        # Read file content
        content = file.read()
        file_size = len(content)
        
        # Validate file
        self._validate_file(safe_filename, file_size)
        
        file_hash = self._compute_file_hash(content)
        
        # Check for duplicates
        existing = await self._check_duplicate(user_id, file_hash)
        if existing:
            logger.info(f"Duplicate document found: {safe_filename} for user {user_id}")
            return existing
            
        # Save file
        user_dir = self._get_user_dir(user_id)
        storage_filename = f"{file_hash[:16]}_{safe_filename}"
        file_path = user_dir / storage_filename
        
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
            
        # Create document record
        doc = Document(
            user_id=user_id,
            title=metadata.get("title", safe_filename) if metadata else safe_filename,
            doc_type=doc_type,
            file_path=str(file_path),
            metadata={
                **(metadata or {}),
                "file_hash": file_hash,
                "original_filename": filename,
                "safe_filename": safe_filename,
                "file_size": file_size,
            },
        )
        
        # Extract content
        doc.content = await self._extract_content(file_path, doc_type)
        doc.status = DocumentStatus.PROCESSING
        
        # Process and index
        if self.vector_store:
            await self._process_and_index(doc)
            
        return doc
    
    async def _extract_content(
        self,
        file_path: Path,
        doc_type: DocumentType,
    ) -> str:
        """Extract text content from document."""
        content = ""
        
        try:
            if doc_type == DocumentType.PDF:
                content = await self._extract_pdf(file_path)
            elif doc_type == DocumentType.DOCX:
                content = await self._extract_docx(file_path)
            elif doc_type in [DocumentType.TXT, DocumentType.MARKDOWN]:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
            elif doc_type == DocumentType.HTML:
                content = await self._extract_html(file_path)
            else:
                logger.warning(f"Unsupported document type: {doc_type}")
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            
        return content
    
    async def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            import pypdf
            content_parts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
            return "\n\n".join(content_parts)
        except ImportError:
            logger.warning("pypdf not installed, falling back to PyPDF2")
            import PyPDF2
            content_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
            return "\n\n".join(content_parts)
    
    async def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            return "\n\n".join([para.text for para in doc.paragraphs if para.text])
        except ImportError:
            logger.error("python-docx not installed")
            return ""
    
    async def _extract_html(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            from bs4 import BeautifulSoup
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                html_content = await f.read()
            soup = BeautifulSoup(html_content, "lxml")
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            logger.error("beautifulsoup4 and lxml not installed")
            return ""
    
    async def _process_and_index(self, doc: Document) -> None:
        """Process document into chunks and index in vector store."""
        if not self.vector_store or not doc.content:
            return
            
        # Chunk document
        chunks = self._chunk_document(doc)
        doc.chunks = [chunk.model_dump() for chunk in chunks]
        
        # Generate embeddings and index
        await self.vector_store.index_document_chunks(doc.id, doc.user_id, chunks)
        
        doc.status = DocumentStatus.INDEXED
        doc.updated_at = datetime.utcnow()
        
    def _chunk_document(
        self,
        doc: Document,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """
        Split document into overlapping chunks for vector storage.
        
        Uses sentence-aware chunking for better context preservation.
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        content = doc.content
        if not content:
            return []
            
        chunks = []
        
        # Try to use LlamaIndex for smart chunking
        try:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core import Document as LlamaDocument
            
            parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            llama_doc = LlamaDocument(text=content, metadata=doc.metadata)
            nodes = parser.get_nodes_from_documents([llama_doc])
            
            for i, node in enumerate(nodes):
                chunks.append(DocumentChunk(
                    document_id=doc.id,
                    user_id=doc.user_id,
                    content=node.text,
                    chunk_index=i,
                    metadata={
                        **doc.metadata,
                        "title": doc.title,
                        "source_url": doc.source_url,
                    },
                ))
        except ImportError:
            # Fallback to simple chunking
            logger.warning("LlamaIndex not available, using simple chunking")
            start = 0
            chunk_idx = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_content = content[start:end]
                
                chunks.append(DocumentChunk(
                    document_id=doc.id,
                    user_id=doc.user_id,
                    content=chunk_content,
                    chunk_index=chunk_idx,
                    metadata={
                        **doc.metadata,
                        "title": doc.title,
                        "source_url": doc.source_url,
                    },
                ))
                
                start += chunk_size - overlap
                chunk_idx += 1
                
        return chunks
    
    async def _check_duplicate(
        self,
        user_id: str,
        file_hash: str,
    ) -> Optional[Document]:
        """Check if a document with the same hash already exists."""
        # This would query the vector store or database
        # For now, return None (no duplicate found)
        return None
    
    async def create_citation_document(
        self,
        user_id: str,
        paper_id: str,
        paper_data: Dict[str, Any],
    ) -> Document:
        """
        Create a document from a Semantic Scholar paper citation.
        
        Args:
            user_id: User ID
            paper_id: Semantic Scholar paper ID
            paper_data: Paper metadata from Semantic Scholar
            
        Returns:
            Document object for the citation
        """
        # Generate citation key
        authors_str = "_".join(
            [a.get("name", "").split()[-1] for a in paper_data.get("authors", [])[:2]]
        )
        year = paper_data.get("year", "n.d.")
        citation_key = f"{authors_str}_{year}"
        
        # Build content from paper metadata
        content_parts = []
        if paper_data.get("title"):
            content_parts.append(f"Title: {paper_data['title']}")
        if paper_data.get("abstract"):
            content_parts.append(f"Abstract: {paper_data['abstract']}")
        if paper_data.get("authors"):
            authors = [a.get("name", "") for a in paper_data["authors"]]
            content_parts.append(f"Authors: {', '.join(authors)}")
        if paper_data.get("year"):
            content_parts.append(f"Year: {paper_data['year']}")
        if paper_data.get("venue"):
            content_parts.append(f"Venue: {paper_data['venue']}")
        if paper_data.get("citationCount") is not None:
            content_parts.append(f"Citations: {paper_data['citationCount']}")
            
        content = "\n\n".join(content_parts)
        
        doc = Document(
            user_id=user_id,
            title=paper_data.get("title", "Untitled"),
            content=content,
            doc_type=DocumentType.SEMANTIC_SCHOLAR,
            source_url=paper_data.get("url"),
            metadata={
                "semantic_scholar_id": paper_id,
                "citation_count": paper_data.get("citationCount", 0),
                "reference_count": paper_data.get("referenceCount", 0),
                "publication_venue": paper_data.get("venue"),
                "publication_date": paper_data.get("publicationDate"),
            },
            citation_key=citation_key,
            authors=[a.get("name", "") for a in paper_data.get("authors", [])],
            year=paper_data.get("year"),
            venue=paper_data.get("venue"),
            doi=paper_data.get("externalIds", {}).get("DOI"),
            semantic_scholar_id=paper_id,
        )
        
        # Process and index
        if self.vector_store:
            await self._process_and_index(doc)
            
        return doc
    
    async def get_user_documents(
        self,
        user_id: str,
        doc_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """
        Retrieve documents for a user with optional filtering.
        
        Args:
            user_id: User ID
            doc_type: Filter by document type
            status: Filter by processing status
            limit: Maximum number of documents to return
            offset: Pagination offset
            
        Returns:
            List of Document objects
        """
        if self.vector_store:
            return await self.vector_store.get_user_documents(
                user_id=user_id,
                doc_type=doc_type,
                status=status,
                limit=limit,
                offset=offset,
            )
        return []
    
    async def delete_document(
        self,
        user_id: str,
        document_id: str,
    ) -> bool:
        """
        Delete a document and all associated chunks.
        
        Args:
            user_id: User ID (for authorization)
            document_id: Document ID to delete
            
        Returns:
            True if deletion was successful
        """
        if self.vector_store:
            return await self.vector_store.delete_document(user_id, document_id)
        return False
    
    async def search_user_documents(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search within user's documents.
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        if self.vector_store:
            return await self.vector_store.search_user_documents(
                user_id=user_id,
                query=query,
                top_k=top_k,
            )
        return []
