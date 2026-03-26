"""
Tests for RAG Pipeline

Unit and integration tests for the enhanced RAG pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import tempfile
import os

# Test the imports
from rag.config import RAGConfig, get_config, OpenSearchConfig, OpenAIConfig, SemanticScholarConfig, RedisConfig
from rag.document_manager import Document, DocumentType, DocumentStatus, DocumentChunk, DocumentManager
from rag.citation_engine import Citation, CitationStyle, CitationEngine
from rag.semantic_scholar import Paper, Author, SemanticScholarClient
from rag.vector_store import SearchResult, VectorStoreManager
from rag.query_engine import QueryEngine, QuerySource, QueryResult


# ============ Fixtures ============

@pytest.fixture
def mock_config():
    """Create a mock RAG configuration."""
    return RAGConfig(
        opensearch=OpenSearchConfig(
            endpoint="http://localhost:9200",
            user="admin",
            password="admin",
            general_index="test_general",
            user_docs_index="test_user_docs",
            citations_index="test_citations",
            embedding_dimension=1536,
            default_top_k=5,
        ),
        openai=OpenAIConfig(
            api_key="test-key",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.1,
        ),
        semantic_scholar=SemanticScholarConfig(api_key="test-s2-key"),
        redis=RedisConfig(enabled=False),
        chunk_size=512,
        chunk_overlap=50,
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    client.embeddings = Mock()
    client.embeddings.create = Mock(return_value=Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    ))
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    ))
    return client


@pytest.fixture
def sample_paper_data():
    """Sample Semantic Scholar paper data."""
    return {
        "paperId": "test-paper-123",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract for the paper.",
        "authors": [
            {"authorId": "author-1", "name": "John Smith"},
            {"authorId": "author-2", "name": "Jane Doe"},
        ],
        "year": 2023,
        "venue": "Test Conference",
        "citationCount": 42,
        "referenceCount": 15,
        "url": "https://example.com/paper",
        "externalIds": {"DOI": "10.1234/test.doi"},
        "openAccessPdf": {"url": "https://example.com/pdf"},
    }


@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(
        id="doc-123",
        user_id="user-456",
        title="Test Document",
        content="This is test content for the document. " * 100,
        doc_type=DocumentType.PDF,
        status=DocumentStatus.INDEXED,
        authors=["Test Author"],
        year=2023,
    )


# ============ Config Tests ============

class TestRAGConfig:
    """Tests for RAG configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = RAGConfig()
        
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.opensearch.embedding_dimension == 1536
        
    def test_config_from_env(self):
        """Test configuration from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = RAGConfig.from_env()
            assert config.openai.api_key == "test-key"


# ============ Document Manager Tests ============

class TestDocument:
    """Tests for Document model."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            user_id="user-1",
            title="Test Doc",
            doc_type=DocumentType.PDF,
        )
        
        assert doc.user_id == "user-1"
        assert doc.title == "Test Doc"
        assert doc.doc_type == DocumentType.PDF
        assert doc.status == DocumentStatus.PENDING
        assert isinstance(doc.created_at, datetime)
        
    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            user_id="user-1",
            title="Test",
            doc_type=DocumentType.TXT,
            metadata={"custom": "value"},
        )
        
        assert doc.metadata["custom"] == "value"


class TestDocumentChunk:
    """Tests for DocumentChunk."""
    
    def test_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            document_id="doc-1",
            user_id="user-1",
            content="Test content",
            chunk_index=0,
        )
        
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None


class TestDocumentManager:
    """Tests for DocumentManager."""
    
    @pytest.mark.asyncio
    async def test_chunk_document(self, mock_config, sample_document):
        """Test document chunking."""
        manager = DocumentManager(config=mock_config)
        chunks = manager._chunk_document(sample_document, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(c.document_id == sample_document.id for c in chunks)
        assert all(c.user_id == sample_document.user_id for c in chunks)
        
    def test_compute_file_hash(self, mock_config):
        """Test file hash computation."""
        manager = DocumentManager(config=mock_config)
        content = b"test content"
        hash_result = manager._compute_file_hash(content)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex length
    
    def test_sanitize_filename(self, mock_config):
        """Test filename sanitization for security."""
        manager = DocumentManager(config=mock_config)
        
        # Test path traversal attempts
        assert ".." not in manager._sanitize_filename("../../../etc/passwd")
        assert "/" not in manager._sanitize_filename("path/to/file.pdf")
        assert "\\" not in manager._sanitize_filename("C:\\Windows\\System32\\config")
        
        # Test null bytes
        assert "\x00" not in manager._sanitize_filename("file\x00.pdf")
        
        # Test special characters
        assert manager._sanitize_filename("file<script>.pdf") == "file_script_.pdf"
        
        # Test empty filename
        assert manager._sanitize_filename("") != ""
        assert manager._sanitize_filename(".pdf") != ""
    
    def test_validate_file_size(self, mock_config):
        """Test file size validation."""
        from rag.document_manager import MAX_FILE_SIZE
        
        manager = DocumentManager(config=mock_config)
        
        # Should pass for small file
        manager._validate_file("test.pdf", 1024)  # 1KB
        
        # Should raise for too large file
        with pytest.raises(ValueError, match="exceeds maximum"):
            manager._validate_file("test.pdf", MAX_FILE_SIZE + 1)
        
        # Should raise for empty file
        with pytest.raises(ValueError, match="Empty file"):
            manager._validate_file("test.pdf", 0)
    
    def test_validate_file_type(self, mock_config):
        """Test file type validation."""
        from rag.document_manager import ALLOWED_EXTENSIONS
        
        manager = DocumentManager(config=mock_config)
        
        # Should pass for allowed types
        manager._validate_file("test.pdf", 1024)
        manager._validate_file("test.docx", 1024)
        manager._validate_file("test.txt", 1024)
        
        # Should raise for disallowed types
        with pytest.raises(ValueError, match="not allowed"):
            manager._validate_file("test.exe", 1024)
        
        with pytest.raises(ValueError, match="not allowed"):
            manager._validate_file("test.sh", 1024)


# ============ Citation Engine Tests ============

class TestCitation:
    """Tests for Citation model."""
    
    def test_citation_creation(self):
        """Test creating a citation."""
        citation = Citation(
            id="cite-1",
            title="Test Paper",
            authors=["John Smith", "Jane Doe"],
            year=2023,
            venue="Test Conference",
            doi="10.1234/test",
        )
        
        assert citation.title == "Test Paper"
        assert len(citation.authors) == 2
        assert citation.year == 2023
        
    def test_citation_key_generation(self):
        """Test citation key generation."""
        key = Citation._generate_citation_key(
            ["John Smith", "Jane Doe"],
            2023
        )
        
        assert key == "Smith_2023"
        
    def test_citation_from_paper(self, sample_paper_data):
        """Test creating citation from Paper object."""
        paper = Paper.from_api_response(sample_paper_data)
        citation = Citation.from_paper(paper, user_id="user-1")
        
        assert citation.title == sample_paper_data["title"]
        assert citation.authors == ["John Smith", "Jane Doe"]
        assert citation.year == 2023


class TestCitationEngine:
    """Tests for CitationEngine."""
    
    def test_format_apa(self):
        """Test APA citation formatting."""
        engine = CitationEngine()
        citation = Citation(
            id="cite-1",
            title="Test Paper Title",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            venue="Test Journal",
            volume="10",
            pages="1-15",
            doi="10.1234/test",
        )
        
        formatted = engine.format_citation(citation, CitationStyle.APA)
        
        assert "2023" in formatted
        assert "Test Paper Title" in formatted
        
    def test_format_mla(self):
        """Test MLA citation formatting."""
        engine = CitationEngine()
        citation = Citation(
            id="cite-1",
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            venue="Test Journal",
        )
        
        formatted = engine.format_citation(citation, CitationStyle.MLA)
        
        assert '"Test Paper."' in formatted
        
    def test_format_bibtex(self):
        """Test BibTeX formatting."""
        engine = CitationEngine()
        citation = Citation(
            id="cite-1",
            title="Test Paper",
            authors=["John Smith", "Jane Doe"],
            year=2023,
            venue="Test Journal",
            doi="10.1234/test",
            citation_key="Smith_2023",
        )
        
        formatted = engine.format_citation(citation, CitationStyle.BIBTEX)
        
        assert "@article{Smith_2023," in formatted
        assert "title = {Test Paper}" in formatted
        assert "year = {2023}" in formatted
        
    def test_extract_citations_from_text(self):
        """Test citation extraction from text."""
        engine = CitationEngine()
        
        text = "According to (Smith, 2023) and confirmed by [1], the results show..."
        citations = engine.extract_citations_from_text(text)
        
        assert len(citations) >= 1
        
    def test_generate_bibliography(self):
        """Test bibliography generation."""
        engine = CitationEngine()
        
        citations = [
            Citation(id="1", title="Paper A", authors=["Smith, John"], year=2023),
            Citation(id="2", title="Paper B", authors=["Doe, Jane"], year=2022),
        ]
        
        bibliography = engine.generate_bibliography(citations, CitationStyle.APA)
        
        assert "Paper A" in bibliography
        assert "Paper B" in bibliography


# ============ Semantic Scholar Tests ============

class TestPaper:
    """Tests for Paper model."""
    
    def test_paper_from_api_response(self, sample_paper_data):
        """Test creating Paper from API response."""
        paper = Paper.from_api_response(sample_paper_data)
        
        assert paper.paper_id == sample_paper_data["paperId"]
        assert paper.title == sample_paper_data["title"]
        assert len(paper.authors) == 2
        assert paper.year == 2023
        assert paper.citation_count == 42
        
    def test_paper_to_dict(self, sample_paper_data):
        """Test converting Paper to dictionary."""
        paper = Paper.from_api_response(sample_paper_data)
        data = paper.to_dict()
        
        assert data["paper_id"] == sample_paper_data["paperId"]
        assert data["title"] == sample_paper_data["title"]


class TestSemanticScholarClient:
    """Tests for Semantic Scholar client."""
    
    @pytest.mark.asyncio
    async def test_search_papers_mock(self, mock_config):
        """Test paper search with mocked response."""
        client = SemanticScholarClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "Test Paper 1",
                        "year": 2023,
                        "authors": [],
                    }
                ]
            }
            
            papers = await client.search_papers("test query")
            
            assert len(papers) == 1
            assert papers[0].title == "Test Paper 1"
            
    @pytest.mark.asyncio
    async def test_get_paper_mock(self, mock_config):
        """Test getting a paper with mocked response."""
        client = SemanticScholarClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "paperId": "paper-1",
                "title": "Test Paper",
                "year": 2023,
                "authors": [{"authorId": "1", "name": "Author"}],
            }
            
            paper = await client.get_paper("paper-1")
            
            assert paper is not None
            assert paper.title == "Test Paper"


# ============ Vector Store Tests ============

class TestSearchResult:
    """Tests for SearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            id="result-1",
            document_id="doc-1",
            user_id="user-1",
            content="Test content",
            score=0.95,
            source_type="document",
        )
        
        assert result.score == 0.95
        assert result.source_type == "document"


class TestVectorStoreManager:
    """Tests for VectorStoreManager."""
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, mock_config, mock_openai_client):
        """Test embedding generation."""
        store = VectorStoreManager(
            config=mock_config,
            openai_client=mock_openai_client,
        )
        
        embedding = await store.generate_embedding("test text")
        
        assert embedding is not None
        assert len(embedding) == 1536


# ============ Query Engine Tests ============

class TestQueryResult:
    """Tests for QueryResult."""
    
    def test_query_result_creation(self):
        """Test creating a query result."""
        result = QueryResult(
            query="test query",
            response="test response",
            sources=[{"id": "1", "title": "Test"}],
            citations=[],
            context_used="test context",
            retrieval_time=0.1,
            generation_time=0.2,
            total_time=0.3,
        )
        
        assert result.query == "test query"
        assert result.response == "test response"
        assert result.total_time == 0.3
        
    def test_query_result_to_dict(self):
        """Test converting QueryResult to dict."""
        result = QueryResult(
            query="test",
            response="response",
            sources=[],
            citations=[],
            context_used="context",
            retrieval_time=0.1,
            generation_time=0.1,
            total_time=0.2,
        )
        
        data = result.to_dict()
        
        assert "query" in data
        assert "response" in data
        assert "sources" in data
        assert "timing" in data


class TestQueryEngine:
    """Tests for QueryEngine."""
    
    def test_get_search_sources(self):
        """Test source mapping."""
        engine = QueryEngine()
        
        assert engine._get_search_sources(QuerySource.ALL) == ["general", "user_docs", "citations"]
        assert engine._get_search_sources(QuerySource.GENERAL) == ["general"]
        assert engine._get_search_sources(QuerySource.USER_DOCS) == ["user_docs"]


# ============ Integration Tests ============

class TestIntegration:
    """Integration tests for the RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_citation_to_document_flow(self, mock_config, sample_paper_data):
        """Test the flow from Paper -> Citation -> Document."""
        # Create paper
        paper = Paper.from_api_response(sample_paper_data)
        
        # Create citation from paper
        citation = Citation.from_paper(paper, user_id="user-1")
        
        # Create document from citation
        doc = Document(
            user_id="user-1",
            title=citation.title,
            content=f"Title: {citation.title}\nAbstract: {paper.abstract}",
            doc_type=DocumentType.SEMANTIC_SCHOLAR,
            authors=citation.authors,
            year=citation.year,
            semantic_scholar_id=paper.paper_id,
        )
        
        assert doc.title == paper.title
        assert doc.semantic_scholar_id == paper.paper_id


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
