# ISAAC RAG Pipeline v2.0

An enhanced RAG (Retrieval-Augmented Generation) pipeline optimized for scientific and research workflows.

## Features

### 📄 Unified Document Management
- Upload and process PDF, DOCX, TXT, MD, HTML files
- Automatic text extraction and chunking
- User-isolated document storage
- Deduplication via content hashing

### 📚 Semantic Scholar Integration
- Search academic papers
- Import citations to personal library
- Automatic citation metadata extraction
- Paper embeddings for semantic search

### 🔍 Hybrid Search
- BM25 + Vector similarity search
- Multi-source retrieval (general docs, user docs, citations)
- User-isolated search
- Relevance ranking

### 📝 Citation Support
- Multiple citation styles (APA, MLA, Chicago, IEEE, Harvard, Vancouver, BibTeX)
- Automatic citation extraction from text
- In-text citation generation
- Bibliography formatting

### ⚡ Performance
- Async processing throughout
- Efficient chunking with LlamaIndex
- Configurable embedding and chunk sizes
- Query rewriting for better retrieval

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  /rag/query  /rag/documents  /rag/papers  /rag/citations    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Query Engine                            │
│  Query Rewriting → Hybrid Search → Context Assembly → LLM   │
└─────────────────────────────────────────────────────────────┘
                              │
┌──────────────────┬──────────────────┬───────────────────────┐
│ Document Manager │   Vector Store   │  Semantic Scholar     │
│  - Upload        │  - OpenSearch    │  - Paper Search       │
│  - Chunking      │  - Embeddings    │  - Citation Import    │
│  - Processing    │  - Hybrid Search │  - Metadata           │
└──────────────────┴──────────────────┴───────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# OpenAI
OPENAI_API_KEY=your-api-key

# OpenSearch
OPENSEARCH_ENDPOINT=http://localhost:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin

# Semantic Scholar (optional, for higher rate limits)
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379
```

## API Endpoints

### Query

```bash
POST /api/rag/query
{
    "query": "What is machine learning?",
    "user_id": "user-123",
    "sources": "all",  # "all" | "general" | "user_docs" | "citations"
    "top_k": 5,
    "citation_style": "apa",
    "language": "en"
}
```

Response:
```json
{
    "query": "What is machine learning?",
    "response": "Machine learning is...",
    "sources": [
        {
            "id": "doc-1",
            "title": "Introduction to ML",
            "url": "https://...",
            "score": 0.95,
            "type": "document"
        }
    ],
    "citations": [...],
    "timing": {
        "retrieval_ms": 150,
        "generation_ms": 500,
        "total_ms": 650
    }
}
```

### Document Upload

```bash
POST /api/rag/documents/upload
Content-Type: multipart/form-data

file: <file>
user_id: user-123
title: My Research Paper
```

### List Documents

```bash
GET /api/rag/documents/{user_id}?limit=50&offset=0
```

### Search Papers (Semantic Scholar)

```bash
GET /api/rag/papers/search?query=machine%20learning&limit=10
```

### Import Citation

```bash
POST /api/rag/citations/import
{
    "paper_id": "semantic-scholar-paper-id",
    "user_id": "user-123"
}
```

### Format Citation

```bash
GET /api/rag/citations/format?paper_id=xxx&style=apa
```

## Usage Examples

### Python

```python
import httpx

async def query_rag(query: str, user_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8088/api/rag/query",
            json={
                "query": query,
                "user_id": user_id,
                "sources": "all",
                "top_k": 5,
            }
        )
        return response.json()

# Upload a document
async def upload_document(file_path: str, user_id: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "http://localhost:8088/api/rag/documents/upload",
                files={"file": f},
                data={"user_id": user_id}
            )
        return response.json()
```

### cURL

```bash
# Query
curl -X POST http://localhost:8088/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "user_id": "user-123"}'

# Upload document
curl -X POST http://localhost:8088/api/rag/documents/upload \
  -F "file=@paper.pdf" \
  -F "user_id=user-123"

# Search papers
curl "http://localhost:8088/api/rag/papers/search?query=neural%20networks&limit=5"
```

## Module Reference

### `rag.config`
Configuration management with environment variable support.

### `rag.document_manager`
Document lifecycle management including upload, processing, and chunking.

### `rag.vector_store`
OpenSearch integration for vector storage and hybrid search.

### `rag.citation_engine`
Citation formatting and extraction for multiple academic styles.

### `rag.semantic_scholar`
Semantic Scholar API client for academic paper search.

### `rag.query_engine`
Main RAG orchestration engine combining all components.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=rag --cov-report=html
```

## Development

### Project Structure

```
isaac_rag/
├── rag/
│   ├── __init__.py
│   ├── config.py           # Configuration
│   ├── document_manager.py # Document handling
│   ├── vector_store.py     # Vector storage
│   ├── citation_engine.py  # Citation formatting
│   ├── semantic_scholar.py # S2 API client
│   ├── query_engine.py     # RAG orchestration
│   └── api.py              # FastAPI endpoints
├── utils/                   # Existing utilities
├── tests/
│   └── test_rag_pipeline.py
├── main.py                  # Application entry
├── api.py                   # Legacy API
└── requirements.txt
```

### Adding a New Citation Style

1. Add style to `CitationStyle` enum in `citation_engine.py`
2. Implement formatter method (e.g., `_format_newstyle`)
3. Register in `style_formatters` dict
4. Add tests

### Adding a New Document Type

1. Add type to `DocumentType` enum in `document_manager.py`
2. Implement extraction method (e.g., `_extract_newtype`)
3. Update upload logic
4. Add tests

## Performance Considerations

- **Chunking**: Default 512 tokens with 50 token overlap
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Search**: Hybrid search combines BM25 (keyword) + KNN (semantic)
- **Caching**: Redis caching for embeddings and search results (optional)

## Security

- User isolation enforced at the vector store level
- API keys loaded from environment variables
- No hardcoded credentials
- Input validation with Pydantic

## License

[Specify License]

## Contributing

[Contributing Guidelines]
