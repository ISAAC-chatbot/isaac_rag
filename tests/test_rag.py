"""
Tests for the scientific RAG pipeline.
Run with: pytest tests/ -v
"""
import math
import pytest
from unittest.mock import MagicMock, patch


# -----------------------------------------------------------------------
# BM25 / HybridRetriever
# -----------------------------------------------------------------------

class TestHybridRetriever:
    def _make_retriever(self):
        from rag.hybrid_retriever import HybridRetriever
        return HybridRetriever(embed_fn=None, alpha=0.0)  # BM25 only

    def test_empty_returns_empty(self):
        r = self._make_retriever()
        assert r.retrieve("anything", top_k=5) == []

    def test_basic_retrieval(self):
        r = self._make_retriever()
        r.add_texts(
            ["BERT is a language model", "transformers changed NLP", "cooking recipes"],
            [{"id": "0"}, {"id": "1"}, {"id": "2"}],
        )
        results = r.retrieve("BERT language", top_k=2)
        assert len(results) == 2
        assert results[0]["text"] == "BERT is a language model"

    def test_top_k_respected(self):
        r = self._make_retriever()
        r.add_texts([f"doc {i}" for i in range(10)])
        results = r.retrieve("doc", top_k=3)
        assert len(results) <= 3

    def test_clear(self):
        r = self._make_retriever()
        r.add_texts(["hello world"])
        r.clear()
        assert r.retrieve("hello", top_k=5) == []

    def test_user_id_isolation(self):
        r = self._make_retriever()
        r.add_texts(
            ["alice secret", "bob public"],
            [{"user_id": "alice"}, {"user_id": "bob"}],
        )
        results = r.retrieve("secret", top_k=5, user_id="bob")
        for res in results:
            uid = res["metadata"].get("user_id")
            assert uid in (None, "bob")

    def test_vector_alpha(self):
        calls = []

        def fake_embed(texts):
            calls.append(texts)
            # Return simple 2-d vectors
            return [[float(i), 0.0] for i in range(len(texts))]

        from rag.hybrid_retriever import HybridRetriever
        r = HybridRetriever(embed_fn=fake_embed, alpha=1.0)  # vector only
        r.add_texts(["a", "b", "c"])
        r.retrieve("x", top_k=3)
        assert len(calls) > 0


# -----------------------------------------------------------------------
# CitationManager
# -----------------------------------------------------------------------

class TestCitationManager:
    _PAPER = {
        "paperId": "abc123",
        "title": "Attention Is All You Need",
        "year": 2017,
        "venue": "NeurIPS",
        "authors": [
            {"name": "Ashish Vaswani"},
            {"name": "Noam Shazeer"},
            {"name": "Niki Parmar"},
        ],
        "externalIds": {"DOI": "10.5555/3295222.3295349"},
        "url": "https://arxiv.org/abs/1706.03762",
    }

    def _cm(self):
        from rag.citation_manager import CitationManager
        return CitationManager()

    def test_apa(self):
        c = self._cm().format(self._PAPER, "apa")
        assert "Vaswani" in c
        assert "2017" in c
        assert "Attention Is All You Need" in c

    def test_bibtex(self):
        c = self._cm().format(self._PAPER, "bibtex")
        assert "@article" in c
        assert "Attention Is All You Need" in c
        assert "2017" in c

    def test_ieee(self):
        c = self._cm().format(self._PAPER, "ieee")
        assert "Attention Is All You Need" in c
        assert "NeurIPS" in c

    def test_invalid_style(self):
        with pytest.raises(ValueError):
            self._cm().format(self._PAPER, "nonexistent")

    def test_format_many(self):
        results = self._cm().format_many([self._PAPER, self._PAPER], "mla")
        assert len(results) == 2


# -----------------------------------------------------------------------
# DocumentManager
# -----------------------------------------------------------------------

class TestDocumentManager:
    def _make_tmpfile(self, content: str, suffix: str = ".txt", tmp_path=None):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.write(fd, content.encode())
        os.close(fd)
        return path

    def test_txt_add_and_retrieve(self):
        from rag.document_manager import DocumentManager
        dm = DocumentManager(chunk_size=50, chunk_overlap=10)
        path = self._make_tmpfile("Hello world " * 20)
        doc = dm.add_document(path, user_id="u1")
        assert len(doc.chunks) > 0
        assert dm.get_document(doc.doc_id) is not None

    def test_deduplication(self):
        from rag.document_manager import DocumentManager
        dm = DocumentManager()
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, b"unique content here")
        os.close(fd)
        doc1 = dm.add_document(path, user_id="u1")
        doc2 = dm.add_document(path, user_id="u1")
        assert doc1.doc_id == doc2.doc_id

    def test_delete(self):
        from rag.document_manager import DocumentManager
        dm = DocumentManager()
        path = self._make_tmpfile("delete me please yes")
        doc = dm.add_document(path, user_id="u2")
        assert dm.delete_document(doc.doc_id) is True
        assert dm.get_document(doc.doc_id) is None

    def test_unsupported_type(self):
        from rag.document_manager import DocumentManager
        dm = DocumentManager()
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        with pytest.raises(ValueError, match="Unsupported"):
            dm.add_document(path, user_id="u1")

    def test_list_by_user(self):
        from rag.document_manager import DocumentManager
        dm = DocumentManager()
        for i, user in enumerate(["alice", "alice", "bob"]):
            path = self._make_tmpfile(f"content {i} {'x'*50}")
            dm.add_document(path, user_id=user)
        assert len(dm.list_documents(user_id="alice")) == 2
        assert len(dm.list_documents(user_id="bob")) == 1


# -----------------------------------------------------------------------
# SemanticScholarClient (mocked HTTP)
# -----------------------------------------------------------------------

class TestSemanticScholarClient:
    def test_paper_to_text(self):
        from rag.semantic_scholar import SemanticScholarClient
        paper = {
            "title": "Test Paper",
            "abstract": "An abstract.",
            "year": 2023,
            "authors": [{"name": "Alice Bob"}],
            "venue": "ICML",
        }
        text = SemanticScholarClient.paper_to_text(paper)
        assert "Test Paper" in text
        assert "Alice Bob" in text
        assert "ICML" in text

    def test_search_success(self):
        from rag.semantic_scholar import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"paperId": "p1", "title": "Result One"}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "get", return_value=mock_resp):
            results = client.search("transformers")
        assert len(results) == 1
        assert results[0]["title"] == "Result One"

    def test_get_paper_404(self):
        from rag.semantic_scholar import SemanticScholarClient
        import requests
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_err = requests.HTTPError(response=mock_resp)
        with patch.object(client._session, "get", side_effect=http_err):
            result = client.get_paper("nonexistent")
        assert result is None
