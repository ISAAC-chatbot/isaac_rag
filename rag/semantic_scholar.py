"""
Semantic Scholar API client.
Searches for academic papers and fetches metadata + abstracts.
"""
import logging
import time
from typing import List, Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_GRAPH_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,abstract,year,authors,venue,externalIds,"
    "referenceCount,citationCount,influentialCitationCount,openAccessPdf,url"
)


class SemanticScholarClient:
    """
    Thin wrapper around the Semantic Scholar Graph API.
    Rate-limit: 1 req/s without an API key; pass api_key for higher limits.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        fields: str = DEFAULT_FIELDS,
        year_range: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text paper search.
        year_range e.g. "2020-2024" or "2020-"
        Returns a list of paper dicts.
        """
        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
        if year_range:
            params["year"] = year_range

        data = self._get("/paper/search", params=params)
        return data.get("data", [])

    def get_paper(
        self, paper_id: str, fields: str = DEFAULT_FIELDS
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single paper by paperId / DOI / ArXiv ID etc."""
        params = {"fields": fields}
        try:
            return self._get(f"/paper/{paper_id}", params=params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("Paper '%s' not found on Semantic Scholar.", paper_id)
                return None
            raise

    def get_references(
        self, paper_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return papers cited by this paper."""
        params = {
            "fields": "paperId,title,year,authors",
            "limit": limit,
        }
        data = self._get(f"/paper/{paper_id}/references", params=params)
        return [r.get("citedPaper", {}) for r in data.get("data", [])]

    def get_citations(
        self, paper_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return papers that cite this paper."""
        params = {
            "fields": "paperId,title,year,authors",
            "limit": limit,
        }
        data = self._get(f"/paper/{paper_id}/citations", params=params)
        return [r.get("citingPaper", {}) for r in data.get("data", [])]

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = SEMANTIC_SCHOLAR_GRAPH_URL + path
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            # Basic rate-limit backoff
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                logger.warning(
                    "Rate-limited by Semantic Scholar. Waiting %s s.", retry_after
                )
                time.sleep(retry_after)
                resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Semantic Scholar request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def paper_to_text(paper: Dict[str, Any]) -> str:
        """Flatten a paper dict into a string suitable for embedding/retrieval."""
        parts = []
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        year = paper.get("year") or ""
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])
        )
        venue = paper.get("venue") or ""
        if title:
            parts.append(f"Title: {title}")
        if authors:
            parts.append(f"Authors: {authors}")
        if year:
            parts.append(f"Year: {year}")
        if venue:
            parts.append(f"Venue: {venue}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        return "\n".join(parts)
