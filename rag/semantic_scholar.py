"""
Semantic Scholar API Integration

Provides integration with Semantic Scholar API for academic paper
search, citation import, and reference management.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Author:
    """Paper author information."""
    author_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    
    
@dataclass
class Paper:
    """
    Semantic Scholar paper model.
    
    Contains comprehensive paper information for citations and retrieval.
    """
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[Author] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    publication_date: Optional[datetime] = None
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    corpus_id: Optional[str] = None
    open_access_pdf: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)
    s2_fields_of_study: List[Dict[str, Any]] = field(default_factory=list)
    tldr: Optional[Dict[str, str]] = None
    embedding: Optional[List[float]] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Paper":
        """Create Paper instance from API response."""
        # Parse authors
        authors = []
        for author_data in data.get("authors", []):
            author = Author(
                author_id=author_data.get("authorId", ""),
                name=author_data.get("name", "Unknown"),
                affiliations=author_data.get("affiliations", []),
            )
            authors.append(author)
            
        # Parse publication date
        pub_date = None
        if data.get("publicationDate"):
            try:
                pub_date = datetime.fromisoformat(data["publicationDate"].replace("Z", "+00:00"))
            except ValueError:
                pass
                
        # Parse external IDs
        external_ids = data.get("externalIds", {})
        
        # Parse open access PDF
        open_access = data.get("openAccessPdf", {})
        pdf_url = open_access.get("url") if open_access else None
        
        # Parse embedding
        embedding = None
        embedding_data = data.get("embedding")
        if embedding_data and "vector" in embedding_data:
            embedding = embedding_data["vector"]
            
        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", "Untitled"),
            abstract=data.get("abstract"),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            publication_date=pub_date,
            citation_count=data.get("citationCount", 0),
            reference_count=data.get("referenceCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            url=data.get("url"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            pubmed_id=external_ids.get("PubMed"),
            corpus_id=external_ids.get("CorpusId"),
            open_access_pdf=pdf_url,
            fields_of_study=data.get("fieldsOfStudy", []),
            s2_fields_of_study=data.get("s2FieldsOfStudy", []),
            tldr=data.get("tldr"),
            embedding=embedding,
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [{"author_id": a.author_id, "name": a.name} for a in self.authors],
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "open_access_pdf": self.open_access_pdf,
            "fields_of_study": self.fields_of_study,
        }


class SemanticScholarClient:
    """
    Semantic Scholar API client for paper search and citation management.
    
    Features:
    - Paper search with relevance ranking
    - Paper details retrieval
    - Bulk paper lookup
    - Rate limiting and caching
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.config = get_config().semantic_scholar
        self.api_key = api_key or self.config.api_key
        self.timeout = timeout or self.config.timeout
        self.base_url = self.config.base_url
        
        # Rate limiting
        self._rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self._request_timestamps: List[float] = []
        
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with optional API key."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        import time
        current_time = time.time()
        
        # Remove timestamps older than 5 minutes
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < 300
        ]
        
        # Check if we're at the limit
        if len(self._request_timestamps) >= self.config.rate_limit_requests:
            wait_time = 300 - (current_time - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
        self._request_timestamps.append(current_time)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an API request with retry logic."""
        await self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with self._rate_limiter:
                try:
                    async with session.get(
                        url,
                        headers=self._get_headers(),
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status == 429:
                            # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            raise Exception("Rate limited")
                            
                        response.raise_for_status()
                        return await response.json()
                        
                except aiohttp.ClientError as e:
                    logger.error(f"API request failed: {e}")
                    raise
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        year_range: Optional[tuple] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False,
    ) -> List[Paper]:
        """
        Search for papers using Semantic Scholar search API.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Pagination offset
            fields: Fields to return
            year_range: Tuple of (start_year, end_year)
            venue: Filter by publication venue
            fields_of_study: Filter by fields of study
            open_access_only: Only return open access papers
            
        Returns:
            List of Paper objects
        """
        # Default fields to return
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year", "authors",
                "venue", "citationCount", "referenceCount", "url",
                "externalIds", "openAccessPdf", "fieldsOfStudy",
                "publicationDate", "tldr",
            ]
            
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": ",".join(fields),
        }
        
        # Add filters
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        if venue:
            params["venue"] = venue
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""
            
        try:
            response = await self._make_request("paper/search", params)
            papers = []
            for item in response.get("data", []):
                papers.append(Paper.from_api_response(item))
            return papers
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            return []
    
    async def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
        include_embedding: bool = False,
    ) -> Optional[Paper]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID or DOI
            fields: Fields to return
            include_embedding: Include paper embedding vector
            
        Returns:
            Paper object or None if not found
        """
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year", "authors",
                "venue", "citationCount", "referenceCount", "url",
                "externalIds", "openAccessPdf", "fieldsOfStudy",
                "publicationDate", "tldr", "influentialCitationCount",
                "s2FieldsOfStudy",
            ]
            
        if include_embedding:
            fields.append("embedding")
            
        params = {
            "fields": ",".join(fields),
        }
        
        try:
            response = await self._make_request(f"paper/{paper_id}", params)
            return Paper.from_api_response(response)
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None
    
    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """
        Get papers cited by a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references
            offset: Pagination offset
            
        Returns:
            List of cited Paper objects
        """
        fields = [
            "paperId", "title", "year", "authors", "venue",
            "citationCount", "url",
        ]
        
        params = {
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
        }
        
        try:
            response = await self._make_request(
                f"paper/{paper_id}/references",
                params
            )
            papers = []
            for item in response.get("data", []):
                cited_paper = item.get("citedPaper", {})
                if cited_paper:
                    papers.append(Paper.from_api_response(cited_paper))
            return papers
        except Exception as e:
            logger.error(f"Failed to get references for {paper_id}: {e}")
            return []
    
    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """
        Get papers that cite a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations
            offset: Pagination offset
            
        Returns:
            List of citing Paper objects
        """
        fields = [
            "paperId", "title", "year", "authors", "venue",
            "citationCount", "url",
        ]
        
        params = {
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
        }
        
        try:
            response = await self._make_request(
                f"paper/{paper_id}/citations",
                params
            )
            papers = []
            for item in response.get("data", []):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    papers.append(Paper.from_api_response(citing_paper))
            return papers
        except Exception as e:
            logger.error(f"Failed to get citations for {paper_id}: {e}")
            return []
    
    async def bulk_get_papers(
        self,
        paper_ids: List[str],
        fields: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Get multiple papers by ID in a single request.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs
            fields: Fields to return
            
        Returns:
            List of Paper objects
        """
        if not paper_ids:
            return []
            
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year", "authors",
                "venue", "citationCount", "url", "externalIds",
            ]
            
        # Semantic Scholar bulk endpoint uses POST
        url = f"{self.base_url}/paper/batch"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json={
                        "ids": paper_ids,
                        "fields": fields,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    papers = []
                    for item in data:
                        papers.append(Paper.from_api_response(item))
                    return papers
                    
            except Exception as e:
                logger.error(f"Bulk paper lookup failed: {e}")
                return []
    
    async def get_author(
        self,
        author_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get author information.
        
        Args:
            author_id: Semantic Scholar author ID
            
        Returns:
            Author information dict
        """
        params = {
            "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex",
        }
        
        try:
            response = await self._make_request(f"author/{author_id}", params)
            return response
        except Exception as e:
            logger.error(f"Failed to get author {author_id}: {e}")
            return None
    
    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """
        Get papers by a specific author.
        
        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers
            offset: Pagination offset
            
        Returns:
            List of Paper objects
        """
        fields = [
            "paperId", "title", "year", "venue", "citationCount",
            "url", "publicationDate",
        ]
        
        params = {
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
        }
        
        try:
            response = await self._make_request(
                f"author/{author_id}/papers",
                params
            )
            papers = []
            for item in response.get("data", []):
                papers.append(Paper.from_api_response(item))
            return papers
        except Exception as e:
            logger.error(f"Failed to get papers for author {author_id}: {e}")
            return []
