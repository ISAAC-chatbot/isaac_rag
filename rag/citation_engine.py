"""
Citation Engine Module

Provides citation extraction, formatting, and management for academic
and research workflows.

Supports:
- Multiple citation styles (APA, MLA, Chicago, IEEE, etc.)
- Automatic citation extraction from user documents
- Semantic Scholar integration for citation metadata
- BibTeX export/import
"""

import re
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .semantic_scholar import Paper, Author

logger = logging.getLogger(__name__)


class CitationStyle(str, Enum):
    """Supported citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"
    BIBTEX = "bibtex"


@dataclass
class Citation:
    """
    Academic citation model.
    
    Contains all necessary information for citation generation
    across multiple styles.
    """
    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    citation_key: Optional[str] = None
    abstract: Optional[str] = None
    
    # Source tracking
    source_type: str = "citation"  # "citation", "document", "generated"
    document_id: Optional[str] = None
    paper_id: Optional[str] = None  # Semantic Scholar ID
    
    @classmethod
    def from_paper(cls, paper: Paper, user_id: str = "") -> "Citation":
        """Create Citation from Semantic Scholar Paper."""
        authors = [a.name for a in paper.authors]
        citation_key = cls._generate_citation_key(authors, paper.year)
        
        return cls(
            id=f"{paper.paper_id}_{user_id}",
            title=paper.title,
            authors=authors,
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            url=paper.url,
            citation_key=citation_key,
            abstract=paper.abstract,
            source_type="citation",
            paper_id=paper.paper_id,
        )
    
    @staticmethod
    def _generate_citation_key(authors: List[str], year: Optional[int]) -> str:
        """Generate a citation key (e.g., Smith_2023)."""
        if authors:
            last_name = authors[0].split()[-1]
        else:
            last_name = "Unknown"
        year_str = str(year) if year else "n.d."
        return f"{last_name}_{year_str}".replace(" ", "")


class CitationEngine:
    """
    Citation management and formatting engine.
    
    Features:
    - Multiple citation style formatting
    - Citation extraction from text
    - BibTeX generation
    - Context-aware citation suggestions
    """
    
    def __init__(self):
        self.style_formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.IEEE: self._format_ieee,
            CitationStyle.HARVARD: self._format_harvard,
            CitationStyle.VANCOUVER: self._format_vancouver,
            CitationStyle.BIBTEX: self._format_bibtex,
        }
    
    def format_citation(
        self,
        citation: Citation,
        style: CitationStyle = CitationStyle.APA,
        include_url: bool = True,
    ) -> str:
        """
        Format a citation in the specified style.
        
        Args:
            citation: Citation object
            style: Citation style
            include_url: Include URL in citation
            
        Returns:
            Formatted citation string
        """
        formatter = self.style_formatters.get(style, self._format_apa)
        return formatter(citation, include_url)
    
    def _format_apa(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in APA style."""
        # Authors: Last, F. M., & Last, F. M.
        authors_str = self._format_authors_apa(citation.authors)
        
        # Year
        year_str = f"({citation.year})" if citation.year else "(n.d.)"
        
        # Title (italicized for journal, regular for others)
        title = citation.title
        
        # Source (journal/conference)
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.volume:
            source_parts.append(citation.volume)
        if citation.issue:
            source_parts.append(f"({citation.issue})")
        if citation.pages:
            source_parts.append(citation.pages)
        
        # DOI or URL
        identifier = ""
        if citation.doi:
            identifier = f"https://doi.org/{citation.doi}"
        elif include_url and citation.url:
            identifier = citation.url
        
        # Assemble
        parts = [authors_str, year_str, f"{title}."]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        if identifier:
            parts.append(identifier)
        
        return " ".join(parts)
    
    def _format_mla(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in MLA style."""
        # Authors: Last, First. (et al. for 3+)
        authors_str = self._format_authors_mla(citation.authors)
        
        # Title in quotes
        title = f'"{citation.title}."'
        
        # Source
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.year:
            source_parts.append(str(citation.year))
        
        # Pages
        if citation.pages:
            source_parts.append(f"pp. {citation.pages}")
        
        # URL (in angle brackets)
        url_part = ""
        if include_url and citation.url:
            url_part = f"<{citation.url}>."
        
        parts = [authors_str, title]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        if url_part:
            parts.append(url_part)
        
        return " ".join(parts)
    
    def _format_chicago(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in Chicago style (notes-bibliography)."""
        # Authors: Last, First, and First Last.
        authors_str = self._format_authors_chicago(citation.authors)
        
        # Title (italicized)
        title = f"{citation.title}."
        
        # Source
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.volume and citation.issue:
            source_parts.append(f"{citation.volume}, no. {citation.issue}")
        elif citation.volume:
            source_parts.append(citation.volume)
        if citation.year:
            source_parts.append(f"({citation.year})")
        if citation.pages:
            source_parts.append(citation.pages)
        
        # DOI
        doi_part = ""
        if citation.doi:
            doi_part = f"https://doi.org/{citation.doi}."
        
        parts = [authors_str, title]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        if doi_part:
            parts.append(doi_part)
        elif include_url and citation.url:
            parts.append(citation.url + ".")
        
        return " ".join(parts)
    
    def _format_ieee(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in IEEE style."""
        # Authors: F. Last, F. Last, and F. Last
        authors_str = self._format_authors_ieee(citation.authors)
        
        # Title in quotes
        title = f'"{citation.title},"'
        
        # Source
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.volume:
            source_parts.append(f"vol. {citation.volume}")
        if citation.issue:
            source_parts.append(f"no. {citation.issue}")
        if citation.pages:
            source_parts.append(f"pp. {citation.pages}")
        if citation.year:
            source_parts.append(str(citation.year))
        
        parts = [authors_str, title]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        
        return " ".join(parts)
    
    def _format_harvard(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in Harvard style."""
        # Authors: Last, F., Last, F. and Last, F.
        authors_str = self._format_authors_harvard(citation.authors)
        
        # Year
        year_str = f"({citation.year})" if citation.year else "(n.d.)"
        
        # Title
        title = f"{citation.title}."
        
        # Source
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.volume:
            source_parts.append(citation.volume)
        if citation.issue:
            source_parts.append(f"({citation.issue})")
        if citation.pages:
            source_parts.append(citation.pages)
        
        # URL
        url_part = ""
        if include_url and citation.url:
            url_part = f"Available at: {citation.url}"
        
        parts = [authors_str, year_str, title]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        if url_part:
            parts.append(url_part)
        
        return " ".join(parts)
    
    def _format_vancouver(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation in Vancouver style."""
        # Authors: Last FM, Last FM.
        authors_str = self._format_authors_vancouver(citation.authors)
        
        # Title
        title = f"{citation.title}."
        
        # Source
        source_parts = []
        if citation.venue:
            source_parts.append(citation.venue)
        if citation.year:
            source_parts.append(str(citation.year))
        if citation.volume:
            source_parts.append(citation.volume)
        if citation.pages:
            source_parts.append(citation.pages)
        
        # DOI
        doi_part = ""
        if citation.doi:
            doi_part = f"doi: {citation.doi}"
        
        parts = [authors_str, title]
        if source_parts:
            parts.append(", ".join(source_parts) + ".")
        if doi_part:
            parts.append(doi_part)
        
        return " ".join(parts)
    
    def _format_bibtex(self, citation: Citation, include_url: bool = True) -> str:
        """Format citation as BibTeX entry."""
        entry_type = "article" if citation.venue else "misc"
        key = citation.citation_key or f"ref_{citation.id[:8]}"
        
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{citation.title}}},")
        
        if citation.authors:
            authors_str = " and ".join(citation.authors)
            lines.append(f"  author = {{{authors_str}}},")
        
        if citation.year:
            lines.append(f"  year = {{{citation.year}}},")
        
        if citation.venue:
            lines.append(f"  journal = {{{citation.venue}}},")
        
        if citation.volume:
            lines.append(f"  volume = {{{citation.volume}}},")
        
        if citation.issue:
            lines.append(f"  number = {{{citation.issue}}},")
        
        if citation.pages:
            lines.append(f"  pages = {{{citation.pages}}},")
        
        if citation.doi:
            lines.append(f"  doi = {{{citation.doi}}},")
        
        if include_url and citation.url:
            lines.append(f"  url = {{{citation.url}}},")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    # Author formatting helpers
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors in APA style."""
        if not authors:
            return ""
        if len(authors) == 1:
            return self._author_to_apa(authors[0])
        if len(authors) == 2:
            return f"{self._author_to_apa(authors[0])} & {self._author_to_apa(authors[1])}"
        # 3+ authors
        return f"{self._author_to_apa(authors[0])} et al."
    
    def _author_to_apa(self, name: str) -> str:
        """Convert full name to APA format: Last, F. M."""
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        last = parts[-1]
        initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
        return f"{last}, {initials}"
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors in MLA style."""
        if not authors:
            return ""
        if len(authors) <= 2:
            return ". ".join(authors) + "."
        return f"{authors[0]}, et al."
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Format authors in Chicago style."""
        if not authors:
            return ""
        if len(authors) == 1:
            return authors[0] + "."
        if len(authors) == 2:
            return f"{authors[0]}, and {authors[1]}."
        # 3+ authors
        return f"{authors[0]}, et al."
    
    def _format_authors_ieee(self, authors: List[str]) -> str:
        """Format authors in IEEE style: F. Last."""
        formatted = []
        for name in authors:
            parts = name.split()
            if len(parts) == 1:
                formatted.append(parts[0])
            else:
                initials = " ".join(p[0].upper() + "." for p in parts[:-1])
                formatted.append(f"{initials} {parts[-1]}")
        return ", ".join(formatted[:6]) + (", et al." if len(authors) > 6 else "")
    
    def _format_authors_harvard(self, authors: List[str]) -> str:
        """Format authors in Harvard style."""
        if not authors:
            return ""
        if len(authors) == 1:
            return self._author_to_apa(authors[0])
        if len(authors) == 2:
            return f"{self._author_to_apa(authors[0])} and {self._author_to_apa(authors[1])}"
        return f"{self._author_to_apa(authors[0])} et al."
    
    def _format_authors_vancouver(self, authors: List[str]) -> str:
        """Format authors in Vancouver style: Last FM."""
        formatted = []
        for name in authors[:6]:
            parts = name.split()
            if len(parts) == 1:
                formatted.append(parts[0])
            else:
                initials = "".join(p[0].upper() for p in parts[:-1])
                formatted.append(f"{parts[-1]} {initials}")
        return ", ".join(formatted) + (", et al" if len(authors) > 6 else "")
    
    def extract_citations_from_text(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract citation references from text.
        
        Detects:
        - Author-year citations: (Smith, 2023) or (Smith and Jones, 2023)
        - Numeric citations: [1] or [1, 2, 3]
        - DOI references: doi:10.xxxx/xxxxx
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of detected citation references
        """
        citations = []
        
        # Author-year pattern
        author_year_pattern = r'\(([A-Z][a-z]+(?:\s+(?:and|&,?)?\s*[A-Z][a-z]+)*)[,\s]+(\d{4}[a-z]?)\)'
        for match in re.finditer(author_year_pattern, text):
            citations.append({
                "type": "author_year",
                "text": match.group(0),
                "authors": match.group(1),
                "year": match.group(2),
                "position": match.start(),
            })
        
        # Numeric citation pattern
        numeric_pattern = r'\[(\d+(?:\s*[-,]\s*\d+)*)\]'
        for match in re.finditer(numeric_pattern, text):
            numbers = re.findall(r'\d+', match.group(1))
            citations.append({
                "type": "numeric",
                "text": match.group(0),
                "numbers": [int(n) for n in numbers],
                "position": match.start(),
            })
        
        # DOI pattern
        doi_pattern = r'(?:doi:|https?://doi\.org/|https?://dx\.doi\.org/)(10\.\d{4,}/[^\s]+)'
        for match in re.finditer(doi_pattern, text, re.IGNORECASE):
            citations.append({
                "type": "doi",
                "text": match.group(0),
                "doi": match.group(1),
                "position": match.start(),
            })
        
        return citations
    
    def generate_in_text_citation(
        self,
        citation: Citation,
        style: CitationStyle = CitationStyle.APA,
        location: Optional[str] = None,
    ) -> str:
        """
        Generate in-text citation format.
        
        Args:
            citation: Citation object
            style: Citation style
            location: Page or section reference
            
        Returns:
            In-text citation string
        """
        if style in [CitationStyle.APA, CitationStyle.HARVARD]:
            # (Author, Year) or Author (Year)
            if citation.authors:
                author = citation.authors[0].split()[-1]
            else:
                author = "Unknown"
            year = citation.year or "n.d."
            
            if location:
                return f"({author}, {year}, {location})"
            return f"({author}, {year})"
        
        elif style == CitationStyle.MLA:
            # (Author Page)
            if citation.authors:
                author = citation.authors[0].split()[-1]
            else:
                author = "Unknown"
            page = location or citation.pages or ""
            return f"({author} {page})".strip()
        
        elif style == CitationStyle.IEEE:
            # [1]
            return f"[{citation.id}]"
        
        elif style == CitationStyle.VANCOUVER:
            # Superscript numeric
            return f"[{citation.id}]"
        
        return f"({citation.title[:20]}...)"
    
    def generate_bibliography(
        self,
        citations: List[Citation],
        style: CitationStyle = CitationStyle.APA,
        sort_by: str = "author",  # "author", "year", "title"
    ) -> str:
        """
        Generate formatted bibliography from citations.
        
        Args:
            citations: List of Citation objects
            style: Citation style
            sort_by: Sorting method
            
        Returns:
            Formatted bibliography string
        """
        # Sort citations
        if sort_by == "author":
            sorted_citations = sorted(citations, key=lambda c: (c.authors[0] if c.authors else "", c.year or 0))
        elif sort_by == "year":
            sorted_citations = sorted(citations, key=lambda c: c.year or 0, reverse=True)
        else:
            sorted_citations = sorted(citations, key=lambda c: c.title)
        
        # Format each citation
        formatted = []
        for i, citation in enumerate(sorted_citations, 1):
            if style == CitationStyle.VANCOUVER or style == CitationStyle.IEEE:
                formatted.append(f"{i}. {self.format_citation(citation, style)}")
            else:
                formatted.append(self.format_citation(citation, style))
        
        return "\n\n".join(formatted)
