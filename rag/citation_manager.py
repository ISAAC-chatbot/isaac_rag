"""
Citation manager – formats references in multiple academic styles.
Supports APA, MLA, Chicago, IEEE, Harvard, Vancouver, BibTeX.
"""
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_STYLES = {"apa", "mla", "chicago", "ieee", "harvard", "vancouver", "bibtex"}


def _authors_string(authors: List[Dict[str, Any]]) -> List[str]:
    return [a.get("name", "Unknown") for a in authors or []]


def _first_author_last(authors: List[str]) -> str:
    if not authors:
        return "Unknown"
    first = authors[0]
    parts = first.rsplit(" ", 1)
    return parts[-1] if len(parts) > 1 else first


class CitationManager:
    """
    Formats Semantic Scholar paper dicts into citation strings.

    Usage::

        cm = CitationManager()
        citation = cm.format(paper, style="apa")
    """

    def format(self, paper: Dict[str, Any], style: str = "apa") -> str:
        style = style.lower()
        if style not in SUPPORTED_STYLES:
            raise ValueError(
                f"Unknown citation style '{style}'. "
                f"Supported: {SUPPORTED_STYLES}"
            )
        formatter = getattr(self, f"_fmt_{style}")
        return formatter(paper)

    def format_many(
        self, papers: List[Dict[str, Any]], style: str = "apa"
    ) -> List[str]:
        return [self.format(p, style) for p in papers]

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _fmt_apa(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        url = p.get("url") or ""
        doi = (p.get("externalIds") or {}).get("DOI", "")

        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) <= 7:
            last_author = authors[-1]
            author_str = ", ".join(authors[:-1]) + ", & " + last_author
        else:
            author_str = ", ".join(authors[:6]) + ", ... " + authors[-1]

        parts = [f"{author_str} ({year}). {title}."]
        if venue:
            parts.append(f" *{venue}*.")
        if doi:
            parts.append(f" https://doi.org/{doi}")
        elif url:
            parts.append(f" {url}")
        return "".join(parts)

    def _fmt_mla(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        url = p.get("url") or ""

        if not authors:
            author_str = "Unknown"
        elif len(authors) == 1:
            # "Last, First"
            name = authors[0]
            parts = name.rsplit(" ", 1)
            author_str = f"{parts[-1]}, {parts[0]}" if len(parts) > 1 else name
        else:
            first = authors[0]
            parts = first.rsplit(" ", 1)
            author_str = (
                f"{parts[-1]}, {parts[0]}, et al."
                if len(parts) > 1
                else first + ", et al."
            )

        parts_out = [f'{author_str}. "{title}."']
        if venue:
            parts_out.append(f" *{venue}*,")
        parts_out.append(f" {year}.")
        if url:
            parts_out.append(f" {url}.")
        return " ".join(parts_out)

    def _fmt_chicago(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        url = p.get("url") or ""

        author_str = " and ".join(authors) if authors else "Unknown"
        parts = [f'{author_str}. "{title}."']
        if venue:
            parts.append(f" *{venue}*")
        parts.append(f" ({year}).")
        if url:
            parts.append(f" {url}.")
        return "".join(parts)

    def _fmt_ieee(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        doi = (p.get("externalIds") or {}).get("DOI", "")

        # IEEE uses initials for first names
        def _init(name: str) -> str:
            parts = name.split()
            if len(parts) > 1:
                return ". ".join(w[0] for w in parts[:-1]) + ". " + parts[-1]
            return name

        author_str = " and ".join(_init(a) for a in authors) if authors else "Unknown"
        parts = [f'{author_str}, "{title},"']
        if venue:
            parts.append(f" in *{venue}*,")
        parts.append(f" {year}.")
        if doi:
            parts.append(f" doi: {doi}.")
        return " ".join(parts)

    def _fmt_harvard(self, p: Dict[str, Any]) -> str:
        # Harvard is very close to APA but uses (year) differently
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        doi = (p.get("externalIds") or {}).get("DOI", "")

        if not authors:
            author_str = "Unknown"
        elif len(authors) <= 3:
            author_str = " and ".join(authors)
        else:
            author_str = authors[0] + " et al."

        parts = [f"{author_str} ({year}) '{title}'"]
        if venue:
            parts.append(f", *{venue}*")
        parts.append(".")
        if doi:
            parts.append(f" Available at: https://doi.org/{doi}")
        return "".join(parts)

    def _fmt_vancouver(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        doi = (p.get("externalIds") or {}).get("DOI", "")

        # Vancouver: Surname IN, ...  Title. Journal. Year;...
        def _van_name(name: str) -> str:
            parts = name.split()
            if len(parts) > 1:
                last = parts[-1]
                initials = "".join(w[0] for w in parts[:-1])
                return f"{last} {initials}"
            return name

        if len(authors) > 6:
            author_str = ", ".join(_van_name(a) for a in authors[:6]) + ", et al."
        else:
            author_str = ", ".join(_van_name(a) for a in authors)

        parts = [f"{author_str}. {title}."]
        if venue:
            parts.append(f" {venue}.")
        parts.append(f" {year}.")
        if doi:
            parts.append(f" doi:{doi}.")
        return " ".join(parts)

    def _fmt_bibtex(self, p: Dict[str, Any]) -> str:
        authors = _authors_string(p.get("authors") or [])
        year = str(p.get("year") or "0000")
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        doi = (p.get("externalIds") or {}).get("DOI", "")
        paper_id = re.sub(r"\W+", "", p.get("paperId") or "unknown")[:12]

        author_str = " and ".join(authors) if authors else "Unknown"
        lines = [
            f"@article{{{paper_id},",
            f"  author  = {{{author_str}}},",
            f"  title   = {{{title}}},",
            f"  year    = {{{year}}},",
        ]
        if venue:
            lines.append(f"  journal = {{{venue}}},")
        if doi:
            lines.append(f"  doi     = {{{doi}}},")
        lines.append("}")
        return "\n".join(lines)
