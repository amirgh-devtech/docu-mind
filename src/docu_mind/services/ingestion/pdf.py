"""PDF parser using PyMuPDF (fitz) with multi-signal heading detection.

This parser uses a combination of heuristics to detect headings
without any LLM calls:

    1. **Font size**: larger than the document's body font.
    2. **Bold / weight**: font has the bold attribute.
    3. **Font family**: differs from the dominant body font.
    4. **ALL CAPS**: most letters are uppercase.
    5. **Numbered prefixes**: "1.", "1.1", "Chapter 3", "Section 2.4".
    6. **Layout**: short text with no ending punctuation.

Body font and dominant fonts are computed per-document (weighted by
character count) so the parser adapts to each PDF's typography.

Header and footer noise (repeated text on every page, page numbers)
is filtered out before heading detection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import pymupdf

from docu_mind.core.exceptions import ParseError
from docu_mind.core.logging import get_logger
from docu_mind.services.ingestion.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────

#: Font size must be at least this many times body_size to be a heading signal.
_SIZE_RATIO_STRONG = 1.20  # very likely a heading
_SIZE_RATIO_WEAK = 1.10  # possibly a heading (needs other signals)

#: Absolute minimum font size (points) for a heading candidate.
_MIN_HEADING_SIZE = 11.0

#: Max characters / words for heading text.
_MAX_HEADING_CHARS = 120
_MAX_HEADING_WORDS = 15

#: Score threshold for classifying a span as a heading.
_HEADING_SCORE_THRESHOLD = 4

#: PDF text span flag for bold (bit 4).
_BOLD_FLAG = 1 << 4
_ITALIC_FLAG = 1 << 1

#: Vertical margin fraction for header/footer detection.
_HEADER_FOOTER_MARGIN = 0.08  # top/bottom 8% of page height

#: Minimum number of pages for header/footer detection to kick in.
_MIN_PAGES_FOR_HEADER_DETECTION = 3

#: Max number of chars for a line to be considered header/footer text.
_MAX_HEADER_FOOTER_CHARS = 120


# Patterns for numbered headings
_NUMBERED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+(\.\d+)*\.?\s+\S"),  # "1.", "1.1", "1.2.3"
    re.compile(r"^(chapter|section|part)\s+\d+", re.IGNORECASE),  # "Chapter 1"
    re.compile(r"^[ivxlcdm]+\.\s+\S", re.IGNORECASE),  # "IV." Roman numerals
    re.compile(r"^[A-Z]\.\s+\S"),  # "A." letter
)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass(slots=True)
class _Span:
    """A text span extracted from the PDF, with all metadata we need."""

    text: str
    size: float
    bold: bool
    italic: bool
    font: str
    page: int
    block: int
    line: int
    #: Bounding box: (x0, y0, x1, y1) in page coordinates.
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_short(self) -> bool:
        return self.char_count <= _MAX_HEADING_CHARS and self.word_count <= _MAX_HEADING_WORDS

    @property
    def ends_cleanly(self) -> bool:
        """True if the text doesn't end with sentence punctuation."""
        return not self.text.rstrip().endswith((".", "!", "?", ",", ";", ":"))


@dataclass(slots=True)
class _DocStats:
    """Document-level typography statistics."""

    body_size: float
    body_fonts: set[str] = field(default_factory=set)
    page_heights: dict[int, float] = field(default_factory=dict)
    median_page_height: float = 792.0  # US Letter default

    @property
    def min_heading_size(self) -> float:
        return max(
            self.body_size * _SIZE_RATIO_WEAK,
            _MIN_HEADING_SIZE,
        )


# ─────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────
class PDFParser(BaseParser):
    """Parse PDF files into section-based chunks using heuristics."""

    extensions = ("pdf",)

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a PDF file into section-based ParsedPage items."""
        try:
            doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = f"Failed to open PDF {path.name}: {e}"
            raise ParseError(msg) from e

        try:
            # 1. Extract spans
            spans = self._extract_spans(doc)
            if not spans:
                logger.warning("ingestion.pdf.no_text", filename=path.name)
                return ParsedDocument(
                    pages=[],
                    metadata={
                        "page_count": doc.page_count,
                        "scanned": True,
                        "warning": "no text layer found — may need OCR",
                    },
                )

            # 2. Compute document stats
            stats = self._compute_stats(doc, spans)

            # 3. Filter noise (headers, footers, page numbers)
            spans = self._filter_noise(spans, stats)

            if not spans:
                logger.warning("ingestion.pdf.all_filtered", filename=path.name)
                return ParsedDocument(
                    pages=[],
                    metadata={"page_count": doc.page_count},
                )

            # 4 + 5 + 6. Detect headings, infer levels, group into sections
            pages = self._group_into_sections(spans, stats)

            # 7. Merge tiny sections (if any)
            pages = self._merge_tiny_sections(pages)

            metadata: dict[str, Any] = {
                "page_count": doc.page_count,
                "title": (doc.metadata or {}).get("title") or None,
                "author": (doc.metadata or {}).get("author") or None,
                "producer": (doc.metadata or {}).get("producer") or None,
                "body_font_size": stats.body_size,
                "body_fonts": sorted(stats.body_fonts),
                "section_count": len(pages),
                "total_spans": len(spans),
            }

            logger.info(
                "ingestion.pdf.parsed",
                filename=path.name,
                sections=len(pages),
                chars=sum(len(p.text) for p in pages),
                body_size=stats.body_size,
                total_spans=len(spans),
            )

            return ParsedDocument(pages=pages, metadata=metadata)
        finally:
            doc.close()  # type: ignore[no-untyped-call]

    # ─────────────────────────────────────────────
    # Step 1: Extract spans
    # ─────────────────────────────────────────────
    @staticmethod
    def _extract_spans(doc: Any) -> list[_Span]:
        """Extract all text spans with metadata (size, font, bbox, etc.)."""
        result: list[_Span] = []

        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")

            for block_idx, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type") != 0:  # skip images
                    continue

                for line_idx, line in enumerate(block.get("lines", [])):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        flags = span.get("flags", 0)
                        bbox_raw = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                        bbox = tuple(float(x) for x in bbox_raw[:4])

                        result.append(
                            _Span(
                                text=text,
                                size=round(float(span.get("size", 0.0)), 1),
                                bold=bool(flags & _BOLD_FLAG),
                                italic=bool(flags & _ITALIC_FLAG),
                                font=str(span.get("font", "")),
                                page=page_num,
                                block=block_idx,
                                line=line_idx,
                                bbox=bbox,  # type: ignore[arg-type]
                            )
                        )
        return result

    # ─────────────────────────────────────────────
    # Step 2: Compute document stats
    # ─────────────────────────────────────────────
    @staticmethod
    def _compute_stats(doc: Any, spans: list[_Span]) -> _DocStats:
        """Compute document-level typography statistics."""
        # 2a. Body size = char-weighted mode of sizes
        size_weights: Counter[float] = Counter()
        for span in spans:
            if span.size > 0:
                size_weights[span.size] += span.char_count

        body_size = size_weights.most_common(1)[0][0] if size_weights else 10.0

        # 2b. Body fonts = fonts used at body size
        body_fonts: set[str] = set()
        for span in spans:
            if abs(span.size - body_size) < 0.5 and span.font:
                body_fonts.add(span.font)

        # 2c. Page heights
        page_heights: dict[int, float] = {}
        for page_num, page in enumerate(doc, start=1):
            rect = page.rect
            page_heights[page_num] = float(rect.height)

        # 2d. Median page height
        if page_heights:
            heights = sorted(page_heights.values())
            median = heights[len(heights) // 2]
        else:
            median = 792.0

        return _DocStats(
            body_size=body_size,
            body_fonts=body_fonts,
            page_heights=page_heights,
            median_page_height=median,
        )

    # ─────────────────────────────────────────────
    # Step 3: Filter noise (headers, footers)
    # ─────────────────────────────────────────────
    def _filter_noise(
        self,
        spans: list[_Span],
        stats: _DocStats,
    ) -> list[_Span]:
        """Remove header/footer text and page-number-like spans."""
        if not spans:
            return spans

        # If the document is too short, skip header/footer detection
        unique_pages = {s.page for s in spans}
        if len(unique_pages) < _MIN_PAGES_FOR_HEADER_DETECTION:
            return self._drop_page_numbers(spans)

        # Find spans in the top/bottom margins that repeat on many pages
        top_texts: Counter[str] = Counter()
        bottom_texts: Counter[str] = Counter()

        for span in spans:
            if span.char_count > _MAX_HEADER_FOOTER_CHARS:
                continue
            page_height = stats.page_heights.get(span.page, stats.median_page_height)
            margin = page_height * _HEADER_FOOTER_MARGIN
            y0 = span.bbox[1]
            y1 = span.bbox[3]

            if y1 <= margin:
                top_texts[span.text] += 1
            elif y0 >= page_height - margin:
                bottom_texts[span.text] += 1

        # Text appearing in >50% of pages in same zone → header/footer
        threshold = max(2, len(unique_pages) // 2)
        header_texts = {t for t, c in top_texts.items() if c >= threshold}
        footer_texts = {t for t, c in bottom_texts.items() if c >= threshold}

        if header_texts or footer_texts:
            logger.info(
                "ingestion.pdf.noise_detected",
                headers=len(header_texts),
                footers=len(footer_texts),
            )

        filtered: list[_Span] = []
        for span in spans:
            if span.text in header_texts or span.text in footer_texts:
                continue
            filtered.append(span)

        return self._drop_page_numbers(filtered)

    @staticmethod
    def _drop_page_numbers(spans: list[_Span]) -> list[_Span]:
        """Drop spans that look like page numbers (e.g. "42", "- 42 -")."""
        page_num_pattern = re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$")  # noqa: RUF001
        return [s for s in spans if not page_num_pattern.match(s.text)]

    # ─────────────────────────────────────────────
    # Step 4: Heading detection (multi-signal)
    # ─────────────────────────────────────────────
    def _is_heading(self, span: _Span, stats: _DocStats) -> bool:
        """Return True if the span looks like a heading.

        Combines multiple weak signals into a single score.
        """
        # Reject obvious non-headings first
        if not span.is_short or not span.ends_cleanly:
            return False

        score = 0

        # Signal 1: font size
        size_ratio = span.size / stats.body_size if stats.body_size > 0 else 1.0
        if size_ratio >= _SIZE_RATIO_STRONG:
            score += 4
        elif size_ratio >= _SIZE_RATIO_WEAK:
            score += 2

        # Signal 2: bold
        if span.bold:
            score += 2

        # Signal 3: different font family (not the body font)
        if span.font and stats.body_fonts and span.font not in stats.body_fonts:
            score += 1

        # Signal 4: ALL CAPS (most chars uppercase)
        if self._is_mostly_caps(span.text):
            score += 2

        # Signal 5: numbered prefix
        if self._has_numbered_prefix(span.text):
            score += 3

        # Signal 6: short + no punctuation
        if span.word_count <= 12:
            score += 1

        return score >= _HEADING_SCORE_THRESHOLD

    @staticmethod
    def _is_mostly_caps(text: str) -> bool:
        """True if ≥70% of alphabetic chars are uppercase."""
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 2:
            return False
        uppercase = sum(1 for c in letters if c.isupper())
        return uppercase / len(letters) >= 0.7

    @staticmethod
    def _has_numbered_prefix(text: str) -> bool:
        """True if the text starts with a numbered/lettered prefix."""
        return any(p.match(text) for p in _NUMBERED_PATTERNS)

    # ─────────────────────────────────────────────
    # Step 5: Level inference
    # ─────────────────────────────────────────────
    @staticmethod
    def _infer_level(span: _Span, stats: _DocStats) -> int:
        """Infer a heading level (1-4) from size, bold, and numbering."""
        text = span.text
        size_ratio = span.size / stats.body_size if stats.body_size > 0 else 1.0

        # Numbered patterns override size-based inference
        # "1." → level 1; "1.1" → level 2; "1.1.1" → level 3
        m = re.match(r"^(\d+(?:\.\d+)*)", text)
        if m:
            depth = m.group(1).count(".") + 1
            return min(depth, 4)

        # "Chapter N" / "Section N" → level 1
        if re.match(r"^(chapter|part)\s+\d+", text, re.IGNORECASE):
            return 1

        # Size-based fallback
        if size_ratio >= 1.6:
            return 1
        if size_ratio >= 1.35:
            return 2
        if span.bold and size_ratio >= 1.15:
            return 2
        return 3

    # ─────────────────────────────────────────────
    # Step 6: Section grouping
    # ─────────────────────────────────────────────
    def _group_into_sections(
        self,
        spans: list[_Span],
        stats: _DocStats,
    ) -> list[ParsedPage]:
        """Group spans into sections (heading + following body text)."""
        pages: list[ParsedPage] = []
        heading_stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        current_page = spans[0].page if spans else 1

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else None
                pages.append(
                    ParsedPage(
                        text=text,
                        page_number=current_page,
                        heading_path=heading_path,
                    )
                )
            buffer.clear()

        for span in spans:
            if self._is_heading(span, stats):
                flush()
                level = self._infer_level(span, stats)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, span.text))
                current_page = span.page
            else:
                buffer.append(span.text)

        flush()
        return pages

    # ─────────────────────────────────────────────
    # Step 7: Merge tiny sections
    # ─────────────────────────────────────────────
    @staticmethod
    def _merge_tiny_sections(
        pages: list[ParsedPage],
        min_chars: int = 40,
    ) -> list[ParsedPage]:
        """Merge sections that are too short into the next one.

        Very short sections (e.g. a heading followed by a single line)
        produce low-quality chunks; merging them improves retrieval.
        """
        if not pages:
            return pages

        merged: list[ParsedPage] = []
        pending: ParsedPage | None = None

        for original in pages:
            # If the current page is too short, buffer it
            if len(original.text) < min_chars:
                if pending is None:
                    pending = original
                else:
                    pending = ParsedPage(
                        text=f"{pending.text}\n\n{original.text}",
                        page_number=pending.page_number,
                        heading_path=pending.heading_path,
                    )
                continue

            # Current page is long enough — emit it
            if pending is not None:
                merged.append(
                    ParsedPage(
                        text=f"{pending.text}\n\n{original.text}",
                        page_number=original.page_number,
                        heading_path=original.heading_path,
                    )
                )
                pending = None
            else:
                merged.append(original)

        # Flush any remaining pending content
        if pending is not None:
            merged.append(pending)

        return merged
