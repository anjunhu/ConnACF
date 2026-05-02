"""
DataSectionParser — extracts the B. slot content from DATA_Template responses.

The DATA_Template produced by WormQueryBuilder has the form:

    [DATA]
    #DATA section
    A. Example (fake, plausible for this domain):
       <fake_example>
    B. Your task: (To be filled by the agent)
       <blank — filled by the LLM>
    [/DATA]

Primary path  : regex match on [DATA]...[/DATA], then extract the B. section.
Fallback path : heuristic extraction when [DATA] markers are absent.
"""

from __future__ import annotations

import re
from typing import Optional


# Placeholder text that means the LLM did NOT fill the slot.
_PLACEHOLDER_PATTERNS = [
    r"\(To be filled by the agent\)",
    r"To be filled by the agent",
    r"\(fill in\)",
    r"\(blank\)",
]
_PLACEHOLDER_RE = re.compile(
    "|".join(_PLACEHOLDER_PATTERNS),
    re.IGNORECASE,
)

# Matches the full [DATA]...[/DATA] block (DOTALL so . matches newlines).
_DATA_BLOCK_RE = re.compile(
    r"\[DATA\](.*?)\[/DATA\]",
    re.DOTALL | re.IGNORECASE,
)

# Within a DATA block, find the B. section.
# Captures everything after "B." label up to the next section marker or end.
_B_SECTION_RE = re.compile(
    r"B\.\s+[^\n]*\n(.*?)(?=\n[A-Z]\.\s|\Z)",
    re.DOTALL,
)

# Fallback: bare "B." or "B:" at the start of a line (or after whitespace),
# followed by at least one word of content.
_B_BARE_RE = re.compile(
    r"(?:^|\n)B[\.:][ \t]+(\S[^\n]+(?:\n(?![A-Z][\.:]|\[).*)*)",
    re.DOTALL,
)

# Heuristic: correction / numbered-step patterns.
_CORRECTION_RE = re.compile(
    r"(?:correction|corrected version|corrected text)[:\s]+(.+?)(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_NUMBERED_STEP_RE = re.compile(
    r"(?:step\s*1|1\.)\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)


class DataSectionParser:
    """Parse DATA_Template responses and extract the B. slot content."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, response: str) -> Optional[str]:
        """
        Extract B. slot content from a DATA_Template response.

        Primary: regex match on [DATA]...[/DATA] block, extract B. section.
        Fallback: regex match on 'B.' or 'B:' followed by content.
        Returns None if no structured content found or response is empty.

        Requirements: 1.4, 1.7
        """
        if not response or not response.strip():
            return None

        # --- Primary: structured [DATA]...[/DATA] block ---
        block_match = _DATA_BLOCK_RE.search(response)
        if block_match:
            block_content = block_match.group(1)
            result = self._extract_b_from_block(block_content)
            if result is not None:
                return result

        # --- Fallback: bare B. / B: pattern ---
        b_match = _B_BARE_RE.search(response)
        if b_match:
            text = b_match.group(1).strip()
            if text and not _PLACEHOLDER_RE.search(text):
                return text

        return None

    def parse_with_fallback(self, response: str) -> Optional[str]:
        """
        Try structured parse first; fall back to heuristic extraction
        (correction patterns, numbered steps, etc.) if structured fails.

        Requirements: 1.4, 1.7
        """
        if not response or not response.strip():
            return None

        # Try structured parse first.
        structured = self.parse(response)
        if structured is not None:
            return structured

        # Heuristic 1: correction patterns.
        m = _CORRECTION_RE.search(response)
        if m:
            text = m.group(1).strip()
            if text and not _PLACEHOLDER_RE.search(text):
                return text

        # Heuristic 2: numbered steps.
        m = _NUMBERED_STEP_RE.search(response)
        if m:
            text = m.group(1).strip()
            if text and not _PLACEHOLDER_RE.search(text):
                return text

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_b_from_block(self, block_content: str) -> Optional[str]:
        """Extract the B. section from the interior of a [DATA] block."""
        m = _B_SECTION_RE.search(block_content)
        if m:
            text = m.group(1).strip()
            if text and not _PLACEHOLDER_RE.search(text):
                return text
        return None
