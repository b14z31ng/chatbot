"""Retrieval-only answer extractor for simple factual queries.

Extracts entity values directly from retrieved context using regex patterns
and keyword-proximity scanning. Zero Gemini API calls.

Supported query types
---------------------
- Company / organization name
- Cost / price / amount
- Email address
- Phone number
- Date / invoice date
- Document / invoice number
- Address
- Person name (prepared by)
- Generic "what is X" label lookup
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Email
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Phone / mobile — catches various formats
_PHONE_RE = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,5}",
    re.IGNORECASE,
)

# Date patterns
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4})\b",
    re.IGNORECASE,
)

# Currency / cost / price
_COST_RE = re.compile(
    r"(?:USD|BDT|SGD|EUR|GBP|RM|INR|AUD|CAD)?\s*[\$£€₹¥]\s?\d[\d,\.]*"
    r"|\d[\d,\.]*\s*(?:USD|BDT|SGD|EUR|GBP|RM|INR|AUD|CAD|taka|dollars?|euros?|pounds?)",
    re.IGNORECASE,
)

# Document reference number (invoice, project, ref, PO)
_DOC_NUMBER_RE = re.compile(
    r"(?:invoice|inv|project|ref|po|order|quotation|quote|doc|document)\s*(?:no\.?|number|#)?\s*:?\s*([A-Z0-9\-\/]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Label→value proximity scanner
# ---------------------------------------------------------------------------

def _extract_after_label(text: str, labels: list[str], max_chars: int = 80) -> str | None:
    """Find the first occurrence of any label in text and return the value after it.

    Strips common separators (: = -) and returns the remainder up to newline or
    `max_chars` characters.
    """
    t_lower = text.lower()
    for label in labels:
        idx = t_lower.find(label.lower())
        if idx == -1:
            continue
        after = text[idx + len(label):].lstrip(" \t:=–—-")
        # Take up to end of line or max_chars
        value = after.split("\n")[0].strip()[:max_chars].strip(" ,;.")
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Query-type detection helpers
# ---------------------------------------------------------------------------

def _q(query: str) -> str:
    return query.lower()


def _contains(query: str, *words: str) -> bool:
    q = _q(query)
    return any(w in q for w in words)


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------

def extract_answer(query: str, context_text: str) -> str | None:
    """Attempt to extract a factual answer from context without calling Gemini.

    Returns the extracted string if confident, or None to signal the caller
    should fall back to Gemini.
    """
    if not context_text.strip():
        return None

    # ── Email ──────────────────────────────────────────────────────────────
    if _contains(query, "email", "e-mail", "mail"):
        m = _EMAIL_RE.search(context_text)
        if m:
            return m.group(0)

    # ── Phone ──────────────────────────────────────────────────────────────
    if _contains(query, "phone", "telephone", "mobile", "contact number", "fax"):
        # Prefer line containing "phone" / "tel"
        for line in context_text.splitlines():
            if re.search(r"phone|tel|mob|fax|contact", line, re.IGNORECASE):
                m = _PHONE_RE.search(line)
                if m:
                    return m.group(0).strip()
        m = _PHONE_RE.search(context_text)
        if m:
            return m.group(0).strip()

    # ── Date ───────────────────────────────────────────────────────────────
    if _contains(query, "date", "issued", "dated", "when"):
        # Prefer line with "date"
        for line in context_text.splitlines():
            if re.search(r"\bdate\b", line, re.IGNORECASE):
                m = _DATE_RE.search(line)
                if m:
                    return m.group(0).strip()
        m = _DATE_RE.search(context_text)
        if m:
            return m.group(0).strip()

    # ── Cost / price / amount ──────────────────────────────────────────────
    if _contains(query, "cost", "price", "amount", "total", "fee", "charge",
                 "quotation", "quote", "budget", "rate", "value"):
        # Prefer line with "total" or "amount"
        for keyword in ("total", "amount", "price", "cost", "fee", "charge"):
            for line in context_text.splitlines():
                if re.search(keyword, line, re.IGNORECASE):
                    m = _COST_RE.search(line)
                    if m:
                        return m.group(0).strip()
        m = _COST_RE.search(context_text)
        if m:
            return m.group(0).strip()

    # ── Document / invoice number ──────────────────────────────────────────
    if _contains(query, "invoice number", "document number", "reference number",
                 "ref number", "project number", "po number", "order number"):
        m = _DOC_NUMBER_RE.search(context_text)
        if m:
            return m.group(1).strip()

    # ── Company / organization / client name ──────────────────────────────
    if _contains(query, "company", "organization", "organisation", "firm",
                 "business", "company name", "client", "client name",
                 "vendor", "supplier", "customer"):
        value = _extract_after_label(
            context_text,
            ["company:", "company name:", "organization:", "organisation:",
             "prepared by:", "submitted by:", "from:", "client:",
             "client name:", "prepared for:", "for:", "to:"],
            max_chars=120,
        )
        if value:
            return value.split("\n")[0].strip()

    # ── Person / prepared by ───────────────────────────────────────────────
    if _contains(query, "prepared by", "who prepared", "submitted by",
                 "authored by", "created by", "who created", "who wrote"):
        value = _extract_after_label(
            context_text,
            ["prepared by:", "submitted by:", "authored by:", "from:"],
        )
        if value:
            return value.split("\n")[0].strip()

    # ── Address ────────────────────────────────────────────────────────────
    if _contains(query, "address", "location", "office", "city", "country",
                 "zip", "postal"):
        value = _extract_after_label(
            context_text,
            ["address:", "location:", "office address:", "city:", "country:"],
            max_chars=150,
        )
        if value:
            return value

    # ── Generic "what is the X" → label scan ──────────────────────────────
    # e.g. "what is the project name" → look for "project name: ..."
    generic_match = re.search(
        r"what\s+is\s+(?:the\s+|a\s+)?(.+?)(?:\?|$)", _q(query)
    )
    if generic_match:
        label = generic_match.group(1).strip().rstrip("?")
        value = _extract_after_label(context_text, [label + ":"])
        if value:
            return value.split("\n")[0].strip()

    # Nothing found — signal caller to use Gemini fallback
    return None
