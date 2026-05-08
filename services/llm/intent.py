"""Query intent classifier — routes queries to retrieval-only or Gemini pipelines.

Classification is purely keyword/pattern-based (zero latency, zero API cost).

Taxonomy
--------
SIMPLE   →  factual lookup (entity, value, date, contact info, number)
COMPLEX  →  reasoning, synthesis, summarization, explanation, comparison

Design goal: minimize false-positives (classifying complex queries as simple),
since a missed complex query just adds a Gemini call, whereas a missed simple
query wastes a free extraction opportunity.
"""
from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Triggers for COMPLEX mode (Gemini required)
_COMPLEX_TRIGGERS: frozenset[str] = frozenset([
    # Summarization
    "summarize", "summary", "summarise", "summarization",
    # Explanation / analysis
    "explain", "explanation", "analyze", "analyse", "analysis",
    "elaborate", "elaboration", "breakdown", "break down",
    # Description
    "describe", "description", "overview", "outline",
    # Comparison / synthesis
    "compare", "comparison", "contrast", "versus", "vs",
    "synthesize", "synthesis",
    # Detail requests
    "detailed", "in depth", "in-depth", "comprehensive", "complete", "full",
    "details", "tell me about", "tell me more",
    # Opinion / evaluation
    "evaluate", "assessment", "assess", "review",
    "main points", "key points", "key findings",
    "what does this mean", "what are the main",
    "give me a", "provide a", "write a",
    # Document-type phrase triggers (added for summary routing fix)
    "what is this",
    "what is the pdf",
    "what does this document",
    "what does this agreement",
    "what does this quotation",
    "what does this proposal",
    "project",
    "agreement",
    "proposal",
    "contract",
    "scope",
    "purpose",
])

# Triggers for SIMPLE mode (regex extraction — no Gemini)
_SIMPLE_TRIGGERS: frozenset[str] = frozenset([
    # Identity
    "company name", "company", "organization", "organisation",
    "prepared by", "prepared", "who made", "who created", "who wrote",
    # Financial
    "cost", "price", "amount", "total", "fee", "charge", "invoice",
    "quotation", "quote", "budget", "rate", "value",
    # Contact
    "email", "e-mail", "phone", "telephone", "mobile", "fax",
    "address", "location", "office", "city", "country", "zip", "postal",
    # Document metadata
    "date", "issued", "dated", "version", "number", "reference", "ref",
    "invoice number", "document number", "project number",
    # Named entity lookups
    "what is the", "what is", "who is",
    "what was", "when was", "where is", "when is",
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

QueryType = Literal["simple", "complex"]


def classify_query(query: str) -> QueryType:
    """Classify a query as 'simple' (retrieval-only) or 'complex' (Gemini).

    Priority: complex keywords take precedence over simple ones to avoid
    degrading quality on ambiguous queries.
    """
    q = query.lower().strip()

    # Complex wins if any trigger present
    if any(kw in q for kw in _COMPLEX_TRIGGERS):
        return "complex"

    # Simple if any simple trigger present
    if any(kw in q for kw in _SIMPLE_TRIGGERS):
        return "simple"

    # Short queries (≤ 6 words) without complex markers → treat as simple
    word_count = len(q.split())
    if word_count <= 6:
        return "simple"

    # Default: complex (safer — uses Gemini, preserves quality)
    return "complex"


def is_complex_query(query: str) -> bool:
    """Return True if the query requires Gemini generation."""
    return classify_query(query) == "complex"


def top_k_for_query(query: str) -> int:
    """Return the ideal FAISS retrieval depth for this query type."""
    return 12 if is_complex_query(query) else 8


def max_tokens_for_query(query: str) -> int:
    """Return the ideal Gemini max_output_tokens for this query type."""
    return 400 if is_complex_query(query) else 150
