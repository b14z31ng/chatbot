from dataclasses import dataclass


@dataclass
class QueryIntent:
    mode: str
    top_k: int
    max_tokens: int


ENTITY_PATTERNS = [
    "company name",
    "client name",
    "customer name",
    "invoice number",
    "quotation number",
    "reference number",
    "email address",
    "phone number",
    "contact number",
    "issue date",
    "due date",
    "vendor name",
    "organization name",
    "prepared by",
]

COMPLEX_TERMS = [
    "summary",
    "summarize",
    "explain in detail",
    "describe",
    "overview",
    "details",
    "scope",
    "purpose",
    "what is the pdf about",
    "what does this document say",
    "agreement",
    "proposal",
    "contract",
]

def classify_query(query: str) -> QueryIntent:
    query_lower = query.lower().strip()

    if any(p in query_lower for p in ENTITY_PATTERNS):
        return QueryIntent(
            mode="entity",
            top_k=3,
            max_tokens=80,
        )

    if any(query_lower.startswith(w) for w in [
        "what",
        "when",
        "why",
        "how",
        "who",
        "where",
        "which",
        "explain",
        "describe",
        "summarize",
    ]):
        return QueryIntent(
            mode="complex",
            top_k=12,
            max_tokens=500,
        )

    return QueryIntent(
        mode="factual",
        top_k=6,
        max_tokens=220,
    )


def is_complex_query(query: str) -> bool:
    return classify_query(query).mode == "complex"


def top_k_for_query(query: str) -> int:
    return classify_query(query).top_k


def max_tokens_for_query(query: str) -> int:
    return classify_query(query).max_tokens