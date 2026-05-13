"""Compact prompt builder with conversational continuity support."""

from services.llm.memory import Message
from services.rag.retriever import ContextChunk


_SIMPLE_SYSTEM = """
You are a strict document question-answering assistant.

Answer ONLY from the provided document context.

Rules:
- Keep answers concise and factual.
- Return exact names and values from the document.
- Do not hallucinate.
- If the answer is missing, say:
'I cannot answer this from the provided context.'
"""


_FACTUAL_SYSTEM = """
You are a document question-answering assistant.

Answer ONLY using the provided context.

Rules:
- Provide complete explanations.
- Keep answers concise but COMPLETE.
- Never cut sentences halfway.
- Use 2-5 sentences when necessary.
- Use exact names and facts from the document.
- If the answer is missing, say:
'I cannot answer this from the provided context.'
"""


_COMPLEX_SYSTEM = """
You are a document analysis assistant.

Answer ONLY using the provided document context.

Rules:
- Provide complete and natural explanations.
- Use multiple sentences when needed.
- Summarize clearly and professionally.
- Explain the purpose, scope, companies, services, and goals if present.
- Do NOT shorten answers unnecessarily.
- Do NOT return sentence fragments.
- Use exact entity names from the document.
- If the answer is missing, say:
'I cannot answer this from the provided context.'
"""


_CONTINUATION_NOTE = (
    "Note: 'they', 'it', 'this', 'the company', 'the document' refer to the "
    "uploaded document and its subject matter discussed in the conversation above."
)


def _has_pronoun_reference(query: str) -> bool:
    """Return True when query contains reference pronouns needing prior-turn resolution."""
    q = " " + query.lower() + " "

    pronouns = (
        " they ", " their ", " them ", " theirs ",
        " it ", " its ",
        " this ", " these ", " that ", " those ",
    )

    return any(p in q for p in pronouns)


def _has_followup_reference(query: str) -> bool:
    q = f" {query.lower()} "

    refs = [
        " it ",
        " this ",
        " that ",
        " they ",
        " them ",
        " its ",
        " their ",
        " incident ",
        " event ",
        " movement ",
    ]

    return any(r in q for r in refs)


def build_prompt(
    contexts: list[ContextChunk],
    user_query: str,
    chat_history: list[Message],
    mode: str = "factual",
    **kwargs,
) -> str:
    """
    Build a grounded prompt with conversational continuity.
    """

    q_lower = user_query.lower()

    is_date_query = any(
        x in q_lower
        for x in [
            "when",
            "what year",
            "which year",
            "what date",
            "date of",
            "year of",
            "timeline",
            "occurred",
            "happened",
        ]
    )

    if "is_complex" in kwargs:
        mode = "complex" if kwargs["is_complex"] else "entity"

    if is_date_query:
        system = (
            "Answer ONLY using the provided context. "
            "For date or year questions, return the exact year, date, or time period explicitly mentioned in the document. "
            "Do not shorten the answer before including the actual date or year. "
            "Use complete factual sentences."
        )
    elif mode == "entity":
        system = _SIMPLE_SYSTEM
    elif mode == "factual":
        system = _FACTUAL_SYSTEM
    else:
        system = _COMPLEX_SYSTEM

    # =========================
    # CONTEXT BUILDING
    # =========================
    if contexts:
        if mode == "complex":
            context_text = "\n\n---\n\n".join(
                c.content for c in contexts[:5]
            )
        else:
            context_text = contexts[0].content
    else:
        context_text = "No relevant context retrieved."

    # =========================
    # CHAT HISTORY
    # =========================
    history_lines: list[str] = []

    for msg in chat_history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.content}")

    history_text = "\n".join(history_lines)

    # =========================
    # PROMPT ASSEMBLY
    # =========================
    parts: list[str] = [system.strip(), "\n\n"]

    if history_text:
        parts += [
            "Conversation history:\n",
            history_text,
            "\n\n",
        ]

    if history_text and _has_followup_reference(user_query):
        parts += [
            "Follow-up question detected. "
            "Resolve references like 'it', 'that', 'this incident', "
            "'they', or 'the movement' using the conversation history first.\n\n",
        ]

    parts += [
        "Context from uploaded document:\n",
        context_text,
        "\n\n",
        "Question:\n",
        user_query,
        "\n\n",
        "Answer:",
    ]

    return "".join(parts)


def _is_summary_query(query: str) -> bool:
    """Legacy alias — prefer intent.is_complex_query() for new code."""
    from services.llm.intent import is_complex_query
    return is_complex_query(query)