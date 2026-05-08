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


def build_prompt(
    contexts: list[ContextChunk],
    user_query: str,
    chat_history: list[Message],
    is_complex: bool = True,
) -> str:
    """
    Build a grounded prompt with conversational continuity.
    """

    system = _COMPLEX_SYSTEM if is_complex else _SIMPLE_SYSTEM

    # =========================
    # CONTEXT BUILDING
    # =========================
    if contexts:
        if is_complex:
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
        prefix = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {msg.content}")

    history_text = "\n".join(history_lines)

    # =========================
    # PROMPT ASSEMBLY
    # =========================
    parts: list[str] = [system.strip(), "\n\n"]

    if history_text:
        parts += [
            "Conversation so far:\n",
            history_text,
            "\n\n",
        ]

    if history_text and _has_pronoun_reference(user_query):
        parts += [
            _CONTINUATION_NOTE,
            "\n\n",
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