from services.llm.memory import Message
from services.rag.retriever import ContextChunk




def build_prompt(
    contexts: list[ContextChunk],
    user_query: str,
    chat_history: list[Message],
) -> str:
    """Build a grounded prompt with context and conversation history."""

    context_text = contexts[0].content if contexts else "None"

    return (
        "You are a strict QA system.\n"
        "Answer ONLY using the provided context.\n"
        "Do NOT use outside knowledge.\n\n"

        "If the answer is not clearly present, say:\n"
        "'Not in knowledge base.'\n\n"

        "Give a short, clean answer.\n\n"

        f"Context:\n{context_text}\n\n"
        f"Question:\n{user_query}\n\n"
        "Answer:"
    )
