# Project Overview: AI RAG Chatbot

A production-ready retrieval-augmented chatbot with per-chat document memory, a FastAPI backend, and a React UI. Each chat session has its own document set and vector store, so context never leaks across conversations.

## What This Project Is For
- Chat over uploaded PDFs with context that stays scoped to a single chat.
- Provide accurate, grounded answers using retrieval + LLM generation.
- Support multi-chat workflows where each conversation has isolated memory.

## Key Capabilities
- Chat-scoped PDF ingestion and vector search.
- Natural-language Q&A with citations.
- JWT authentication and admin seeding.
- Structured JSON logging for traceability.

## Architecture (High Level)
- API layer (FastAPI): auth, chats, messages, uploads.
- Service layer: chat pipeline, prompt building, memory.
- RAG layer: ingestion, embeddings, FAISS per chat.
- LLM layer: Gemini via LangChain.
- UI layer: React + Vite.

## End-to-End Flow
1. User creates a chat session.
2. User uploads PDFs for that chat.
3. Backend chunks PDFs, embeds text, and stores vectors in a chat-scoped FAISS index.
4. User asks a question.
5. Backend retrieves the most relevant chunks for that chat only.
6. Prompt is built with recent chat history plus retrieved context.
7. Gemini generates a grounded answer.
8. Response is returned with citations and stored in the chat history.

## Code Snippets (Where Each Step Happens)

Entry point (API -> service):
```python
# apps/api/api/chat_messages.py
result = handle_chat(
	message=payload.message,
	conversation_id=payload.chat_id,
	document_ids=doc_ids,
	chat_id=payload.chat_id,
	request_id=request_id,
)
```

Retrieval + follow-up enrichment:
```python
# services/chat_service.py
retrieval_query = message

if _is_followup_query(message):
	recent_history = get_history(conversation_id)[-4:]
	history_text = " ".join(
		msg.content
		for msg in recent_history
		if msg.role == "user"
	)
	retrieval_query = f"{history_text} {message}"

contexts = _retrieve_context(retrieval_query, top_k, document_ids, chat_id, request_id)
```

FAISS retrieval:
```python
# services/rag/retriever.py
store = FAISS.load_local(
	str(store_path),
	_get_embeddings(),
	allow_dangerous_deserialization=True,
	index_name="index",
)
```

Prompt building with history + context:
```python
# services/llm/prompt_builder.py
history_lines = []
for msg in chat_history[-6:]:
	role = "User" if msg.role == "user" else "Assistant"
	history_lines.append(f"{role}: {msg.content}")
history_text = "\n".join(history_lines)

parts = [system.strip(), "\n\n"]
if history_text:
	parts += ["Conversation history:\n", history_text, "\n\n"]
```

LLM call:
```python
# services/llm/generator.py
response = llm.invoke([HumanMessage(content=prompt)])
text = (response.content or "").strip()
```

Validation and response shaping:
```python
# services/chat_service.py
is_valid, reason = _validate_answer(raw_answer)
validated_answer = raw_answer if is_valid else RETRIEVAL_EMPTY_ANSWER
```

## Retrieval and Answering
- Retrieval uses MMR to balance relevance and diversity.
- Query expansion improves recall for natural-language questions.
- Prompt routing selects the correct answer style (entity, factual, complex).

## Conversation Memory
- Conversation history is included in the prompt for follow-up questions.
- Follow-up references are resolved using recent history.
- Memory is scoped to the current chat only.

## Data and Storage
- SQLite DB: rag_chat.db
- Uploads: data/uploads/
- Vector stores per chat: data/vector_store/chat_<chat_id>/

## Configuration
Environment variables (backend):
- GEMINI_API_KEY: Gemini API key (required).
- SECRET_KEY: JWT signing key (required).
- ADMIN_PASSWORD: Admin seed password (default: admin).
- LOG_LEVEL: INFO, WARNING, ERROR.
- LOG_FORMAT: json or text.
- HOST: default 0.0.0.0
- PORT: default 8000

## Local Development
Backend:
1. Install dependencies.
2. Set environment variables.
3. Run the API server.

Frontend:
1. Install dependencies.
2. Run the dev server.

See README.md for exact commands and ports.

## API Surface (Core)
- POST /login
- GET /chats, POST /chats, PATCH /chats/{id}, DELETE /chats/{id}
- GET /chats/{id}/messages, POST /chats/{id}/messages
- POST /chats/{id}/upload, GET /chats/{id}/documents, DELETE /documents/{id}

## Logging and Observability
Structured JSON logs are emitted for key steps such as retrieval, LLM calls, and validation. This makes it easier to trace failures and performance bottlenecks.

## Common Questions
- Does context leak between chats? No, vector stores are per chat.
- Are answers grounded? Yes, responses are restricted to retrieved context.
- Can it handle follow-up questions? Yes, recent history is included in prompts.

## When To Use This
- Document Q&A for teams with multiple conversations.
- PDF analysis with strict context isolation.
- Systems needing traceable, auditable outputs.
