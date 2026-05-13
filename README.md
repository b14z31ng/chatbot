# AI RAG Chatbot

Production-ready retrieval-augmented chatbot with chat-scoped document memory, structured logging, and a React UI.

## Overview
This project provides a FastAPI backend and a React frontend for a multi-chat RAG experience. Each chat keeps its own document set and vector store, so context never leaks across conversations.

## Features
- Chat-scoped PDF uploads and document memory
- LangChain-powered PDF chunking + FAISS vector search
- Universal query expansion for better retrieval recall
- Conversational follow-up handling using recent chat history
- Gemini 2.5 Flash responses with safe fallbacks
- JWT authentication and admin seeding
- Structured JSON logging
- React UI with multi-chat UX

## Architecture
- API layer (FastAPI): auth, chats, messages, uploads
- Service layer: chat pipeline, prompt building, memory
- RAG layer: ingestion, embeddings, FAISS per chat
- LLM layer: Gemini via LangChain

## Answer Routing
The backend now routes questions into three answer styles:
- Entity: short metadata lookups such as company name, invoice number, or prepared by.
- Factual: complete but concise answers for general factual questions.
- Complex: longer answers for WH, explanation, summary, and follow-up questions.

Typical token budgets:
- Entity: 80 tokens
- Factual: 220 tokens
- Complex: 500 tokens

This helps prevent mid-sentence truncation for date, history, and explanation questions while keeping metadata answers short.

## Tech Stack
- FastAPI, SQLAlchemy (SQLite)
- LangChain, FAISS (faiss-cpu)
- sentence-transformers (all-MiniLM-L6-v2)
- Google Gemini (langchain-google-genai)
- React + Vite

## Quick Start
Backend:
```bash
python -m pip install -r requirements.txt
set GEMINI_API_KEY=your_key
set SECRET_KEY=your_secret
set ADMIN_PASSWORD=your_admin_password
python -m uvicorn apps.api.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Open:
- API docs: http://localhost:8000/docs
- Frontend: http://localhost:5173

## Configuration
Environment variables (backend):

| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | Gemini API key (required). |
| `SECRET_KEY` | JWT signing key (required). |
| `ADMIN_PASSWORD` | Admin seed password (default: admin). |
| `LOG_LEVEL` | Logging level (INFO, WARNING, ERROR). |
| `LOG_FORMAT` | Logging format (`json` or `text`). |
| `HOST` | Server host (default: 0.0.0.0). |
| `PORT` | Server port (default: 8000). |

Frontend API base URL is set in [frontend/src/services/api.js](frontend/src/services/api.js).

## Data and Storage
- SQLite DB: `rag_chat.db`
- Uploads: `data/uploads/`
- Vector stores: `data/vector_store/chat_<chat_id>/index.faiss` + `index.pkl`

## Conversation Handling
- Recent chat history is included in prompts so follow-up questions can resolve references like "it", "that incident", or "they".
- Retrieval is also enriched with recent same-chat user turns for follow-up questions, improving recall without leaking memory across chats.
- Memory stays scoped to the active conversation only.

## API Endpoints (Core)
- `POST /login`
- `GET /chats`, `POST /chats`, `PATCH /chats/{id}`, `DELETE /chats/{id}`
- `GET /chats/{id}/messages`, `POST /chats/{id}/messages`
- `POST /chats/{id}/upload`, `GET /chats/{id}/documents`, `DELETE /documents/{id}`
- Legacy: `POST /chat`, `POST /upload` (admin)

Use `/docs` for the full OpenAPI spec.

## Logging
Structured JSON logs are written to stdout. Key events include:
- `chat.raw_gemini_answer`
- `chat.validator`
- `chat.final_answer_before_response`
- `chat.response_payload`

## Recent Changes
- Improved intent routing so WH questions are treated as complex questions instead of ultra-short entity lookups.
- Added follow-up query enrichment before retrieval.
- Increased token budgets for factual and complex answers to reduce truncation.
- Pinned `bcrypt` to `4.0.1` for compatibility with `passlib[bcrypt]`.

## Deployment
Docker Compose:
```bash
docker-compose up --build
```

Backend only:
```bash
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

## Notes
- Change the default admin password before production use.
- Per-chat vector stores isolate document context between conversations.
