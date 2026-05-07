# AI RAG Chatbot Backend

A secure, production-ready RAG chatbot backend with grounded LLM responses and role-based access control.

## Project Overview
A production-grade backend for a retrieval-augmented chatbot. The API retrieves context from an internal knowledge base, builds a grounded prompt, and generates an answer with strict no-hallucination rules.

## Features
- PDF ingestion with chunking
- Embeddings and FAISS vector search
- Grounded LLM responses with strict fallback
- Short-term conversation memory
- JWT authentication with admin-only uploads
- OpenAPI docs at `/docs`

## Architecture

The system follows a layered RAG architecture:

- API Layer (FastAPI): Handles HTTP requests and authentication
- Service Layer: Business logic (chat, auth)
- RAG Layer: Retrieval (ingestion, embeddings, FAISS)
- LLM Layer: Prompt building and grounded answer generation

## Tech Stack
- FastAPI
- sentence-transformers
- FAISS (faiss-cpu)
- OpenAI (gpt-4o-mini)
- pypdf
- python-jose + passlib

## Quick Start

```bash
git clone <repo-url>
cd ai-chatbot
pip install -r requirements.txt
cp .env.example .env
uvicorn apps.api.main:app --reload
```

## Setup Instructions
1. Install deps:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and update all required values before running.
3. Run API:
   ```bash
   python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Open Swagger UI:
   - http://localhost:8000/docs

## Environment Variables
| Variable | Description |
| --- | --- |
| `SECRET_KEY` | JWT signing key. Must be set. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime in minutes. |
| `ADMIN_PASSWORD` | Admin password; hashed at startup for in-memory store. |
| `APP_NAME` | Application name for FastAPI. |
| `LOG_LEVEL` | Logging level (INFO, WARNING, ERROR). |
| `LOG_FORMAT` | Logging format (`json` or `text`). |
| `HOST` | Server host. |
| `PORT` | Server port. |

## Authentication Flow
1. POST `/login` with username/password.
2. Receive `access_token`.
3. Use `Authorization: Bearer <token>` on `/chat` and `/upload`.
4. `/upload` requires admin role.

⚠️ IMPORTANT: Change the default admin password before running in production.

## API Endpoints

| Endpoint | Method | Auth | Description |
|---------|--------|------|------------|
| /login | POST | No | Authenticate and get JWT |
| /chat | POST | Yes | Ask grounded question |
| /upload | POST | Admin | Upload and ingest PDF |

## API Usage
### Login
```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}'
```

### Chat (requires token)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"message":"What is this document about?","conversation_id":"conv-1"}'
```

Example response:
```json
{
  "answer": "Not in knowledge base.",
  "citations": [],
  "confidence": 0.0,
  "conversation_id": "conv-1"
}
```

### Upload (admin-only)
```bash
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@path/to/doc.pdf"
```

## Folder Structure
```
/apps
  /api
    /api
      auth.py
      chat.py
      upload.py
    config.py
    logger.py
    main.py
    schemas.py
/services
  /auth
  /llm
  /rag
/tests
/docs
```

## Frontend Usage

The React frontend is in the `frontend/` directory.

### Local Development
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — frontend connects to backend at `http://localhost:8000`.

### Environment
Set `VITE_API_URL` in `frontend/.env` to change the backend URL:
```
VITE_API_URL=http://localhost:8000
```

## Deployment

### Docker Compose (recommended)
```bash
# Start both backend + frontend
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Backend Only
```bash
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

### Environment Variables
All secrets are loaded from `.env`. Never hardcode secrets. Required:
- `OPENROUTER_API_KEY` — LLM provider key
- `SECRET_KEY` — JWT signing key
- `ADMIN_PASSWORD` — Admin user password

## Demo Flow

1. **Login**: Open frontend → enter `admin` / your admin password → get JWT
2. **Chat**: Ask a question → system retrieves context → LLM generates grounded answer
3. **Upload** (API only): `POST /upload` with PDF → admin-only document ingestion
4. **Observe**: Check structured JSON logs for `chat.metrics`, `request.summary`

## Performance Notes

- **Free API Limits**: OpenRouter free-tier models have rate limits (requests/min, tokens/day). Monitor `llm.retry` and `llm.error` logs for throttling.
- **Fallback Behavior**: When LLM fails after retry or context retrieval returns nothing, the system returns `"Not in knowledge base."` with `confidence: 0.0` — no hallucination.
- **Caching**: In-memory LRU cache (20 entries) deduplicates identical queries against the same context. Cache hit rate logged in `chat.metrics`.
- **Rate Limiting**: API enforces 30 requests/minute per IP (in-memory). Returns `429` when exceeded.
- **LLM Timeout**: LLM calls timeout at 20s with 1 automatic retry before failing gracefully.

## Limitations
- In-memory vector store (no persistence).
- In-memory user store (single admin user).
- No refresh tokens.
- No multi-tenant isolation.

## Future Improvements
- Persist FAISS index to disk or managed vector DB.
- User management and role provisioning.
- Refresh tokens and token rotation.
- Retrieval evaluation and monitoring.
