# RAG Chatbot — Full Project Audit Report

**Audit Date:** 2026-05-07  
**System:** AI RAG Chatbot (FastAPI + FAISS + Gemini 2.0 Flash)  
**Status:** Running (uvicorn active)

---

## 1. Project Structure

```
e:/ML XAI/
├── .env                          ← Active secrets (GEMINI_API_KEY exposed!)
├── .env.example                  ← Outdated — still references OPENROUTER
├── .gitignore / .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_rag.py                   ← Root-level script, not in /tests
├── data/
│   ├── test_doc.pdf
│   └── uploads/                  ← PDF uploads stored here
├── apps/
│   └── api/
│       ├── config.py
│       ├── logger.py
│       ├── main.py               ← FastAPI app factory
│       ├── schemas.py
│       └── api/
│           ├── auth.py           ← POST /login
│           ├── chat.py           ← POST /chat
│           └── upload.py         ← POST /upload
├── services/
│   ├── chat_service.py           ← Core orchestration layer
│   ├── auth/
│   │   ├── auth_service.py
│   │   ├── dependencies.py
│   │   └── jwt_handler.py
│   ├── llm/
│   │   ├── generator.py          ← Gemini 2.0 Flash (google-genai)
│   │   ├── memory.py             ← In-memory conversation history
│   │   └── prompt_builder.py
│   ├── rag/
│   │   ├── embeddings.py         ← all-MiniLM-L6-v2 (sentence-transformers)
│   │   ├── ingestion.py          ← PDF chunker
│   │   ├── retriever.py          ← Query rewrite + FAISS search
│   │   └── vector_store.py       ← In-memory FAISS IndexFlatL2
│   └── ingestion/                ← EMPTY DIRECTORY
├── tests/
│   ├── unit/                     ← EMPTY
│   ├── integration/              ← EMPTY
│   └── e2e/
├── frontend/                     ← Vite + React
│   └── src/
│       ├── App.jsx
│       ├── components/
│       └── services/
├── docs/
└── raag/                         ← Unknown purpose (likely leftover)
```

---

## 2. Issues by Severity

---

### 🔴 CRITICAL

#### C-1: Real API Key in `.env`
**File:** `.env` line 9  
**Issue:** `GEMINI_API_KEY=AIzaSyDvF...` is a live Google API key stored in plaintext. If ever committed to Git, it is fully exposed.  
**Risk:** Unauthorized billing, account compromise, key theft.  
**Fix:** Rotate the key immediately via Google Cloud Console. Verify `.env` is not tracked by git (`git status --short`).

#### C-2: Ephemeral Vector Store — Data Lost on Restart
**File:** `services/rag/vector_store.py` line 73-74  
**Issue:** `_STORE = InMemoryVectorStore()` — the FAISS index and all ingested documents exist only in RAM. Every server restart, crash, or redeployment wipes all uploaded PDFs.  
**Risk:** Complete data loss on any restart.  
**Fix:** Persist the FAISS index with `faiss.write_index()` and the `texts`/`metadata` arrays with `pickle` or `json`. Reload from disk on startup.

#### C-3: Hardcoded Single Admin User with Default Password
**File:** `services/auth/auth_service.py` lines 9–19  
**Issue:** `ADMIN_PASSWORD` defaults to `"admin"`. Your `.env` also sets it to `"admin"`. Only one hardcoded user exists with no way to add users without code changes.  
**Risk:** Trivially brute-forceable. No multi-user support.  
**Fix:** Move users to a persistent store (SQLite minimum). Enforce password complexity validation.

#### C-4: No Input Sanitization on Chat Messages
**File:** `apps/api/api/chat.py`, `services/chat_service.py`  
**Issue:** Raw user messages are injected directly into LLM prompts with no length cap, no character filtering, and no injection protection.  
**Risk:** Prompt injection attacks, runaway token costs, context flooding.  
**Fix:** Add `max_length=1000` validator on `ChatRequest.message`. Strip control characters before prompt assembly.

---

### 🟠 HIGH

#### H-1: `main.py` Reports Wrong LLM Provider and Model
**File:** `apps/api/main.py` lines 79–83, 88, 111–112  
**Issue:** Code still checks for `OPENROUTER_API_KEY`, logs `llm_model: "gemma-2b-it (huggingface)"`, and the `/health` endpoint returns `"llm_provider": "huggingface"`. The actual backend is **Gemini 2.0 Flash** via `google-genai`.  
**Fix:** Check `GEMINI_API_KEY`, log `"gemini-2.0-flash"`, set `llm_provider: "google"`.

#### H-2: `.env.example` References Wrong Provider
**File:** `.env.example`  
**Issue:** Still contains `OPENROUTER_API_KEY` and `LLM_MODEL=mistral:7b-instruct`. Any new developer will configure the wrong provider entirely.  
**Fix:** Replace with `GEMINI_API_KEY=your_key_here` and remove `LLM_MODEL`.

#### H-3: `retrieve_context()` Silently Ignores Its `top_k` Parameter
**File:** `services/chat_service.py` line 88  
**Issue:** `retrieve_context()` accepts `top_k` as a parameter but internally hardcodes `top_k=5`, ignoring the argument entirely. The `retrieve()` function defaults to `top_k=10`. Silent mismatch.  
**Fix:** Pass `top_k` through: `results = retrieve(query, top_k=top_k)`.

#### H-4: FAISS L2 Distance Used as Similarity Score (Inverted Logic)
**File:** `services/chat_service.py` lines 32–43  
**Issue:** `v = c.score` is a raw L2 **distance** (lower = better). The hybrid formula `0.6 * v` treats higher distances as more relevant, so the *worst* FAISS matches get the highest score.  
**Fix:** `v = 1 / (1 + c.score)` to convert distance to similarity before scoring.

#### H-5: Two Conflicting Grounding Check Functions
**File:** `services/chat_service.py` lines 135–154 and 157–158  
**Issue:** `_is_answer_grounded()` (token regex, 3+ chars) and `is_grounded()` (word membership) both exist and are called at different points. They return different results for the same input.  
**Fix:** Delete one. Consolidate into a single grounding function used throughout.

#### H-6: `openai` in `requirements.txt` but Unused; `google-genai` Missing
**File:** `requirements.txt`  
**Issue:** `openai>=1.50.0` is listed but never imported. `google-genai` (actually used in `generator.py`) is not listed at all. A fresh `pip install -r requirements.txt` will crash at runtime.  
**Fix:** Remove `openai`. Add `google-genai`.

#### H-7: CORS Fully Open with Credentials
**File:** `apps/api/main.py` lines 204–209  
**Issue:** `allow_origins=["*"]` combined with `allow_credentials=True` is both a browser-spec violation and a security hole — any website can call the API.  
**Fix:** Restrict to explicit origins: `["http://localhost:5173", "http://localhost:3000"]`.

---

### 🟡 MEDIUM

#### M-1: Chat History Collected but Never Used in Prompt
**File:** `services/llm/prompt_builder.py`  
**Issue:** `build_prompt()` receives `chat_history` but never inserts it into the returned string. Conversation memory is silently discarded.  
**Fix:** Include recent history in the prompt, or remove the parameter from the signature.

#### M-2: Hard-coded `rewrite_query()` Rules
**File:** `services/rag/retriever.py` lines 20–29  
**Issue:** Keyword rewrites only fire for exact phrases like `"company"` + `"name"`. Queries like "who is the vendor?" or "which firm made this?" bypass rewriting entirely.  
**Fix:** Remove or generalize. Either trust the embedding model or use LLM-based query expansion.

#### M-3: Domain-Specific `header_boost()` and `lexical_score()` Hacks
**File:** `services/chat_service.py` lines 11–26  
**Issue:** Boost rules contain `"prepared by"`, `"quotation"`, `"project price"` — hardcoded for one specific PDF. These will misfire on any other document type.  
**Fix:** Remove document-specific rules. Rely on vector similarity + LLM reasoning.

#### M-4: Conversation Memory Has No TTL or Persistence
**File:** `services/llm/memory.py`  
**Issue:** `_STORE` is a module-level dict with no expiry. Old sessions accumulate forever, and everything is wiped on restart.  
**Fix:** Add TTL cleanup. Use Redis or SQLite for production.

#### M-5: No Duplicate Document Detection on Upload
**File:** `apps/api/api/upload.py`  
**Issue:** Every upload generates a new UUID and ingests independently. Uploading the same PDF twice doubles all chunks in the vector store, causing duplicated and inconsistent retrieval results.  
**Fix:** SHA-256 hash the file content and check for existing documents before ingesting.

#### M-6: `test_rag.py` Is a Manual Script, Not a Test
**File:** `test_rag.py`  
**Issue:** Root-level script with hardcoded `admin/admin` credentials, no assertions, no cleanup, and requires a live server. Not executable by `pytest`.  
**Fix:** Move to `tests/integration/`. Rewrite as proper `pytest` tests with fixtures and assertions.

#### M-7: `services/ingestion/` Is an Empty Directory
**Fix:** Delete it. Actual ingestion logic is in `services/rag/ingestion.py`.

#### M-8: `raag/` Directory Has No Purpose
**Fix:** Delete it (likely a typo artifact of `rag/`).

#### M-9: Cache Hits Return Wrong Confidence Score
**File:** `services/chat_service.py` line 259  
**Issue:** `confidence = min(1.0, 0.3 * len(contexts))` — cached answers always show 30% confidence. Fresh LLM answers show 85%. A cached *correct* answer appears less reliable than a new one.  
**Fix:** Store confidence alongside the cached answer and return it directly on cache hit.

---

### 🔵 LOW / HOUSEKEEPING

#### L-1: Production Code Flooded with `print()` Statements
Raw `print()` calls in `vector_store.py`, `retriever.py`, `chat_service.py`, and `generator.py` dump internal state to stdout on every single request.  
**Fix:** Replace all `print()` with `logger.debug()`. Gate on `LOG_LEVEL=DEBUG`.

#### L-2: `google-genai` Not in `requirements.txt` — Docker Will Fail at Runtime
The Dockerfile installs from `requirements.txt` which is missing `google-genai`. The container will start, then crash on first request.

#### L-3: Zero Automated Test Coverage
All three test directories (`unit/`, `integration/`, `e2e/`) are completely empty. No pytest tests exist anywhere.

#### L-4: `prompt_builder.py` Has Cosmetic Blank Lines
Lines 3–6 have three consecutive blank lines — minor, but signals accumulated edits without cleanup.

#### L-5: `SECRET_KEY` in `.env`
`SECRET_KEY=65012074ee9b903e...` signs all JWTs. Verify this has never been committed to any git branch. If it has, rotate it (invalidates all current tokens).

---

## 3. Summary Table

| ID | Severity | Area | Issue |
|----|----------|------|-------|
| C-1 | 🔴 Critical | Security | Live Gemini API key in `.env` |
| C-2 | 🔴 Critical | Data | Ephemeral vector store — data lost on restart |
| C-3 | 🔴 Critical | Auth | Single hardcoded user, default `admin` password |
| C-4 | 🔴 Critical | Security | No chat input sanitization or length limit |
| H-1 | 🟠 High | Config | `main.py` reports wrong LLM provider/model |
| H-2 | 🟠 High | Config | `.env.example` references wrong provider |
| H-3 | 🟠 High | Logic | `top_k` silently ignored in `retrieve_context()` |
| H-4 | 🟠 High | Logic | L2 distance used as similarity — inverted ranking |
| H-5 | 🟠 High | Logic | Two conflicting grounding check functions |
| H-6 | 🟠 High | Dependencies | `openai` unused; `google-genai` missing from requirements |
| H-7 | 🟠 High | Security | CORS fully open (`*` + credentials) |
| M-1 | 🟡 Medium | Logic | Chat history never injected into prompt |
| M-2 | 🟡 Medium | Retrieval | Hard-coded fragile `rewrite_query()` rules |
| M-3 | 🟡 Medium | Retrieval | Domain-specific `header_boost()` hacks |
| M-4 | 🟡 Medium | Memory | No TTL or persistence for conversation store |
| M-5 | 🟡 Medium | Upload | No duplicate document detection |
| M-6 | 🟡 Medium | Testing | `test_rag.py` is a manual script, not a test |
| M-7 | 🟡 Medium | Structure | `services/ingestion/` is empty |
| M-8 | 🟡 Medium | Structure | `raag/` directory has no purpose |
| M-9 | 🟡 Medium | Logic | Cache hits return wrong confidence score |
| L-1 | 🔵 Low | Code Quality | `print()` debug statements in production code |
| L-2 | 🔵 Low | Docker | `google-genai` missing — container fails at runtime |
| L-3 | 🔵 Low | Testing | Zero automated test coverage |
| L-4 | 🔵 Low | Code Quality | Cosmetic blank lines in prompt_builder.py |
| L-5 | 🔵 Low | Security | `SECRET_KEY` in `.env` — verify never committed |

---

## 4. Immediate Action Plan (Priority Order)

1. 🔴 **Rotate** `GEMINI_API_KEY` in Google Cloud Console right now
2. 🟠 **Add `google-genai`** to `requirements.txt`, remove `openai`
3. 🟠 **Fix `main.py`**: check `GEMINI_API_KEY`, update model/provider labels
4. 🟠 **Update `.env.example`** to use `GEMINI_API_KEY`
5. 🔴 **Persist FAISS index** to disk (fix data loss on restart)
6. 🟠 **Fix L2 distance bug** in `pick_best_context()`: `v = 1 / (1 + c.score)`
7. 🔴 **Add input length limit** to `ChatRequest` schema
8. 🟠 **Consolidate grounding** into one function, remove the duplicate
9. 🔵 **Replace all `print()`** with `logger.debug()` 
10. 🟡 **Remove empty dirs**: `services/ingestion/`, `raag/`
