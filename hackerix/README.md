# Insurance Policy Analysis API

A FastAPI service that analyses insurance policy documents (PDF, DOCX, EML) and answers
natural-language questions about them using **Google Gemini** and **LangChain RAG** pipelines.

## Architecture

```
Client → POST /hackrx/run
           │
           ▼
   [Auth — Bearer token]
           │
           ▼
   [Download document from URL]        ← aiohttp
           │
           ▼
   [Parse document into text]          ← PyPDF / UnstructuredWord / UnstructuredEmail
           │
           ▼
   [Chunk text]                        ← RecursiveCharacterTextSplitter
           │
           ▼
   [Embed chunks → FAISS vector store] ← HuggingFace all-MiniLM-L6-v2
           │
           ▼
   [Retrieve top-k relevant chunks]    ← FAISS similarity search
           │
           ▼
   [Generate structured answer]        ← Google Gemini 1.5 Flash
           │
           ▼
   Structured JSON response
```

## Features

- 📄 **Multi-format** — PDF, DOCX, and EML/MSG email documents
- 🔍 **Semantic search** — FAISS vector store with HuggingFace embeddings
- 🤖 **Gemini-powered** — structured JSON answers with decision, amount, justification, and clause mapping
- 🔒 **Bearer token auth** — protected endpoint
- ⚡ **Async** — non-blocking document download and processing
- 🧹 **Auto cleanup** — temporary files deleted after each request

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/hackerix.git
cd hackerix
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY and BEARER_TOKEN
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey

### 3. Run

```bash
uvicorn app:app --reload
```

API docs (Swagger UI): http://localhost:8000/docs

### 4. Test

```bash
export BEARER_TOKEN=your_token_here
python test_api.py
```

## API Reference

### `POST /hackrx/run`

Analyses a policy document and answers questions.

**Headers**
```
Authorization: Bearer <BEARER_TOKEN>
Content-Type: application/json
```

**Request body**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "Is knee surgery covered under the policy?"
  ]
}
```

**Response**
```json
{
  "success": true,
  "answers": [
    {
      "question": "What is the waiting period for pre-existing diseases?",
      "decision": "covered",
      "amount": null,
      "justification": "Pre-existing diseases are covered after a 48-month waiting period as per clause 4.1.",
      "clause_mapping": [
        {
          "clause_text": "Pre-existing diseases shall be covered after 48 months of continuous coverage.",
          "source": "policy.pdf"
        }
      ]
    }
  ],
  "metadata": {
    "processing_time_seconds": 3.2,
    "source_filename": "policy.pdf",
    "model": "gemini-1.5-flash",
    "chunks_indexed": 42
  }
}
```

### `GET /health`

Returns `{"status": "healthy", "timestamp": "..."}` — no auth required.

## Deployment

### Railway / Render (recommended for a live demo URL)

1. Push to GitHub
2. Connect the repo on [Railway](https://railway.app) or [Render](https://render.com)
3. Add `GOOGLE_API_KEY` and `BEARER_TOKEN` as environment variables in the dashboard
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| LLM | Google Gemini 1.5 Flash (via LangChain) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS (in-memory, per request) |
| Document parsing | PyPDF, Unstructured, python-docx |
| Async HTTP | aiohttp |
| Config | python-dotenv |

## License

MIT
