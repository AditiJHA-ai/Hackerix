"""
Insurance Policy Analysis API
-------------------------------
Two analysis routes:
  POST /hackrx/run     — accepts a document URL + questions (JSON body)
  POST /hackrx/upload  — accepts a file upload + questions (multipart/form-data)

Both routes are protected by a Bearer token.
The frontend never receives or stores credentials.

Run:
    uvicorn app:app --reload
"""

import json
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredEmailLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, HttpUrl

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (all secrets stay on the server, never sent to the browser)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
BEARER_TOKEN: str = os.getenv("BEARER_TOKEN", "")
LLM_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))

# ---------------------------------------------------------------------------
# Shared model singletons
# ---------------------------------------------------------------------------
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_llm: Optional[ChatGroq] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL)
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embedding_model


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Missing GROQ_API_KEY in environment or .env.",
            )
        log.info("Initialising Groq LLM: %s", LLM_MODEL)
        _llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep startup lightweight; models are created lazily on first request.
    log.info("Lifespan started. Models will be initialised on first use.")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Insurance Policy Analysis API",
    description="Analyse insurance documents via URL or file upload.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
        )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": "https://example.com/policy.pdf",
                "questions": [
                    "What is the waiting period for pre-existing diseases?",
                    "Is knee surgery covered?",
                ],
            }
        }
    }


class ClauseMapping(BaseModel):
    clause_text: str
    source: str


class Answer(BaseModel):
    question: str
    decision: str
    amount: Optional[float]
    justification: str
    clause_mapping: List[ClauseMapping]


class Metadata(BaseModel):
    processing_time_seconds: float
    source_filename: str
    model: str
    chunks_indexed: int


class QueryResponse(BaseModel):
    success: bool
    answers: List[Answer]
    metadata: Metadata


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".eml", ".msg"}
ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "message/rfc822": ".eml",
}


def _infer_extension(url: str, content_type: str) -> str:
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return suffix
    return ALLOWED_MIME_TYPES.get(content_type.split(";")[0].strip(), ".pdf")


async def download_document(url: str) -> tuple[str, str]:
    """Download a remote document into a temp file. Returns (path, filename)."""
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url), timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Could not fetch document — remote server returned {resp.status}.",
                )
            content_type = resp.headers.get("Content-Type", "")
            ext = _infer_extension(str(url), content_type)
            data = await resp.read()

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    filename = Path(str(url).split("?")[0]).name or f"document{ext}"
    return tmp.name, filename


async def save_upload(file: UploadFile) -> tuple[str, str]:
    """
    Save an uploaded file to a temp location.
    Validates size and extension. Returns (path, original_filename).
    """
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name, original_name


def load_documents(path: str) -> List[Document]:
    """Load a local file into LangChain Documents."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
    elif ext == ".docx":
        docs = UnstructuredWordDocumentLoader(path).load()
    elif ext in {".eml", ".msg"}:
        docs = UnstructuredEmailLoader(path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext!r}")
    return [d if isinstance(d, Document) else Document(**d) for d in docs]


def build_vector_store(documents: List[Document]) -> tuple[FAISS, int]:
    """Chunk documents and build an in-memory FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, get_embedding_model())
    return db, len(chunks)


# ---------------------------------------------------------------------------
# LLM prompt & chain
# ---------------------------------------------------------------------------
POLICY_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert assistant specialising in insurance and policy document analysis.
Use ONLY the context below to answer the question.
If the answer is not found in the context, set decision to "insufficient information".

Return a JSON object with EXACTLY these fields:
    "decision"      : string  — e.g. "approved", "rejected", "covered", "not covered", "insufficient information"
    "amount"        : number or null  — payout/limit amount if stated, else null
    "justification" : string  — concise explanation referencing the policy
    "clause_mapping": list of {{"clause_text": string, "source": string}}

Output ONLY valid JSON. No markdown fences, no extra keys.

Context:
{context}

Question: {question}
""",
)


def parse_llm_answer(raw: str, question: str, source: str) -> Answer:
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return Answer(
            question=question,
            decision="parse_error",
            amount=None,
            justification=raw,
            clause_mapping=[],
        )

    return Answer(
        question=question,
        decision=data.get("decision", "unknown"),
        amount=data.get("amount"),
        justification=data.get("justification", ""),
        clause_mapping=[
            ClauseMapping(
                clause_text=c.get("clause_text", ""),
                source=c.get("source", source),
            )
            for c in data.get("clause_mapping", [])
        ],
    )


async def _run_pipeline(local_path: str, filename: str, questions: List[str]) -> tuple[List[Answer], int]:
    """Shared core: load → chunk → embed → answer."""
    documents = load_documents(local_path)
    vector_store, n_chunks = build_vector_store(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    chain = POLICY_PROMPT | get_llm()

    answers: List[Answer] = []
    for question in questions:
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
        raw_result = chain.invoke({"context": context_text, "question": question})
        raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
        answers.append(parse_llm_answer(raw_text, question, filename))

    return answers, n_chunks


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Analyse a policy document from a URL",
)
async def analyze_policy_url(request: QueryRequest):
    """Original endpoint — accepts a public document URL."""
    start = time.perf_counter()
    local_path, filename = await download_document(str(request.documents))

    try:
        answers, n_chunks = await _run_pipeline(local_path, filename, request.questions)
    finally:
        Path(local_path).unlink(missing_ok=True)

    return QueryResponse(
        success=True,
        answers=answers,
        metadata=Metadata(
            processing_time_seconds=round(time.perf_counter() - start, 2),
            source_filename=filename,
            model=LLM_MODEL,
            chunks_indexed=n_chunks,
        ),
    )


@app.post(
    "/hackrx/upload",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Analyse an uploaded policy document (multipart/form-data)",
)
async def analyze_policy_upload(
    file: UploadFile = File(..., description="PDF or DOCX policy document"),
    questions: str = Form(..., description="JSON array of question strings"),
):
    """
    New endpoint — accepts a file upload directly from the browser.

    Form fields:
      file      — the document (PDF / DOCX / EML)
      questions — JSON-encoded list of strings, e.g. '["Is X covered?", "What is the deductible?"]'
    """
    # Parse questions from the JSON string sent by the form
    try:
        parsed_questions: List[str] = json.loads(questions)
        if not isinstance(parsed_questions, list) or not parsed_questions:
            raise ValueError
        parsed_questions = [str(q).strip() for q in parsed_questions if str(q).strip()]
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'questions' must be a non-empty JSON array of strings.",
        )

    start = time.perf_counter()
    local_path, filename = await save_upload(file)

    try:
        answers, n_chunks = await _run_pipeline(local_path, filename, parsed_questions)
    finally:
        Path(local_path).unlink(missing_ok=True)

    return QueryResponse(
        success=True,
        answers=answers,
        metadata=Metadata(
            processing_time_seconds=round(time.perf_counter() - start, 2),
            source_filename=filename,
            model=LLM_MODEL,
            chunks_indexed=n_chunks,
        ),
    )


@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Static file serving (optional — place index.html in a ./static folder)
# Uncomment to serve the frontend from the same process:
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")