"""
Insurance Policy Analysis API
-------------------------------
Accepts a URL to a policy document (PDF / DOCX / EML) and a list of
questions. Returns structured JSON answers grounded in the document.

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
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredEmailLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, HttpUrl

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BEARER_TOKEN: str = os.environ["BEARER_TOKEN"]
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "erudite-imprint-469117-p3")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Shared model singletons (loaded once at startup)
# ---------------------------------------------------------------------------
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_llm: Optional[ChatGoogleGenerativeAI] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL)
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embedding_model


def get_llm():
    global _llm
    if _llm is None:
        log.info("Initialising Gemini LLM: %s", GEMINI_MODEL)
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models at startup so the first real request isn't slow
    get_embedding_model()
    get_llm()
    log.info("✅ Models ready.")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Insurance Policy Analysis API",
    description="Upload a policy document URL and ask questions — get structured JSON answers.",
    version="2.0.0",
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


def _infer_extension(url: str, content_type: str) -> str:
    """Try to get file extension from URL path, fall back to Content-Type."""
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return suffix
    ct_map = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "message/rfc822": ".eml",
    }
    return ct_map.get(content_type.split(";")[0].strip(), ".pdf")


async def download_document(url: str) -> tuple[str, str]:
    """
    Download a document from *url* into a temp file.
    Returns (local_file_path, original_filename).
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PolicyBot/1.0)"}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            str(url), timeout=aiohttp.ClientTimeout(total=30), headers=headers
        ) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Could not fetch document — remote server returned {resp.status}.",
                )
            content_type = resp.headers.get("Content-Type", "")

            # Reject HTML responses — the URL likely hit a login wall or redirect
            if "text/html" in content_type:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "The URL returned an HTML page instead of a document. "
                        "Make sure the URL points directly to a PDF or DOCX file "
                        "(it should end in .pdf or .docx and be publicly accessible)."
                    ),
                )

            ext = _infer_extension(str(url), content_type)
            data = await resp.read()

    # Write to a named temp file so loaders can open it by path
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    filename = Path(str(url).split("?")[0]).name or f"document{ext}"
    return tmp.name, filename


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
    """Parse the LLM's raw string output into an Answer object."""
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Graceful fallback — return the raw text so nothing crashes
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


def build_fallback_answer(question: str, documents: List[Document], source: str, reason: str) -> Answer:
    """Create a grounded fallback answer when the Gemini model is unavailable."""
    text_parts = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
    combined_text = " ".join(text_parts)
    if combined_text:
        sentences = re.split(r"(?<=[.!?])\s+", combined_text)
        snippet = " ".join(sentence.strip() for sentence in sentences if sentence.strip())[:600]
        clause_mapping = [ClauseMapping(clause_text=snippet, source=source)] if snippet else []
        justification = (
            "The Gemini model was unavailable, so this answer is an extractive fallback from the document text. "
            f"Relevant excerpt: {snippet}"
        )
    else:
        clause_mapping = []
        justification = (
            "The Gemini model was unavailable and no usable document text could be extracted. "
            f"Fallback reason: {reason}"
        )

    return Answer(
        question=question,
        decision="insufficient information",
        amount=None,
        justification=justification,
        clause_mapping=clause_mapping,
    )


async def process_document(doc_url: str, questions: List[str]) -> tuple[List[Answer], int]:
    """Full pipeline: download → load → chunk → embed → answer each question."""
    local_path, filename = await download_document(doc_url)

    try:
        try:
            documents = load_documents(local_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse document: {e}. Ensure the URL points to a valid, uncorrupted PDF or DOCX.",
            )
        vector_store, n_chunks = build_vector_store(documents)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chain = create_stuff_documents_chain(llm=get_llm(), prompt=POLICY_PROMPT)

        answers: List[Answer] = []
        for question in questions:
            relevant_docs = retriever.invoke(question)
            try:
                raw_result = chain.invoke({"context": relevant_docs, "question": question})
                answers.append(parse_llm_answer(raw_result, question, filename))
            except Exception as exc:
                log.warning("Vertex request failed for question %r: %s", question, exc)
                answers.append(build_fallback_answer(question, relevant_docs, filename, str(exc)))

        return answers, n_chunks

    finally:
        # Always clean up the temp file
        Path(local_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Analyse a policy document and answer questions",
)
async def analyze_policy(request: QueryRequest):
    start = time.perf_counter()

    answers, n_chunks = await process_document(str(request.documents), request.questions)

    return QueryResponse(
        success=True,
        answers=answers,
        metadata=Metadata(
            processing_time_seconds=round(time.perf_counter() - start, 2),
            source_filename=Path(str(request.documents).split("?")[0]).name,
            model=GEMINI_MODEL,
            chunks_indexed=n_chunks,
        ),
    )


@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
