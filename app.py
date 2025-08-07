from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import os
import json
import time
from datetime import datetime
import tempfile
import aiohttp
from pyngrok import ngrok

# Import your document processing and LLM components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
NGROK_TOKEN = os.getenv("NGROK_TOKEN")

app = FastAPI(title="Insurance Policy Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini and other components
def init_components():
    global gemini_llm, embedding_model, retriever
    
    # Initialize Gemini
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialize embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Load FAISS index if exists
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embedding_model)
        retriever = db.as_retriever()

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Metadata(BaseModel):
    processing_time: str
    source: str
    model: str = "Gemini-Pro"
    chunks_processed: Optional[int] = None

class QueryResponse(BaseModel):
    success: bool
    answers: List[str]
    metadata: Metadata

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    init_components()
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(8000)
        print(f"🔗 Public API URL: {public_url}")

@app.post("/hackrx/run", response_model=QueryResponse)
async def analyze_policy(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...)
):
    # Verify token
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    start_time = time.time()
    
    try:
        # Process document and questions
        answers = await process_document(
            request.documents,
            request.questions,
            background_tasks
        )
        
        return QueryResponse(
            success=True,
            answers=answers,
            metadata=Metadata(
                processing_time=f"{time.time() - start_time:.1f}s",
                source=str(request.documents).split("/")[-1]
            )
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            answers=[f"Error: {str(e)}"],
            metadata=Metadata(
                processing_time=f"{time.time() - start_time:.1f}s",
                source=str(request.documents).split("/")[-1]
            )
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
