# Insurance Policy Analysis API

This project implements an API for analyzing insurance policy documents using LangChain, Google's Gemini LLM, and FastAPI.

## Features

- 📄 Multi-format document support (PDF, DOCX, Email)
- 🔍 Semantic search using FAISS and HuggingFace embeddings
- 🤖 Advanced policy analysis using Google's Gemini LLM
- 🚀 FastAPI endpoint with authentication
- 📊 Structured JSON responses

## API Endpoints

### POST /hackrx/run

Analyzes insurance policy documents and answers questions.

```bash
curl -X POST "https://your-domain/hackrx/run" \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "documents": "https://example.com/policy.pdf",
  "questions": ["What is the waiting period for pre-existing diseases?"]
}'
```

Response format:
```json
{
  "success": true,
  "answers": ["Pre-existing diseases have a waiting period of 48 months..."],
  "metadata": {
    "processing_time": "2.5s",
    "source": "policy.pdf",
    "model": "Gemini-Pro"
  }
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export BEARER_TOKEN="your_api_token"
```

3. Run the API:
```bash
python app.py
```

## Development

- Written in Python using Jupyter Notebook
- Uses LangChain for document processing
- FAISS for vector storage
- Google Gemini for LLM capabilities
- FastAPI for API implementation
- Ngrok for public access

## Security

- Bearer token authentication
- Secure document handling
- Temporary file cleanup
- Rate limiting support

## License

MIT License
