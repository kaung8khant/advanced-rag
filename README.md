# basic rag

`basic rag` is a minimal Retrieval-Augmented Generation (RAG) app built with FastAPI, Ollama, and FAISS.

It supports:
- uploading `.pdf` / `.md` / `.markdown` files
- chunking text with token windows + overlap
- embedding chunks with Ollama embeddings
- storing vectors in FAISS
- query rewriting before retrieval
- retrieving top chunks for a question
- generating a final answer with an Ollama chat model

## Project Structure

- `main.py`: app bootstrap + router registration
- `api/routes.py`: FastAPI endpoints (`/upload`, `/ask`)
- `schemas/api_schemas.py`: request/response DTOs
- `services/rag_service.py`: upload/ask orchestration + app constants
- `services/file_extractor.py`: PDF/Markdown text extraction + metadata enrichment
- `services/token_chunker.py`: chunking + token counts
- `services/vectorizer.py`: text/chunk embedding via Ollama OpenAI-compatible API
- `services/query_rewriter.py`: query rewrite with Ollama
- `services/answer_generator.py`: final answer generation from retrieved context
- `services/retrieval_service.py`: retrieval orchestration (rank/filter/map)
- `stores/faiss_store.py`: store/search vectors in FAISS
- `stores/chunk_store.py`: chunk JSON persistence helpers
- `clients/ollama_client.py`: shared Ollama OpenAI-compatible client factory

## Constants (in `services/rag_service.py`)

- `EMBEDDING_MODEL = "nomic-embed-text"`
- `ANSWER_MODEL = "llama3.2:latest"`
- `RETRIEVAL_TOP_K = 5`
- `MIN_RETRIEVAL_SCORE = 0.55`

## Data Folders

- `raw/`: uploaded files
- `vector_store/`: FAISS index file (`index.faiss`)
- `chunk_store/`: vectorized chunks saved as JSON (includes `faiss_id`)

## Setup

From `level3`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and ensure models are available:

```powershell
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:latest
```

Run API:

```powershell
uvicorn main:app --reload
```

## API Endpoints

### `POST /upload`

Uploads a file and indexes its chunks.

Form-data:
- `file`: file upload (`.pdf`, `.md`, `.markdown`)
- `owner_id` (optional): default `"1"`

What it does:
1. Extracts text segments
2. Builds chunks (`chunk_size=300`, `token_overlap=50`)
3. Embeds each chunk
4. Stores vectors in FAISS
5. Attaches returned `faiss_id` to each chunk
6. Saves chunk JSON to `chunk_store/<doc_id>.json`

### `POST /ask`

Request body:

```json
{
  "question": "Your question here"
}
```

What it does:
1. Rewrites query for retrieval
2. Embeds rewritten query
3. Searches FAISS
4. Filters by score threshold (`>= MIN_RETRIEVAL_SCORE`)
5. Ranks and returns top matches (`RETRIEVAL_TOP_K`)
6. Builds context from matches
7. Generates final answer with Ollama LLM

Response includes:
- `question`
- `top_k`
- `answer`
- `matches` (ranked chunks with `k`, `faiss_id`, `score`, `chunk`)

## Notes

- FAISS layer stores vectors only.
- Chunk metadata + `faiss_id` are stored in JSON files under `chunk_store/`.
- If you want a clean reset, remove and recreate `raw/`, `vector_store/`, and `chunk_store/`.

## Limitations

- Query-document asymmetry:
  Query text is short and user-style, while document chunks are longer and domain-specific. Even with query rewriting, semantic mismatch can still reduce recall.
- Query rewriting risk:
  Rewriting can drift from original intent in edge cases and may retrieve less relevant chunks.
- No reranker:
  Retrieval currently depends on embedding similarity + score threshold only.
