# Advanced RAG

`advanced-rag` is a FastAPI-based Retrieval-Augmented Generation app using Ollama, FAISS, BM25, and a Hugging Face reranker.

It supports:
- uploading `.pdf` / `.md` / `.markdown` files
- extracting and chunking document text
- embedding chunks with Ollama embeddings
- storing vectors in FAISS
- multi-query semantic retrieval
- BM25 keyword retrieval
- merging and deduplicating retrieval results
- reranking merged candidates with a Hugging Face cross-encoder
- generating a final answer with an Ollama chat model

<img width="689" height="702" alt="Screenshot 2026-03-24 at 9 25 25 AM" src="https://github.com/user-attachments/assets/f4fad685-151c-47cc-84c8-e78b7ddb81d3" />

## Current Pipeline

`POST /upload`
1. Save uploaded file to `raw/`
2. Extract text and metadata
3. Build token chunks
4. Embed chunks with `nomic-embed-text`
5. Store vectors in FAISS
6. Save chunk metadata with `faiss_id` into `chunk_store/`

`POST /ask`
1. Generate multiple query variants for semantic retrieval
2. Retrieve semantic matches from FAISS
3. Retrieve lexical matches with BM25
4. Merge and deduplicate matches
5. Rerank merged matches with `BAAI/bge-reranker-base`
6. Build context from the reranked top `RETRIEVAL_TOP_K`
7. Generate an answer with the Ollama answer model

## Project Structure

- `main.py`: FastAPI app bootstrap
- `api/routes.py`: HTTP routes for `/upload` and `/ask`
- `schemas/api_schemas.py`: request and response DTOs
- `clients/ollama_client.py`: shared Ollama OpenAI-compatible client
- `services/rag_service.py`: upload flow, retrieval merge, rerank, and answer orchestration
- `services/file_extractor.py`: PDF/Markdown extraction and metadata enrichment
- `services/token_chunker.py`: chunk building
- `services/vectorizer.py`: embedding calls through Ollama
- `services/multi_query_retriever.py`: multi-query semantic retrieval
- `services/bm25_service.py`: BM25 retrieval over stored chunks
- `services/reranker.py`: Hugging Face cross-encoder reranking
- `services/answer_generator.py`: context building and final answer generation
- `services/retrieval_service.py`: FAISS retrieval mapping and persistence helpers
- `stores/faiss_store.py`: FAISS write/search helpers
- `stores/chunk_store.py`: chunk JSON helpers

## Configuration

Configuration is loaded from `.env`.

Current environment variables:
- `UPLOAD_DIR=raw`
- `CHUNK_STORE_DIR=chunk_store`
- `VECTOR_STORE_DIR=vector_store`
- `EMBEDDING_MODEL=nomic-embed-text`
- `ANSWER_MODEL=deepseek-r1:1.5b`
- `RERANKING_MODEL=BAAI/bge-reranker-base`
- `RETRIEVAL_TOP_K=5`
- `MIN_RETRIEVAL_SCORE=0.55`
- `MULTI_QUERY_COUNT=3`

## Data Folders

- `raw/`: uploaded source files
- `chunk_store/`: stored chunk metadata JSON files
- `vector_store/`: FAISS index data

These are runtime artifacts and should generally not be committed.

## Setup

Using `uv`:

```powershell
uv sync
```

Or with a virtual environment and pip:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and pull the required models:

```powershell
ollama serve
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```

Run the API:

```powershell
uv run uvicorn main:app --reload
```

Or if using the virtual environment directly:

```powershell
uvicorn main:app --reload
```

## API

### `POST /upload`

Uploads and indexes a file.

Form-data:
- `file`: `.pdf`, `.md`, or `.markdown`
- `owner_id`: optional, defaults to `"1"`

Response includes:
- `message`
- `extracted_text`
- `token_chunks`

### `POST /ask`

Request body:

```json
{
  "question": "Your question here"
}
```

Response includes:
- `question`
- `top_k`
- `answer`
- `matches`

Each returned match may include:
- `k`
- `faiss_id`
- `score`
- `rerank_score`
- `retrieval_method`
- `matched_queries`
- `chunk`

## Notes

- Multi-query retrieval is used for semantic recall.
- BM25 is used for lexical recall.
- The reranker decides the final top results returned to the LLM.
- The current answer step uses only the reranked top `RETRIEVAL_TOP_K` matches as context.
- If you want a clean reset, remove generated contents from `raw/`, `chunk_store/`, and `vector_store/`.

## Limitations

- There is no automated evaluation suite yet.
- Retrieval quality still depends heavily on chunk quality and source document structure.
- Multi-query generation depends on Ollama availability.
- The final answer quality depends on both retrieval quality and the answer model.
