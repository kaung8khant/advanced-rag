# Repository Guidelines

## Project Structure & Module Organization
`main.py` boots the FastAPI app and registers routes. HTTP handlers live in `api/`, request and response models in `schemas/`, core RAG logic in `services/`, storage helpers in `stores/`, and Ollama integration in `clients/`. Runtime data is written to `raw/`, `chunk_store/`, and `vector_store/`; treat those as generated artifacts, not source. There is no dedicated `tests/` directory yet.

## Build, Test, and Development Commands
Create an environment and install dependencies with `python -m venv .venv`, `.\.venv\Scripts\activate`, and `pip install -r requirements.txt`. Run the API locally with `uvicorn main:app --reload`. Start Ollama separately with `ollama serve`, then pull the required models with `ollama pull nomic-embed-text` and `ollama pull llama3.2:latest`. If you need a clean local reset, remove generated contents from `raw/`, `chunk_store/`, and `vector_store/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, and small focused modules. Use `snake_case` for files, functions, and variables, `UPPER_SNAKE_CASE` for configuration constants such as `RETRIEVAL_TOP_K`, and `PascalCase` for DTOs and schema classes. Keep FastAPI route handlers thin and push orchestration into `services/`. Match the current import style: standard library first, then third-party, then local modules.

## Testing Guidelines
This repository does not currently include an automated test suite. For new work, add `pytest` tests under `tests/` using names such as `test_routes.py` or `test_rag_service.py`. Prioritize coverage for file validation, chunk generation, retrieval ranking, and API error handling. Until a suite exists, verify changes by running `uvicorn main:app --reload` and exercising `POST /upload` and `POST /ask`.

## Commit & Pull Request Guidelines
Recent history uses short messages like `bug fix`, `update ignore`, and `initial setup`. Keep commits concise and imperative, but make them more specific, for example `fix empty-question validation` or `add bm25 retrieval tests`. Pull requests should describe behavior changes, list any required Ollama models or setup steps, and include example requests or responses when API behavior changes.

## Runtime & Configuration Notes
Model names and retrieval thresholds are defined in `services/rag_service.py`. Update them there instead of scattering configuration across modules. Do not commit generated FAISS indexes, uploaded documents, or chunk JSON unless the change explicitly requires fixture data.
