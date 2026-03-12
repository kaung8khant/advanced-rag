from __future__ import annotations

from fastapi import FastAPI

from api.routes import router
from services.rag_service import ensure_runtime_dirs

ensure_runtime_dirs()

app = FastAPI()
app.include_router(router)
