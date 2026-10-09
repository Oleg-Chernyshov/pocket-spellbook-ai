from __future__ import annotations

import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logs import configure_logging
from app.schemas import SearchResponse
from app.search import SearchEngine
from app import search as search_module

logger = logging.getLogger(__name__)

_rebuild_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Loading spell search model...")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    search_module.engine = SearchEngine.load_or_build()
    logger.info("Search engine is ready")
    yield
    search_module.engine = None


app = FastAPI(title="Pocket Spellbook AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _status(engine: SearchEngine) -> dict:
    return {
        "status": "ok",
        "spells": {language: len(index.items) for language, index in engine.indexes.items()},
        "semantic": engine.semantic,
        "model": {key: engine.meta.get(key) for key in ("builtAt", "dataHash", "sklearn", "embedding")},
    }


@app.get("/health")
def health():
    engine = search_module.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return _status(engine)


@app.post("/admin/rebuild")
def rebuild(
    source: Literal["auto", "api", "file"] = "auto",
    x_admin_token: str | None = Header(default=None),
):
    if not settings.admin_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token.encode("utf-8"), settings.admin_token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    if not _rebuild_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Rebuild is already running")
    try:
        engine = SearchEngine.build(source=source)
    except Exception as exc:
        logger.exception("Model rebuild failed")
        raise HTTPException(status_code=502, detail=f"Rebuild failed: {exc}") from exc
    finally:
        _rebuild_lock.release()
    search_module.engine = engine
    logger.info("Search engine rebuilt from source=%s", source)
    return _status(engine)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    language: Literal["en", "ru"] = "ru",
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    level: str | None = None,
    school: str | None = None,
    source: str | None = None,
    characterClass: int | None = Query(default=None, alias="characterClass"),
):
    engine = search_module.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return engine.search(
        query=q,
        language=language,
        page=page,
        limit=limit,
        level=level,
        school=school,
        source=source,
        character_class=characterClass,
    )
