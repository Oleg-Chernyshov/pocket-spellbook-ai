from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import SearchResponse
from app.search import SearchEngine
from app import search as search_module

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    engine = search_module.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {
        "status": "ok",
        "spells": {language: len(index.items) for language, index in engine.indexes.items()},
    }


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
