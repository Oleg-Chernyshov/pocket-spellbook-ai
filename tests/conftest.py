from pathlib import Path

import pytest

from app.embeddings import Embedder, get_embedder
from app.loader import load_from_file
from app.schemas import SpellRecord
from app.search import SearchEngine
from app.train import train_indexes

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def records() -> list[SpellRecord]:
    return load_from_file(ROOT / "data" / "spells_raw.json")


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    loaded = get_embedder()
    assert loaded is not None, "embedding model failed to load"
    return loaded


@pytest.fixture(scope="session")
def hybrid_engine(records, embedder) -> SearchEngine:
    return SearchEngine(train_indexes(records, embedder), embedder=embedder)


@pytest.fixture(scope="session")
def lexical_engine(records) -> SearchEngine:
    return SearchEngine(train_indexes(records))
