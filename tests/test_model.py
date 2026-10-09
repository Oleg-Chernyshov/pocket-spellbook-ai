import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import loader
from app.config import Settings, settings
from app.embeddings import embedding_config
from app.loader import load_from_file
from app.search import SearchEngine
from app.train import build_meta, is_compatible, meta_path_for, read_meta, save_indexes, train_indexes

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", ROOT / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "embedding_model", "")

    def api_unavailable(*args, **kwargs):
        raise RuntimeError("API is disabled in tests")

    monkeypatch.setattr(loader, "load_from_api", api_unavailable)
    return tmp_path


def test_settings_reject_invalid_semantic_values():
    with pytest.raises(ValidationError):
        Settings(semantic_weight=1.5)
    with pytest.raises(ValidationError):
        Settings(semantic_span=0)


def test_saved_model_is_compatible(isolated_settings):
    records = load_from_file(ROOT / "data" / "spells_raw.json")
    save_indexes(train_indexes(records), build_meta(records))
    meta = read_meta()
    assert is_compatible(meta)
    assert len(meta["dataHash"]) == 64


def test_model_from_other_library_version_is_rebuilt(isolated_settings):
    records = load_from_file(ROOT / "data" / "spells_raw.json")
    meta = build_meta(records)
    save_indexes(train_indexes(records), {**meta, "sklearn": "0.0.0"})
    assert not is_compatible(read_meta())

    engine = SearchEngine.load_or_build()

    assert engine.meta["sklearn"] == meta["sklearn"]
    assert is_compatible(read_meta())


def test_model_with_other_embedding_settings_is_incompatible(isolated_settings, monkeypatch):
    records = load_from_file(ROOT / "data" / "spells_raw.json")
    save_indexes(train_indexes(records), build_meta(records))
    assert is_compatible(read_meta())

    monkeypatch.setattr(settings, "embedding_model", "intfloat/multilingual-e5-small")
    assert not is_compatible(read_meta())

    save_indexes(train_indexes(records), build_meta(records, embedding_config()))
    assert is_compatible(read_meta())

    monkeypatch.setattr(settings, "embedding_max_tokens", 256)
    assert not is_compatible(read_meta())


def test_build_stores_embeddings(embedder, isolated_settings, monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "intfloat/multilingual-e5-small")

    engine = SearchEngine.load_or_build()

    assert engine.semantic
    assert read_meta()["embedding"] == embedding_config()
    assert SearchEngine.load_or_build().semantic


def test_model_without_meta_is_rebuilt(isolated_settings):
    settings.model_path.write_bytes(b"not a model")
    assert read_meta() is None

    engine = SearchEngine.load_or_build()

    assert engine.indexes
    assert meta_path_for(settings.model_path).exists()
    assert json.loads(meta_path_for(settings.model_path).read_text(encoding="utf-8"))["format"]
