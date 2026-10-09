from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from app.config import settings
from app.embeddings import Embedder, embedding_config, get_embedder
from app.loader import load_spells
from app.preprocess import ProcessedSpell, build_passage, preprocess_records, tokenize_en, tokenize_ru
from app.schemas import Language, SpellRecord

logger = logging.getLogger(__name__)

# Bump when LanguageIndex or preprocessing changes so stored models get rebuilt.
MODEL_FORMAT = 2


@dataclass
class LanguageIndex:
    language: Language
    vectorizer: FeatureUnion
    svd: TruncatedSVD
    matrix: np.ndarray
    items: list[ProcessedSpell]
    embeddings: np.ndarray | None = None


def _tokenizer_for(language: Language):
    return tokenize_ru if language == "ru" else tokenize_en


def _build_vectorizer(language: Language) -> FeatureUnion:
    tokenizer = _tokenizer_for(language)
    word = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
        lowercase=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def _fit_language(language: Language, items: list[ProcessedSpell], embedder: Embedder | None) -> LanguageIndex:
    docs = [item.doc for item in items]
    vectorizer = _build_vectorizer(language)
    features = vectorizer.fit_transform(docs)
    n_components = min(settings.svd_components, max(1, features.shape[0] - 1), max(1, features.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix = normalize(svd.fit_transform(features))
    logger.info(
        "Fitted %s index: %s spells, %s components, explained=%.3f",
        language,
        len(items),
        n_components,
        float(svd.explained_variance_ratio_.sum()) if hasattr(svd, "explained_variance_ratio_") else 0.0,
    )
    embeddings = None
    if embedder is not None:
        embeddings = embedder.encode_passages([build_passage(item.translation) for item in items])
        logger.info("Embedded %s %s spells", len(items), language)
    return LanguageIndex(
        language=language,
        vectorizer=vectorizer,
        svd=svd,
        matrix=matrix,
        items=items,
        embeddings=embeddings,
    )


def train_indexes(records: list[SpellRecord], embedder: Embedder | None = None) -> dict[Language, LanguageIndex]:
    processed = preprocess_records(records)
    indexes: dict[Language, LanguageIndex] = {}
    for language in ("ru", "en"):
        items = [item for item in processed if item.language == language]
        if not items:
            continue
        indexes[language] = _fit_language(language, items, embedder)
    if not indexes:
        raise RuntimeError("No valid spell translations to train on")
    return indexes


def records_hash(records: list[SpellRecord]) -> str:
    payload = json.dumps([record.model_dump() for record in records], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_meta(records: list[SpellRecord], embedding: dict | None = None) -> dict:
    return {
        "format": MODEL_FORMAT,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "embedding": embedding,
        "dataHash": records_hash(records),
        "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def meta_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def read_meta(path: Path | None = None) -> dict | None:
    meta_path = meta_path_for(path or settings.model_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_compatible(meta: dict | None) -> bool:
    if not meta:
        return False
    return (
        meta.get("format") == MODEL_FORMAT
        and meta.get("sklearn") == sklearn.__version__
        and meta.get("numpy") == np.__version__
        and meta.get("embedding") == embedding_config()
    )


def save_indexes(indexes: dict[Language, LanguageIndex], meta: dict, path: Path | None = None) -> Path:
    model_path = path or settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    joblib.dump(indexes, tmp_path)
    os.replace(tmp_path, model_path)
    meta_path_for(model_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved model to %s", model_path)
    return model_path


def load_indexes(path: Path | None = None) -> dict[Language, LanguageIndex]:
    model_path = path or settings.model_path
    indexes = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return indexes


def build_and_save(source: str = "auto", path: Path | None = None) -> tuple[dict[Language, LanguageIndex], dict]:
    records = load_spells(source)
    embedder = get_embedder()
    indexes = train_indexes(records, embedder)
    meta = build_meta(records, embedding_config() if embedder is not None else None)
    save_indexes(indexes, meta, path)
    return indexes, meta
