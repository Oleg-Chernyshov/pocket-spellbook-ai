from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from app.config import settings
from app.loader import load_spells
from app.preprocess import ProcessedSpell, preprocess_records, tokenize_en, tokenize_ru
from app.schemas import Language, SpellRecord

logger = logging.getLogger(__name__)


@dataclass
class LanguageIndex:
    language: Language
    vectorizer: FeatureUnion
    svd: TruncatedSVD
    matrix: np.ndarray
    items: list[ProcessedSpell]


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


def _fit_language(language: Language, items: list[ProcessedSpell]) -> LanguageIndex:
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
    return LanguageIndex(
        language=language,
        vectorizer=vectorizer,
        svd=svd,
        matrix=matrix,
        items=items,
    )


def train_indexes(records: list[SpellRecord]) -> dict[Language, LanguageIndex]:
    processed = preprocess_records(records)
    indexes: dict[Language, LanguageIndex] = {}
    for language in ("ru", "en"):
        items = [item for item in processed if item.language == language]
        if not items:
            continue
        indexes[language] = _fit_language(language, items)
    if not indexes:
        raise RuntimeError("No valid spell translations to train on")
    return indexes


def save_indexes(indexes: dict[Language, LanguageIndex], path: Path | None = None) -> Path:
    model_path = path or settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(indexes, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path


def load_indexes(path: Path | None = None) -> dict[Language, LanguageIndex]:
    model_path = path or settings.model_path
    indexes = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return indexes


def build_and_save(source: str = "auto", path: Path | None = None) -> dict[Language, LanguageIndex]:
    records = load_spells(source)
    indexes = train_indexes(records)
    save_indexes(indexes, path)
    return indexes
