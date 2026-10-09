from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import normalize

from app.config import settings
from app.embeddings import Embedder, get_embedder
from app.preprocess import normalize_school, normalize_text, expand_query
from app.schemas import Language, LocalizedSpell, PaginationMeta, SearchResponse
from app.train import LanguageIndex, build_and_save, is_compatible, load_indexes, read_meta

logger = logging.getLogger(__name__)


def rescale_semantic(similarity: np.ndarray) -> np.ndarray:
    # E5 similarities sit in a narrow, query-dependent band (~0.75-0.9), so measure them
    # from the query's median with a fixed span instead of using raw cosine values.
    baseline = float(np.median(similarity))
    return np.clip((similarity - baseline) / settings.semantic_span, 0.0, 1.0)


@dataclass
class FilterColumns:
    levels: np.ndarray
    school_norms: np.ndarray
    school_texts: np.ndarray
    sources: np.ndarray
    classes: dict[int, np.ndarray]


def _filter_columns(index: LanguageIndex) -> FilterColumns:
    items = index.items
    classes: dict[int, np.ndarray] = {}
    for position, item in enumerate(items):
        for class_id in item.record.classIds:
            classes.setdefault(class_id, np.zeros(len(items), dtype=bool))[position] = True
    return FilterColumns(
        levels=np.array([item.record.level for item in items], dtype=object),
        school_norms=np.array([item.school_norm for item in items], dtype=object),
        school_texts=np.array([normalize_text(item.translation.school) for item in items], dtype=object),
        sources=np.array([item.source for item in items], dtype=object),
        classes=classes,
    )


@dataclass
class SearchEngine:
    indexes: dict[Language, LanguageIndex]
    meta: dict = field(default_factory=dict)
    embedder: Embedder | None = None
    columns: dict[Language, FilterColumns] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = {language: _filter_columns(index) for language, index in self.indexes.items()}

    @property
    def semantic(self) -> bool:
        return self.embedder is not None and all(index.embeddings is not None for index in self.indexes.values())

    @classmethod
    def build(cls, source: str = "auto") -> "SearchEngine":
        indexes, meta = build_and_save(source=source)
        return cls(indexes, meta, get_embedder())

    @classmethod
    def load_or_build(cls) -> "SearchEngine":
        meta = read_meta()
        if settings.model_path.exists() and is_compatible(meta):
            try:
                return cls(load_indexes(), meta, get_embedder())
            except Exception as exc:
                logger.warning("Failed to load model, rebuilding: %s", exc)
        else:
            logger.info("Model is missing or was built with other settings or library versions, rebuilding")
        return cls.build(source="auto")

    def search(
        self,
        query: str,
        language: Language = "ru",
        page: int = 1,
        limit: int = 20,
        level: str | None = None,
        school: str | None = None,
        source: str | None = None,
        character_class: int | None = None,
    ) -> SearchResponse:
        page = max(1, page)
        limit = min(max(1, limit), settings.page_size_max)
        empty = SearchResponse(
            data=[],
            pagination=PaginationMeta(
                page=page,
                limit=limit,
                total=0,
                totalPages=0,
                hasNext=False,
                hasPrev=False,
            ),
        )
        if not query.strip():
            return empty

        index = self.indexes.get(language)
        if index is None:
            return empty
        columns = self.columns[language]

        query_doc = expand_query(query)
        features = index.vectorizer.transform([query_doc])
        query_vec = normalize(index.svd.transform(features))
        scores = (index.matrix @ query_vec.T).ravel()
        if self.embedder is not None and index.embeddings is not None:
            similarity = index.embeddings @ self.embedder.encode_queries([query_doc])[0]
            weight = settings.semantic_weight
            scores = (1 - weight) * scores + weight * rescale_semantic(similarity)

        mask = scores >= settings.min_score
        if level:
            mask &= columns.levels == str(level)
        if school:
            school_norm = normalize_school(school)
            if school_norm:
                mask &= columns.school_norms == school_norm
            else:
                mask &= columns.school_texts == normalize_text(school)
        if source:
            mask &= columns.sources == source
        if character_class is not None:
            class_mask = columns.classes.get(character_class)
            if class_mask is None:
                mask[:] = False
            else:
                mask &= class_mask

        candidates = np.flatnonzero(mask)
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        total = len(ranked)
        total_pages = math.ceil(total / limit) if total else 0
        start = (page - 1) * limit
        page_items = ranked[start : start + limit]

        data = []
        for idx in page_items:
            item = index.items[idx]
            translation = item.translation
            data.append(
                LocalizedSpell(
                    id=item.record.id,
                    name=translation.name,
                    level=item.record.level,
                    text=translation.text,
                    school=translation.school,
                    castingTime=translation.castingTime,
                    range=translation.range,
                    materials=translation.materials,
                    components=translation.components,
                    duration=translation.duration,
                    source=translation.source,
                    score=round(float(scores[idx]), 4),
                )
            )

        return SearchResponse(
            data=data,
            pagination=PaginationMeta(
                page=page,
                limit=limit,
                total=total,
                totalPages=total_pages,
                hasNext=page < total_pages,
                hasPrev=page > 1,
            ),
        )


engine: SearchEngine | None = None
