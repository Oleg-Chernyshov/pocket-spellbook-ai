from __future__ import annotations

import math
from dataclasses import dataclass

from sklearn.preprocessing import normalize

from app.config import settings
from app.preprocess import normalize_school, normalize_text, expand_query
from app.schemas import Language, LocalizedSpell, PaginationMeta, SearchResponse
from app.train import LanguageIndex, load_indexes, build_and_save


@dataclass
class SearchEngine:
    indexes: dict[Language, LanguageIndex]

    @classmethod
    def load_or_build(cls) -> "SearchEngine":
        if settings.model_path.exists():
            return cls(load_indexes())
        return cls(build_and_save(source="auto"))

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

        query_doc = expand_query(query)
        features = index.vectorizer.transform([query_doc])
        query_vec = normalize(index.svd.transform(features))
        scores = (index.matrix @ query_vec.T).ravel()

        school_norm = normalize_school(school) if school else None
        ranked: list[tuple[int, float]] = []
        for idx, item in enumerate(index.items):
            if scores[idx] < settings.score_threshold:
                continue
            if level and item.record.level != str(level):
                continue
            if school_norm and item.school_norm != school_norm:
                continue
            if school and not school_norm and normalize_text(item.translation.school) != normalize_text(school):
                continue
            if source and item.source != source:
                continue
            if character_class is not None and character_class not in item.record.classIds:
                continue
            ranked.append((idx, float(scores[idx])))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        total = len(ranked)
        total_pages = math.ceil(total / limit) if total else 0
        start = (page - 1) * limit
        page_items = ranked[start : start + limit]

        data = []
        for idx, score in page_items:
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
                    score=round(score, 4),
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
