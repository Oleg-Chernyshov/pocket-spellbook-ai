from collections import Counter

import pytest

from app.preprocess import normalize_school
from app.search import SearchEngine


@pytest.fixture(params=["hybrid", "lexical"])
def engine(request) -> SearchEngine:
    return request.getfixturevalue(f"{request.param}_engine")


def test_hybrid_engine_uses_embeddings(hybrid_engine: SearchEngine, lexical_engine: SearchEngine):
    assert hybrid_engine.semantic
    assert not lexical_engine.semantic


def _names(engine: SearchEngine, query: str, language: str, **filters) -> list[str]:
    result = engine.search(query=query, language=language, limit=5, **filters)
    return [item.name for item in result.data]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
            ("огненный шар по площади", "Огненный шар"),
            ("исцелить словом", "Исцеляющее слово"),
        ("кислотные брызги", "Кислотные брызги"),
        ("стать невидимым", "Невидимость"),
        ("магическая стрела", "Магическая стрела"),
        ("поднять мертвого", "Поднять мертвого"),
        ("полет", "Полет"),
        ("контрзаклятье", "Контрзаклятье"),
        ("удержать гуманоида", "Удержание персоны"),
        ("громовая волна", "Громовая волна"),
    ],
)
def test_russian_query_ranks_expected_spell(engine: SearchEngine, query: str, expected: str):
    names = _names(engine, query, "ru")
    assert expected in names, f"{query!r} -> {names}"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("area fire explosion", "Fireball"),
        ("heal ally at a distance", "Healing Word"),
        ("become invisible", "Invisibility"),
        ("unguided magic darts", "Magic Missile"),
    ],
)
def test_english_query_ranks_expected_spell(engine: SearchEngine, query: str, expected: str):
    names = _names(engine, query, "en")
    assert expected in names, f"{query!r} -> {names}"


def test_level_filter(engine: SearchEngine):
    result = engine.search("огонь", language="ru", level="0", limit=20)
    assert result.data
    assert all(item.level == "0" for item in result.data)


def test_school_filter(engine: SearchEngine):
    result = engine.search("огонь", language="ru", school="Воплощение", limit=50)
    assert result.data
    assert all(normalize_school(item.school) == "evocation" for item in result.data)


def test_character_class_filter(engine: SearchEngine):
    index = engine.indexes["ru"]
    class_id = Counter(cid for item in index.items for cid in item.record.classIds).most_common(1)[0][0]
    allowed = {item.record.id for item in index.items if class_id in item.record.classIds}
    result = engine.search("магия", language="ru", character_class=class_id, limit=100)
    assert result.data
    assert all(item.id in allowed for item in result.data)


def test_unknown_character_class_returns_nothing(engine: SearchEngine):
    result = engine.search("магия", language="ru", character_class=999_999)
    assert result.data == []
    assert result.pagination.total == 0


def test_results_are_sorted_by_score(engine: SearchEngine):
    result = engine.search("огненный шар", language="ru", limit=20)
    scores = [item.score for item in result.data]
    assert scores == sorted(scores, reverse=True)


def test_pagination_matches_nestjs_shape(engine: SearchEngine):
    result = engine.search("магия", language="ru", page=1, limit=5)
    assert result.pagination.page == 1
    assert result.pagination.limit == 5
    assert result.pagination.total >= len(result.data)
    assert result.data[0].score is not None
