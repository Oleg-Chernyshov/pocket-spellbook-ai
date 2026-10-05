from pathlib import Path

import pytest

from app.config import settings
from app.loader import load_from_file
from app.search import SearchEngine
from app.train import train_indexes

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def engine() -> SearchEngine:
    settings.data_dir = ROOT / "data"
    settings.models_dir = ROOT / "models"
    records = load_from_file(ROOT / "data" / "spells_raw.json")
    return SearchEngine(train_indexes(records))


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


def test_pagination_matches_nestjs_shape(engine: SearchEngine):
    result = engine.search("магия", language="ru", page=1, limit=5)
    assert result.pagination.page == 1
    assert result.pagination.limit == 5
    assert result.pagination.total >= len(result.data)
    assert result.data[0].score is not None
