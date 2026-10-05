from app.preprocess import (
    build_document,
    is_valid_translation,
    normalize_school,
    normalize_text,
    strip_html,
)
from app.schemas import SpellTranslation


def test_strip_html_and_yo():
    text = strip_html("Огнём<br>шар <b>по площади</b>")
    assert "огнём" in text.lower() or "Огнём" in text
    assert "<br>" not in text
    assert "<b>" not in text
    assert "по площади" in text
    assert normalize_text("Ёлка<br>зелёная") == "елка зеленая"


def test_school_aliases():
    assert normalize_school("Проявление") == "evocation"
    assert normalize_school("Воплощение") == "evocation"
    assert normalize_school("Проявлние") == "evocation"
    assert normalize_school("Призыв") == "conjuration"
    assert normalize_school("Вызов") == "conjuration"
    assert normalize_school("Изенение") == "transmutation"
    assert normalize_school("Evocation") == "evocation"


def test_skips_broken_translations():
    broken_source = SpellTranslation(
        name="Тест",
        text="Описание",
        school="Проявление",
        source="В, С, М",
    )
    leaked_school = SpellTranslation(
        name="Тест",
        text="Описание",
        school="Исцеляющий дух",
        source="PHB",
    )
    ok = SpellTranslation(
        name="Огненный шар",
        text="Вспышка пламени",
        school="Проявление",
        source="PHB",
    )
    assert is_valid_translation(broken_source) is False
    assert is_valid_translation(leaked_school) is False
    assert is_valid_translation(ok) is True


def test_document_repeats_name():
    translation = SpellTranslation(
        name="Fireball",
        text="A bright streak flashes",
        school="Evocation",
    )
    document = build_document(translation)
    assert document.startswith("fireball fireball fireball")
    assert "evocation" in document
    assert "bright streak" in document


def test_expand_query_adds_synonyms():
    from app.preprocess import expand_query

    expanded = expand_query("вылечить союзника на расстоянии")
    assert "исцелить" in expanded
    assert "дальности" in expanded
    assert "существо" in expanded
