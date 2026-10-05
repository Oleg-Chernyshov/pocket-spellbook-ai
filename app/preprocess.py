from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

import snowballstemmer

from app.schemas import Language, SpellRecord, SpellTranslation

TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
COMPONENT_CHARS = set("всмvsm")

QUERY_SYNONYMS = {
    "вылечить": "исцелить восстанавливает",
    "лечить": "исцелить восстанавливает",
    "лечения": "исцеление восстанавливает",
    "союзника": "существо",
    "расстоянии": "дальности футов",
    "площади": "сфере радиус область",
    "невидимым": "невидимость",
    "heal": "healing restore",
    "ally": "creature",
    "distance": "range feet",
    "invisible": "invisibility",
}

RU_STEMMER = snowballstemmer.stemmer("russian")
EN_STEMMER = snowballstemmer.stemmer("english")

SCHOOL_ALIASES = {
    "abjuration": "abjuration",
    "ограждение": "abjuration",
    "conjuration": "conjuration",
    "вызов": "conjuration",
    "призыв": "conjuration",
    "divination": "divination",
    "прорицание": "divination",
    "enchantment": "enchantment",
    "очарование": "enchantment",
    "evocation": "evocation",
    "воплощение": "evocation",
    "проявление": "evocation",
    "проявлние": "evocation",
    "illusion": "illusion",
    "иллюзия": "illusion",
    "necromancy": "necromancy",
    "некромантия": "necromancy",
    "transmutation": "transmutation",
    "преобразование": "transmutation",
    "превращение": "transmutation",
    "изенение": "transmutation",
}


@dataclass
class ProcessedSpell:
    record: SpellRecord
    language: Language
    doc: str
    school_norm: str
    source: str
    translation: SpellTranslation


def strip_html(text: str) -> str:
    cleaned = unescape(text or "")
    cleaned = BR_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_text(text: str) -> str:
    return strip_html(text).lower().replace("ё", "е")


def normalize_school(school: str) -> str | None:
    key = normalize_text(school)
    return SCHOOL_ALIASES.get(key)


def expand_query(text: str) -> str:
    normalized = normalize_text(text)
    extras = [
        QUERY_SYNONYMS[token]
        for token in TOKEN_RE.findall(normalized)
        if token in QUERY_SYNONYMS
    ]
    if extras:
        return f"{normalized} {' '.join(extras)}"
    return normalized


def is_component_source(source: str) -> bool:
    compact = re.sub(r"[\s,.;/]+", "", normalize_text(source))
    return bool(compact) and set(compact) <= COMPONENT_CHARS


def is_valid_translation(translation: SpellTranslation) -> bool:
    if not translation.name.strip() or not translation.text.strip():
        return False
    if is_component_source(translation.source):
        return False
    school = translation.school.strip()
    if not school:
        return False
    if normalize_school(school) is None and " " in school:
        return False
    return True


def build_document(translation: SpellTranslation) -> str:
    name = normalize_text(translation.name)
    school = normalize_text(translation.school)
    text = normalize_text(translation.text)
    range_text = normalize_text(translation.range)
    return f"{name} {name} {name} {school} {range_text} {text}".strip()


def tokenize(text: str, language: Language) -> list[str]:
    stemmer = RU_STEMMER if language == "ru" else EN_STEMMER
    return [stemmer.stemWord(token.lower().replace("ё", "е")) for token in TOKEN_RE.findall(text)]


def tokenize_ru(text: str) -> list[str]:
    return tokenize(text, "ru")


def tokenize_en(text: str) -> list[str]:
    return tokenize(text, "en")


def preprocess_records(records: list[SpellRecord]) -> list[ProcessedSpell]:
    processed: list[ProcessedSpell] = []
    for record in records:
        for language in ("ru", "en"):
            translation: SpellTranslation = getattr(record, language)
            if not is_valid_translation(translation):
                continue
            processed.append(
                ProcessedSpell(
                    record=record,
                    language=language,
                    doc=build_document(translation),
                    school_norm=normalize_school(translation.school) or normalize_text(translation.school),
                    source=translation.source.strip(),
                    translation=translation,
                )
            )
    return processed
