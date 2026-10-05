from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from app.config import settings
from app.schemas import SpellRecord, SpellTranslation

logger = logging.getLogger(__name__)

TRANSLATION_FIELDS = (
    "name",
    "text",
    "school",
    "castingTime",
    "range",
    "materials",
    "components",
    "duration",
    "source",
)


def _translation_from_api(item: dict) -> SpellTranslation:
    return SpellTranslation(
        **{field: item.get(field) or "" for field in TRANSLATION_FIELDS}
    )


def _fetch_page(client: httpx.Client, params: dict) -> dict:
    response = client.get("/spells", params=params)
    response.raise_for_status()
    return response.json()


def _fetch_all(client: httpx.Client, language: str, extra: dict | None = None) -> list[dict]:
    page = 1
    items: list[dict] = []
    while True:
        params = {"language": language, "limit": 100, "page": page}
        if extra:
            params.update(extra)
        payload = _fetch_page(client, params)
        items.extend(payload.get("data") or [])
        pagination = payload.get("pagination") or {}
        if not pagination.get("hasNext"):
            break
        page += 1
    return items


def load_from_api(base_url: str | None = None) -> list[SpellRecord]:
    url = (base_url or settings.spellbook_api_url).rstrip("/")
    logger.info("Loading spells from %s", url)
    with httpx.Client(base_url=url, timeout=30.0) as client:
        ru_items = _fetch_all(client, "ru")
        en_items = _fetch_all(client, "en")
        classes = client.get("/spells/classes", params={"language": "en"})
        classes.raise_for_status()
        class_list = classes.json()

        spell_classes: dict[int, list[int]] = {}
        for cls in class_list:
            if not cls.get("hasSpells"):
                continue
            class_id = int(cls["id"])
            for item in _fetch_all(client, "en", {"characterClass": class_id}):
                spell_classes.setdefault(int(item["id"]), []).append(class_id)

    en_by_id = {int(item["id"]): item for item in en_items}
    records: list[SpellRecord] = []
    for item in ru_items:
        spell_id = int(item["id"])
        english = en_by_id.get(spell_id, {})
        records.append(
            SpellRecord(
                id=spell_id,
                level=str(item.get("level") or ""),
                classIds=spell_classes.get(spell_id, []),
                ru=_translation_from_api(item),
                en=_translation_from_api(english),
            )
        )
    logger.info("Loaded %s spells from API", len(records))
    return records


def load_from_file(path: Path | None = None) -> list[SpellRecord]:
    snapshot = path or settings.snapshot_path
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    records = [SpellRecord.model_validate(item) for item in payload["spells"]]
    logger.info("Loaded %s spells from %s", len(records), snapshot)
    return records


def save_snapshot(records: list[SpellRecord], path: Path | None = None) -> Path:
    snapshot = path or settings.snapshot_path
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spells": [record.model_dump() for record in records]}
    snapshot.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved snapshot with %s spells to %s", len(records), snapshot)
    return snapshot


def load_spells(source: str = "auto") -> list[SpellRecord]:
    if source not in {"auto", "api", "file"}:
        raise ValueError(f"Unknown source: {source}")

    if source in {"auto", "api"}:
        try:
            records = load_from_api()
            save_snapshot(records)
            return records
        except Exception as exc:
            logger.warning("API load failed: %s", exc)
            if source == "api":
                raise

    return load_from_file()
