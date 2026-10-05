from typing import Literal

from pydantic import BaseModel, Field


Language = Literal["en", "ru"]


class SpellTranslation(BaseModel):
    name: str = ""
    text: str = ""
    school: str = ""
    castingTime: str = ""
    range: str = ""
    materials: str = ""
    components: str = ""
    duration: str = ""
    source: str = ""


class SpellRecord(BaseModel):
    id: int
    level: str
    classIds: list[int] = Field(default_factory=list)
    ru: SpellTranslation
    en: SpellTranslation


class LocalizedSpell(BaseModel):
    id: int
    name: str
    level: str
    text: str
    school: str
    castingTime: str
    range: str
    materials: str
    components: str
    duration: str
    source: str
    score: float | None = None


class PaginationMeta(BaseModel):
    page: int
    limit: int
    total: int
    totalPages: int
    hasNext: bool
    hasPrev: bool


class SearchResponse(BaseModel):
    data: list[LocalizedSpell]
    pagination: PaginationMeta
