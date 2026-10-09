from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    spellbook_api_url: str = "http://localhost:3000"
    min_score: float = 0.35
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    page_size_max: int = 100
    svd_components: int = 80
    cors_origins: str = "http://localhost:8080"
    admin_token: str = ""
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_file: str = "onnx/model_qint8_avx512_vnni.onnx"
    embedding_max_tokens: int = Field(128, ge=8, le=512)
    embeddings_dir: Path = Path("models/embeddings")
    semantic_weight: float = Field(0.5, ge=0, le=1)
    semantic_span: float = Field(0.08, gt=0)

    @property
    def snapshot_path(self) -> Path:
        return self.data_dir / "spells_raw.json"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "model.joblib"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
