from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    spellbook_api_url: str = "http://localhost:3000"
    score_threshold: float = 0.08
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    page_size_max: int = 100
    svd_components: int = 80

    @property
    def snapshot_path(self) -> Path:
        return self.data_dir / "spells_raw.json"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "model.joblib"


settings = Settings()
