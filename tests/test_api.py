from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import loader
from app import search as search_module
from app.config import settings
from app.main import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", ROOT / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "embedding_model", "")

    def api_unavailable(*args, **kwargs):
        raise RuntimeError("API is disabled in tests")

    monkeypatch.setattr(loader, "load_from_api", api_unavailable)
    monkeypatch.setattr(search_module, "engine", None)
    return TestClient(app)


def test_rebuild_is_disabled_without_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "")
    assert client.post("/admin/rebuild").status_code == 404


def test_rebuild_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "secret")
    assert client.post("/admin/rebuild").status_code == 401
    assert client.post("/admin/rebuild", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_rebuild_replaces_engine(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "secret")
    assert client.get("/health").status_code == 503

    response = client.post("/admin/rebuild?source=file", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["model"]["dataHash"]
    assert client.get("/health").status_code == 200
    assert client.get("/search", params={"q": "огненный шар"}).json()["data"]


def test_cors_allows_only_configured_origins(client):
    allowed = client.get("/health", headers={"Origin": "http://localhost:8080"})
    denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8080"
    assert "access-control-allow-origin" not in denied.headers
