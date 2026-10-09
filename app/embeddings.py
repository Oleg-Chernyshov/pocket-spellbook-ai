from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import normalize
from tokenizers import Tokenizer

from app.config import settings

logger = logging.getLogger(__name__)

TOKENIZER_FILE = "onnx/tokenizer.json"
BATCH_SIZE = 32


class Embedder:
    """Sentence encoder for E5-family models: expects "query: " / "passage: " prefixes and mean pooling."""

    def __init__(self, model_path: Path, tokenizer_path: Path, max_tokens: int):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_tokens)
        self.tokenizer.enable_padding()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {text}" for text in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), BATCH_SIZE):
            encodings = self.tokenizer.encode_batch(texts[start : start + BATCH_SIZE])
            ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
            mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = self.session.run(None, {name: value for name, value in feeds.items() if name in self.input_names})[0]
            weights = mask[..., None].astype(np.float32)
            chunks.append((hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None))
        return normalize(np.vstack(chunks)).astype(np.float32)


def embedding_config() -> dict | None:
    if not settings.embedding_model:
        return None
    return {
        "model": settings.embedding_model,
        "revision": settings.embedding_revision,
        "file": settings.embedding_file,
        "maxTokens": settings.embedding_max_tokens,
    }


def _fetch(repo: str, revision: str, filename: str, local_dir: Path) -> Path:
    path = local_dir / filename
    if path.exists():
        return path
    logger.info("Downloading %s/%s@%s", repo, filename, revision[:12])
    return Path(hf_hub_download(repo, filename, revision=revision, local_dir=local_dir))


_embedders: dict[tuple, Embedder] = {}


def get_embedder() -> Embedder | None:
    config = embedding_config()
    if config is None:
        return None
    key = tuple(config.values())
    if key not in _embedders:
        local_dir = settings.embeddings_dir / config["model"].replace("/", "--") / config["revision"][:12]
        try:
            model_path = _fetch(config["model"], config["revision"], config["file"], local_dir)
            tokenizer_path = _fetch(config["model"], config["revision"], TOKENIZER_FILE, local_dir)
            _embedders[key] = Embedder(model_path, tokenizer_path, config["maxTokens"])
        except Exception as exc:
            logger.warning("Semantic search is disabled, failed to load %s: %s", config["model"], exc)
            return None
        logger.info("Loaded embedding model %s", config["model"])
    return _embedders[key]
