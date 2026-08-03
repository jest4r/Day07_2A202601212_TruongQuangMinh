from __future__ import annotations

import hashlib
import math
import os
from dotenv import load_dotenv

# Multilingual model suitable for the Vietnamese corpora used in this Lab.
# The local backend remains optional; required checkpoints use MockEmbedder.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        load_dotenv()
        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
        self._backend_name = self.model_name
        self.model = SentenceTransformer(self.model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        load_dotenv()
        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        self._backend_name = self.model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


class OllamaEmbedder:
    """Ollama API-backed embedder."""

    def __init__(self, model_name: str | None = None, host: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._backend_name = f"ollama-{self.model_name}"

    def __call__(self, text: str) -> list[float]:
        import json
        import urllib.request

        # Giới hạn tối đa 3000 ký tự cho prompt embedding để không vượt context length của Ollama
        truncated_text = text[:3000] if text else ""
        url = f"{self.host}/api/embeddings"
        payload = json.dumps({"model": self.model_name, "prompt": truncated_text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return [float(value) for value in res_data.get("embedding", [])]


_mock_embed = MockEmbedder()


def get_embedder():
    """Tự động chọn Embedder dựa trên biến môi trường EMBEDDING_PROVIDER trong .env"""
    load_dotenv()
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "ollama":
        return OllamaEmbedder()
    elif provider == "local":
        return LocalEmbedder()
    elif provider == "openai":
        return OpenAIEmbedder()
    return _mock_embed


