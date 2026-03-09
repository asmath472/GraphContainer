from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Install `openai` first.") from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.model = model
        self._client = OpenAI(api_key=key)

    def embed(self, text: str) -> List[float]:
        value = str(text or "").strip()
        if not value:
            raise ValueError("Cannot embed an empty query.")

        response = self._client.embeddings.create(
            model=self.model,
            input=value,
        )
        embedding = response.data[0].embedding
        return [float(x) for x in embedding]


class BGEEmbedder:
    def __init__(
        self,
        *,
        model: str = "BAAI/bge-m3",
        use_fp16: bool = True,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            raise RuntimeError(
                "FlagEmbedding is not installed. Install `flagembedding` first."
            ) from exc

        self.model_name = model
        self._model = BGEM3FlagModel(model, use_fp16=use_fp16)

    def embed(self, text: str) -> List[float]:
        value = str(text or "").strip()
        if not value:
            raise ValueError("Cannot embed an empty query.")
        output = self._model.encode(value, return_dense=True)["dense_vecs"]
        if hasattr(output, "tolist"):
            output = output.tolist()
        if output and isinstance(output[0], list):
            output = output[0]
        return [float(x) for x in output]


class EmbedderRouter:
    def __init__(
        self,
        *,
        default_provider: Optional[str] = None,
        default_bge_model: Optional[str] = None,
        default_openai_model: Optional[str] = None,
    ) -> None:
        self.default_provider = (default_provider or os.getenv("GRAPH_RAG_EMBEDDER", "bge")).strip().lower()
        self.default_bge_model = (
            default_bge_model or os.getenv("GRAPH_RAG_BGE_MODEL", os.getenv("GRAPH_RAG_EMBED_MODEL", "BAAI/bge-m3"))
        ).strip()
        self.default_openai_model = (
            default_openai_model
            or os.getenv("GRAPH_RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small")
        ).strip()
        self._cache: Dict[Tuple[str, str], object] = {}

    def _normalize_provider(self, provider: Optional[str]) -> str:
        value = str(provider or self.default_provider).strip().lower()
        if value in {"baai", "bge", "bge-m3", "flagembedding"}:
            return "bge"
        if value in {"openai", "oai"}:
            return "openai"
        return value

    def _resolve(self, provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
        normalized_provider = self._normalize_provider(provider)
        chosen_model = str(model or "").strip()
        if not chosen_model:
            if normalized_provider == "openai":
                chosen_model = self.default_openai_model
            else:
                normalized_provider = "bge"
                chosen_model = self.default_bge_model
        return normalized_provider, chosen_model

    def _get_embedder(self, provider: str, model: str) -> object:
        key = (provider, model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if provider == "openai":
            embedder = OpenAIEmbedder(model=model)
        elif provider == "bge":
            embedder = BGEEmbedder(model=model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        self._cache[key] = embedder
        return embedder

    def embed(self, text: str, *, provider: Optional[str] = None, model: Optional[str] = None) -> List[float]:
        resolved_provider, resolved_model = self._resolve(provider, model)
        embedder = self._get_embedder(resolved_provider, resolved_model)
        return embedder.embed(text)
