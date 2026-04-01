from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _as_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Cannot embed an empty query.")
    return text


def _as_vector(value: Any) -> List[float]:
    output = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, (list, tuple)):
        raise ValueError("Embedding backend returned a non-vector output.")
    return [float(x) for x in output]


def _dedup(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _split_models(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return _dedup(part.strip() for part in str(raw).split(","))


def _normalize_models(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return _split_models(values)
    if isinstance(values, (list, tuple, set)):
        return _dedup(str(item or "").strip() for item in values)
    return []


class OpenAIEmbedder:
    def __init__(self, *, model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> None:
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
        response = self._client.embeddings.create(model=self.model, input=_as_text(text))
        return _as_vector(response.data[0].embedding)


class HuggingFaceEmbedder:
    """
    Generic HuggingFace embedder.
    - BGE-M3: tries FlagEmbedding first.
    - Others: uses sentence-transformers.
    """

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = str(model or "").strip() or "BAAI/bge-m3"
        self.normalize_embeddings = bool(normalize_embeddings)
        self._backend = ""
        self._model: Any = None

        if "bge-m3" in self.model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
            except Exception:
                pass
            else:
                self._model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
                self._backend = "flagembedding"

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError(
                    "Install `sentence-transformers` (or `flagembedding` for BGE-M3)."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence-transformers"

    def embed(self, text: str) -> List[float]:
        value = _as_text(text)
        if self._backend == "flagembedding":
            output = self._model.encode(value, return_dense=True)["dense_vecs"]
        else:
            output = self._model.encode(value, normalize_embeddings=self.normalize_embeddings)
        return _as_vector(output)


class BGEEmbedder(HuggingFaceEmbedder):
    """Backward-compatible alias."""

    def __init__(self, *, model: str = "BAAI/bge-m3", use_fp16: bool = True) -> None:
        super().__init__(model=model, use_fp16=use_fp16)


class EmbeddingService:
    _ALIASES = {
        "openai": "openai",
        "oai": "openai",
        "hf": "hf",
        "huggingface": "hf",
        "sentence-transformers": "hf",
        "sentence_transformers": "hf",
        "st": "hf",
        "bge": "hf",
        "bge-m3": "hf",
        "baai": "hf",
        "flagembedding": "hf",
    }

    def __init__(
        self,
        *,
        default_provider: Optional[str] = None,
        default_hf_model: Optional[str] = None,
        default_bge_model: Optional[str] = None,  # backward compatibility
        default_openai_model: Optional[str] = None,
        hf_models: Optional[Iterable[str]] = None,
        openai_models: Optional[Iterable[str]] = None,
        model_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_provider = self._canonical_provider(default_provider or "hf")
        self.default_hf_model = (
            default_hf_model
            or default_bge_model
            or "BAAI/bge-m3"
        ).strip()
        self.default_openai_model = (default_openai_model or "text-embedding-3-small").strip()

        self._cache: Dict[Tuple[str, str], object] = {}
        self._defaults = {"hf": self.default_hf_model, "openai": self.default_openai_model}
        self._catalog = {
            "hf": _dedup([self.default_hf_model] + _normalize_models(hf_models)),
            "openai": _dedup([self.default_openai_model] + _normalize_models(openai_models)),
        }
        self._merge_catalog(model_catalog)

    def _canonical_provider(self, provider: Optional[str]) -> str:
        value = str(provider or "").strip().lower()
        if not value:
            return "hf"
        return self._ALIASES.get(value, value)

    def _merge_catalog(self, catalog: Optional[Dict[str, Any]]) -> None:
        if not catalog or not isinstance(catalog, dict):
            return

        for raw_provider, raw_models in catalog.items():
            provider = self._canonical_provider(raw_provider)
            if provider not in self._catalog:
                continue
            models = _normalize_models(raw_models)
            self._catalog[provider] = _dedup(self._catalog[provider] + models)

    def _resolve(self, provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
        resolved_provider = self._canonical_provider(provider or self.default_provider)
        if resolved_provider not in self._defaults:
            supported = ", ".join(sorted(self._defaults))
            raise ValueError(f"Unsupported embedding provider: {resolved_provider}. Supported: {supported}")
        resolved_model = str(model or "").strip() or self._defaults[resolved_provider]
        return resolved_provider, resolved_model

    def _new_embedder(self, provider: str, model: str) -> object:
        if provider == "openai":
            return OpenAIEmbedder(model=model)
        return HuggingFaceEmbedder(model=model)

    def _get_embedder(self, provider: str, model: str) -> object:
        key = (provider, model)
        embedder = self._cache.get(key)
        if embedder is None:
            embedder = self._new_embedder(provider, model)
            self._cache[key] = embedder
        return embedder

    def embed(self, text: str, *, provider: Optional[str] = None, model: Optional[str] = None) -> List[float]:
        resolved_provider, resolved_model = self._resolve(provider, model)
        return self._get_embedder(resolved_provider, resolved_model).embed(text)

    def list_options(self) -> Dict[str, Any]:
        providers: List[Dict[str, Any]] = []
        options: List[Dict[str, str]] = []

        for provider, default_model in self._defaults.items():
            models = _dedup(self._catalog.get(provider, []) + [default_model])
            providers.append({"provider": provider, "default_model": default_model, "models": models})
            options.extend(
                {
                    "provider": provider,
                    "model": model,
                    "value": f"{provider}:{model}",
                    "label": f"Embedding: {provider}/{model}",
                }
                for model in models
            )

        resolved_provider, resolved_model = self._resolve(self.default_provider, None)
        return {
            "default_provider": resolved_provider,
            "default_model": resolved_model,
            "default_value": f"{resolved_provider}:{resolved_model}",
            "providers": providers,
            "options": options,
        }
