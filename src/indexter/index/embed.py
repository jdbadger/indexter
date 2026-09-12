"""Embedding backends: turn composed text into vectors for `vectors`.

Two lazy backends behind one `Embedder` protocol -- see design.md decision 8.
Constructing an embedder loads nothing; `tokenizer()` loads only the
tokenizer (no ML framework), and `embed()` loads the model on first call.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from indexter.config import Settings

TOKENIZER_FILENAME = "tokenizer.json"


class EmbeddingError(Exception):
    """Base class for embedding-layer errors."""


class BackendNotAvailable(EmbeddingError):
    """The configured backend's package isn't importable."""

    def __init__(self, backend: str, extra: str) -> None:
        self.backend = backend
        self.extra = extra
        super().__init__(
            f"The {backend!r} embedding backend requires the {extra!r} extra. "
            f"Install it with `uv add indexter[{extra}]` (or `pip install indexter[{extra}]`)."
        )


class ModelAcquisitionError(EmbeddingError):
    """The model or tokenizer isn't cached locally and couldn't be downloaded."""

    def __init__(self, model_name: str, detail: str) -> None:
        self.model_name = model_name
        self.detail = detail
        super().__init__(
            f"Could not load {model_name!r}: {detail}. Network access is needed once to "
            "download this model; after that it is cached locally and works offline."
        )


class DimensionMismatch(EmbeddingError):
    """The model's actual output dimension doesn't match configured `embedding_dim`."""

    def __init__(self, model_name: str, actual: int, configured: int) -> None:
        self.model_name = model_name
        self.actual = actual
        self.configured = configured
        super().__init__(
            f"{model_name!r} produces {actual}-dimensional vectors, but embedding_dim is "
            f"configured as {configured}. Set embedding_dim = {actual}."
        )


@runtime_checkable
class Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...
    @property
    def offsets(self) -> list[tuple[int, int]]: ...


@runtime_checkable
class TokenizerLike(Protocol):
    def encode(self, text: str, /, *, add_special_tokens: bool = ...) -> Encoding: ...


@runtime_checkable
class Embedder(Protocol):
    model_name: str

    def tokenizer(self) -> TokenizerLike: ...
    def embed(self, texts: Sequence[str]) -> list[bytes]: ...


def _pack_f32(values: Sequence[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _load_tokenizer(model_name: str) -> TokenizerLike:
    """Load `model_name`'s tokenizer, preferring the local Hugging Face cache
    and falling back to a download. Truncation and padding are disabled --
    the shipped tokenizer.json for all-MiniLM-L6-v2 has both enabled, which
    would silently cap every token count at 128 (see design.md decision 7).
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    try:
        path = hf_hub_download(model_name, filename=TOKENIZER_FILENAME, local_files_only=True)
    except OSError:
        try:
            path = hf_hub_download(model_name, filename=TOKENIZER_FILENAME)
        except OSError as e:
            raise ModelAcquisitionError(model_name, str(e)) from e

    tok = Tokenizer.from_file(path)
    tok.no_truncation()
    tok.no_padding()
    return tok


class SentenceTransformerEmbedder:
    """Default backend: sentence-transformers (torch)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings if settings is not None else Settings()
        self.model_name = settings.embedding_model
        self._settings = settings
        self._tokenizer: TokenizerLike | None = None
        self._model = None

    def tokenizer(self) -> TokenizerLike:
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_name)
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(self.model_name)
        except OSError as e:
            raise ModelAcquisitionError(self.model_name, str(e)) from e

        model.max_seq_length = self._settings.embed_max_tokens
        actual_dim = model.get_embedding_dimension()
        if actual_dim is None:
            raise ModelAcquisitionError(self.model_name, "the model reports no embedding dimension")
        if actual_dim != self._settings.embedding_dim:
            raise DimensionMismatch(self.model_name, actual_dim, self._settings.embedding_dim)

        self._model = model
        return self._model

    def embed(self, texts: Sequence[str]) -> list[bytes]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._settings.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.astype("float32").tobytes() for vector in vectors]


class FastEmbedEmbedder:
    """Optional backend: fastembed (ONNX runtime, the `onnx` extra)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings if settings is not None else Settings()
        self.model_name = settings.embedding_model
        self._settings = settings
        self._tokenizer: TokenizerLike | None = None
        self._model = None
        self._dimension_checked = False

    def tokenizer(self) -> TokenizerLike:
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_name)
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextEmbedding  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise BackendNotAvailable("fastembed", "onnx") from e

        try:
            model = TextEmbedding(model_name=self.model_name)
        except OSError as e:
            raise ModelAcquisitionError(self.model_name, str(e)) from e

        self._model = model
        return self._model

    def embed(self, texts: Sequence[str]) -> list[bytes]:
        if not texts:
            return []
        model = self._load_model()
        out = []
        for raw in model.embed(list(texts), batch_size=self._settings.embed_batch_size):
            values = [float(v) for v in raw]
            if not self._dimension_checked:
                self._dimension_checked = True
                if len(values) != self._settings.embedding_dim:
                    raise DimensionMismatch(self.model_name, len(values), self._settings.embedding_dim)
            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                values = [v / norm for v in values]
            out.append(_pack_f32(values))
        return out


def make_embedder(settings: Settings) -> Embedder:
    """Select the embedder backend named by `settings.embedding_backend`."""
    if settings.embedding_backend == "fastembed":
        return FastEmbedEmbedder(settings)
    return SentenceTransformerEmbedder(settings)


_WORD_RE = re.compile(r"\S+")


class _WhitespaceEncoding:
    def __init__(self, ids: list[int], offsets: list[tuple[int, int]]) -> None:
        self.ids = ids
        self.offsets = offsets


class _WhitespaceTokenizer:
    """FakeEmbedder's tokenizer: splits on whitespace, one token per run of
    non-space characters, with character offsets into the original text.
    """

    def encode(self, text: str, *, add_special_tokens: bool = True) -> _WhitespaceEncoding:  # noqa: ARG002
        ids = []
        offsets = []
        for match in _WORD_RE.finditer(text):
            ids.append(hash(match.group()) & 0xFFFF)
            offsets.append((match.start(), match.end()))
        return _WhitespaceEncoding(ids, offsets)


def _hash_vector(text: str, dim: int) -> bytes:
    """A deterministic, hash-derived unit vector: same text -> same vector,
    different text -> (almost certainly) a different one.
    """
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for i in range(0, len(digest) - 1, 2):
            if len(values) >= dim:
                break
            raw = int.from_bytes(digest[i : i + 2], "big")
            values.append((raw / 32767.5) - 1.0)
        counter += 1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return _pack_f32([v / norm for v in values])


class FakeEmbedder:
    """Test double: no real tokenizer or model, just deterministic output and
    load counters so tests can assert "the model was never loaded".
    """

    def __init__(self, dim: int = 384, model_name: str = "fake-model") -> None:
        self.model_name = model_name
        self.dim = dim
        self.tokenizer_loads = 0
        self.model_loads = 0
        self._tokenizer: TokenizerLike | None = None
        self._model_loaded = False

    def tokenizer(self) -> TokenizerLike:
        if self._tokenizer is None:
            self.tokenizer_loads += 1
            self._tokenizer = _WhitespaceTokenizer()
        return self._tokenizer

    def embed(self, texts: Sequence[str]) -> list[bytes]:
        if not texts:
            return []
        if not self._model_loaded:
            self.model_loads += 1
            self._model_loaded = True
        return [_hash_vector(text, self.dim) for text in texts]
