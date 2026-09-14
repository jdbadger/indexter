import struct
import sys
import types

import pytest

from indexter.config import Settings
from indexter.index.embed import (
    BackendNotAvailable,
    DimensionMismatch,
    FakeEmbedder,
    FastEmbedEmbedder,
    ModelAcquisitionError,
    SentenceTransformerEmbedder,
    make_embedder,
)

DEFAULT_MODEL = Settings().embedding_model


@pytest.fixture
def stub_tokenizer_download(monkeypatch, tmp_path):
    """Patch `hf_hub_download` to hand back a real, minimal tokenizer file
    built locally, so a test can exercise `_load_tokenizer`'s real path with
    no network access and regardless of what's in the local HF cache.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    path = tmp_path / "tokenizer.json"
    Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")).save(str(path))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(path))
    return path


def _unpack(vector: bytes) -> tuple[float, ...]:
    n = len(vector) // 4
    return struct.unpack(f"{n}f", vector)


class _FakeVector:
    """Mimics the numpy row sentence-transformers' `encode()` returns."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def astype(self, _dtype: str) -> "_FakeVector":
        return self

    def tobytes(self) -> bytes:
        return struct.pack(f"{len(self._values)}f", *self._values)


def _make_sentence_transformers_stub(*, dim: int, vectors: dict[str, list[float]] | None = None, fail: bool = False):
    module = types.ModuleType("sentence_transformers")
    calls = {"constructions": 0}

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            if fail:
                raise OSError("could not resolve host")
            self.model_name = model_name
            self.max_seq_length = None
            calls["constructions"] += 1

        def get_embedding_dimension(self) -> int:
            return dim

        def encode(self, texts, *, batch_size, normalize_embeddings, convert_to_numpy):  # noqa: ARG002
            return [_FakeVector(vectors[t]) if vectors and t in vectors else _FakeVector([0.0] * dim) for t in texts]

    module.SentenceTransformer = _FakeSentenceTransformer
    return module, calls


def _make_fastembed_stub(*, dim: int, fail: bool = False):
    module = types.ModuleType("fastembed")
    calls = {"constructions": 0}

    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:  # noqa: ARG002
            if fail:
                raise OSError("could not resolve host")
            calls["constructions"] += 1

        def embed(self, texts, *, batch_size):  # noqa: ARG002
            for _ in texts:
                yield [1.0] + [0.0] * (dim - 1)

    module.TextEmbedding = _FakeTextEmbedding
    return module, calls


class TestConstructionIsLazy:
    def test_sentence_transformer_construction_loads_nothing(self):
        embedder = SentenceTransformerEmbedder(Settings())
        assert embedder._model is None
        assert embedder._tokenizer is None

    def test_fastembed_construction_loads_nothing(self):
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed"))
        assert embedder._model is None
        assert embedder._tokenizer is None

    def test_tokenizer_without_model(self, stub_tokenizer_download):
        embedder = SentenceTransformerEmbedder(Settings())
        embedder.tokenizer()
        assert embedder._model is None

    def test_fastembed_tokenizer_without_model(self, stub_tokenizer_download):
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed"))
        tokenizer = embedder.tokenizer()
        assert tokenizer is not None
        assert embedder._model is None

    def test_tokenizer_download_failure_is_actionable(self, monkeypatch):
        import huggingface_hub

        def _always_fails(*args, **kwargs):  # noqa: ARG001
            raise OSError("no network")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _always_fails)
        embedder = SentenceTransformerEmbedder(Settings(embedding_model="some/uncached-model"))

        with pytest.raises(ModelAcquisitionError, match="some/uncached-model"):
            embedder.tokenizer()


class TestSentenceTransformerBackend:
    def test_model_loads_once_across_batches(self, monkeypatch):
        module, calls = _make_sentence_transformers_stub(dim=4)
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=4))

        embedder.embed(["a", "b"])
        embedder.embed(["c"])

        assert calls["constructions"] == 1

    def test_output_order_and_count(self, monkeypatch):
        module, _ = _make_sentence_transformers_stub(dim=2, vectors={"first": [1.0, 0.0], "second": [0.0, 1.0]})
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=2))

        result = embedder.embed(["first", "second"])

        assert len(result) == 2
        assert _unpack(result[0]) == pytest.approx((1.0, 0.0))
        assert _unpack(result[1]) == pytest.approx((0.0, 1.0))

    def test_empty_input_loads_nothing(self, monkeypatch):
        module, calls = _make_sentence_transformers_stub(dim=4)
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=4))

        assert embedder.embed([]) == []
        assert calls["constructions"] == 0

    def test_dimension_none_is_reported_as_acquisition_failure(self, monkeypatch):
        module, _ = _make_sentence_transformers_stub(dim=4)
        module.SentenceTransformer.get_embedding_dimension = lambda self: None
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=4))

        with pytest.raises(ModelAcquisitionError):
            embedder.embed(["text"])

    def test_dimension_mismatch_names_both_dimensions(self, monkeypatch):
        module, _ = _make_sentence_transformers_stub(dim=384)
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=768))

        with pytest.raises(DimensionMismatch, match="384") as exc_info:
            embedder.embed(["text"])
        assert "768" in str(exc_info.value)
        assert "embedding_dim" in str(exc_info.value)

    def test_download_failure_names_model_and_network(self, monkeypatch):
        module, _ = _make_sentence_transformers_stub(dim=4, fail=True)
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = SentenceTransformerEmbedder(Settings(embedding_dim=4, embedding_model="some/model"))

        with pytest.raises(ModelAcquisitionError, match="some/model") as exc_info:
            embedder.embed(["text"])
        assert "network access" in str(exc_info.value).lower()


class TestFastEmbedBackend:
    def test_not_installed_names_onnx_extra(self):
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed"))
        with pytest.raises(BackendNotAvailable, match="onnx"):
            embedder.embed(["text"])

    def test_model_loads_once_across_batches(self, monkeypatch):
        module, calls = _make_fastembed_stub(dim=4)
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed", embedding_dim=4))

        embedder.embed(["a", "b"])
        embedder.embed(["c"])

        assert calls["constructions"] == 1

    def test_download_failure_names_model_and_network(self, monkeypatch):
        module, _ = _make_fastembed_stub(dim=4, fail=True)
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder(
            Settings(embedding_backend="fastembed", embedding_dim=4, embedding_model="some/model")
        )

        with pytest.raises(ModelAcquisitionError, match="some/model") as exc_info:
            embedder.embed(["text"])
        assert "network access" in str(exc_info.value).lower()

    def test_normalizes_and_matches_dimension(self, monkeypatch):
        module, _ = _make_fastembed_stub(dim=4)
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed", embedding_dim=4))

        (vector,) = embedder.embed(["text"])
        values = _unpack(vector)
        norm = sum(v * v for v in values) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_dimension_mismatch(self, monkeypatch):
        module, _ = _make_fastembed_stub(dim=8)
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed", embedding_dim=4))

        with pytest.raises(DimensionMismatch, match="embedding_dim"):
            embedder.embed(["text"])

    def test_empty_input_loads_nothing(self, monkeypatch):
        module, calls = _make_fastembed_stub(dim=4)
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder(Settings(embedding_backend="fastembed", embedding_dim=4))

        assert embedder.embed([]) == []
        assert calls["constructions"] == 0


class TestBackendSelection:
    def test_default_is_sentence_transformers(self):
        assert isinstance(make_embedder(Settings()), SentenceTransformerEmbedder)

    def test_fastembed_selected_explicitly(self):
        assert isinstance(make_embedder(Settings(embedding_backend="fastembed")), FastEmbedEmbedder)


class TestFakeEmbedder:
    def test_tokenizer_loads_without_model(self):
        embedder = FakeEmbedder()
        embedder.tokenizer()
        assert embedder.tokenizer_loads == 1
        assert embedder.model_loads == 0

    def test_tokenizer_cached_across_calls(self):
        embedder = FakeEmbedder()
        embedder.tokenizer()
        embedder.tokenizer()
        assert embedder.tokenizer_loads == 1

    def test_model_loads_once(self):
        embedder = FakeEmbedder()
        embedder.embed(["a"])
        embedder.embed(["b", "c"])
        assert embedder.model_loads == 1

    def test_empty_input_does_not_count_as_a_load(self):
        embedder = FakeEmbedder()
        assert embedder.embed([]) == []
        assert embedder.model_loads == 0

    def test_deterministic_unit_vectors(self):
        embedder = FakeEmbedder(dim=16)
        (a,) = embedder.embed(["hello world"])
        (b,) = embedder.embed(["hello world"])
        (c,) = embedder.embed(["something else"])

        assert a == b
        assert a != c
        norm = sum(v * v for v in _unpack(a)) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_deterministic_unit_vectors_at_a_non_dividing_dimension(self):
        embedder = FakeEmbedder(dim=3)
        (vector,) = embedder.embed(["a"])
        norm = sum(v * v for v in _unpack(vector)) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_whitespace_tokenizer_offsets(self):
        embedder = FakeEmbedder()
        encoding = embedder.tokenizer().encode("hello  world")
        assert len(encoding.ids) == 2
        starts_ends = encoding.offsets
        assert "hello  world"[starts_ends[0][0] : starts_ends[0][1]] == "hello"
        assert "hello  world"[starts_ends[1][0] : starts_ends[1][1]] == "world"


def _tokenizer_cached_locally(model_name: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_name, filename="tokenizer.json", local_files_only=True)
    except OSError:
        return False
    return True


def _model_cached_locally(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
    except OSError:
        return False
    return True


TOKENIZER_CACHED = _tokenizer_cached_locally(DEFAULT_MODEL)
MODEL_CACHED = _model_cached_locally(DEFAULT_MODEL)


@pytest.mark.skipif(not TOKENIZER_CACHED, reason=f"{DEFAULT_MODEL} tokenizer is not cached locally")
class TestRealTokenizer:
    def test_shipped_truncation_does_not_cap_counts(self):
        embedder = SentenceTransformerEmbedder(Settings())
        tokenizer = embedder.tokenizer()

        encoding = tokenizer.encode("word " * 300, add_special_tokens=False)

        assert len(encoding.ids) > 128


@pytest.mark.skipif(not MODEL_CACHED, reason=f"{DEFAULT_MODEL} model is not cached locally")
class TestRealModel:
    def test_dimension_and_semantic_ordering(self):
        embedder = SentenceTransformerEmbedder(Settings())

        cat_a, cat_b, unrelated = embedder.embed(
            [
                "the cat sat on the mat",
                "a cat is sitting on a mat",
                "quarterly stock prices fell sharply today",
            ]
        )

        assert len(cat_a) // 4 == 384

        def cosine(x: bytes, y: bytes) -> float:
            return sum(a * b for a, b in zip(_unpack(x), _unpack(y), strict=True))

        assert cosine(cat_a, cat_b) > cosine(cat_a, unrelated)
