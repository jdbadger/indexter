from __future__ import annotations

import os

from indexter.config.store import VectorEmbeddingSettings, VectorStoreSettings

# ============================================================================
# Tests for VectorEmbeddingSettings
# ============================================================================


def test_vector_embedding_settings_defaults(clean_env):
    """Test VectorEmbeddingSettings with default values."""
    settings = VectorEmbeddingSettings()

    assert settings.model_name == "BAAI/bge-small-en-v1.5"
    assert settings.dimension == 384


def test_vector_embedding_settings_custom_values(clean_env):
    """Test VectorEmbeddingSettings with custom initialization."""
    settings = VectorEmbeddingSettings(model_name="custom/model", dimension=512)

    assert settings.model_name == "custom/model"
    assert settings.dimension == 512


def test_vector_embedding_settings_env_vars(set_embedding_env_vars):
    """Test VectorEmbeddingSettings loads from environment variables."""
    set_embedding_env_vars(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=768)

    settings = VectorEmbeddingSettings()

    assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.dimension == 768


def test_vector_embedding_settings_env_prefix(clean_env):
    """Test VectorEmbeddingSettings uses INDEXTER_EMBEDDING_ prefix."""
    os.environ["INDEXTER_EMBEDDING_MODEL_NAME"] = "test/model"
    os.environ["INDEXTER_EMBEDDING_DIMENSION"] = "256"

    settings = VectorEmbeddingSettings()

    assert settings.model_name == "test/model"
    assert settings.dimension == 256


def test_vector_embedding_settings_partial_env_override(set_embedding_env_vars):
    """Test partial override with environment variables."""
    set_embedding_env_vars(dimension=1024)

    settings = VectorEmbeddingSettings()

    # model_name should use default
    assert settings.model_name == "BAAI/bge-small-en-v1.5"
    # dimension should use env var
    assert settings.dimension == 1024


def test_vector_embedding_settings_explicit_overrides_env(clean_env):
    """Test explicit values override environment variables."""
    os.environ["INDEXTER_EMBEDDING_MODEL_NAME"] = "env/model"
    os.environ["INDEXTER_EMBEDDING_DIMENSION"] = "512"

    settings = VectorEmbeddingSettings(model_name="explicit/model", dimension=256)

    assert settings.model_name == "explicit/model"
    assert settings.dimension == 256


def test_vector_embedding_settings_model_config(clean_env):
    """Test VectorEmbeddingSettings model_config settings."""
    settings = VectorEmbeddingSettings()

    assert settings.model_config["env_prefix"] == "INDEXTER_EMBEDDING_"


def test_vector_embedding_settings_serialization(clean_env):
    """Test VectorEmbeddingSettings serialization."""
    settings = VectorEmbeddingSettings(model_name="test/model", dimension=512)

    data = settings.model_dump()

    assert data["model_name"] == "test/model"
    assert data["dimension"] == 512


def test_vector_embedding_settings_json_serialization(clean_env):
    """Test VectorEmbeddingSettings JSON serialization."""
    settings = VectorEmbeddingSettings()

    json_str = settings.model_dump_json()

    assert "BAAI/bge-small-en-v1.5" in json_str
    assert "384" in json_str


def test_vector_embedding_settings_dimension_types(clean_env):
    """Test VectorEmbeddingSettings dimension accepts various numeric types."""
    # From string (environment variable scenario)
    os.environ["INDEXTER_EMBEDDING_DIMENSION"] = "768"
    settings1 = VectorEmbeddingSettings()
    assert settings1.dimension == 768

    # Clean for next test
    del os.environ["INDEXTER_EMBEDDING_DIMENSION"]

    # From int
    settings2 = VectorEmbeddingSettings(dimension=1024)
    assert settings2.dimension == 1024


# ============================================================================
# Tests for VectorStoreSettings
# ============================================================================


def test_vector_store_settings_defaults(clean_env):
    """Test VectorStoreSettings with default values."""
    settings = VectorStoreSettings()

    assert settings.host == "localhost"
    assert settings.port == 6333
    assert settings.grpc_port == 6334
    assert settings.api_key is None
    assert settings.use_grpc is False


def test_vector_store_settings_custom_values(clean_env):
    """Test VectorStoreSettings with custom initialization."""
    settings = VectorStoreSettings(
        host="vector.example.com", port=8080, grpc_port=8081, api_key="secret123", use_grpc=True
    )

    assert settings.host == "vector.example.com"
    assert settings.port == 8080
    assert settings.grpc_port == 8081
    assert settings.api_key == "secret123"
    assert settings.use_grpc is True


def test_vector_store_settings_env_vars(set_store_env_vars):
    """Test VectorStoreSettings loads from environment variables."""
    set_store_env_vars(
        host="qdrant.cloud", port=6333, grpc_port=6334, api_key="my-api-key", use_grpc="true"
    )

    settings = VectorStoreSettings()

    assert settings.host == "qdrant.cloud"
    assert settings.port == 6333
    assert settings.grpc_port == 6334
    assert settings.api_key == "my-api-key"
    assert settings.use_grpc is True


def test_vector_store_settings_env_prefix(clean_env):
    """Test VectorStoreSettings uses INDEXTER_STORE_ prefix."""
    os.environ["INDEXTER_STORE_HOST"] = "custom-host"
    os.environ["INDEXTER_STORE_PORT"] = "9999"

    settings = VectorStoreSettings()

    assert settings.host == "custom-host"
    assert settings.port == 9999


def test_vector_store_settings_partial_env_override(set_store_env_vars):
    """Test partial override with environment variables."""
    set_store_env_vars(host="remote.host", api_key="key123")

    settings = VectorStoreSettings()

    # Overridden values
    assert settings.host == "remote.host"
    assert settings.api_key == "key123"

    # Default values
    assert settings.port == 6333
    assert settings.grpc_port == 6334
    assert settings.use_grpc is False


def test_vector_store_settings_explicit_overrides_env(clean_env):
    """Test explicit values override environment variables."""
    os.environ["INDEXTER_STORE_HOST"] = "env-host"
    os.environ["INDEXTER_STORE_PORT"] = "8888"

    settings = VectorStoreSettings(host="explicit-host", port=7777)

    assert settings.host == "explicit-host"
    assert settings.port == 7777


def test_vector_store_settings_url_property(clean_env):
    """Test VectorStoreSettings.url property generates correct URL."""
    settings = VectorStoreSettings()

    assert settings.url == "http://localhost:6333"


def test_vector_store_settings_url_property_custom(clean_env):
    """Test VectorStoreSettings.url property with custom values."""
    settings = VectorStoreSettings(host="vector.example.com", port=8080)

    assert settings.url == "http://vector.example.com:8080"


def test_vector_store_settings_url_property_env(set_store_env_vars):
    """Test VectorStoreSettings.url property with env vars."""
    set_store_env_vars(host="qdrant.cloud", port=443)

    settings = VectorStoreSettings()

    assert settings.url == "http://qdrant.cloud:443"


def test_vector_store_settings_model_config(clean_env):
    """Test VectorStoreSettings model_config settings."""
    settings = VectorStoreSettings()

    assert settings.model_config["env_prefix"] == "INDEXTER_STORE_"


def test_vector_store_settings_api_key_none_by_default(clean_env):
    """Test VectorStoreSettings api_key is None by default."""
    settings = VectorStoreSettings()

    assert settings.api_key is None


def test_vector_store_settings_api_key_from_env(set_store_env_vars):
    """Test VectorStoreSettings api_key from environment."""
    set_store_env_vars(api_key="env-secret")

    settings = VectorStoreSettings()

    assert settings.api_key == "env-secret"


def test_vector_store_settings_use_grpc_boolean(clean_env):
    """Test VectorStoreSettings use_grpc as boolean."""
    settings_false = VectorStoreSettings(use_grpc=False)
    settings_true = VectorStoreSettings(use_grpc=True)

    assert settings_false.use_grpc is False
    assert settings_true.use_grpc is True


def test_vector_store_settings_use_grpc_from_string_env(clean_env):
    """Test VectorStoreSettings use_grpc parses string env vars."""
    # Test various string representations
    test_cases = [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ]

    for env_value, expected in test_cases:
        os.environ["INDEXTER_STORE_USE_GRPC"] = env_value
        settings = VectorStoreSettings()
        assert settings.use_grpc is expected, f"Failed for env_value={env_value}"
        del os.environ["INDEXTER_STORE_USE_GRPC"]


def test_vector_store_settings_serialization(clean_env):
    """Test VectorStoreSettings serialization."""
    settings = VectorStoreSettings(
        host="test-host", port=5555, grpc_port=5556, api_key="test-key", use_grpc=True
    )

    data = settings.model_dump()

    assert data["host"] == "test-host"
    assert data["port"] == 5555
    assert data["grpc_port"] == 5556
    assert data["api_key"] == "test-key"
    assert data["use_grpc"] is True


def test_vector_store_settings_json_serialization(clean_env):
    """Test VectorStoreSettings JSON serialization."""
    settings = VectorStoreSettings(host="json-host", api_key="json-key")

    json_str = settings.model_dump_json()

    assert "json-host" in json_str
    assert "json-key" in json_str


def test_vector_store_settings_port_types(clean_env):
    """Test VectorStoreSettings port accepts various numeric types."""
    # From string (environment variable scenario)
    os.environ["INDEXTER_STORE_PORT"] = "7777"
    os.environ["INDEXTER_STORE_GRPC_PORT"] = "7778"
    settings1 = VectorStoreSettings()
    assert settings1.port == 7777
    assert settings1.grpc_port == 7778

    # Clean for next test
    del os.environ["INDEXTER_STORE_PORT"]
    del os.environ["INDEXTER_STORE_GRPC_PORT"]

    # From int
    settings2 = VectorStoreSettings(port=8888, grpc_port=8889)
    assert settings2.port == 8888
    assert settings2.grpc_port == 8889


def test_vector_store_settings_ipv6_host(clean_env):
    """Test VectorStoreSettings with IPv6 host."""
    settings = VectorStoreSettings(host="::1")

    assert settings.host == "::1"
    assert settings.url == "http://::1:6333"


def test_vector_store_settings_domain_name_host(clean_env):
    """Test VectorStoreSettings with domain name."""
    settings = VectorStoreSettings(host="vector-db.company.internal")

    assert settings.host == "vector-db.company.internal"
    assert settings.url == "http://vector-db.company.internal:6333"


# ============================================================================
# Tests for combined scenarios
# ============================================================================


def test_both_settings_independent_env_prefixes(clean_env):
    """Test that VectorEmbeddingSettings and VectorStoreSettings use independent prefixes."""
    os.environ["INDEXTER_EMBEDDING_MODEL_NAME"] = "embedding-model"
    os.environ["INDEXTER_STORE_HOST"] = "store-host"

    embedding_settings = VectorEmbeddingSettings()
    store_settings = VectorStoreSettings()

    assert embedding_settings.model_name == "embedding-model"
    assert store_settings.host == "store-host"

    # Verify no cross-contamination
    assert store_settings.port == 6333  # default, not affected by EMBEDDING_ vars


def test_both_settings_can_coexist(clean_env):
    """Test that both settings classes can be instantiated together."""
    embedding = VectorEmbeddingSettings(model_name="test/model", dimension=512)
    store = VectorStoreSettings(host="test-host", port=8080)

    assert embedding.model_name == "test/model"
    assert embedding.dimension == 512
    assert store.host == "test-host"
    assert store.port == 8080


def test_settings_immutability_after_creation(clean_env):
    """Test that settings remain stable after environment changes."""
    os.environ["INDEXTER_EMBEDDING_DIMENSION"] = "512"
    os.environ["INDEXTER_STORE_PORT"] = "7777"

    embedding = VectorEmbeddingSettings()
    store = VectorStoreSettings()

    # Change environment after creation
    os.environ["INDEXTER_EMBEDDING_DIMENSION"] = "1024"
    os.environ["INDEXTER_STORE_PORT"] = "8888"

    # Original settings should remain unchanged
    assert embedding.dimension == 512
    assert store.port == 7777
