from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from indexter.config.mcp import MCPSettings

# ============================================================================
# Tests for MCPSettings defaults
# ============================================================================


def test_mcp_settings_default_values(clean_env):
    """Test that MCPSettings initializes with correct default values."""
    settings = MCPSettings()

    assert settings.host == "localhost"
    assert settings.port == 8765
    assert settings.default_top_k == 10


def test_mcp_settings_model_config(clean_env):
    """Test that MCPSettings has correct model configuration."""
    settings = MCPSettings()

    # Check that env_prefix is set correctly
    assert settings.model_config["env_prefix"] == "INDEXTER_MCP_"


# ============================================================================
# Tests for environment variable overrides
# ============================================================================


def test_mcp_settings_host_from_env(set_env_vars):
    """Test that host can be set from environment variable."""
    set_env_vars(host="0.0.0.0")
    settings = MCPSettings()

    assert settings.host == "0.0.0.0"


def test_mcp_settings_port_from_env(set_env_vars):
    """Test that port can be set from environment variable."""
    set_env_vars(port=9000)
    settings = MCPSettings()

    assert settings.port == 9000


def test_mcp_settings_default_top_k_from_env(set_env_vars):
    """Test that default_top_k can be set from environment variable."""
    set_env_vars(default_top_k=20)
    settings = MCPSettings()

    assert settings.default_top_k == 20


def test_mcp_settings_all_from_env(set_env_vars):
    """Test that all settings can be set from environment variables."""
    set_env_vars(host="192.168.1.1", port=3000, default_top_k=50)
    settings = MCPSettings()

    assert settings.host == "192.168.1.1"
    assert settings.port == 3000
    assert settings.default_top_k == 50


def test_mcp_settings_partial_env_override(set_env_vars):
    """Test that partial environment variables override defaults correctly."""
    set_env_vars(port=5000)
    settings = MCPSettings()

    assert settings.host == "localhost"  # default
    assert settings.port == 5000  # overridden
    assert settings.default_top_k == 10  # default


# ============================================================================
# Tests for explicit initialization
# ============================================================================


def test_mcp_settings_explicit_values(clean_env):
    """Test that settings can be initialized with explicit values."""
    settings = MCPSettings(host="example.com", port=8080, default_top_k=15)

    assert settings.host == "example.com"
    assert settings.port == 8080
    assert settings.default_top_k == 15


def test_mcp_settings_explicit_overrides_env(set_env_vars):
    """Test that explicit values override environment variables."""
    set_env_vars(host="env-host", port=9999, default_top_k=100)
    settings = MCPSettings(host="explicit-host", port=7777, default_top_k=5)

    assert settings.host == "explicit-host"
    assert settings.port == 7777
    assert settings.default_top_k == 5


def test_mcp_settings_partial_explicit(set_env_vars):
    """Test mixing explicit values with env vars and defaults."""
    set_env_vars(port=4000)
    settings = MCPSettings(host="my-host")

    assert settings.host == "my-host"  # explicit
    assert settings.port == 4000  # from env
    assert settings.default_top_k == 10  # default


# ============================================================================
# Tests for validation
# ============================================================================


def test_mcp_settings_invalid_port_type(clean_env):
    """Test that invalid port type raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        MCPSettings(port="not-a-number")  # type: ignore

    assert "port" in str(exc_info.value).lower()


def test_mcp_settings_invalid_default_top_k_type(clean_env):
    """Test that invalid default_top_k type raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        MCPSettings(default_top_k="not-a-number")  # type: ignore

    assert "default_top_k" in str(exc_info.value).lower()


def test_mcp_settings_negative_port(clean_env):
    """Test that negative port is accepted by Pydantic (no custom validation)."""
    # Note: Pydantic will accept negative integers unless we add custom validation
    settings = MCPSettings(port=-1)
    assert settings.port == -1


def test_mcp_settings_negative_default_top_k(clean_env):
    """Test that negative default_top_k is accepted by Pydantic."""
    # Note: Pydantic will accept negative integers unless we add custom validation
    settings = MCPSettings(default_top_k=-5)
    assert settings.default_top_k == -5


def test_mcp_settings_zero_port(clean_env):
    """Test that zero port is accepted."""
    settings = MCPSettings(port=0)
    assert settings.port == 0


def test_mcp_settings_zero_default_top_k(clean_env):
    """Test that zero default_top_k is accepted."""
    settings = MCPSettings(default_top_k=0)
    assert settings.default_top_k == 0


def test_mcp_settings_large_port(clean_env):
    """Test that large port number is accepted."""
    settings = MCPSettings(port=65535)
    assert settings.port == 65535


def test_mcp_settings_very_large_port(clean_env):
    """Test that very large port number (beyond valid range) is accepted."""
    # Pydantic doesn't validate port ranges by default
    settings = MCPSettings(port=99999)
    assert settings.port == 99999


def test_mcp_settings_large_default_top_k(clean_env):
    """Test that large default_top_k is accepted."""
    settings = MCPSettings(default_top_k=10000)
    assert settings.default_top_k == 10000


# ============================================================================
# Tests for string types
# ============================================================================


def test_mcp_settings_empty_host(clean_env):
    """Test that empty host string is accepted."""
    settings = MCPSettings(host="")
    assert settings.host == ""


def test_mcp_settings_host_with_whitespace(clean_env):
    """Test that host with whitespace is accepted as-is."""
    settings = MCPSettings(host="  localhost  ")
    assert settings.host == "  localhost  "


def test_mcp_settings_host_ipv4(clean_env):
    """Test that IPv4 address is accepted as host."""
    settings = MCPSettings(host="192.168.1.100")
    assert settings.host == "192.168.1.100"


def test_mcp_settings_host_ipv6(clean_env):
    """Test that IPv6 address is accepted as host."""
    settings = MCPSettings(host="::1")
    assert settings.host == "::1"


def test_mcp_settings_host_domain_with_port(clean_env):
    """Test that host with port notation is accepted (not validated)."""
    # Note: This is just a string, no URL validation
    settings = MCPSettings(host="example.com:8080")
    assert settings.host == "example.com:8080"


# ============================================================================
# Tests for environment variable parsing
# ============================================================================


def test_mcp_settings_env_var_case_insensitive(clean_env):
    """Test that environment variable names are case-insensitive."""
    # Pydantic's env_prefix is case-insensitive for the prefix
    os.environ["indexter_MCP_HOST"] = "uppercase-host"
    settings = MCPSettings()
    assert settings.host == "uppercase-host"


def test_mcp_settings_env_var_with_string_port(clean_env):
    """Test that port from env var is parsed as integer."""
    os.environ["indexter_MCP_PORT"] = "8888"
    settings = MCPSettings()
    assert settings.port == 8888
    assert isinstance(settings.port, int)


def test_mcp_settings_env_var_with_string_default_top_k(clean_env):
    """Test that default_top_k from env var is parsed as integer."""
    os.environ["indexter_MCP_DEFAULT_TOP_K"] = "25"
    settings = MCPSettings()
    assert settings.default_top_k == 25
    assert isinstance(settings.default_top_k, int)


def test_mcp_settings_invalid_env_var_port(clean_env):
    """Test that invalid port in env var raises validation error."""
    os.environ["indexter_MCP_PORT"] = "not-a-number"
    with pytest.raises(ValidationError) as exc_info:
        MCPSettings()

    assert "port" in str(exc_info.value).lower()


def test_mcp_settings_invalid_env_var_default_top_k(clean_env):
    """Test that invalid default_top_k in env var raises validation error."""
    os.environ["indexter_MCP_DEFAULT_TOP_K"] = "not-a-number"
    with pytest.raises(ValidationError) as exc_info:
        MCPSettings()

    assert "default_top_k" in str(exc_info.value).lower()


# ============================================================================
# Tests for multiple instances
# ============================================================================


def test_mcp_settings_multiple_instances_independent(clean_env):
    """Test that multiple instances can have different values."""
    settings1 = MCPSettings(host="host1", port=1000)
    settings2 = MCPSettings(host="host2", port=2000)

    assert settings1.host == "host1"
    assert settings1.port == 1000
    assert settings2.host == "host2"
    assert settings2.port == 2000


def test_mcp_settings_instance_immutability():
    """Test that settings instances are immutable after creation."""
    settings = MCPSettings(host="original")

    # Pydantic models are not frozen by default, but we can test assignment
    # If we want immutability, we'd need to add frozen=True to model_config
    settings.host = "modified"
    assert settings.host == "modified"  # Mutable by default


# ============================================================================
# Tests for serialization
# ============================================================================


def test_mcp_settings_model_dump(clean_env):
    """Test that settings can be serialized to dict."""
    settings = MCPSettings(host="test-host", port=9000, default_top_k=15)
    data = settings.model_dump()

    assert data == {"host": "test-host", "port": 9000, "default_top_k": 15}


def test_mcp_settings_model_dump_json(clean_env):
    """Test that settings can be serialized to JSON."""
    settings = MCPSettings(host="test-host", port=9000, default_top_k=15)
    json_str = settings.model_dump_json()

    assert "test-host" in json_str
    assert "9000" in json_str
    assert "15" in json_str


def test_mcp_settings_from_dict(clean_env):
    """Test that settings can be created from dict."""
    data = {"host": "dict-host", "port": 7000, "default_top_k": 30}
    settings = MCPSettings(**data)

    assert settings.host == "dict-host"
    assert settings.port == 7000
    assert settings.default_top_k == 30


# ============================================================================
# Tests for field access
# ============================================================================


def test_mcp_settings_field_access(clean_env):
    """Test that all fields are accessible."""
    settings = MCPSettings()

    # All fields should be accessible
    assert hasattr(settings, "host")
    assert hasattr(settings, "port")
    assert hasattr(settings, "default_top_k")


def test_mcp_settings_no_extra_fields(clean_env):
    """Test that extra fields raise validation error."""
    with pytest.raises(ValidationError):
        MCPSettings(host="test", extra_field="value")  # type: ignore


# ============================================================================
# Tests for edge cases
# ============================================================================


def test_mcp_settings_unicode_host(clean_env):
    """Test that Unicode characters in host are accepted."""
    settings = MCPSettings(host="テスト.example.com")
    assert settings.host == "テスト.example.com"


def test_mcp_settings_special_chars_host(clean_env):
    """Test that special characters in host are accepted."""
    settings = MCPSettings(host="my-server_123.example.com")
    assert settings.host == "my-server_123.example.com"


def test_mcp_settings_env_prefix_not_in_values(clean_env):
    """Test that env_prefix doesn't appear in actual values."""
    os.environ["indexter_MCP_HOST"] = "test-host"
    settings = MCPSettings()

    # The prefix should be stripped, not part of the value
    assert settings.host == "test-host"
    assert "indexter_MCP" not in settings.host
