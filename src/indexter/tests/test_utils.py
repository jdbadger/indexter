"""Tests for the utils module."""

import hashlib
from unittest.mock import patch

from indexter.utils import compute_hash

# --- Basic functionality tests ---


def test_compute_hash_basic():
    """Test compute_hash with basic string."""
    content = "hello"
    result = compute_hash(content)

    # Verify it returns a string
    assert isinstance(result, str)

    # Verify it's a valid SHA256 hash (64 hex characters)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_hash_matches_hashlib():
    """Test compute_hash matches hashlib.sha256 directly."""
    content = "test content"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_empty_string():
    """Test compute_hash with empty string."""
    content = ""
    result = compute_hash(content)

    # SHA256 of empty string is well-known
    expected = hashlib.sha256(b"").hexdigest()
    assert result == expected


def test_compute_hash_empty_string_is_deterministic():
    """Test compute_hash always returns same hash for empty string."""
    hash1 = compute_hash("")
    hash2 = compute_hash("")

    assert hash1 == hash2


def test_compute_hash_single_character():
    """Test compute_hash with single character."""
    content = "a"
    result = compute_hash(content)

    expected = hashlib.sha256(b"a").hexdigest()
    assert result == expected


def test_compute_hash_long_string():
    """Test compute_hash with long string."""
    content = "x" * 10000
    result = compute_hash(content)

    # Should still return valid hash
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# --- Determinism tests ---


def test_compute_hash_is_deterministic():
    """Test that compute_hash is deterministic."""
    content = "deterministic test"
    hash1 = compute_hash(content)
    hash2 = compute_hash(content)
    hash3 = compute_hash(content)

    assert hash1 == hash2 == hash3


def test_compute_hash_same_content_same_hash():
    """Test same content produces same hash across calls."""
    for _ in range(5):
        assert compute_hash("same") == compute_hash("same")


# --- Sensitivity to changes ---


def test_compute_hash_different_content():
    """Test different content produces different hashes."""
    hash1 = compute_hash("content1")
    hash2 = compute_hash("content2")

    assert hash1 != hash2


def test_compute_hash_case_sensitive():
    """Test compute_hash is case-sensitive."""
    hash_lower = compute_hash("hello")
    hash_upper = compute_hash("HELLO")

    assert hash_lower != hash_upper


def test_compute_hash_whitespace_matters():
    """Test compute_hash is sensitive to whitespace."""
    hash_no_space = compute_hash("hello world")
    hash_extra_space = compute_hash("hello  world")
    hash_no_space_at_end = compute_hash("hello world ")

    assert hash_no_space != hash_extra_space
    assert hash_no_space != hash_no_space_at_end


def test_compute_hash_single_char_difference():
    """Test single character difference produces different hash."""
    hash1 = compute_hash("abcdef")
    hash2 = compute_hash("abcxef")

    assert hash1 != hash2


def test_compute_hash_order_matters():
    """Test character order matters."""
    hash1 = compute_hash("abc")
    hash2 = compute_hash("bca")

    assert hash1 != hash2


# --- Unicode and special characters ---


def test_compute_hash_unicode():
    """Test compute_hash with unicode characters."""
    content = "Hello 世界 🌍"
    result = compute_hash(content)

    # Should return valid hash
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_hash_unicode_consistency():
    """Test unicode hashes are consistent."""
    content = "Café"
    hash1 = compute_hash(content)
    hash2 = compute_hash(content)

    assert hash1 == hash2


def test_compute_hash_emoji():
    """Test compute_hash with emoji."""
    content = "🔒🔐🗝️"
    result = compute_hash(content)

    # Should return valid hash
    assert len(result) == 64


def test_compute_hash_newlines():
    """Test compute_hash with newlines."""
    content = "line1\nline2\nline3"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_tabs_and_spaces():
    """Test compute_hash with tabs and spaces."""
    content = "spaces  \ttabs"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_carriage_return():
    """Test compute_hash with carriage return."""
    content = "line1\r\nline2"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_null_character():
    """Test compute_hash with null character."""
    content = "before\x00after"
    result = compute_hash(content)

    # Should handle null character without issues
    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


# --- Special content tests ---


def test_compute_hash_multiline_string():
    """Test compute_hash with multiline string."""
    content = """Line 1
Line 2
Line 3
Line 4"""
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_json_like_content():
    """Test compute_hash with JSON-like content."""
    content = '{"key": "value", "number": 42}'
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_code_snippet():
    """Test compute_hash with code snippet."""
    content = """def hello():
    print("world")
    return 42"""
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_xml_content():
    """Test compute_hash with XML content."""
    content = "<root><item>value</item></root>"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


# --- Hash format tests ---


def test_compute_hash_returns_lowercase_hex():
    """Test compute_hash returns lowercase hex digits."""
    content = "TEST"
    result = compute_hash(content)

    # All characters should be lowercase hex
    assert result == result.lower()
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_hash_always_64_chars():
    """Test compute_hash always returns 64 character string (SHA256)."""
    test_cases = [
        "",
        "a",
        "short",
        "This is a longer string with multiple words",
        "x" * 1000,
    ]

    for content in test_cases:
        result = compute_hash(content)
        assert len(result) == 64, f"Hash length mismatch for content: {content!r}"


# --- Encoding tests ---


def test_compute_hash_string_encoding():
    """Test compute_hash uses UTF-8 encoding."""
    content = "café"
    result = compute_hash(content)

    # Verify it matches UTF-8 encoding
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result == expected


def test_compute_hash_ascii_subset():
    """Test compute_hash with ASCII subset."""
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


def test_compute_hash_special_ascii():
    """Test compute_hash with special ASCII characters."""
    content = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    result = compute_hash(content)

    expected = hashlib.sha256(content.encode()).hexdigest()
    assert result == expected


# --- Real-world scenarios ---


def test_compute_hash_file_content_like():
    """Test compute_hash with content resembling file content."""
    content = """#!/usr/bin/env python3
# This is a comment
def main():
    print("Hello")

if __name__ == "__main__":
    main()
"""
    result = compute_hash(content)

    # Should return valid hash
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_hash_large_content():
    """Test compute_hash with large content."""
    # 1MB of content
    content = "x" * (1024 * 1024)
    result = compute_hash(content)

    # Should still return 64 character hash
    assert len(result) == 64


def test_compute_hash_binary_like_string():
    """Test compute_hash with content containing binary-like bytes."""
    content = "\x00\x01\x02\x03\x04\x05" * 100
    result = compute_hash(content)

    # Should handle without issues
    assert len(result) == 64


# --- Mock and patch tests ---


def test_compute_hash_uses_sha256():
    """Test compute_hash uses sha256 algorithm."""
    content = "test"

    with patch("indexter.utils.hashlib.sha256") as mock_sha256:
        mock_sha256.return_value.hexdigest.return_value = "mocked_hash"

        result = compute_hash(content)

        # Verify sha256 was called
        mock_sha256.assert_called_once()
        # Verify it was called with encoded bytes
        args = mock_sha256.call_args[0]
        assert args[0] == content.encode()

        # Verify hexdigest was called
        mock_sha256.return_value.hexdigest.assert_called_once()

        # Verify result is the hexdigest return value
        assert result == "mocked_hash"


def test_compute_hash_encodes_string_to_bytes():
    """Test compute_hash properly encodes string to bytes."""
    content = "test string"

    with patch("indexter.utils.hashlib.sha256") as mock_sha256:
        mock_sha256.return_value.hexdigest.return_value = "hash"

        compute_hash(content)

        # Verify encode was used (check the bytes passed to sha256)
        called_with = mock_sha256.call_args[0][0]
        assert called_with == b"test string"
        assert isinstance(called_with, bytes)


def test_compute_hash_returns_hexdigest():
    """Test compute_hash returns the hexdigest output."""
    content = "test"
    expected_hash = "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"

    with patch("indexter.utils.hashlib.sha256") as mock_sha256:
        mock_instance = mock_sha256.return_value
        mock_instance.hexdigest.return_value = expected_hash

        result = compute_hash(content)

        assert result == expected_hash
        mock_instance.hexdigest.assert_called_once()


# --- Error handling and edge cases ---


def test_compute_hash_does_not_raise_on_various_inputs():
    """Test compute_hash doesn't raise exceptions on various inputs."""
    test_cases = [
        "",
        "a",
        "normal text",
        "text with\nnewlines",
        "unicode: 你好世界",
        "emoji: 🎉🚀💡",
        "special: !@#$%^&*()",
        'symbols: <>?:"{}|',
        "tabs\tand\tspaces  ",
    ]

    for content in test_cases:
        # Should not raise
        result = compute_hash(content)
        assert result is not None


# --- Type tests ---


def test_compute_hash_returns_string():
    """Test compute_hash always returns a string."""
    result = compute_hash("test")
    assert isinstance(result, str)


def test_compute_hash_returns_string_various_inputs():
    """Test compute_hash returns string for various inputs."""
    test_cases = ["", "a", "test", "x" * 1000, "unicode: 中文"]

    for content in test_cases:
        result = compute_hash(content)
        assert isinstance(result, str), f"Expected str, got {type(result)} for {content!r}"
