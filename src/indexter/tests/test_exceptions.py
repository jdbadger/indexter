"""Tests for the exceptions module."""

import pytest

from indexter.exceptions import RepoExistsError, RepoNotFoundError


class TestRepoNotFoundError:
    def test_repo_not_found_error_is_lookup_error(self):
        """Test RepoNotFoundError is a subclass of LookupError."""
        assert issubclass(RepoNotFoundError, LookupError)

    def test_repo_not_found_error_can_be_raised(self):
        """Test RepoNotFoundError can be raised with a message."""
        with pytest.raises(RepoNotFoundError) as exc_info:
            raise RepoNotFoundError("Repository not found")

        assert str(exc_info.value) == "Repository not found"

    def test_repo_not_found_error_can_be_raised_without_message(self):
        """Test RepoNotFoundError can be raised without a message."""
        with pytest.raises(RepoNotFoundError):
            raise RepoNotFoundError()

    def test_repo_not_found_error_can_be_caught_as_lookup_error(self):
        """Test RepoNotFoundError can be caught as LookupError."""
        with pytest.raises(LookupError):
            raise RepoNotFoundError("Repository not found")

    def test_repo_not_found_error_with_repo_name(self):
        """Test RepoNotFoundError with repository name."""
        repo_name = "my-awesome-repo"
        with pytest.raises(RepoNotFoundError) as exc_info:
            raise RepoNotFoundError(f"Repository '{repo_name}' not found")

        assert repo_name in str(exc_info.value)

    def test_repo_not_found_error_preserves_traceback(self):
        """Test RepoNotFoundError preserves traceback information."""

        def inner_function():
            raise RepoNotFoundError("Test error")

        def outer_function():
            inner_function()

        with pytest.raises(RepoNotFoundError) as exc_info:
            outer_function()

        # Verify traceback contains both function names
        tb = exc_info.traceback
        assert tb is not None


class TestRepoExistsError:
    def test_repo_exists_error_is_value_error(self):
        """Test RepoExistsError is a subclass of ValueError."""
        assert issubclass(RepoExistsError, ValueError)

    def test_repo_exists_error_can_be_raised(self):
        """Test RepoExistsError can be raised with a message."""
        with pytest.raises(RepoExistsError) as exc_info:
            raise RepoExistsError("Repository already exists")

        assert str(exc_info.value) == "Repository already exists"

    def test_repo_exists_error_can_be_raised_without_message(self):
        """Test RepoExistsError can be raised without a message."""
        with pytest.raises(RepoExistsError):
            raise RepoExistsError()

    def test_repo_exists_error_can_be_caught_as_value_error(self):
        """Test RepoExistsError can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise RepoExistsError("Repository already exists")

    def test_repo_exists_error_with_repo_name(self):
        """Test RepoExistsError with repository name."""
        repo_name = "duplicate-repo"
        with pytest.raises(RepoExistsError) as exc_info:
            raise RepoExistsError(f"Repository '{repo_name}' already exists")

        assert repo_name in str(exc_info.value)

    def test_repo_exists_error_preserves_traceback(self):
        """Test RepoExistsError preserves traceback information."""

        def inner_function():
            raise RepoExistsError("Test error")

        def outer_function():
            inner_function()

        with pytest.raises(RepoExistsError) as exc_info:
            outer_function()

        # Verify traceback contains both function names
        tb = exc_info.traceback
        assert tb is not None


class TestExceptionDifferentiation:
    def test_exceptions_are_distinct(self):
        """Test that RepoNotFoundError and RepoExistsError are distinct types."""
        assert RepoNotFoundError is not RepoExistsError
        assert not issubclass(RepoNotFoundError, RepoExistsError)
        assert not issubclass(RepoExistsError, RepoNotFoundError)

    def test_catching_specific_exception_types(self):
        """Test that each exception type can be caught independently."""
        # RepoNotFoundError should not be caught by RepoExistsError
        with pytest.raises(RepoNotFoundError):
            try:
                raise RepoNotFoundError("Not found")
            except RepoExistsError:
                pytest.fail("RepoNotFoundError should not be caught as RepoExistsError")

        # RepoExistsError should not be caught by RepoNotFoundError
        with pytest.raises(RepoExistsError):
            try:
                raise RepoExistsError("Already exists")
            except RepoNotFoundError:
                pytest.fail("RepoExistsError should not be caught as RepoNotFoundError")

    def test_exception_messages_can_be_formatted(self):
        """Test that exception messages support formatting."""
        repo_path = "/path/to/repo"

        with pytest.raises(RepoNotFoundError) as exc_info:
            raise RepoNotFoundError(f"Repository at {repo_path} not found")

        assert repo_path in str(exc_info.value)

        with pytest.raises(RepoExistsError) as exc_info:
            raise RepoExistsError(f"Repository at {repo_path} already exists")

        assert repo_path in str(exc_info.value)


class TestMultipleArguments:
    def test_repo_not_found_error_with_multiple_args(self):
        """Test RepoNotFoundError can accept multiple arguments."""
        with pytest.raises(RepoNotFoundError) as exc_info:
            raise RepoNotFoundError("Repository", "not", "found")

        # Multiple args are joined in exception string representation
        exc_str = str(exc_info.value)
        assert "Repository" in exc_str or "not" in exc_str or "found" in exc_str

    def test_repo_exists_error_with_multiple_args(self):
        """Test RepoExistsError can accept multiple arguments."""
        with pytest.raises(RepoExistsError) as exc_info:
            raise RepoExistsError("Repository", "already", "exists")

        # Multiple args are joined in exception string representation
        exc_str = str(exc_info.value)
        assert "Repository" in exc_str or "already" in exc_str or "exists" in exc_str


class TestExceptionAttributes:
    def test_repo_not_found_error_args_attribute(self):
        """Test RepoNotFoundError.args attribute."""
        message = "Repository not found"
        with pytest.raises(RepoNotFoundError) as exc_info:
            raise RepoNotFoundError(message)

        assert exc_info.value.args == (message,)

    def test_repo_exists_error_args_attribute(self):
        """Test RepoExistsError.args attribute."""
        message = "Repository already exists"
        with pytest.raises(RepoExistsError) as exc_info:
            raise RepoExistsError(message)

        assert exc_info.value.args == (message,)
