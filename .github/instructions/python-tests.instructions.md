---
applyTo: "**/*.py"
---

When writing Python tests:

## Test Structure Essentials
- Use pytest as the primary testing framework
- Follow AAA pattern: Arrange, Act, Assert
- Write descriptive test names that explain the behavior being tested
- Keep tests focused on one specific behavior
- All imports should be at the top of the file unless necesssary

## Key Testing Practices
- Use pytest fixtures for setup and teardown. Module-specific fixtures should be placed in the test module itself, while shared fixtures should go in a `conftest.py` file at the appropriate directory level
- Make judicious use of unittest.mock. Mock external dependencies (databases, APIs, file operations, etc.)
- Use parameterized tests for testing multiple similar scenarios
- Test edge cases and error conditions, not just happy paths
- Ensure an appropriate number of integration tests to cover interactions between components

## Before writing tests:
To ensure a clean state before writing tests:
- Run linting against the source module (the file under test) with `uv run --group dev ruff check --fix <path>`
- Run type checking against the source module (the file under test) with `uv run --group dev ty check <path>`

## After writing tests:
To ensure a clean state and adequate coverage after writing tests:
- Run linting against the source module (the file under test) with `uv run --group dev ruff check --fix <path>`
- Run type checking against the source module (the file under test) with `uv run --group dev ty check <path>`
- Run linting against the test module with `uv run --group dev ruff check --fix <path>`
- Run a coverage report against the test module with `uv run --group test pytest <path> --cov=<module> --cov-fail-under=95 --cov-report=term-missing`

## Example Test Pattern
```python
import pytest
from unittest.mock import Mock, patch


class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()

    @pytest.mark.parametrize("invalid_email", ["", "invalid", "@test.com"])
    def test_should_reject_invalid_emails(self, user_service, invalid_email):
        with pytest.raises(ValueError, match="Invalid email"):
            user_service.create_user({"email": invalid_email})

    @patch('src.user_service.email_validator')
    def test_should_handle_validation_failure(self, mock_validator, user_service):
        mock_validator.validate.side_effect = ConnectionError()

        with pytest.raises(ConnectionError):
            user_service.create_user({"email": "test@example.com"})
```
