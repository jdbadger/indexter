import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from indexter.store.models import IndexResult, SearchResult, SearchResults


class TestIndexResult:
    """Tests for IndexResult model."""

    @pytest.fixture
    def valid_index_result_data(self):
        """Fixture providing valid IndexResult data."""
        return {
            "repo": "test-repo",
            "repo_path": "/home/user/repos/test-repo",
            "documents_indexed": ["src/main.py", "src/utils.py"],
            "documents_deleted": ["src/old.py"],
            "nodes_added": 50,
            "nodes_deleted": 10,
            "duration": 1.5,
            "errors": ["Error parsing file.py"],
        }

    def test_should_create_index_result_with_valid_data(self, valid_index_result_data):
        """Test IndexResult creation with all valid fields."""
        result = IndexResult(**valid_index_result_data)

        assert result.repo == "test-repo"
        assert result.repo_path == "/home/user/repos/test-repo"
        assert result.documents_indexed == ["src/main.py", "src/utils.py"]
        assert result.documents_deleted == ["src/old.py"]
        assert result.nodes_added == 50
        assert result.nodes_deleted == 10
        assert result.duration == 1.5
        assert result.errors == ["Error parsing file.py"]

    def test_should_create_index_result_with_minimal_data(self):
        """Test IndexResult creation with only required fields."""
        result = IndexResult(repo="minimal-repo", repo_path="/path/to/repo")

        assert result.repo == "minimal-repo"
        assert result.repo_path == "/path/to/repo"
        assert result.documents_indexed == []
        assert result.documents_deleted == []
        assert result.nodes_added == 0
        assert result.nodes_deleted == 0
        assert result.duration == 0.0
        assert result.errors == []

    @pytest.mark.parametrize(
        "field_name",
        ["repo", "repo_path"],
    )
    def test_should_reject_missing_required_field(self, valid_index_result_data, field_name):
        """Test IndexResult rejects missing required fields."""
        data = valid_index_result_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            IndexResult(**data)

        assert field_name in str(exc_info.value)

    def test_should_have_indexed_at_timestamp(self):
        """Test IndexResult has indexed_at timestamp set automatically."""
        before = datetime.now(UTC)
        result = IndexResult(repo="test", repo_path="/path")
        after = datetime.now(UTC)

        assert before <= result.indexed_at <= after
        assert result.indexed_at.tzinfo == UTC

    def test_should_accept_custom_indexed_at(self):
        """Test IndexResult accepts custom indexed_at timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = IndexResult(repo="test", repo_path="/path", indexed_at=custom_time)

        assert result.indexed_at == custom_time

    @pytest.mark.parametrize(
        "documents",
        [[], ["file1.py"], ["file1.py", "file2.js", "file3.md"]],
    )
    def test_should_accept_various_document_lists(self, documents):
        """Test IndexResult accepts various document lists."""
        result = IndexResult(repo="test", repo_path="/path", documents_indexed=documents)

        assert result.documents_indexed == documents

    @pytest.mark.parametrize(
        "errors",
        [[], ["Error 1"], ["Error 1", "Error 2", "Error 3"]],
    )
    def test_should_accept_various_error_lists(self, errors):
        """Test IndexResult accepts various error lists."""
        result = IndexResult(repo="test", repo_path="/path", errors=errors)

        assert result.errors == errors

    @pytest.mark.parametrize(
        "duration",
        [0.0, 0.5, 1.0, 10.5, 100.0, 1234.56],
    )
    def test_should_accept_various_durations(self, duration):
        """Test IndexResult accepts various duration values."""
        result = IndexResult(repo="test", repo_path="/path", duration=duration)

        assert result.duration == duration

    @pytest.mark.parametrize(
        "count",
        [0, 1, 10, 100, 1000],
    )
    def test_should_accept_various_node_counts(self, count):
        """Test IndexResult accepts various node count values."""
        result = IndexResult(
            repo="test",
            repo_path="/path",
            nodes_added=count,
            nodes_deleted=count,
        )

        assert result.nodes_added == count
        assert result.nodes_deleted == count

    def test_should_reject_invalid_duration_type(self):
        """Test IndexResult rejects invalid duration type."""
        with pytest.raises(ValidationError) as exc_info:
            IndexResult(repo="test", repo_path="/path", duration="not_a_number")  # type: ignore[arg-type]

        assert "duration" in str(exc_info.value).lower()

    def test_should_reject_invalid_count_type(self):
        """Test IndexResult rejects invalid count type."""
        with pytest.raises(ValidationError) as exc_info:
            IndexResult(repo="test", repo_path="/path", nodes_added="not_a_number")  # type: ignore[arg-type]

        assert "nodes_added" in str(exc_info.value).lower()

    def test_should_generate_summary_with_data(self):
        """Test summary property generates correct summary."""
        result = IndexResult(
            repo="test",
            repo_path="/path",
            documents_indexed=["file1.py", "file2.py"],
            nodes_added=10,
            nodes_deleted=5,
            duration=2.5,
        )

        summary = result.summary

        assert "2 documents" in summary
        assert "+10 nodes added" in summary
        assert "-5 nodes deleted" in summary
        assert "2.50s" in summary

    def test_should_generate_summary_with_empty_data(self):
        """Test summary property with no indexed files."""
        result = IndexResult(repo="test", repo_path="/path", duration=1.0)

        summary = result.summary

        assert "0 documents" in summary
        assert "+0 nodes added" in summary
        assert "1.00s" in summary

    def test_should_serialize_to_dict(self, valid_index_result_data):
        """Test IndexResult can be serialized to dict."""
        result = IndexResult(**valid_index_result_data)

        data = result.model_dump()

        assert isinstance(data, dict)
        assert data["repo"] == "test-repo"
        assert data["documents_indexed"] == ["src/main.py", "src/utils.py"]
        assert "indexed_at" in data
        assert "summary" in data

    def test_should_serialize_to_json(self, valid_index_result_data):
        """Test IndexResult can be serialized to JSON."""
        result = IndexResult(**valid_index_result_data)

        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-repo" in json_str
        assert "src/main.py" in json_str

    def test_should_deserialize_from_dict(self, valid_index_result_data):
        """Test IndexResult can be deserialized from dict."""
        result = IndexResult.model_validate(valid_index_result_data)

        assert result.repo == valid_index_result_data["repo"]
        assert result.documents_indexed == valid_index_result_data["documents_indexed"]

    def test_should_support_field_descriptions(self):
        """Test IndexResult fields have descriptions."""
        schema = IndexResult.model_json_schema()

        assert "properties" in schema
        assert "repo" in schema["properties"]
        assert "description" in schema["properties"]["repo"]


class TestSearchResult:
    """Tests for SearchResult model."""

    @pytest.fixture
    def valid_search_result_data(self):
        """Fixture providing valid SearchResult data."""
        return {
            "content": "def hello():\n    print('Hello, World!')",
            "score": 0.95,
            "metadata": {
                "path": "src/main.py",
                "language": "python",
                "node_type": "function_definition",
            },
        }

    def test_should_create_search_result_with_valid_data(self, valid_search_result_data):
        """Test SearchResult creation with all valid fields."""
        result = SearchResult(**valid_search_result_data)

        assert result.content == "def hello():\n    print('Hello, World!')"
        assert result.score == 0.95
        assert result.metadata["path"] == "src/main.py"
        assert result.metadata["language"] == "python"

    @pytest.mark.parametrize(
        "field_name",
        ["content", "score", "metadata"],
    )
    def test_should_reject_missing_required_field(self, valid_search_result_data, field_name):
        """Test SearchResult rejects missing required fields."""
        data = valid_search_result_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            SearchResult(**data)

        assert field_name in str(exc_info.value)

    @pytest.mark.parametrize(
        "score",
        [0.0, 0.1, 0.5, 0.75, 0.99, 1.0],
    )
    def test_should_accept_various_scores(self, valid_search_result_data, score):
        """Test SearchResult accepts various score values."""
        data = valid_search_result_data.copy()
        data["score"] = score

        result = SearchResult(**data)

        assert result.score == score

    def test_should_reject_invalid_score_type(self, valid_search_result_data):
        """Test SearchResult rejects invalid score type."""
        data = valid_search_result_data.copy()
        data["score"] = "not_a_number"

        with pytest.raises(ValidationError) as exc_info:
            SearchResult(**data)

        assert "score" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "content",
        ["", "single line", "multi\nline\ncontent", "unicode: café 日本語 🚀"],
    )
    def test_should_accept_various_content_types(self, valid_search_result_data, content):
        """Test SearchResult accepts various content types."""
        data = valid_search_result_data.copy()
        data["content"] = content

        result = SearchResult(**data)

        assert result.content == content

    def test_should_accept_empty_metadata(self, valid_search_result_data):
        """Test SearchResult accepts empty metadata dict."""
        data = valid_search_result_data.copy()
        data["metadata"] = {}

        result = SearchResult(**data)

        assert result.metadata == {}

    def test_should_accept_complex_metadata(self, valid_search_result_data):
        """Test SearchResult accepts complex metadata structures."""
        data = valid_search_result_data.copy()
        data["metadata"] = {
            "path": "src/utils/helper.py",
            "language": "python",
            "node_type": "class_definition",
            "line_start": 10,
            "line_end": 25,
            "tags": ["utility", "helper"],
            "nested": {"key": "value"},
        }

        result = SearchResult(**data)

        assert result.metadata["path"] == "src/utils/helper.py"
        assert result.metadata["tags"] == ["utility", "helper"]
        assert result.metadata["nested"]["key"] == "value"

    def test_should_serialize_to_dict(self, valid_search_result_data):
        """Test SearchResult can be serialized to dict."""
        result = SearchResult(**valid_search_result_data)

        data = result.model_dump()

        assert isinstance(data, dict)
        assert data["content"] == valid_search_result_data["content"]
        assert data["score"] == valid_search_result_data["score"]
        assert data["metadata"] == valid_search_result_data["metadata"]

    def test_should_serialize_to_json(self, valid_search_result_data):
        """Test SearchResult can be serialized to JSON."""
        result = SearchResult(**valid_search_result_data)

        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
        assert "hello" in json_str
        assert "0.95" in json_str

    def test_should_deserialize_from_dict(self, valid_search_result_data):
        """Test SearchResult can be deserialized from dict."""
        result = SearchResult.model_validate(valid_search_result_data)

        assert result.content == valid_search_result_data["content"]
        assert result.score == valid_search_result_data["score"]

    def test_should_support_field_descriptions(self):
        """Test SearchResult fields have descriptions."""
        schema = SearchResult.model_json_schema()

        assert "properties" in schema
        assert "content" in schema["properties"]
        assert "description" in schema["properties"]["content"]


class TestSearchResults:
    """Tests for SearchResults model."""

    @pytest.fixture
    def valid_search_results_data(self):
        """Fixture providing valid SearchResults data."""
        return {
            "repo": "test-repo",
            "repo_path": "/home/user/repos/test-repo",
            "results": [
                {
                    "content": "def hello():",
                    "score": 0.95,
                    "metadata": {"path": "main.py"},
                },
                {
                    "content": "def world():",
                    "score": 0.85,
                    "metadata": {"path": "utils.py"},
                },
            ],
            "query": "hello world function",
            "filters": {"language": "python", "min_score": 0.8},
        }

    def test_should_create_search_results_with_valid_data(self, valid_search_results_data):
        """Test SearchResults creation with all valid fields."""
        results = SearchResults(**valid_search_results_data)

        assert results.repo == "test-repo"
        assert results.repo_path == "/home/user/repos/test-repo"
        assert len(results.results) == 2
        assert results.query == "hello world function"
        assert results.filters["language"] == "python"

    def test_should_create_search_results_with_minimal_data(self):
        """Test SearchResults creation with only required fields."""
        results = SearchResults(results=[], query="test", filters={})

        assert results.repo is None
        assert results.repo_path is None
        assert results.results == []
        assert results.query == "test"
        assert results.filters == {}

    @pytest.mark.parametrize(
        "field_name",
        ["results", "query", "filters"],
    )
    def test_should_reject_missing_required_field(self, valid_search_results_data, field_name):
        """Test SearchResults rejects missing required fields."""
        data = valid_search_results_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            SearchResults(**data)

        assert field_name in str(exc_info.value)

    def test_should_accept_empty_results_list(self):
        """Test SearchResults accepts empty results list."""
        results = SearchResults(results=[], query="no matches", filters={})

        assert results.results == []
        assert results.count == 0

    def test_should_accept_multiple_results(self, valid_search_results_data):
        """Test SearchResults with multiple results."""
        results = SearchResults(**valid_search_results_data)

        assert len(results.results) == 2
        assert all(isinstance(r, SearchResult) for r in results.results)

    def test_should_compute_count_property(self, valid_search_results_data):
        """Test count property returns correct number of results."""
        results = SearchResults(**valid_search_results_data)

        assert results.count == 2

    def test_should_compute_count_for_empty_results(self):
        """Test count property returns 0 for empty results."""
        results = SearchResults(results=[], query="test", filters={})

        assert results.count == 0

    def test_should_accept_none_for_optional_fields(self):
        """Test SearchResults accepts None for optional repo fields."""
        results = SearchResults(repo=None, repo_path=None, results=[], query="test", filters={})

        assert results.repo is None
        assert results.repo_path is None

    def test_should_accept_empty_filters(self):
        """Test SearchResults accepts empty filters dict."""
        results = SearchResults(results=[], query="test", filters={})

        assert results.filters == {}

    def test_should_accept_complex_filters(self):
        """Test SearchResults accepts complex filter structures."""
        results = SearchResults(
            results=[],
            query="test",
            filters={
                "language": "python",
                "min_score": 0.8,
                "max_results": 10,
                "node_types": ["function", "class"],
                "tags": {"type": "utility"},
            },
        )

        assert results.filters["language"] == "python"
        assert results.filters["node_types"] == ["function", "class"]
        assert results.filters["tags"]["type"] == "utility"

    def test_should_convert_result_dicts_to_search_result_objects(self):
        """Test SearchResults converts result dicts to SearchResult objects."""
        data = {
            "results": [
                {"content": "code", "score": 0.9, "metadata": {"path": "file.py"}},
            ],
            "query": "test",
            "filters": {},
        }

        results = SearchResults(**data)  # type: ignore[arg-type]

        assert isinstance(results.results[0], SearchResult)
        assert results.results[0].content == "code"

    def test_should_serialize_to_dict(self, valid_search_results_data):
        """Test SearchResults can be serialized to dict."""
        results = SearchResults(**valid_search_results_data)

        data = results.model_dump()

        assert isinstance(data, dict)
        assert data["repo"] == "test-repo"
        assert len(data["results"]) == 2
        assert data["count"] == 2

    def test_should_serialize_to_json(self, valid_search_results_data):
        """Test SearchResults can be serialized to JSON."""
        results = SearchResults(**valid_search_results_data)

        json_str = results.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-repo" in json_str
        assert "hello world function" in json_str

    def test_should_deserialize_from_dict(self, valid_search_results_data):
        """Test SearchResults can be deserialized from dict."""
        results = SearchResults.model_validate(valid_search_results_data)

        assert results.repo == valid_search_results_data["repo"]
        assert len(results.results) == 2

    def test_should_support_field_descriptions(self):
        """Test SearchResults fields have descriptions."""
        schema = SearchResults.model_json_schema()

        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "description" in schema["properties"]["query"]


class TestIndexResultIntegration:
    """Integration tests for IndexResult."""

    def test_should_roundtrip_through_dict_serialization(self):
        """Test IndexResult can roundtrip through dict serialization."""
        original = IndexResult(
            repo="integration-repo",
            repo_path="/integration/path",
            documents_indexed=["file1.py", "file2.js"],
            documents_deleted=["old.py"],
            nodes_added=25,
            nodes_deleted=5,
            duration=3.14,
            errors=["Error 1"],
        )

        # Serialize and deserialize
        data = original.model_dump()
        # Remove computed field for validation
        data.pop("summary", None)
        restored = IndexResult.model_validate(data)

        assert restored.repo == original.repo
        assert restored.documents_indexed == original.documents_indexed
        assert restored.nodes_added == original.nodes_added

    def test_should_roundtrip_through_json_serialization(self):
        """Test IndexResult can roundtrip through JSON serialization."""
        original = IndexResult(
            repo="json-repo",
            repo_path="/json/path",
            documents_indexed=["test.py"],
            nodes_added=10,
        )

        # Serialize to JSON and deserialize
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        # Remove computed field
        data.pop("summary", None)
        restored = IndexResult.model_validate(data)

        assert restored.repo == original.repo
        assert restored.documents_indexed == original.documents_indexed

    def test_should_handle_datetime_serialization(self):
        """Test IndexResult handles datetime serialization correctly."""
        custom_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = IndexResult(repo="test", repo_path="/path", indexed_at=custom_time)

        # Serialize and deserialize
        data = result.model_dump()
        data.pop("summary", None)
        restored = IndexResult.model_validate(data)

        # Datetimes should be equal
        assert restored.indexed_at.year == custom_time.year
        assert restored.indexed_at.month == custom_time.month
        assert restored.indexed_at.day == custom_time.day


class TestSearchResultIntegration:
    """Integration tests for SearchResult."""

    def test_should_roundtrip_through_dict_serialization(self):
        """Test SearchResult can roundtrip through dict serialization."""
        original = SearchResult(
            content="def test(): pass",
            score=0.88,
            metadata={"path": "test.py", "language": "python"},
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = SearchResult.model_validate(data)

        assert restored.content == original.content
        assert restored.score == original.score
        assert restored.metadata == original.metadata

    def test_should_roundtrip_through_json_serialization(self):
        """Test SearchResult can roundtrip through JSON serialization."""
        original = SearchResult(
            content="class MyClass:\n    pass",
            score=0.92,
            metadata={"path": "class.py"},
        )

        # Serialize to JSON and deserialize
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = SearchResult.model_validate(data)

        assert restored.content == original.content
        assert restored.score == original.score


class TestSearchResultsIntegration:
    """Integration tests for SearchResults."""

    def test_should_roundtrip_through_dict_serialization(self):
        """Test SearchResults can roundtrip through dict serialization."""
        original = SearchResults(
            repo="int-repo",
            repo_path="/int/path",
            results=[
                SearchResult(content="code1", score=0.9, metadata={"p": "1"}),
                SearchResult(content="code2", score=0.8, metadata={"p": "2"}),
            ],
            query="integration test",
            filters={"lang": "py"},
        )

        # Serialize and deserialize
        data = original.model_dump()
        # Remove computed field
        data.pop("count", None)
        restored = SearchResults.model_validate(data)

        assert restored.repo == original.repo
        assert len(restored.results) == len(original.results)
        assert restored.query == original.query

    def test_should_roundtrip_through_json_serialization(self):
        """Test SearchResults can roundtrip through JSON serialization."""
        original = SearchResults(
            results=[
                SearchResult(content="test", score=0.95, metadata={}),
            ],
            query="json test",
            filters={},
        )

        # Serialize to JSON and deserialize
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        # Remove computed field
        data.pop("count", None)
        restored = SearchResults.model_validate(data)

        assert len(restored.results) == 1
        assert restored.query == original.query

    def test_should_handle_nested_search_results(self):
        """Test SearchResults properly handles nested SearchResult objects."""
        results = SearchResults(
            results=[
                SearchResult(
                    content=f"function_{i}()",
                    score=0.9 - i * 0.1,
                    metadata={"index": i},
                )
                for i in range(5)
            ],
            query="functions",
            filters={},
        )

        assert results.count == 5
        for i, result in enumerate(results.results):
            assert isinstance(result, SearchResult)
            assert f"function_{i}" in result.content
            assert result.metadata["index"] == i

    def test_should_validate_complete_search_workflow(self):
        """Test complete search results structure validation."""
        results = SearchResults(
            repo="complete-repo",
            repo_path="/complete/repo/path",
            results=[
                SearchResult(
                    content="def authenticate(user, password):\n    return True",
                    score=0.95,
                    metadata={
                        "path": "src/auth.py",
                        "language": "python",
                        "node_type": "function_definition",
                        "line_start": 10,
                        "line_end": 12,
                    },
                ),
                SearchResult(
                    content="class User:\n    def __init__(self):\n        pass",
                    score=0.87,
                    metadata={
                        "path": "src/models.py",
                        "language": "python",
                        "node_type": "class_definition",
                        "line_start": 5,
                        "line_end": 8,
                    },
                ),
            ],
            query="user authentication",
            filters={
                "language": "python",
                "min_score": 0.8,
                "max_results": 10,
                "node_types": ["function_definition", "class_definition"],
            },
        )

        # Verify all fields
        assert results.repo == "complete-repo"
        assert results.repo_path == "/complete/repo/path"
        assert results.count == 2
        assert results.query == "user authentication"
        assert results.filters["language"] == "python"
        assert results.results[0].score == 0.95
        assert results.results[1].metadata["node_type"] == "class_definition"
