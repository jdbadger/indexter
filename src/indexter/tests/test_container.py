"""Tests for indexter.container – Docker lifecycle management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from docker.errors import ImageNotFound, NotFound

from indexter.container import (
    CONTAINER_NAME,
    HEALTH_POLL_INTERVAL,
    HEALTH_TIMEOUT,
    check_container_health,
    start_qdrant_container,
    stop_qdrant_container,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings(tmp_path):
    """Return a Settings-like object with sensible defaults."""
    store = MagicMock()
    store.image = "qdrant/qdrant:latest"
    store.port = 6333
    store.grpc_port = 6334
    store.host = "localhost"

    settings = MagicMock()
    settings.store = store
    settings.data_dir = tmp_path / "data"
    return settings


@pytest.fixture
def mock_docker_client():
    """Patch docker.from_env and return the mock client."""
    with patch("indexter.container.docker") as docker_mod:
        client = MagicMock()
        docker_mod.from_env.return_value = client
        yield client


# ---------------------------------------------------------------------------
# start_qdrant_container
# ---------------------------------------------------------------------------


class TestStartQdrantContainerExisting:
    """Scenarios where the container already exists."""

    def test_already_running(self, mock_docker_client, mock_settings):
        # Arrange
        container = MagicMock(status="running")
        mock_docker_client.containers.get.return_value = container

        # Act
        result = start_qdrant_container(mock_settings)

        # Assert
        assert result is container
        container.start.assert_not_called()

    def test_stopped_container_is_started(self, mock_docker_client, mock_settings):
        # Arrange
        container = MagicMock(status="exited")
        mock_docker_client.containers.get.return_value = container

        # Act
        result = start_qdrant_container(mock_settings)

        # Assert
        assert result is container
        container.start.assert_called_once()


class TestStartQdrantContainerNew:
    """Scenarios where no existing container is found."""

    def test_image_exists_locally(self, mock_docker_client, mock_settings):
        # Arrange
        mock_docker_client.containers.get.side_effect = NotFound("nope")
        new_container = MagicMock()
        mock_docker_client.containers.run.return_value = new_container

        # Act
        result = start_qdrant_container(mock_settings)

        # Assert
        assert result is new_container
        mock_docker_client.images.get.assert_called_once_with("qdrant/qdrant:latest")
        mock_docker_client.images.pull.assert_not_called()
        mock_docker_client.containers.run.assert_called_once()

    def test_image_pulled_when_missing(self, mock_docker_client, mock_settings):
        # Arrange
        mock_docker_client.containers.get.side_effect = NotFound("nope")
        mock_docker_client.images.get.side_effect = ImageNotFound("nope")
        new_container = MagicMock()
        mock_docker_client.containers.run.return_value = new_container

        # Act
        result = start_qdrant_container(mock_settings)

        # Assert
        assert result is new_container
        mock_docker_client.images.pull.assert_called_once_with("qdrant/qdrant:latest")

    def test_container_run_args(self, mock_docker_client, mock_settings):
        # Arrange
        mock_docker_client.containers.get.side_effect = NotFound("nope")
        new_container = MagicMock()
        mock_docker_client.containers.run.return_value = new_container

        # Act
        start_qdrant_container(mock_settings)

        # Assert
        call_kwargs = mock_docker_client.containers.run.call_args
        assert call_kwargs[0][0] == "qdrant/qdrant:latest"
        assert call_kwargs[1]["name"] == CONTAINER_NAME
        assert call_kwargs[1]["ports"] == {
            "6333/tcp": 6333,
            "6334/tcp": 6334,
        }
        assert call_kwargs[1]["detach"] is True

    def test_storage_dir_created(self, mock_docker_client, mock_settings):
        # Arrange
        mock_docker_client.containers.get.side_effect = NotFound("nope")
        mock_docker_client.containers.run.return_value = MagicMock()

        # Act
        start_qdrant_container(mock_settings)

        # Assert
        qdrant_storage = mock_settings.data_dir / "qdrant"
        assert qdrant_storage.exists()

    def test_volume_mount(self, mock_docker_client, mock_settings):
        # Arrange
        mock_docker_client.containers.get.side_effect = NotFound("nope")
        mock_docker_client.containers.run.return_value = MagicMock()

        # Act
        start_qdrant_container(mock_settings)

        # Assert
        call_kwargs = mock_docker_client.containers.run.call_args[1]
        qdrant_storage = mock_settings.data_dir / "qdrant"
        assert str(qdrant_storage) in call_kwargs["volumes"]
        mount = call_kwargs["volumes"][str(qdrant_storage)]
        assert mount["bind"] == "/qdrant/storage"
        assert mount["mode"] == "rw"


# ---------------------------------------------------------------------------
# stop_qdrant_container
# ---------------------------------------------------------------------------


class TestStopQdrantContainer:
    """Tests for stop_qdrant_container."""

    def test_stop_and_remove(self):
        # Arrange
        container = MagicMock()

        # Act
        stop_qdrant_container(container)

        # Assert
        container.stop.assert_called_once_with(timeout=10)
        container.remove.assert_called_once()

    def test_already_removed(self):
        # Arrange
        container = MagicMock()
        container.stop.side_effect = NotFound("gone")

        # Act – no exception raised
        stop_qdrant_container(container)

        # Assert
        container.remove.assert_not_called()

    def test_generic_error_logged(self):
        # Arrange
        container = MagicMock()
        container.stop.side_effect = RuntimeError("boom")

        # Act – no exception raised
        stop_qdrant_container(container)

        # Assert – function does not re-raise
        container.remove.assert_not_called()


# ---------------------------------------------------------------------------
# check_container_health
# ---------------------------------------------------------------------------


class TestCheckContainerHealth:
    """Tests for check_container_health."""

    @patch("indexter.container.time")
    @patch("urllib.request.urlopen")
    def test_healthy_on_first_poll(self, mock_urlopen, mock_time, mock_settings):
        # Arrange
        mock_time.monotonic.side_effect = [0.0, 0.5]  # start, first check
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        # Act
        result = check_container_health(mock_settings)

        # Assert
        assert result is True
        mock_urlopen.assert_called_once()

    @patch("indexter.container.time")
    @patch("urllib.request.urlopen")
    def test_healthy_after_retries(self, mock_urlopen, mock_time, mock_settings):
        import urllib.error

        # Arrange – fail twice, then succeed
        mock_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0]
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.side_effect = [
            urllib.error.URLError("fail"),
            OSError("fail"),
            resp,
        ]

        # Act
        result = check_container_health(mock_settings)

        # Assert
        assert result is True
        assert mock_urlopen.call_count == 3
        assert mock_time.sleep.call_count == 2

    @patch("indexter.container.time")
    @patch("urllib.request.urlopen")
    def test_timeout_raises(self, mock_urlopen, mock_time, mock_settings):
        import urllib.error

        # Arrange – always fail, deadline exceeded after first check
        mock_time.monotonic.side_effect = [0.0, HEALTH_TIMEOUT + 1]
        mock_urlopen.side_effect = urllib.error.URLError("fail")

        # Act / Assert
        with pytest.raises(TimeoutError, match="did not become healthy"):
            check_container_health(mock_settings)

    @patch("indexter.container.time")
    @patch("urllib.request.urlopen")
    def test_url_uses_settings(self, mock_urlopen, mock_time, mock_settings):
        # Arrange
        mock_settings.store.host = "192.168.1.100"
        mock_settings.store.port = 9999
        mock_time.monotonic.side_effect = [0.0, 0.5]
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        # Act
        check_container_health(mock_settings)

        # Assert
        call_url = mock_urlopen.call_args[0][0]
        assert call_url == "http://192.168.1.100:9999/healthz"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify exposed constants."""

    def test_container_name(self):
        assert CONTAINER_NAME == "indexter-qdrant"

    def test_health_poll_interval(self):
        assert HEALTH_POLL_INTERVAL == 1.0

    def test_health_timeout(self):
        assert HEALTH_TIMEOUT == 60.0
