"""Docker container lifecycle management for Qdrant vector store."""

from __future__ import annotations

import logging
import time

import docker
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

from .config import Settings

logger = logging.getLogger(__name__)

CONTAINER_NAME = "indexter-qdrant"
HEALTH_POLL_INTERVAL = 1.0
HEALTH_TIMEOUT = 60.0


def start_qdrant_container(settings: Settings) -> Container:
    """Start the Qdrant Docker container.

    Pulls the image if not present, then creates and starts the container
    with port bindings and a volume mount for persistent storage.

    If a container with the expected name already exists and is running,
    it is returned directly. If it exists but is stopped, it is started.

    Args:
        settings: Global application settings with store configuration.

    Returns:
        The running Docker Container object.

    Raises:
        docker.errors.DockerException: If Docker is not available or the
            container cannot be started.
    """
    client = docker.from_env()
    store = settings.store

    # Check for existing container
    try:
        container = client.containers.get(CONTAINER_NAME)
        if container.status == "running":
            logger.info(f"Qdrant container '{CONTAINER_NAME}' is already running")
            return container
        logger.info(f"Starting existing Qdrant container '{CONTAINER_NAME}'")
        container.start()
        return container
    except NotFound:
        pass

    # Pull image if needed
    image = store.image
    try:
        client.images.get(image)
        logger.debug(f"Image '{image}' already available")
    except ImageNotFound:
        logger.info(f"Pulling image '{image}'...")
        client.images.pull(image)

    # Create and start container
    qdrant_storage = settings.data_dir / "qdrant"
    qdrant_storage.mkdir(parents=True, exist_ok=True)

    container = client.containers.run(
        image,
        name=CONTAINER_NAME,
        ports={
            "6333/tcp": store.port,
            "6334/tcp": store.grpc_port,
        },
        volumes={
            str(qdrant_storage): {"bind": "/qdrant/storage", "mode": "rw"},
        },
        detach=True,
    )
    logger.info(f"Started Qdrant container '{CONTAINER_NAME}' (image={image})")
    return container


def stop_qdrant_container(container: Container) -> None:
    """Stop and remove the Qdrant Docker container.

    Args:
        container: The Docker Container object to stop.
    """
    try:
        container.stop(timeout=10)
        container.remove()
        logger.info(f"Stopped and removed Qdrant container '{container.name}'")
    except NotFound:
        logger.debug("Container already removed")
    except Exception as e:
        logger.warning(f"Error stopping container: {e}")


def check_container_health(settings: Settings) -> bool:
    """Wait for the Qdrant container to be healthy by polling its HTTP API.

    Polls the Qdrant health endpoint at the configured host and port
    until it responds or the timeout is reached.

    Args:
        settings: Global settings with store host/port configuration.

    Returns:
        True if Qdrant is healthy.

    Raises:
        TimeoutError: If the container does not become healthy within the timeout.
    """
    import urllib.error
    import urllib.request

    url = f"http://{settings.store.host}:{settings.store.port}/healthz"
    deadline = time.monotonic() + HEALTH_TIMEOUT

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    logger.info("Qdrant container is healthy")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    raise TimeoutError(f"Qdrant container did not become healthy within {HEALTH_TIMEOUT}s (url={url})")
