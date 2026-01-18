"""
Store management CLI commands.

This module provides CLI commands for managing the Qdrant vector store
Docker container. It handles container lifecycle operations including
initialization, starting, stopping, and removal.

Commands:
    init: Initialize and optionally start the Qdrant container
    start: Start an existing container
    status: Show container status and connection info
    stop: Stop a running container
    remove: Remove the container and optionally its data

Example:
    $ indexter store init
    $ indexter store status
    $ indexter store stop
"""

import logging
import os
import shutil
from typing import Annotated

import docker
import typer
from docker.errors import DockerException
from rich.console import Console
from rich.table import Table

from indexter.config import settings

logger = logging.getLogger(__name__)

# Container configuration
CONTAINER_NAME = "indexter-qdrant"
CONTAINER_MOUNT_PATH = "/qdrant/storage"

store_app = typer.Typer(
    name="store",
    help="Manage the Qdrant vector store container.",
    no_args_is_help=True,
)

console = Console()


class DockerNotAvailableError(Exception):
    """Raised when Docker daemon is not available."""

    pass


class ContainerNotFoundError(Exception):
    """Raised when the container does not exist."""

    pass


class ContainerAlreadyExistsError(Exception):
    """Raised when trying to create a container that already exists."""

    pass


def get_docker_client():
    """Get a Docker client instance.

    Returns:
        docker.DockerClient: A Docker client connected to the daemon.

    Raises:
        DockerNotAvailableError: If Docker daemon is not running or accessible.
    """
    try:
        client = docker.from_env()
        client.ping()
        return client
    except ImportError:
        raise DockerNotAvailableError("Docker SDK not installed. Install with: uv add indexter[cli]") from None
    except DockerException as e:
        raise DockerNotAvailableError(f"Cannot connect to Docker daemon. Is Docker running?\n{e}") from None


def get_container(client):
    """Get the indexter-qdrant container if it exists.

    Args:
        client: Docker client instance.

    Returns:
        Container object or None if not found.
    """
    try:
        return client.containers.get(CONTAINER_NAME)
    except Exception:
        return None


def pull_image(client, image: str) -> None:
    """Pull a Docker image with progress display.

    Args:
        client: Docker client instance.
        image: Image name and tag to pull.
    """
    console.print(f"[blue]Pulling image {image}...[/blue]")
    try:
        client.images.pull(image)
        console.print(f"[green]✓[/green] Image {image} pulled successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to pull image: {e}")
        raise


def create_container(client, image: str, no_start: bool = False):
    """Create the Qdrant container with appropriate configuration.

    Args:
        client: Docker client instance.
        image: Docker image to use.
        no_start: If True, create but don't start the container.

    Returns:
        The created container object.

    Raises:
        ContainerAlreadyExistsError: If container already exists.
    """
    existing = get_container(client)
    if existing:
        raise ContainerAlreadyExistsError(
            f"Container '{CONTAINER_NAME}' already exists. "
            "Use 'indexter store start' to start it or 'indexter store remove' to remove it."
        )

    # Ensure data directories exist
    # Qdrant needs write access to storage and snapshots directories
    data_path = settings.data_dir / "qdrant"
    storage_path = data_path / "storage"
    snapshots_path = data_path / "snapshots"
    storage_path.mkdir(parents=True, exist_ok=True)
    snapshots_path.mkdir(parents=True, exist_ok=True)

    # Port configuration
    port_bindings = {
        "6333/tcp": settings.store.port,
        "6334/tcp": settings.store.grpc_port,
    }

    # Volume mounts (bind mount to XDG data dir)
    # Qdrant requires both storage and snapshots directories to be writable
    volumes = {
        str(storage_path): {"bind": "/qdrant/storage", "mode": "rw"},
        str(snapshots_path): {"bind": "/qdrant/snapshots", "mode": "rw"},
    }

    # Run as current user to ensure proper file ownership in bind mounts
    uid = os.getuid()
    gid = os.getgid()

    console.print(f"[blue]Creating container {CONTAINER_NAME}...[/blue]")
    console.print(f"  Data directory: {data_path}")
    console.print(f"  HTTP port: {settings.store.port}")
    console.print(f"  gRPC port: {settings.store.grpc_port}")

    container = client.containers.create(
        image=image,
        name=CONTAINER_NAME,
        ports=port_bindings,
        volumes=volumes,
        user=f"{uid}:{gid}",
        detach=True,
    )

    console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' created")

    if not no_start:
        container.start()
        console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' started")

    return container


def start_container(client) -> None:
    """Start the Qdrant container.

    Args:
        client: Docker client instance.

    Raises:
        ContainerNotFoundError: If container doesn't exist.
    """
    container = get_container(client)
    if not container:
        raise ContainerNotFoundError(f"Container '{CONTAINER_NAME}' not found. Run 'indexter store init' first.")

    if container.status == "running":
        console.print(f"[yellow]Container '{CONTAINER_NAME}' is already running[/yellow]")
        return

    container.start()
    console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' started")


def stop_container(client) -> None:
    """Stop the Qdrant container.

    Args:
        client: Docker client instance.

    Raises:
        ContainerNotFoundError: If container doesn't exist.
    """
    container = get_container(client)
    if not container:
        raise ContainerNotFoundError(f"Container '{CONTAINER_NAME}' not found.")

    if container.status != "running":
        console.print(f"[yellow]Container '{CONTAINER_NAME}' is not running[/yellow]")
        return

    container.stop()
    console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' stopped")


def remove_container(client, remove_volumes: bool = False) -> None:
    """Remove the Qdrant container and optionally its data.

    Args:
        client: Docker client instance.
        remove_volumes: If True, also delete the data directory.

    Raises:
        ContainerNotFoundError: If container doesn't exist.
    """
    container = get_container(client)
    if not container:
        raise ContainerNotFoundError(f"Container '{CONTAINER_NAME}' not found.")

    # Stop if running
    if container.status == "running":
        container.stop()
        console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' stopped")

    container.remove()
    console.print(f"[green]✓[/green] Container '{CONTAINER_NAME}' removed")

    if remove_volumes:
        data_path = settings.data_dir / "qdrant"
        if data_path.exists():
            shutil.rmtree(data_path)
            console.print(f"[green]✓[/green] Data directory removed: {data_path}")


@store_app.command(name="init")
def store_init(
    no_start: Annotated[
        bool,
        typer.Option("--no-start", "-n", help="Create container without starting it"),
    ] = False,
) -> None:
    """Initialize the Qdrant vector store container.

    Pulls the Qdrant Docker image and creates a container with the configured
    ports and data directory. By default, starts the container after creation.

    The container data is stored at ~/.local/share/indexter/qdrant (or
    $XDG_DATA_HOME/indexter/qdrant if set).
    """
    try:
        client = get_docker_client()
    except DockerNotAvailableError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None

    try:
        # Pull the image
        image = settings.store.image
        pull_image(client, image)

        # Create (and optionally start) the container
        create_container(client, image, no_start=no_start)

        if not no_start:
            console.print()
            console.print("[green]Store is ready![/green]")
            console.print(f"  HTTP endpoint: http://{settings.store.host}:{settings.store.port}")
            console.print(f"  gRPC endpoint: {settings.store.host}:{settings.store.grpc_port}")
    except ContainerAlreadyExistsError as e:
        console.print(f"[yellow]![/yellow] {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize store: {e}")
        raise typer.Exit(1) from None


@store_app.command(name="start")
def store_start() -> None:
    """Start the Qdrant vector store container.

    Starts an existing container. If the container doesn't exist,
    prompts to run 'indexter store init' first.
    """
    try:
        client = get_docker_client()
    except DockerNotAvailableError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None

    try:
        start_container(client)
        console.print()
        console.print(f"  HTTP endpoint: http://{settings.store.host}:{settings.store.port}")
        console.print(f"  gRPC endpoint: {settings.store.host}:{settings.store.grpc_port}")
    except ContainerNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start store: {e}")
        raise typer.Exit(1) from None


@store_app.command(name="status")
def store_status() -> None:
    """Show the status of the Qdrant vector store container.

    Displays container state, port mappings, and connection information.
    """
    try:
        client = get_docker_client()
    except DockerNotAvailableError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None

    container = get_container(client)
    if not container:
        console.print(f"[yellow]Container '{CONTAINER_NAME}' not found.[/yellow]")
        console.print("Run 'indexter store init' to create it.")
        raise typer.Exit(0)

    # Refresh container info
    container.reload()

    # Build status table
    table = Table(title="Qdrant Store Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    status_color = "green" if container.status == "running" else "yellow"
    table.add_row("Container", CONTAINER_NAME)
    table.add_row("Status", f"[{status_color}]{container.status}[/{status_color}]")
    table.add_row("Image", container.image.tags[0] if container.image.tags else "unknown")

    # Data directory
    data_path = settings.data_dir / "qdrant"
    table.add_row("Data Directory", str(data_path))

    # Ports
    table.add_row("HTTP Port", str(settings.store.port))
    table.add_row("gRPC Port", str(settings.store.grpc_port))
    table.add_row("Prefer gRPC", str(settings.store.prefer_grpc))

    if container.status == "running":
        table.add_row("HTTP Endpoint", f"http://{settings.store.host}:{settings.store.port}")
        table.add_row("gRPC Endpoint", f"{settings.store.host}:{settings.store.grpc_port}")

    console.print(table)


@store_app.command(name="stop")
def store_stop() -> None:
    """Stop the Qdrant vector store container.

    Stops the running container. The container and its data are preserved.
    """
    try:
        client = get_docker_client()
    except DockerNotAvailableError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None

    try:
        stop_container(client)
    except ContainerNotFoundError as e:
        console.print(f"[yellow]![/yellow] {e}")
        raise typer.Exit(0) from None
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to stop store: {e}")
        raise typer.Exit(1) from None


@store_app.command(name="remove")
def store_remove(
    volumes: Annotated[
        bool,
        typer.Option("--volumes", "-v", help="Also remove the data directory"),
    ] = False,
) -> None:
    """Remove the Qdrant vector store container.

    Stops and removes the container. Use --volumes to also delete
    the data directory at ~/.local/share/indexter/qdrant.
    """
    try:
        client = get_docker_client()
    except DockerNotAvailableError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from None

    try:
        remove_container(client, remove_volumes=volumes)
    except ContainerNotFoundError as e:
        console.print(f"[yellow]![/yellow] {e}")
        raise typer.Exit(0) from None
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to remove store: {e}")
        raise typer.Exit(1) from None
