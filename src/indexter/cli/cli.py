"""Main CLI app and commands."""

import logging
from pathlib import Path
from typing import Annotated

import anyio
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from indexter import __version__
from indexter.config import settings
from indexter.exceptions import InvalidGitRepositoryError, RepoExistsError, RepoNotFoundError
from indexter.models import Repo

from .config import config_app

app = typer.Typer(
    name="indexter",
    help="indexter - Enhanced codebase context for AI agents via RAG.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(config_app, name="config")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"indexter {__version__}")
        raise typer.Exit()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def setup_global_config() -> None:
    """Ensure global config file exists."""
    if not settings.global_config_file.exists():
        settings.create_global_config()
        console.print(f"[dim]Created global config: {settings.global_config_file}[/dim]")


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version"),
    ] = None,
) -> None:
    """indexter - Enhanced codebase context for AI agents via RAG."""
    setup_global_config()
    setup_logging(verbose)


@app.command()
def init(
    repo_path: Annotated[Path, typer.Argument(help="Path to the git repository to index")],
) -> None:
    """Initialize a git repository for indexing."""
    try:
        repo = anyio.run(Repo.init, repo_path.resolve())
        console.print(f"[green]✓[/green] Added [bold]{repo.name}[/bold] to indexter")
    except InvalidGitRepositoryError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e
    except RepoExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    # Show next steps
    console.print()
    console.print(f"[bold]Repository '{repo.name}' initialized successfully![/bold]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Run [bold]indexter index {repo.name}[/bold] to index the repository.")
    console.print(
        f"  2. Use [bold]indexter search 'your query' {repo.name}[/bold] "
        f"to search the indexed code."
    )
    console.print()


@app.command()
def index(
    name: Annotated[str, typer.Argument(help="Name of the repository to index")],
    full: Annotated[
        bool,
        typer.Option(
            "--full", "-f", help="Force full re-indexing of the repository", show_default=True
        ),
    ] = False,
) -> None:
    """
    Sync a git repository to the vector store.

    If the repository is already indexed, only changed files are
    synced unless '--full' is specified.
    """
    try:
        repo = anyio.run(Repo.get, name)
    except RepoNotFoundError as e:
        console.print(f"[red]✗[/red] Repository not found: {name}")
        console.print("Run 'indexter init <repo_path>' to initialize the repository first.")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Syncing {repo.name}...", total=None)

        result = anyio.run(repo.index, full)

        if result.files_synced == 0:
            console.print(f"  [dim]●[/dim] {repo.name}: up to date")
            console.print(
                f" [green]✓[/green] No changes detected. {result.files_checked} "
                f"files checked. Repository is up to date."
            )
        else:
            console.print(
                f"  [green]✓[/green] {repo.name}: "
                f"+{result.nodes_added} ~{result.nodes_updated} -{result.nodes_deleted} "
                f"({len(result.files_synced)} files synced) "
                f"({len(result.files_deleted)} files deleted)"
            )

        if result.errors:
            console.print(f"  [yellow]Errors: {len(result.errors)}[/yellow]")
            for error in result.errors[:5]:
                console.print(f"    - {error}")
            if len(result.errors) > 5:
                console.print(f"    ... and {len(result.errors) - 5} more")
            console.print(
                "  [yellow]Some files could not be indexed. Please check the errors above.[/yellow]"
            )
            return

        if result.skipped_files:
            console.print(f"  [yellow]Skipped: {len(result.skipped_files)} files[/yellow]")
            for skipped in result.skipped_files[:5]:
                console.print(f"    - {skipped}")
            if len(result.skipped_files) > 5:
                console.print(f"    ... and {len(result.skipped_files) - 5} more")
            console.print(
                "  [yellow]Some files skipped during indexing due to maximum allowed "
                "file limit being exceeded.[/yellow]"
            )

        console.print("[green]Indexing complete![/green]")
        return


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    name: Annotated[str, typer.Argument(help="Name of the repository to search")],
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Number of results to return", show_default=True)
    ] = 10,
) -> None:
    """Search indexed nodes in a repository."""
    try:
        repo = anyio.run(Repo.get, name)
    except RepoNotFoundError as e:
        console.print(f"[red]✗[/red] Repository not found: {name}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Searching {repo.name}...", total=None)

        results = anyio.run(repo.search, query, limit)

        if not results:
            console.print(f"[yellow]No results found for query:[/yellow] {query}")
            return

        table = Table(title=f"Search Results for '{query}' in '{repo.name}'")
        table.add_column("Score", justify="right", style="cyan", no_wrap=True)
        table.add_column("Content", style="magenta")
        table.add_column("Document Path", style="green")

        for score, content, doc_path in results:
            table.add_row(
                f"{score:.4f}", content.strip().replace("\n", " ")[:50] + "...", str(doc_path)
            )

        console.print(table)


@app.command()
def status() -> None:
    """Show status of indexed repositories."""
    # Repository status
    repos = anyio.run(Repo.list)

    if not repos:
        console.print("[bold]Repositories[/bold]")
        console.print(
            "  No repositories indexed. Run 'indexter index <repo_path>' to index a repository."
        )
        console.print()
        return

    table = Table(title="Indexed Repositories")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Nodes", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Stale Files", justify="right")

    for repo in repos:
        try:
            status = anyio.run(repo.status)
            table.add_row(
                repo.name,
                str(repo.path),
                str(status.get("nodes_indexed", "-")),
                str(status.get("documents_indexed", "-")),
                str(status.get("documents_indexed_stale", "-")),
            )
        except Exception as e:
            table.add_row(
                repo.name,
                str(repo.path),
                "-",
                f"[red]Error: {e}[/red]",
                "-",
            )

    console.print(table)
    console.print()


@app.command()
def forget(
    name: Annotated[str, typer.Argument(help="Name of the repository to forget")],
) -> None:
    """Forget a repository (remove from indexter and delete indexed data)."""
    try:
        anyio.run(Repo.remove, name)
    except RepoNotFoundError as e:
        console.print(f"[red]✗[/red] Repository not found: {name}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e
    else:
        console.print(f"[green]✓[/green] Repository '{name}' is forgotten.")
