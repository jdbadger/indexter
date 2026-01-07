"""
Main CLI application and command definitions.

This module provides the main entry point for the Indexter CLI application,
which enables semantic code context retrieval for AI agents via RAG (Retrieval
Augmented Generation). It includes commands for initializing repositories,
indexing code, searching indexed content, and managing repository status.

The CLI is built using Typer for command-line parsing and Rich for enhanced
terminal output with colors, tables, and progress indicators.

Typical usage:
    $ indexter init /path/to/repo
    $ indexter index repo-name
    $ indexter search "query" repo-name
    $ indexter status
"""

# NOTE: This file uses 'cast' from typing to help with type checking of anyio.run()
# anyio.run() is typed as returning T | None, but in practice, the functions it calls
# always return a value of type T or raise an exception. To satisfy the type checker, we use
# cast() to assert the expected return type. This does not affect runtime behavior, only
# static type checking.
#
# e.g.:
# repo = cast(Repo, anyio.run(Repo.init, repo_path.resolve()))
#
# This tells the type checker that we expect Repo.init to return a Repo instance.

import logging
from pathlib import Path
from typing import Annotated, cast

import anyio
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from indexter import __version__
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import IndexResult, Repo

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
    """Print the application version and exit.

    This callback is triggered when the --version flag is used. It displays
    the current version of Indexter and exits the application.

    Args:
        value: If True, print the version and exit. If False, do nothing.

    Raises:
        typer.Exit: Always raised when value is True to exit the application.
    """
    if value:
        console.print(f"indexter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version"),
    ] = None,
) -> None:
    """
    Indexter - Semantic Code Context For Your LLM.

    This is the main callback function for the CLI application. It sets up
    logging configuration and handles global options like verbose output
    and version display.

    Args:
        verbose: Enable verbose debug logging output. Defaults to False.
        version: When provided, displays version and exits. This parameter
            is handled by version_callback. Defaults to None.

    Returns:
        None: This function configures logging and does not return a value.
    """
    # Set up logging with rich handler
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@app.command()
def init(
    path: Annotated[str, typer.Option("--path", "-p", help="Path to the git repository to index")] = ".",
    no_index: Annotated[
        bool,
        typer.Option("--no-index", "-n", help="Do not index the repository after initialization", show_default=True),
    ] = False,
) -> None:
    """
    Initialize a git repository for indexing.

    Registers a git repository with Indexter, preparing it for semantic indexing.
    This command validates the repository path, creates necessary metadata, and
    adds it to the list of managed repositories.

    Args:
        path: a string representing a filesystem path to the git repository to initialize.
            The path will be resolved to an absolute path.
        no_index: If True, do not index the repository after initialization.
            Defaults to False.

    Raises:
        typer.Exit: Exits with code 1 if the repository already exists or if an
            unexpected error occurs during initialization or indexing.

    Examples:
        $ indexter init /home/user/projects/myrepo
        ✓ Added myrepo to indexter

        Repository 'myrepo' initialized successfully and indexed successfully!

        Next steps:
          1. Use indexter search 'your query' myrepo to search the indexed code.
    """

    async def _init() -> tuple[Repo, IndexResult | None]:
        """Run all init operations in a single event loop."""
        resolved_path = Path(path).resolve()
        repo = await Repo.init(resolved_path)
        if no_index:
            return repo, None
        # Call index after init
        result = await repo.index()
        return repo, result

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_description = "Initializing..." if no_index else "Initializing and indexing..."
            progress.add_task(task_description, total=None)
            repo, result = cast(tuple[Repo, IndexResult | None], anyio.run(_init))
        console.print(f"[green]✓[/green] Added [bold]{repo.name}[/bold] to indexter")
    except RepoExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    if no_index:
        # Show next steps
        console.print()
        console.print(f"[bold]Repository '{repo.name}' initialized successfully![/bold]")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. Run [bold]indexter index {repo.name}[/bold] to index the repository.")
        console.print(f"  2. Use [bold]indexter search 'your query' {repo.name}[/bold] to search the indexed code.")
        console.print()
    else:
        # Show next steps and index summary
        console.print()
        console.print(f"[bold]Repository '{repo.name}' initialized and indexed successfully![/bold]")
        if result:
            console.print(f"  [green]✓[/green] {repo.name}: {result.summary}")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. Use [bold]indexter search 'your query' {repo.name}[/bold] to search the indexed code.")
        console.print()


@app.command()
def index(
    name: Annotated[str, typer.Argument(help="Name of the repository to index")],
    full: Annotated[
        bool,
        typer.Option("--full", "-f", help="Force full re-indexing of the repository", show_default=True),
    ] = False,
) -> None:
    """
    Index a git repository in the vector store.

    Performs semantic indexing of the specified git repository, storing code
    snippets as vector embeddings for efficient retrieval.

    By default, only changed files are indexed for efficiency. Use --full to
    force complete re-indexing of all files.

    The command tracks added, updated, and deleted nodes, and reports any
    errors encountered during the indexing process.

    Args:
        name: Name of the repository to index. Must be a repository previously
            initialized with 'indexter init'.
        full: If True, forces full re-indexing of all files in the repository,
            ignoring incremental change detection. Defaults to False.

    Raises:
        typer.Exit: Exits with code 1 if the repository is not found or if an
            unexpected error occurs.

    Examples:
        $ indexter index myrepo
        ✓ myrepo: +15 ~3 -2 (5 files synced) (1 files deleted)
        Indexing complete!

        $ indexter index myrepo --full
        ✓ myrepo: +150 ~0 -0 (50 files synced) (0 files deleted)
        Indexing complete!
    """

    async def _index() -> tuple[Repo, IndexResult]:
        """Run all index operations in a single event loop."""
        repo = await Repo.get(name)
        result = await repo.index(full)
        return repo, result

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Indexing...", total=None)
            repo, result = cast(tuple[Repo, IndexResult], anyio.run(_index))
    except RepoNotFoundError as e:
        console.print(f"[red]✗[/red] Repository not found: {name}")
        console.print("Run 'indexter init <repo_path>' to initialize the repository first.")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    if result.files_indexed == 0:
        console.print(f"  [dim]●[/dim] {repo.name}: up to date")
        console.print(
            f" [green]✓[/green] No changes detected. {result.files_checked} files checked. Repository is up to date."
        )
    else:
        console.print(f"  [green]✓[/green] {repo.name}: {result.summary}")

    if result.errors:
        console.print(f"  [yellow]Errors: {len(result.errors)}[/yellow]")
        for error in result.errors[:5]:
            console.print(f"    - {error}")
        if len(result.errors) > 5:
            console.print(f"    ... and {len(result.errors) - 5} more")
        console.print("  [yellow]Some files could not be indexed. Please check the errors above.[/yellow]")
        return

    if result.skipped_files:
        console.print(f"  [yellow]Skipped: {result.skipped_files} files[/yellow]")
        console.print(
            "  [yellow]Some files skipped during indexing due to maximum allowed file limit being exceeded.[/yellow]"
        )

    console.print("[green]Indexing complete![/green]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    name: Annotated[str, typer.Argument(help="Name of the repository to search")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results to return", show_default=True)] = 10,
) -> None:
    """Search indexed nodes in a repository.

    Performs semantic search across the indexed codebase using vector similarity.
    Returns the most relevant code snippets ranked by similarity score, displayed
    in a formatted table with scores, content previews, and file paths.

    Args:
        query: Natural language search query describing the code you're looking for.
        name: Name of the repository to search. Must be an indexed repository.
        limit: Maximum number of search results to return. Defaults to 10.

    Raises:
        typer.Exit: Exits with code 1 if the repository is not found or if an
            unexpected error occurs.

    Examples:
        $ indexter search "authentication middleware" myrepo
        ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
        ┃ Score ┃ Content          ┃ Document Path    ┃
        ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
        │ 0.856 │ def authenticate...│ src/auth/mid...  │
        └───────┴──────────────────┴──────────────────┘

        $ indexter search "error handling" myrepo --limit 5
    """

    async def _search() -> tuple[Repo, list]:
        """Run all search operations in a single event loop."""
        repo = await Repo.get(name)
        results = await repo.search(query, limit=limit)
        return repo, results

    try:
        repo, results = cast(tuple[Repo, list], anyio.run(_search))
    except RepoNotFoundError as e:
        console.print(f"[red]✗[/red] Repository not found: {name}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1) from e

    if not results:
        console.print(f"[yellow]No results found for query:[/yellow] {query}")
        return

    table = Table(title=f"Search Results for '{query}' in '{repo.name}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Content", style="magenta")
    table.add_column("Document Path", style="green")

    for result in results:
        table.add_row(
            f"{result['score']:.4f}",
            result["content"].strip().replace("\n", " ")[:50] + "...",
            str(result["file_path"]),
        )

    console.print(table)


@app.command()
def status() -> None:
    """Show status of indexed repositories.

    Displays a table of all repositories managed by Indexter, including their
    paths, indexing statistics (nodes, files, stale files), and current status.
    This helps track which repositories are indexed and identify those needing
    updates.

    Returns:
        None: This function prints a formatted table to the console and does
            not return a value.

    Examples:
        $ indexter status
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ Name    ┃ Path           ┃ Nodes ┃ Files ┃ Stale Files ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━┩
        │ myrepo  │ /home/user/... │ 1250  │ 45    │ 2           │
        │ webapp  │ /home/user/... │ 3420  │ 128   │ 0           │
        └─────────┴────────────────┴───────┴───────┴─────────────┘
    """

    async def _get_all_statuses() -> list[tuple[Repo, dict | None, str | None]]:
        """Get status for all repos in a single event loop."""
        repos = await Repo.list()
        results: list[tuple[Repo, dict | None, str | None]] = []
        for repo in repos:
            try:
                repo_status = await repo.status()
                results.append((repo, repo_status, None))
            except Exception as e:
                results.append((repo, None, str(e)))
        return results

    statuses = cast(list[tuple[Repo, dict | None, str | None]], anyio.run(_get_all_statuses))

    if not statuses:
        console.print("[bold]Repositories[/bold]")
        console.print("  No repositories indexed. Run 'indexter index <repo_path>' to index a repository.")
        console.print()
        return

    table = Table(title="Indexed Repositories")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Nodes", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Stale Files", justify="right")

    for repo, repo_status, error in statuses:
        if error:
            table.add_row(
                repo.name,
                str(repo.path),
                "-",
                f"[red]Error: {error}[/red]",
                "-",
            )
        elif repo_status:
            table.add_row(
                repo.name,
                str(repo.path),
                str(repo_status.get("nodes_indexed", "-")),
                str(repo_status.get("documents_indexed", "-")),
                str(repo_status.get("documents_indexed_stale", "-")),
            )

    console.print(table)
    console.print()


@app.command()
def forget(
    name: Annotated[str, typer.Argument(help="Name of the repository to forget")],
) -> None:
    """Forget a repository (remove from indexter and delete indexed data).

    Removes a repository from Indexter's management and deletes all associated
    indexed data from the vector store. This operation cannot be undone. The
    original repository files remain unchanged.

    Args:
        name: Name of the repository to remove. Must be a previously initialized
            repository.

    Raises:
        typer.Exit: Exits with code 1 if the repository is not found or if an
            unexpected error occurs during removal.

    Examples:
        $ indexter forget myrepo
        ✓ Repository 'myrepo' is forgotten.
    """
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
