"""CLI entry point. Setup-only commands: `list` and `remove` over the central
database directory (no registry, everything read from each database's own
state), plus `init` and `reindex` to build and refresh a repository's index.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from indexter.config import ConfigError, load_settings
from indexter.db import queries
from indexter.db.connection import IndexterDBError, delete_database_files
from indexter.index.embed import EmbeddingError, make_embedder
from indexter.index.graph import ResolveReport
from indexter.index.sync import IndexResult, index_repository
from indexter.parse.models import RefKind
from indexter.paths import data_dir, db_path

app = typer.Typer(
    name="indexter",
    help="indexter - local code search and graph for AI agents.",
    no_args_is_help=True,
)


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _format_time(timestamp: float | None) -> str:
    if timestamp is None:
        return "never"
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")  # noqa: DTZ006 - local time is fine for CLI display


def _render_summary(summary: queries.RepoSummary) -> str:
    missing = " [missing]" if not summary.repo_exists else ""
    return (
        f"{summary.repo_path}{missing}\n"
        f"  nodes={summary.node_count} model={summary.model} dim={summary.dim} "
        f"schema_version={summary.schema_version} size={_format_size(summary.size_bytes)} "
        f"indexed_at={_format_time(summary.indexed_at)}"
    )


def _render_resolution(resolution: ResolveReport) -> None:
    summary = resolution.summary

    edges_by_kind: dict[str, int] = {}
    for (kind, _confidence), count in summary.edges_by_kind_confidence.items():
        edges_by_kind[kind] = edges_by_kind.get(kind, 0) + count
    edges_str = " ".join(f"{kind}={count}" for kind, count in sorted(edges_by_kind.items()))

    calls_by_outcome: dict[str, int] = {}
    for (ref_kind, status, confidence), count in summary.refs_by_outcome.items():
        if ref_kind != RefKind.CALLS.value:
            continue
        label = status if confidence is None else f"{status}/{confidence}"
        calls_by_outcome[label] = calls_by_outcome.get(label, 0) + count
    calls_str = " ".join(f"{label}={count}" for label, count in sorted(calls_by_outcome.items()))

    typer.echo(
        f"  resolution: edges[{edges_str}] calls[{calls_str}] "
        f"external_modules={summary.external_node_count} elapsed={resolution.elapsed_seconds:.2f}s"
    )


def _render_index_result(repo: Path, result: IndexResult) -> None:
    report = result.report
    if result.status == "created":
        typer.echo(f"Initialized {repo} -> {result.db_path}")
    elif result.status == "rebuilt":
        typer.echo(f"Rebuilt {repo} -> {result.db_path}")
    else:
        typer.echo(f"{repo} already initialized -> {result.db_path}")

    typer.echo(
        f"  added={len(report.added)} changed={len(report.changed)} removed={len(report.removed)} "
        f"unchanged={len(report.unchanged)} nodes_written={report.nodes_written} "
        f"nodes_deleted={report.nodes_deleted} refs_written={report.refs_written} "
        f"texts_embedded={report.texts_embedded} elapsed={report.elapsed_seconds:.2f}s"
    )
    if report.resolution is not None:
        _render_resolution(report.resolution)
    for path, error in sorted(report.errors.items()):
        typer.echo(f"  error: {path}: {error}", err=True)


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Repository path to index")] = Path(),  # noqa: B008
) -> None:
    """Create (or re-sync) a repository's index."""
    try:
        if not path.is_dir():
            typer.echo(f"{path} is not a directory.", err=True)
            raise typer.Exit(1)

        settings = load_settings(path)
        embedder = make_embedder(settings)
        result = index_repository(path, settings, embedder)
        _render_index_result(path, result)
    except (IndexterDBError, ConfigError, EmbeddingError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command()
def reindex(
    path: Annotated[Path, typer.Argument(help="Repository path to re-index")] = Path(),  # noqa: B008
    full: Annotated[bool, typer.Option("--full", help="Delete and rebuild the database before syncing")] = False,
) -> None:
    """Re-sync a previously initialized repository's index."""
    try:
        if not path.is_dir():
            typer.echo(f"{path} is not a directory.", err=True)
            raise typer.Exit(1)

        if not db_path(path).is_file():
            typer.echo(f"No index found for {path}. Run `indexter init {path}` first.", err=True)
            raise typer.Exit(1)

        settings = load_settings(path)
        embedder = make_embedder(settings)
        result = index_repository(path, settings, embedder, full=full)
        _render_index_result(path, result)
    except (IndexterDBError, ConfigError, EmbeddingError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command("list")
def list_repos() -> None:
    """List indexed repositories."""
    directory = data_dir()
    db_files = sorted(directory.glob("*.db")) if directory.is_dir() else []

    if not db_files:
        typer.echo("No repositories indexed.")
        return

    for path in db_files:
        summary = queries.read_summary(path)
        if isinstance(summary, queries.CorruptDatabase):
            typer.echo(f"{path.name}: corrupt database ({summary.error})")
        else:
            typer.echo(_render_summary(summary))


def _resolve_target(target: str) -> Path:
    """A remove target is either a database filename in the data directory,
    or a repository path -- resolved through the same derivation used to
    create its database, so it works even if the repository no longer exists.
    """
    candidate = data_dir() / target
    if candidate.suffix == ".db" or candidate.is_file():
        return candidate
    return db_path(target)


@app.command()
def remove(
    target: Annotated[str, typer.Argument(help="Repository path or database filename to remove")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")] = False,
) -> None:
    """Remove an indexed repository's database. Never touches the repository itself."""
    try:
        path = _resolve_target(target)
        if not path.is_file():
            typer.echo(f"No indexed database found for {target!r}.", err=True)
            raise typer.Exit(1)

        if not yes and not typer.confirm(f"Remove database for {target}?"):
            return

        delete_database_files(path)
        typer.echo(f"Removed {path}")
    except IndexterDBError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
