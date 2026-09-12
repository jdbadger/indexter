"""CLI entry point. Setup-only commands for M1: `list` and `remove` over the
central database directory -- no registry, everything read from each
database's own state.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from indexter.db import queries
from indexter.db.connection import IndexterDBError
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

        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        typer.echo(f"Removed {path}")
    except IndexterDBError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
