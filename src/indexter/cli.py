"""CLI entry point. Setup-only commands: `list` and `remove` over the central
database directory (no registry, everything read from each database's own
state), plus `init` and `reindex` to build and refresh a repository's index.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from indexter.banner import show_banner
from indexter.config import ConfigError, load_settings
from indexter.db import queries
from indexter.db.connection import IndexterDBError, delete_database_files
from indexter.index.embed import EmbeddingError, make_embedder
from indexter.index.graph import ResolveReport
from indexter.index.sync import IndexResult, index_repository
from indexter.mcp.server import run_server
from indexter.parse.models import RefKind
from indexter.paths import data_dir, db_path
from indexter.progress import ConsoleProgress, NullProgress, Progress

app = typer.Typer(
    name="indexter",
    help="indexter - local code search and graph for AI agents.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"indexter {importlib.metadata.version('indexter')}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    pass


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


def _render_index_result(repo: Path, result: IndexResult, *, existing: str) -> None:
    """Print the index summary; `existing` is the headline for an index that
    already existed, which differs between `init` and `reindex`."""
    report = result.report
    if result.status == "created":
        typer.echo(f"Initialized {repo} -> {result.db_path}")
    elif result.status == "rebuilt":
        typer.echo(f"Rebuilt {repo} -> {result.db_path}")
    else:
        typer.echo(f"{existing} -> {result.db_path}")

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


QuietOption = Annotated[bool, typer.Option("--quiet", help="Suppress progress narration on stderr.")]
ProgressOption = Annotated[
    bool, typer.Option("--progress", help="Narrate progress on stderr even when it is not a terminal.")
]


def _stderr_is_interactive() -> bool:
    return sys.stderr.isatty()


def _check_narration_flags(quiet: bool, progress: bool) -> None:
    if quiet and progress:
        typer.echo("--quiet and --progress cannot be used together.", err=True)
        raise typer.Exit(2)


@contextmanager
def _narration(quiet: bool, progress: bool) -> Iterator[Progress]:
    """Narrate on stderr when it is a terminal, unless `--quiet`; `--progress`
    narrates regardless. Results stay on stdout either way.
    """
    if quiet or not (progress or _stderr_is_interactive()):
        yield NullProgress()
        return
    with ConsoleProgress() as narrator:
        yield narrator


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Repository path to index")] = Path(),  # noqa: B008
    quiet: QuietOption = False,
    progress: ProgressOption = False,
) -> None:
    """Create (or re-sync) a repository's index."""
    _check_narration_flags(quiet, progress)
    try:
        if not path.is_dir():
            typer.echo(f"{path} is not a directory.", err=True)
            raise typer.Exit(1)

        first_time = not db_path(path).is_file()
        settings = load_settings(path)
        embedder = make_embedder(settings)
        if first_time and not quiet and _stderr_is_interactive():
            show_banner(version=importlib.metadata.version("indexter"))
        with _narration(quiet, progress) as narrator:
            result = index_repository(path, settings, embedder, progress=narrator)
        _render_index_result(path, result, existing=f"{path} already initialized")
    except (IndexterDBError, ConfigError, EmbeddingError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command()
def reindex(
    path: Annotated[Path, typer.Argument(help="Repository path to re-index")] = Path(),  # noqa: B008
    full: Annotated[bool, typer.Option("--full", help="Delete and rebuild the database before syncing")] = False,
    quiet: QuietOption = False,
    progress: ProgressOption = False,
) -> None:
    """Re-sync a previously initialized repository's index."""
    _check_narration_flags(quiet, progress)
    try:
        if not path.is_dir():
            typer.echo(f"{path} is not a directory.", err=True)
            raise typer.Exit(1)

        if not db_path(path).is_file():
            typer.echo(f"No index found for {path}. Run `indexter init {path}` first.", err=True)
            raise typer.Exit(1)

        settings = load_settings(path)
        embedder = make_embedder(settings)
        with _narration(quiet, progress) as narrator:
            result = index_repository(path, settings, embedder, full=full, progress=narrator)
        _render_index_result(path, result, existing=f"Reindexed {path}")
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


@app.command()
def mcp(
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Repository to serve; defaults to the working directory")
    ] = None,
) -> None:
    """Start the MCP server over stdio."""
    if repo is not None and not repo.is_dir():
        typer.echo(f"{repo} is not a directory.", err=True)
        raise typer.Exit(1)
    run_server(repo)


def _claude_config_dir() -> Path:
    value = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(value) if value else Path.home() / ".claude"


def _skill_content() -> str:
    return importlib.resources.files("indexter.skill").joinpath("SKILL.md").read_text()


@app.command()
def skill(
    install: Annotated[bool, typer.Option("--install", help="Install the skill instead of printing it")] = False,
    install_dir: Annotated[
        Path | None,
        typer.Option("--dir", help="Install directory, replacing <config>/skills/indexter (requires --install)"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing, different skill file (requires --install)")
    ] = False,
) -> None:
    """Print the indexter skill, or install it into an agent's skills directory."""
    if not install:
        if install_dir is not None or force:
            typer.echo("--dir and --force require --install.", err=True)
            raise typer.Exit(1)
        typer.echo(_skill_content(), nl=False)
        return

    target_dir = install_dir if install_dir is not None else _claude_config_dir() / "skills" / "indexter"
    target = target_dir / "SKILL.md"
    content = _skill_content()

    if target.is_file():
        if target.read_text() == content:
            typer.echo(f"{target} is up to date.")
            return
        if not force:
            typer.echo(
                f"{target} already exists and differs from the packaged skill. Use --force to overwrite.", err=True
            )
            raise typer.Exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    typer.echo(f"Installed {target}")
