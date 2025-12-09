"""Configuration management CLI commands."""

import os
import subprocess

import typer
from rich.console import Console
from rich.syntax import Syntax

from indexter.config import settings

config_app = typer.Typer(
    name="config",
    help="Manage global indexter configuration.",
    no_args_is_help=True,
)

console = Console()


@config_app.command(name="show")
def config_show() -> None:
    """Show global configuration."""
    console.print("[bold]Global Configuration[/bold]")
    console.print(
        f"  Config file: {str(settings.global_config_file)}", overflow="ignore", crop=False
    )
    console.print()

    if settings.global_config_file.exists():
        content = settings.global_config_file.read_text()
        syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        console.print("[dim]No global config file. Run 'indexter config init' to create one.[/dim]")


@config_app.command(name="init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    """Create the global configuration file."""
    config_path = settings.global_config_file

    if config_path.exists() and not force:
        console.print(f"[yellow]Global config already exists:[/yellow] {config_path}")
        console.print("Use --force to overwrite")
        return

    # Delete existing file if force=True
    if force and config_path.exists():
        config_path.unlink()

    settings.create_global_config()
    console.print(f"[green]✓[/green] Created global config: {config_path}")
    console.print()
    console.print("Edit the file to customize indexter settings.")
    console.print()
    console.print("[dim]For per-repository settings, add a indexter.toml to your repo[/dim]")
    console.print("[dim]or add \\[tool.indexter] to pyproject.toml.[/dim]")


@config_app.command(name="edit")
def config_edit() -> None:
    """Open global configuration in $EDITOR."""
    config_path = settings.global_config_file

    if not config_path.exists():
        console.print("[dim]No global config. Creating one first...[/dim]")
        settings.create_global_config()

    editor = os.environ.get("EDITOR", "vim")
    try:
        subprocess.run([editor, str(config_path)])  # noqa: S603
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] Editor '{editor}' not found")
        console.print(f"Set $EDITOR or edit manually: {config_path}")
        raise typer.Exit(1) from e


@config_app.command(name="path")
def config_path() -> None:
    """Print the path to the global config file."""
    # Use print instead of console.print to avoid Rich formatting/wrapping
    print(settings.global_config_file)
