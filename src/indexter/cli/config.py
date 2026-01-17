"""
Configuration CLI commands.

This module provides CLI commands for viewing Indexter's global
configuration settings. It includes commands to display the configuration file
contents and retrieve the configuration file path.
"""

import typer
from rich.console import Console
from rich.syntax import Syntax

from indexter.config import settings

config_app = typer.Typer(
    name="config",
    help="View Indexter global settings.",
    no_args_is_help=True,
)

console = Console()


@config_app.command(name="show")
def config_show() -> None:
    """Show Indexter global settings config.

    Displays the Indexter configuration file path and its contents in a
    formatted view. The configuration file is displayed with syntax
    highlighting using the Monokai theme. If the configuration file
    does not exist, a message is displayed indicating this.

    Returns:
        None: This function prints output to the console and does not
            return a value.

    Examples:
        $ indexter config show
        Indexter Settings
          Config file: /home/user/.config/indexter/config.toml

        [config file contents with syntax highlighting]
    """
    console.print("[bold]Indexter Settings[/bold]")
    console.print(f"  Config file: {str(settings.config_file)}", overflow="ignore", crop=False)
    console.print()

    if settings.config_file.exists():
        content = settings.config_file.read_text()
        syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        console.print("[dim]Config file not found.[/dim]")


@config_app.command(name="path")
def config_path() -> None:
    """Print the path to the Indexter settings config file.

    Outputs the absolute file system path to the Indexter configuration
    file. This uses plain print() instead of Rich's console.print() to
    avoid any formatting or text wrapping, making the output suitable
    for use in scripts or command substitution.

    Returns:
        None: This function prints output to stdout and does not return
            a value.

    Examples:
        $ indexter config path
        /home/user/.config/indexter/config.toml

        # Use in shell scripts
        $ cat $(indexter config path)
    """
    # Use print instead of console.print to avoid Rich formatting/wrapping
    print(settings.config_file)
