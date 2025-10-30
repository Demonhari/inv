# logger.py
from rich.console import Console
from rich.panel import Panel

console = Console()

def stage(title: str, body: str | None = None):
    console.rule(f"[bold cyan]{title}")
    if body:
        console.print(Panel.fit(body, border_style="cyan"))

def info(msg: str):
    console.print(f"[cyan]- {msg}[/cyan]")

def success(msg: str):
    console.print(f"[bold green]✓ {msg}[/bold green]")

def warn(msg: str):
    console.print(f"[bold yellow]! {msg}[/bold yellow]")

def error(msg: str):
    console.print(f"[bold red]✗ {msg}[/bold red]")
