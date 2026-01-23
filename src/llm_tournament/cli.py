"""CLI for LLM Tournament."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Literal

import structlog
import typer
from rich.console import Console
from rich.logging import RichHandler

from llm_tournament import __version__
from llm_tournament.core.config import TournamentConfig, load_config
from llm_tournament.core.errors import ConfigurationError
from llm_tournament.pipeline import run_tournament
from llm_tournament.services.llm import create_client
from llm_tournament.services.llm.client import LLMClient

# Preset configurations for common use cases
SCOPE_PRESETS: dict[Literal["small", "medium", "full"], dict[str, int | None]] = {
    "small": {
        "max_topics": 1,
        "max_writers": 3,
        "max_critics": 3,
        "rounds": 2,
    },
    "medium": {
        "max_topics": 2,
        "max_writers": 5,
        "max_critics": 5,
        "rounds": 5,
    },
    "full": {
        "max_topics": None,
        "max_writers": None,
        "max_critics": None,
        "rounds": None,
    },
}

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

app = typer.Typer(
    name="llm-tournament",
    help="LLM Tournament Evaluator - Compare OpenRouter models via pairwise Elo ranking",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"llm-tournament v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """LLM Tournament Evaluator CLI."""


def _apply_scope_preset(
    scope: Literal["small", "medium", "full"] | None,
    max_topics: int | None,
    max_writers: int | None,
    max_critics: int | None,
    rounds: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if scope is None:
        return max_topics, max_writers, max_critics, rounds

    preset_values = SCOPE_PRESETS[scope]
    return (
        max_topics if max_topics is not None else preset_values["max_topics"],
        max_writers if max_writers is not None else preset_values["max_writers"],
        max_critics if max_critics is not None else preset_values["max_critics"],
        rounds if rounds is not None else preset_values["rounds"],
    )


def _apply_cli_overrides(
    config: TournamentConfig,
    simple_mode: bool | None,
    rounds: int | None,
    ranking_algorithm: str | None,
) -> None:
    if simple_mode is not None:
        config.simple_mode = simple_mode
    if rounds is not None:
        config.ranking.rounds = rounds
    if ranking_algorithm is not None:
        config.ranking.algorithm = ranking_algorithm


def _create_client(
    config: TournamentConfig,
    cache_path: Path | None,
    use_cache: bool,
    dry_run: bool,
) -> LLMClient:
    if dry_run:
        console.print("[yellow]DRY RUN MODE - using fake LLM responses[/yellow]")
        return create_client(dry_run=True, seed=config.seed)

    api_key = config.get_api_key()
    return create_client(
        api_key=api_key,
        cache_path=cache_path,
        use_cache=use_cache,
        dry_run=False,
    )


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="Path to config YAML file")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Use fake LLM responses, no API calls")
    ] = False,
    use_cache: Annotated[
        bool, typer.Option("--use-cache/--no-cache", help="Use cached API responses")
    ] = True,
    simple_mode: Annotated[
        bool | None, typer.Option("--simple-mode", help="Rank only v0 essays")
    ] = None,
    scope: Annotated[
        Literal["small", "medium", "full"] | None,
        typer.Option("--scope", help="Execution scope preset (small/medium/full)"),
    ] = None,
    max_topics: Annotated[
        int | None, typer.Option("--max-topics", help="Limit number of topics")
    ] = None,
    max_writers: Annotated[
        int | None, typer.Option("--max-writers", help="Limit number of writers")
    ] = None,
    max_critics: Annotated[
        int | None, typer.Option("--max-critics", help="Limit number of critics")
    ] = None,
    rounds: Annotated[int | None, typer.Option("--rounds", help="Number of ranking rounds")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Custom run ID")] = None,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", help="Maximum concurrent API calls")
    ] = 5,
    ranking_algorithm: Annotated[
        str | None,
        typer.Option("--ranking", help="Ranking algorithm: elo or trueskill"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Verbose output")] = False,
) -> None:
    """Run a tournament with the given configuration.

    Args:
        config_path: Path to YAML configuration file.
        dry_run: If True, use fake LLM client.
        use_cache: Whether to cache API responses.
        simple_mode: Override config simple_mode setting.
        scope: Execution scope preset (small/medium/full). Applies predefined
            limits for max_topics, max_writers, max_critics, and rounds.
        max_topics: Limit number of topics to process.
        max_writers: Limit number of writers.
        max_critics: Limit number of critics.
        rounds: Override number of ranking rounds.
        run_id: Custom run ID (default: timestamp).
        max_concurrency: Maximum concurrent API calls.
        ranking_algorithm: Override ranking algorithm.
        verbose: Enable verbose logging.
    """
    # Configure logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    try:
        # Load config
        console.print(f"[bold]Loading config:[/bold] {config_path}")
        config = load_config(config_path)

        max_topics, max_writers, max_critics, rounds = _apply_scope_preset(
            scope, max_topics, max_writers, max_critics, rounds
        )
        _apply_cli_overrides(config, simple_mode, rounds, ranking_algorithm)

        # Create client
        cache_path = Path(config.output_dir) / ".cache" / "api_cache.duckdb" if use_cache else None

        client = _create_client(config, cache_path, use_cache, dry_run)

        # Run tournament (async)
        console.print("[bold green]Starting tournament...[/bold green]")
        console.print(f"  Ranking algorithm: {config.ranking.algorithm}")
        console.print(f"  Max concurrency: {max_concurrency}")

        async def _run() -> None:
            store = await run_tournament(
                config=config,
                client=client,
                run_id=run_id,
                max_topics=max_topics,
                max_writers=max_writers,
                max_critics=max_critics,
                max_concurrency=max_concurrency,
            )
            console.print("\n[bold green]Tournament complete![/bold green]")
            console.print(f"Results saved to: {store.base_dir}")

            # Close client
            await client.close()
            await store.close()

        asyncio.run(_run())

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except ConfigurationError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command()
def validate(
    config_path: Annotated[Path, typer.Argument(help="Path to config YAML file")],
) -> None:
    """Validate a configuration file without running.

    Args:
        config_path: Path to YAML configuration file.
    """
    try:
        config = load_config(config_path)
        console.print("[green]Configuration is valid![/green]")
        console.print(f"  Writers: {len(config.writers)}")
        console.print(f"  Critics: {len(config.critics)}")
        console.print(f"  Judges: {len(config.judges)}")
        console.print(f"  Topics: {len(config.topics)}")
        console.print(f"  Simple mode: {config.simple_mode}")
        console.print(f"  Ranking algorithm: {config.ranking.algorithm}")
        console.print(f"  Rounds: {config.ranking.rounds}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except ConfigurationError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def info() -> None:
    """Show tool information and example commands."""
    console.print("[bold]LLM Tournament Evaluator[/bold]")
    console.print(f"Version: {__version__}\n")

    console.print("[bold]Example Commands:[/bold]")
    console.print("  # Dry run (no API calls)")
    console.print("  uv run llm-tournament run config.yaml --dry-run\n")

    console.print("  # Real run for one topic")
    console.print("  uv run llm-tournament run config.yaml --max-topics 1\n")

    console.print("  # Simple mode (skip revision)")
    console.print("  uv run llm-tournament run config.yaml --simple-mode\n")

    console.print("  # Use TrueSkill ranking")
    console.print("  uv run llm-tournament run config.yaml --ranking trueskill\n")

    console.print("  # High concurrency (faster)")
    console.print("  uv run llm-tournament run config.yaml --max-concurrency 10\n")

    console.print("  # Limit scope")
    console.print("  uv run llm-tournament run config.yaml --max-writers 3 --max-critics 3\n")

    console.print("  # Validate config")
    console.print("  uv run llm-tournament validate config.yaml")


if __name__ == "__main__":
    app()
