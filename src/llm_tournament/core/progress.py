"""Progress tracking utilities for tournament operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


class TournamentProgress:
    """Progress tracking for tournament operations.

    Provides a simple interface for tracking progress of long-running
    operations like essay generation, ranking rounds, etc.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize progress tracker.

        Args:
            console: Optional console instance. If None, creates a new one.
        """
        self.console = console or Console()

    async def track_generation(
        self,
        items: list[Any],
        generator_func: Callable[[Any], Any],
        description: str = "Processing",
    ) -> None:
        """Track progress for processing multiple items.

        Args:
            items: List of items to process.
            generator_func: Async function to call for each item.
            description: Description of the operation.

        Returns:
            None.
        """
        with Progress(
            SpinnerColumn("line"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]{description}...", total=len(items))

            for item in items:
                await generator_func(item)
                label = getattr(item, "title", str(item))
                progress.update(task, advance=1, description=f"[cyan]{description}: {label}")

    async def track_rounds(
        self,
        rounds: int,
        round_func: Callable[[int], Any],
        description: str = "Running rounds",
    ) -> None:
        """Track progress for multiple rounds.

        Args:
            rounds: Number of rounds to process.
            round_func: Async function to call for each round.
            description: Description of the operation.

        Returns:
            None.
        """
        with Progress(
            SpinnerColumn("line"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[green]{description}...", total=rounds)

            for round_num in range(1, rounds + 1):
                await round_func(round_num)
                desc = f"[green]{description}: {round_num}/{rounds}"
                progress.update(task, advance=1, description=desc)
