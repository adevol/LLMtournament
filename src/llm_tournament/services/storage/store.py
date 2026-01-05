"""Unified tournament storage layer composing file and database stores."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog
import yaml
from sqlmodel import SQLModel, create_engine

from llm_tournament import __version__
from llm_tournament.core.config import TournamentConfig

from .db_store import DBStore
from .file_store import FileStore
from .report_store import ReportStore

logger = structlog.get_logger()


class TournamentStore:
    """Unified persistence layer composing file and database stores.

    Handles:
    - File-based storage for essays, feedback, revisions (Markdown)
    - SQLModel-based storage for matches, ratings (DuckDB/SQL)
    - JSONL backup files for matches

    Attributes:
        files: FileStore for essay/feedback/revision operations.
        db: DBStore for match and rating operations.
        reports: ReportStore for report generation and export.
    """

    def __init__(
        self,
        config: TournamentConfig,
        run_id: str | None = None,
    ) -> None:
        """Initialize tournament store.

        Args:
            config: Tournament configuration.
            run_id: Optional run identifier (defaults to timestamp).
        """
        self.config = config
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(config.output_dir) / self.run_id
        self._db_path = self.base_dir / "tournament.duckdb"
        self._engine = None
        self._init_directories()
        self._init_db()
        self._init_stores()
        self._save_metadata()

    def _init_directories(self) -> None:
        """Create base output directory."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("store_init", run_id=self.run_id, path=str(self.base_dir))

    def _init_db(self) -> None:
        """Initialize DuckDB database and create tables."""
        db_url = f"duckdb:///{self._db_path}"
        self._engine = create_engine(db_url)
        SQLModel.metadata.create_all(self._engine)

    def _init_stores(self) -> None:
        """Initialize composed storage components."""
        self.files = FileStore(self.base_dir)
        self.db = DBStore(self._engine, self.files, self.base_dir)
        self.reports = ReportStore(self.base_dir, self.files)

    def _save_metadata(self) -> None:
        """Save config snapshot and run metadata."""
        config_path = self.base_dir / "config_snapshot.yaml"
        with config_path.open("w") as f:
            config_dict = self.config.model_dump(exclude={"api_key"})
            yaml.dump(config_dict, f, default_flow_style=False)

        metadata = {
            "run_id": self.run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "seed": self.config.seed,
            "llm_tournament_version": __version__,
            "ranking_algorithm": self.config.ranking.algorithm,
        }
        metadata_path = self.base_dir / "run_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

    async def close(self) -> None:
        """Dispose of the database engine."""
        if self._engine:
            self._engine.dispose()

    def close_sync(self) -> None:
        """Synchronously dispose of the database engine."""
        if self._engine:
            self._engine.dispose()
