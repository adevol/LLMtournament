"""Tests for pipeline in dry-run mode."""

import json
import tempfile

import pytest

from llm_tournament.core.config import TopicConfig, TournamentConfig
from llm_tournament.pipeline import (
    TournamentPipeline,
    run_tournament,
)
from llm_tournament.services.llm import FakeLLMClient
from llm_tournament.services.storage import TournamentStore


@pytest.fixture
def minimal_config():
    """Create minimal test configuration."""
    return TournamentConfig(
        writers=["test/writer-a", "test/writer-b"],
        critics=["test/critic-a", "test/critic-b"],
        judges=["test/judge-a", "test/judge-b"],
        topics=[
            TopicConfig(
                title="Test Topic",
                prompts={"Essay": "Write a comprehensive essay"},
            )
        ],
        seed=42,
        simple_mode=True,  # Faster for tests
    )


@pytest.fixture
def minimal_config_full_mode():
    """Create minimal test configuration with full mode."""
    return TournamentConfig(
        writers=["test/writer-a", "test/writer-b"],
        critics=["test/critic-a"],
        judges=["test/judge-a", "test/judge-b"],
        topics=[
            TopicConfig(
                title="Test Topic",
                prompts={"Essay": "Write a comprehensive essay"},
            )
        ],
        seed=42,
        simple_mode=False,
    )


class TestPipelineDryRun:
    """Tests for pipeline dry-run execution."""

    async def test_run_simple_mode(self, minimal_config):
        """Test pipeline runs in simple mode (v0 only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config, run_id="test_run")

            pipeline = TournamentPipeline(minimal_config, client, store)
            await pipeline.run()

            # Check v0 essays were created
            v0_dir = store.paths.topic_dir("test-topic") / "v0"
            assert (v0_dir / "test__writer-a.md").exists()
            assert (v0_dir / "test__writer-b.md").exists()

            # Check ranking outputs
            ranking_dir = store.paths.topic_dir("test-topic") / "ranking"
            assert (ranking_dir / "leaderboard.csv").exists()
            assert (ranking_dir / "leaderboard.md").exists()

            store.close_sync()

    async def test_run_full_mode(self, minimal_config_full_mode):
        """Test pipeline runs in full mode (with critique and revision)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config_full_mode.output_dir = tmpdir
            minimal_config_full_mode.ranking.rounds = 1  # Faster
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config_full_mode, run_id="test_run")

            pipeline = TournamentPipeline(minimal_config_full_mode, client, store)
            await pipeline.run()

            # Check v0 essays
            v0_dir = store.paths.topic_dir("test-topic") / "v0"
            assert (v0_dir / "test__writer-a.md").exists()

            # Check feedback was created
            feedback_dir = store.paths.topic_dir("test-topic") / "feedback"
            assert (feedback_dir / "test__writer-a__test__critic-a.md").exists()

            # Check v1 essays
            v1_dir = store.paths.topic_dir("test-topic") / "v1"
            assert (v1_dir / "test__writer-a__test__critic-a.md").exists()

            store.close_sync()

    async def test_run_with_limits(self, minimal_config):
        """Test pipeline runs with limited config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            # Limit writers in config directly
            minimal_config.writers = [minimal_config.writers[0]]
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config, run_id="test_run")

            pipeline = TournamentPipeline(minimal_config, client, store)
            await pipeline.run()

            # Only one writer should have essay
            v0_dir = store.paths.topic_dir("test-topic") / "v0"
            essays = list(v0_dir.glob("*.md"))
            assert len(essays) == 1

            store.close_sync()

    async def test_metadata_saved(self, minimal_config):
        """Test run metadata is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            store = TournamentStore(minimal_config, run_id="test_run")

            # Check metadata files exist
            assert (store.base_dir / "config_snapshot.yaml").exists()
            assert (store.base_dir / "run_metadata.json").exists()

            store.close_sync()

    async def test_matches_jsonl_created(self, minimal_config):
        """Test matches JSONL is created during ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config, run_id="test_run")

            pipeline = TournamentPipeline(minimal_config, client, store)
            await pipeline.run()

            jsonl_path = store.paths.topic_dir("test-topic") / "ranking" / "matches.jsonl"
            assert jsonl_path.exists()

            # Check content is valid JSONL
            with jsonl_path.open() as f:
                lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                data = json.loads(line)
                assert "winner" in data
                assert "essay_a_id" in data

            store.close_sync()

    async def test_convenience_function(self, minimal_config):
        """Test run_tournament convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            client = FakeLLMClient(seed=42)

            store = await run_tournament(
                config=minimal_config,
                client=client,
                run_id="conv_test",
            )

            assert store.run_id == "conv_test"
            assert (store.base_dir / "test-topic" / "ranking" / "leaderboard.md").exists()

            store.close_sync()

    async def test_convenience_function_with_limits(self):
        """Test run_tournament applies optional topic/writer/critic limits."""
        config = TournamentConfig(
            writers=["test/writer-a", "test/writer-b"],
            critics=["test/critic-a", "test/critic-b"],
            judges=["test/judge-a", "test/judge-b"],
            topics=[
                TopicConfig(title="Test Topic A", prompts={"Essay": "Write essay A"}),
                TopicConfig(title="Test Topic B", prompts={"Essay": "Write essay B"}),
            ],
            seed=42,
            simple_mode=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.ranking.rounds = 1
            client = FakeLLMClient(seed=42)

            store = await run_tournament(
                config=config,
                client=client,
                run_id="limit_test",
                max_topics=1,
                max_writers=1,
                max_critics=1,
            )

            assert len(store.config.topics) == 1
            assert len(store.config.writers) == 1
            assert len(store.config.critics) == 1
            assert (store.base_dir / "test-topic-a").exists()
            assert not (store.base_dir / "test-topic-b").exists()

            store.close_sync()

    async def test_convenience_function_rejects_non_positive_limits(self, minimal_config):
        """Test run_tournament validates optional limits."""
        client = FakeLLMClient(seed=42)
        with pytest.raises(ValueError, match="max_topics must be greater than 0"):
            await run_tournament(
                config=minimal_config,
                client=client,
                max_topics=0,
            )


class TestEssayContent:
    """Tests for essay content structure."""

    async def test_v0_essay_has_sections(self, minimal_config):
        """Test v0 essays have required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config, run_id="test")

            pipeline = TournamentPipeline(minimal_config, client, store)
            await pipeline.run()

            essay = await store.essays.load_essay("test-topic", "test__writer-a", "v0")
            assert "## Essay" in essay

            store.close_sync()


class TestTrueSkillRanking:
    """Tests for TrueSkill ranking algorithm."""

    async def test_run_with_trueskill(self, minimal_config):
        """Test pipeline runs with TrueSkill ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            minimal_config.ranking.algorithm = "trueskill"
            client = FakeLLMClient(seed=42)
            store = TournamentStore(minimal_config, run_id="test_trueskill")

            pipeline = TournamentPipeline(minimal_config, client, store)
            await pipeline.run()

            # Check ranking outputs exist
            ranking_dir = store.paths.topic_dir("test-topic") / "ranking"
            assert (ranking_dir / "leaderboard.csv").exists()
            assert (ranking_dir / "leaderboard.md").exists()

            # Check leaderboard from DB
            leaderboard = await store.ratings.get_leaderboard("test-topic")
            assert len(leaderboard) > 0

            store.close_sync()
