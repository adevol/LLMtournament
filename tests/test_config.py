"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pydantic
import pytest
import yaml

from llm_tournament.core.config import (
    TopicConfig,
    TournamentConfig,
    WriterConfig,
    hash_messages,
    load_config,
)
from llm_tournament.core.slug import SlugGenerator


class TestTopicConfig:
    """Tests for TopicConfig."""

    def test_slug_generation(self):
        """Test slug is generated correctly from title."""
        topic = TopicConfig(
            title="AI Ethics & Safety",
            prompts={"Essay": "Write an essay about AI ethics"},
        )
        assert topic.slug == "ai-ethics-safety"

    def test_slug_handles_special_chars(self):
        """Test slug handles special characters."""
        topic = TopicConfig(
            title="What's the Deal with AI?!",
            prompts={"Essay": "test"},
        )
        assert topic.slug == "what-s-the-deal-with-ai"


class TestTournamentConfig:
    """Tests for TournamentConfig."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = TournamentConfig(
            writers=["model/a"],
            critics=["model/b"],
            judges=["model/c"],
            topics=[
                TopicConfig(
                    title="Test",
                    prompts={"Essay": "test"},
                )
            ],
        )
        assert len(config.writers) == 1
        assert config.simple_mode is False
        assert config.seed == 42

    def test_empty_model_list_fails(self):
        """Test that empty model lists are rejected."""
        with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
            TournamentConfig(
                writers=[],
                critics=["model/b"],
                judges=["model/c"],
                topics=[
                    TopicConfig(
                        title="Test",
                        prompts={"Essay": "test"},
                    )
                ],
            )

    def test_empty_model_id_fails(self):
        """Test that empty model IDs are rejected."""
        with pytest.raises(pydantic.ValidationError, match="cannot be empty"):
            TournamentConfig(
                writers=["model/a", ""],
                critics=["model/b"],
                judges=["model/c"],
                topics=[
                    TopicConfig(
                        title="Test",
                        prompts={"Essay": "test"},
                    )
                ],
            )

    def test_default_token_caps(self):
        """Test default token caps are set."""
        config = TournamentConfig(
            writers=["model/a"],
            critics=["model/b"],
            judges=["model/c"],
            topics=[
                TopicConfig(
                    title="Test",
                    prompts={"Essay": "test"},
                )
            ],
        )
        assert config.writer_tokens == 1200
        assert config.critic_tokens == 300


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML config."""
        config_data = {
            "writers": ["openai/gpt-4"],
            "critics": ["openai/gpt-4"],
            "judges": ["openai/gpt-4"],
            "topics": [
                {
                    "title": "Test Topic",
                    "prompts": {"Essay": "Write an essay"},
                }
            ],
            "seed": 123,
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config_data, f)
            f.flush()

            config = load_config(f.name)
            assert config.seed == 123
            assert len(config.topics) == 1

        Path(f.name).unlink()

    def test_load_missing_file_fails(self):
        """Test loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestSafeId:
    """Tests for safe_id helper."""

    def test_basic_slug(self):
        """Test basic model ID to slug conversion."""
        assert SlugGenerator.safe_id("openai/gpt-4-turbo") == "openai__gpt-4-turbo"

    def test_slug_with_colon(self):
        """Test model ID with colon."""
        assert SlugGenerator.safe_id("openai/gpt-4:latest") == "openai__gpt-4_latest"


class TestHashMessages:
    """Tests for message hashing."""

    def test_same_input_same_hash(self):
        """Test same input produces same hash."""
        messages = [{"role": "user", "content": "hello"}]
        params = {"model": "test", "temperature": 0.7}

        hash1 = hash_messages(messages, params)
        hash2 = hash_messages(messages, params)

        assert hash1 == hash2

    def test_different_input_different_hash(self):
        """Test different input produces different hash."""
        messages1 = [{"role": "user", "content": "hello"}]
        messages2 = [{"role": "user", "content": "world"}]
        params = {"model": "test", "temperature": 0.7}

        hash1 = hash_messages(messages1, params)
        hash2 = hash_messages(messages2, params)

        assert hash1 != hash2


class TestWriterConfig:
    """Tests for WriterConfig."""

    def test_slug_from_model_id(self):
        """Test slug generation from model_id only."""
        writer = WriterConfig(model_id="openai/gpt-4")
        assert writer.get_slug() == "openai__gpt-4"

    def test_slug_with_custom_name(self):
        """Test slug uses custom name when provided."""
        writer = WriterConfig(
            model_id="openai/gpt-4",
            name="my-custom-writer",
        )
        assert writer.get_slug() == "my-custom-writer"

    def test_slug_with_system_prompt_hash(self):
        """Test slug includes hash when system_prompt is set."""
        writer = WriterConfig(
            model_id="openai/gpt-4",
            system_prompt="You are a pirate.",
        )
        slug = writer.get_slug()
        # Should have model_id + 6-char hash suffix
        assert slug.startswith("openai__gpt-4_")
        assert len(slug) == len("openai__gpt-4_") + 6

    def test_different_prompts_different_slugs(self):
        """Test different system prompts generate different slugs."""
        writer1 = WriterConfig(
            model_id="openai/gpt-4",
            system_prompt="You are a pirate.",
        )
        writer2 = WriterConfig(
            model_id="openai/gpt-4",
            system_prompt="You are a ninja.",
        )
        assert writer1.get_slug() != writer2.get_slug()

    def test_empty_model_id_validation(self):
        """Test that empty model_id in WriterConfig is rejected."""
        with pytest.raises(pydantic.ValidationError, match="cannot be empty"):
            TournamentConfig(
                writers=[WriterConfig(model_id="")],
                critics=["model/b"],
                judges=["model/c"],
                topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
            )


class TestTournamentConfigWriterHelpers:
    """Tests for TournamentConfig writer helper methods."""

    def test_get_writer_slug_string(self):
        """Test get_writer_slug with string input."""
        config = TournamentConfig(
            writers=["openai/gpt-4"],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_slug("openai/gpt-4") == "openai__gpt-4"

    def test_get_writer_slug_writer_config(self):
        """Test get_writer_slug with WriterConfig input."""
        writer = WriterConfig(model_id="openai/gpt-4", name="custom")
        config = TournamentConfig(
            writers=[writer],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_slug(writer) == "custom"

    def test_get_writer_model_id_string(self):
        """Test get_writer_model_id with string input."""
        config = TournamentConfig(
            writers=["openai/gpt-4"],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_model_id("openai/gpt-4") == "openai/gpt-4"

    def test_get_writer_model_id_writer_config(self):
        """Test get_writer_model_id with WriterConfig input."""
        writer = WriterConfig(model_id="openai/gpt-4")
        config = TournamentConfig(
            writers=[writer],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_model_id(writer) == "openai/gpt-4"

    def test_get_writer_system_prompt_string(self):
        """Test get_writer_system_prompt returns None for string input."""
        config = TournamentConfig(
            writers=["openai/gpt-4"],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_system_prompt("openai/gpt-4") is None

    def test_get_writer_system_prompt_writer_config(self):
        """Test get_writer_system_prompt returns prompt from WriterConfig."""
        writer = WriterConfig(
            model_id="openai/gpt-4",
            system_prompt="You are a pirate.",
        )
        config = TournamentConfig(
            writers=[writer],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert config.get_writer_system_prompt(writer) == "You are a pirate."

    def test_mixed_writers_list(self):
        """Test TournamentConfig accepts mixed string and WriterConfig list."""
        writer_config = WriterConfig(
            model_id="openai/gpt-4",
            system_prompt="Custom prompt",
        )
        config = TournamentConfig(
            writers=["anthropic/claude-3", writer_config],
            critics=["model/b"],
            judges=["model/c"],
            topics=[TopicConfig(title="Test", prompts={"Essay": "test"})],
        )
        assert len(config.writers) == 2
        assert isinstance(config.writers[0], str)
        assert isinstance(config.writers[1], WriterConfig)
