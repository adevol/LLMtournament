"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_tournament.core.config import (
    TopicConfig,
    TournamentConfig,
    hash_messages,
    load_config,
    model_slug,
)


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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        assert config.token_caps.writer_tokens == 1200
        assert config.token_caps.critic_tokens == 300


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


class TestModelSlug:
    """Tests for model slug generation."""

    def test_basic_slug(self):
        """Test basic model ID to slug conversion."""
        assert model_slug("openai/gpt-4-turbo") == "openai__gpt-4-turbo"

    def test_slug_with_colon(self):
        """Test model ID with colon."""
        assert model_slug("openai/gpt-4:latest") == "openai__gpt-4_latest"


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
