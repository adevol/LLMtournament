"""Custom exceptions for configuration and setup errors."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Base exception for configuration errors with optional suggestions."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"[Configuration Error] {self.message}"
        if self.suggestion:
            msg += f"\n[Suggestion] {self.suggestion}"
        return msg


class MissingFieldError(ConfigurationError):
    """Error when a required configuration field is missing."""

    def __init__(self, field: str, config_path: str) -> None:
        super().__init__(
            f"Missing required field '{field}' in {config_path}",
            "Add the field to your configuration.",
        )


class EmptyModelError(ConfigurationError):
    """Error when a model ID is empty."""

    def __init__(self, role: str, index: int) -> None:
        super().__init__(
            f"Empty {role} model ID at index {index}",
            "Provide a valid model ID (e.g., 'openai/gpt-4').",
        )


class APIKeyError(ConfigurationError):
    """Error when API key is missing."""

    def __init__(self) -> None:
        super().__init__(
            "API key required for real API calls",
            "Set OPENROUTER_API_KEY or add api_key to config.yaml.",
        )


class ValidationError(ConfigurationError):
    """Error when configuration validation fails."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            f"Invalid value for '{field}'",
            f"{reason}",
        )
