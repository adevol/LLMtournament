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


class APIKeyError(ConfigurationError):
    """Error when API key is missing."""

    def __init__(self) -> None:
        super().__init__(
            "API key required for real API calls",
            "Set OPENROUTER_API_KEY or add api_key to config.yaml.",
        )
