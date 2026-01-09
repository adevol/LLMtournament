"""LLM API call tracking model for cost intelligence."""

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class LLMCall(SQLModel, table=True):
    """Tracks individual LLM API calls with cost data."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    model: str = Field(index=True)
    role: str = Field(index=True)  # "writer", "judge", "critic", "analysis"
    topic_slug: str | None = Field(default=None, index=True)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
