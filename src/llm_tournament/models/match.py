import uuid
from datetime import UTC, datetime

from sqlalchemy import Column
from sqlmodel import JSON, Field, SQLModel


class Match(SQLModel, table=True):
    """A single pairwise match between two essays."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    topic_slug: str = Field(index=True)
    essay_a_id: str
    essay_b_id: str
    winner: str
    confidence: float
    reasons: list[str] = Field(sa_column=Column(JSON))
    winner_edge: str
    primary_judge: str
    audit_judges: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    final_decision: str = "primary"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
