import uuid

from sqlmodel import Field, SQLModel


class Rating(SQLModel, table=True):
    """Rating for a candidate (writer + optional critic) on a topic."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    topic_slug: str = Field(index=True)
    candidate_id: str = Field(index=True)
    rating: float
    mu: float | None = None
    sigma: float | None = None
    matches: int = 0
    wins: int = 0
    losses: int = 0
    writer_slug: str
    critic_slug: str | None = None
