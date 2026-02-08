from __future__ import annotations

import asyncio

import structlog

from llm_tournament.core.config import TopicConfig, TournamentConfig, WriterConfig
from llm_tournament.prompts import (
    critic_system_prompt,
    critic_user_prompt,
    revision_system_prompt,
    revision_user_prompt,
    writer_system_prompt,
    writer_user_prompt,
)
from llm_tournament.rag import build_rag_context
from llm_tournament.services.llm import LLMClient
from llm_tournament.services.storage import TournamentStore

logger = structlog.get_logger()


class SubmissionService:
    """Handles essay generation, critique, and revision."""

    def __init__(
        self,
        config: TournamentConfig,
        client: LLMClient,
        store: TournamentStore,
        semaphore: asyncio.Semaphore,
    ):
        self.config = config
        self.client = client
        self.store = store
        self._semaphore = semaphore

    async def run_generation_batch(
        self, topic: TopicConfig, writers: list[str | WriterConfig]
    ) -> None:
        """Generate essays for all writers on a topic."""
        tasks = []
        for writer in writers:
            writer_slug = self.config.get_writer_slug(writer)
            writer_model_id = self.config.get_writer_model_id(writer)
            system_prompt = self.config.get_writer_system_prompt(writer)
            use_rag = self.config.writer_uses_rag(writer)
            tasks.append(
                self._generate_one(topic, writer_model_id, writer_slug, system_prompt, use_rag)
            )
        await asyncio.gather(*tasks)

    async def _generate_one(
        self,
        topic: TopicConfig,
        writer_model_id: str,
        writer_slug: str,
        system_prompt: str | None = None,
        use_rag: bool = False,
    ) -> None:
        """Generate essay sections for a single writer."""
        prompts_map = writer_user_prompt(topic)
        sections: dict[str, str] = {}

        # Priority: per-writer override > config default > prompts.yaml
        base_system_prompt = (
            system_prompt or self.config.writer_system_prompt or writer_system_prompt()
        )

        async def generate_section(genre: str, prompt: str) -> None:
            async with self._semaphore:
                logger.debug("generating_section", writer=writer_slug, genre=genre)

                # Build effective system prompt with optional RAG context
                effective_system_prompt = base_system_prompt
                if use_rag and self.config.retriever and topic.rag_queries:
                    rag_query = topic.rag_queries.get(genre)
                    if rag_query:
                        rag_context = build_rag_context(self.config.retriever, rag_query)
                        effective_system_prompt = f"{rag_context}\n\n{base_system_prompt}"

                content = await self.client.complete_prompt(
                    writer_model_id,
                    effective_system_prompt,
                    prompt,
                    self.config.writer_tokens,
                    self.config.writer_temp,
                )
                sections[genre] = content.content

        await asyncio.gather(
            *[generate_section(genre, prompt) for genre, prompt in prompts_map.items()]
        )

        # Stitch sections in order defined in config (prompts_map preserves insertion order of dict)
        full_text_parts = []
        for genre in prompts_map:
            content = sections.get(genre, "")
            full_text_parts.append(f"## {genre}\n\n{content}")

        full_essay = "\n\n".join(full_text_parts)
        await self.store.save_essay(topic.slug, writer_slug, full_essay, "v0")

    async def run_critique_batch(
        self, topic: TopicConfig, writers: list[str | WriterConfig], critics: list[str]
    ) -> None:
        """Generate critiques for all writer-critic combinations."""
        writer_slugs = [self.config.get_writer_slug(w) for w in writers]
        critic_slugs = [self.config.get_slug_model(c) for c in critics]

        tasks = []
        for writer_slug in writer_slugs:
            for critic_id, critic_slug in zip(critics, critic_slugs, strict=True):
                tasks.append(self._critique_one(topic.slug, writer_slug, critic_id, critic_slug))
        await asyncio.gather(*tasks)

    async def _critique_one(
        self, topic_slug: str, writer_slug: str, critic_id: str, critic_slug: str
    ) -> None:
        async with self._semaphore:
            logger.debug("generating_critique", writer=writer_slug, critic=critic_slug)
            essay = await self.store.load_essay(topic_slug, writer_slug, "v0")
            feedback = await self.client.complete_prompt(
                critic_id,
                critic_system_prompt(),
                critic_user_prompt(essay),
                self.config.critic_tokens,
                self.config.critic_temp,
            )
            await self.store.save_feedback(topic_slug, writer_slug, critic_slug, feedback.content)

    async def run_revision_batch(
        self, topic: TopicConfig, writers: list[str | WriterConfig], critics: list[str]
    ) -> None:
        """Generate revisions for all writer-critic combinations."""
        tasks = []
        for writer in writers:
            writer_slug = self.config.get_writer_slug(writer)
            writer_model_id = self.config.get_writer_model_id(writer)
            for critic in critics:
                critic_slug = self.config.get_slug_model(critic)
                tasks.append(
                    self._revise_one(topic.slug, writer_model_id, writer_slug, critic_slug)
                )
        await asyncio.gather(*tasks)

    async def _revise_one(
        self, topic_slug: str, writer_model_id: str, writer_slug: str, critic_slug: str
    ) -> None:
        async with self._semaphore:
            logger.debug("generating_revision", writer=writer_slug, critic=critic_slug)
            original_essay = await self.store.load_essay(topic_slug, writer_slug, "v0")
            feedback = await self.store.load_feedback(topic_slug, writer_slug, critic_slug)
            revised = await self.client.complete_prompt(
                writer_model_id,
                revision_system_prompt(),
                revision_user_prompt(original_essay, feedback),
                self.config.revision_tokens,
                self.config.revision_temp,
            )
            await self.store.save_revision(topic_slug, writer_slug, critic_slug, revised.content)
