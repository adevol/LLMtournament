from __future__ import annotations

import asyncio

import structlog

from llm_tournament.core.config import TournamentConfig, model_slug
from llm_tournament.prompts import (
    critic_system_prompt,
    critic_user_prompt,
    revision_system_prompt,
    revision_user_prompt,
    writer_system_prompt,
    writer_user_prompt,
)
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

    async def run_generation_batch(self, topic, writers: list[str]) -> None:
        writer_slugs = [model_slug(w, self.config.slug_max_length) for w in writers]

        tasks = []
        for writer_id, writer_slug in zip(writers, writer_slugs, strict=True):
            tasks.append(self._generate_one(topic, writer_id, writer_slug))
        await asyncio.gather(*tasks)

    async def _generate_one(self, topic, writer_id: str, writer_slug: str) -> None:
        prompts_map = writer_user_prompt(topic)
        sections: dict[str, str] = {}

        async def generate_section(genre: str, prompt: str) -> None:
            async with self._semaphore:
                logger.debug("generating_section", writer=writer_slug, genre=genre)
                messages = [
                    {"role": "system", "content": writer_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
                content = await self.client.complete(
                    writer_id,
                    messages,
                    self.config.token_caps.writer_tokens,
                    self.config.temperatures.writer,
                )
                sections[genre] = content

        await asyncio.gather(
            *[generate_section(genre, prompt) for genre, prompt in prompts_map.items()]
        )

        # Stitch sections in order defined in config (prompts_map preserves insertion order of dict)
        full_text_parts = []
        for genre in prompts_map:
            content = sections.get(genre, "")
            # Add header
            full_text_parts.append(f"## {genre}\n\n{content}")

        full_essay = "\n\n".join(full_text_parts)
        await self.store.files.save_essay(topic.slug, writer_slug, full_essay, "v0")

    async def run_critique_batch(self, topic, writers: list[str], critics: list[str]) -> None:
        writer_slugs = [model_slug(w, self.config.slug_max_length) for w in writers]
        critic_slugs = [model_slug(c, self.config.slug_max_length) for c in critics]

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
            essay = await self.store.files.load_essay(topic_slug, writer_slug, "v0")
            messages = [
                {"role": "system", "content": critic_system_prompt()},
                {"role": "user", "content": critic_user_prompt(essay)},
            ]
            feedback = await self.client.complete(
                critic_id,
                messages,
                self.config.token_caps.critic_tokens,
                self.config.temperatures.critic,
            )
            await self.store.files.save_feedback(topic_slug, writer_slug, critic_slug, feedback)

    async def run_revision_batch(self, topic, writers: list[str], critics: list[str]) -> None:
        writer_slugs = [model_slug(w, self.config.slug_max_length) for w in writers]
        critic_slugs = [model_slug(c, self.config.slug_max_length) for c in critics]

        tasks = []
        for writer_id, writer_slug in zip(writers, writer_slugs, strict=True):
            for critic_slug in critic_slugs:
                tasks.append(self._revise_one(topic.slug, writer_id, writer_slug, critic_slug))
        await asyncio.gather(*tasks)

    async def _revise_one(
        self, topic_slug: str, writer_id: str, writer_slug: str, critic_slug: str
    ) -> None:
        async with self._semaphore:
            logger.debug("generating_revision", writer=writer_slug, critic=critic_slug)
            original_essay = await self.store.files.load_essay(topic_slug, writer_slug, "v0")
            feedback = await self.store.files.load_feedback(topic_slug, writer_slug, critic_slug)
            messages = [
                {"role": "system", "content": revision_system_prompt()},
                {
                    "role": "user",
                    "content": revision_user_prompt(original_essay, feedback),
                },
            ]
            revised = await self.client.complete(
                writer_id,
                messages,
                self.config.token_caps.revision_tokens,
                self.config.temperatures.revision,
            )
            await self.store.files.save_revision(topic_slug, writer_slug, critic_slug, revised)
