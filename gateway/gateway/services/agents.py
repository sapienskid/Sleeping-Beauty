"""Agent routing — intent detection and dispatch configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Intent detection (Phase 1: rule-based) ─────────────────────────────────

PASSTHROUGH_PREFIX = "!norag"

REMEMBER_KEYWORDS = ["remember", "note that", "keep in mind", "from now on"]
DRAFT_KEYWORDS = [
    "draft",
    "write me",
    "compose",
    "email to",
    "reply to",
    "blog post",
    "write a",
    "write an",
]


def detect_intent(query: str) -> str:
    """Classify user intent from query text.

    Returns one of: "passthrough", "remember", "draft", "search"
    """
    query_lower = query.lower().strip()

    if query_lower.startswith(PASSTHROUGH_PREFIX):
        return "passthrough"

    if any(kw in query_lower for kw in REMEMBER_KEYWORDS):
        return "remember"

    if any(kw in query_lower for kw in DRAFT_KEYWORDS):
        return "draft"

    return "search"


def strip_norag_prefix(query: str) -> str:
    """Remove the !norag prefix if present."""
    if query.lower().strip().startswith(PASSTHROUGH_PREFIX):
        return query.strip()[len(PASSTHROUGH_PREFIX) :].strip()
    return query


# ─── Agent config ────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    name: str
    system_prompt_template: str | None = None
    collections: list[str] | str = "all"  # "all" or list of names
    memory_types: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.35

    def load_prompt(self, prompts_dir: str = "/app/prompts") -> str | None:
        """Load the system prompt template from file."""
        if self.system_prompt_template is None:
            return None
        path = Path(prompts_dir) / self.system_prompt_template
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        logger.warning("Prompt template not found: %s", path)
        return None


# ─── Default agent configs ───────────────────────────────────────────────────

DEFAULT_AGENTS: dict[str, AgentConfig] = {
    "search": AgentConfig(
        name="search",
        system_prompt_template="search.txt",
        collections="all",
        memory_types=["preference", "fact"],
        confidence_threshold=0.35,
    ),
    "draft": AgentConfig(
        name="draft",
        system_prompt_template="draft.txt",
        collections="all",
        memory_types=["preference", "fact", "writing-style"],
        confidence_threshold=0.25,
    ),
    "remember": AgentConfig(
        name="remember",
        system_prompt_template=None,
        collections=[],
        memory_types=[],
    ),
    "passthrough": AgentConfig(
        name="passthrough",
        system_prompt_template="passthrough.txt",
        collections=[],
        memory_types=["preference", "fact"],
    ),
}


def get_agent(intent: str) -> AgentConfig:
    """Get agent config for an intent."""
    return DEFAULT_AGENTS.get(intent, DEFAULT_AGENTS["search"])
