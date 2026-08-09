"""Runtime configuration, loaded from the environment."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"
DEFAULT_MAX_REVISIONS = 2
DEFAULT_RESULTS_PER_QUERY = 2


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs to run, resolved once at startup."""

    model: str = DEFAULT_MODEL
    tavily_api_key: str | None = None
    max_revisions: int = DEFAULT_MAX_REVISIONS
    results_per_query: int = DEFAULT_RESULTS_PER_QUERY

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        """Load settings from `.env` / the environment, then apply overrides."""
        load_dotenv()
        defaults = {
            "model": os.environ.get("ESSAYWRITER_MODEL", DEFAULT_MODEL),
            "tavily_api_key": os.environ.get("TAVILY_API_KEY"),
            "max_revisions": int(
                os.environ.get("ESSAYWRITER_MAX_REVISIONS", DEFAULT_MAX_REVISIONS)
            ),
            "results_per_query": int(
                os.environ.get(
                    "ESSAYWRITER_RESULTS_PER_QUERY", DEFAULT_RESULTS_PER_QUERY
                )
            ),
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)  # type: ignore[arg-type]
