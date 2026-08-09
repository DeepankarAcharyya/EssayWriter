"""Graph state and the structured shapes the model is asked to produce."""

import operator
from typing import Annotated, List, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State threaded through every node of the essay graph.

    `content` accumulates: nodes return only the snippets they found and
    LangGraph appends them, so no node mutates the list in place.
    """

    task: str
    plan: str
    draft: str
    critique: str
    content: Annotated[List[str], operator.add]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    """A batch of search queries produced by a research node."""

    queries: List[str] = Field(
        description="A list of search queries to gather relevant information."
    )


class Critique(BaseModel):
    """A graded critique of a draft."""

    score: int = Field(
        ge=1,
        le=10,
        description=(
            "Quality of the draft on a 1-10 scale, where 8 or above is "
            "publishable."
        ),
    )
    feedback: str = Field(
        description="Concrete, actionable revisions the writer should make."
    )
