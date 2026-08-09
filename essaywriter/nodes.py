"""The node functions of the essay graph.

Each node takes the graph state and returns only the keys it updates.
They are bound to a model and a search backend by `EssayNodes` so the
graph itself stays free of construction details.
"""

from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from essaywriter.config import Settings
from essaywriter.prompts import (
    PLAN_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
)
from essaywriter.research import SearchBackend, TavilySearch
from essaywriter.state import AgentState, Queries


class EssayNodes:
    """Holds the model and search backend the nodes run against."""

    def __init__(self, model: BaseChatModel, search: SearchBackend) -> None:
        self._model = model
        self._search = search

    @classmethod
    def from_settings(cls, settings: Settings) -> "EssayNodes":
        return cls(
            model=init_chat_model(settings.model),
            search=TavilySearch(
                settings.tavily_api_key, settings.results_per_query
            ),
        )

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Turn the topic into a section-by-section outline."""
        response = self._model.invoke(
            [
                SystemMessage(content=PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
        return {"plan": response.content}

    def research_plan(self, state: AgentState) -> Dict[str, Any]:
        """Turn the outline into search queries; collect the results."""
        return {"content": self._research(RESEARCH_PLAN_PROMPT, state["task"])}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Write (or rewrite) the draft from the outline, research and critique."""
        content = "\n\n".join(state.get("content") or [])
        response = self._model.invoke(
            [
                SystemMessage(content=WRITER_PROMPT.format(content=content)),
                HumanMessage(
                    content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
                ),
            ]
        )
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """Critique the draft — grade it and list concrete gaps."""
        response = self._model.invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=state["draft"]),
            ]
        )
        return {"critique": response.content}

    def research_critique(self, state: AgentState) -> Dict[str, Any]:
        """Search for what the critique says is missing."""
        return {
            "content": self._research(RESEARCH_CRITIQUE_PROMPT, state["critique"])
        }

    def _research(self, system_prompt: str, subject: str) -> list[str]:
        """Ask the model for queries about `subject`, then run them."""
        queries = self._model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=subject),
            ]
        )
        return self._search.gather(queries.queries)


def should_continue(state: AgentState) -> str:
    """After a draft: stop at the revision limit, otherwise critique and loop."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
