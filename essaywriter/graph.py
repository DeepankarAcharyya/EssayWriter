"""Wiring of the plan → research → write → critique loop."""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from essaywriter.config import Settings
from essaywriter.nodes import EssayNodes, is_good_enough, should_continue
from essaywriter.state import AgentState


def build_graph(
    nodes: Optional[EssayNodes] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    settings: Optional[Settings] = None,
):
    """Build and compile the essay graph.

    Pass `nodes` to inject a model/search backend (tests do this); otherwise
    they are constructed from `settings`, defaulting to the environment.
    """
    if nodes is None:
        nodes = EssayNodes.from_settings(settings or Settings.from_env())

    graph = StateGraph(AgentState)
    graph.add_node("plan", nodes.plan)
    graph.add_node("research_plan", nodes.research_plan)
    graph.add_node("generate", nodes.generate)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("research_critique", nodes.research_critique)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research_plan")
    graph.add_edge("research_plan", "generate")
    graph.add_conditional_edges(
        "generate",
        should_continue,
        {END: END, "reflect": "reflect"},
    )
    graph.add_conditional_edges(
        "reflect",
        is_good_enough,
        {END: END, "research_critique": "research_critique"},
    )
    graph.add_edge("research_critique", "generate")

    return graph.compile(checkpointer=checkpointer)
