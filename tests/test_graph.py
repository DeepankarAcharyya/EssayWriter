"""End-to-end loop behaviour, driven entirely by fakes."""

import pytest

from essaywriter.cli import recursion_limit
from essaywriter.graph import build_graph
from essaywriter.nodes import EssayNodes
from essaywriter.state import Critique

from tests.conftest import FakeModel, FakeSearch


def run(model, search, *, max_revisions=3, quality_threshold=8):
    """Run the graph to completion and return the final state."""
    graph = build_graph(nodes=EssayNodes(model=model, search=search))
    return graph.invoke(
        {
            "task": "a topic",
            "content": [],
            "revision_number": 1,
            "max_revisions": max_revisions,
            "quality_threshold": quality_threshold,
            "score": 0,
        }
    )


def test_a_good_first_draft_ends_the_run():
    model = FakeModel(
        text_replies=["an outline", "draft one"],
        critiques=[Critique(score=9, feedback="Ship it.")],
    )
    search = FakeSearch()

    final = run(model, search)

    assert final["draft"] == "draft one"
    assert final["score"] == 9
    assert search.queries == ["q1"]  # only research_plan searched


def test_a_weak_draft_loops_until_it_clears_the_bar():
    model = FakeModel(
        text_replies=["an outline", "draft one", "draft two"],
        critiques=[
            Critique(score=5, feedback="Thin on evidence."),
            Critique(score=8, feedback="Good enough."),
        ],
    )
    search = FakeSearch()

    final = run(model, search)

    assert final["draft"] == "draft two"
    assert final["score"] == 8
    assert final["critique"] == "Good enough."
    assert len(search.queries) == 2  # the plan, then the critique


def test_the_cap_stops_a_draft_that_never_improves():
    model = FakeModel(
        text_replies=["an outline", "a draft"],
        critiques=[Critique(score=4, feedback="Still weak.")],
    )

    final = run(model, FakeSearch(), max_revisions=3)

    assert final["revision_number"] == 4  # three drafts written
    assert final["score"] == 4


@pytest.mark.parametrize(
    "max_revisions, expected_revision_number",
    [
        (0, 2),  # should_continue sends the first draft straight to END
        (1, 2),  # one generate still overshoots the cap immediately after
        (20, 21),  # 20 generates, 19 reflects, 19 research_critiques
    ],
)
def test_a_full_run_at_the_cap_fits_inside_the_recursion_limit(
    max_revisions, expected_revision_number
):
    model = FakeModel(
        text_replies=["outline", "draft"],
        critiques=[Critique(score=4, feedback="weak")],
    )
    graph = build_graph(nodes=EssayNodes(model=model, search=FakeSearch()))
    final = graph.invoke(
        {
            "task": "t",
            "content": [],
            "revision_number": 1,
            "max_revisions": max_revisions,
            "quality_threshold": 8,
            "score": 0,
        },
        {"recursion_limit": recursion_limit(max_revisions)},
    )
    assert final["revision_number"] == expected_revision_number
