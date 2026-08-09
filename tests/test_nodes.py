"""Node behaviour, exercised against fakes."""

from langgraph.graph import END

from essaywriter.nodes import EssayNodes, is_good_enough, should_continue
from essaywriter.state import Critique

from tests.conftest import FakeModel, FakeSearch


def test_reflect_splits_the_grade_from_the_feedback():
    model = FakeModel(critiques=[Critique(score=6, feedback="Add sources.")])
    nodes = EssayNodes(model=model, search=FakeSearch())

    update = nodes.reflect({"draft": "a draft"})

    assert update == {"critique": "Add sources.", "score": 6}
    assert model.calls == ["critique"]


def test_should_continue_critiques_while_under_the_cap():
    assert should_continue({"revision_number": 2, "max_revisions": 20}) == "reflect"


def test_should_continue_stops_past_the_cap():
    assert should_continue({"revision_number": 21, "max_revisions": 20}) == END


def test_is_good_enough_accepts_a_score_at_the_threshold():
    assert is_good_enough({"score": 8, "quality_threshold": 8}) == END


def test_is_good_enough_researches_below_the_threshold():
    state = {"score": 7, "quality_threshold": 8}
    assert is_good_enough(state) == "research_critique"
