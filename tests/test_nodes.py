"""Node behaviour, exercised against fakes."""

from essaywriter.nodes import EssayNodes
from essaywriter.state import Critique

from tests.conftest import FakeModel, FakeSearch


def test_reflect_splits_the_grade_from_the_feedback():
    model = FakeModel(critiques=[Critique(score=6, feedback="Add sources.")])
    nodes = EssayNodes(model=model, search=FakeSearch())

    update = nodes.reflect({"draft": "a draft"})

    assert update == {"critique": "Add sources.", "score": 6}
    assert model.calls == ["critique"]
