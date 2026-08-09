"""The structured shapes the model is asked to produce."""

import pytest
from pydantic import ValidationError

from essaywriter.state import Critique


def test_critique_carries_a_score_and_feedback():
    critique = Critique(score=7, feedback="Add sources to paragraph three.")
    assert critique.score == 7
    assert critique.feedback == "Add sources to paragraph three."


@pytest.mark.parametrize("score", [0, 11, -3])
def test_critique_rejects_scores_off_the_scale(score):
    with pytest.raises(ValidationError):
        Critique(score=score, feedback="...")
