"""Fakes that let the whole graph run with no model and no network."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from essaywriter.state import Critique, Queries


@dataclass
class FakeResponse:
    """Stands in for a chat model's message; nodes read `.content`."""

    content: str


class FakeModel:
    """Duck-typed chat model returning scripted replies and recording calls."""

    def __init__(
        self,
        text_replies: Sequence[str] = ("text",),
        critiques: Sequence[Critique] = (),
        queries: Sequence[str] = ("q1",),
    ) -> None:
        self._text_replies = list(text_replies)
        self._critiques = list(critiques)
        self._queries = list(queries)
        self.calls: List[str] = []

    def invoke(self, messages) -> FakeResponse:
        self.calls.append("invoke")
        return FakeResponse(_take(self._text_replies))

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)


class _FakeStructured:
    """What `FakeModel.with_structured_output(schema)` hands back."""

    def __init__(self, model: FakeModel, schema) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages):
        if self._schema is Queries:
            self._model.calls.append("queries")
            return Queries(queries=list(self._model._queries))
        self._model.calls.append("critique")
        return _take(self._model._critiques)


class FakeSearch:
    """Search backend that records its queries and returns fixed snippets."""

    def __init__(self, snippets: Sequence[str] = ("snippet",)) -> None:
        self._snippets = list(snippets)
        self.queries: List[str] = []

    def gather(self, queries: Iterable[str]) -> List[str]:
        self.queries.extend(queries)
        return list(self._snippets)


def _take(scripted: list):
    """Pop the next scripted item, repeating the last one forever after."""
    if not scripted:
        raise AssertionError("the fake ran out of scripted responses")
    if len(scripted) == 1:
        return scripted[0]
    return scripted.pop(0)
