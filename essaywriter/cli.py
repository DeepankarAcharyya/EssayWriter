"""Command line entry point: take a topic, write an essay."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from langgraph.checkpoint.sqlite import SqliteSaver

from essaywriter.config import Settings
from essaywriter.graph import build_graph


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="essaywriter",
        description="Plan, research, draft and revise an essay on a topic.",
    )
    parser.add_argument("topic", help="What the essay should be about.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the final essay here instead of stdout.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help="How many revision loops to allow (default: from env, else 20).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="Score (1-10) at which the draft is accepted (default: from env, else 8).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Chat model to use, e.g. anthropic:claude-sonnet-4-5.",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=":memory:",
        help="SQLite path for checkpoints so long runs can resume.",
    )
    parser.add_argument(
        "--thread-id",
        default="1",
        help="Checkpoint thread to run under; reuse it to resume a run.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-node progress output.",
    )
    return parser.parse_args(argv)


def recursion_limit(max_revisions: int) -> int:
    """Floor for LangGraph's limit: plan + research_plan + N generates + (N-1) reflects + (N-1) research_critiques, plus headroom."""
    return max(4, 3 * max_revisions + 3)


def _progress(node: str, update: dict, settings: Settings) -> str:
    """One stderr line per node, carrying the grade when there is one."""
    if "score" in update:
        return (
            f"[{node}] score {update['score']}/10 "
            f"(target {settings.quality_threshold})"
        )
    return f"[{node}]"


def write_essay(
    topic: str,
    settings: Settings,
    checkpoint_db: str = ":memory:",
    thread_id: str = "1",
    verbose: bool = True,
) -> str:
    """Run the graph to completion and return the final draft."""
    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, settings=settings)
        initial_state = {
            "task": topic,
            "content": [],
            "revision_number": 1,
            "max_revisions": settings.max_revisions,
            "quality_threshold": settings.quality_threshold,
            "score": 0,
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit(settings.max_revisions),
        }

        draft = ""
        for step in graph.stream(initial_state, config):
            for node, update in step.items():
                if verbose:
                    print(_progress(node, update, settings), file=sys.stderr)
                if "draft" in update:
                    draft = update["draft"]
        return draft


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        settings = Settings.from_env(
            model=args.model,
            max_revisions=args.max_revisions,
            quality_threshold=args.quality_threshold,
        )
        essay = write_essay(
            topic=args.topic,
            settings=settings,
            checkpoint_db=args.checkpoint_db,
            thread_id=args.thread_id,
            verbose=not args.quiet,
        )
    except ValueError as exc:  # missing credentials, bad config
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        args.output.write_text(essay)
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(essay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
