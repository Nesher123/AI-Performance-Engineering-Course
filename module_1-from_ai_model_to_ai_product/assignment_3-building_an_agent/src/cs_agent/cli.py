"""Interactive CLI for the customer-service data-analyst agent.

Run with::

    uv run cs-agent --session demo --user ofir
    # or
    uv run python -m cs_agent.cli --session demo --user ofir

The REPL prints every reasoning step (router decision, tool calls,
observations, fallbacks, final answer) in a colour-coded ``rich`` trace, so the
grader can see *how* the agent arrived at its answer — not just the answer.

Within a single CLI session, conversation messages accumulate in memory so
follow-up turns work naturally ("Show me 3 examples of REFUND" → "show me 3
more"). Persistence across restarts is added in Task 2a via SqliteSaver.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from cs_agent.agent.graph import build_graph

# Suppress library log spam from inside agent_node ("loop detected — short-circuiting")
# unless the user explicitly raises the level.
logging.basicConfig(level=logging.WARNING, format="%(message)s")

EXIT_WORDS = {"exit", "quit", ":q"}
TOOL_RESULT_PREVIEW_CHARS = 240


def _format_args(args: dict[str, Any]) -> str:
    """Render a tool-call's argument dict compactly for the trace."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if v is None:
            continue
        s = repr(v) if isinstance(v, str) else str(v)
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _render_router(update: dict[str, Any], console: Console) -> None:
    route = update.get("route")
    console.print(f"  [dim]router → [bold]{route}[/][/]")


def _render_agent(update: dict[str, Any], console: Console) -> None:
    for msg in update.get("messages", []):
        if not isinstance(msg, AIMessage):
            continue
        if msg.tool_calls:
            for tc in msg.tool_calls:
                console.print(f"  [yellow]→ {tc['name']}([dim]{_format_args(tc.get('args') or {})}[/])[/]")
        elif msg.content:
            console.print(
                Panel(
                    Text(str(msg.content)),
                    title="agent",
                    border_style="green",
                    padding=(0, 1),
                )
            )


def _render_tools(update: dict[str, Any], console: Console) -> None:
    for msg in update.get("messages", []):
        if not isinstance(msg, ToolMessage):
            continue
        preview = _truncate(str(msg.content))
        console.print(f"  [green]← {msg.name}[/] [dim]→[/] {preview}")


def _render_terminal(node: str, update: dict[str, Any], console: Console) -> None:
    """Render decline_node / fallback_node output."""
    border_style = "red" if node == "decline" else "yellow"
    title = "out-of-scope decline" if node == "decline" else "fallback (loop / max-iter)"
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage) and msg.content:
            console.print(
                Panel(
                    Text(str(msg.content)),
                    title=title,
                    border_style=border_style,
                    padding=(0, 1),
                )
            )


def _render_chunk(
    chunk: dict[str, dict[str, Any]],
    console: Console,
    messages: list[BaseMessage],
) -> None:
    """Print a single LangGraph stream update and accumulate its messages."""
    for node_name, update in chunk.items():
        for msg in update.get("messages", []) or []:
            messages.append(msg)
        if node_name == "router":
            _render_router(update, console)
        elif node_name == "agent":
            _render_agent(update, console)
        elif node_name == "tools":
            _render_tools(update, console)
        elif node_name in {"decline", "fallback"}:
            _render_terminal(node_name, update, console)


def _print_banner(console: Console, session: str, user: str) -> None:
    body = (
        "[bold]Customer Service Data Analyst[/] — Bitext dataset agent\n"
        f"session: [cyan]{session}[/]   user: [cyan]{user}[/]\n"
        f"type [yellow]{', '.join(sorted(EXIT_WORDS))}[/] (or Ctrl-C / Ctrl-D) to quit"
    )
    console.print(Panel.fit(body, title="cs-agent", border_style="blue"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cs-agent",
        description="Interactive ReAct agent for the Bitext customer-support dataset.",
    )
    p.add_argument(
        "--session",
        default="default",
        help="Session id. In Task 2a this becomes the LangGraph thread_id "
        "for persistent memory across restarts. Default: 'default'.",
    )
    p.add_argument(
        "--user",
        default="anon",
        help="User id. In Task 2b this keys the per-user profile. Default: 'anon'.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show INFO-level logs (router fallbacks, loop-detection, etc.).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    console = Console()
    graph = build_graph()

    _print_banner(console, args.session, args.user)

    messages: list[BaseMessage] = []

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]you[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]")
            return 0

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            console.print("[dim]bye[/]")
            return 0

        messages.append(HumanMessage(question))
        initial: dict[str, Any] = {
            "messages": messages,
            "iterations": 0,
            "user_id": args.user,
            "route": None,
        }

        try:
            for chunk in graph.stream(initial, stream_mode="updates"):
                _render_chunk(chunk, console, messages)
        except Exception as exc:  # noqa: BLE001 — interactive REPL: never crash on a single bad turn
            console.print(f"[red]error during turn:[/] {exc!r}")
            # Keep messages list consistent: drop the human message we appended
            # so the next turn doesn't replay the failed one.
            if messages and messages[-1] is initial["messages"][-1]:
                messages.pop()


if __name__ == "__main__":
    sys.exit(main())
