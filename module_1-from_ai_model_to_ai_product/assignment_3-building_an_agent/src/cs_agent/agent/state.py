"""Shared graph state shape.

Kept deliberately small. ``messages`` uses LangGraph's ``add_messages`` reducer
so that node updates *append* messages instead of replacing the list.

Fields populated in Task 1:
- messages, route, iterations, user_id

Fields reserved for later tasks (declared upfront so adding them later is a
no-op for existing nodes that just don't touch them):
- pending_query  — Bonus B: a recommended-but-not-yet-executed query string.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Route = Literal["structured", "unstructured", "out_of_scope"]


class GraphState(TypedDict, total=False):
    """LangGraph state for the customer-service data-analyst agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history. ``add_messages`` reducer appends, never overwrites."""

    route: Route | None
    """Output of ``router_node``. Drives the conditional edge after the router."""

    iterations: int
    """Number of times ``agent_node`` has run for the current turn. Reset per turn."""

    user_id: str
    """Stable identifier for the human user (separate from thread_id). Used
    by the user-profile module in Task 2b."""

    pending_query: str | None
    """Reserved for Bonus B (Query Recommender). Holds a suggested query the
    user has not yet confirmed. Unused in Task 1."""
