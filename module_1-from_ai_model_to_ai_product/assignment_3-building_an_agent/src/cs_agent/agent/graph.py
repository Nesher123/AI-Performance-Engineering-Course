"""Compile the full LangGraph StateGraph for the customer-service agent.

Topology:

    START
      |
      v
    router  ── out_of_scope ─►  decline ──► END
      |
      ▼
    agent  ◄────────────────┐
      | tool_calls?         │
      ├── yes ──► tools ────┘
      └── no ──► END

A ``checkpointer`` may be passed through; Task 1 leaves it as ``None`` (no
persistence yet) and Task 2a will pass a ``SqliteSaver``.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cs_agent.agent.nodes import (
    agent_node,
    decline_node,
    fallback_node,
    should_continue,
)
from cs_agent.agent.router import route_from_router, router_node
from cs_agent.agent.state import GraphState
from cs_agent.tools.registry import DATA_TOOLS


def build_graph(checkpointer: Any | None = None):
    """Compile the agent graph. Pass a checkpointer in Task 2a; ``None`` in Task 1."""
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(DATA_TOOLS))
    builder.add_node("decline", decline_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_from_router,
        {"agent": "agent", "decline": "decline"},
    )
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "fallback": "fallback", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("decline", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
