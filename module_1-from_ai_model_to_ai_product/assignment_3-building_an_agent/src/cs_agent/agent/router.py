"""Query router: classifies the user's latest message into one of three routes.

Implementation detail: we use ``ChatOpenAI.with_structured_output`` against a
small Llama 3.1 8B model. Structured output gives us a typed ``RouterDecision``
back instead of free-form JSON we'd have to parse defensively. If the model
ever fails to comply with the schema we fall back to ``out_of_scope`` so the
agent is never silently routed somewhere it shouldn't be.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from cs_agent.agent.prompts import ROUTER_SYSTEM
from cs_agent.agent.state import GraphState, Route
from cs_agent.llm import get_router_llm

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Structured output schema returned by the router LLM."""

    route: Route = Field(
        ...,
        description=(
            "One of 'structured' (concrete data lookup), 'unstructured' "
            "(summary/synthesis), or 'out_of_scope' (unrelated to the dataset)."
        ),
    )
    reason: str = Field(
        ...,
        description="One short sentence justifying the chosen route.",
    )


def classify(message: str) -> RouterDecision:
    """Classify a single user message. Exposed for tests and the smoke script."""
    structured = get_router_llm().with_structured_output(RouterDecision)
    raw = structured.invoke([SystemMessage(ROUTER_SYSTEM), HumanMessage(message)])
    return RouterDecision.model_validate(raw)


def router_node(state: GraphState) -> dict:
    """Router node: read the latest human message, return the chosen route.

    Returns a partial state update that only sets ``route`` — the messages
    list is left untouched so the agent loop sees the original user query.
    """
    messages = state.get("messages") or []
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        logger.warning("router_node called with no HumanMessage in state; declining.")
        return {"route": "out_of_scope"}

    try:
        decision = classify(str(last_human.content))
        logger.debug("router decision: %s — %s", decision.route, decision.reason)
        return {"route": decision.route}
    except Exception as exc:  # noqa: BLE001 — fall back safely on any LLM/parsing error
        logger.warning("router LLM failed (%s); defaulting to out_of_scope", exc)
        return {"route": "out_of_scope"}


def route_from_router(state: GraphState) -> Literal["agent", "decline"]:
    """Conditional edge function: maps the route to the next node label.

    'structured' / 'unstructured' both go to the agent loop; only out-of-scope
    diverts to the decline node. The agent's own system prompt distinguishes
    structured vs unstructured behaviour from there.
    """
    return "decline" if state.get("route") == "out_of_scope" else "agent"
