"""Query router: classifies the user's latest message into one of three routes.

Implementation detail: we use ``ChatOpenAI.with_structured_output`` against a
small classifier model (Qwen3-32B by default). Structured output gives us a
typed ``RouterDecision`` back instead of free-form JSON to parse defensively.

Resilience: if the primary router model fails (timeout, 404, schema violation),
we transparently retry with the larger agent model (Llama 3.3 70B). If even
that fails the final last-resort fallback is route='structured' — sending the
question into the agent loop where the system prompt's scoped-fallback
paragraph keeps the response honest. Defaulting to 'structured' rather than
'out_of_scope' avoids unfairly declining legitimate questions during transient
Nebius outages.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from cs_agent.agent.prompts import ROUTER_SYSTEM
from cs_agent.agent.state import GraphState, Route
from cs_agent.llm import get_agent_llm, get_router_llm

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


def _classify_with(llm: BaseChatModel, message: str) -> RouterDecision:
    structured = llm.with_structured_output(RouterDecision)
    raw = structured.invoke([SystemMessage(ROUTER_SYSTEM), HumanMessage(message)])
    return RouterDecision.model_validate(raw)


def classify(message: str) -> RouterDecision:
    """Classify a single user message. Exposed for tests and the smoke script.

    Tries the primary router model first. On any failure (network, timeout,
    schema), retries once with the larger agent model. If both fail, raises
    the second exception so callers can decide a last-resort fallback.
    """
    try:
        return _classify_with(get_router_llm(), message)
    except Exception as primary_exc:  # noqa: BLE001 — see below
        logger.warning("router primary failed (%s); retrying with agent model", primary_exc)
        return _classify_with(get_agent_llm(), message)


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
        logger.warning("router_node called with no HumanMessage in state; defaulting structured.")
        return {"route": "structured"}

    try:
        decision = classify(str(last_human.content))
        logger.debug("router decision: %s — %s", decision.route, decision.reason)
        return {"route": decision.route}
    except Exception as exc:  # noqa: BLE001 — fall back safely on any LLM/parsing error
        # Both primary and fallback models failed. Default to 'structured' so
        # legitimate questions aren't declined during a transient Nebius outage.
        # The agent's system prompt still keeps it from inventing answers.
        logger.warning("router fully failed (%s); defaulting to 'structured'", exc)
        return {"route": "structured"}


def route_from_router(state: GraphState) -> Literal["agent", "decline"]:
    """Conditional edge function: maps the route to the next node label.

    'structured' / 'unstructured' both go to the agent loop; only out-of-scope
    diverts to the decline node. The agent's own system prompt distinguishes
    structured vs unstructured behaviour from there.
    """
    return "decline" if state.get("route") == "out_of_scope" else "agent"
