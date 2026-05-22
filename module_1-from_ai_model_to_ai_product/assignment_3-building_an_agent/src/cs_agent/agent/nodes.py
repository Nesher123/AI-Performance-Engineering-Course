"""Graph nodes other than the router itself.

- ``agent_node``: one step of the ReAct loop. Calls the agent LLM with all
  tools bound. Honours ``MAX_ITERATIONS`` as a graceful fallback.
- ``decline_node``: terminal node for out-of-scope queries.
- ``should_continue``: conditional-edge function deciding whether the agent
  asked for a tool call (loop back to the tool node) or is done (END).
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from cs_agent.agent.prompts import build_agent_system
from cs_agent.agent.state import GraphState
from cs_agent.config import MAX_ITERATIONS
from cs_agent.llm import get_agent_llm
from cs_agent.tools.registry import DATA_TOOLS

logger = logging.getLogger(__name__)

DECLINE_MESSAGE = (
    "That question is outside the scope of this customer-service data agent. "
    "I can help with the Bitext customer-support dataset — categories, intents, "
    "examples, distributions, or summaries. Try one of those?"
)

MAX_ITER_MESSAGE = (
    "I couldn't reach a confident answer within my reasoning budget "
    "(max {max_iter} steps). Could you rephrase the question or break it into "
    "smaller pieces?"
)

LOOP_FALLBACK_TEMPLATE = "Based on the {tool_name} tool result:\n\n{content}"


def _is_loop(state: GraphState) -> tuple[bool, ToolMessage | None]:
    """Detect a tool-call loop: the last AIMessage emits exactly the same tool
    call (name + args) as one of the previous AIMessages in this turn.

    Returns (is_loop, last_tool_message).
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return False, None

    # Build a signature for the latest tool calls
    def sig(ai: AIMessage) -> tuple:
        return tuple((tc["name"], tuple(sorted((tc.get("args") or {}).items()))) for tc in ai.tool_calls)

    latest_sig = sig(last)
    for prev in messages[:-1]:
        if isinstance(prev, AIMessage) and prev.tool_calls and sig(prev) == latest_sig:
            # Find the most recent ToolMessage to surface as the answer
            last_tool_msg = next(
                (m for m in reversed(messages) if isinstance(m, ToolMessage)),
                None,
            )
            return True, last_tool_msg
    return False, None


def agent_node(state: GraphState) -> dict:
    """One step of the ReAct loop.

    Each visit:
    1. Checks the iteration budget; emits a graceful fallback if exceeded.
    2. Builds a route-aware system prompt (structured/unstructured steering).
    3. Invokes the agent LLM with all tools bound. The LLM either emits a
       final answer (no tool calls) or one or more ``tool_calls`` that the
       graph will execute via ``ToolNode`` and then route back here.
    """
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        logger.info("max_iterations (%d) reached; emitting fallback", MAX_ITERATIONS)
        return {
            "messages": [AIMessage(MAX_ITER_MESSAGE.format(max_iter=MAX_ITERATIONS))],
            "iterations": iterations + 1,
        }

    route = state.get("route")
    system_prompt = build_agent_system(route=route)

    llm_with_tools = get_agent_llm().bind_tools(DATA_TOOLS)
    messages = state.get("messages") or []
    response = llm_with_tools.invoke([SystemMessage(system_prompt), *messages])

    return {"messages": [response], "iterations": iterations + 1}


def decline_node(state: GraphState) -> dict:
    """Terminal node for out-of-scope queries — never calls a tool, never an LLM."""
    return {"messages": [AIMessage(DECLINE_MESSAGE)]}


def should_continue(state: GraphState) -> Literal["tools", "fallback", "end"]:
    """Conditional edge after agent_node. Four outcomes (third route below
    short-circuits a stuck-loop pattern that some open-source models exhibit).

    - "tools": the LLM emitted tool_calls and we still have iteration budget,
      AND the call is not a duplicate of an earlier call in this turn.
    - "fallback": either (a) we ran out of iteration budget mid tool-call,
      OR (b) the LLM is repeating a previously-issued tool call verbatim
      (a clear loop signal). Either way we emit a graceful answer.
    - "end": the LLM produced a final natural-language answer. Done.
    """
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    has_pending_tool_call = isinstance(last, AIMessage) and bool(last.tool_calls)

    if not has_pending_tool_call:
        return "end"

    is_loop, _ = _is_loop(state)
    if is_loop:
        logger.info("loop detected — short-circuiting to fallback")
        return "fallback"

    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "fallback"
    return "tools"


def fallback_node(state: GraphState) -> dict:
    """Emit a graceful answer when we abandoned the ReAct loop early.

    If we hit the loop because the LLM was re-calling the same tool, we
    surface the tool's most recent result as the user-facing answer (the
    LLM had the answer; it just refused to use it). Otherwise we fall back
    to the generic "ran out of budget" message.
    """
    is_loop, last_tool = _is_loop(state)
    if is_loop and last_tool is not None:
        logger.info("loop fallback: surfacing %s tool result as final answer", last_tool.name)
        text = LOOP_FALLBACK_TEMPLATE.format(
            tool_name=last_tool.name,
            content=str(last_tool.content).strip(),
        )
        return {"messages": [AIMessage(text)]}
    return {"messages": [AIMessage(MAX_ITER_MESSAGE.format(max_iter=MAX_ITERATIONS))]}
