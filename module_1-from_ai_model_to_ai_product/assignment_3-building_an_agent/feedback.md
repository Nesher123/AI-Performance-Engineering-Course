Feedback - Score: 120

Per-Task Feedback
Task 1 — Query Router — 15 / 15
Full marks. Dedicated router_node runs at graph entry (START → router) using Qwen3-32B with with_structured_output(RouterDecision) for typed 3-way classification. Out-of-scope queries hit decline_node with a canned message — no LLM general knowledge leaks. Resilient fallback chain (primary → agent model → default structured) is a thoughtful addition.

Task 1 — Tools & Schemas — 15 / 15
Seven well-designed tools (list_categories, list_intents, get_distribution, count_rows, get_examples, search_by_keyword, summarize) covering all required query types. Every tool has a dedicated Pydantic schema in tools/schemas.py with rich Field(description=...) annotations. The LLMToolBase model validator for cleaning LLM artifacts (string "null", JSON-encoded lists) is a smart design choice. Docstrings are clear and tell the LLM exactly when to select each tool.

Task 1 — Multi-step Reasoning — 10 / 10
Genuine ReAct loop: agent → tools → agent via should_continue. Tools are granular — multi-tool queries (e.g., list intents then count per intent) require actual chaining. Loop detection (_is_loop) catches duplicate tool-call patterns and short-circuits gracefully.

Task 1 — CLI — 5 / 5
Full interactive REPL in cli.py with rich color-coded reasoning trace (router decisions, tool calls, tool results, final answers in styled panels). Exit handling via keywords and Ctrl-C/Ctrl-D.

Task 1 — Max-iterations Fallback — 5 / 5
MAX_ITERATIONS = 12 configurable via env var. Checked in both agent_node and should_continue. fallback_node emits either the last tool result (loop case) or a polite "ran out of budget" message.

Task 2a — Episodic Memory — 20 / 20
SqliteSaver backed by checkpoints.sqlite on disk — genuinely persistent across restarts. --session + --user compose the thread_id via make_thread_id. Per-turn invoke passes only the new HumanMessage with explicit iterations=0, route=None reset; the add_messages reducer and checkpointer own the running history. Follow-ups and reference resolution work naturally.

Task 2b — User Profile — 10 / 10
UserProfile Pydantic model with distilled fields (name, role, topics_of_interest, preferences, notable_facts) stored as profiles/<user_id>.json — clearly separate from conversation history. profile_update_node with cheap regex gate (is_personal_info_bearing) avoids wasting LLM calls on dataset Q&A turns. Profile recall via dedicated profile_recall_node with render_recall_answer(). Persists on disk, survives restarts, keyed by user (not session).

Task 3 — MCP Server — 20 / 20
FastMCP server in mcp_server/server.py exposing 6 tools (all structured tools, summarize intentionally excluded since it requires an API key). Each MCP wrapper delegates directly to TOOLS_BY_NAME[<name>].invoke(...) — single source of truth. README includes server launch instructions, a complete Python client example, and Cursor/Claude Desktop configuration. Clear and actionable.

Bonus A — Streamlit UI — 10 / 10
Full chat UI with streaming, reasoning expander per turn, and sidebar with session/user ID switcher + reload button. @st.cache_resource shares graph/checkpointer across Streamlit reruns. Pure-Python rendering helpers in rendering.py enable offline testing. Cross-transport resume (CLI ↔ Streamlit) works by sharing the same checkpointer.

Bonus B — Query Recommender — 10 / 10
Complete suggest → refine → confirm → reject flow. Suggestions grounded in user profile + recent conversation context via _suggest. Intent classification (confirm/refine/reject) via structured output. pending_query persisted in graph state via checkpointer — survives restarts. Router short-circuits to recommender when pending_query is set. Only executes on explicit confirmation via synthetic HumanMessage.

Cross-cutting
No deductions. pyproject.toml has version-bounded dependencies with a uv.lock for exact reproducibility. README is thorough with architecture overview, model justification, mermaid diagrams, and walkthrough examples. Both models are Nebius Token Factory hosted. Repo is cleanly organized into logical modules (agent/, tools/, memory/, mcp_server/, ui/, data/) with comprehensive tests (120 tests across unit and integration). Exemplary submission.
