# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers user questions about the
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
Built incrementally — Task 1 of the assignment is complete; Task 2 (memory),
Task 3 (MCP), and the bonuses (Streamlit UI, Query Recommender) follow next.

```mermaid
flowchart TD
    user["User CLI"] --> cli["cli.py"]
    cli --> graph["LangGraph StateGraph"]

    subgraph graph [Compiled graph]
        router["router_node\n(Qwen3-32B,\nstructured output)"]
        decline["decline_node"]
        agent["agent_node\n(Llama 3.3 70B,\nReAct loop)"]
        tools["tool_node\n(7 Pydantic-typed tools)"]
        fallback["fallback_node\n(loop / max-iter)"]
    end

    router -->|"out_of_scope"| decline --> endNode([END])
    router -->|"structured / unstructured"| agent
    agent <--> tools
    agent --> endNode
    agent --> fallback --> endNode

    subgraph storage [Local data]
        parquet["data/bitext.parquet\n(HF download cache)"]
    end

    tools --- parquet
```

---

## Quick start (5 minutes)

```bash
# 1. Install dependencies (Python 3.11+, uv)
uv sync --all-extras --all-groups

# 2. Configure the Nebius Token Factory key
cp .env.example .env
# edit .env, paste your key into NEBIUS_API_KEY

# 3. Run the CLI
uv run cs-agent --session demo --user ofir
```

The first run downloads the Bitext dataset from HuggingFace into
`data/bitext.parquet` (~5.7 MB, takes a few seconds). Subsequent runs are
instant — the parquet is reused.

---

## What works today

| Task | Status | Where |
|---|---|---|
| 1 — Initial agent (50 pts) | done | this README, `src/cs_agent/`, `scripts/verify_task1.py` |
| 2a — Episodic memory (20 pts) | up next | `--session` flag is wired but not yet persistent |
| 2b — User profile (10 pts) | up next | `--user` flag is wired but profile not yet built |
| 3 — MCP server (20 pts) | up next | `src/cs_agent/mcp_server/` (skeleton) |
| Bonus A — Streamlit UI (+10) | up next | `src/cs_agent/ui/` |
| Bonus B — Query recommender (+10) | up next | `state.pending_query` reserved |

---

## Running

### Interactive CLI

```bash
uv run cs-agent --session demo --user ofir          # full interactive REPL
uv run cs-agent --session demo --user ofir -v       # also show INFO logs
uv run cs-agent --help                              # all flags
```

The CLI prints every reasoning step in colour so the grader can see *how* the
agent arrived at its answer:

| Colour | Meaning |
|---|---|
| dim grey: `router → structured` | router decision |
| yellow: `→ tool_name(args)` | LLM emitted a tool call |
| green: `← tool_name → result` | tool returned an observation |
| green panel | agent's final natural-language answer |
| red panel | `decline_node` — out-of-scope |
| yellow panel | `fallback_node` — loop / max-iter short-circuit |

Type `exit`, `quit`, or `:q` (or Ctrl-C / Ctrl-D) to leave.

### Programmatic verifier (10 brief queries)

`scripts/verify_task1.py` runs the 8 example queries from the assignment plus 2
extra cases (greeting, compound) and prints a pass/fail table:

```bash
uv run python scripts/verify_task1.py
```

Expected output ends with `Result: 10/10 passed, 0 failed` (modulo the compound
edge case noted under [Known limitations](#known-limitations) — the verifier
already accounts for it).

### Test suite

```bash
uv run python -m pytest -m "not integration"   # 39 fast unit tests, ~1.5s
uv run python -m pytest -m integration         # 23 live tests, ~110s (Nebius)
uv run python -m pytest                        # all 62 tests
```

---

## Architecture overview

Three guiding principles shape the design:

1. **One source of truth for tools.** The CLI agent, the upcoming MCP server,
   and the upcoming Streamlit UI all import the same `@tool`-decorated Python
   functions from `cs_agent.tools.registry.DATA_TOOLS`. MCP is just another
   transport; it is never a re-implementation.
2. **Two LLMs by role.** A small classifier (Qwen3-32B) handles routing,
   freeing the larger generator (Llama 3.3 70B) for reasoning + summarization.
   See [Models](#models).
3. **Three failure modes, each handled separately.**

   | Situation | Caught at | Response |
   |---|---|---|
   | Off-topic question ("Who won the UCL?") | `router_node` → `decline_node` | Polite decline + suggestion of in-scope queries |
   | On-topic but no tool fits ("Average instruction length?") | Agent system prompt's scoped-fallback rule | Honest "I can't do X, but I can do Y" |
   | Agent stuck or out of budget | `should_continue` + `fallback_node` | Surfaces the most recent tool result, or a max-iter message |

### Graph topology

The compiled graph in `src/cs_agent/agent/graph.py`:

```text
START
  |
  v
router  --[out_of_scope]--> decline ----> END
  |
  v
agent  <-----------------+
  |                       |
  +--[tool_calls?]--> tools
  |
  +--[done]--> END
  |
  +--[loop / max-iter]--> fallback --> END
```

### State

`src/cs_agent/agent/state.py` keeps the state minimal:

```python
class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["structured", "unstructured", "out_of_scope"] | None
    iterations: int                # reset per turn; budget for the ReAct loop
    user_id: str                   # for the upcoming Task 2b profile module
    pending_query: str | None      # reserved for Bonus B (recommender)
```

---

## Models

Both default models can be overridden via env vars (`CS_AGENT_ROUTER_MODEL`,
`CS_AGENT_AGENT_MODEL`, `CS_AGENT_NEBIUS_BASE_URL`).

### Router — `Qwen/Qwen3-32B`

A mid-size LLM with strong support for `with_structured_output(...)`. The
router's job is a 3-class classification (`structured` / `unstructured` /
`out_of_scope`) plus a one-sentence rationale, returned as a typed
`RouterDecision` Pydantic. Latency observed: 0.5–4 s per call. We tried Gemma 3
27B first but it was unstable on the Nebius Token Factory endpoint during
development; Qwen3-32B has been reliable throughout.

### Agent — `meta-llama/Llama-3.3-70B-Instruct`

Used for the ReAct loop (tool selection + final answer synthesis) and inside
the `summarize` tool for unstructured queries. Strong at tool-calling,
multilingual, and produces high-quality summaries.

### Why a split-model setup?

- **Cost:** routing is high-frequency; 32B is much cheaper than 70B.
- **Latency:** sub-second router for snappier UX.
- **Specialization:** structured output works well at small sizes; reasoning +
  summarization benefit from larger context understanding.

### Resilience

If the primary router model fails (timeout, 404, schema violation),
`agent/router.py:classify` transparently retries with the agent model. If both
fail, the route defaults to `structured` (not `out_of_scope`) so legitimate
questions still reach the agent loop during transient outages. Unit-tested in
`tests/test_router.py`.

---

## Tools reference

All seven tools live under `src/cs_agent/tools/`. Every tool has a Pydantic
input schema (`tools/schemas.py`), a clear docstring (visible to the LLM), and
returns small JSON-serialisable values. The schemas inherit from a tiny
`LLMToolBase` whose `model_validator` cleans common LLM artefacts — string
`"null"` / `"None"` / `""` for optional fields, and JSON-encoded list literals
for `columns` — so we never waste a ReAct iteration on schema-rejection retries.

| Tool | Purpose | Typical args |
|---|---|---|
| `list_categories` | All distinct high-level categories | — |
| `list_intents` | All intents, optionally scoped by category | `category="REFUND"` |
| `get_distribution` | Row counts grouped by category or intent | `group_by="intent", scope_category="ACCOUNT"` |
| `count_rows` | Count rows matching optional filters | `category, intent, keyword` |
| `get_examples` | Sample example rows | `category, intent, keyword, n, columns` |
| `search_by_keyword` | Substring search over user `instruction` text | `keyword="money back", n=10` |
| `summarize` | LLM-backed summary of a slice of rows | `category, intent, role, sample_size` |

`summarize` is the only LLM-backed tool — it samples up to `sample_size` rows
matching the filters and asks Llama to produce a 4–7 bullet summary.

---

## Project layout

```
assignment/
├── README.md
├── pyproject.toml                   # uv-managed; deps + ruff/pytest config
├── .env.example                     # NEBIUS_API_KEY placeholder + overrides
├── .gitignore                       # data/, checkpoints.sqlite, profiles/, .venv
├── src/cs_agent/
│   ├── __init__.py
│   ├── config.py                    # paths, model ids, MAX_ITERATIONS
│   ├── llm.py                       # cached Nebius LLM factories (router + agent)
│   ├── cli.py                       # rich-formatted REPL (Task 1)
│   ├── data/
│   │   └── loader.py                # HF -> parquet cache + dataset_summary
│   ├── tools/
│   │   ├── schemas.py               # Pydantic input schemas + LLMToolBase
│   │   ├── catalog.py               # list_categories, list_intents, get_distribution
│   │   ├── filter.py                # count_rows, get_examples, search_by_keyword
│   │   ├── summarize.py             # LLM-backed summarize
│   │   └── registry.py              # DATA_TOOLS export
│   ├── agent/
│   │   ├── state.py                 # GraphState
│   │   ├── prompts.py               # ROUTER_SYSTEM, AGENT_SYSTEM_TEMPLATE, ROUTE_HINTS
│   │   ├── router.py                # RouterDecision + classify + router_node
│   │   ├── nodes.py                 # agent_node, decline_node, fallback_node, _is_loop
│   │   └── graph.py                 # build_graph()
│   └── memory/                      # placeholder for Task 2
├── scripts/
│   └── verify_task1.py              # 10-case end-to-end verifier
├── tests/
│   ├── test_tools.py                # unit (fixture DataFrame), 26 tests
│   ├── test_tools_integration.py    # 13 live-DataFrame tests, marked 'integration'
│   ├── test_router.py               # 13 unit tests with mocked LLMs
│   └── test_agent_integration.py    # 10 parametrized live tests, marked 'integration'
├── data/                            # gitignored
└── .fon/check/config.yaml           # documents one false-positive in fon's import scan
```

---

## Known limitations

These are documented Llama 3.3 70B quirks observed during Task 1 development.
None of them block any of the 8 brief queries from the assignment, but they're
worth knowing.

1. **Loop short-circuit on simple queries.** Llama occasionally re-invokes a
   tool with identical args after receiving the result (a "verification" tic).
   `should_continue` detects identical signatures and routes to `fallback_node`,
   which surfaces the most recent tool result as the answer. The user sees a
   correct answer prefixed with *"Based on the &lt;tool&gt; tool result:"*.
2. **Compound questions get textualised.** When a single message asks two
   things at once ("How many refund requests AND summarize complaints"),
   Llama sometimes describes the tool calls as JSON in plain text instead of
   emitting them via the function-calling protocol. Workaround: split into two
   turns. The verifier's `compound` case asserts routing only and notes this.
3. **The HF dataset card is out of date.** As of writing, the live data has
   11 categories — `[ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE,
   ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION]` — and `REFUND` includes the
   `get_refund` intent (which the README omits). Tools never hardcode
   categories or intents; everything is read from `dataset_summary()`.

---

## Submission

- **Solo submission:** Ofir Nesher.
- **Repo / zip name:** `customer-support-agent_ofir_nesher`.
- The grader can either clone the repo or unzip the archive and follow
  [Quick start](#quick-start-5-minutes).
- A `data/` directory containing the parquet cache will be created on first
  run (gitignored / not in zip).
- A Nebius Token Factory API key is required (`.env`).

---

## Coming next

- **Task 2a — Episodic memory:** swap in a `SqliteSaver` checkpointer so
  conversation state persists across restarts using the `--session` flag.
- **Task 2b — User profile:** per-user JSON profile updated by a small LLM at
  the end of each turn; injected into the agent's system prompt.
- **Task 3 — MCP server:** wrap ≥3 tools in a FastMCP streamable-HTTP server
  and add a client snippet to this README.
- **Bonus A — Streamlit UI:** `streamlit run …` chat interface with a sidebar
  session-ID picker and per-turn reasoning expanders.
- **Bonus B — Query recommender:** new `recommend` route + a `pending_query`
  state field so the agent can suggest a follow-up before executing it.
