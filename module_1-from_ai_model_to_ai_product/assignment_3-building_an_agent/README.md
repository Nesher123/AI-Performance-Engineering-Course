# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers questions about the
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
Tools are exposed in-process for the CLI / Streamlit UI and over MCP via FastMCP.

> Status: scaffold in place. CLI, tools, router, memory, MCP, and Streamlit UI are
> being built incrementally — see the per-task plans for progress.

## Quick start (5-minute path)

```bash
# 1. Install
uv sync

# 2. Configure
cp .env.example .env
# paste your Nebius Token Factory key into NEBIUS_API_KEY

# 3. Run the CLI (once Task 1 is done)
uv run cs-agent --session demo --user ofir
```

A complete README with run instructions, model justification, architecture diagram,
tools reference, MCP client snippet, and smoke-test script will land at the end of
the implementation.
