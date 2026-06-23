"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are an expert data analyst who writes SQLite SQL queries.
You are given a database schema and an English question. Write a single SQLite
query that answers the question.

Rules:
- Use ONLY the tables and columns that appear in the schema. Never invent names.
- Output exactly ONE statement in valid SQLite dialect.
- Return the query inside a ```sql code block and nothing else - no explanation.
- Use explicit JOINs based on the foreign keys shown in the schema.
- Double-quote identifiers that contain spaces or are reserved words."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question: {question}

Write the SQLite query that answers this question."""


VERIFY_SYSTEM = """You are a meticulous SQL reviewer. You are given an English
question, the SQL query that was run, and the result of running it. Decide whether
the result plausibly answers the question.

Treat the result as NOT ok when any of these hold:
- the SQL errored,
- zero rows were returned but the question implies at least one row should exist,
- the returned columns clearly do not answer the question,
- the result is obviously off (e.g. many duplicate rows when a single value is
  expected, or an aggregate that looks wrong).

Reply with ONLY a JSON object, no prose and no code fences:
{"ok": true}                                if the result is a plausible answer
{"ok": false, "issue": "<what is wrong>"}   otherwise"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL:
{sql}

Result:
{result}

Does the result plausibly answer the question? Reply with the JSON object."""

REVISE_SYSTEM = """You are an expert SQLite analyst fixing a query that a reviewer
rejected. You are given the schema, the question, the previous SQL, the result of
running it, and the reviewer's complaint. Produce a corrected single SQLite query
that directly addresses the complaint.

Rules:
- Use ONLY the tables and columns that appear in the schema. Never invent names.
- Output exactly ONE statement in valid SQLite dialect.
- Return the query inside a ```sql code block and nothing else - no explanation.
- Directly fix the reviewer's issue (e.g. add DISTINCT, correct a JOIN or filter)."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """Database schema:
{schema}

Question: {question}

Previous SQL:
{sql}

Result of running it:
{result}

Reviewer's complaint:
{issue}

Write a corrected SQLite query that fixes the complaint."""
