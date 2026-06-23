# LLM Inference + Observability — Assignment Report

Text-to-SQL PoC over BIRD-bench, served by vLLM (`Qwen/Qwen3-30B-A3B-Instruct-2507`)
with a LangGraph self-consistency agent, Prometheus + Grafana for serving o11y, and
Langfuse for agent tracing.

> Submission note: Phases 3–5 were first validated locally against Nebius Token Factory
> (hosted Qwen3-30B). The booked H100 slot was initially missed; I later cloned an H100 VM
> in the course Nebius project (`ofir-mlops-hw2`) and completed the self-hosted runs:
>
> - Phase 1 (serving config) — deployed on 1× H100 80GB (§1).
> - Phase 2 (Grafana) — live vLLM `/metrics` + `screenshots/grafana_serving.png` (§2).
> - Phase 3 (agent), Phase 4 (tracing), Phase 5 (evals) — complete on H100 (§3–5).
> - Phase 6 (SLO) — load-tested on H100; SLO missed with metric-grounded diagnosis (§6).
>
> Eval accuracy matched between hosted and self-hosted backends (36.7%), confirming the
> agent quality is a function of model + prompts, not the serving layer.

---

## 1. Serving configuration (Phase 1)

Deployed on 1× NVIDIA H100 80GB (`gpu-h100-sxm`), Ubuntu 24.04 + CUDA 13, model
`Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~3.3B active params per token).

Measured flags (vLLM 0.10.2, V0 engine):

| Flag | Value | Why |
| --- | --- | --- |
| `--max-model-len` | `8192` | Agent prompts are ~1.5–3K tokens; default 262K would waste KV cache. |
| `--gpu-memory-utilization` | `0.92` | Model weights use ~57 GiB; leaves ~15 GiB for KV cache at this max len. |
| `--enforce-eager` | on | Avoids CUDA-graph compile issues on first boot; acceptable for eval/load-test. |
| `VLLM_USE_V1=0` | V0 engine | V1 engine hit Triton/gcc build failures on the VM image. |

Startup profile: 56.9 GiB model weights, ~14.3 GiB KV cache, max concurrency ~19 sequences
at 8192 tokens (vLLM startup log).

Not yet tuned: FP8 quantization, `--max-num-seqs`, `--enable-chunked-prefill` — the levers
I would iterate for the SLO (§6).

---

## 2. Observability dashboard (Phase 2)

Prometheus + Grafana run from `docker-compose.yml` on the H100 VM. Dashboard JSON:
`infra/grafana/provisioning/dashboards/serving.json`. Panels:

- Request latency percentiles (p50/p95/p99) via `vllm:e2e_request_latency_seconds_bucket`.
- Time to first token p95 via `vllm:time_to_first_token_seconds_bucket`.
- Requests running vs waiting via `vllm:num_requests_running` and `vllm:num_requests_waiting`.
- KV-cache utilization via `vllm:gpu_cache_usage_perc`.
- Prompt/generated token throughput and request throughput.

Captured during eval + load test: `screenshots/grafana_serving.png` (p50 ~3–4s, p95 ~8–10s
during the 22:40–22:46 burst; an earlier p99 spike to ~45s from a few multi-revise requests).

---

## 3. Agent design (Phase 3)

A self-consistency-inspired LangGraph agent (`agent/graph.py`):

```text
attach_schema -> generate_sql -> execute -> verify
                                              | ok=true  -> END
                                              | ok=false -> revise -> execute -> verify (loop)
```

- `generate_sql`, `verify`, `revise` each make one vLLM call; `execute` runs the SQL
  read-only against the target SQLite DB.
- `verify` returns a JSON verdict `{"ok": bool, "issue": str}`, parsed defensively
  (`_parse_verdict`); on unparseable output it defaults to `ok=true` to avoid loops.
- `route_after_verify` loops into `revise` unless verify is satisfied or
  `MAX_ITERATIONS` (currently 3) is reached.
- Prompts in `agent/prompts.py` are schema-grounded and demand a single fenced SQL
  statement so `_extract_sql` can parse reliably.
- Served via FastAPI at `http://localhost:8001/answer` (`agent/server.py`).

Validation: the loop genuinely fires. Example (eval Q1, `formula_1`): the first query
returned 11 duplicate coordinate rows; `verify` flagged it ("implausible for a single
circuit"); `revise` added `SELECT DISTINCT`; `verify` then passed. Across the eval set,
12/30 questions triggered at least one revise.

---

## 4. Agent tracing (Phase 4)

Langfuse (local, from `docker-compose.yml`) captures every run as a span waterfall:
`generate_sql` / `verify` / (sometimes) `revise`, each with prompt, response, latency,
and token counts. Enabled purely via `.env` keys (the `CallbackHandler` in
`agent/server.py` activates when keys are present). Traces tagged with metadata
(`phase`, `dataset`, `backend`) for Phase 6 filtering.

Observation worth noting: `revise` is the most expensive call (~1.6K prompt tokens) because
it carries schema + question + failed SQL + result + verifier complaint. Revised
questions cost ~3 LLM calls; this is the dominant latency path.

Deliverables: `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png`. Traces
also captured on the H100 VM (Langfuse via `docker compose`, keys from the VM instance).

---

## 5. Baseline eval results (Phase 5)

Execution accuracy: agent's final SQL vs. gold SQL, compared on canonicalized row sets
(rows sorted, cells stringified, NULL→''). 30 BIRD questions, `temperature=0`.

**H100 (self-hosted vLLM)** — `results/eval_h100.json`:

- Overall pass rate: 36.7% (11/30)
- Pass rate by iteration (carry-forward):
  - Iter 0 (generate only): 26.7% (8/30)
  - Iter 1 (after 1st revise): 40.0% (12/30)
  - Iter 2 (after 2nd revise): 36.7% (11/30)
- Iteration distribution: 18 stop at 1, 5 at 2, 7 hit the cap at 3.
- Revise triggered on 12/30 questions; mean iterations 1.63.

**Hosted baseline (Nebius Token Factory)** — `results/eval_baseline.json`: identical
overall pass rate (36.7%) and per-iteration shape. This confirms eval numbers are
model/prompt-dependent, not serving-layer-dependent.

Commentary: the first revise is clearly net-positive (+13.3 points, fixes 4 questions),
so the architecture earns its keep. The second revise regresses by one question — see §6–7.
Absolute accuracy is modest because BIRD is hard and the agent is deliberately simple;
the per-iteration delta, not the absolute number, is the signal of interest.

---

## 6. Hitting the SLO (Phase 6)

SLO target: P95 end-to-end agent latency < 5s at 10+ RPS for 5 minutes.

### Hosted baseline (pre-H100)

`results/load_test_rps1.json`, `results/load_test_rps3.json` — agent backed by Nebius
hosted Qwen3-30B from a laptop:

- 1 RPS / 45s: p50 4.0s, p95 11.0s; achieved 0.88 RPS; 6/45 HTTP 500.
- 3 RPS / 60s: p50 2.8s, p95 13.1s; achieved 1.5 RPS; 23/180 HTTP 500.

Diagnosis: tail latency + ~13% upstream 500s from shared hosted tier and network RTT.

### H100 self-hosted run

`results/load_test.json` — same driver, agent on VM, vLLM on localhost:8000:

- 3 RPS / 60s: p50 4.6s, p95 **26.1s**, p99 42.0s; achieved **1.76 RPS**; 157/180 ok,
  23 HTTP 500.

Verdict: **SLO still missed** on H100, but the failure mode changed.

### Iteration log

- **saw (hosted)** → p95 ~11–13s even at 1 RPS + ~13% HTTP 500s; achieved RPS << requested.
- **hypothesized** → remote API throttling + agent↔model network RTT + multi-call revise tail.
- **changed** → self-hosted vLLM on H100 (no network hop, dedicated GPU, Prometheus metrics).
- **result (H100)** → upstream 500 rate unchanged (~13%), p95 **worse** (26s vs 13s),
  achieved RPS slightly better (1.76 vs 1.5) but still far below 3 RPS offered.

- **saw (H100 + Grafana)** → vLLM serving p50 ~3–4s during burst but p95/p99 spikes to
  8–45s; errors correlate with concurrent agent requests each making 2–3 sequential LLM calls.
- **hypothesized** → bottleneck is agent architecture (serial LLM calls per request) +
  vLLM queueing under offered concurrency, not raw single-request generation speed.
  `--enforce-eager` and no `--max-num-seqs` tuning leave throughput headroom on the table.
- **would change next** → cap `MAX_ITERATIONS` at 2 (quality + fewer calls); raise
  `--max-num-seqs`; try FP8 + chunked prefill; re-run `load_test/driver.py` at 10 RPS.

Not completed: `results/eval_after_tuning.json`, before/after Grafana pair after tuning.

---

## 7. Agent value — did the loop help?

Yes, the loop adds measurable value, but with a clear caveat. The first revise lifts pass
rate from 26.7% to 40.0% — it recovers 4 questions the single-shot model got wrong. That
is the self-consistency architecture paying for itself.

The caveat: the second revise *regresses* to 36.7%. Root cause is structural, not random.
The agent has no ground truth; it acts only on `verify`'s judgment, which is itself an
imperfect LLM classifier. A `verify` **false negative** — rejecting an already-correct
result (e.g. a legitimately empty or single-row answer it deems "implausible") — forces a
needless revise that can break a correct query. Compounding this, the agent always returns
the *last* SQL, so a bad late revise overwrites a good earlier one.

---

## 8. What I'd do with more time (specific)

- **Cap iterations at 2.** The data shows iteration 2 only hurts; stopping after the first
  revise would likely raise overall accuracy to ~40%.
- **Make `verify` higher-precision.** Restrict `ok=false` to strong signals (SQL error,
  zero rows when the question implies rows, columns that don't answer the question) so it
  stops rejecting borderline-correct results. Measure verifier precision/recall against gold.
- **Keep the best candidate, not the last.** Track all iteration SQLs and return the one
  the verifier was most confident in, instead of blindly the final attempt.
- **Schema retrieval for wide DBs.** Some BIRD schemas are large; selecting only relevant
  tables/columns would shrink prompts (cheaper, faster) and reduce hallucinated joins.
- **Few-shot exemplars** of question→SQL per DB to lift the iteration-0 pass rate, which
  is the cheapest place to gain accuracy.
