# Issue ideas backlog

This file contains copy-pastable GitHub issue drafts.

## Conventions

- Each issue is written with a clear **Title** and a detailed **Description**.
- Where possible, each issue includes **Acceptance Criteria**.

---

# Internal roadmap (epics)

## 1) Add real MCP health checks (replace port-open probing)

**Description**

`SumItUpApp._wait_for_servers_ready()` currently checks whether a TCP port is open. This can be a false positive if the server process is alive but not ready (e.g., models still loading, missing env vars, crashed after binding, etc.).

Add a lightweight health endpoint (tool/resource) per MCP server and change the app readiness logic to call it.

**Acceptance Criteria**

- Each MCP server exposes a `health` tool or resource that returns JSON:
  - `status: "ok" | "error"`
  - `ready: bool`
  - `service: str`
  - `version: str` (optional)
  - `details: dict` (optional)
- `SumItUpApp._wait_for_servers_ready()` calls each server health endpoint instead of checking port-open.
- Startup failures include actionable errors (missing env var, model load error, etc.).

---

## 2) Request correlation across the pipeline (`request_id`)

**Description**

Introduce a per-request `request_id` to correlate logs across:

- App process
- Orchestrator
- MCP servers

The project already has `sum_it_up_agent.observability.logger` with `contextvars` support. Use it to bind a `request_id` per `process_request()`.

**Acceptance Criteria**

- Orchestrator generates a UUID per `process_request`.
- Orchestrator binds it using `bind_request_id(request_id)`.
- All logs include `[request_id]` (already supported by logging filter).
- (Optional) MCP calls propagate `request_id` so each server can bind it too.

---

## 3) Environment variable consistency + startup validation

**Description**

Environment variable names differ between components. For example the communicator server expects `SMTP_SERVER` / `SMTP_PORT` / `SENDER_EMAIL_*`, but the app currently prepares `EMAIL_*` variables. This leads to confusing runtime failures.

Standardize env vars and validate them at startup.

**Acceptance Criteria**

- One consistent env var spec documented in `.env.example` and README.
- App passes the correct variables when booting each MCP server.
- Each MCP server validates required vars and returns a clear error via health endpoint.

---

## 4) Structured error taxonomy for pipeline failures

**Description**

Current error handling mostly passes raw exception strings. Introduce a small error taxonomy so callers can reason about failures programmatically.

**Acceptance Criteria**

- Define error codes (e.g., `E_AUDIO_FORMAT`, `E_IO_NOT_FOUND`, `E_MCP_UNAVAILABLE`, `E_LLM_AUTH`).
- `PipelineResult` includes `error_code` and `error_details`.
- MCP servers map expected exceptions to error codes.

---

## 5) Proper CLI entrypoint (Typer/Click)

**Description**

Add a CLI for non-interactive usage and automation:

- Start/stop/status servers
- Process an audio file with prompt

**Acceptance Criteria**

- `sum-it-up servers start|stop|status`
- `sum-it-up process <audio> --prompt "..." [--output-dir ...]`
- Non-zero exit codes on failure
- CLI help text and examples in README

---

## 6) Export & artifact management (single output folder per run)

**Description**

Organize pipeline outputs (transcription JSON, topic classification, summary JSON, logs) under a single run directory.

**Acceptance Criteria**

- A stable directory layout:
  - `<output_dir>/<request_id>/transcription.json`
  - `<output_dir>/<request_id>/topic.json`
  - `<output_dir>/<request_id>/summary.json`
  - `<output_dir>/<request_id>/run.log`
- Summary JSON contains references to all produced artifacts.

---

## 7) Cache eviction policy improvements (LRU/TTL + metrics)

**Description**

MCP servers cache model-heavy objects. Improve cache eviction strategies and expose metrics.

**Acceptance Criteria**

- Configurable eviction: `max_cached`, `ttl_seconds`, and/or LRU.
- Logs/metrics for cache hits, misses, and evictions.

---

# Contributor-friendly issues (smaller, well-scoped)

## A) Add issue templates (bug report + feature request)

**Description**

Add GitHub issue templates under `.github/ISSUE_TEMPLATE/`.

**Acceptance Criteria**

- `bug_report.yml` template
- `feature_request.yml` template
- Templates request:
  - component (agent/audio/summarizer/topic/communicator)
  - steps to reproduce
  - expected vs actual
  - logs

---

## B) Add `health` tool to one MCP server (starter)

**Description**

Implement a `health` tool/resource in a single MCP server (e.g., communicator) as a reference implementation.

**Acceptance Criteria**

- A tool/resource callable via MCP returning `{status, ready, service}`.
- Documented in that server module.

---

## C) Fix communicator env-var mismatch

**Description**

Align environment variables passed from `SumItUpApp` to match what `EmailCommunicator` uses.

**Acceptance Criteria**

- Update `app.py` communicator env var mapping.
- Update `.env.example` if needed.
- Confirm sending still works with env-only secrets.

---

## D) Add unit tests for prompt-length fallback

**Description**

`PromptParser.parse_prompt` has logic to skip LLM parsing if prompt exceeds `prompt_limit`. Add tests for this behavior.

**Acceptance Criteria**

- Tests cover:
  - prompt shorter than limit -> uses provider
  - prompt longer than limit -> returns fallback intent

---

## E) Add basic linting (ruff) and formatting (black) in CI

**Description**

Add `ruff` and `black` checks to CI for consistent style.

**Acceptance Criteria**

- CI runs `ruff check` and `black --check`.
- Minimal configuration committed.

---

## F) Add a "dry run" mode to orchestrator (no external calls)

**Description**

Add a mode that runs pipeline without calling external providers (LLMs, email).

**Acceptance Criteria**

- Config flag `dry_run=True`.
- Summarizer/communicator steps are skipped or stubbed.
- Pipeline result clearly indicates which steps were skipped.

---

# Nice-to-have features (future)

## X) Slack/Discord/Jira communicator adapters

**Description**

Extend communicator to support additional channels behind the same interface.

**Acceptance Criteria**

- Add at least one new channel adapter behind `ICommunicator`.
- Orchestrator can route to it from prompt intent.

---

## Y) Persist run history + simple UI

**Description**

Add a small local run history store (SQLite) and a minimal UI or CLI command to browse results.

**Acceptance Criteria**

- Store request id, input file, timestamps, artifact paths.
- List runs and open artifacts.
