# ADR: Store prompts as versioned text files (document-as-implementation)

- Status: Accepted
- Date: 2026-02-14
- Deciders: Project maintainers
- Tags: prompting, evaluation, maintainability, reproducibility

## Context

Sum-It-Up Agent relies on prompts that strongly influence behavior, output structure, and quality. Prompt iteration is frequent (wording, constraints, output formats, tone), while the surrounding orchestration code may remain stable.

Two recurring needs drive this decision:

1) **Change prompts without changing code**
- Prompt updates should not require modifying or redeploying application logic.
- Prompt edits should be reviewable independently from code changes.
- Prompt hotfixes should not risk introducing unrelated code regressions.

2) **Maintain multiple prompt versions for evaluation**
- We need to compare prompt variants (A/B, baselines, regressions) using offline evaluation.
- The evaluation harness must be able to pin an exact prompt version and reproduce results.
- We need a clear history of prompt evolution and the ability to roll back.

If prompts are embedded directly in code, prompt iteration becomes coupled to code changes, and prompt variants become harder to manage, compare, and reproduce.

## Decision

Store prompts as plain text files in the repository and load them at runtime. Prompts are treated as versioned artifacts (“document-as-implementation prompts”).

- Prompts live in a dedicated `prompts/` directory and are referenced by path.
- Multiple prompt versions are represented as separate files (or versioned subfolders), enabling evaluation to select a specific variant without code changes.
- Git history serves as the canonical versioning mechanism for prompts.

## Options considered

### Option A — Inline prompts in code (constants / string literals)
**Pros**
- Simple to implement
- No file packaging concerns

**Cons**
- Prompt edits require code edits (tight coupling)
- Harder to maintain multiple prompt versions and run reproducible evaluations
- PR diffs mix prompt changes with logic changes

### Option B — Versioned text prompt files in repo (chosen)
**Pros**
- Prompt changes decoupled from code logic
- Easy to keep and compare multiple prompt versions for evaluation
- Clear diffs and review process via git
- Easy rollbacks to known-good prompt versions

**Cons**
- Requires prompt loading/management
- Packaging must include prompt files

### Option C — External prompt storage (DB/config service)
**Pros**
- Hot updates without redeploy

**Cons**
- Reproducibility requires additional snapshot/version discipline
- Adds operational dependency and complexity

## Consequences

### Positive
- Faster prompt iteration with reduced risk to code stability.
- Supports evaluation workflows: baseline vs candidate prompts, regression tracking, reproducible runs.
- Encourages modularity: prompts can be reused across components without duplication.

### Negative / Risks
- Runtime failures if prompt files are missing/mispackaged.
- Path refactors can break references unless guarded.

## Implementation notes

Recommended directory structure:
- `prompts/system/`
- `prompts/intent/`
- `prompts/summarization/`
- `prompts/formatting/`

Versioning strategies:
- File-level: `prompts/intent/v1.md`, `v2.md`, `baseline.md`, `candidate.md`
- Or folder-level: `prompts/intent/v1/`, `v2/`

Runtime selection:
- Default prompt path in config.
- Evaluation harness can override the prompt path/version per run (CLI/config), without code changes.

Testing:
- Validate configured prompt paths exist.
- For structured outputs, add schema/golden tests per prompt version used in evaluation.

## Follow-ups

- Add an evaluation config that enumerates prompt versions to run (baseline + candidates).
- Add CI checks to ensure prompt paths exist and structured outputs remain valid for pinned prompt versions.
