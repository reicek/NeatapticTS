---
name: triaging-test-failures
description: 'Use when: a validation command fails and the workflow needs failure ownership, root-cause grouping, reroute decisions, or unrelated-failure separation.'
argument-hint: 'Provide the failing command, relevant output summary, changed files, and whether failures may be unrelated or environment-owned.'
user-invocable: false
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Triaging Test Failures

This skill interprets failing validation output and assigns ownership to each failure without widening the fix scope. It classifies failures by type, identifies the smallest owner boundary, and recommends the next agent or command rather than attempting a broad repair.

## When to Use

- A Jest run produced failures and it is unclear which failures are caused by the active change versus pre-existing issues.
- Multiple failures span unrelated files and need to be grouped before deciding where to focus.
- A CI run failed and the failure output needs to be distilled into actionable ownership items.
- A test was expected to be green but is red, and the root cause is not immediately obvious.
- Deciding whether to proceed with a fix, skip a pre-existing failure, or escalate to a different skill.
- An environment-owned failure (flaky test, missing dependency, port conflict) needs to be separated from a code defect.

## Task Packet

Include the failing command, a summary of the failure output (first error lines per failing test), the files changed in the active change, and whether any failures are suspected to be pre-existing or environment-owned.

```text
Use triaging-test-failures for <failing command>.
Failure summary: <failing test names and first error lines>
Active change files: <list of files changed>
Pre-existing suspected: <yes | no | unknown>
Environment factor: <none | possible port conflict | missing build artifact | flaky>
```

## Required Workflow

1. Read the failing command output; extract the failing test names and the first meaningful error line for each.
2. Classify each failure into one category:
   - `active-change`: caused by the current edit (import error, assertion mismatch, type error in a touched file).
   - `pre-existing`: present before the active change (visible in recent CI history or git blame shows no recent touch).
   - `environment`: port conflict, missing build artifact, missing env var, or OS-specific issue.
   - `flaky`: non-deterministic, passes on retry without code changes.
   - `unknown`: cannot classify without more information.
3. Identify the smallest owner boundary for each failure: the single file or function responsible.
4. For active-change failures: describe the root cause concisely and recommend the next action (fix, remove dead code, update test).
5. For pre-existing failures: confirm they are unrelated and note them so they are not treated as defects of the active change.
6. For environment failures: identify the missing precondition and recommend how to resolve it (rebuild, set env var, free port).
7. For flaky failures: recommend a retry and note the flakiness in the tracker.
8. Recommend the next agent or command — do not attempt a broad fix within this skill.
9. Preserve raw evidence as concise command/result snippets; do not paste full stack traces.

## Guardrails

- Do not attempt to fix failures within this skill; triage only — route fixes to the appropriate implementation or test skill.
- Do not classify a failure as pre-existing without evidence (git blame, CI history, or explicit confirmation).
- Do not recommend a broad refactor as the next action; recommend the smallest targeted fix.
- Do not conflate multiple failures into a single root cause unless you have direct evidence they share a source.
- Do not skip the classification step and jump directly to a fix recommendation; classification is the primary output of this skill.
- Do not include full stack traces in the triage report; distill to the failing test name and the first meaningful error line.

## Expected Final Output

- A structured failure list: failing test name → category → owner boundary → root cause or unknown.
- Separation of active-change failures from pre-existing, environment, and flaky failures.
- A concrete next action for each active-change failure (file to fix, test to update, dead code to remove).
- Next validation command to run after the recommended action.
