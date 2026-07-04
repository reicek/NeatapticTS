---
name: bug-triage
description: 'Use when: triaging a bug report through assess → fix → test with URL-trust discipline and evidence artifacts.'
argument-hint: 'Provide the bug report text or URL, an optional slug, and any known suspect files or reproduction steps.'
user-invocable: false
disable-model-invocation: false
skills:
  - triaging-test-failures
  - test-fix-workflow
  - red-test-contracts
  - tracker-handoff
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Bug Triage

Use this skill when a bug report needs a dedicated artifact path and a
structured `bug-assess` → `bug-fix` → `bug-test` workflow. This skill owns
the layout under `.github/bugs/<slug>/`, the URL-trust rules for fetched bug
reports, and the regression-test gate that keeps a bug from being closed
without coverage.

It does **not** replace the existing `triaging-test-failures` and
`test-fix-workflow` skills. Instead, it delegates to them for failure
classification and multi-failure repair discipline while adding the Spec
Kit-style artifact path and gap-type vocabulary.

## When to Use

- A bug report arrives as pasted text, a URL, or a mix and needs a durable
  assessment artifact.
- The fix requires a focused red-phase regression test before the bug can be
  considered resolved.
- The report may contain untrusted URLs that must be classified before fetching.
- The work spans more than one session and needs a resumable slug directory.

## When NOT to use

Do NOT use for triaging test failures without a bug report - use `triaging-test-failures` first. Do NOT use for systematic multi-failure repair - use `test-fix-workflow` after triage.

## Workflow Diagram

```text
Flowchart summary: "Bug report" → "bug-assess"; "bug-assess" → "bug-fix"; "bug-fix" → "bug-test"; "bug-test" → "Close bug?"; "Close bug?" → "closed with regression test" (Yes), "reopen" (No)
```

## Gap Types

When judging a bug report or the resulting fix, classify any deviation against
the intended behavior using the four Spec Kit gap types:

- **missing** — the behavior was required but is absent.
- **partial** — the behavior exists but is incomplete, inconsistent, or only
  works in a subset of cases.
- **contradicts** — the behavior does the opposite of what is required or
  violates an established contract.
- **unrequested** — the report asks for something outside scope; document why
  before closing as not-a-bug.

Record the gap type in `assess.md` so `fix.md` and `test.md` can target the
right contract.

## Task Packet

Pass a compact packet that includes the bug description or URL, an optional
slug, and any known suspect boundary.

```text
Use bug-triage for worker transport payload corruption.
Report: pasted stack trace + https://github.com/owner/repo/issues/123
Slug: worker-payload-corruption (optional)
Suspect boundary: src/multithreading/multi.utils.ts
```

## Required Workflow

1. **bug-assess** — create `.github/bugs/<slug>/assess.md`.
   - Resolve the slug (user-provided, asked, or auto-generated with a
     disambiguating suffix). Never overwrite an existing slug directory.
   - Apply the URL-trust policy from `.github/bugs/README.md` before fetching
     any URL. Record the verbatim URL, parsed host, and branch taken
     (`allowlisted`, `confirmed-by-user`, or `auto-refused: <reason>`).
   - Ingest the report, summarize the symptom, and mark unknowns as
     `[NEEDS CLARIFICATION]` rather than guessing.
   - Search the codebase for suspect symbols, file paths, error strings, and
     route names.
   - Judge merit (`valid` / `likely valid, needs reproduction` / `invalid`)
     and assign severity (`critical` / `high` / `medium` / `low`).
   - Propose a preferred remediation, files likely to change, and tests to add.
   - Never modify source code during assessment.

2. **bug-fix** — create `.github/bugs/<slug>/fix.md`.
   - Confirm the assessment contract. If the verdict is `invalid`, stop.
   - Apply the smallest remediation that resolves the gap type recorded in the
     assessment.
   - Add or update a **regression test** that fails before the fix and passes
     after it. This test is mandatory before the bug can be closed.
   - Run focused checks on the changed paths.
   - Record files changed, tests added, local verification, and any deviations
     from the assessment.
   - If the assessment was wrong, stop modifying code, document the finding,
     and recommend re-running `bug-assess`.

3. **bug-test** — create `.github/bugs/<slug>/test.md`.
   - Re-run the reproduction steps from the assessment (or their automated
     equivalent).
   - Run the regression test added in `bug-fix`.
   - Run any broader regression suite that touches the changed files, but
     never run destructive or network-dependent checks without user consent.
   - Judge the outcome as `verified`, `partial`, or `failed`.
   - A bug may be marked closed only when the result is `verified` and the
     regression test passes. Downgrade to `partial` if the original
     reproduction could not be exercised.

## Artifact Layout

```text
.github/bugs/<slug>/
  assess.md       — symptom, root-cause hypothesis, severity, proposed remediation
  fix.md          — files changed, tests added, local verification notes
  test.md         — reproduction re-run results, regression test result, verdict
  evidence/       — screenshots, logs, stack traces, telemetry excerpts, URLs
```

## URL-Trust Policy

All fetched URLs are untrusted input. See `.github/bugs/README.md` for the
trusted-host allowlist, refused-host categories, and evidence-handling rules.

## Decision Tree

```text
Flowchart summary: "Bug report" → "Need durable artifact?"; "Need durable artifact?" → "bug-triage" (Yes), "triaging-test-failures" (No, just classify); "bug-triage" → "assess → fix → test"; "assess → fix → test" → "regression test passes?"; "regression test passes?" → "close" (Yes), "reopen" (No)
```

## Before / After Examples

**Before:**

```text
Tests are failing in worker transport. Maybe a timing issue.
```

**After:**

```text
Slug: worker-payload-corruption
Assessment: .github/bugs/worker-payload-corruption/assess.md
Gap type: contradicts (worker output differs from portable predictor)
Fix: ordered result assembly restored in multi.utils.ts
Test: regression test added in multi.utils.test.ts
Verification: .github/bugs/worker-payload-corruption/test.md → verified
```

## Guardrails

- Do not modify source code during `bug-assess`.
- Do not close a bug without a passing regression test.
- Do not overwrite an existing `assess.md`, `fix.md`, or `test.md` without
  confirmation.
- Do not fetch untrusted URLs without applying the URL-trust policy.
- Do not treat a `partial` verification as a closed bug.
- Do not let demo-local compensations hide library-level gaps.

## Expected Final Output

A strong bug-triage pass should report:

- the bug slug and `.github/bugs/<slug>/` path,
- the gap type (`missing`, `partial`, `contradicts`, `unrequested`),
- the files changed and the regression test added,
- the verification verdict (`verified`, `partial`, `failed`),
- references to any delegated skills (`triaging-test-failures`, `test-fix-workflow`).
