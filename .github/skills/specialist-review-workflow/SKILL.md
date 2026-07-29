---
name: specialist-review-workflow
description: 'Use when: deciding how many specialist reviewers a fix or implementation slice needs.'
argument-hint: 'List the changed files and any uncertainty about whether the change is trivial (tests/docs/formatting) or non-trivial (runtime logic).'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
  - green-validation-gates
  - implementation-standards
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

> **Shared validation rule:** Run `shared-validation.gate.mjs` once before specialist dispatch. Specialists receive the JSON artifact and do not re-run tests, build, or lint.

# Specialist Review Workflow

Use this skill when an orchestrator must decide how many Tier-3 specialist
reviewers to dispatch before green testing.

The `execute` skill requires specialist review for `04-implementing` slices
that touch runtime code, but not every fix is equally risky. Documentation,
policy, test-only, and formatting changes need zero specialist review. Runtime
code changes need exactly 1 specialist. This skill owns the severity classifier
that distinguishes trivial fixes from non-trivial ones and tells the orchestrator
how many specialists to dispatch (0 or 1).

When tracker files need updating, `tracker-handoff` owns the plan/log shape.
When the classifier itself needs changing, `implementation-standards` and
`green-validation-gates` define the validation bar.

## When to Use

- An `04-implementing` slice is ready for specialist review.
- A fix-loop retry arrives and the orchestrator must decide whether the new
  change set is still non-trivial.
- A plan or agent wants to document why a slice received 1 specialist instead
  of 3+.
- The `specialist-review-severity.gate.mjs` gate needs interpretation.

## When NOT to use

Do NOT use for skipping specialist review on non-trivial runtime changes. Do NOT use for replacing the mandatory pre-green specialist review gate — only for sizing it.

## Workflow Diagram

```text
Flowchart summary: "Changed files" → "shared-validation.gate.mjs"; "shared-validation.gate.mjs" → "pass?"; "pass?" → "classifySeverity" (Yes), "handoff to 04-implementing" (No); "classifySeverity" → "TRIVIAL"; "classifySeverity" → "FULL"; "TRIVIAL" → "skip specialist review (0 specialists)"; "FULL" → "dispatch 1 specialist with shared artifact"; "skip specialist review (0 specialists)" → "green testing"; "dispatch 1 specialist with shared artifact" → "green testing".
```

## Task Packet

Pass a compact packet that lists the changed files and asks for the severity
classification.

```text
Use specialist-review-workflow to classify a fix.
Changed files:
  - testing/architecture/network.test.ts
  - testing/architecture/network.utils.test.ts
Gate: scripts/agent-customization/gates/specialist-review-severity.gate.mjs
Question: How many specialists should the orchestrator dispatch before green testing?
```

## Required Workflow

1. Collect the repository-relative paths of all files changed in the fix or
   implementation slice.
2. Run the shared validation gate once for the changed files:
   ```bash
   node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=path1,path2,...
   ```
3. Read the returned gate contract. If `pass` is `false`, stop specialist
   dispatch and hand the failing runner output back to `04-implementing`.
4. Run the severity classifier:
   ```bash
   node scripts/agent-customization/gates/specialist-review-severity.gate.mjs --json --input=path1,path2,...
   ```
5. Read the returned `severity` field:
   - `TRIVIAL` — the fix changes only test files, Markdown/JSDoc surfaces,
     formatting configuration, or lock files.
   - `FULL` — at least one file touches runtime logic under `src/`,
     `examples/`, or `benchmarks/`.
6. Dispatch the appropriate specialist review panel:
   - `TRIVIAL`: dispatch **0** specialists — skip specialist review entirely.
     The shared-validation artifact is sufficient. Proceed directly to
     green testing.
   - `FULL`: dispatch **1** Tier-3 specialist (usually
     `implementation-pattern-scout` or a domain-specific scout) for a focused
     code-quality and pattern check.
7. Record the classification, the shared-validation artifact path, and the
   dispatched specialist list in the plan's `VALIDATION_EVIDENCE` section before
   dispatching `05-green-testing`.
8. If any specialist returns `REQUEST_CHANGES`, compile observations into a
   fix packet, dispatch a fresh `04-implementing` instance, and re-run the
   shared-validation gate on the new changed-file set before re-review.

## Shared Validation and Perspective Review

Before any specialist reviews code, run the shared-validation gate exactly once.
The gate executes the focused test set, the build, and lint for the changed
files and writes a JSON artifact (default `artifacts/shared-validation.json`).
Every specialist in the review panel receives the same artifact and uses it as
their validation baseline.

### What specialists do NOT repeat

- Re-running the full test suite, build, or lint independently.
- Re-deriving changed files or test coverage from scratch.

### What specialists DO add

Each specialist applies their own perspective to the changed files and the
shared artifact:

- `implementation-pattern-scout` — naming conventions, module boundaries,
  duplication, and repo pattern alignment.
- `performance-trace-specialist` — runtime cost, algorithmic complexity, and
  hot-path regressions (when applicable).
- `coverage-scout` — whether the shared artifact shows adequate coverage and
  whether tests exercise the changed behavior.
- Domain scouts (e.g., `browser-runtime-scout`, `worker-payload-scout`) —
  environment-specific correctness and integration fit.

The shared artifact lets reviewers focus on perspective-specific judgment instead
of redundant validation.

## Severity Rules

### TRIVIAL classification

A fix is classified `TRIVIAL` when **every** changed file falls into one of
these categories by path inspection:

- Test files: `*.test.{ts,js,mjs,cjs}` or `*.spec.{ts,js,mjs,cjs}`.
- Documentation/Markdown files: `*.md` (including JSDoc-bearing source files
  when the diff is confirmed to be JSDoc-only by the orchestrator).
- Formatting configuration: `.prettierrc*`, `.prettierignore`,
  `.editorconfig`, `.gitattributes`, `.eslintignore`, `.gitignore`, `.npmrc`,
  `.nvmrc`.
- Lock files: `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`.

### FULL classification

A fix is classified `FULL` when any changed file:

- Lives under `src/` and is not a test or doc file.
- Lives under `examples/` and is not a test or doc file.
- Lives under `benchmarks/` and is not a test or doc file.
- Contains executable runtime logic that is not matched by a trivial pattern.

Mixed changes (e.g., `src/foo.ts` plus `src/foo.test.ts`) are always `FULL`,
but the classifier partitions the trivial and non-trivial files for the
orchestrator's record.

### Orchestrator override

The classifier is a path-based heuristic. The orchestrator may override the
classification when the actual diff shows that a source file change was purely
JSDoc or that a formatting-only change touched a non-trivial path. The override
must be recorded in `VALIDATION_EVIDENCE` with the reason.

## Decision Tree

```text
Flowchart summary: "Fix ready for review" → "Run shared-validation.gate.mjs"; "Run shared-validation.gate.mjs" → "pass?"; "pass?" → "Run classifySeverity" (Yes), "Return to 04-implementing" (No); "Run classifySeverity" → "TRIVIAL?"; "TRIVIAL?" → "Skip specialist review (0 specialists)" (Yes), "Dispatch 1 specialist with shared artifact" (No); "Skip specialist review (0 specialists)" → "Record evidence" → "Green testing"; "Dispatch 1 specialist with shared artifact" → "Record evidence" → "Green testing".
```

## Before / After Examples

**Before:**

```text
Issue: 3+ specialists were dispatched for a one-line test fixture change.
Result: wasted review cycles on low-risk edits.
```

**After:**

```text
Changed files: testing/foo.test.ts
Classification: TRIVIAL
Action: skip specialist review entirely — shared-validation artifact is sufficient.
Result: faster feedback, same quality bar for non-trivial changes preserved.
```

## Guardrails

- Do not classify a fix as TRIVIAL when any changed file touches runtime
  logic under `src/`, `examples/`, or `benchmarks/`.
- Do not dispatch more than 1 specialist for any slice. The balanced policy
  caps specialist review at 1 reviewer for FULL changes and 0 for TRIVIAL.
- Do not rely solely on path inspection when the diff is ambiguous; use the
  override mechanism and document the reason.
- Do not let any specialist re-run tests, build, or lint independently — use
  the shared-validation gate once and pass its artifact to the reviewer.
- Do not prepend calendar dates to plan headings or session logs.

## Expected Final Output

A strong specialist-review workflow pass should report:

- the changed files classified,
- the returned severity (`TRIVIAL` or `FULL`),
- the number and names of specialists dispatched,
- the override reason if the orchestrator deviated from the classifier,
- the recorded `VALIDATION_EVIDENCE` location.
