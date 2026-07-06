---
name: planning-acceptance-criteria
description: 'Use when: turning user intent into acceptance criteria and scope boundaries.'
argument-hint: 'Describe the user request, target surface, edge cases, non-goals, and validation or done-state expectations.'
user-invocable: false
disable-model-invocation: false
skills:
  - plan-alignment
  - red-test-contracts
  - tracker-handoff
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Planning Acceptance Criteria

This skill converts user intent into a precise, testable acceptance criteria set. Acceptance criteria are the unit tests for English that turn intent into observable, implementation-agnostic contracts. Each criterion is tied to an observable behavior or validation method, non-goals are made explicit to prevent scope creep, and open assumptions are surfaced before implementation begins.

## When to Use

- At the start of a planning session before implementation, to ensure the done-state is unambiguous.
- When a user request is broad or open-ended and implementation could proceed in multiple directions.
- Before writing a tracker entry to confirm that the task has measurable completion criteria.
- When a prior implementation did not satisfy the user's intent and you need to clarify expectations before retrying.
- When preparing a handoff packet for an implementation agent that needs to know exactly what "done" looks like.
- When scope is likely to widen during implementation and explicit non-goals are needed to contain it.

## When NOT to use

Do NOT use for plan alignment checking - use `plan-alignment` instead. Do NOT use for red test contracts - use `red-test-contracts` instead.

## Workflow Diagram

```text
Flowchart summary: "Feature request" → "Identify observable behaviors"; "Identify observable behaviors" → "Write acceptance criteria"; "Write acceptance criteria" → "Can it be split?"; "Can it be split?" → "Decompose into slices" (Yes), "Keep as single step" (No); "Decompose into slices" → "Define each slice boundary"; "Keep as single step" → "Proceed to red testing"; "Define each slice boundary" → "Proceed to red testing"; "Proceed to red testing".
```

## Task Packet

Include the user request, the target surface, suspected edge cases, explicit non-goals, and how completion should be validated.

```text
Use planning-acceptance-criteria for <brief task description>.
Target surface: <files, API, or behavior being changed>
Edge cases: <list of conditions that might be overlooked>
Non-goals: <what is explicitly out of scope for this task>
Validation method: <command, manual check, or observable output>
Done-state: <what must be true for this to be complete>
```

## Required Workflow

1. Restate the user intent in your own words to confirm understanding before defining criteria.
2. Write each acceptance criterion as an observable behavior: "Given X, when Y, then Z" or an equivalent concrete statement. Optionally give each acceptance criterion a stable, unique `id: AC-###` (for example `id: AC-001`). The `id` field is optional for acceptance criteria but recommended for any criterion that maps to a specific file change or validation command, so it can be referenced from traceability tables, gate evidence, and phase-compression notes.
3. Tie each criterion to a validation method: a Jest command, a `node` script, a manual check, or a visible output.
4. Add at least one edge-case criterion for each failure mode or boundary condition that is easy to overlook.
5. Write an explicit non-goals list: capabilities that are adjacent to the task but should not be addressed in this pass.
6. Identify open assumptions: decisions that depend on user preference or environmental state that cannot be determined from the task description alone.
7. Keep the criteria concise enough to fit into a handoff packet; aim for five to ten well-scoped criteria rather than an exhaustive checklist.
8. Surface open assumptions to the user or record them in the tracker before implementation proceeds.

## No Deferred Cleanup — Mandatory Acceptance Criterion

For any step, slice, or task that involves migration, refactoring, or API
replacement, the acceptance criteria MUST include a criterion verifying that
old code is removed in the same step:

> "The old API/implementation is fully removed (no backward-compatibility
> wrappers, no dual-path code, no deferred cleanup) in this step."

This is a non-negotiable acceptance criterion. A step that introduces new code
alongside old code without removing the old code MUST NOT pass acceptance
review. The criterion MUST be observable: cite the specific files or exports
that were deleted, not just "old code removed."

## Before/After Examples: Vague to Precise

**Before (vague, no numbered identifier):**

```md
- The builder should work correctly.
```

**After (precise, with optional AC-### IDs):**

```md
- id: AC-001
  text: buildMLP() with default config produces a network with exactly 5 nodes
  (2 inputs, 2 hidden, 1 output) and 6 connections.
  validation: npx jest --testPathPattern=builders/mlp
- id: AC-002
  text: buildMLP() with empty hiddenLayers throws an error naming the field.
  validation: npx jest --testPathPattern=builders/mlp
- id: AC-003
  text: Same config + same seed produces identical network shape.
  validation: npx jest --testPathPattern=builders/mlp
```

## Decision Tree: Splitting Work

```text
Flowchart summary: "Acceptance criteria" → "Multiple independent behaviors?"; "Multiple independent behaviors?" → "Split into slices" (Yes), "Single step" (No); "Split into slices" → "Each slice has its own files_to_change"; "Single step"; "Each slice has its own files_to_change" → "Each slice has its own acceptance criteria"; "Each slice has its own acceptance criteria" → "Define parallelizable flag"; "Define parallelizable flag".
```

## Guardrails

- Do not write implementation-wish criteria ("use X algorithm"); prefer observable behavior criteria ("given input Y, output Z is produced").
- Do not omit non-goals when the task scope is ambiguous; explicit non-goals prevent silent scope creep.
- Do not include more than ten criteria without splitting the task; large criteria sets indicate the task needs decomposition.
- Do not proceed to implementation recommendations within this skill; this skill outputs criteria only.
- Do not tie a criterion to a validation method that does not exist yet; if the validation command needs to be written, note it as a dependency.

## Expected Final Output

- A numbered acceptance criteria list: each criterion observable, each tied to a validation method.
- An explicit non-goals list.
- A list of open assumptions with a recommended resolution action for each.
- Criteria compact enough to be included in a tracker entry or handoff packet.
