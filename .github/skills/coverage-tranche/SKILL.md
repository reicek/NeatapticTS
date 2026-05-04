---
name: coverage-tranche
description: 'Systematically expand test coverage for a specific source boundary in NeatapticTS toward 100% statements, branches, functions, and lines. Use when a source file has a known coverage gap and passing tests, not for fixing test failures.'
argument-hint: 'Name the target source file, provide the current coverage % or uncovered line count from lcov.info or a focused run, and state whether this is reconnaissance, implementation, or validation.'
user-invocable: true
disable-model-invocation: false
---

# Coverage Tranche Playbook

Use this skill to bring a specific `src/` boundary from its current coverage
level up to the repo-wide 100% requirement.

**100% coverage across all four categories — statements, branches, functions,
lines — is a hard requirement for every file in `src/`, not a goal to
approach.** This skill is the structured workflow for closing gaps that exist
on already-passing code. For post-change regression repair use `coverage-guard`
instead; for failing tests use `test-fix-workflow` first.

When this skill updates the coverage tracker, `tracker-handoff` owns the
`.plans.md` and `Handoff query` shape.

## Skill Relationships

| Situation | Correct skill |
|---|---|
| Tests are failing | `test-fix-workflow` first, then `coverage-guard` |
| Tests pass, file below 100% | `coverage-tranche` (this skill) |
| Change just landed, verify no regression | `coverage-guard` |
| Identify next file below 100% | `Coverage Scout` agent |

Do not invoke this skill when tests are red. Fix failures first, then
return to coverage expansion.

## When to Use

- A source file is below 100% in one or more coverage categories.
- The test suite is green.
- The task is to add the **smallest** owner-local test that exercises an
  uncovered path — not to fix a broken assertion.
- Dead code should be confirmed and removed as part of the tranche.

## Task Packet

Pass a compact packet that includes:

- target source file path,
- current coverage metrics (from `coverage/lcov.info` or a focused run),
- uncovered line or branch count,
- most recent green baseline (suite count, passing state),
- the plan file to update (`plans/test-repair-and-coverage.plans.md`),
- whether this is reconnaissance, implementation, or validation.

Compact example:

```text
Use coverage-tranche for src/neat/diversity/core/diversity.core.ts.
Coverage: 96.55% lines (28/29). One uncovered branch.
Baseline: 296 suites / 2570 tests green.
Plan: plans/test-repair-and-coverage.plans.md.
Mode: implementation.
```

## Required Workflow

1. Read the target source file in full.
2. Read the nearest existing test file for that boundary.
3. Identify the uncovered path from `coverage/lcov.info` or a focused run:
   ```bash
   npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=<file>
   ```
4. Classify the uncovered path:
   - **Reachable path:** add the smallest test that exercises it.
   - **Dead code:** remove the dead production branch; do not write a test to
     force an unreachable path.
5. Add the test (if reachable) to the nearest existing test file for that
   boundary — never create a new test file unless no test file exists for the
   boundary.
6. Validate with a focused Jest slice for the single file:
   ```bash
   npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=<file>
   ```
   Confirm 100% across all four categories for the target file.
7. Run the repo-wide suite to confirm no regressions:
   ```bash
   npm run test:silent
   ```
8. Update `plans/test-repair-and-coverage.plans.md` with the completed tranche:
   - Mark the tranche `[DONE]`.
   - State the new coverage baseline (suite count, test count).
   - Identify the next target from `coverage/lcov.info` ordering.
   - Refresh the `Handoff query` section to name the next target.

## Dead Code Rule

When a test exposes a branch that no combination of legal inputs can reach:

- **Remove the dead production branch** instead of writing a contorted test.
- Note the removal in the plan so the decision is recorded.
- Dead code removal is a correctness improvement, not just a cosmetic fix —
  several real bugs and bookkeeping gaps have been found and fixed this way in
  this repo.

## Single-Expect Rule

Each new `it()` block must have **exactly one top-level `expect(...)`**.
Group by scenario, not by assertion: one `it()` per observable behavior.

## Validation Rules

- During the tranche: focused Jest slice for the target file only.
- After the tranche: `npm run test:silent` for repo-wide green confirmation.
- Do not run the broad suite during implementation; run only the focused slice.

## Companion Agent and Sibling Skill

Use `Coverage Scout` (`coverage-scout`) when you need to identify the next
coverage tranche target from `coverage/lcov.info` before starting a tranche.
The scout is read-only recon; this skill is the execution workflow.

Use `coverage-guard` when a code change has just landed and you need to verify
the touched files have not dropped below 100%. `coverage-tranche` is for
forward progress; `coverage-guard` is for regression prevention after changes.

## Guardrails

- Do not create a new test file when an existing owner-local test file exists.
- Do not write a test whose sole purpose is to inflate a coverage number.
  If an uncovered path is dead code, remove the production branch.
- Do not skip `npm run test:silent` after the focused tranche succeeds.
- Do not mark a tranche `[DONE]` in the plan before the repo-wide suite is
  confirmed green.
- Do not modify the test runner configuration; use the existing
  `jest.config.mjs` and `--testPathPattern` flag only.
- Do not pad `plans/test-repair-and-coverage.plans.md` with verbose session
  transcripts; keep tranche entries concise.
- Do not prepend specific calendar dates to plan headings or tranche entries.
  Use stable content-focused labels instead.

## Expected Final Output

A strong tranche run should report:

- the source file processed,
- the coverage metric before and after (per category),
- whether the gap was a live path (test added) or dead code (branch removed),
- the focused Jest validation result (100% confirmed in all four categories),
- the repo-wide suite result (`npm run test:silent` green),
- the new green baseline (suite count, test count),
- the updated plan state and next target.

A tranche is only complete when the focused Jest slice confirms 100% in all
four categories **and** the repo-wide suite is green. Partial results are not
acceptable.
