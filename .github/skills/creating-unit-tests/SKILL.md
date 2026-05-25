---
name: creating-unit-tests
description: 'Use when: writing focused unit tests, red tests, failing tests, test fixtures, mocks, assertions, or coverage for a scoped change.'
argument-hint: 'Describe the behavior under test, owner-local test file or folder, expected red/green state, and focused command.'
user-invocable: false
disable-model-invocation: false
---

# Creating Unit Tests

This skill writes the smallest test that proves a specific behavior in the NeatapticTS codebase. It follows the TDD order (red → implement → green), respects the repo's single-expect rule, and places tests in owner-local files near the source boundary being tested.

## When to Use

- Writing a red test to define expected behavior before implementing a fix.
- Adding a focused unit test for an uncovered code path identified by `coverage/lcov.info`.
- Creating test fixtures or mocks for a scoped behavior that currently has no test.
- Verifying that a new implementation detail is exercised by at least one assertion.
- Expanding coverage for a specific file to reach 100% as part of a coverage tranche.
- Writing a regression test for a bug that was fixed so it cannot silently recur.

## Task Packet

Include the behavior under test, the owner-local test file path, the expected initial state (red or green), and the focused Jest command to validate.

```text
Use creating-unit-tests for <behavior description>.
Source file: <src/path/to/source.ts>
Test file: <src/path/to/source.test.ts>
Expected state: <red (failing until implementation) | green (behavior already exists)>
Focused command: npx jest --config=jest.config.mjs --no-cache --testPathPattern=<path>
```

## Required Workflow

1. Read the source file being tested to understand the existing implementation and public surface.
2. Read the nearest existing test file for the same module to learn framework conventions, import patterns, and describe/it structure used in this codebase.
3. Identify the specific behavior or branch to test; keep each test focused on exactly one behavior.
4. Write the test using the repo's single-expect rule: each `it()` block contains exactly one top-level `expect(...)` call.
5. Place the test in an owner-local file near the source boundary (e.g., `src/neat/mutation/` tests alongside `src/neat/mutation/` source).
6. Follow existing framework conventions: describe block naming, beforeEach setup, mock/stub patterns already used in sibling tests.
7. Run the focused Jest command to confirm the expected initial state (red if TDD, green if verifying existing behavior).
8. Do not run the full suite until the focused test is in the expected state.
9. Report the command run, exit status, and the expected red/green state.

## Guardrails

- Do not use broad snapshots unless the project already uses them for the target surface.
- Do not write multiple top-level `expect(...)` calls in a single `it()` block; split into separate tests instead.
- Do not place tests in a shared or unrelated folder; keep them owner-local near the source boundary.
- Do not run `npm test` (full suite) until the focused test is in the correct state; use the targeted Jest command first.
- Do not use single-letter local variable names except `i` and `j` in trivial loops; match the descriptive naming style of the codebase.
- Do not write a test to force an unreachable path; if a branch cannot be reached, remove it from production code instead.

## Expected Final Output

- A new or updated owner-local test file with one or more focused `it()` blocks, each with a single top-level `expect(...)`.
- The focused Jest command passes in the expected state (red before implementation, green after).
- Behavior covered, test file path, focused command, and red/green status reported in the session output.
