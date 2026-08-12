---
name: creating-unit-tests
description: 'Use when: writing focused unit tests, fixtures, mocks, or coverage-aware tests.'
argument-hint: 'Describe the behavior under test, owner-local test file or folder, expected red/green state, and focused command.'
user-invocable: false
disable-model-invocation: false
skills:
  - red-test-contracts
  - coverage-tranche
  - coverage-guard
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Creating Unit Tests

This skill writes the smallest test that proves a specific behavior in the NeatapticTS codebase. It follows the TDD order (red → implement → green), respects the repo's relaxed single-expect rule, and places tests in owner-local files near the source boundary being tested.

The relaxed rule: an `it()` block may contain up to three related `expect(...)` calls when they all verify the same behavior state (for example, the same returned object or the same side effect). Unrelated assertions still belong in separate tests.

## When to Use

- Writing a red test to define expected behavior before implementing a fix.
- Adding a focused unit test for an uncovered code path identified by `coverage/lcov.info`.
- Creating test fixtures or mocks for a scoped behavior that currently has no test.
- Verifying that a new implementation detail is exercised by at least one assertion.
- Expanding coverage for a specific file to reach 100% as part of a coverage tranche.
- Writing a regression test for a bug that was fixed so it cannot silently recur.

## When NOT to use

Do NOT use for red test contract design - use `red-test-contracts` instead. Do NOT use for coverage gap analysis - use `coverage-tranche` instead.

## Workflow Diagram

```text
Flowchart summary: "Need test for code path" → "Find owner-local test file"; "Find owner-local test file" → "Write single it() block"; "Write single it() block" → "Up to 3 related expect() calls on same state"; "Up to 3 related expect() calls on same state" → "Run focused slice"; "Run focused slice" → "Pass?"; "Pass?" → "Done" (Yes), "Fix test or code" (No); "Done"; "Fix test or code" → "Run focused slice".
```

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
4. Write the test using the repo's relaxed single-expect rule: each `it()` block should answer one behavioral question. You may use up to three related top-level `expect(...)` calls when they all verify the same behavior state; unrelated assertions belong in separate `it()` blocks.
5. Place the test in an owner-local file near the source boundary (e.g., `src/neat/mutation/` tests alongside `src/neat/mutation/` source).
6. Follow existing framework conventions: describe block naming, beforeEach setup, mock/stub patterns already used in sibling tests.
7. Run the focused Jest command to confirm the expected initial state (red if TDD, green if verifying existing behavior).
8. Do not run the full suite until the focused test is in the expected state and the user or active step packet explicitly requires a repo-wide run.
9. Report the command run, exit status, and the expected red/green state.

## Why the Relaxed Single-Expect Rule Exists

Each `it()` block should answer one behavioral question. When unrelated assertions share a test, a failure in the first assertion can mask failures in later ones and makes the failing contract harder to read. The relaxed rule keeps the focus on one behavior state while allowing a small number of related checks (up to three) that describe the same state from slightly different angles — for example, verifying both the value and type of a single returned object. Group by scenario and state, not by assertion count.

## Test Pattern Examples

**AAA (Arrange-Act-Assert):**

```ts
it('returns sorted array', () => {
  const input = [3, 1, 2]; // Arrange
  const result = input.toSorted(); // Act
  expect(result).toEqual([1, 2, 3]); // Assert
});
```

**Single-expect:**

```ts
it('does not mutate the original array', () => {
  const original = [3, 1, 2];
  original.toSorted();
  expect(original).toEqual([3, 1, 2]);
});
```

**Up to three related assertions on the same state:**

```ts
it('expires a token exactly at the threshold', () => {
  const token = makeToken({ ttl: 1000 });
  jest.advanceTimersByTime(999);
  expect(token.isActive()).toBe(true);
  jest.advanceTimersByTime(1);
  expect(token.isActive()).toBe(false);
  expect(token.expiredAt).toBe(1000);
});
```

**Owner-local (use nearest existing test file):**

```ts
// In testing/architecture/network/builders/gru.test.ts
it('produces deterministic output under fixed seed', () => {
  const net1 = buildGRU({ units: 4 });
  const net2 = buildGRU({ units: 4 });
  expect(net1.nodes.length).toBe(net2.nodes.length);
});
```

## Decision Tree

```text
Flowchart summary: "Need test for code path" → "What is missing?"; "What is missing?" → "Write new it() in owner-local file" (No test exists), "Extract shared fixture" (Setup duplication), "Add mock or stub" (Dependency not isolated), "Clarify related assertions" (Assertion grouping unclear); "Write new it() in owner-local file" → "Run focused slice"; "Extract shared fixture" → "Run focused slice"; "Add mock or stub" → "Run focused slice"; "Clarify related assertions" → "Run focused slice"; "Run focused slice".
```

## Before / After Examples

**Before:**

```ts
it('should work', () => {
  const token = makeToken({ ttl: 1000 });
  expect(token.issuedAt).toBe(0);
  jest.advanceTimersByTime(999);
  expect(token.isActive()).toBe(true);
  jest.advanceTimersByTime(1);
  expect(token.isActive()).toBe(false);
  expect(token.expiredAt).toBe(1000);
});
```

**After:**

```ts
it('records the original issue timestamp', () => {
  const token = makeToken({ ttl: 1000 });
  expect(token.issuedAt).toBe(0);
});

it('expires exactly at the threshold', () => {
  const token = makeToken({ ttl: 1000 });
  jest.advanceTimersByTime(999);
  expect(token.isActive()).toBe(true);
  jest.advanceTimersByTime(1);
  expect(token.isActive()).toBe(false);
  expect(token.expiredAt).toBe(1000);
});
```

## Guardrails

- Do not use broad snapshots unless the project already uses them for the target surface.
- Do not write more than three top-level `expect(...)` calls in a single `it()` block, and do not mix unrelated assertions in one test; split independent contracts into separate `it()` blocks.
- Do not place tests in a shared or unrelated folder; keep them owner-local near the source boundary.
- Do not run `npm test` or any full-suite command (`npm run test:silent`, `npm run jest:esm-ts`, `npm run jest:mjs`) speculatively. Use the targeted Jest command first. Only run the full suite when the user or active step packet explicitly requires it.
- Do not use single-letter local variable names except `i` and `j` in trivial loops; match the descriptive naming style of the codebase.
- Do not write a test to force an unreachable path; if a branch cannot be reached, remove it from production code instead.

## Expected Final Output

- A new or updated owner-local test file with one or more focused `it()` blocks, each answering one behavioral question. Each `it()` may contain up to three related top-level `expect(...)` calls when they verify the same behavior state.
- The focused Jest command passes in the expected state (red before implementation, green after).
- Behavior covered, test file path, focused command, and red/green status reported in the session output.
