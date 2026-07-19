---
name: red-test-contracts
description: 'Use when: designing red-phase failing tests, contracts, or mocks.'
argument-hint: 'Describe the behavior change, target file or skill, expected failing assertion, and validation command.'
user-invocable: false
disable-model-invocation: false
skills:
  - creating-unit-tests
  - test-fix-workflow
  - coverage-guard
  - coverage-tranche
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Red Test Contracts

Use this skill before implementation changes that alter observable behavior.
The red phase exists to make the target behavior explicit as a failing assertion
before any production code changes.

This skill owns the workflow for identifying the smallest testable contract,
authoring the failing test in the correct owner-local file, confirming it
actually fails, and handing off cleanly to the implementation phase.

## When to Use

- A behavior change is planned and no failing test yet exists that targets it.
- An existing test needs to be updated to reflect the new expected behavior
  before the implementation is changed.
- A coverage gap has been identified and the missing branch needs a red-phase
  test before a production edit removes dead code.
- A phase agent is about to implement something but has not yet confirmed the
  red contract is in place.
- A plan file records a step that requires red-phase evidence and that evidence
  is missing or stale.
- Multiple tests are failing and the repair scope needs to be narrowed to the
  smallest honest failing assertion first.

## When NOT to use

Do NOT use for writing unit tests - use `creating-unit-tests` instead. Do NOT use for fixing failing tests - use `test-fix-workflow` instead.

## Workflow Diagram

```text
Flowchart summary: "Feature to test" → "Define expected behavior"; "Define expected behavior" → "Write failing test"; "Write failing test" → "Run test"; "Run test" → "Fails for right reason?"; "Fails for right reason?" → "Red contract complete" (Yes), "Fix test fixture" (No, fails for wrong reason), "Check if test is meaningful" (No, passes already); "Red contract complete"; "Fix test fixture" → "Run test"; "Check if test is meaningful" → "Add edge case or remove"; "Add edge case or remove".
```

## Task Packet

Pass a compact packet naming the target behavior, the file to test, the expected
failing assertion, and the validation command.

```text
Use red-test-contracts for network topology utils uncovered branch.
Target: src/architecture/network/topology/network.topology.utils.ts line 28.
Expected behavior: throws RangeError when node count is zero.
Owner-local test file: src/architecture/network/topology/network.topology.utils.test.ts.
Validate with: npx jest --config=jest.config.mjs --no-cache --testPathPattern=network.topology.utils
```

## Required Workflow

1. Identify the smallest observable behavior that should fail before the fix.
   Start from the uncovered branch, the missing public contract, or the plan's
   stated invariant — not from the full function.
2. Add or update the narrowest owner-local test or eval assertion first. Prefer
   the existing test file for the module under test; do not create a broad
   integration test when an owner-local unit test is possible.
3. Follow AAA structure: Arrange the fixture, Act on the unit under test, Assert
   the expected outcome. Keep each section concise.
4. Use nested `describe` blocks to mirror the module structure: outer describe
   names the module, inner describe names the function or scenario, `it` names
   the specific contract.
5. Write exactly one top-level `expect(...)` per test. Multiple independent
   assertions belong in separate `it` blocks.
6. Prefer deterministic fixtures over broad integration runs during the red
   phase. Use seeded values, small static networks, or mocked dependencies
   rather than pulling in large live datasets.
7. Run the focused test to confirm it actually fails for the right reason:
   `npx jest --config=jest.config.mjs --no-cache --testPathPattern=<file>`
8. Record the failing command in the active plan step or note why it cannot
   be run if the environment blocks it.
9. Hand off to the implementation phase only after the red contract is confirmed.

## Coordination

- Use `test-fix-workflow` when multiple tests are already failing and the suite
  needs repair rather than a new red contract.
- Use `coverage-tranche` when the goal is expanding coverage on passing code,
  not writing a new failing assertion first.
- Use `coverage-guard` after any `src/` change to confirm 100% coverage is
  maintained on every touched file.

## Test Pattern Examples

**Behavior-first contract:**

```ts
it('throws when hiddenLayers is empty', () => {
  expect(() => buildMLP({ hiddenLayers: [] })).toThrow(/hiddenLayers/);
});
```

**Determinism contract:**

```ts
it('produces identical output for same config and seed', () => {
  const net1 = buildGRU({ units: 4, seed: 42 });
  const net2 = buildGRU({ units: 4, seed: 42 });
  expect(net1.toJSON()).toEqual(net2.toJSON());
});
```

**Shape contract:**

```ts
it('produces correct node count for 3-layer MLP', () => {
  const net = buildMLP({ hiddenLayers: [4, 4], inputSize: 2, outputSize: 1 });
  expect(net.nodes.length).toBe(7);
});
```

## Decision Tree

```text
Flowchart summary: "Test work" → "What kind?"; "What kind?" → "red-test-contracts (create red contract)" (No failing test yet exists), "test-fix-workflow" (Tests already failing, need repair), "coverage-tranche" (Passing code needs more coverage), "coverage-guard" (Post-change coverage gate); "red-test-contracts (create red contract)"; "test-fix-workflow"; "coverage-tranche"; "coverage-guard".
```

## Before / After Examples

**Before:**

```ts
it('should work', () => {
  expect(fn()).toBeDefined();
});
```

**After:**

```ts
it('throws RangeError when node count is zero', () => {
  expect(() => buildMLP({ hiddenLayers: [] })).toThrow(/hiddenLayers/);
});
```

## Guardrails

- Do not write a test that is trivially impossible to fail; the test must
  genuinely fail before the implementation change.
- Do not write a test that passes immediately because it tests the existing
  (incorrect) behavior; the red phase requires a test that expresses the
  desired future behavior.
- Do not add multiple top-level `expect(...)` calls in one `it`; each
  independent contract needs its own `it` block.
- Do not reach for broad integration runs when a focused owner-local test can
  express the same contract.
- Do not skip the focused run confirmation; a test that does not actually fail
  is not a red contract.

## Expected Final Output

A strong red-test pass should produce:

- the new or updated failing test in the owner-local test file,
- the focused Jest run output confirming the test fails for the expected reason,
- the failing command recorded in the active plan step,
- a clear handoff note naming the behavior that must be implemented to make the
  test pass.
