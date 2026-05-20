---
name: red-test-contracts
description: 'Design red-phase tests and eval assertions for NeatapticTS before behavior changes. Use when a phase agent needs failing tests, AAA structure, single top-level expectation, nested describe blocks, coverage targets, or eval assertions before implementation.'
argument-hint: 'Describe the behavior change, target file or skill, expected failing assertion, and validation command.'
user-invocable: false
disable-model-invocation: false
---

# Red Test Contracts

Use this skill before implementation changes that alter behavior.

## Workflow

1. Identify the smallest observable behavior that should fail before the fix.
2. Add or update the narrowest owner-local test or eval assertion first.
3. Follow AAA structure, nested `describe`, and one top-level `expect(...)` per test.
4. Prefer deterministic fixtures over broad integration runs during the red phase.
5. Record the failing command or reason it cannot be run in the active plan.
6. Hand off to implementation only after the red contract is clear.

## Coordination

- Use `test-fix-workflow` for multiple failing tests.
- Use `coverage-tranche` when expanding passing coverage.