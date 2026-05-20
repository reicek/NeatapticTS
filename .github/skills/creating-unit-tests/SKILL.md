---
name: creating-unit-tests
description: 'Use when: writing focused unit tests, red tests, failing tests, test fixtures, mocks, assertions, or coverage for a scoped change.'
argument-hint: 'Describe the behavior under test, owner-local test file or folder, expected red/green state, and focused command.'
user-invocable: false
disable-model-invocation: false
---

# Creating Unit Tests

Use this skill to create the smallest test that proves a behavior.

Rules:
- Follow the nearest existing test framework and file conventions.
- Prefer owner-local tests near the source boundary.
- Keep red tests narrow enough that the intended implementation is obvious.
- Preserve the repo single-expect rule for Jest tests.
- Avoid broad snapshots unless the project already uses them for the target surface.

Return files changed, behavior covered, command to run, and any known red/green expectation.