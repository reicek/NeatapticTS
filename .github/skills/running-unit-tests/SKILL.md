---
name: running-unit-tests
description: 'Use when: running focused unit tests, confirming red or green status, choosing Jest command scope, or summarizing bounded test output.'
argument-hint: 'Provide the focused command or test path, expected red/green state, and whether output should be summarized or rerouted.'
user-invocable: false
disable-model-invocation: false
---

# Running Unit Tests

Use this skill to validate a narrow test boundary before widening.

Rules:
- Prefer focused Jest commands before broad suite runs.
- Report command, exit status, and the smallest meaningful failure summary.
- Separate unrelated pre-existing failures from failures caused by the active change.
- Do not claim green until the requested command passes.

Return commands run, pass/fail evidence, reroute recommendation, and residual risk.