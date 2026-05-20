---
name: triaging-test-failures
description: 'Use when: a validation command fails and the workflow needs failure ownership, root-cause grouping, reroute decisions, or unrelated-failure separation.'
user-invocable: false
disable-model-invocation: false
---

# Triaging Test Failures

Use this skill to interpret failing validation without widening the fix scope.

Rules:
- Identify the smallest owner boundary for each failure.
- Classify failures as active-change, pre-existing, environment, flaky, or unknown.
- Recommend the next agent or command, not a broad refactor.
- Preserve raw evidence only as concise command/result snippets.

Return failure groups, likely owner, reroute target, and next validation command.