---
name: summarizing-session-log
description: 'Use when: summarizing session activity, changed files, validation evidence, learning events, residual risks, or next-step handoff context without replaying the transcript.'
argument-hint: 'Describe the workstream, files changed, validations run, learning events, and whether this is a chat summary, tracker note, or log entry.'
user-invocable: false
disable-model-invocation: false
---

# Summarizing Session Log

Use this skill to produce compact continuity summaries.

Rules:
- Do not create a session log when the user explicitly forbids it.
- Prefer high-signal evidence: files changed, validations, blockers, and next action.
- Keep public summaries free of private or chat-only detail that does not help continuation.
- Separate durable tracker/log updates from final chat summaries.

Return summary type, files changed, validation evidence, learning event status, risks, and next action.