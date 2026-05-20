---
name: capturing-learning-event
description: 'Use when: recording an ISO-42001-style local evidence event for agent-system gaps, agent updates, skill updates, routing updates, model changes, or output-contract fixes.'
user-invocable: false
disable-model-invocation: false
---

# Capturing Learning Event

Use this skill to append compact, public-project-friendly learning evidence.

Schema:
```json
{"timestamp":"<ISO timestamp>","eventType":"agent-system-gap|agent-update|skill-update|routing-update|output-contract-fix","triggeringTask":"<brief>","gap":"<what was missing>","resolution":"<what changed>","filesChanged":["<path>"],"agentsAffected":["<agent-name>"],"skillsAffected":["<skill-name>"],"confirmation":"not-required|user-confirmed|deferred","resumeAction":"<how work continued>"}
```

Rules:
- Do not claim certification or compliance.
- Prefer `.github/ai-learning/learning-log.jsonl` for append-only events.
- Keep events factual and concise.

Return learning status, file changed, event type, and resume action.