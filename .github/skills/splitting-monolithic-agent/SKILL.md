---
name: splitting-monolithic-agent
description: 'Use when: splitting an overloaded custom agent into an orchestrator, sub-orchestrator, specialists, reusable skills, and explicit output contracts.'
argument-hint: 'Describe the source agent, broad responsibilities, desired compatibility surface, candidate specialists, skills to extract, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Splitting Monolithic Agent

Use this skill when an agent carries too much context or too many responsibilities.

Rules:
- Preserve the original user-visible API unless the user explicitly approves a rename.
- Move coordination into an orchestrator or sub-orchestrator.
- Move narrow, repeatable, tool-specific jobs into hidden specialists.
- Move reusable procedure into skills.
- Keep delegation bounded; avoid `agents: '*'`.
- Add eval or validation coverage for the new routing boundary.

Return split map, files changed, compatibility decision, validation commands, and residual manual-review items.