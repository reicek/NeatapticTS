---
name: creating-specialist-agent
description: 'Use when: creating a hidden specialist or auxiliary .agent.md for one narrow reusable job, including tools, model tier, output contract, and parent routing.'
argument-hint: 'Describe the missing specialist job, parent orchestrator, required tools, model tier, output fields, and validation commands.'
user-invocable: false
disable-model-invocation: false
---

# Creating Specialist Agent

Use this skill when a reusable workflow gap needs a new hidden agent.

Rules:
- Prefer a skill over an agent when no isolated context or tool restriction is needed.
- Default new specialists to `user-invocable: false`.
- Use the narrowest tool set and an explicit `agents: []` list unless delegation is required.
- Give every hidden agent a compact structured output contract.
- Add the new agent only to the smallest parent allow-list that needs it.
- Validate with agent frontmatter and graph scripts.

Return agent file, parent routing changes, model choice, output contract, and validation evidence.