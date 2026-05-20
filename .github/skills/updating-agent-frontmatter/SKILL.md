---
name: updating-agent-frontmatter
description: 'Use when: updating .agent.md YAML frontmatter, names, descriptions, tools, agents allow-lists, model fallback arrays, handoffs, or visibility flags.'
user-invocable: false
disable-model-invocation: false
---

# Updating Agent Frontmatter

Use this skill to make safe custom-agent metadata edits.

Rules:
- Keep public agent names stable and intentional.
- Use `user-invocable: false` for internal specialists and auxiliaries.
- Prefer bounded `agents` allow-lists and include `agent` in tools when subagents are listed.
- Use qualified model names and fallback arrays.
- Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after edits.

Return files changed, metadata changed, validation command, and compatibility risk.