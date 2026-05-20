---
name: agent-frontmatter-standards
description: 'Validate and design VS Code custom agent frontmatter for NeatapticTS. Use when creating or editing .agent.md files, setting tools, agents, model, user-invocable, disable-model-invocation, handoffs, or diagnosing silent customization loading failures.'
argument-hint: 'Describe the agent file or phase, visibility target, allowed subagents, model routing, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Agent Frontmatter Standards

Use this skill when a task changes `.github/agents/*.agent.md`.

## Workflow

1. Read `plans/Agentic_Workflow_Architecture.plans.md` first and update it after the step.
2. Keep exactly seven user-facing phase agents as the final target.
3. Hide specialists with `user-invocable: false` while keeping them callable by listed parent agents.
4. Use explicit `agents: [...]` allow-lists on phase agents; avoid unrestricted delegation.
5. Use qualified model strings or fallback arrays validated by `model-routing-and-budget`.
6. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after edits.
7. Run strict validation only when a phase claims the final seven-agent surface is ready.

## Gotchas

- YAML frontmatter failures can be silent. Prefer single-line quoted descriptions and explicit booleans.
- `agents: []` means no subagents. Omitted `agents` can allow broader delegation in VS Code.
- Do not copy durable workflow policy into agent bodies; put it in skills and have the agent invoke the skill.

## Sources

- VS Code custom agents documentation defines `.agent.md` fields such as `model`, `agents`, `handoffs`, `user-invocable`, and `disable-model-invocation`.
- The repo tracker is the authoritative local policy for the seven-phase architecture.