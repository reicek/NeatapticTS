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

1. Read the active customization tracker when one is open; otherwise keep edits scoped to the requested customization surface.
2. Keep exactly eight user-facing SDLC orchestrators as the final target: `00-helping`, `01-planning`, `02-researching`, `03-red-testing`, `04-implementing`, `05-green-testing`, `06-documenting`, and `07-logging`.
3. Hide specialists with `user-invocable: false` while keeping them callable by listed parent agents.
4. Use explicit `agents: [...]` allow-lists on phase agents; avoid unrestricted delegation.
5. Use qualified model strings or fallback arrays validated by `model-routing-and-budget`.
6. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after edits.
7. Run strict validation only when the eight-agent SDLC surface is expected to be ready.

## Gotchas

- YAML frontmatter failures can be silent. Prefer single-line quoted descriptions and explicit booleans.
- `agents: []` means no subagents. Omitted `agents` can allow broader delegation in VS Code.
- Do not copy durable workflow policy into agent bodies; put it in skills and have the agent invoke the skill.

## Sources

- VS Code custom agents documentation defines `.agent.md` fields such as `model`, `agents`, `handoffs`, `user-invocable`, and `disable-model-invocation`.
- The archived meta-workflow tracker records the baseline architecture; active customization work should follow the current eight-orchestrator surface in this skill.