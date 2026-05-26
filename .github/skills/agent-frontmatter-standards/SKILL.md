---
name: agent-frontmatter-standards
description: 'Validate and design VS Code custom agent frontmatter for NeatapticTS. Use when creating or editing .agent.md files, setting tools, agents, model, user-invocable, disable-model-invocation, handoffs, or diagnosing silent customization loading failures.'
argument-hint: 'Describe the agent file or phase, visibility target, allowed subagents, model routing, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Agent Frontmatter Standards

This skill governs the design and validation of YAML frontmatter in `.github/agents/*.agent.md` files. It ensures that every agent in the NeatapticTS customization system carries correct visibility flags, bounded delegation, valid model strings, and passes automated frontmatter and graph validation scripts.

## When to Use

- Creating a new `.agent.md` file and need to establish correct initial frontmatter.
- Editing tools, `agents` allow-lists, model strings, or handoff fields in an existing agent.
- Diagnosing why a customization change appears to have no effect (silent loading failure).
- Reviewing whether the eight SDLC orchestrators (`00-helping` through `07-logging`) are correctly surfaced as `user-invocable: true`.
- Auditing hidden specialists to confirm they carry `user-invocable: false` and bounded `agents: []`.
- Preparing validation evidence before or after a customization batch.

## Task Packet

Include the agent filename, the specific frontmatter fields being changed, whether this is a user-facing orchestrator or hidden specialist, intended subagent allow-list, and which validation mode to use (normal vs. `--strict`).

```text
Use agent-frontmatter-standards for <agent-name>.agent.md.
Fields: <e.g. model, agents, user-invocable>
Visibility: <user-facing orchestrator | hidden specialist>
Subagents: <explicit list or none>
Validate with: node scripts/agent-customization/validate-agent-frontmatter.mjs --json
```

## Required Workflow

1. Read the active customization tracker if one is open; otherwise scope edits only to the requested agent file.
2. Confirm the eight user-facing SDLC orchestrators are the target surface: `00-helping`, `01-planning`, `02-researching`, `03-red-testing`, `04-implementing`, `05-green-testing`, `06-documenting`, `07-logging`.
3. Set `user-invocable: false` on every hidden specialist or auxiliary agent; set `true` only for the eight orchestrators.
4. Use an explicit `agents: [...]` allow-list on phase agents; never omit `agents` where broader delegation is not intended.
5. Include `agent` in the `tools` list whenever `agents` is non-empty.
6. Use confirmed qualified model strings (for example `GPT-5.4 (copilot)`) or validated fallback arrays; prefer the form documented in `model-routing-and-budget`.
7. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after every edit.
8. Run with `--strict` only when the full eight-agent SDLC surface is expected to be complete and correct.
9. Record validation output in the active plan or tracker as evidence.

## Guardrails

- Do not copy durable workflow policy into agent body text; put procedure in skills and have the agent invoke the skill by name.
- Do not use `agents: '*'` or omit `agents` on orchestrators that should have bounded delegation.
- Do not use unqualified or invented model strings; always use strings validated by the model-routing-and-budget skill.
- Do not run `--strict` validation during a migration that has not yet reached the eight-agent target state.
- YAML frontmatter parse failures are silent in VS Code; always prefer single-line quoted descriptions, same-line inline arrays, and explicit boolean values.
- Do not grant a hidden specialist `user-invocable: true` without explicit intent and user approval.

## Expected Final Output

- The target `.agent.md` file has correct, validated frontmatter with explicit `user-invocable`, `disable-model-invocation`, `agents`, and `model` fields.
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` exits cleanly with no errors.
- If the eight-orchestrator surface is complete, `--strict` also passes.
- Validation output is recorded in the active plan or chat summary.
