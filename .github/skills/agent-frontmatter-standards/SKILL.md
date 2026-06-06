---
name: agent-frontmatter-standards
description: 'Use when: validating or designing VS Code custom agent frontmatter for NeatapticTS, including tier, skills, tools, agents allow-lists, model routing, visibility, handoffs, migration state, CI validation, or diagnosing silent customization loading failures.'
argument-hint: 'Describe the agent file(s), tier and visibility target, tools/skills/agents changes, model routing, migration state, observed failure symptom, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Agent Frontmatter Standards

This skill governs the design and validation of YAML frontmatter in `.github/agents/*.agent.md` files. It keeps the NeatapticTS customization system discoverable, bounded, and auditable by enforcing correct visibility, tier, delegation, skills, model routing, and validation evidence across local edits and CI.

## When to Use

- Creating a new `.agent.md` file and need to establish correct initial frontmatter.
- Editing tools, `agents` allow-lists, model strings, or handoff fields in an existing agent.
- Adding or correcting `tier`, `skills`, `description`, or other metadata that affects routing and discoverability.
- Diagnosing why a customization change appears to have no effect (silent loading failure).
- Reviewing whether the eight SDLC orchestrators (`00-helping` through `07-logging`) are correctly surfaced as `user-invocable: true`.
- Auditing hidden specialists to confirm they carry `user-invocable: false` and bounded `agents: []`.
- Preparing validation evidence before or after a customization batch, migration step, or CI gate.

## Task Packet

Include the agent filename, the specific frontmatter fields being changed, whether this is a new agent, migration, or repair, whether this is a user-facing orchestrator or hidden specialist, intended subagent allow-list, observed failure symptoms, and which validation mode to use (normal vs. `--strict`).

```text
Use agent-frontmatter-standards for <agent-name>.agent.md.
Change type: <new agent | migration | repair>
Fields: <e.g. tier, skills, model, agents, user-invocable>
Visibility: <user-facing orchestrator | hidden specialist>
Subagents: <explicit list or none>
Failure symptom: <silent load | validator error | routing drift | none>
Validate with: node scripts/agent-customization/validate-agent-frontmatter.mjs --json
```

## Required Workflow

1. Classify the change as a new agent, staged migration, or repair so validation expectations are explicit before editing.
2. Read the active customization tracker if one is open; otherwise scope edits only to the requested agent file.
3. Confirm filename, `name`, folder path, and `tier` intent all match the agent's actual role.
4. Confirm the eight user-facing SDLC orchestrators are the only valid `user-invocable: true` surface: `00-helping`, `01-planning`, `02-researching`, `03-red-testing`, `04-implementing`, `05-green-testing`, `06-documenting`, `07-logging`.
5. Set `user-invocable: false` on every hidden specialist or auxiliary agent; set `true` only for the eight orchestrators.
6. Use an explicit `agents: [...]` allow-list on phase agents; never omit `agents` where broader delegation is not intended.
7. Include `agent` in the `tools` list whenever `agents` is non-empty.
8. Verify `skills: [...]` is present even when the list is empty, and keep durable policy in skills rather than agent body text.
9. Use confirmed qualified single model strings; NeatapticTS targets CLI-compatible scalar `model` frontmatter documented in `model-routing-and-budget`.
10. Normalize YAML for VS Code safety: single-line quoted descriptions, explicit boolean values, and inline arrays where the parser is known to be fragile.
11. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after every edit.
12. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict` only when the targeted migration state is complete and the full required surface is expected to pass.
13. Run `node scripts/agent-customization/validate-agent-graph.mjs --json` whenever `tier`, `user-invocable`, or `agents` changes alter delegation boundaries.
14. Refresh shared metadata with `npm run agents:routing-table` when agent names, skills, or routing metadata changes could stale the canonical table.
15. Record validator output, migration state, and any residual routing risk in the active plan or tracker as evidence.

## Error Recovery and Silent Failure Triage

1. Treat a no-effect customization edit as a likely metadata or YAML failure before assuming runtime logic is broken.
2. Check for the highest-signal failure sources first: filename and `name` mismatch, missing `tier`, missing `skills`, invalid model strings, omitted `agent` tool when `agents` is non-empty, or malformed YAML.
3. If validation fails, keep the error visible in the tracker or chat summary, correct the frontmatter, and rerun the narrowest relevant validator before making additional edits.
4. If VS Code still appears to ignore the agent after metadata fixes, refresh the routing table, rerun strict validation when appropriate, and capture the exact symptom so the next pass starts from evidence instead of guesswork.

## Migration and Strict-Mode Boundaries

- Use non-strict validation during partial migrations when the repo is intentionally between stable states.
- Record why `--strict` is deferred, what remains incomplete, and which agent or field is still in transition.
- Move back to `--strict` as soon as the intended target state is restored; do not leave migrations indefinitely in a partially validated state.
- When a migration changes tiering, visibility, or bounded delegation, pair frontmatter validation with graph validation instead of treating them as separate concerns.

## Edge Cases and Exceptions

- Prefer `agents: []` and `skills: []` explicitly when an agent should not delegate or when no durable skill binding exists yet.
- If emergency work appears to require broader delegation, record the exception, bound it to the smallest temporary allow-list, and define the rollback condition before shipping the change.
- Do not create self-referential or circular delegation chains; if the delegation path is hard to explain in one sentence, simplify it before validating.
- When repairing a legacy array-valued `model`, preserve the first listed qualified string unless the user explicitly wants a routing change.

## CI/CD and Validation Evidence

- Local edits should be validated with `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`.
- Stable surfaces should also pass `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict`.
- Delegation changes should pass `node scripts/agent-customization/validate-agent-graph.mjs --json`.
- Shared metadata changes should be followed by `npm run agents:routing-table` and `npm run agents:routing-table:gate` so CI and local state stay aligned.
- Treat validator output as delivery evidence, not optional cleanup.

## Change History Guidance

- Record which frontmatter fields changed, why they changed, and whether the edit was a repair, migration step, or policy alignment.
- Note the validator commands that were run and whether each passed in normal mode, strict mode, or both.
- If any residual risk remains, describe the exact boundary and the next validation step instead of leaving a generic warning.

## Guardrails

- Do not copy durable workflow policy into agent body text; put procedure in skills and have the agent invoke the skill by name.
- Do not use `agents: '*'` or omit `agents` on orchestrators that should have bounded delegation.
- Do not use unqualified or invented model strings; always use strings validated by the model-routing-and-budget skill.
- Do not run `--strict` validation during a migration that has not yet reached the intended stable target state.
- YAML frontmatter parse failures are silent in VS Code; always prefer single-line quoted descriptions, same-line inline arrays, and explicit boolean values.
- Do not omit `tier` or `skills`; both are part of the repo's enforced agent contract.
- Do not grant a hidden specialist `user-invocable: true` without explicit intent and user approval.
- Do not hide validation failures, route drift, or emergency exceptions; surface them with the exact command output or failure symptom.

## Expected Final Output

- The target `.agent.md` file has correct, validated frontmatter with explicit `tier`, `skills`, `user-invocable`, `disable-model-invocation`, `agents`, and `model` fields.
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` exits cleanly with no errors.
- When the target state is migration-complete, `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict` also passes.
- When delegation boundaries changed, `node scripts/agent-customization/validate-agent-graph.mjs --json` also passes.
- Shared metadata changes are reflected in the canonical routing table when relevant.
- Validation output, migration state, and residual routing risk are recorded in the active plan or chat summary.
