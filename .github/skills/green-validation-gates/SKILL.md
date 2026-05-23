---
name: green-validation-gates
description: 'Run and interpret green-phase validation for NeatapticTS agent workflow changes. Use when deciding focused test order, rerouting failed checks, enforcing coverage guard, or proving scripts, plans, agents, and skills are consistent.'
argument-hint: 'Describe changed files, expected validations, latest failures, and whether strict customization validation should pass yet.'
user-invocable: false
disable-model-invocation: false
---

# Green Validation Gates

Use this skill after implementation edits.

## Workflow

1. Run the narrowest validation that matches the changed surface.
2. For customization scripts, run each script in audit mode and targeted strict mode when the target state should hold.
3. For plan-only changes, run markdown whitespace checks and plan-sync validation.
4. For `.github/agents/`, run agent frontmatter and graph validation.
5. For `.github/skills/`, run skill frontmatter validation.
6. Route failures back to Red Testing or Implementation instead of continuing to Documentation.
7. Record validation evidence in the active plan.

## Gate-Aware Contract

Gates in `.github/flows/` return structured JSON with four required fields:

```json
{
  "pass": true,
  "evidence": { "...": "..." },
  "fixHint": "What to do if this gate fails",
  "owner": "gate-script-name.gate.mjs or validate-*.mjs"
}
```

A flow is only complete when every declared gate returns `"pass": true`.
When a gate fails:
- Record the exception with `node scripts/agent-customization/gates/record-gate-exception.mjs`.
- The exception is appended to `.github/ai-learning/learning-log.jsonl`.
- Three consecutive gate failures in the same session escalate to `00-helping`
  via `00.cross-tier-helper`.
- Do not mark a phase step `[DONE]` if any declared gate has `"pass": false`.

**Tier-1 gate catalog:**

| Gate ID | Check | Owner |
|---|---|---|
| `plan-sync` | Plan registered in README + Roadmap; status coherent | `scripts/agent-customization/gates/plan-sync.gate.mjs` |
| `step-packet` | Active step has yaml block, status, next_step, validation, stop conditions | `scripts/agent-customization/gates/step-packet.gate.mjs` |
| `agent-graph` | All flow/gate/agent references resolve to real files | `scripts/agent-customization/gates/agent-graph.gate.mjs` |
| `learning-event` | A learning event exists for any gate exception or cross-tier call | `scripts/agent-customization/gates/learning-event.gate.mjs` |



- Run `npm run build` or `npm run lint` when TypeScript, package scripts, or source code changes require it.
- Run `npm run docs` only when docs-generation inputs are touched.