---
name: model-routing-and-budget
description: 'Choose, validate, and document model routing for NeatapticTS custom agents. Use when assigning GPT-5.4, GPT-5.4-mini, Claude Sonnet 4.6, Claude Haiku 4.6, fallback arrays, phase-specific model budgets, or when a model string must be verified before frontmatter changes.'
argument-hint: 'Describe the agent phase, desired model tier, available model names, and whether validation should be advisory or strict.'
user-invocable: false
disable-model-invocation: false
---

# Model Routing And Budget

Use this skill before writing or changing an agent `model` field, and whenever
a frontmatter model string must be verified as a valid qualified name before
it is committed.

This skill owns the phase-default routing table, qualified name validation
process, and fallback array design for NeatapticTS custom agents.

## When to Use

- A new agent file is being created and its `model` field needs a phase-
  appropriate qualified name and fallback array.
- An existing agent's model string is producing routing errors or resolving to
  an unexpected model tier.
- The available Copilot model list has changed and agent frontmatter needs to
  be updated to match.
- A phase is being delegated to a lower-cost tier and the routing change needs
  to be validated before wiring.
- Frontmatter shape needs to be confirmed with `validate-agent-frontmatter.mjs`
  before the agent is used in a gate or handoff.

## Task Packet

Pass a compact packet describing the agent, its phase, and the routing decision
to validate.

```text
Use model-routing-and-budget for the 03-red-testing agent frontmatter.
Agent: .github/agents/03-red-testing.agent.md.
Phase: 03 Red Testing.
Desired tier: Full (GPT-5.4 or Claude Sonnet 4.6).
Validation: advisory — confirm qualified name before committing.
```

## Required Workflow

1. Discover the exact qualified model names available in the active Copilot
   client before assigning any name.
2. Treat session-local availability constraints as controlling for frontmatter
   edits. Under the current cost-tier restriction, `GPT-5.5 (copilot)` must
   not be written to frontmatter.
3. Use `GPT-5.4 (copilot)` for coding-heavy implementation and red-test
   synthesis when available.
4. Use `Claude Sonnet 4.6 (copilot)` for planning, documentation synthesis,
   nuanced maintenance, and ambiguity-heavy coordination when available.
5. Use `GPT-5.4-mini (copilot)` for bounded research, validation, and subagent
   work where coding or tool strength still matters.
6. Use `Claude Haiku 4.6 (copilot)` for narrow checklist, summarization, and
   mechanical assistant work. If the model picker exposes only a different Haiku
   generation, update the qualified name before strict validation.
7. Use fallback arrays so agents degrade to an available qualified Copilot model
   when the primary is unavailable.
8. Validate frontmatter shape with
   `node scripts/agent-customization/validate-agent-frontmatter.mjs`.

## Phase Defaults

| Phase             | Tier          | Reason                                                                    |
| ----------------- | ------------- | ------------------------------------------------------------------------- |
| 00 Helping        | Sonnet / Full | Maintenance and gap resolution need nuanced synthesis plus safe fallback. |
| 01 Planning       | Sonnet / Full | Architecture decisions and cross-plan tradeoffs need broad reasoning.     |
| 02 Research       | Mini / Haiku  | Retrieval and summarization should be cheap and bounded.                  |
| 03 Red Testing    | Full          | Test contracts need careful judgment.                                     |
| 04 Implementation | Full          | Implementation needs deeper reasoning and edge-case handling.             |
| 05 Green Testing  | Mini / Haiku  | Verification is mostly mechanical.                                        |
| 06 Documentation  | Sonnet / Mini | Educational docs benefit from stronger writing after facts exist.         |
| 07 Logging        | Haiku / Mini  | Summarization and tracker updates should be lightweight.                  |

## Guardrails

- Do not commit a model string that has not been confirmed as a valid qualified
  name in the active Copilot client; an invalid name causes silent fallback or
  routing errors.
- When a model is rejected by the active session, keep that session-local
  restriction out of frontmatter. Under the current cost-tier restriction,
  `GPT-5.5 (copilot)` must not be written to frontmatter.
- Do not assign a Full-tier model to phases where a Mini or Haiku tier is
  sufficient; unnecessary cost undermines the budget design.
- Do not omit a fallback array for agents that must be resilient to model
  availability changes.
- Do not hand-edit qualified names without re-running
  `validate-agent-frontmatter.mjs`; the validator catches typos and schema
  drift that manual review misses.

## Expected Final Output

A strong model-routing pass should produce:

- the confirmed qualified model name for the target agent and phase,
- the fallback array if the model has known availability constraints,
- the `validate-agent-frontmatter.mjs` result confirming schema validity,
- an updated agent frontmatter `model` field ready to commit.
