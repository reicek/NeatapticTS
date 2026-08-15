---
name: model-routing-and-budget
description: 'Use when: choosing or validating custom-agent model routing and budget.'
argument-hint: 'Describe the agent phase, desired model tier, available model names, and whether validation should be advisory or strict.'
user-invocable: false
disable-model-invocation: false
skills:
  - customize-cloud-agent
  - agent-frontmatter-standards
  - routing-optimization-policy
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Model Routing And Budget

Use this skill before writing or changing an agent `model` field, and whenever
a frontmatter model string must be verified as a valid qualified name before
it is committed.

This skill owns the phase-default routing table, qualified name validation
process, and CLI-compatible scalar `model` policy for NeatapticTS custom
agents.

## When to Use

- A new agent file is being created and its `model` field needs a phase-
  appropriate qualified name.
- An existing agent's model string is producing routing errors or resolving to
  an unexpected model tier.
- The available Copilot model list has changed and agent frontmatter needs to
  be updated to match.
- A phase is being delegated to a lower-cost tier and the routing change needs
  to be validated before wiring.
- Frontmatter shape needs to be confirmed with `validate-agent-frontmatter.mjs`
  before the agent is used in a gate or handoff.

## When NOT to use

Do NOT use for cloud agent setup - use `customize-cloud-agent` instead. Do NOT use for frontmatter validation - use `agent-frontmatter-standards` instead.

## Workflow Diagram

```text
Flowchart summary: "Select model" → "Agent tier?"; "Agent tier?" → "Use high-capability model" (Tier 0-1), "Use mid-capability model" (Tier 2-3), "Use lightweight model" (Tier 4); "Use high-capability model" → "Check context budget"; "Use mid-capability model" → "Check context budget"; "Use lightweight model" → "Check context budget"; "Check context budget" → "Within budget?"; "Within budget?" → "Approve routing" (Yes), "Downgrade model" (No); "Approve routing"; "Downgrade model" → "Check context budget".
```

## Task Packet

Pass a compact packet describing the agent, its phase, and the routing decision
to validate.

```text
Use model-routing-and-budget for the 03-red-testing agent frontmatter.
Agent: .github/agents/03-red-testing.agent.md.
Phase: 03 Red Testing.
Desired tier: Full (glm-5.2:cloud).
Validation: advisory — confirm qualified name before committing.
```

## Required Workflow

1. Discover the exact qualified model names available in the active Copilot
   client before assigning any name.
2. Treat session-local availability constraints as controlling for frontmatter
   edits. Under the current cost-tier restriction, `GPT-5.5 (copilot)` must
   not be written to frontmatter.
3. Use `glm-5.2:cloud` for coding-heavy implementation and red-test
   synthesis when available.
4. Use `glm-5.2:cloud` for planning, documentation synthesis,
   nuanced maintenance, and ambiguity-heavy coordination when available.
5. Use `kimi-k2.7-code:cloud` for bounded research, validation, and subagent
   work where coding or tool strength still matters.
6. Use `kimi-k2.7-code:cloud` for narrow checklist, summarization, and
   mechanical assistant work.
7. Write a single qualified model string in `model:`. When repairing a legacy
   array-valued `model`, preserve the first listed entry unless the user
   explicitly requests a different routing decision.
8. Validate frontmatter shape with
   `node scripts/agent-customization/validate-agent-frontmatter.mjs`.
9. Additional models may be added to the approved pool only with explicit user
   approval. Do not introduce unapproved model names into agent frontmatter or
   this skill's routing tables.

## Phase Defaults

| Phase             | Tier                 | Reason                                                                    |
| ----------------- | -------------------- | ------------------------------------------------------------------------- |
| 00 Helping        | glm-5.2:cloud        | Maintenance and gap resolution need nuanced synthesis plus safe fallback. |
| 01 Planning       | glm-5.2:cloud        | Architecture decisions and cross-plan tradeoffs need broad reasoning.     |
| 02 Research       | kimi-k2.7-code:cloud | Retrieval and summarization should be cheap and bounded.                  |
| 03 Red Testing    | glm-5.2:cloud        | Test contracts need careful judgment.                                     |
| 04 Implementation | glm-5.2:cloud        | Implementation needs deeper reasoning and edge-case handling.             |
| 05 Green Testing  | kimi-k2.7-code:cloud | Verification is mostly mechanical.                                        |
| 06 Documentation  | glm-5.2:cloud        | Educational docs benefit from stronger writing after facts exist.         |
| 07 Logging        | kimi-k2.7-code:cloud | Summarization and tracker updates should be lightweight.                  |

## Decision Tree: Model Selection by Tier

```text
Flowchart summary: "Need model for agent" → "What tier?"; "What tier?" → "glm-5.2:cloud" (Tier 0 (Agent Zero)), "glm-5.2:cloud" (Tier 1 (SDLC)), "kimi-k2.7-code:cloud" (Tier 2 (Coordinators)), "kimi-k2.7-code:cloud" (Tier 3 (Scouts)), "kimi-k2.7-code:cloud" (Tier 4 (Auxiliaries)); "glm-5.2:cloud"; "kimi-k2.7-code:cloud".
```

## Before / After Examples

**Before:**

```yaml
---
model: claude-sonnet
---
```

**After:**

```yaml
---
# Qualified name confirmed in the active Copilot client; tier budget matches phase default.
model: glm-5.2:cloud
---
```

## Guardrails

- Do not commit a model string that has not been confirmed as a valid qualified
  name in the active Copilot client; an invalid name causes silent fallback or
  routing errors.
- When a model is rejected by the active session, keep that session-local
  restriction out of frontmatter. Under the current cost-tier restriction,
  `GPT-5.5 (copilot)` must not be written to frontmatter.
- Do not assign a Full-tier model (`glm-5.2:cloud`) to phases where a
  light-tier model (`kimi-k2.7-code:cloud`) is sufficient; unnecessary cost
  undermines the budget design.
- Do not write arrays into `model:` frontmatter in this repo; NeatapticTS
  targets Copilot CLI-compatible scalar model strings.
- Do not hand-edit qualified names without re-running
  `validate-agent-frontmatter.mjs`; the validator catches typos and schema
  drift that manual review misses.

## Expected Final Output

A strong model-routing pass should produce:

- the confirmed qualified model name for the target agent and phase,
- the `validate-agent-frontmatter.mjs` result confirming schema validity,
- an updated scalar `model` value ready to commit.
