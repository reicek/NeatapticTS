---
name: model-routing-and-budget
description: 'Choose, validate, and document model routing for NeatapticTS custom agents. Use when assigning GPT-5.4, GPT-5.4-mini, fallback arrays, phase-specific model budgets, or when a model string must be verified before frontmatter changes.'
argument-hint: 'Describe the agent phase, desired model tier, available model names, and whether validation should be advisory or strict.'
user-invocable: false
disable-model-invocation: false
---

# Model Routing And Budget

Use this skill before writing or changing an agent `model` field.

## Workflow

1. Discover the exact qualified model names available in the active Copilot client.
2. Prefer `GPT-5.4 (copilot)` for Full phases when it is available.
3. Prefer `GPT-5.4-mini (copilot)` for Mini phases when it is available.
4. Fall back from Mini to `GPT-5.4 (copilot)` when the mini variant is unavailable.
5. Use fallback arrays so agents degrade to an available qualified Copilot model.
6. Record the resolved model names in `plans/Agentic_Workflow_Architecture.plans.md` before editing agents.
7. Validate frontmatter shape with `validate-agent-frontmatter.mjs`.

## Phase Defaults

| Phase | Tier | Reason |
|---|---|---|
| 01 Planning | Full | Architecture decisions and cross-plan tradeoffs need broad reasoning. |
| 02 Research | Mini | Retrieval and summarization should be cheap and bounded. |
| 03 Red Testing | Full | Test contracts need careful judgment. |
| 04 Implementation | Full | Implementation needs deeper reasoning and edge-case handling. |
| 05 Green Testing | Mini | Verification is mostly mechanical. |
| 06 Documentation | Mini | Structured writing can be delegated after implementation facts exist. |
| 07 Session Logging | Mini | Summarization and tracker updates should be lightweight. |

## Sources

- VS Code custom agents documentation allows a single `model` string or prioritized fallback array with qualified names such as `GPT-5.4 (copilot)`.