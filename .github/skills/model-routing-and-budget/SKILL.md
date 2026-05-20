---
name: model-routing-and-budget
description: 'Choose, validate, and document model routing for NeatapticTS custom agents. Use when assigning GPT-5.4, GPT-5.4-mini, Claude Sonnet 4.6, Claude Haiku 4.6, fallback arrays, phase-specific model budgets, or when a model string must be verified before frontmatter changes.'
argument-hint: 'Describe the agent phase, desired model tier, available model names, and whether validation should be advisory or strict.'
user-invocable: false
disable-model-invocation: false
---

# Model Routing And Budget

Use this skill before writing or changing an agent `model` field.

## Workflow

1. Discover the exact qualified model names available in the active Copilot client.
2. Use `GPT-5.4 (copilot)` for coding-heavy implementation and red-test synthesis when available.
3. Use `Claude Sonnet 4.6 (copilot)` for planning, documentation synthesis, nuanced maintenance, and ambiguity-heavy coordination when available.
4. Use `GPT-5.4-mini (copilot)` for bounded research, validation, and subagent work where coding/tool strength still matters.
5. Use `Claude Haiku 4.6 (copilot)` for narrow checklist, summarization, and mechanical assistant work; if the model picker exposes only a different Haiku generation, update the qualified name before strict validation.
6. Use fallback arrays so agents degrade to an available qualified Copilot model.
7. Validate frontmatter shape with `validate-agent-frontmatter.mjs`.

## Phase Defaults

| Phase | Tier | Reason |
|---|---|---|
| 00 Helping | Sonnet / Full | Maintenance and gap resolution need nuanced synthesis plus safe fallback. |
| 01 Planning | Sonnet / Full | Architecture decisions and cross-plan tradeoffs need broad reasoning. |
| 02 Research | Mini / Haiku | Retrieval and summarization should be cheap and bounded. |
| 03 Red Testing | Full | Test contracts need careful judgment. |
| 04 Implementation | Full | Implementation needs deeper reasoning and edge-case handling. |
| 05 Green Testing | Mini / Haiku | Verification is mostly mechanical. |
| 06 Documentation | Sonnet / Mini | Educational docs benefit from stronger writing after facts exist. |
| 07 Logging | Haiku / Mini | Summarization and tracker updates should be lightweight. |

## Sources

- VS Code custom agents documentation allows a single `model` string or prioritized fallback array with qualified names such as `GPT-5.4 (copilot)` or `Claude Sonnet 4.5 (copilot)`.
- OpenAI model guidance positions GPT-5.4 for coding and professional work and GPT-5.4-mini for lower-latency, lower-cost coding and subagent workloads.
- Anthropic model guidance positions Claude Sonnet 4.6 as the best combination of speed and intelligence, and Haiku-class models as the fastest economical tier.