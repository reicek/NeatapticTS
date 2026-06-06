# GitHub Copilot Model Comparison — Research Summary

> Generated: 2026-05-25 | Source: https://docs.github.com/en/copilot/reference/ai-models/model-comparison and further reading links
> Purpose: Inform model assignment decisions for NeatapticTS custom agents

---

## Model Comparison Overview

The GitHub Copilot model comparison page organizes models into three primary task categories:
- **Fast help** — lightweight, repetitive coding questions; lowest latency
- **General-purpose** — inline suggestions, explanations, agentic codebase exploration
- **Deep reasoning** — multi-step problem solving, complex debugging, architecture analysis

---

## Full Model Table (as documented)

| Model Name (docs label) | Task Category | Coding Strength | Reasoning Depth | Speed | Premium Multiplier |
|---|---|---|---|---|---|
| GPT-4.1 | General-purpose | Standard | Standard | Fast | 0× (included) |
| GPT-4o | General-purpose | Standard | Standard | Fast | 0× (included) |
| GPT-5 mini | General-purpose / Deep reasoning | Standard to strong | Standard to strong | Fast | 0× (included) |
| GPT-5.2 | Deep reasoning | Strong | Deep | Moderate | 1× |
| GPT-5.2-Codex | Agentic software development | Strong | Deep | Moderate | 1× |
| GPT-5.3-Codex | Agentic software development | Strong | Deep | Moderate | 1× |
| GPT-5.4 | Deep reasoning | Strong | Deep | Moderate | 1× |
| GPT-5.4 mini | Agentic / General-purpose | Moderate | Moderate | Fast | 0.33× |
| GPT-5.4 nano | Fast help | Light | Light | Very fast | 0.25× |
| **GPT-5.5** | **Deep reasoning** | **Very strong** | **Very deep** | **Moderate** | **7.5× (promotional)** |
| Claude Haiku 4.5 | Fast help | Light | Light | Very fast | 0.33× |
| Claude Opus 4.5 | Deep reasoning | Very strong | Very deep | Slow | 3× |
| Claude Opus 4.6 | Deep reasoning | Very strong | Very deep | Slow | 3× |
| Claude Opus 4.6 (fast mode, preview) | Deep reasoning | Very strong | Very deep | Fast (fast mode) | 30× |
| Claude Opus 4.7 | Deep reasoning | Very strong | Very deep | Slow | 15× |
| Claude Sonnet 4.5 | General-purpose + agents | Strong | Deep | Moderate | 1× |
| Claude Sonnet 4.6 | General-purpose + agents | Strong | Deep | Moderate | 1× |
| Gemini 2.5 Pro | Deep reasoning | Strong | Deep | Moderate | 1× |
| Gemini 3 Flash | Fast help | Light | Light | Very fast | 0.33× |
| Gemini 3.1 Pro | Deep reasoning | Strong | Very deep | Moderate | 1× |
| Gemini 3.5 Flash | Fast help | Light | Light | Very fast | 14× |
| Raptor mini | General-purpose | Standard | Standard | Fast | 0× (included) |
| Goldeneye | Deep reasoning | Strong | Very deep | Moderate | Not listed (free plan) |
| Qwen2.5 | General-purpose | Strong | Standard | Fast | Not listed |

> **Note on context window sizes:** The primary comparison page does not list context window sizes inline. They are documented on the supported-ai-models detail page (URL listed in Further Reading below).

---

## Key Decision Factors for Custom Agent Assignment

### Included Models (no premium cost on paid plans)
These are the budget-safe choices for high-frequency or low-stakes agents:
- `GPT-5 mini` — 0× multiplier, reasonable reasoning
- `GPT-4.1` — 0× multiplier, fast inline suggestions
- `GPT-4o` — 0× multiplier
- `Raptor mini` — 0× multiplier, fast inline suggestions

### Economy Premium (≤ 0.33×)
Good for mini/haiku replacement candidates:
- `GPT-5.4 mini` — 0.33× (vs current `GPT-5.4 mini (copilot)` in our fleet)
- `GPT-5.4 nano` — 0.25×
- `Claude Haiku 4.5` — 0.33× (one-generation older than current `Claude Haiku 4.6`)
- `Gemini 3 Flash` — 0.33×

### Balanced Full-Tier (1×)
Current fleet primary full-tier models fall here:
- `GPT-5.4` — 1×, strong coding, deep reasoning (current fleet primary)
- `Claude Sonnet 4.6` — 1×, nuanced synthesis (current fleet primary)
- `GPT-5.2`, `GPT-5.3-Codex` — 1×, agentic specialization
- `Gemini 2.5 Pro`, `Gemini 3.1 Pro` — 1×, long context / scientific

### High-Cost (3×–30×)
Deep reasoning with significant budget impact:
- `Claude Opus 4.6` — 3×, highest-quality Anthropic reasoning
- `Claude Opus 4.7` — 15×
- `Claude Opus 4.6 (fast mode, preview)` — 30×
- `GPT-5.5` — **7.5× (promotional rate as of docs date)**

---

## GPT-5.5 Availability Verdict

### What the docs confirm

GPT-5.5 is explicitly listed in the GitHub Copilot model comparison documentation under the **Deep reasoning** category with the description:
> "Great at complex reasoning, code analysis, and technical decision-making."

The model-hosting page documents:
> "GPT-5.5 is available at a promotional multiplier of 7.5x."

GPT-5.5 is listed among the **OpenAI** models available through GitHub Copilot. It is real and available in GitHub Copilot at a premium.

### What the docs do NOT confirm

- The docs tables use the bare label **`GPT-5.5`** — not `GPT-5.5 (copilot)`.
- There is **no mention of `GPT-5.5 (copilot)`** as a qualified agent model string in any fetched documentation page.
- The docs pages covering custom agent model availability (VS Code Copilot extension agent model selection) could not be fetched; their custom-agent-specific availability status cannot be confirmed from docs alone.

### Local environment evidence

The existing NeatapticTS fleet uses the `(copilot)` qualifier suffix for all four current model strings:
- `GPT-5.4 (copilot)` — confirmed working
- `GPT-5.4 mini (copilot)` — confirmed working
- `Claude Sonnet 4.6 (copilot)` — confirmed working
- `Claude Haiku 4.6 (copilot)` — confirmed working

The `model-routing-and-budget` skill (`.github/skills/model-routing-and-budget/SKILL.md`) does **not** list `GPT-5.5 (copilot)` in its phase defaults table. This is the canonical source of truth for validated model strings in this environment.

### Verdict

**`GPT-5.5 (copilot)` has NOT been confirmed as a valid qualified model string for NeatapticTS custom agents.**

- Docs confirm `GPT-5.5` exists in GitHub Copilot as a model.
- Docs do not confirm the `(copilot)` qualified variant or its availability in `.agent.md` files specifically.
- The `model-routing-and-budget` skill does not list it as a validated string.
- **Before using `GPT-5.5 (copilot)` in any agent frontmatter, run `model-name-auditor` to confirm the exact qualified name and then validate with `validate-agent-frontmatter.mjs`.**
- Using an unconfirmed model string causes silent fallback or routing errors per the model-routing-and-budget guardrails.

---

## Further Reading Links (from primary page)

| Page | URL | Relevance |
|---|---|---|
| Supported AI models in Copilot | https://docs.github.com/en/copilot/using-github-copilot/ai-models/supported-ai-models-in-copilot | Detailed model specs, context windows, pricing |
| Comparing AI models using different tasks | https://docs.github.com/en/copilot/using-github-copilot/ai-models/comparing-ai-models-using-different-tasks | Task-specific model comparison examples |
| Changing the AI model for Copilot Chat | https://docs.github.com/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat | Model switching in Copilot Chat |
| Change the completion model | https://docs.github.com/en/copilot/how-tos/use-ai-models/change-the-completion-model | Inline suggestion model switching |
| Model hosting | https://docs.github.com/en/copilot/reference/ai-models/model-hosting | Hosting information and premium multipliers |
| Auto model selection | https://docs.github.com/en/copilot/concepts/auto-model-selection | Auto model selection behavior |
| About premium requests | https://docs.github.com/en/copilot/managing-copilot/monitoring-usage-and-entitlements/about-premium-requests | Premium request billing details |

> **Billing note:** Starting June 1, 2026, GitHub is moving Copilot from request-based billing to usage-based billing per the model-hosting docs.

---

## Upgrade Candidate Summary for NeatapticTS Agent Fleet

| Current String | Potential Upgrade | Multiplier Δ | Notes |
|---|---|---|---|
| `GPT-5.4 (copilot)` | `GPT-5.5 (copilot)` | 1× → 7.5× | Significant cost increase; requires qualified-name confirmation |
| `Claude Sonnet 4.6 (copilot)` | `Claude Opus 4.6 (copilot)` | 1× → 3× | Higher reasoning ceiling; requires qualified-name confirmation |
| `GPT-5.4 mini (copilot)` | `GPT-5.4 nano (copilot)` | 0.33× → 0.25× | Slight cost reduction; requires qualified-name confirmation |
| `Claude Haiku 4.6 (copilot)` | No direct upgrade listed | — | Haiku 4.6 appears to be current generation; Haiku 4.5 is older |

> All upgrade candidates require `model-name-auditor` confirmation of the exact `(copilot)` qualified string before any `.agent.md` frontmatter changes.
