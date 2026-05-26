# Canonical Agent Model Reference

> Canonical decision aid for NeatapticTS custom-agent model assignment.
> Use this file before adding a new agent or changing any `model:` frontmatter.

## Purpose And Maintenance Workflow

This file is the durable reference for model-selection decisions in the
NeatapticTS agent fleet. It complements the generated
`.github/agent-skill-routing-table.md` by explaining why a given model belongs
on a given agent and by recording the currently confirmed qualified model
strings that are safe to write into frontmatter.

Maintenance workflow:

1. Confirm exact qualified model names in the active Copilot client before any
   frontmatter edit.
2. Update this file when a model becomes confirmed, deprecated, or when the
   routing rationale changes.
3. Edit `.github/agents/*.agent.md` files only after the assignment decision is
   clear and validated.
   Write fallback arrays as same-line inline arrays; the local customization
   parser intentionally does not support multi-line YAML blocks.
4. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs` after
   any frontmatter change.
5. Refresh `.github/agent-skill-routing-table.md` only after agent or skill
   frontmatter changes are complete.

> Warning: unverified model strings must not be written to frontmatter.
> A guessed or stale model string can trigger silent fallback or routing drift.

## Confirmed Qualified Model Strings

These are the only model strings confirmed by the current NeatapticTS fleet and
safe local evidence.

| Qualified model string        | Role in fleet                           | Capability note                                                                                                                   |
| ----------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `GPT-5.4 (copilot)`           | Full-tier coding and implementation     | Best default for code-heavy implementation, red-test authorship, and specialist work that needs stronger tool-aware reasoning.    |
| `Claude Sonnet 4.6 (copilot)` | Full-tier synthesis and coordination    | Best default for planning, documentation, maintenance, ambiguity-heavy coordination, and nuanced policy reasoning.                |
| `GPT-5.4-mini (copilot)`      | Budget-conscious general-purpose work   | Good default for bounded research, validation, subagent coordination, and specialists that still benefit from solid tool use.     |
| `Claude Haiku 4.6 (copilot)`  | Lightweight checklist and summarization | Good default for narrow audits, summaries, small recon tasks, and one-shot helpers where latency and cost matter more than depth. |

Other Copilot models may exist in docs or the client, but they are not
canonical for NeatapticTS agent frontmatter until the exact qualified string is
confirmed in the active client and validated locally.

## Canonical Decision Flow

1. Verify availability first.
   Use the active Copilot model picker or `model-name-auditor` to confirm the
   exact qualified string before touching frontmatter.
2. Classify the agent job.
   Decide whether the agent is primarily implementation, synthesis,
   research/validation, or lightweight checklist/summarization.
3. Choose the lowest-cost model that still fits the job.
   Use `GPT-5.4` for coding-heavy implementation, `Claude Sonnet 4.6` for
   nuanced synthesis, `GPT-5.4-mini` for bounded research and validation, and
   `Claude Haiku 4.6` for narrow mechanical work.
4. Design the fallback array deliberately.
   The first fallback should preserve behavior quality when possible; later
   fallbacks can trade cost for resilience. Do not omit fallbacks for agents
   that must remain usable across client availability changes.
5. Account for cost and performance.
   Reserve Full-tier models for work that truly needs deeper reasoning or
   stronger coding. Prefer Mini or Haiku for high-frequency mechanical tasks.
6. Validate before committing.
   Run `node scripts/agent-customization/validate-agent-frontmatter.mjs` after
   any frontmatter edit.
7. Refresh the generated routing table only after frontmatter changes land.
   This reference is hand-maintained; `.github/agent-skill-routing-table.md` is
   generated and should be refreshed after the actual agent changes, not before.

## Default Assignment Patterns

| Agent need                                               | Canonical primary             | Typical fallback shape           | Notes                                                                            |
| -------------------------------------------------------- | ----------------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| Coding-heavy implementation or test synthesis            | `GPT-5.4 (copilot)`           | Sonnet, then Mini                | Prefer when code generation quality and edge-case handling matter most.          |
| Ambiguity-heavy planning, maintenance, or docs synthesis | `Claude Sonnet 4.6 (copilot)` | GPT-5.4, then Mini               | Prefer when the agent must weigh tradeoffs, policy, or prose quality.            |
| Bounded research, validation, or scout work              | `GPT-5.4-mini (copilot)`      | Haiku or GPT-5.4 depending depth | Prefer Mini when the agent still uses tools heavily or needs moderate reasoning. |
| Lightweight checklist, summaries, or one-shot helpers    | `Claude Haiku 4.6 (copilot)`  | Mini, then Full                  | Prefer Haiku when the task is narrow, mechanical, and frequent.                  |

## Current Agent Assignments

### Tier 1 — SDLC Orchestrators

| Agent              | Tier | Job category                                                | Current primary               | Current fallback array                                                             | Recommendation / rationale                                                                                                            |
| ------------------ | ---- | ----------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `00-helping`       | 1    | AI system maintenance / workflow gap resolution             | `Claude Sonnet 4.6 (copilot)` | [`GPT-5.4 (copilot)`, `GPT-5.4-mini (copilot)`]                                    | Keep current. Maintenance and gap resolution are ambiguity-heavy and benefit from Sonnet-first synthesis with strong coding fallback. |
| `01-planning`      | 1    | Planning / decomposition / acceptance criteria              | `Claude Sonnet 4.6 (copilot)` | [`GPT-5.4 (copilot)`, `GPT-5.4-mini (copilot)`]                                    | Keep current. Cross-plan reasoning and acceptance-criteria work fit Sonnet-first routing.                                             |
| `02-researching`   | 1    | Codebase research / API exploration / architecture recon    | `GPT-5.4-mini (copilot)`      | [`Claude Haiku 4.6 (copilot)`, `Claude Sonnet 4.6 (copilot)`, `GPT-5.4 (copilot)`] | Keep current. Research should stay cheap by default but preserve escalation paths for harder recon.                                   |
| `03-red-testing`   | 1    | Red test creation / failing test authorship                 | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]                          | Keep current. Red-test design needs stronger coding judgment and careful edge-case framing.                                           |
| `04-implementing`  | 1    | Scoped code implementation                                  | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]                          | Keep current. Implementation is the clearest Full-tier coding surface in the fleet.                                                   |
| `05-green-testing` | 1    | Test validation / regression triage / behavior verification | `GPT-5.4-mini (copilot)`      | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`]                                | Keep current. Validation is usually mechanical, with GPT-5.4 reserved for harder triage.                                              |
| `06-documenting`   | 1    | JSDoc / API docs / generated README / changelogs            | `Claude Sonnet 4.6 (copilot)` | [`GPT-5.4-mini (copilot)`, `GPT-5.4 (copilot)`]                                    | Keep current. Documentation quality and explanatory synthesis justify Sonnet-first routing.                                           |
| `07-logging`       | 1    | Session summaries / tracker updates / logging               | `Claude Haiku 4.6 (copilot)`  | [`GPT-5.4-mini (copilot)`, `GPT-5.4 (copilot)`]                                    | Keep current. Logging is a lightweight summarization surface and should stay inexpensive.                                             |

### Tier 2 — Coordinators And Sub-Orchestrators

| Agent                                   | Tier | Job category                                                 | Current primary               | Current fallback array                                        | Recommendation / rationale                                                                                        |
| --------------------------------------- | ---- | ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `flappy-architecture-polish`            | 2    | Flappy Bird architecture profile tuning                      | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]     | Keep current. Tuning loops span code, metrics, and behavior adjustments, so Full-tier coding remains appropriate. |
| `green-test-failure-triage-coordinator` | 2    | Test failure ownership / coverage gate interpretation        | `GPT-5.4-mini (copilot)`      | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`]           | Keep current. Most triage is bounded classification work with occasional escalation.                              |
| `helping-agent-maintenance-coordinator` | 2    | Agent file maintenance / frontmatter repair                  | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]     | Keep current. Frontmatter repair is code-adjacent and benefits from stronger structured-edit reasoning.           |
| `helping-gap-resolution-coordinator`    | 2    | Missing specialist / routing gap resolution                  | `Claude Sonnet 4.6 (copilot)` | [`GPT-5.4 (copilot)`, `GPT-5.4-mini (copilot)`]               | Keep current. Gap resolution is ambiguity-heavy and often policy-driven.                                          |
| `implementation-pattern-coordinator`    | 2    | Pattern discovery / refactor routing / specialist assignment | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]     | Keep current. Routing implementation work still benefits from a coding-first lead model.                          |
| `planning-context-coordinator`          | 2    | Project context / ownership / README evidence for planning   | `GPT-5.4-mini (copilot)`      | [`Claude Haiku 4.6 (copilot)`, `Claude Sonnet 4.6 (copilot)`] | Keep current. This is bounded evidence gathering with a sensible synthesis fallback.                              |
| `planning-risk-coordinator`             | 2    | Ambiguity review / blast-radius / reversibility analysis     | `Claude Sonnet 4.6 (copilot)` | [`GPT-5.4 (copilot)`, `GPT-5.4-mini (copilot)`]               | Keep current. Risk framing and policy tradeoffs fit Sonnet-first reasoning.                                       |
| `planning-test-strategy-coordinator`    | 2    | Acceptance criteria / red-test scope / coverage order        | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]     | Keep current. Test-strategy design is close to red-test authorship and benefits from stronger coding judgment.    |
| `research-codebase-coordinator`         | 2    | Multi-area source research / domain scout coordination       | `GPT-5.4-mini (copilot)`      | [`Claude Haiku 4.6 (copilot)`, `Claude Sonnet 4.6 (copilot)`] | Keep current. High-frequency coordination should stay budget-aware while keeping synthesis escalation.            |
| `solid-split`                           | 2    | SOLID module split / folderization / JSDoc / split plan      | `GPT-5.4 (copilot)`           | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`]     | Keep current. Large structural refactors are coding-heavy and need stronger implementation judgment.              |

### Tier 3 — Hidden Scouts And Specialists

| Agent                           | Tier | Job category                                                     | Current primary              | Current fallback array                                    | Recommendation / rationale                                                                                                                                 |
| ------------------------------- | ---- | ---------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `academic-docs-auditor`         | 3    | Educational docs audit / JSDoc / Mermaid / citations             | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Most audit passes are checklist-heavy, with Sonnet reserved for deeper prose judgment.                                                       |
| `agent-frontmatter-auditor`     | 3    | Agent frontmatter validation                                     | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Validation is narrow and mechanical, but Sonnet remains a safe ambiguity fallback.                                                           |
| `boundary-mapper`               | 3    | Module boundary mapping / refactor planning                      | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Initial boundary recon is lightweight and can escalate when the split is complex.                                                            |
| `browser-runtime-scout`         | 3    | Browser runtime / bundle / smoke-test recon                      | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Browser recon is usually bounded evidence gathering with occasional synthesis escalation.                                                    |
| `checkpoint-scout`              | 3    | Save/resume / checkpoint boundary mapping                        | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Most checkpoint scouting is structured mapping rather than deep implementation.                                                              |
| `cortex-embeddings-scout`       | 3    | ONNX embeddings / dense retrieval index recon                    | `Claude Haiku 4.6 (copilot)` | []                                                        | Keep current for now. Single-model routing is acceptable here because the scope is very narrow, but add a Mini fallback later if this agent broadens.      |
| `coverage-guard`                | 3    | Post-change 100% coverage enforcement                            | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Coverage enforcement is mostly checklist and command interpretation work.                                                                    |
| `coverage-scout`                | 3    | lcov gap identification / next tranche target                    | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. This is narrow evidence extraction, not full reasoning by default.                                                                           |
| `determinism-scout`             | 3    | RNG / seed / replay / ordering boundary mapping                  | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Determinism recon starts as structured checklist work and can escalate when contracts conflict.                                              |
| `docs-scout`                    | 3    | Generated README drift / JSDoc gap recon                         | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Most docs recon is cheap evidence gathering.                                                                                                 |
| `evaluation-pool-scout`         | 3    | Worker pool / queueing / ordered results recon                   | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Pool scouting is typically bounded and observational.                                                                                        |
| `failure-triage-specialist`     | 3    | Validation failure root-cause / reroute                          | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                           | Keep current. Triage is narrow and pattern-based unless it encounters deeper ambiguity.                                                                    |
| `hybrid-interop-scout`          | 3    | Parameter vector / fine-tuning / Lamarckian persistence recon    | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. This scout still benefits from moderate technical reasoning and tool use.                                                                    |
| `implementation-pattern-scout`  | 3    | Source patterns / naming conventions / helper boundaries recon   | `GPT-5.4-mini (copilot)`     | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`]       | Keep current. Pattern scouting is tool-heavy and fits Mini-first routing well.                                                                             |
| `license-attribution-auditor`   | 3    | Source attribution / license note checking                       | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Attribution checks need moderate reasoning more than prose depth.                                                                            |
| `mcp-runtime-scout`             | 3    | MCP runtime visibility gap mapping                               | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Runtime scouting is tool-centric and fits Mini-first routing.                                                                                |
| `mcp-server-architect`          | 3    | MCP server contracts / tool/resource schemas                     | `GPT-5.4 (copilot)`          | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`] | Keep current. Contract design is implementation-shaped and benefits from stronger coding depth.                                                            |
| `mcp-validation-auditor`        | 3    | MCP workflow / allow-list / plan packet validation               | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Validation and contract checks are bounded but still technical.                                                                              |
| `model-name-auditor`            | 3    | Qualified model name discovery and validation                    | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. This agent should stay cheap and precise, with a stronger fallback for tricky availability questions.                                        |
| `neatchat-scout`                | 3    | NEATchat memory / retrieval / session boundary recon             | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. System scouting needs more than checklist depth but not a default Full-tier lead.                                                            |
| `nge-benchmark-scout`           | 3    | NGE benchmark methodology / fairness / observability recon       | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Benchmark recon is technical but still bounded enough for Mini-first.                                                                        |
| `nge-core-scout`                | 3    | NGE DNA / lifecycle / neuromodulation boundary recon             | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. The domain is advanced, but the scout role is still bounded recon.                                                                           |
| `phase-handoff-designer`        | 3    | Sequential SDLC handoff design and audit                         | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Handoff shaping is structured and can escalate if policy tradeoffs get thorny.                                                               |
| `plan-registration-auditor`     | 3    | Plan registration / roadmap / tracker sync validation            | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Tracker sync checks are mostly mechanical and validation-oriented.                                                                           |
| `plan-scout`                    | 3    | Roadmap alignment / plan document selection                      | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Plan scouting is bounded read-and-route work that should stay inexpensive.                                                                   |
| `repo-cortex-scout`             | 3    | Cortex index freshness / corpus rebuild / MCP binding recon      | `Claude Haiku 4.6 (copilot)` | []                                                        | Keep current for now. Single-model routing is acceptable because the job is narrow and operational, though a Mini fallback would improve resilience later. |
| `skill-frontmatter-auditor`     | 3    | SKILL.md frontmatter / folder-name / visibility audit            | `GPT-5.4-mini (copilot)`     | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`]       | Keep current. The work is structured but still benefits from decent tool reasoning.                                                                        |
| `skill-inventory-auditor`       | 3    | Skills and agents inventory / drift evidence                     | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Inventory work is bounded and should remain budget-aware.                                                                                    |
| `skill-output-eval-grader`      | 3    | Skill output grading / assertion / baseline comparison           | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Grading is structured evaluation work with modest reasoning needs.                                                                           |
| `skill-trigger-eval-designer`   | 3    | Trigger evals / should-trigger / false-positive prevention       | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Eval design is bounded enough for Mini-first routing.                                                                                        |
| `unit-test-runner`              | 3    | Focused test execution / red-green result confirmation           | `GPT-5.4-mini (copilot)`     | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`]       | Keep current. Command-driven validation is a strong Mini use case.                                                                                         |
| `unit-test-writer`              | 3    | Unit test writing / fixtures / mocks / assertions                | `GPT-5.4 (copilot)`          | [`Claude Sonnet 4.6 (copilot)`, `GPT-5.4-mini (copilot)`] | Keep current. Test authorship is a coding-heavy specialist and fits Full-tier routing.                                                                     |
| `visualizer-scout`              | 3    | Visualizer UI / layout / hover / parity recon                    | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. UI scouting is investigative and tool-heavy but not usually Full-tier by default.                                                            |
| `vscode-ai-extensibility-scout` | 3    | VS Code AI extensibility / MCP / hooks / agent plugins recon     | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Extensibility recon is technical but bounded enough for Mini-first.                                                                          |
| `worker-payload-scout`          | 3    | Worker payload / structured clone / transfer-list boundary recon | `GPT-5.4-mini (copilot)`     | [`GPT-5.4 (copilot)`]                                     | Keep current. Payload scouting is tool-heavy and benefits from the Mini tier's cost profile.                                                               |

### Tier 4 — Auxiliaries And One-Shot Helpers

| Agent                        | Tier | Job category                                 | Current primary              | Current fallback array                              | Recommendation / rationale                                                                                         |
| ---------------------------- | ---- | -------------------------------------------- | ---------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `acceptance-criteria-writer` | 4    | Acceptance criteria authorship               | `GPT-5.4-mini (copilot)`     | [`Claude Haiku 4.6 (copilot)`, `GPT-5.4 (copilot)`] | Keep current. This is short-form structured writing and does not need a default Full-tier lead.                    |
| `docs-example-writer`        | 4    | JSDoc examples / README usage snippets       | `Claude Haiku 4.6 (copilot)` | [`Claude Sonnet 4.6 (copilot)`]                     | Keep current. Example snippets are narrow, repetitive, and cheap unless prose quality becomes unusually important. |
| `file-change-summarizer`     | 4    | Changed file summarization / logging handoff | `Claude Haiku 4.6 (copilot)` | [`GPT-5.4-mini (copilot)`, `GPT-5.4 (copilot)`]     | Keep current. Summarization is the clearest lightweight-helper use case in the fleet.                              |
| `learning-event-capturer`    | 4    | ISO-42001-style learning event capture       | `Claude Haiku 4.6 (copilot)` | [`GPT-5.4-mini (copilot)`, `GPT-5.4 (copilot)`]     | Keep current. Structured evidence capture should remain low-cost by default.                                       |

## Current Upgrade Posture

- Treat `GPT-5.5`, `Claude Opus 4.6`, `Claude Opus 4.7`, `GPT-5.4 nano`, and
  any other candidate upgrade as unverified for `.agent.md` frontmatter until
  the exact qualified `(copilot)` string is confirmed locally.
- Do not adopt a higher-cost model solely because it appears in public Copilot
  docs; local qualified-name validation is the controlling requirement.
- Revisit this file when the active client exposes additional qualified names
  that can be validated with `model-name-auditor` and
  `validate-agent-frontmatter.mjs`.
