# NeatapticTS Agent Inventory

> Generated: 2026-06-03 | Source: `.github/agents/*.agent.md` | Total: 57 agents

---

## Full Agent Table

| filename | name | tier | user-invocable | model | skills | job category |
|---|---|---|---|---|---|---|
| 00-helping.agent.md | 00-helping | 1 | true | glm-5.2:cloud (ollama) | agent-frontmatter-standards, model-routing-and-budget, agent-inventory-audit, subagent-delegation-patterns | AI system maintenance / workflow gap resolution |
| 01-planning.agent.md | 01-planning | 1 | true | glm-5.2:cloud (ollama) | plan-alignment, tracker-handoff, phase-handoff-workflow, agent-frontmatter-standards, model-routing-and-budget, license-attribution-audit | Planning / decomposition / acceptance criteria |
| 02-researching.agent.md | 02-researching | 1 | true | glm-5.2:cloud (ollama) | subagent-delegation-patterns | Codebase research / API exploration / architecture recon |
| 03-red-testing.agent.md | 03-red-testing | 1 | true | glm-5.2:cloud (ollama) | red-test-contracts, test-fix-workflow, coverage-tranche | Red test creation / failing test authorship |
| 04-implementing.agent.md | 04-implementing | 1 | true | glm-5.2:cloud (ollama) | _(empty)_ | Scoped code implementation |
| 05-green-testing.agent.md | 05-green-testing | 1 | true | glm-5.2:cloud (ollama) | green-validation-gates, coverage-guard, plan-sync-validation | Test validation / regression triage / behavior verification |
| 06-documenting.agent.md | 06-documenting | 1 | true | glm-5.2:cloud (ollama) | educational-docs, docs-academic-citation-audit, license-attribution-audit | JSDoc / API docs / generated README / changelogs |
| 07-logging.agent.md | 07-logging | 1 | true | Claude Haiku 4.6 (copilot) | tracker-handoff, plan-sync-validation, capturing-learning-event | Session summaries / tracker updates / logging |
| academic-docs-auditor.agent.md | academic-docs-auditor | 3 | false | Claude Haiku 4.6 (copilot) | docs-academic-citation-audit | Educational docs audit / JSDoc / Mermaid / citations |
| acceptance-criteria-writer.agent.md | acceptance-criteria-writer | 4 | false | glm-5.2:cloud (ollama) | planning-acceptance-criteria | Acceptance criteria authorship |
| agent-frontmatter-auditor.agent.md | agent-frontmatter-auditor | 3 | false | Claude Haiku 4.6 (copilot) | agent-frontmatter-standards | Agent frontmatter validation |
| boundary-mapper.agent.md | boundary-mapper | 3 | false | Claude Haiku 4.6 (copilot) | solid-split | Module boundary mapping / refactor planning |
| browser-runtime-scout.agent.md | browser-runtime-scout | 3 | false | Claude Haiku 4.6 (copilot) | browser-build | Browser runtime / bundle / smoke-test recon |
| checkpoint-scout.agent.md | checkpoint-scout | 3 | false | Claude Haiku 4.6 (copilot) | checkpointing-persistence | Save/resume / checkpoint boundary mapping |
| cortex-embeddings-scout.agent.md | cortex-embeddings-scout | 3 | false | Claude Haiku 4.6 (copilot) | _(empty)_ | ONNX embeddings / dense retrieval index recon |
| coverage-guard.agent.md | coverage-guard | 3 | false | Claude Haiku 4.6 (copilot) | coverage-guard | Post-change 100% coverage enforcement |
| coverage-scout.agent.md | coverage-scout | 3 | false | Claude Haiku 4.6 (copilot) | coverage-tranche | lcov gap identification / next tranche target |
| determinism-scout.agent.md | determinism-scout | 3 | false | Claude Haiku 4.6 (copilot) | reproducibility-contracts | RNG / seed / replay / ordering boundary mapping |
| docs-example-writer.agent.md | docs-example-writer | 4 | false | Claude Haiku 4.6 (copilot) | _(empty)_ | JSDoc examples / README usage snippets |
| docs-scout.agent.md | docs-scout | 3 | false | Claude Haiku 4.6 (copilot) | educational-docs | Generated README drift / JSDoc gap recon |
| evaluation-pool-scout.agent.md | evaluation-pool-scout | 3 | false | Claude Haiku 4.6 (copilot) | multithread-evaluation | Worker pool / queueing / ordered results recon |
| failure-triage-specialist.agent.md | failure-triage-specialist | 3 | false | Claude Haiku 4.6 (copilot) | triaging-test-failures | Validation failure root-cause / reroute |
| file-change-summarizer.agent.md | file-change-summarizer | 4 | false | Claude Haiku 4.6 (copilot) | summarizing-session-log | Changed file summarization / logging handoff |
| flappy-architecture-polish.agent.md | flappy-architecture-polish | 2 | false | glm-5.2:cloud (ollama) | flappy-architecture-polish | Flappy Bird architecture profile tuning |
| green-test-failure-triage-coordinator.agent.md | green-test-failure-triage-coordinator | 2 | false | glm-5.2:cloud (ollama) | green-validation-gates | Test failure ownership / coverage gate interpretation |
| helping-agent-maintenance-coordinator.agent.md | helping-agent-maintenance-coordinator | 2 | false | glm-5.2:cloud (ollama) | agent-frontmatter-standards, model-routing-and-budget, agent-inventory-audit | Agent file maintenance / frontmatter repair |
| helping-gap-resolution-coordinator.agent.md | helping-gap-resolution-coordinator | 2 | false | Claude Sonnet 4.6 (copilot) | agent-frontmatter-standards, model-routing-and-budget, agent-inventory-audit, subagent-delegation-patterns | Missing specialist / routing gap resolution |
| hybrid-interop-scout.agent.md | hybrid-interop-scout | 3 | false | glm-5.2:cloud (ollama) | hybrid-training-interop | Parameter vector / fine-tuning / Lamarckian persistence recon |
| implementation-pattern-coordinator.agent.md | implementation-pattern-coordinator | 2 | false | glm-5.2:cloud (ollama) | subagent-delegation-patterns | Pattern discovery / refactor routing / specialist assignment |
| implementation-pattern-scout.agent.md | implementation-pattern-scout | 3 | false | glm-5.2:cloud (ollama) | _(empty)_ | Source patterns / naming conventions / helper boundaries recon |
| learning-event-capturer.agent.md | learning-event-capturer | 4 | false | Claude Haiku 4.6 (copilot) | capturing-learning-event | ISO-42001-style learning event capture |
| license-attribution-auditor.agent.md | license-attribution-auditor | 3 | false | glm-5.2:cloud (ollama) | license-attribution-audit | Source attribution / license note checking |
| mcp-runtime-scout.agent.md | mcp-runtime-scout | 3 | false | glm-5.2:cloud (ollama) | mcp-local-server-workflow | MCP runtime visibility gap mapping |
| mcp-server-architect.agent.md | mcp-server-architect | 3 | false | glm-5.2:cloud (ollama) | mcp-local-server-workflow | MCP server contracts / tool/resource schemas |
| mcp-validation-auditor.agent.md | mcp-validation-auditor | 3 | false | glm-5.2:cloud (ollama) | mcp-local-server-workflow | MCP workflow / allow-list / plan packet validation |
| model-name-auditor.agent.md | model-name-auditor | 3 | false | glm-5.2:cloud (ollama) | model-routing-and-budget | Qualified model name discovery and validation |
| neatchat-scout.agent.md | neatchat-scout | 3 | false | glm-5.2:cloud (ollama) | neatchat-systems | NEATchat memory / retrieval / session boundary recon |
| nge-benchmark-scout.agent.md | nge-benchmark-scout | 3 | false | glm-5.2:cloud (ollama) | nge-benchmark-workflow | NGE benchmark methodology / fairness / observability recon |
| nge-core-scout.agent.md | nge-core-scout | 3 | false | glm-5.2:cloud (ollama) | nge-core-algorithm | NGE DNA / lifecycle / neuromodulation boundary recon |
| phase-handoff-designer.agent.md | phase-handoff-designer | 3 | false | glm-5.2:cloud (ollama) | phase-handoff-workflow | Sequential SDLC handoff design and audit |
| plan-registration-auditor.agent.md | plan-registration-auditor | 3 | false | glm-5.2:cloud (ollama) | plan-sync-validation | Plan registration / roadmap / tracker sync validation |
| plan-scout.agent.md | plan-scout | 3 | false | glm-5.2:cloud (ollama) | plan-alignment | Roadmap alignment / plan document selection |
| planning-context-coordinator.agent.md | planning-context-coordinator | 2 | false | glm-5.2:cloud (ollama) | plan-alignment | Project context / ownership / README evidence for planning |
| planning-risk-coordinator.agent.md | planning-risk-coordinator | 2 | false | Claude Sonnet 4.6 (copilot) | model-routing-and-budget, license-attribution-audit | Ambiguity review / blast-radius / reversibility analysis |
| planning-test-strategy-coordinator.agent.md | planning-test-strategy-coordinator | 2 | false | glm-5.2:cloud (ollama) | planning-acceptance-criteria, red-test-contracts | Acceptance criteria / red-test scope / coverage order |
| repo-cortex-scout.agent.md | repo-cortex-scout | 3 | false | Claude Haiku 4.6 (copilot) | repo-cortex-workflow | Cortex index freshness / corpus rebuild / MCP binding recon |
| research-codebase-coordinator.agent.md | research-codebase-coordinator | 2 | false | glm-5.2:cloud (ollama) | subagent-delegation-patterns | Multi-area source research / domain scout coordination |
| skill-frontmatter-auditor.agent.md | skill-frontmatter-auditor | 3 | false | glm-5.2:cloud (ollama) | skill-frontmatter-standards | SKILL.md frontmatter / folder-name / visibility audit |
| skill-inventory-auditor.agent.md | skill-inventory-auditor | 3 | false | glm-5.2:cloud (ollama) | agent-inventory-audit | Skills and agents inventory / drift evidence |
| skill-output-eval-grader.agent.md | skill-output-eval-grader | 3 | false | glm-5.2:cloud (ollama) | skill-output-evals | Skill output grading / assertion / baseline comparison |
| skill-trigger-eval-designer.agent.md | skill-trigger-eval-designer | 3 | false | glm-5.2:cloud (ollama) | skill-description-evals | Trigger evals / should-trigger / false-positive prevention |
| solid-split.agent.md | solid-split | 2 | false | glm-5.2:cloud (ollama) | solid-split | SOLID module split / folderization / JSDoc / split plan |
| unit-test-runner.agent.md | unit-test-runner | 3 | false | glm-5.2:cloud (ollama) | running-unit-tests | Focused test execution / red-green result confirmation |
| unit-test-writer.agent.md | unit-test-writer | 3 | false | glm-5.2:cloud (ollama) | creating-unit-tests | Unit test writing / fixtures / mocks / assertions |
| visualizer-scout.agent.md | visualizer-scout | 3 | false | glm-5.2:cloud (ollama) | visualizer-workflow | Visualizer UI / layout / hover / parity recon |
| vscode-ai-extensibility-scout.agent.md | vscode-ai-extensibility-scout | 3 | false | glm-5.2:cloud (ollama) | _(empty)_ | VS Code AI extensibility / MCP / hooks / agent plugins recon |
| worker-payload-scout.agent.md | worker-payload-scout | 3 | false | glm-5.2:cloud (ollama) | worker-inference-transport | Worker payload / structured clone / transfer-list boundary recon |

---

## Aggregate Counts

### Model String Frequency

All 57 agents now use scalar `model:` strings. Every agent has a full `tier:` field and a full `model:` field — no agents have missing fields.

| Qualified Model String | Agents Using This Model | % of 57 agents |
|---|---|---|
| `glm-5.2:cloud (ollama)` | 31 | 54.4% |
| `Claude Haiku 4.6 (copilot)` | 17 | 29.8% |
| `glm-5.2:cloud (ollama)` | 7 | 12.3% |
| `Claude Sonnet 4.6 (copilot)` | 2 | 3.5% |

### Agents Per Tier

| Tier | Label | Count |
|---|---|---|
| 1 | SDLC Orchestrators (user-invocable) | 8 |
| 2 | Named Coordinators / Sub-Orchestrators | 10 |
| 3 | Hidden Scouts & Specialists | 35 |
| 4 | Auxiliaries & One-Shot Helpers | 4 |
| **Total** | | **57** |

### Validation Status

- ✅ All 57 agents have `name`, `tier`, `user-invocable`, `model`, and `skills` fields.
- ✅ `user-invocable: true` appears only on Tier 1 agents (8/8).
- ✅ All agents now use scalar `model:` strings compatible with the Copilot CLI.
- ✅ No agents are missing a `model:` field.
- ✅ No agents are missing a `tier:` field.
- ⚠️ `cortex-embeddings-scout`, `implementation-pattern-scout`, `vscode-ai-extensibility-scout`, and `docs-example-writer` have empty `skills` arrays — expected per their narrow recon scope.

### Current Model Usage Summary

The fleet currently uses these scalar model strings:

1. `glm-5.2:cloud (ollama)`
2. `Claude Haiku 4.6 (copilot)`
3. `glm-5.2:cloud (ollama)`
4. `Claude Sonnet 4.6 (copilot)`
