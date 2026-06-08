<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: 87569999c29c845a7bdae62c9d1d26461db368192988097790b4f54185a76a55 -->
<!-- source-file-count: 109 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| 00-helping | 1 | gemma4:latest (ollama) | helping-gap-resolution-coordinator<br>helping-agent-maintenance-coordinator<br>skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>skill-trigger-eval-designer<br>skill-output-eval-grader<br>coverage-guard<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| 01-planning | 1 | gemma4:latest (ollama) | planning-context-coordinator<br>planning-risk-coordinator<br>planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>plan-scout<br>model-name-auditor<br>plan-registration-auditor<br>helping-gap-resolution-coordinator | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit |
| 02-researching | 1 | gemma4:latest (ollama) | research-codebase-coordinator<br>plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>skill-inventory-auditor<br>helping-gap-resolution-coordinator | subagent-delegation-patterns |
| 03-red-testing | 1 | gemma4:latest (ollama) | planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>unit-test-writer<br>coverage-scout<br>determinism-scout<br>plan-scout<br>helping-gap-resolution-coordinator | red-test-contracts<br>test-fix-workflow<br>coverage-tranche |
| 04-implementing | 1 | gemma4:latest (ollama) | implementation-pattern-coordinator<br>boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>solid-split<br>flappy-architecture-polish<br>agent-frontmatter-auditor<br>phase-handoff-designer<br>mcp-server-architect<br>helping-gap-resolution-coordinator | - |
| 05-green-testing | 1 | gemma4:latest (ollama) | green-test-failure-triage-coordinator<br>coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>determinism-scout<br>plan-registration-auditor<br>mcp-validation-auditor<br>helping-gap-resolution-coordinator | green-validation-gates<br>coverage-guard<br>plan-sync-validation |
| 06-documenting | 1 | gemma4:latest (ollama) | docs-scout<br>academic-docs-auditor<br>docs-example-writer<br>plan-scout<br>license-attribution-auditor<br>vscode-ai-extensibility-scout<br>helping-gap-resolution-coordinator | educational-docs<br>docs-academic-citation-audit<br>license-attribution-audit |
| 07-logging | 1 | gemma4:latest (ollama) | plan-scout<br>plan-registration-auditor<br>learning-event-capturer<br>file-change-summarizer<br>helping-gap-resolution-coordinator | tracker-handoff<br>plan-sync-validation<br>capturing-learning-event |
| academic-docs-auditor | 3 | gemma4:latest (ollama) | - | docs-academic-citation-audit |
| acceptance-criteria-writer | 4 | gemma4:latest (ollama) | - | planning-acceptance-criteria |
| agent-frontmatter-auditor | 3 | gemma4:latest (ollama) | - | agent-frontmatter-standards |
| boundary-mapper | 3 | gemma4:latest (ollama) | - | solid-split |
| browser-runtime-scout | 3 | gemma4:latest (ollama) | - | browser-build |
| checkpoint-scout | 3 | gemma4:latest (ollama) | - | checkpointing-persistence |
| cortex-embeddings-scout | 3 | gemma4:latest (ollama) | - | - |
| coverage-guard | 3 | gemma4:latest (ollama) | - | coverage-guard |
| coverage-scout | 3 | gemma4:latest (ollama) | - | coverage-tranche |
| determinism-scout | 3 | gemma4:latest (ollama) | - | reproducibility-contracts |
| docs-example-writer | 4 | gemma4:latest (ollama) | - | - |
| docs-scout | 3 | gemma4:latest (ollama) | - | educational-docs |
| evaluation-pool-scout | 3 | gemma4:latest (ollama) | - | multithread-evaluation |
| failure-triage-specialist | 3 | gemma4:latest (ollama) | - | triaging-test-failures |
| file-change-summarizer | 4 | gemma4:latest (ollama) | - | summarizing-session-log |
| flappy-architecture-polish | 2 | GPT-5.4 (copilot) | plan-scout | flappy-architecture-polish |
| green-test-failure-triage-coordinator | 2 | gemma4:latest (ollama) | coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>plan-registration-auditor<br>mcp-validation-auditor | green-validation-gates |
| helping-agent-maintenance-coordinator | 2 | GPT-5.4 (copilot) | agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>skill-inventory-auditor<br>model-name-auditor<br>learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit |
| helping-gap-resolution-coordinator | 2 | gemma4:latest (ollama) | skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| hybrid-interop-scout | 3 | gemma4:latest (ollama) | - | hybrid-training-interop |
| implementation-pattern-coordinator | 2 | GPT-5.4 (copilot) | implementation-pattern-scout<br>boundary-mapper<br>docs-scout<br>agent-frontmatter-auditor | subagent-delegation-patterns |
| implementation-pattern-scout | 3 | gemma4:latest (ollama) | - | - |
| learning-event-capturer | 4 | gemma4:latest (ollama) | - | capturing-learning-event |
| license-attribution-auditor | 3 | gemma4:latest (ollama) | - | license-attribution-audit |
| mcp-runtime-scout | 3 | gemma4:latest (ollama) | - | mcp-local-server-workflow |
| mcp-server-architect | 3 | GPT-5.4 (copilot) | - | mcp-local-server-workflow |
| mcp-validation-auditor | 3 | gemma4:latest (ollama) | - | mcp-local-server-workflow |
| model-name-auditor | 3 | gemma4:latest (ollama) | - | model-routing-and-budget |
| neatchat-scout | 3 | gemma4:latest (ollama) | - | neatchat-systems |
| nge-benchmark-scout | 3 | gemma4:latest (ollama) | - | nge-benchmark-workflow |
| nge-core-scout | 3 | gemma4:latest (ollama) | - | nge-core-algorithm |
| phase-handoff-designer | 3 | gemma4:latest (ollama) | - | phase-handoff-workflow |
| plan-registration-auditor | 3 | gemma4:latest (ollama) | - | plan-sync-validation |
| plan-scout | 3 | gemma4:latest (ollama) | - | plan-alignment |
| planning-context-coordinator | 2 | gemma4:latest (ollama) | plan-scout<br>docs-scout<br>boundary-mapper | plan-alignment |
| planning-risk-coordinator | 2 | gemma4:latest (ollama) | plan-scout<br>determinism-scout<br>license-attribution-auditor<br>model-name-auditor | model-routing-and-budget<br>license-attribution-audit |
| planning-test-strategy-coordinator | 2 | GPT-5.4 (copilot) | coverage-scout<br>determinism-scout<br>acceptance-criteria-writer<br>unit-test-writer | planning-acceptance-criteria<br>red-test-contracts |
| repo-cortex-scout | 3 | gemma4:latest (ollama) | - | repo-cortex-workflow |
| research-codebase-coordinator | 2 | gemma4:latest (ollama) | plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout | subagent-delegation-patterns |
| skill-frontmatter-auditor | 3 | gemma4:latest (ollama) | - | skill-frontmatter-standards |
| skill-inventory-auditor | 3 | gemma4:latest (ollama) | - | agent-inventory-audit |
| skill-output-eval-grader | 3 | gemma4:latest (ollama) | - | skill-output-evals |
| skill-trigger-eval-designer | 3 | gemma4:latest (ollama) | - | skill-description-evals |
| solid-split | 2 | GPT-5.4 (copilot) | boundary-mapper<br>plan-scout<br>docs-scout | solid-split |
| unit-test-runner | 3 | gemma4:latest (ollama) | - | running-unit-tests |
| unit-test-writer | 3 | GPT-5.4 (copilot) | - | creating-unit-tests |
| visualizer-scout | 3 | gemma4:latest (ollama) | - | visualizer-workflow |
| vscode-ai-extensibility-scout | 3 | gemma4:latest (ollama) | - | - |
| worker-payload-scout | 3 | gemma4:latest (ollama) | - | worker-inference-transport |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | 00-helping<br>01-planning<br>agent-frontmatter-auditor<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator | self |
| agent-inventory-audit | skill | - | 00-helping<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>skill-inventory-auditor | self |
| agent-json-body-to-md | skill | - | - | self |
| agent-script-tooling | skill | - | - | self |
| architecture-builder | skill | - | - | self |
| auditing-js-docs | skill | - | - | self |
| browser-build | skill | - | browser-runtime-scout | self |
| capturing-learning-event | skill | - | 07-logging<br>learning-event-capturer | self |
| checkpointing-persistence | skill | - | checkpoint-scout | self |
| coverage-guard | skill | - | 05-green-testing<br>coverage-guard | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-scout | self |
| creating-specialist-agent | skill | - | - | self |
| creating-unit-tests | skill | - | unit-test-writer | self |
| docs-academic-citation-audit | skill | - | 06-documenting<br>academic-docs-auditor | self |
| educational-docs | skill | - | 06-documenting<br>docs-scout | self |
| flappy-architecture-polish | skill | - | flappy-architecture-polish | self |
| green-validation-gates | skill | - | 05-green-testing<br>green-test-failure-triage-coordinator | self |
| hybrid-training-interop | skill | - | hybrid-interop-scout | self |
| license-attribution-audit | skill | - | 01-planning<br>06-documenting<br>license-attribution-auditor<br>planning-risk-coordinator | self |
| mcp-local-server-workflow | skill | - | mcp-runtime-scout<br>mcp-server-architect<br>mcp-validation-auditor | self |
| model-routing-and-budget | skill | - | 00-helping<br>01-planning<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>model-name-auditor<br>planning-risk-coordinator | self |
| multithread-evaluation | skill | - | evaluation-pool-scout | self |
| neatchat-systems | skill | - | neatchat-scout | self |
| nge-benchmark-workflow | skill | - | nge-benchmark-scout | self |
| nge-core-algorithm | skill | - | nge-core-scout | self |
| onnx-work | skill | - | - | self |
| performance-optimization | skill | - | - | self |
| phase-handoff-workflow | skill | - | 01-planning<br>phase-handoff-designer | self |
| plan-alignment | skill | - | 01-planning<br>plan-scout<br>planning-context-coordinator | self |
| plan-sync-validation | skill | - | 05-green-testing<br>07-logging<br>plan-registration-auditor | self |
| planning-acceptance-criteria | skill | - | acceptance-criteria-writer<br>planning-test-strategy-coordinator | self |
| red-test-contracts | skill | - | 03-red-testing<br>planning-test-strategy-coordinator | self |
| repo-cortex-workflow | skill | - | repo-cortex-scout | self |
| reproducibility-contracts | skill | - | determinism-scout | self |
| running-unit-tests | skill | - | unit-test-runner | self |
| skill-description-evals | skill | - | skill-trigger-eval-designer | self |
| skill-frontmatter-standards | skill | - | skill-frontmatter-auditor | self |
| skill-output-evals | skill | - | skill-output-eval-grader | self |
| solid-split | skill | - | boundary-mapper<br>solid-split | self |
| splitting-monolithic-agent | skill | - | - | self |
| subagent-delegation-patterns | skill | - | 00-helping<br>02-researching<br>helping-gap-resolution-coordinator<br>implementation-pattern-coordinator<br>research-codebase-coordinator | self |
| summarizing-session-log | skill | - | file-change-summarizer | self |
| test-fix-workflow | skill | - | 03-red-testing | self |
| trace-analyzer-extension | skill | - | - | self |
| trace-audit-reporting | skill | - | - | self |
| tracker-handoff | skill | - | 01-planning<br>07-logging | self |
| triaging-test-failures | skill | - | failure-triage-specialist | self |
| updating-agent-frontmatter | skill | - | - | self |
| updating-js-docs | skill | - | - | self |
| updating-skill-frontmatter | skill | - | - | self |
| visualizer-workflow | skill | - | visualizer-scout | self |
| worker-inference-transport | skill | - | worker-payload-scout | self |
