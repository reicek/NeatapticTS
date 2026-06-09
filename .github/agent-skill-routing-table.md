<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: e243b0f3281fc7e9f041b709dc0fce8c44d62bf3a92ed998088608a4bf5b8a72 -->
<!-- source-file-count: 117 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| 00-helping | 1 | glm-5.1:cloud (ollama) | - | - |
| 01-planning | 1 | glm-5.1:cloud (ollama) | - | - |
| 02-researching | 1 | glm-5.1:cloud (ollama) | - | - |
| 03-red-testing | 1 | glm-5.1:cloud (ollama) | - | red-test-contracts<br>test-fix-workflow<br>coverage-tranche |
| 04-implementing | 1 | glm-5.1:cloud (ollama) | - | - |
| 05-green-testing | 1 | glm-5.1:cloud (ollama) | - | - |
| 06-documenting | 1 | glm-5.1:cloud (ollama) | - | - |
| 07-logging | 1 | glm-5.1:cloud (ollama) | - | tracker-handoff<br>plan-sync-validation<br>capturing-learning-event |
| academic-docs-auditor | 3 | glm-5.1:cloud (ollama) | - | docs-academic-citation-audit<br>auditing-js-docs |
| acceptance-criteria-writer | 4 | glm-5.1:cloud (ollama) | - | planning-acceptance-criteria |
| agent-frontmatter-auditor | 3 | glm-5.1:cloud (ollama) | - | agent-frontmatter-standards<br>updating-agent-frontmatter |
| boundary-mapper | 3 | glm-5.1:cloud (ollama) | - | solid-split |
| browser-runtime-scout | 3 | glm-5.1:cloud (ollama) | - | browser-build |
| checkpoint-scout | 3 | glm-5.1:cloud (ollama) | - | checkpointing-persistence |
| code-quality-auditor | 3 | glm-5.1:cloud (ollama) | file-change-summarizer | green-validation-gates<br>implementation-standards |
| cortex-embeddings-scout | 3 | glm-5.1:cloud (ollama) | - | repo-cortex-embeddings |
| coverage-guard | 3 | glm-5.1:cloud (ollama) | - | coverage-guard |
| coverage-scout | 3 | glm-5.1:cloud (ollama) | - | coverage-tranche<br>coverage-guard |
| determinism-scout | 3 | glm-5.1:cloud (ollama) | - | reproducibility-contracts |
| docs-example-writer | 4 | glm-5.1:cloud (ollama) | - | educational-docs |
| docs-scout | 3 | glm-5.1:cloud (ollama) | - | educational-docs |
| evaluation-pool-scout | 3 | glm-5.1:cloud (ollama) | - | multithread-evaluation |
| failure-triage-specialist | 3 | glm-5.1:cloud (ollama) | - | triaging-test-failures |
| file-change-summarizer | 4 | glm-5.1:cloud (ollama) | - | summarizing-session-log |
| flappy-architecture-polish | 2 | glm-5.1:cloud (ollama) | plan-scout | flappy-architecture-polish<br>architecture-builder |
| green-test-failure-triage-coordinator | 2 | glm-5.1:cloud (ollama) | - | green-validation-gates |
| helping-agent-maintenance-coordinator | 2 | glm-5.1:cloud (ollama) | - | - |
| helping-gap-resolution-coordinator | 2 | glm-5.1:cloud (ollama) | - | - |
| hybrid-interop-scout | 3 | glm-5.1:cloud (ollama) | - | hybrid-training-interop |
| implementation-executor | 2 | glm-5.1:cloud (ollama) | - | implementation-standards<br>coverage-guard |
| implementation-pattern-coordinator | 2 | glm-5.1:cloud (ollama) | - | subagent-delegation-patterns |
| implementation-pattern-scout | 3 | glm-5.1:cloud (ollama) | - | implementation-standards |
| learning-event-capturer | 4 | glm-5.1:cloud (ollama) | - | capturing-learning-event |
| license-attribution-auditor | 3 | glm-5.1:cloud (ollama) | - | license-attribution-audit |
| mcp-runtime-scout | 3 | glm-5.1:cloud (ollama) | - | mcp-local-server-workflow |
| mcp-server-architect | 3 | glm-5.1:cloud (ollama) | - | mcp-local-server-workflow |
| mcp-validation-auditor | 3 | glm-5.1:cloud (ollama) | - | mcp-local-server-workflow |
| model-name-auditor | 3 | glm-5.1:cloud (ollama) | - | model-routing-and-budget |
| neatchat-scout | 3 | glm-5.1:cloud (ollama) | - | neatchat-systems |
| nge-benchmark-scout | 3 | glm-5.1:cloud (ollama) | - | nge-benchmark-workflow |
| nge-core-scout | 3 | glm-5.1:cloud (ollama) | - | nge-core-algorithm |
| phase-handoff-designer | 3 | glm-5.1:cloud (ollama) | - | phase-handoff-workflow |
| plan-registration-auditor | 3 | glm-5.1:cloud (ollama) | - | plan-sync-validation |
| plan-scout | 3 | glm-5.1:cloud (ollama) | - | plan-alignment |
| planning-context-coordinator | 2 | glm-5.1:cloud (ollama) | plan-scout<br>docs-scout<br>boundary-mapper | plan-alignment |
| planning-risk-coordinator | 2 | glm-5.1:cloud (ollama) | - | model-routing-and-budget<br>license-attribution-audit |
| planning-test-strategy-coordinator | 2 | glm-5.1:cloud (ollama) | - | planning-acceptance-criteria<br>red-test-contracts |
| repo-cortex-scout | 3 | glm-5.1:cloud (ollama) | - | repo-cortex-workflow |
| research-codebase-coordinator | 2 | glm-5.1:cloud (ollama) | - | subagent-delegation-patterns<br>repo-cortex-workflow |
| research-synthesis-specialist | 3 | glm-5.1:cloud (ollama) | acceptance-criteria-writer<br>file-change-summarizer | research-methodology<br>plan-alignment |
| skill-frontmatter-auditor | 3 | glm-5.1:cloud (ollama) | - | skill-frontmatter-standards<br>updating-skill-frontmatter |
| skill-inventory-auditor | 3 | glm-5.1:cloud (ollama) | - | agent-inventory-audit |
| skill-output-eval-grader | 3 | glm-5.1:cloud (ollama) | - | skill-output-evals |
| skill-trigger-eval-designer | 3 | glm-5.1:cloud (ollama) | - | skill-description-evals |
| solid-split | 2 | glm-5.1:cloud (ollama) | boundary-mapper<br>plan-scout<br>docs-scout | solid-split |
| test-coverage-analyst | 3 | glm-5.1:cloud (ollama) | - | coverage-guard<br>coverage-tranche |
| unit-test-runner | 3 | glm-5.1:cloud (ollama) | - | running-unit-tests |
| unit-test-writer | 3 | glm-5.1:cloud (ollama) | - | creating-unit-tests |
| visualizer-scout | 3 | glm-5.1:cloud (ollama) | - | visualizer-workflow |
| vscode-ai-extensibility-scout | 3 | glm-5.1:cloud (ollama) | - | - |
| worker-payload-scout | 3 | glm-5.1:cloud (ollama) | - | worker-inference-transport |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | agent-frontmatter-auditor | self |
| agent-inventory-audit | skill | - | skill-inventory-auditor | self |
| agent-json-body-to-md | skill | - | - | self |
| agent-script-tooling | skill | - | - | self |
| architecture-builder | skill | - | flappy-architecture-polish | self |
| auditing-js-docs | skill | - | academic-docs-auditor | self |
| browser-build | skill | - | browser-runtime-scout | self |
| capturing-learning-event | skill | - | 07-logging<br>learning-event-capturer | self |
| checkpointing-persistence | skill | - | checkpoint-scout | self |
| coverage-guard | skill | - | coverage-guard<br>coverage-scout<br>implementation-executor<br>test-coverage-analyst | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-scout<br>test-coverage-analyst | self |
| creating-specialist-agent | skill | - | - | self |
| creating-unit-tests | skill | - | unit-test-writer | self |
| docs-academic-citation-audit | skill | - | academic-docs-auditor | self |
| educational-docs | skill | - | docs-example-writer<br>docs-scout | self |
| flappy-architecture-polish | skill | - | flappy-architecture-polish | self |
| green-validation-gates | skill | - | code-quality-auditor<br>green-test-failure-triage-coordinator | self |
| hybrid-training-interop | skill | - | hybrid-interop-scout | self |
| implementation-standards | skill | - | code-quality-auditor<br>implementation-executor<br>implementation-pattern-scout | self |
| license-attribution-audit | skill | - | license-attribution-auditor<br>planning-risk-coordinator | self |
| mcp-local-server-workflow | skill | - | mcp-runtime-scout<br>mcp-server-architect<br>mcp-validation-auditor | self |
| model-routing-and-budget | skill | - | model-name-auditor<br>planning-risk-coordinator | self |
| multithread-evaluation | skill | - | evaluation-pool-scout | self |
| neatchat-systems | skill | - | neatchat-scout | self |
| nge-benchmark-workflow | skill | - | nge-benchmark-scout | self |
| nge-core-algorithm | skill | - | nge-core-scout | self |
| onnx-work | skill | - | - | self |
| performance-optimization | skill | - | - | self |
| phase-handoff-workflow | skill | - | phase-handoff-designer | self |
| plan-alignment | skill | - | plan-scout<br>planning-context-coordinator<br>research-synthesis-specialist | self |
| plan-sync-validation | skill | - | 07-logging<br>plan-registration-auditor | self |
| planning-acceptance-criteria | skill | - | acceptance-criteria-writer<br>planning-test-strategy-coordinator | self |
| red-test-contracts | skill | - | 03-red-testing<br>planning-test-strategy-coordinator | self |
| repo-cortex-embeddings | skill | - | cortex-embeddings-scout | self |
| repo-cortex-workflow | skill | - | repo-cortex-scout<br>research-codebase-coordinator | self |
| reproducibility-contracts | skill | - | determinism-scout | self |
| research-methodology | skill | - | research-synthesis-specialist | self |
| routing-optimization-policy | skill | - | - | self |
| running-unit-tests | skill | - | unit-test-runner | self |
| skill-description-evals | skill | - | skill-trigger-eval-designer | self |
| skill-frontmatter-standards | skill | - | skill-frontmatter-auditor | self |
| skill-output-evals | skill | - | skill-output-eval-grader | self |
| solid-split | skill | - | boundary-mapper<br>solid-split | self |
| splitting-monolithic-agent | skill | - | - | self |
| subagent-delegation-patterns | skill | - | implementation-pattern-coordinator<br>research-codebase-coordinator | self |
| summarizing-session-log | skill | - | file-change-summarizer | self |
| test-fix-workflow | skill | - | 03-red-testing | self |
| trace-analyzer-extension | skill | - | - | self |
| trace-audit-reporting | skill | - | - | self |
| tracker-handoff | skill | - | 07-logging | self |
| triaging-test-failures | skill | - | failure-triage-specialist | self |
| updating-agent-frontmatter | skill | - | agent-frontmatter-auditor | self |
| updating-js-docs | skill | - | - | self |
| updating-skill-frontmatter | skill | - | skill-frontmatter-auditor | self |
| visualizer-workflow | skill | - | visualizer-scout | self |
| worker-inference-transport | skill | - | worker-payload-scout | self |
