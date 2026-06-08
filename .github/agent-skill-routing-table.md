<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: 4503222ad35b5a6065a31fe1bf3f53d8f4697db2ef33063e85d428dafb4837d8 -->
<!-- source-file-count: 116 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| 00-helping | 1 | qwen3.5:cloud (ollama) | helping-gap-resolution-coordinator<br>helping-agent-maintenance-coordinator<br>skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>skill-trigger-eval-designer<br>skill-output-eval-grader<br>coverage-guard<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| 01-planning | 1 | qwen3.5:cloud (ollama) | planning-context-coordinator<br>planning-risk-coordinator<br>planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>plan-scout<br>model-name-auditor<br>plan-registration-auditor<br>helping-gap-resolution-coordinator<br>research-synthesis-specialist<br>phase-handoff-designer | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit |
| 02-researching | 1 | qwen3.5:cloud (ollama) | research-codebase-coordinator<br>plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>skill-inventory-auditor<br>helping-gap-resolution-coordinator | subagent-delegation-patterns |
| 03-red-testing | 1 | qwen3.5:cloud (ollama) | planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>unit-test-writer<br>coverage-scout<br>determinism-scout<br>plan-scout<br>helping-gap-resolution-coordinator | red-test-contracts<br>test-fix-workflow<br>coverage-tranche |
| 04-implementing | 1 | qwen3.5:cloud (ollama) | implementation-pattern-coordinator<br>implementation-executor<br>boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>solid-split<br>flappy-architecture-polish<br>agent-frontmatter-auditor<br>phase-handoff-designer<br>mcp-server-architect<br>helping-gap-resolution-coordinator | - |
| 05-green-testing | 1 | qwen3.5:cloud (ollama) | green-test-failure-triage-coordinator<br>coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>determinism-scout<br>plan-registration-auditor<br>mcp-validation-auditor<br>helping-gap-resolution-coordinator<br>code-quality-auditor<br>test-coverage-analyst | green-validation-gates<br>coverage-guard<br>plan-sync-validation |
| 06-documenting | 1 | qwen3.5:cloud (ollama) | docs-scout<br>academic-docs-auditor<br>docs-example-writer<br>plan-scout<br>license-attribution-auditor<br>vscode-ai-extensibility-scout<br>helping-gap-resolution-coordinator | educational-docs<br>docs-academic-citation-audit<br>license-attribution-audit |
| 07-logging | 1 | qwen3.5:cloud (ollama) | plan-scout<br>plan-registration-auditor<br>learning-event-capturer<br>file-change-summarizer<br>helping-gap-resolution-coordinator<br>phase-handoff-designer | tracker-handoff<br>plan-sync-validation<br>capturing-learning-event |
| academic-docs-auditor | 3 | qwen3.5:cloud (ollama) | - | docs-academic-citation-audit |
| acceptance-criteria-writer | 4 | qwen3.5:cloud (ollama) | - | planning-acceptance-criteria |
| agent-frontmatter-auditor | 3 | qwen3.5:cloud (ollama) | - | agent-frontmatter-standards |
| boundary-mapper | 3 | qwen3.5:cloud (ollama) | - | solid-split |
| browser-runtime-scout | 3 | qwen3.5:cloud (ollama) | - | browser-build |
| checkpoint-scout | 3 | qwen3.5:cloud (ollama) | - | checkpointing-persistence |
| code-quality-auditor | 3 | qwen3.5:cloud (ollama) | file-change-summarizer | green-validation-gates<br>implementation-standards |
| cortex-embeddings-scout | 3 | qwen3.5:cloud (ollama) | - | - |
| coverage-guard | 3 | qwen3.5:cloud (ollama) | - | coverage-guard |
| coverage-scout | 3 | qwen3.5:cloud (ollama) | - | coverage-tranche |
| determinism-scout | 3 | qwen3.5:cloud (ollama) | - | reproducibility-contracts |
| docs-example-writer | 4 | qwen3.5:cloud (ollama) | - | - |
| docs-scout | 3 | qwen3.5:cloud (ollama) | - | educational-docs |
| evaluation-pool-scout | 3 | qwen3.5:cloud (ollama) | - | multithread-evaluation |
| failure-triage-specialist | 3 | qwen3.5:cloud (ollama) | - | triaging-test-failures |
| file-change-summarizer | 4 | qwen3.5:cloud (ollama) | - | summarizing-session-log |
| flappy-architecture-polish | 2 | qwen3.5:cloud (ollama) | plan-scout | flappy-architecture-polish |
| green-test-failure-triage-coordinator | 2 | qwen3.5:cloud (ollama) | coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>plan-registration-auditor<br>mcp-validation-auditor | green-validation-gates |
| helping-agent-maintenance-coordinator | 2 | qwen3.5:cloud (ollama) | agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>skill-inventory-auditor<br>model-name-auditor<br>learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit |
| helping-gap-resolution-coordinator | 2 | qwen3.5:cloud (ollama) | skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| hybrid-interop-scout | 3 | qwen3.5:cloud (ollama) | - | hybrid-training-interop |
| implementation-executor | 2 | qwen3.5:cloud (ollama) | boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>checkpoint-scout<br>determinism-scout<br>helping-gap-resolution-coordinator | implementation-standards<br>coverage-guard |
| implementation-pattern-coordinator | 2 | qwen3.5:cloud (ollama) | implementation-pattern-scout<br>boundary-mapper<br>docs-scout<br>agent-frontmatter-auditor | subagent-delegation-patterns |
| implementation-pattern-scout | 3 | qwen3.5:cloud (ollama) | - | - |
| learning-event-capturer | 4 | qwen3.5:cloud (ollama) | - | capturing-learning-event |
| license-attribution-auditor | 3 | qwen3.5:cloud (ollama) | - | license-attribution-audit |
| mcp-runtime-scout | 3 | qwen3.5:cloud (ollama) | - | mcp-local-server-workflow |
| mcp-server-architect | 3 | qwen3.5:cloud (ollama) | - | mcp-local-server-workflow |
| mcp-validation-auditor | 3 | qwen3.5:cloud (ollama) | - | mcp-local-server-workflow |
| model-name-auditor | 3 | qwen3.5:cloud (ollama) | - | model-routing-and-budget |
| neatchat-scout | 3 | qwen3.5:cloud (ollama) | - | neatchat-systems |
| nge-benchmark-scout | 3 | qwen3.5:cloud (ollama) | - | nge-benchmark-workflow |
| nge-core-scout | 3 | qwen3.5:cloud (ollama) | - | nge-core-algorithm |
| phase-handoff-designer | 3 | qwen3.5:cloud (ollama) | - | phase-handoff-workflow |
| plan-registration-auditor | 3 | qwen3.5:cloud (ollama) | - | plan-sync-validation |
| plan-scout | 3 | qwen3.5:cloud (ollama) | - | plan-alignment |
| planning-context-coordinator | 2 | qwen3.5:cloud (ollama) | plan-scout<br>docs-scout<br>boundary-mapper | plan-alignment |
| planning-risk-coordinator | 2 | qwen3.5:cloud (ollama) | plan-scout<br>determinism-scout<br>license-attribution-auditor<br>model-name-auditor | model-routing-and-budget<br>license-attribution-audit |
| planning-test-strategy-coordinator | 2 | qwen3.5:cloud (ollama) | coverage-scout<br>determinism-scout<br>acceptance-criteria-writer<br>unit-test-writer | planning-acceptance-criteria<br>red-test-contracts |
| repo-cortex-scout | 3 | qwen3.5:cloud (ollama) | - | repo-cortex-workflow |
| research-codebase-coordinator | 2 | qwen3.5:cloud (ollama) | plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>research-synthesis-specialist | subagent-delegation-patterns |
| research-synthesis-specialist | 3 | qwen3.5:cloud (ollama) | acceptance-criteria-writer<br>file-change-summarizer | research-methodology<br>plan-alignment |
| skill-frontmatter-auditor | 3 | qwen3.5:cloud (ollama) | - | skill-frontmatter-standards |
| skill-inventory-auditor | 3 | qwen3.5:cloud (ollama) | - | agent-inventory-audit |
| skill-output-eval-grader | 3 | qwen3.5:cloud (ollama) | - | skill-output-evals |
| skill-trigger-eval-designer | 3 | qwen3.5:cloud (ollama) | - | skill-description-evals |
| solid-split | 2 | qwen3.5:cloud (ollama) | boundary-mapper<br>plan-scout<br>docs-scout | solid-split |
| test-coverage-analyst | 3 | qwen3.5:cloud (ollama) | - | coverage-guard<br>coverage-tranche |
| unit-test-runner | 3 | qwen3.5:cloud (ollama) | - | running-unit-tests |
| unit-test-writer | 3 | qwen3.5:cloud (ollama) | - | creating-unit-tests |
| visualizer-scout | 3 | qwen3.5:cloud (ollama) | - | visualizer-workflow |
| vscode-ai-extensibility-scout | 3 | qwen3.5:cloud (ollama) | - | - |
| worker-payload-scout | 3 | qwen3.5:cloud (ollama) | - | worker-inference-transport |

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
| coverage-guard | skill | - | 05-green-testing<br>coverage-guard<br>implementation-executor<br>test-coverage-analyst | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-scout<br>test-coverage-analyst | self |
| creating-specialist-agent | skill | - | - | self |
| creating-unit-tests | skill | - | unit-test-writer | self |
| docs-academic-citation-audit | skill | - | 06-documenting<br>academic-docs-auditor | self |
| educational-docs | skill | - | 06-documenting<br>docs-scout | self |
| flappy-architecture-polish | skill | - | flappy-architecture-polish | self |
| green-validation-gates | skill | - | 05-green-testing<br>code-quality-auditor<br>green-test-failure-triage-coordinator | self |
| hybrid-training-interop | skill | - | hybrid-interop-scout | self |
| implementation-standards | skill | - | code-quality-auditor<br>implementation-executor | self |
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
| plan-alignment | skill | - | 01-planning<br>plan-scout<br>planning-context-coordinator<br>research-synthesis-specialist | self |
| plan-sync-validation | skill | - | 05-green-testing<br>07-logging<br>plan-registration-auditor | self |
| planning-acceptance-criteria | skill | - | acceptance-criteria-writer<br>planning-test-strategy-coordinator | self |
| red-test-contracts | skill | - | 03-red-testing<br>planning-test-strategy-coordinator | self |
| repo-cortex-workflow | skill | - | repo-cortex-scout | self |
| reproducibility-contracts | skill | - | determinism-scout | self |
| research-methodology | skill | - | research-synthesis-specialist | self |
| routing-optimization-policy | skill | - | - | self |
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
