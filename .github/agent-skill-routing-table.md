<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: ddac23f9d041599b4c6b448c8910fc0758861dbc5a5b1bb4633d18d9cdca94a5 -->
<!-- source-file-count: 132 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| 00-helping | 1 | kimi-k2.7-code:cloud | helping-gap-resolution-coordinator<br>helping-agent-maintenance-coordinator<br>skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>skill-trigger-eval-designer<br>skill-output-eval-grader<br>coverage-guard<br>learning-event-capturer<br>file-change-summarizer<br>slice-orchestration-scheduler | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>customize-cloud-agent<br>subagent-delegation-patterns<br>capturing-learning-event<br>routing-optimization-policy<br>phase-handoff-workflow<br>tracker-handoff<br>execute |
| 01-planning | 1 | kimi-k2.7-code:cloud | planning-context-coordinator<br>planning-risk-coordinator<br>planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>plan-scout<br>model-name-auditor<br>plan-registration-auditor<br>helping-gap-resolution-coordinator<br>research-synthesis-specialist<br>phase-handoff-designer | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit<br>planning-acceptance-criteria<br>plan-sync-validation<br>spec-checklist<br>research-methodology<br>execute |
| 02-researching | 1 | kimi-k2.7-code:cloud | research-codebase-coordinator<br>plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>skill-inventory-auditor<br>helping-gap-resolution-coordinator<br>cortex-embeddings-scout | subagent-delegation-patterns<br>research-methodology<br>repo-cortex-workflow<br>execute |
| 03-red-testing | 1 | kimi-k2.7-code:cloud | planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>unit-test-writer<br>test-coverage-analyst<br>coverage-scout<br>determinism-scout<br>nge-core-scout<br>plan-scout<br>helping-gap-resolution-coordinator<br>performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist | red-test-contracts<br>nge-core-algorithm<br>reproducibility-contracts<br>creating-unit-tests<br>test-fix-workflow<br>coverage-tranche<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools |
| 04-implementing | 1 | kimi-k2.7-code:cloud | implementation-pattern-coordinator<br>implementation-pattern-scout<br>implementation-executor<br>boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>solid-split<br>flappy-architecture-polish<br>agent-frontmatter-auditor<br>phase-handoff-designer<br>mcp-server-architect<br>helping-gap-resolution-coordinator<br>browser-harness-specialist | implementation-standards<br>nge-core-algorithm<br>reproducibility-contracts<br>tracker-handoff<br>architecture-builder<br>onnx-work<br>performance-optimization<br>trace-analyzer-extension<br>worker-inference-transport<br>research-methodology<br>execute<br>browser-testing-harness |
| 05-green-testing | 1 | kimi-k2.7-code:cloud | green-test-failure-triage-coordinator<br>coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>determinism-scout<br>plan-registration-auditor<br>mcp-validation-auditor<br>helping-gap-resolution-coordinator<br>code-quality-auditor<br>test-coverage-analyst<br>performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist | green-validation-gates<br>coverage-guard<br>test-fix-workflow<br>plan-sync-validation<br>spec-checklist<br>trace-audit-reporting<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools |
| 06-documenting | 1 | kimi-k2.7-code:cloud | docs-scout<br>nge-core-scout<br>academic-docs-auditor<br>docs-example-writer<br>plan-scout<br>license-attribution-auditor<br>vscode-ai-extensibility-scout<br>helping-gap-resolution-coordinator<br>browser-harness-specialist | educational-docs<br>nge-core-algorithm<br>docs-academic-citation-audit<br>license-attribution-audit<br>auditing-js-docs<br>updating-js-docs<br>research-methodology<br>execute<br>browser-testing-harness |
| 07-logging | 1 | kimi-k2.7-code:cloud | plan-scout<br>plan-registration-auditor<br>learning-event-capturer<br>file-change-summarizer<br>helping-gap-resolution-coordinator<br>phase-handoff-designer | tracker-handoff<br>summarizing-session-log<br>plan-sync-validation<br>capturing-learning-event<br>research-methodology<br>execute |
| academic-docs-auditor | 3 | kimi-k2.7-code:cloud | - | docs-academic-citation-audit<br>auditing-js-docs |
| acceptance-criteria-writer | 4 | glm-5.2:cloud (ollama) | - | planning-acceptance-criteria<br>research-methodology |
| agent-frontmatter-auditor | 3 | kimi-k2.7-code:cloud | - | agent-frontmatter-standards<br>updating-agent-frontmatter |
| assimilator | 3 | kimi-k2.7-code:cloud | - | external-tool-assimilation<br>license-attribution-audit<br>research-methodology<br>capturing-learning-event |
| boundary-mapper | 3 | kimi-k2.7-code:cloud | - | solid-split<br>implementation-standards |
| browser-harness-specialist | 3 | kimi-k2.7-code:cloud | - | chrome-devtools-mcp<br>research-methodology |
| browser-memory-specialist | 3 | kimi-k2.7-code:cloud | - | chrome-devtools-mcp |
| browser-runtime-scout | 3 | kimi-k2.7-code:cloud | - | browser-build |
| browser-ui-specialist | 3 | kimi-k2.7-code:cloud | - | chrome-devtools-mcp |
| checkpoint-scout | 3 | kimi-k2.7-code:cloud | - | checkpointing-persistence |
| code-quality-auditor | 3 | kimi-k2.7-code:cloud | file-change-summarizer | green-validation-gates<br>implementation-standards |
| cortex-embeddings-scout | 3 | kimi-k2.7-code:cloud | - | repo-cortex-embeddings |
| coverage-guard | 3 | kimi-k2.7-code:cloud | - | coverage-guard |
| coverage-scout | 3 | kimi-k2.7-code:cloud | - | coverage-tranche<br>coverage-guard |
| determinism-scout | 3 | kimi-k2.7-code:cloud | - | reproducibility-contracts |
| docs-example-writer | 4 | kimi-k2.7-code:cloud | - | educational-docs |
| docs-scout | 3 | kimi-k2.7-code:cloud | - | educational-docs |
| evaluation-pool-scout | 3 | kimi-k2.7-code:cloud | - | multithread-evaluation |
| failure-triage-specialist | 3 | kimi-k2.7-code:cloud | - | triaging-test-failures<br>test-fix-workflow |
| file-change-summarizer | 4 | kimi-k2.7-code:cloud | - | summarizing-session-log |
| flappy-architecture-polish | 2 | kimi-k2.7-code:cloud | plan-scout | flappy-architecture-polish<br>architecture-builder<br>performance-optimization<br>execute |
| green-test-failure-triage-coordinator | 2 | kimi-k2.7-code:cloud | coverage-guard<br>coverage-scout<br>code-quality-auditor<br>failure-triage-specialist<br>unit-test-runner<br>plan-registration-auditor<br>mcp-validation-auditor | green-validation-gates<br>test-fix-workflow<br>execute |
| helping-agent-maintenance-coordinator | 2 | kimi-k2.7-code:cloud | agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>skill-inventory-auditor<br>model-name-auditor<br>learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>agent-json-body-to-md<br>agent-script-tooling<br>creating-specialist-agent<br>splitting-monolithic-agent<br>execute |
| helping-gap-resolution-coordinator | 2 | kimi-k2.7-code:cloud | skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>mcp-runtime-scout<br>model-name-auditor<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>creating-specialist-agent<br>subagent-delegation-patterns<br>execute |
| hybrid-interop-scout | 3 | kimi-k2.7-code:cloud | - | hybrid-training-interop |
| implementation-executor | 2 | kimi-k2.7-code:cloud | boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>checkpoint-scout<br>determinism-scout | implementation-standards<br>coverage-guard<br>execute |
| implementation-pattern-coordinator | 2 | kimi-k2.7-code:cloud | implementation-pattern-scout<br>boundary-mapper<br>docs-scout<br>agent-frontmatter-auditor | subagent-delegation-patterns<br>implementation-standards<br>execute |
| implementation-pattern-scout | 3 | kimi-k2.7-code:cloud | - | implementation-standards |
| learning-event-capturer | 4 | kimi-k2.7-code:cloud | - | capturing-learning-event |
| license-attribution-auditor | 3 | kimi-k2.7-code:cloud | - | license-attribution-audit |
| mcp-runtime-scout | 3 | kimi-k2.7-code:cloud | - | mcp-local-server-workflow |
| mcp-server-architect | 3 | kimi-k2.7-code:cloud | - | mcp-local-server-workflow |
| mcp-validation-auditor | 3 | kimi-k2.7-code:cloud | - | mcp-local-server-workflow |
| model-name-auditor | 3 | kimi-k2.7-code:cloud | - | model-routing-and-budget |
| neatchat-scout | 3 | kimi-k2.7-code:cloud | - | neatchat-systems |
| nge-benchmark-scout | 3 | kimi-k2.7-code:cloud | - | nge-benchmark-workflow |
| nge-core-scout | 3 | kimi-k2.7-code:cloud | - | nge-core-algorithm |
| performance-trace-specialist | 3 | kimi-k2.7-code:cloud | - | chrome-devtools-mcp<br>trace-audit-reporting<br>trace-analyzer-extension |
| phase-handoff-designer | 3 | kimi-k2.7-code:cloud | - | phase-handoff-workflow |
| plan-registration-auditor | 3 | kimi-k2.7-code:cloud | - | plan-sync-validation |
| plan-scout | 3 | kimi-k2.7-code:cloud | - | plan-alignment |
| planning-context-coordinator | 2 | kimi-k2.7-code:cloud | plan-scout<br>docs-scout<br>boundary-mapper | plan-alignment<br>execute |
| planning-risk-coordinator | 2 | kimi-k2.7-code:cloud | plan-scout<br>boundary-mapper<br>implementation-pattern-scout<br>determinism-scout<br>license-attribution-auditor<br>model-name-auditor | model-routing-and-budget<br>license-attribution-audit<br>planning-acceptance-criteria<br>execute |
| planning-test-strategy-coordinator | 2 | kimi-k2.7-code:cloud | coverage-scout<br>test-coverage-analyst<br>determinism-scout<br>acceptance-criteria-writer<br>unit-test-writer | planning-acceptance-criteria<br>red-test-contracts<br>execute |
| repo-cortex-scout | 3 | kimi-k2.7-code:cloud | - | repo-cortex-workflow |
| research-codebase-coordinator | 2 | kimi-k2.7-code:cloud | plan-scout<br>docs-scout<br>repo-cortex-scout<br>boundary-mapper<br>implementation-pattern-scout<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>research-synthesis-specialist | subagent-delegation-patterns<br>repo-cortex-workflow<br>research-methodology<br>execute |
| research-synthesis-specialist | 3 | kimi-k2.7-code:cloud | acceptance-criteria-writer<br>file-change-summarizer | research-methodology<br>plan-alignment |
| skill-frontmatter-auditor | 3 | kimi-k2.7-code:cloud | - | skill-frontmatter-standards<br>updating-skill-frontmatter |
| skill-inventory-auditor | 3 | kimi-k2.7-code:cloud | - | agent-inventory-audit |
| skill-output-eval-grader | 3 | kimi-k2.7-code:cloud | - | skill-output-evals |
| skill-trigger-eval-designer | 3 | kimi-k2.7-code:cloud | - | skill-description-evals |
| slice-orchestration-scheduler | 3 | kimi-k2.7-code:cloud | - | subagent-delegation-patterns<br>phase-handoff-workflow<br>tracker-handoff<br>execute |
| solid-split | 2 | kimi-k2.7-code:cloud | boundary-mapper<br>plan-scout<br>docs-scout | solid-split<br>implementation-standards<br>execute |
| test-coverage-analyst | 3 | kimi-k2.7-code:cloud | - | coverage-guard<br>coverage-tranche |
| unit-test-runner | 3 | kimi-k2.7-code:cloud | - | running-unit-tests |
| unit-test-writer | 3 | kimi-k2.7-code:cloud | - | creating-unit-tests<br>red-test-contracts |
| visualizer-scout | 3 | kimi-k2.7-code:cloud | - | visualizer-workflow |
| vscode-ai-extensibility-scout | 3 | kimi-k2.7-code:cloud | - | mcp-local-server-workflow |
| worker-payload-scout | 3 | kimi-k2.7-code:cloud | - | worker-inference-transport |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | 00-helping<br>01-planning<br>agent-frontmatter-auditor<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator | self |
| agent-inventory-audit | skill | - | 00-helping<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>skill-inventory-auditor | self |
| agent-json-body-to-md | skill | - | helping-agent-maintenance-coordinator | self |
| agent-script-tooling | skill | - | helping-agent-maintenance-coordinator | self |
| architecture-builder | skill | - | 04-implementing<br>flappy-architecture-polish | self |
| auditing-js-docs | skill | - | 06-documenting<br>academic-docs-auditor | self |
| browser-build | skill | - | browser-runtime-scout | self |
| browser-testing-harness | skill | - | 03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting | self |
| capturing-learning-event | skill | - | 00-helping<br>07-logging<br>assimilator<br>learning-event-capturer | self |
| checkpointing-persistence | skill | - | checkpoint-scout | self |
| chrome-devtools-mcp | skill | - | 03-red-testing<br>05-green-testing<br>browser-harness-specialist<br>browser-memory-specialist<br>browser-ui-specialist<br>performance-trace-specialist | self |
| coverage-guard | skill | - | 05-green-testing<br>coverage-guard<br>coverage-scout<br>implementation-executor<br>test-coverage-analyst | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-scout<br>test-coverage-analyst | self |
| creating-specialist-agent | skill | - | helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator | self |
| creating-unit-tests | skill | - | 03-red-testing<br>unit-test-writer | self |
| customize-cloud-agent | skill | - | 00-helping | self |
| devtools | skill | - | 03-red-testing<br>05-green-testing | self |
| docs-academic-citation-audit | skill | - | 06-documenting<br>academic-docs-auditor | self |
| educational-docs | skill | - | 06-documenting<br>docs-example-writer<br>docs-scout | self |
| execute | skill | - | 00-helping<br>01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>flappy-architecture-polish<br>green-test-failure-triage-coordinator<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>implementation-executor<br>implementation-pattern-coordinator<br>planning-context-coordinator<br>planning-risk-coordinator<br>planning-test-strategy-coordinator<br>research-codebase-coordinator<br>slice-orchestration-scheduler<br>solid-split | self |
| external-tool-assimilation | skill | - | assimilator | self |
| flappy-architecture-polish | skill | - | flappy-architecture-polish | self |
| green-validation-gates | skill | - | 05-green-testing<br>code-quality-auditor<br>green-test-failure-triage-coordinator | self |
| hybrid-training-interop | skill | - | hybrid-interop-scout | self |
| implementation-standards | skill | - | 04-implementing<br>boundary-mapper<br>code-quality-auditor<br>implementation-executor<br>implementation-pattern-coordinator<br>implementation-pattern-scout<br>solid-split | self |
| license-attribution-audit | skill | - | 01-planning<br>06-documenting<br>assimilator<br>license-attribution-auditor<br>planning-risk-coordinator | self |
| mcp-local-server-workflow | skill | - | mcp-runtime-scout<br>mcp-server-architect<br>mcp-validation-auditor<br>vscode-ai-extensibility-scout | self |
| model-routing-and-budget | skill | - | 00-helping<br>01-planning<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>model-name-auditor<br>planning-risk-coordinator | self |
| multithread-evaluation | skill | - | evaluation-pool-scout | self |
| neatchat-systems | skill | - | neatchat-scout | self |
| nge-benchmark-workflow | skill | - | nge-benchmark-scout | self |
| nge-core-algorithm | skill | - | 03-red-testing<br>04-implementing<br>06-documenting<br>nge-core-scout | self |
| onnx-work | skill | - | 04-implementing | self |
| performance-optimization | skill | - | 04-implementing<br>flappy-architecture-polish | self |
| phase-handoff-workflow | skill | - | 00-helping<br>01-planning<br>phase-handoff-designer<br>slice-orchestration-scheduler | self |
| plan-alignment | skill | - | 01-planning<br>plan-scout<br>planning-context-coordinator<br>research-synthesis-specialist | self |
| plan-sync-validation | skill | - | 01-planning<br>05-green-testing<br>07-logging<br>plan-registration-auditor | self |
| planning-acceptance-criteria | skill | - | 01-planning<br>acceptance-criteria-writer<br>planning-risk-coordinator<br>planning-test-strategy-coordinator | self |
| red-test-contracts | skill | - | 03-red-testing<br>planning-test-strategy-coordinator<br>unit-test-writer | self |
| repo-cortex-embeddings | skill | - | cortex-embeddings-scout | self |
| repo-cortex-workflow | skill | - | 02-researching<br>repo-cortex-scout<br>research-codebase-coordinator | self |
| reproducibility-contracts | skill | - | 03-red-testing<br>04-implementing<br>determinism-scout | self |
| research-methodology | skill | - | 01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>acceptance-criteria-writer<br>assimilator<br>browser-harness-specialist<br>research-codebase-coordinator<br>research-synthesis-specialist | self |
| routing-optimization-policy | skill | - | 00-helping | self |
| running-unit-tests | skill | - | unit-test-runner | self |
| skill-description-evals | skill | - | skill-trigger-eval-designer | self |
| skill-frontmatter-standards | skill | - | skill-frontmatter-auditor | self |
| skill-output-evals | skill | - | skill-output-eval-grader | self |
| solid-split | skill | - | boundary-mapper<br>solid-split | self |
| spec-checklist | skill | - | 01-planning<br>05-green-testing | self |
| specialist-review-workflow | skill | - | - | self |
| splitting-monolithic-agent | skill | - | helping-agent-maintenance-coordinator | self |
| subagent-delegation-patterns | skill | - | 00-helping<br>02-researching<br>helping-gap-resolution-coordinator<br>implementation-pattern-coordinator<br>research-codebase-coordinator<br>slice-orchestration-scheduler | self |
| summarizing-session-log | skill | - | 07-logging<br>file-change-summarizer | self |
| test-fix-workflow | skill | - | 03-red-testing<br>05-green-testing<br>failure-triage-specialist<br>green-test-failure-triage-coordinator | self |
| trace-analyzer-extension | skill | - | 04-implementing<br>performance-trace-specialist | self |
| trace-audit-reporting | skill | - | 05-green-testing<br>performance-trace-specialist | self |
| tracker-handoff | skill | - | 00-helping<br>01-planning<br>04-implementing<br>07-logging<br>slice-orchestration-scheduler | self |
| triaging-test-failures | skill | - | failure-triage-specialist | self |
| updating-agent-frontmatter | skill | - | agent-frontmatter-auditor | self |
| updating-js-docs | skill | - | 06-documenting | self |
| updating-skill-frontmatter | skill | - | skill-frontmatter-auditor | self |
| visualizer-workflow | skill | - | visualizer-scout | self |
| webgpu | skill | - | - | self |
| worker-inference-transport | skill | - | 04-implementing<br>worker-payload-scout | self |
