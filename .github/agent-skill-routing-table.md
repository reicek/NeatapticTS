<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: cbef17dfc723a19d710e1f37dfcdc7c968a4482eb8f41ede8f6e5f5d88ad6cf8 -->
<!-- source-file-count: 108 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| 00-helping | 1 | Claude Sonnet 4.6 (copilot)<br>GPT-5.4 (copilot)<br>GPT-5.4-mini (copilot) | helping-gap-resolution-coordinator<br>helping-agent-maintenance-coordinator<br>skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>skill-trigger-eval-designer<br>skill-output-eval-grader<br>coverage-guard<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| 01-planning | 1 | Claude Sonnet 4.6 (copilot)<br>GPT-5.4 (copilot)<br>GPT-5.4-mini (copilot) | planning-context-coordinator<br>planning-risk-coordinator<br>planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>plan-scout<br>model-name-auditor<br>plan-registration-auditor<br>helping-gap-resolution-coordinator | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit |
| 02-researching | 1 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4 (copilot) | research-codebase-coordinator<br>plan-scout<br>docs-scout<br>boundary-mapper<br>skill-inventory-auditor<br>helping-gap-resolution-coordinator | subagent-delegation-patterns |
| 03-red-testing | 1 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | planning-test-strategy-coordinator<br>acceptance-criteria-writer<br>unit-test-writer<br>coverage-scout<br>determinism-scout<br>plan-scout<br>helping-gap-resolution-coordinator | red-test-contracts<br>test-fix-workflow<br>coverage-tranche |
| 04-implementing | 1 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | implementation-pattern-coordinator<br>boundary-mapper<br>docs-scout<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout<br>solid-split<br>flappy-architecture-polish<br>agent-frontmatter-auditor<br>phase-handoff-designer<br>mcp-server-architect<br>helping-gap-resolution-coordinator | - |
| 05-green-testing | 1 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | green-test-failure-triage-coordinator<br>coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>determinism-scout<br>plan-registration-auditor<br>mcp-validation-auditor<br>helping-gap-resolution-coordinator | green-validation-gates<br>coverage-guard<br>plan-sync-validation |
| 06-documenting | 1 | Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | docs-scout<br>academic-docs-auditor<br>docs-example-writer<br>plan-scout<br>license-attribution-auditor<br>vscode-ai-extensibility-scout<br>helping-gap-resolution-coordinator | educational-docs<br>docs-academic-citation-audit<br>license-attribution-audit |
| 07-logging | 1 | Claude Haiku 4.6 (copilot)<br>GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | plan-scout<br>plan-registration-auditor<br>learning-event-capturer<br>file-change-summarizer<br>helping-gap-resolution-coordinator | tracker-handoff<br>plan-sync-validation |
| academic-docs-auditor | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | docs-academic-citation-audit |
| acceptance-criteria-writer | 4 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | - | planning-acceptance-criteria |
| agent-frontmatter-auditor | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | agent-frontmatter-standards |
| boundary-mapper | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | solid-split |
| browser-runtime-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | browser-build |
| checkpoint-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | checkpointing-persistence |
| cortex-embeddings-scout | 3 | Claude Haiku 4.6 (copilot) | - | - |
| coverage-guard | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | coverage-guard |
| coverage-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | coverage-tranche |
| determinism-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | reproducibility-contracts |
| docs-example-writer | 4 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | - |
| docs-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | educational-docs |
| evaluation-pool-scout | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | multithread-evaluation |
| failure-triage-specialist | 3 | Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | - | triaging-test-failures |
| file-change-summarizer | 4 | Claude Haiku 4.6 (copilot)<br>GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | summarizing-session-log |
| flappy-architecture-polish | 2 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | plan-scout | flappy-architecture-polish |
| green-test-failure-triage-coordinator | 2 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | coverage-guard<br>coverage-scout<br>failure-triage-specialist<br>unit-test-runner<br>plan-registration-auditor<br>mcp-validation-auditor | green-validation-gates |
| helping-agent-maintenance-coordinator | 2 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>skill-inventory-auditor<br>model-name-auditor<br>learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit |
| helping-gap-resolution-coordinator | 2 | Claude Sonnet 4.6 (copilot)<br>GPT-5.4 (copilot)<br>GPT-5.4-mini (copilot) | skill-inventory-auditor<br>agent-frontmatter-auditor<br>skill-frontmatter-auditor<br>model-name-auditor<br>learning-event-capturer<br>file-change-summarizer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>subagent-delegation-patterns |
| hybrid-interop-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | hybrid-training-interop |
| implementation-pattern-coordinator | 2 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | implementation-pattern-scout<br>boundary-mapper<br>docs-scout<br>agent-frontmatter-auditor | subagent-delegation-patterns |
| implementation-pattern-scout | 3 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | - | - |
| learning-event-capturer | 4 | Claude Haiku 4.6 (copilot)<br>GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | capturing-learning-event |
| license-attribution-auditor | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | license-attribution-audit |
| mcp-runtime-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | mcp-local-server-workflow |
| mcp-server-architect | 3 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | - | mcp-local-server-workflow |
| mcp-validation-auditor | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | mcp-local-server-workflow |
| model-name-auditor | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | model-routing-and-budget |
| neatchat-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | neatchat-systems |
| nge-benchmark-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | nge-benchmark-workflow |
| nge-core-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | nge-core-algorithm |
| phase-handoff-designer | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | phase-handoff-workflow |
| plan-registration-auditor | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | plan-sync-validation |
| plan-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | plan-alignment |
| planning-context-coordinator | 2 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | plan-scout<br>docs-scout<br>boundary-mapper | plan-alignment |
| planning-risk-coordinator | 2 | Claude Sonnet 4.6 (copilot)<br>GPT-5.4 (copilot)<br>GPT-5.4-mini (copilot) | plan-scout<br>determinism-scout<br>license-attribution-auditor<br>model-name-auditor | model-routing-and-budget<br>license-attribution-audit |
| planning-test-strategy-coordinator | 2 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | coverage-scout<br>determinism-scout<br>acceptance-criteria-writer<br>unit-test-writer | planning-acceptance-criteria<br>red-test-contracts |
| repo-cortex-scout | 3 | Claude Haiku 4.6 (copilot) | - | repo-cortex-workflow |
| research-codebase-coordinator | 2 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>Claude Sonnet 4.6 (copilot) | plan-scout<br>docs-scout<br>boundary-mapper<br>browser-runtime-scout<br>worker-payload-scout<br>evaluation-pool-scout<br>checkpoint-scout<br>hybrid-interop-scout<br>determinism-scout<br>visualizer-scout<br>nge-core-scout<br>nge-benchmark-scout<br>neatchat-scout | subagent-delegation-patterns |
| skill-frontmatter-auditor | 3 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | - | skill-frontmatter-standards |
| skill-inventory-auditor | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | agent-inventory-audit |
| skill-output-eval-grader | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | skill-output-evals |
| skill-trigger-eval-designer | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | skill-description-evals |
| solid-split | 2 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | boundary-mapper<br>plan-scout<br>docs-scout | solid-split |
| unit-test-runner | 3 | GPT-5.4-mini (copilot)<br>Claude Haiku 4.6 (copilot)<br>GPT-5.4 (copilot) | - | running-unit-tests |
| unit-test-writer | 3 | GPT-5.4 (copilot)<br>Claude Sonnet 4.6 (copilot)<br>GPT-5.4-mini (copilot) | - | creating-unit-tests |
| visualizer-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | visualizer-workflow |
| vscode-ai-extensibility-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | - |
| worker-payload-scout | 3 | GPT-5.4-mini (copilot)<br>GPT-5.4 (copilot) | - | worker-inference-transport |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | 00-helping<br>01-planning<br>agent-frontmatter-auditor<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator | self |
| agent-inventory-audit | skill | - | 00-helping<br>helping-agent-maintenance-coordinator<br>helping-gap-resolution-coordinator<br>skill-inventory-auditor | self |
| agent-script-tooling | skill | - | - | self |
| architecture-builder | skill | - | - | self |
| auditing-js-docs | skill | - | - | self |
| browser-build | skill | - | browser-runtime-scout | self |
| capturing-learning-event | skill | - | learning-event-capturer | self |
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
