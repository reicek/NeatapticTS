<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: c2c6ff50a361d9f2ff4a00b2489a4dde0c9f299563410c7060ace6062515003c -->
<!-- source-file-count: 99 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Complexity | Model | Agents | Skills |
| --- | --- | --- | --- | --- | --- |
| 00-helping | 1 | moderate | kimi-k2.7-code:cloud | agent-maintenance-coordinator<br>coverage-analyst<br>learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>customize-cloud-agent<br>subagent-delegation-patterns<br>capturing-learning-event<br>routing-optimization-policy<br>phase-handoff-workflow<br>tracker-handoff<br>execute<br>skill-frontmatter-standards<br>mcp-local-server-workflow<br>skill-description-evals<br>skill-output-evals |
| 01-planning | 1 | moderate | kimi-k2.7-code:cloud | plan-scout<br>agent-maintenance-coordinator | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit<br>planning-acceptance-criteria<br>plan-sync-validation<br>spec-checklist<br>research-methodology<br>execute<br>red-test-contracts |
| 02-researching | 1 | moderate | kimi-k2.7-code:cloud | research-codebase-coordinator<br>plan-scout<br>docs-scout<br>boundary-mapper<br>agent-maintenance-coordinator<br>license-reviewer<br>dependency-audit-reviewer<br>benchmark-gate-reviewer | subagent-delegation-patterns<br>research-methodology<br>repo-cortex-workflow<br>execute<br>repo-cortex-embeddings |
| 03-red-testing | 1 | moderate | kimi-k2.7-code:cloud | unit-test-writer<br>plan-scout<br>performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist<br>coverage-analyst<br>agent-maintenance-coordinator<br>slice-validator<br>property-based-test-writer<br>boundary-mapper | red-test-contracts<br>nge-core-algorithm<br>reproducibility-contracts<br>creating-unit-tests<br>test-fix-workflow<br>coverage-tranche<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools<br>planning-acceptance-criteria<br>property-based-testing |
| 04-implementing | 1 | moderate | kimi-k2.7-code:cloud | implementation-pattern-scout<br>implementation-executor<br>boundary-mapper<br>docs-scout<br>solid-split<br>browser-harness-specialist<br>agent-maintenance-coordinator<br>plan-scout<br>performance-trace-specialist<br>security-reviewer<br>performance-reviewer<br>api-contract-reviewer<br>determinism-reviewer<br>dependency-audit-reviewer | implementation-standards<br>nge-core-algorithm<br>reproducibility-contracts<br>tracker-handoff<br>architecture-builder<br>onnx-work<br>performance-optimization<br>trace-analyzer-extension<br>worker-inference-transport<br>research-methodology<br>execute<br>browser-testing-harness<br>neatchat-systems<br>mcp-local-server-workflow<br>webgpu<br>multithread-evaluation<br>checkpointing-persistence<br>hybrid-training-interop<br>visualizer-workflow<br>browser-build<br>flappy-architecture-polish<br>security-review<br>dependency-audit |
| 05-green-testing | 1 | moderate | kimi-k2.7-code:cloud | performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist<br>coverage-analyst<br>agent-maintenance-coordinator<br>plan-scout<br>boundary-mapper<br>slice-validator<br>security-reviewer<br>performance-reviewer<br>determinism-reviewer<br>benchmark-gate-reviewer | green-validation-gates<br>coverage-guard<br>test-fix-workflow<br>plan-sync-validation<br>spec-checklist<br>trace-audit-reporting<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools<br>nge-benchmark-workflow<br>reproducibility-contracts<br>running-unit-tests<br>triaging-test-failures<br>mcp-local-server-workflow<br>security-review<br>benchmark-gate |
| 06-documenting | 1 | moderate | kimi-k2.7-code:cloud | docs-scout<br>plan-scout<br>browser-harness-specialist<br>agent-maintenance-coordinator<br>api-contract-reviewer<br>license-reviewer<br>browser-ui-specialist<br>browser-memory-specialist | educational-docs<br>nge-core-algorithm<br>docs-academic-citation-audit<br>license-attribution-audit<br>auditing-js-docs<br>updating-js-docs<br>research-methodology<br>execute<br>browser-testing-harness<br>dependency-audit |
| 07-logging | 1 | moderate | kimi-k2.7-code:cloud | plan-scout<br>learning-event-capturer<br>agent-maintenance-coordinator | tracker-handoff<br>summarizing-session-log<br>plan-sync-validation<br>capturing-learning-event<br>research-methodology<br>execute<br>phase-handoff-workflow |
| agent-maintenance-coordinator | 2 | moderate | kimi-k2.7-code:cloud | learning-event-capturer | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>creating-specialist-agent<br>subagent-delegation-patterns<br>execute<br>agent-json-body-to-md<br>agent-script-tooling<br>splitting-monolithic-agent |
| api-contract-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | implementation-standards |
| benchmark-gate-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | benchmark-gate<br>performance-optimization |
| boundary-mapper | 3 | moderate | kimi-k2.7-code:cloud | - | solid-split<br>implementation-standards |
| browser-harness-specialist | 3 | moderate | kimi-k2.7-code:cloud | - | chrome-devtools-mcp<br>research-methodology |
| browser-memory-specialist | 3 | moderate | kimi-k2.7-code:cloud | - | chrome-devtools-mcp |
| browser-ui-specialist | 3 | moderate | kimi-k2.7-code:cloud | - | chrome-devtools-mcp |
| coverage-analyst | 3 | moderate | kimi-k2.7-code:cloud | - | coverage-guard<br>coverage-tranche |
| dependency-audit-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | dependency-audit |
| determinism-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | reproducibility-contracts |
| docs-scout | 3 | moderate | kimi-k2.7-code:cloud | - | educational-docs<br>auditing-js-docs |
| implementation-executor | 2 | moderate | kimi-k2.7-code:cloud | boundary-mapper<br>docs-scout | implementation-standards<br>coverage-guard<br>execute |
| implementation-pattern-scout | 3 | moderate | kimi-k2.7-code:cloud | - | implementation-standards |
| learning-event-capturer | 4 | moderate | kimi-k2.7-code:cloud | - | capturing-learning-event |
| license-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | license-attribution-audit |
| performance-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | performance-optimization<br>implementation-standards |
| performance-trace-specialist | 3 | moderate | kimi-k2.7-code:cloud | - | chrome-devtools-mcp<br>trace-audit-reporting<br>trace-analyzer-extension |
| plan-scout | 3 | moderate | kimi-k2.7-code:cloud | - | plan-alignment |
| property-based-test-writer | 3 | moderate | kimi-k2.7-code:cloud | - | property-based-testing<br>red-test-contracts<br>creating-unit-tests |
| research-codebase-coordinator | 2 | moderate | kimi-k2.7-code:cloud | plan-scout<br>docs-scout<br>boundary-mapper<br>implementation-pattern-scout | subagent-delegation-patterns<br>repo-cortex-workflow<br>research-methodology<br>execute |
| security-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | security-review<br>implementation-standards |
| slice-validator | 3 | moderate | kimi-k2.7-code:cloud | - | phase-handoff-workflow<br>plan-sync-validation |
| solid-split | 2 | moderate | kimi-k2.7-code:cloud | boundary-mapper<br>plan-scout<br>docs-scout | solid-split<br>implementation-standards<br>execute |
| unit-test-writer | 3 | moderate | kimi-k2.7-code:cloud | - | creating-unit-tests<br>red-test-contracts |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | 00-helping<br>01-planning<br>agent-maintenance-coordinator | self |
| agent-inventory-audit | skill | - | 00-helping<br>agent-maintenance-coordinator | self |
| agent-json-body-to-md | skill | - | agent-maintenance-coordinator | self |
| agent-script-tooling | skill | - | agent-maintenance-coordinator | self |
| architecture-builder | skill | - | 04-implementing | self |
| auditing-js-docs | skill | - | 06-documenting<br>docs-scout | self |
| benchmark-gate | skill | - | 05-green-testing<br>benchmark-gate-reviewer | self |
| browser-build | skill | - | 04-implementing | self |
| browser-testing-harness | skill | - | 03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting | self |
| capturing-learning-event | skill | - | 00-helping<br>07-logging<br>learning-event-capturer | self |
| checkpointing-persistence | skill | - | 04-implementing | self |
| chrome-devtools-mcp | skill | - | 03-red-testing<br>05-green-testing<br>browser-harness-specialist<br>browser-memory-specialist<br>browser-ui-specialist<br>performance-trace-specialist | self |
| coverage-guard | skill | - | 05-green-testing<br>coverage-analyst<br>implementation-executor | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-analyst | self |
| creating-specialist-agent | skill | - | agent-maintenance-coordinator | self |
| creating-unit-tests | skill | - | 03-red-testing<br>property-based-test-writer<br>unit-test-writer | self |
| customize-cloud-agent | skill | - | 00-helping | self |
| dependency-audit | skill | - | 04-implementing<br>06-documenting<br>dependency-audit-reviewer | self |
| devtools | skill | - | 03-red-testing<br>05-green-testing | self |
| docs-academic-citation-audit | skill | - | 06-documenting | self |
| educational-docs | skill | - | 06-documenting<br>docs-scout | self |
| execute | skill | - | 00-helping<br>01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>agent-maintenance-coordinator<br>implementation-executor<br>research-codebase-coordinator<br>solid-split | self |
| flappy-architecture-polish | skill | - | 04-implementing | self |
| green-validation-gates | skill | - | 05-green-testing | self |
| hybrid-training-interop | skill | - | 04-implementing | self |
| implementation-standards | skill | - | 04-implementing<br>api-contract-reviewer<br>boundary-mapper<br>implementation-executor<br>implementation-pattern-scout<br>performance-reviewer<br>security-reviewer<br>solid-split | self |
| license-attribution-audit | skill | - | 01-planning<br>06-documenting<br>license-reviewer | self |
| mcp-local-server-workflow | skill | - | 00-helping<br>04-implementing<br>05-green-testing | self |
| model-routing-and-budget | skill | - | 00-helping<br>01-planning<br>agent-maintenance-coordinator | self |
| multithread-evaluation | skill | - | 04-implementing | self |
| neatchat-systems | skill | - | 04-implementing | self |
| nge-benchmark-workflow | skill | - | 05-green-testing | self |
| nge-core-algorithm | skill | - | 03-red-testing<br>04-implementing<br>06-documenting | self |
| onnx-work | skill | - | 04-implementing | self |
| performance-optimization | skill | - | 04-implementing<br>benchmark-gate-reviewer<br>performance-reviewer | self |
| phase-handoff-workflow | skill | - | 00-helping<br>01-planning<br>07-logging<br>slice-validator | self |
| plan-alignment | skill | - | 01-planning<br>plan-scout | self |
| plan-sync-validation | skill | - | 01-planning<br>05-green-testing<br>07-logging<br>slice-validator | self |
| planning-acceptance-criteria | skill | - | 01-planning<br>03-red-testing | self |
| property-based-testing | skill | - | 03-red-testing<br>property-based-test-writer | self |
| red-test-contracts | skill | - | 01-planning<br>03-red-testing<br>property-based-test-writer<br>unit-test-writer | self |
| repo-cortex-embeddings | skill | - | 02-researching | self |
| repo-cortex-workflow | skill | - | 02-researching<br>research-codebase-coordinator | self |
| reproducibility-contracts | skill | - | 03-red-testing<br>04-implementing<br>05-green-testing<br>determinism-reviewer | self |
| research-methodology | skill | - | 01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>browser-harness-specialist<br>research-codebase-coordinator | self |
| routing-optimization-policy | skill | - | 00-helping | self |
| running-unit-tests | skill | - | 05-green-testing | self |
| security-review | skill | - | 04-implementing<br>05-green-testing<br>security-reviewer | self |
| skill-description-evals | skill | - | 00-helping | self |
| skill-frontmatter-standards | skill | - | 00-helping | self |
| skill-output-evals | skill | - | 00-helping | self |
| solid-split | skill | - | boundary-mapper<br>solid-split | self |
| spec-checklist | skill | - | 01-planning<br>05-green-testing | self |
| splitting-monolithic-agent | skill | - | agent-maintenance-coordinator | self |
| subagent-delegation-patterns | skill | - | 00-helping<br>02-researching<br>agent-maintenance-coordinator<br>research-codebase-coordinator | self |
| summarizing-session-log | skill | - | 07-logging | self |
| test-fix-workflow | skill | - | 03-red-testing<br>05-green-testing | self |
| trace-analyzer-extension | skill | - | 04-implementing<br>performance-trace-specialist | self |
| trace-audit-reporting | skill | - | 05-green-testing<br>performance-trace-specialist | self |
| tracker-handoff | skill | - | 00-helping<br>01-planning<br>04-implementing<br>07-logging | self |
| triaging-test-failures | skill | - | 05-green-testing | self |
| updating-agent-frontmatter | skill | - | - | self |
| updating-js-docs | skill | - | 06-documenting | self |
| updating-skill-frontmatter | skill | - | - | self |
| visualizer-workflow | skill | - | 04-implementing | self |
| webgpu | skill | - | 04-implementing | self |
| worker-inference-transport | skill | - | 04-implementing | self |
