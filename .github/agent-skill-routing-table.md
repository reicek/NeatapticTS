<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->
<!-- source-hash: 96d7b9ffb44d44b74334a0b6ee089f70840d1b858f5aac3c7a889a4e58ef7236 -->
<!-- source-file-count: 105 -->
# Canonical Agent and Skill Routing Table

> Generated file. Do not edit manually.
> Refresh with `npm run agents:routing-table`.
> Validate freshness with `npm run agents:routing-table:gate`.

## Agents

| Name | Tier | Complexity | Model | Agents | Skills |
| --- | --- | --- | --- | --- | --- |
| 00-helping | 1 | moderate | glm-5.2:cloud | agent-maintenance-coordinator<br>coverage-analyst<br>learning-event-capturer<br>frontmatter-auditor<br>repo-cortex-scout | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>customize-cloud-agent<br>subagent-delegation-patterns<br>capturing-learning-event<br>routing-optimization-policy<br>phase-handoff-workflow<br>tracker-handoff<br>execute<br>skill-frontmatter-standards<br>mcp-local-server-workflow<br>skill-description-evals<br>skill-output-evals |
| 01-planning | 1 | moderate | glm-5.2:cloud | plan-scout<br>agent-maintenance-coordinator | plan-alignment<br>tracker-handoff<br>phase-handoff-workflow<br>agent-frontmatter-standards<br>model-routing-and-budget<br>license-attribution-audit<br>planning-acceptance-criteria<br>plan-sync-validation<br>spec-checklist<br>research-methodology<br>execute<br>red-test-contracts<br>solid-split |
| 02-researching | 1 | moderate | kimi-k2.7-code:cloud | plan-scout<br>docs-scout<br>boundary-mapper<br>agent-maintenance-coordinator<br>license-reviewer<br>dependency-audit-reviewer<br>benchmark-gate-reviewer<br>repo-cortex-scout<br>implementation-pattern-scout | subagent-delegation-patterns<br>research-methodology<br>repo-cortex-workflow<br>execute<br>repo-cortex-embeddings<br>solid-split<br>neatchat-systems<br>architecture-builder<br>trace-analyzer-extension<br>nge-benchmark-workflow |
| 03-red-testing | 1 | moderate | glm-5.2:cloud | unit-test-writer<br>plan-scout<br>performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist<br>coverage-analyst<br>agent-maintenance-coordinator<br>slice-validator<br>property-based-test-writer<br>boundary-mapper | red-test-contracts<br>nge-core-algorithm<br>reproducibility-contracts<br>creating-unit-tests<br>test-fix-workflow<br>coverage-tranche<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools<br>planning-acceptance-criteria<br>property-based-testing |
| 04-implementing | 1 | moderate | glm-5.2:cloud | implementation-pattern-scout<br>implementation-executor<br>boundary-mapper<br>docs-scout<br>browser-harness-specialist<br>agent-maintenance-coordinator<br>plan-scout<br>performance-trace-specialist<br>review-coordinator | implementation-standards<br>solid-split<br>reproducibility-contracts<br>tracker-handoff<br>performance-optimization<br>research-methodology<br>execute<br>browser-testing-harness<br>mcp-local-server-workflow<br>checkpointing-persistence<br>hybrid-training-interop<br>flappy-architecture-polish<br>security-review<br>dependency-audit |
| 05-green-testing | 1 | moderate | kimi-k2.7-code:cloud | performance-trace-specialist<br>browser-ui-specialist<br>browser-memory-specialist<br>browser-harness-specialist<br>coverage-analyst<br>agent-maintenance-coordinator<br>plan-scout<br>boundary-mapper<br>slice-validator<br>benchmark-gate-reviewer<br>review-coordinator | green-validation-gates<br>coverage-guard<br>test-fix-workflow<br>plan-sync-validation<br>spec-checklist<br>trace-audit-reporting<br>research-methodology<br>execute<br>chrome-devtools-mcp<br>browser-testing-harness<br>devtools<br>nge-benchmark-workflow<br>reproducibility-contracts<br>running-unit-tests<br>triaging-test-failures<br>mcp-local-server-workflow<br>security-review<br>benchmark-gate<br>worker-inference-transport<br>multithread-evaluation<br>browser-build |
| 06-documenting | 1 | moderate | kimi-k2.7-code:cloud | docs-scout<br>docs-writer<br>plan-scout<br>browser-harness-specialist<br>agent-maintenance-coordinator<br>api-contract-reviewer<br>license-reviewer<br>browser-ui-specialist<br>browser-memory-specialist | educational-docs<br>nge-core-algorithm<br>docs-academic-citation-audit<br>license-attribution-audit<br>auditing-js-docs<br>updating-js-docs<br>research-methodology<br>execute<br>browser-testing-harness<br>dependency-audit<br>neatchat-systems<br>visualizer-workflow |
| 07-logging | 1 | moderate | kimi-k2.7-code:cloud | plan-scout<br>learning-event-capturer<br>agent-maintenance-coordinator<br>session-summarizer | tracker-handoff<br>summarizing-session-log<br>plan-sync-validation<br>capturing-learning-event<br>research-methodology<br>execute<br>phase-handoff-workflow |
| agent-maintenance-coordinator | 2 | moderate | kimi-k2.7-code:cloud | learning-event-capturer<br>frontmatter-auditor | agent-frontmatter-standards<br>model-routing-and-budget<br>agent-inventory-audit<br>creating-specialist-agent<br>subagent-delegation-patterns<br>execute<br>agent-json-body-to-md<br>agent-script-tooling<br>splitting-monolithic-agent |
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
| docs-writer | 3 | moderate | kimi-k2.7-code:cloud | - | educational-docs<br>updating-js-docs<br>auditing-js-docs |
| evolution-correctness-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | nge-core-algorithm<br>reproducibility-contracts<br>implementation-standards |
| frontmatter-auditor | 3 | moderate | kimi-k2.7-code:cloud | - | agent-frontmatter-standards<br>skill-frontmatter-standards<br>updating-agent-frontmatter<br>updating-skill-frontmatter<br>model-routing-and-budget |
| implementation-executor | 2 | moderate | glm-5.2:cloud | boundary-mapper<br>docs-scout | implementation-standards<br>coverage-guard<br>execute |
| implementation-pattern-scout | 3 | moderate | kimi-k2.7-code:cloud | - | implementation-standards |
| learning-event-capturer | 4 | moderate | kimi-k2.7-code:cloud | - | capturing-learning-event |
| license-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | license-attribution-audit |
| onnx-parity-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | onnx-work<br>implementation-standards<br>reproducibility-contracts |
| performance-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | performance-optimization<br>implementation-standards |
| performance-trace-specialist | 3 | moderate | kimi-k2.7-code:cloud | - | chrome-devtools-mcp<br>trace-audit-reporting<br>trace-analyzer-extension |
| plan-scout | 3 | moderate | kimi-k2.7-code:cloud | - | plan-alignment |
| property-based-test-writer | 3 | moderate | kimi-k2.7-code:cloud | - | property-based-testing<br>red-test-contracts<br>creating-unit-tests |
| repo-cortex-scout | 3 | moderate | kimi-k2.7-code:cloud | - | repo-cortex-workflow<br>repo-cortex-embeddings<br>research-methodology |
| review-coordinator | 2 | moderate | kimi-k2.7-code:cloud | security-reviewer<br>performance-reviewer<br>determinism-reviewer<br>api-contract-reviewer<br>dependency-audit-reviewer<br>benchmark-gate-reviewer<br>evolution-correctness-reviewer<br>onnx-parity-reviewer<br>webgpu-parity-reviewer | implementation-standards<br>red-test-contracts<br>security-review<br>dependency-audit |
| security-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | security-review<br>implementation-standards |
| session-summarizer | 3 | moderate | kimi-k2.7-code:cloud | - | summarizing-session-log<br>tracker-handoff<br>research-methodology |
| slice-validator | 3 | moderate | kimi-k2.7-code:cloud | - | phase-handoff-workflow<br>plan-sync-validation |
| unit-test-writer | 3 | moderate | kimi-k2.7-code:cloud | - | creating-unit-tests<br>red-test-contracts |
| webgpu-parity-reviewer | 3 | moderate | kimi-k2.7-code:cloud | - | webgpu<br>implementation-standards<br>performance-optimization |

## Skills

| Name | Tier | Model | Agents | Skills |
| --- | --- | --- | --- | --- |
| agent-frontmatter-standards | skill | - | 00-helping<br>01-planning<br>agent-maintenance-coordinator<br>frontmatter-auditor | self |
| agent-inventory-audit | skill | - | 00-helping<br>agent-maintenance-coordinator | self |
| agent-json-body-to-md | skill | - | agent-maintenance-coordinator | self |
| agent-script-tooling | skill | - | agent-maintenance-coordinator | self |
| architecture-builder | skill | - | 02-researching | self |
| auditing-js-docs | skill | - | 06-documenting<br>docs-scout<br>docs-writer | self |
| benchmark-gate | skill | - | 05-green-testing<br>benchmark-gate-reviewer | self |
| browser-build | skill | - | 05-green-testing | self |
| browser-testing-harness | skill | - | 03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting | self |
| capturing-learning-event | skill | - | 00-helping<br>07-logging<br>learning-event-capturer | self |
| checkpointing-persistence | skill | - | 04-implementing | self |
| chrome-devtools-mcp | skill | - | 03-red-testing<br>05-green-testing<br>browser-harness-specialist<br>browser-memory-specialist<br>browser-ui-specialist<br>performance-trace-specialist | self |
| coverage-guard | skill | - | 05-green-testing<br>coverage-analyst<br>implementation-executor | self |
| coverage-tranche | skill | - | 03-red-testing<br>coverage-analyst | self |
| creating-specialist-agent | skill | - | agent-maintenance-coordinator | self |
| creating-unit-tests | skill | - | 03-red-testing<br>property-based-test-writer<br>unit-test-writer | self |
| customize-cloud-agent | skill | - | 00-helping | self |
| dependency-audit | skill | - | 04-implementing<br>06-documenting<br>dependency-audit-reviewer<br>review-coordinator | self |
| devtools | skill | - | 03-red-testing<br>05-green-testing | self |
| docs-academic-citation-audit | skill | - | 06-documenting | self |
| educational-docs | skill | - | 06-documenting<br>docs-scout<br>docs-writer | self |
| execute | skill | - | 00-helping<br>01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>agent-maintenance-coordinator<br>implementation-executor | self |
| flappy-architecture-polish | skill | - | 04-implementing | self |
| green-validation-gates | skill | - | 05-green-testing | self |
| hybrid-training-interop | skill | - | 04-implementing | self |
| implementation-standards | skill | - | 04-implementing<br>api-contract-reviewer<br>boundary-mapper<br>evolution-correctness-reviewer<br>implementation-executor<br>implementation-pattern-scout<br>onnx-parity-reviewer<br>performance-reviewer<br>review-coordinator<br>security-reviewer<br>webgpu-parity-reviewer | self |
| license-attribution-audit | skill | - | 01-planning<br>06-documenting<br>license-reviewer | self |
| mcp-local-server-workflow | skill | - | 00-helping<br>04-implementing<br>05-green-testing | self |
| model-routing-and-budget | skill | - | 00-helping<br>01-planning<br>agent-maintenance-coordinator<br>frontmatter-auditor | self |
| multithread-evaluation | skill | - | 05-green-testing | self |
| neatchat-systems | skill | - | 02-researching<br>06-documenting | self |
| nge-benchmark-workflow | skill | - | 02-researching<br>05-green-testing | self |
| nge-core-algorithm | skill | - | 03-red-testing<br>06-documenting<br>evolution-correctness-reviewer | self |
| onnx-work | skill | - | onnx-parity-reviewer | self |
| performance-optimization | skill | - | 04-implementing<br>benchmark-gate-reviewer<br>performance-reviewer<br>webgpu-parity-reviewer | self |
| phase-handoff-workflow | skill | - | 00-helping<br>01-planning<br>07-logging<br>slice-validator | self |
| plan-alignment | skill | - | 01-planning<br>plan-scout | self |
| plan-sync-validation | skill | - | 01-planning<br>05-green-testing<br>07-logging<br>slice-validator | self |
| planning-acceptance-criteria | skill | - | 01-planning<br>03-red-testing | self |
| property-based-testing | skill | - | 03-red-testing<br>property-based-test-writer | self |
| red-test-contracts | skill | - | 01-planning<br>03-red-testing<br>property-based-test-writer<br>review-coordinator<br>unit-test-writer | self |
| repo-cortex-embeddings | skill | - | 02-researching<br>repo-cortex-scout | self |
| repo-cortex-workflow | skill | - | 02-researching<br>repo-cortex-scout | self |
| reproducibility-contracts | skill | - | 03-red-testing<br>04-implementing<br>05-green-testing<br>determinism-reviewer<br>evolution-correctness-reviewer<br>onnx-parity-reviewer | self |
| research-methodology | skill | - | 01-planning<br>02-researching<br>03-red-testing<br>04-implementing<br>05-green-testing<br>06-documenting<br>07-logging<br>browser-harness-specialist<br>repo-cortex-scout<br>session-summarizer | self |
| routing-optimization-policy | skill | - | 00-helping | self |
| running-unit-tests | skill | - | 05-green-testing | self |
| security-review | skill | - | 04-implementing<br>05-green-testing<br>review-coordinator<br>security-reviewer | self |
| skill-description-evals | skill | - | 00-helping | self |
| skill-frontmatter-standards | skill | - | 00-helping<br>frontmatter-auditor | self |
| skill-output-evals | skill | - | 00-helping | self |
| solid-split | skill | - | 01-planning<br>02-researching<br>04-implementing<br>boundary-mapper | self |
| spec-checklist | skill | - | 01-planning<br>05-green-testing | self |
| splitting-monolithic-agent | skill | - | agent-maintenance-coordinator | self |
| subagent-delegation-patterns | skill | - | 00-helping<br>02-researching<br>agent-maintenance-coordinator | self |
| summarizing-session-log | skill | - | 07-logging<br>session-summarizer | self |
| test-fix-workflow | skill | - | 03-red-testing<br>05-green-testing | self |
| trace-analyzer-extension | skill | - | 02-researching<br>performance-trace-specialist | self |
| trace-audit-reporting | skill | - | 05-green-testing<br>performance-trace-specialist | self |
| tracker-handoff | skill | - | 00-helping<br>01-planning<br>04-implementing<br>07-logging<br>session-summarizer | self |
| triaging-test-failures | skill | - | 05-green-testing | self |
| updating-agent-frontmatter | skill | - | frontmatter-auditor | self |
| updating-js-docs | skill | - | 06-documenting<br>docs-writer | self |
| updating-skill-frontmatter | skill | - | frontmatter-auditor | self |
| visualizer-workflow | skill | - | 06-documenting | self |
| webgpu | skill | - | webgpu-parity-reviewer | self |
| worker-inference-transport | skill | - | 05-green-testing | self |
