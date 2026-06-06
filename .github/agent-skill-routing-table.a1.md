## Proposed Agent and Skill Routing Table Revision (Gemma4 Local Focus)

**Goal:** Maximize local agent (Gemma4) usage for cost efficiency and reserve cloud agents only for essential, high-complexity tasks.
**Principle:** All standard data manipulation, local file system interaction, or general pattern matching is assigned to a local agent (Gemma4). Cloud agents are reserved for advanced reasoning/proprietary model needs.

| Task/Role                               | Gemma4 (Local) | Qwen3.6 (Local) | Cloud Agents (mini/large/GPT-5.4) | Justification                                                                      |
| :-------------------------------------- | :------------- | :-------------- | :-------------------------------- | :--------------------------------------------------------------------------------- |
| Orchestration (all coordinator roles)   | ✔️             | ✔️              | ⛔                                | Local agents can handle orchestration per MCP guidelines; cloud only for fallback. |
| Routing/Auditing (all auditor roles)    | ✔️             | ✔️              | ⛔                                | Routine checks and routing logic are suitable for local agents.                    |
| Information Gathering (scout, capturer) | ✔️             | ✔️              | ⛔                                | Local agents can efficiently gather and summarize information.                     |
| Routine Skill Evaluation                | ✔️             | ✔️              | ⛔                                | Local agents can perform standard evaluations and inventory audits.                |
| Subagent Delegation Patterns            | ✔️             | ✔️              | ⛔                                | Delegation logic is lightweight and fits local agent capabilities.                 |
| Complex Planning/Strategy               | ⛔             | ⛔              | ✔️                                | Requires advanced reasoning, large context, or high accuracy.                      |
| Acceptance Criteria Writing             | ⛔             | ⛔              | ✔️                                | High-value, critical thinking task best handled by cloud agents.                   |
| Academic/Legal/Compliance Auditing      | ⛔             | ⛔              | ✔️                                | Requires high accuracy, up-to-date knowledge, and advanced language skills.        |
| Multi-file/Codebase Analysis            | ⛔             | ⛔              | ✔️                                | Large context and advanced reasoning needed.                                       |
| Specialized/Heavy-Compute Tasks         | ⛔             | ⛔              | ✔️                                | Only cloud agents should be used for tasks exceeding local agent resource limits.  |

| Agent Name                            | Current Model     | Recommended Model | Justification                                                                                                       |
| :------------------------------------ | :---------------- | :---------------- | :------------------------------------------------------------------------------------------------------------------ |
| 00-helping                            | GPT-5.4-mini      | Gemma4 (local)    | All roles are orchestration, auditing, and info-gathering—well within local agent capability.                       |
| 01-planning                           | GPT-5.4-mini      | Gemma4 (local)    | Planning, coordination, and auditing can be handled locally; escalate to cloud only for complex, multi-phase plans. |
| 02-researching                        | GPT-5.4-mini      | Gemma4 (local)    | Research, scouting, and coordination are suitable for local agent.                                                  |
| 03-red-testing                        | GPT-5.4-mini      | Gemma4 (local)    | Test strategy and coverage can be handled locally; escalate only for advanced test design.                          |
| 04-implementing                       | GPT-5.4-mini      | Gemma4 (local)    | Implementation coordination, boundary mapping, and scouting are local-suitable.                                     |
| 05-green-testing                      | GPT-5.4-mini      | Gemma4 (local)    | Triage, coverage, and validation are routine and fit local agent.                                                   |
| 06-documenting                        | GPT-5.4-mini      | Gemma4 (local)    | Documentation, auditing, and scouting are local-suitable; escalate for academic/legal compliance.                   |
| 07-logging                            | Claude Haiku 4.6  | Gemma4 (local)    | Logging, summarization, and event capture are routine and fit local agent.                                          |
| academic-docs-auditor                 | Claude Haiku 4.6  | GPT-5.4 (copilot) | Academic citation audit requires high accuracy and up-to-date knowledge—keep on cloud.                              |
| acceptance-criteria-writer            | GPT-5.4-mini      | GPT-5.4 (copilot) | Acceptance criteria writing is high-value and benefits from advanced reasoning.                                     |
| agent-frontmatter-auditor             | Claude Haiku 4.6  | Gemma4 (local)    | Frontmatter auditing is routine and fits local agent.                                                               |
| boundary-mapper                       | Claude Haiku 4.6  | Gemma4 (local)    | Boundary mapping is suitable for local agent.                                                                       |
| browser-runtime-scout                 | Claude Haiku 4.6  | Gemma4 (local)    | Browser build scouting is routine.                                                                                  |
| checkpoint-scout                      | Claude Haiku 4.6  | Gemma4 (local)    | Checkpointing is routine.                                                                                           |
| cortex-embeddings-scout               | Claude Haiku 4.6  | Gemma4 (local)    | Embedding scouting is routine.                                                                                      |
| coverage-guard                        | Claude Haiku 4.6  | Gemma4 (local)    | Coverage guarding is routine.                                                                                       |
| coverage-scout                        | Claude Haiku 4.6  | Gemma4 (local)    | Coverage scouting is routine.                                                                                       |
| determinism-scout                     | Claude Haiku 4.6  | Gemma4 (local)    | Reproducibility contracts are routine.                                                                              |
| docs-example-writer                   | Claude Haiku 4.6  | Gemma4 (local)    | Example writing is routine.                                                                                         |
| docs-scout                            | Claude Haiku 4.6  | Gemma4 (local)    | Documentation scouting is routine.                                                                                  |
| evaluation-pool-scout                 | Claude Haiku 4.6  | Gemma4 (local)    | Evaluation pooling is routine.                                                                                      |
| failure-triage-specialist             | Claude Haiku 4.6  | Gemma4 (local)    | Failure triage is routine.                                                                                          |
| file-change-summarizer                | Claude Haiku 4.6  | Gemma4 (local)    | Summarization is routine.                                                                                           |
| flappy-architecture-polish            | GPT-5.4           | GPT-5.4 (copilot) | Architecture polish is complex—keep on cloud.                                                                       |
| green-test-failure-triage-coordinator | GPT-5.4-mini      | Gemma4 (local)    | Triage and validation are routine.                                                                                  |
| helping-agent-maintenance-coordinator | GPT-5.4           | Gemma4 (local)    | Maintenance coordination is routine.                                                                                |
| helping-gap-resolution-coordinator    | Claude Sonnet 4.6 | Gemma4 (local)    | Gap resolution is routine.                                                                                          |
| hybrid-interop-scout                  | GPT-5.4-mini      | Gemma4 (local)    | Interop scouting is routine.                                                                                        |
| implementation-pattern-coordinator    | GPT-5.4           | GPT-5.4 (copilot) | Implementation pattern coordination can be complex—keep on cloud.                                                   |
| implementation-pattern-scout          | GPT-5.4-mini      | Gemma4 (local)    | Pattern scouting is routine.                                                                                        |
| learning-event-capturer               | Claude Haiku 4.6  | Gemma4 (local)    | Event capture is routine.                                                                                           |
| license-attribution-auditor           | GPT-5.4-mini      | Gemma4 (local)    | License auditing is routine.                                                                                        |
| mcp-runtime-scout                     | GPT-5.4-mini      | Gemma4 (local)    | MCP runtime scouting is routine.                                                                                    |
| mcp-server-architect                  | GPT-5.4           | GPT-5.4 (copilot) | Server architecture is complex—keep on cloud.                                                                       |
| mcp-validation-auditor                | GPT-5.4-mini      | Gemma4 (local)    | Validation auditing is routine.                                                                                     |
| model-name-auditor                    | GPT-5.4-mini      | Gemma4 (local)    | Model name auditing is routine.                                                                                     |
| neatchat-scout                        | GPT-5.4-mini      | Gemma4 (local)    | Neatchat scouting is routine.                                                                                       |
| nge-benchmark-scout                   | GPT-5.4-mini      | Gemma4 (local)    | Benchmark scouting is routine.                                                                                      |
| nge-core-scout                        | GPT-5.4-mini      | Gemma4 (local)    | Core algorithm scouting is routine.                                                                                 |
| phase-handoff-designer                | GPT-5.4-mini      | Gemma4 (local)    | Handoff design is routine.                                                                                          |
| plan-registration-auditor             | GPT-5.4-mini      | Gemma4 (local)    | Plan registration auditing is routine.                                                                              |
| plan-scout                            | GPT-5.4-mini      | Gemma4 (local)    | Plan scouting is routine.                                                                                           |
| planning-context-coordinator          | GPT-5.4-mini      | Gemma4 (local)    | Context coordination is routine.                                                                                    |
| planning-risk-coordinator             | Claude Sonnet 4.6 | GPT-5.4 (copilot) | Risk coordination can be complex—keep on cloud.                                                                     |
| planning-test-strategy-coordinator    | GPT-5.4           | GPT-5.4 (copilot) | Test strategy is complex—keep on cloud.                                                                             |
| repo-cortex-scout                     | Claude Haiku 4.6  | Gemma4 (local)    | Repo cortex scouting is routine.                                                                                    |
| research-codebase-coordinator         | GPT-5.4-mini      | Gemma4 (local)    | Codebase research coordination is routine.                                                                          |
| skill-frontmatter-auditor             | GPT-5.4-mini      | Gemma4 (local)    | Skill frontmatter auditing is routine.                                                                              |
| skill-inventory-auditor               | GPT-5.4-mini      | Gemma4 (local)    | Skill inventory auditing is routine.                                                                                |
| skill-output-eval-grader              | GPT-5.4-mini      | Gemma4 (local)    | Output eval grading is routine.                                                                                     |
| skill-trigger-eval-designer           | GPT-5.4-mini      | Gemma4 (local)    | Trigger eval design is routine.                                                                                     |
| solid-split                           | GPT-5.4           | GPT-5.4 (copilot) | Solid split is complex—keep on cloud.                                                                               |
| unit-test-runner                      | GPT-5.4-mini      | Gemma4 (local)    | Unit test running is routine.                                                                                       |
| unit-test-writer                      | GPT-5.4           | GPT-5.4 (copilot) | Unit test writing is complex—keep on cloud.                                                                         |
| visualizer-scout                      | GPT-5.4-mini      | Gemma4 (local)    | Visualization scouting is routine.                                                                                  |
| vscode-ai-extensibility-scout         | GPT-5.4-mini      | Gemma4 (local)    | VSCode extensibility scouting is routine.                                                                           |
| worker-payload-scout                  | GPT-5.4-mini      | Gemma4 (local)    | Worker inference transport scouting is routine.                                                                     |
