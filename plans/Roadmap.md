# Roadmap (Order of Execution)

This roadmap orders every plan in `/plans` into a dependency-aware sequence optimized for **library growth**:

- **Correctness first** (trust + reproducibility)
- **Ergonomics next** (architecture building, deterministic execution)
- **Adoption next** (browser distribution + examples)
- **Scale + deployment** (standalone inference, workers, checkpointing)
- **Interop breadth** (ONNX)
- **Advanced research** last (NEAT Genesis EvoDevo / NGE)

Where it helps, this roadmap uses **lanes** (things that can proceed in parallel) and **gates** (things that should be true before moving on).

Active plans stay in `plans/`; terminally closed reopen baselines and their logs live in `plans/completed/`.

## Recommended primary agent + skill combo by active roadmap phase

- Meta-workflow reopen — `01-planning` + `tracker-handoff` when reopening [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md)
- Phase 3 — `Browser Runtime Scout` + `browser-build`
- Phase 4 — `Plan Scout` + `plan-alignment`
- Phase 5 — `Plan Scout` + `performance-optimization`
- Phase 6 — `Plan Scout` + `onnx-work`
- Phase 7 — `Plan Scout` + `plan-alignment`

## Standalone RAG/index Consolidation Lane [WIP]

**Outcome:** consolidate all Repo Cortex RAG/index scripts and generated
artifacts into a single top-level `rag-index/` directory, add one idempotent
incremental `rag-index/update-rag.mjs` entry point, and update all downstream
consumers so the live `cortex` MCP server and validation gates stay green.

- RAG/index infrastructure reorganization
- Plan: [completed/rag-update.plans.md](completed/rag-update.plans.md) [DONE]
- Current internal state: Phase 1 archived under `plans/completed/`. Target
  layout was flat rename preserving internal subfolders; generated artifacts
  under `rag-index/data/`, `rag-index/models/`, `rag-index/freshness-proofs/`,
  `rag-index/snapshots/`; decision records recorded for snapshot location and
  entity-graph incrementality.

**Coordination rule:** this lane owns `rag-index/` relocation,
`rag-index/` creation, `package.json` script repointing, `.gitignore` updates,
MCP wiring in `.vscode/mcp.json` / `.mcp.json`, and path-reference updates in
`scripts/mcp-semantic/tools/cortex-db.mjs`,
`scripts/agent-customization/gates/cortex-index.gate.mjs`,
`.github/skills/repo-cortex-workflow/SKILL.md`, and
`.github/agents/repo-cortex-scout.agent.md`. It does not change `src/`, core
NEAT algorithms, or cloud/Turso deployment topology. Keep
`data/eval-baselines/` in place; it is persistent evaluation data, not a
generated runtime artifact.

## Standalone CI Failure Hardening Lane [DONE]

**Outcome:** make the `npm test` matrix pass in the GitHub Actions `ubuntu-latest`
runner by removing local-only assumptions and closing reported coverage gaps,
including accurate coverage measurement for `scripts/agent-customization/` `.mjs`
gate scripts via a native-ESM Jest project.

- CI failure hardening
- Plan: [completed/CI_Failure_Hardening.plans.md](completed/CI_Failure_Hardening.plans.md) (`plans/completed/CI_Failure_Hardening.plans.md`) [DONE]
- Current internal state: archived after the native-ESM coverage project and
  `code-coverage` gate baseline were committed.
- Racing Curriculum v2 lane is [DONE]; archived to [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md).

**Coordination rule:** this lane is confined to test files and fixtures under
`scripts/agent-customization/`, `scripts/mcp-semantic/`, `testing/fixtures/`,
plus the three barrel re-export files in `src/` and their new test files.
It must not change core NEAT algorithms or the public API surface beyond
adding required tests/fixtures.

## Racing Curriculum v2 Lane [DONE]

**Outcome:** continue Phase 8 of the racing curriculum now that the upstream
NGE Core Algorithm Workstream and NGE Core Growth Engine Wiring are [DONE].
First v2 slice extended the Tier 4+ observation vector from 95 to 103 channels
and is green-gated. Step 08 fixed the Tier 4 start-line stall
(`resolveControllerInputCountForObservationTier` 95→103 input match) and the
user confirmed cars now move. Current focus is research-before-implementation
on six critical symptoms: NGE networks stalling near ~120 nodes instead of
4k/8k/16k targets; missing red-team pits at Tier 3+; agents not visibly
stopping at pits; tire wear ~3× too fast; Tier 5 failing to activate / red
team stalling from tire failure; and dense NGE visualizer performance collapse
at 4k-16k nodes. Step 09-12 are research-only and each produces a
`docs/research/` report. Step 13 synthesizes findings into red-green
implementation packets. Preserve tier-promotion / carry-reset semantics from
`examples/racing_curriculum/reference.plans.md`.

- Racing Curriculum v2
- Plan: [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE]
- Current internal state: archived after all phases [DONE] and the network-growth blocker fix (N109/C420 -> N523/C1524). Remaining driving-improvement work folded into the Racing Perception Redesign lane.

**Coordination rule:** this lane owns `examples/racing_curriculum/` and any
racing-specific plumbing in `src/` that is required to close the v2 gaps. It
must preserve the Tier 1-6 ladder, promotion rules, carry/reset semantics, and
acceptance criteria defined in `examples/racing_curriculum/reference.plans.md`.
It does not own general NGE core algorithm work; those changes belong in the
completed upstream workstreams.

## NGE Racing Curriculum Oscillation & Sub-Tier Fix Lane [DONE]

**Outcome:** add NGE library-side juvenile sub-tier exhaustion thresholds that scale with network size, and add racing-side steering/score oscillation detection and penalties in the runtime adaptation engine. Library surface changes in `src/neat/nge-juvenile/`; racing surface changes in `examples/racing_curriculum/controller/runtime.adaptation.ts`. Core oscillation fix is green-gated and complete; remaining bundle rebuild, browser smoke validation, and documentation are folded into the Racing Perception Redesign lane.

- NGE Racing Curriculum Oscillation & Sub-Tier Fix
- Plan: [completed/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md](completed/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md) [DONE]
- Current internal state: Phase 4 [DONE] — core oscillation fix verified (82/82 tests pass, NGE specialist APPROVED, all gates green). Phases 5–7 (bundle rebuild, documentation, session logging) [CANCELLED] and folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md).
- Upstream dependency: [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE]

**Coordination rule:** this lane owns the `src/neat/nge-juvenile/` juvenile sub-tier threshold surface and the `examples/racing_curriculum/controller/runtime.adaptation.ts` oscillation-penalty surface. It must not change general NGE core lifecycle or reproduction. Preserve tier-promotion / carry-reset semantics from `examples/racing_curriculum/reference.plans.md`.

## Racing Perception Redesign Lane [PLANNED]

**Outcome:** redesign the racing curriculum observation/perception layer so opponents can be opponent-aware. Phase 1 fixes the speed persistence bug by storing real velocity in `CarState` and piping it into the existing teammate observation slot without changing input dimensions. Phase 2 adds a new Tier 6 observation tier with 103 base channels + 21 opponent-relative channels (3 opponent slots × 7 ego-relative channels) = 124 total inputs. Plan C (track-centric Frenét refactor) remains deferred until continuous arc-length projection is implemented as a standalone prerequisite.

- Racing Perception Redesign
- Plan: [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md) [PLANNED]
- Current internal state: Phase 1 [PLANNED] (Step 01 planning packet authored, green-light verification recorded); Phase 2 [PLANNED].
- Upstream dependency: [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE]

**Coordination rule:** this lane owns the `examples/racing_curriculum/` observation and `CarState` surfaces and any racing-specific perception plumbing in `src/` required for the new opponent-aware tier. It must preserve the Tier 1–6 ladder, promotion rules, carry/reset semantics, and observation dimension contracts from `examples/racing_curriculum/reference.plans.md`. It does not own general NGE core algorithm work; those changes belong in completed upstream workstreams.

## Standalone Workspace MCP Registration Lane [DONE]

**Outcome:** register the repo-owned direct MCP servers in `.vscode/mcp.json`
with a fixed active-plan path so workspace-local MCP discovery can resolve the
repo-static workflow packet and the active-step validation allow-list without
prompt inputs.

## Standalone Folder Quality Gate + Racing Curriculum Hotfix Lane [DONE]

**Outcome:** closed the browser-runtime `NetworkActivateInputSizeMismatchError` (expected 70, got 70)
triggered when `nge.controller.ts` passed a `Float32Array` to `network.activate(...)`, and shipped
`scripts/folder-quality-metrics.mjs` as the fast static folder-quality gate for post-edit workflow
checks.

- Folder quality gate and racing curriculum hotfix
- Plan: [completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md](completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md) [DONE]
- Current internal state: archived after the typed-array runtime fix, the bounded
  `number[] | Float32Array` activation contract update, the new `quality:folder` gate, and green
  validation. Deferred caveats remain explicit: one pre-existing `examples/racing_curriculum`
  lint error, the pre-existing `missing-test-file` smells in `examples/racing_curriculum` and
  `src/architecture/network/activate`, and the PowerShell `npm run quality:folder` forwarding
  quirk.

**Coordination rule:** this lane owns `src/architecture/network/activate/network.activate.core.utils.ts`
(and parallel activate guards) for the bug fix, and `scripts/folder-quality-metrics.mjs` +
`scripts/agent-customization/gates/folder-quality.gate.mjs` + `package.json` `quality:folder`
script + mandatory checklist entries in `.github/copilot-instructions.md` and the
relevant skills for the tooling half. Racing Curriculum Phase 3 (Tier 3: 2v2 Roles) is not part of this lane.

- Workspace MCP registration
- Plan: [completed/workspace-mcp-registration.plans.md](completed/workspace-mcp-registration.plans.md) [DONE]
- Current internal state: the workspace-registration lane is archived as a
  reopen-only baseline after the bounded `.vscode/mcp.json` registration
  pass, plan-sync alignment, and shared stdio transport repair. The
  workspace binding now points at the archived plan path so the fixed plan
  contract remains explicit, and any future live allow-list work must reopen
  from this archive instead of treating it as an active packet.

## Standalone Meta-Workflow Lane — Agentic Workflow Architecture [DONE]

**Outcome:** make AI-assisted development in this repository itself a
first-class, validated product surface: seven numbered user-invocable phase
agents, hidden specialist delegation, skill-first durable workflow knowledge,
model routing, validation scripts, skill evals, and plan updates after every
step.

- Agentic workflow architecture and customization validation
- Plan: [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE]
- Current internal state: the standalone meta-workflow lane is archived as a
  reopen-only baseline after closing the MCP runtime-visibility closure pass.
  The repo now has seven numbered user-invocable phase agents, hidden
  specialist delegation, skill-first workflow knowledge, model routing,
  validation scripts and evals, and the documented MCP ownership boundaries.

**Coordination rule:** this lane is closed for current scope. Any future agent
or skill customization change that needs new workflow architecture should
reopen from
[completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md)
before touching `.github/agents/`, `.github/skills/`, or customization
validation scripts.

## Standalone MCP Validation and Hardening Lane [DONE]

**Outcome:** validated and hardened all three configured MCP servers (`neataptic-gate-mcp`, `neataptic-workflow-mcp`, `neataptic-validation-mcp`) for correct post-restart stdio behavior, JSON-RPC framing, tool/resource/prompt lists, representative tool calls, gate MCP contracts, and allow-list security.

- MCP server validation and hardening
- Plan: [completed/MCP_Server_Validation_and_Hardening.plans.md](completed/MCP_Server_Validation_and_Hardening.plans.md) [DONE]
- Archive note: future reopen work should start from the completed baseline and its matching log, not from a restored active packet.
- Permanent binding: [mcp-active-binding.plans.md](mcp-active-binding.plans.md) [WIP] — stable perpetual `loadActivePlanContext` target for MCP server startup; never archive.

**Coordination rule:** this lane is confined to `.vscode/mcp.json`, `scripts/agent-customization/mcp/**`, `scripts/agent-customization/gates/**`, `.github/flows/**` (MCP flow contracts only), and tracker/log files. Do not change `src/` without explicit user confirmation.

## Standalone Meta-Workflow Lane — Agentic Flows and Gates Upgrade [DONE]

**Outcome:** upgrade the repo customization workflow from nested delegation to
named agent flows, deterministic read-only exit gates, flow-aware local MCP
resources/prompts/tools, evidence-bearing gate exceptions, and universal
`00-helping` escalation with learning-event observability.

- Agentic flows, gates, and universal helper upgrade
- Plan: [completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md) [DONE]
- All six migration phases (A-F) complete: 30 named flows, 12 gates, 3 MCP servers, 61/61 gate tests green, all 9 acceptance criteria satisfied. Archived.

**Coordination rule:** this lane is workflow-only infrastructure. Do not modify
`src/` while executing it, and treat
[completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md)
as the archived baseline for existing agent, skill, model-routing, validation,
and MCP ownership contracts.

## Standalone Meta-Workflow Lane — Orchestration System Optimization [DONE]

**Outcome:** enforce strict Tier 1→Tier 2/3/4 delegation by extracting durable policies into skills, creating targeted specialist agents, and updating flows to eliminate "God-agent" behavior from the eight numbered SDLC orchestrators.

- Orchestration system optimization (mini-agent transition)
- Plan: [completed/Orchestration_System_Optimization.plans.md](completed/Orchestration_System_Optimization.plans.md) [DONE]
- Final state: all 5 phases complete. Structural delegation infrastructure delivered (execute skill, routing table, frontmatter agents arrays, specialist creation, flow specialist references). Archived. The instructional follow-up (per-agent body mandates, output contract tightening, gate enforcement) is handled by the Tier 1 Delegation Remediation lane below.

**Coordination rule:** this lane is confined to `.github/agents/`, `.github/skills/`, `.github/flows/`, `scripts/agent-customization/`, and tracker/log files. Do not modify `src/` or MCP server implementations. Treat [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) and [completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md) as the archived baselines for agent architecture and flow/gates contracts.

## Standalone Meta-Workflow Lane — Chrome DevTools MCP Integration [DONE]

**Outcome:** integrate the Chrome DevTools MCP server into the multi-tier agent orchestration
system to enable direct performance measurements, UI testing, and browser-based validation.
Delivered three new Tier 3 specialists (performance-trace-specialist, browser-ui-specialist,
browser-memory-specialist), two new skills (`execute` for mandatory delegation enforcement and
`devtools` for durable browser tool knowledge), strict sliced RED→IMPLEMENT→GREEN
implementation loop enforcement, updated testing agents with Chrome DevTools MCP awareness,
trace analysis infrastructure scripts, and two new validation gates. All 8 phases complete;
all gates pass; plan archived.

- Chrome DevTools MCP integration
- Plan: [completed/Chrome_DevTools_MCP_Integration.plans.md](completed/Chrome_DevTools_MCP_Integration.plans.md) [DONE]
- Final state: 65 agents (3 new Tier 3), 58 skills (2 new), 52 new tests (100% coverage),
  2 new gates, all 11 flows updated with sliced loop-back protocol. Archived to
  `plans/completed/`.

**Coordination rule:** this lane is confined to `.github/agents/`, `.github/skills/`,
`.github/flows/`, `.github/copilot-instructions.md`, `scripts/agent-customization/`,
`scripts/analyze-trace/`, `.vscode/mcp.json`, `.gitignore`, and plan index files. Do not modify
`src/` library code.

## Standalone Meta-Workflow Lane — Chrome MCP Browser Tests [DONE]

**Outcome:** add a reusable Chrome-MCP-powered browser testing harness and hidden docs URLs
so maintainers can run focused browser scenarios locally against `http://localhost:8080` and
capture performance traces, DOM state, console output, and memory metrics via the existing
Chrome DevTools MCP server and its three specialists.

- Chrome MCP browser tests
- Plan: [completed/Chrome_MCP_Browser_Tests.plans.md](completed/Chrome_MCP_Browser_Tests.plans.md) [DONE]
- Final internal state: all seven phases complete. DevTools-coverage gate has a pre-existing naming mismatch (expects skill `devtools`, canonical skill is `chrome-devtools-mcp`) recorded as a lingering defect; it did not block closure.
- First scenario: WebGPU inference smoke test at
  `docs/browser-tests/webgpu-inference-smoke.html` asserting CPU/GPU output parity for
  `src/architecture/network/gpu/`.

**Coordination rule:** this lane is confined to `.github/skills/browser-testing-harness/`,
`.github/agents/browser-harness-specialist.agent.md`, updates to `03-red-testing`,
`04-implementing`, `05-green-testing`, `06-documenting` and the three existing browser
specialists, `scripts/agent-customization/browser-tests/`, `docs/browser-tests/`, and plan
index files. Do not modify `src/` library code unless a later phase explicitly scopes a WebGPU gap.

## Standalone Meta-Workflow Lane — MCP Lazy-Load Facade [DONE]

**Outcome:** reduce no-op MCP session startup cost by replacing the heavy
`devtools` and `cortex` server registrations with
lightweight stdio facades (`cortex`, `devtools`) that expose only short
stripped tool descriptions at session start and lazily spawn the real servers
on first actual tool use. Preserve the existing tool-name surface and routing
so agents, skills, `copilot-instructions.md`, and validation gates continue to
work after the rename. Remove old server entries from `.vscode/mcp.json` and
`.mcp.json` in the same step that adds the facades — no deferred cleanup and no
dual-path compatibility wrappers.

- MCP lazy-load facade
- Plan: [completed/MCP_Lazy_Load_Facade.plans.md](completed/MCP_Lazy_Load_Facade.plans.md) [DONE]
- Current state: Phase 1 complete and archived. Router-tool facades implemented, all markdown callers migrated, configs updated, routing table and semantic index regenerated, green validation passed. User restart of Copilot CLI / VS Code required before facades become active.
- Stop/reset checkpoint: after `.vscode/mcp.json`/`.mcp.json` are updated and
  green validation passes, the user must restart Copilot CLI / VS Code before
  the new facades are active.

**Coordination rule:** this lane is confined to `scripts/agent-customization/mcp/`,
`.vscode/mcp.json`, `.mcp.json`, `.github/copilot-instructions.md`,
`.github/agents/`, `.github/skills/`, `.github/flows/`,
`scripts/agent-customization/hooks/`, `scripts/agent-customization/gates/`, and
plan index files. It may rename agent/skill/flow references and gate tool names
but must not modify `src/` library code or the core search/embedding logic inside
`scripts/mcp-semantic/`.

## Standalone Meta-Workflow Lane — Step Packet Goal Redesign [DONE]

**Outcome:** replace the step packet YAML `agent` and `agent_file` fields with a `goal` field that declares what outcome a step needs rather than who does it, add an optional `tdd_sequence` field for multi-phase dispatch decomposition, update the orchestrator routing table in `copilot-instructions.md` §3, migrate all existing plan files, and remove backward compatibility.

- Step Packet Goal Redesign
- Plan: [completed/Step_Packet_Goal_Redesign.plans.md](completed/Step_Packet_Goal_Redesign.plans.md) [DONE]
- Final state: goal-based dispatch and tdd_sequence migration complete. Archived.

**Coordination rule:** this lane is confined to `.github/copilot-instructions.md`, `.github/skills/phase-handoff-workflow/SKILL.md`, `scripts/agent-customization/gates/step-packet.gate.mjs`, and plan file step packet YAML. Do not modify `src/` library code, flow YAML files (`.github/flows/*.flow.yml` use `agent:` for flow ownership, a different concern), or runtime enforcement scripts.

## Standalone Meta-Workflow Lane — Holistic Agent & Skill Optimization [DONE]

**Outcome:** grade, fix, and regrade ALL 65 agents (8 Tier 1, 11 Tier 2, 42 Tier 3, 4 Tier 4) and ALL 58 skills to 100/100/100 across three dimensions — Orchestration, Tools & Skills, and Role Knowledge — using a grade→fix→regrade loop where the executor never assesses itself. Then update all WIP/PLANNED plans to use the updated agentic contracts, `delegate_to` fields, and delegation mandates. Supersedes the archived Tier 1 Delegation Remediation lane, which addressed only Tier 1 instructional and output-contract gaps (P0–P5: output contract tightening + gate enforcement, Mission-level delegation mandate, slice `delegate_to` schema field, Tier 1→specialist lookup table, routing table body references, ordered flow dispatch sequences). The structural infrastructure from the archived Orchestration System Optimization lane remains sound; this lane expands remediation to every tier and every skill. Plan archived to `plans/completed/holistic-agent-skill-optimization.plans.md`.

- Holistic agent & skill optimization
- Plan: [completed/holistic-agent-skill-optimization.plans.md](completed/holistic-agent-skill-optimization.plans.md) [DONE]
- Final state: all 65 agents and 58 skills graded/fixed/regraded; P0–P5 remediation gaps addressed; plan archived.
- `agent-json-body-to-md` skill load fix
- Plan: [completed/agent-json-body-to-md-skill-load-fix.plans.md](completed/agent-json-body-to-md-skill-load-fix.plans.md) [DONE]
- Scope: fix unescaped apostrophe in skill frontmatter `description` so strict YAML parsers can load the file; regenerate routing table; verify skill goal and downstream consumers remain intact.

**Coordination rule:** this lane is confined to `.github/agents/*.agent.md` (65 files — all tiers), `.github/skills/*/SKILL.md` (58 files), `.github/skills/execute/SKILL.md`, `.github/flows/*.flow.yml`, `scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs` + `.test.ts`, `.github/agent-skill-routing-table.md` (regeneration only if frontmatter changed), WIP/PLANNED plan files, and tracker/log files. Do not modify `src/` library code or MCP server implementations. Treat [completed/Orchestration_System_Optimization.plans.md](completed/Orchestration_System_Optimization.plans.md) as the archived structural baseline and this lane as its expanded holistic follow-up.

## Standalone External Tool Assimilation Lane [DONE]

**Outcome:** deliver a repeatable skill/agent flow (`external-tool-assimilation` skill,
`assimilator` specialist) that analyzes an external GitHub repository and produces a local
comparison study under `<repo-name>/` with verbatim copies, per-area summaries, and a
final synthesis of actionable assimilation recommendations.

- External tool assimilation
- Plan: [completed/assimilate-repeatable-skill.plans.md](completed/assimilate-repeatable-skill.plans.md) [DONE]
- Final state: all seven steps [DONE]; skill, agent, templates, script, tests, and sample assimilation delivered and validated. Detailed evidence in [completed/assimilate-repeatable-skill.logs.md](completed/assimilate-repeatable-skill.logs.md).

**Coordination rule:** this lane touched `.github/skills/external-tool-assimilation/SKILL.md`,
`.github/agents/assimilator.agent.md`, `.github/templates/assimilation/`,
`scripts/assimilation/`, and plan index files. It did not modify `src/` library code or
MCP server implementations.

## Standalone Context Optimization Lane [DONE]

**Outcome:** reduce Copilot CLI context-window overhead by repo-side changes to instructions and tool-catalog descriptions, while deferring runtime-dependent lazy-loading and schema-split work to future spikes.

- Context optimization
- Plan: [context-optimization.plans.md](completed/context-optimization.plans.md) [DONE]
- Current internal state: Phase 1 complete; all seven steps [DONE]; green validation passed; tracker archived to `plans/completed/`.

**Coordination rule:** this lane is confined to `.github/copilot-instructions.md`, `.github/agents/*.agent.md`, `.github/skills/*/SKILL.md`, `.github/agent-skill-routing-table.md`, `scripts/agent-customization/`, and plan tracker/log files. It must not modify `src/` library code. Runtime-dependent changes (lazy tool schemas, splitting the `task` tool) are recorded as deferred spikes owned by `00-helping`/external CLI team and are out of scope for this lane.

## Standalone Documentation Metrics Contract Lane [DONE]

**Outcome:** permanently eliminate weak-doc metric inconsistency by shipping a canonical
versioned metric contract, deterministic runner artifacts, strict compare guards, MCP/CLI parity,
and CI-enforced validation.

- Docs quality metrics contract and parity
- Plan: [completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md](completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md) [DONE]
- Current internal state: this lane is archived as a reopen-only baseline after closing all seven
  steps, including canonical command wiring (`docs:quality:metrics`, `docs:quality:compare`),
  deterministic normalization and run artifacts, fail-fast mismatch reason codes, MCP and CLI
  parity, docs-quality CI gate wiring, and migration runbook closure.

**Coordination rule:** keep this lane constrained to docs-quality metric ownership surfaces
(`rag-index/**`, `scripts/agent-customization/mcp/**`, gate scripts, package scripts,
CI workflow wiring, and documentation). Avoid unrelated architecture or runtime refactors.

## Standalone Docs-Quality Complexity Cleanup Lane [DONE]

**Outcome:** reduce `npm run docs:quality:metrics` `highComplexity` findings in `src/` to zero by decomposing high-cyclomatic owners into orchestration-first private helpers without changing public APIs or the docs-quality metric contract.

- Docs quality complexity cleanup
- Plan: [completed/docs-quality-complexity-cleanup.plans.md](completed/docs-quality-complexity-cleanup.plans.md) [DONE]
- Current internal state: this lane is archived after reaching a clean docs-quality metrics summary, a clean TypeScript typecheck, and owned `src/architecture/node/node.ts` coverage at `100/100/100/100`; the unrelated `examples/neatChat/core/neatChat.live-flow.safety.test.ts` baseline stays outside this closed boundary.

**Coordination rule:** keep this lane constrained to `src/` complexity decomposition, owner-local test validation, and tracker evidence. Reopen the archived docs-quality contract lane only when changing the metrics contract, artifact schema, or CLI/MCP parity surfaces.

## Standalone Coverage Metrics + Gap Closure Lane [DONE]

**Outcome:** archived standalone docs-quality coverage lane for additive `summary.coverage` reporting in `npm run docs:quality:metrics`, deterministic per-file deficit rows, and final `src` coverage closure at `100/100/100/100`.

- Coverage metrics + gap closure
- Plan: [completed/coverage-metrics-and-gap-closure.plans.md](completed/coverage-metrics-and-gap-closure.plans.md) [DONE]
- Current internal state: archived after `npm run test:silent` passed at `427` suites / `4654` tests with `All files` and `src` at `100/100/100/100`, `npm run --silent docs:quality:metrics` reported `filesBelow100: 0`, and `npm run --silent docs:quality:gate` passed.

## Phase 0 — Hygiene + Refactor Baseline [DONE]

**Outcome:** keep iteration speed high, reduce refactor risk, and finish the structural cleanup needed before broad mechanical modernization.

- Demo refinement and learnability hardening [DONE]
- `asciiMaze` SOLID split before ES2023 modernization [DONE]
- `flappy_bird` reference demo split and documentation baseline [DONE]
- Main app NEAT surface is already SOLID split [DONE]
- Educational documentation and split follow-through lane [DONE]
- [Flappy_Bird_Folder_Documentation_Pass.md](completed/Flappy_Bird_Folder_Documentation_Pass.md) [DONE]
- [architecture-solid-split.plans.md](completed/architecture-solid-split.plans.md) [DONE]
- [methods-solid-split.plans.md](completed/methods-solid-split.plans.md) [DONE]
- [methods-docs.plans.md](completed/methods-docs.plans.md) [DONE]
- [neat-docs.plans.md](completed/neat-docs.plans.md) [DONE]
- [readme-first-section-pass.plans.md](completed/readme-first-section-pass.plans.md) [DONE]
- [utils-docs.plans.md](completed/utils-docs.plans.md) [DONE]
- Both demos are now solid split and the Flappy Bird documentation pass is complete enough to stop being a documentation blocker.
- The architecture split follow-through and the NEAT educational-docs lane are now also complete enough to stop being Phase 0 blockers.
- The repo-wide README first-section pass is now closed across the tracked README surfaces under `src/` and `examples/`.
- The remaining structural polish in Phase 0 is the repository-wide modernization pass plus its validation gate.
- Supporting repair and docs-tooling stabilization lane [DONE]
- [neat-test-surface-repair.plans.md](completed/neat-test-surface-repair.plans.md) [DONE]
- [asciiMaze-typescript-repair.plans.md](completed/asciiMaze-typescript-repair.plans.md) [DONE]
- [generate-docs-solid-split.plans.md](completed/generate-docs-solid-split.plans.md) [DONE]
- [render-docs-html-solid-split.plans.md](completed/render-docs-html-solid-split.plans.md) [DONE]
- [analyze-trace-solid-split.plans.md](completed/analyze-trace-solid-split.plans.md) [DONE]
- These remain roadmap-visible as reopen points and tooling baselines even though they do not change the forward critical path out of Phase 0.
- Source strict-typing cleanup for `src/` explicit-`any` debt [DONE]
- Plan: [src-no-explicit-any-cleanup.plans.md](completed/src-no-explicit-any-cleanup.plans.md)[DONE]
- Scope note: this lane replaced the stale root checklist with a roadmap-tracked plan aligned to the current folderized tree and is now the closed baseline for future reopen-only follow-up.
- ES2023 modernization (completed Phase 0 lane: project-wide named errors with `Error.cause`, targeted syntax cleanup, helper normalization, narrow module-edge cleanup, and lint/CI enforcement) [DONE]
- Plan: [ES2023 migration](completed/ES2023%20migration)
- Scope note: ESM package wiring, the ES2023 TypeScript baseline, the lint scaffold, named-error rollout, targeted syntax cleanup, shared clone-helper normalization, narrow workflow or benchmark edge cleanup, and CI enforcement are now complete. Memory-management or performance-feature reopen work is preserved in [completed/Memory_Optimization.md](completed/Memory_Optimization.md).

**Gate to Phase 1:** satisfied. Both demos are solid split and documented, the main app split is stable, the remaining documentation work is no longer obscuring ownership boundaries, the ES2023 cleanup lane is complete, and `npm run build`, `npm run lint`, and `npm test` are green after the modernization pass.

## Phase 1 — Core Correctness + Determinism Foundations (Critical Path)

**Outcome:** deterministic semantics and “proper NEAT” correctness so everything built on top is reliable.

1. Proper NEAT (historical markings, correct crossover alignment, recurrent/self-connection policy, RNG determinism)

- Plan: [neat.plans.md](completed/neat.plans.md) [DONE]
- Current internal state: the proper-NEAT lane is closed for its current scope through local Phase 7; any later beyond-paper continuation is deferred until an explicit reopen decision.

2. Explicit I/O roles + stable activation ordering (acyclic + recurrent mode semantics)

- Plan: [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md) [DONE]

**Why this ordering:**

- Fixing NEAT correctness prevents “paper cuts” and broken invariants from leaking into every future feature.
- Stable execution semantics (I/O + scheduling) are required for safe builders, serialization, export, and workers.

**Gate to Phase 2:** satisfied for the current roadmap scope. The proper-NEAT and stable activation-ordering lanes are closed, and explicit I/O roles plus deterministic acyclic and recurrent scheduling are now the documented runtime baseline.

## Phase 2 — Architecture DX (Build Graphs Safely)

**Outcome:** a clear, typed, deterministic “build” story for users and for internal features.

3. Architecture primitives (Node/Group/Layer)

- Plan: [Architecture_Primitives_Node_Group_Layer.md](completed/Architecture_Primitives_Node_Group_Layer.md) [DONE]
- Current internal state: the primitive DX baseline is closed; construction-time roles, lightweight descriptors, and source-mapped docs are in place, and the whole-graph construct baseline is now archived in [completed/Construct_From_Parts_Graph_Assembly.md](completed/Construct_From_Parts_Graph_Assembly.md).

4. Deterministic construct-from-parts (validation + scheduling integration)

- Plan: [Construct_From_Parts_Graph_Assembly.md](completed/Construct_From_Parts_Graph_Assembly.md) [DONE]
- Current internal state: the construct baseline is closed for the current Phase 2 scope. Public `Network.construct(...)` now covers deterministic materialization, construct-owned diagnostics, explicit public I/O validation, detached graph snapshots plus summary formatting, and adjacent runtime seam coverage across serialization, training, evolution, crossover, and builder interoperability.

5. Preconfigured architectures (MLP + sequence builders)

- Plan: [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md) [DONE]
- Current internal state: all ten steps are closed. MLP, RandomSparse, NARX, GRU, and LSTM builders are complete; both Flappy Bird and ASCII Maze consume builder-backed seed profiles through the shared example profile contract; cross-demo e2e tests cover the full approved profile matrix; the trainer and worker are both profile-aware for recurrent growth settings; and both demo READMEs document the approved architecture families.

**Gate to Phase 3:** satisfied. Builders use explicit I/O roles, construction is deterministic with actionable diagnostics, and all five preconfigured architecture families (MLP, RandomSparse, NARX, GRU, LSTM) are complete and integrated into both flagship demos.

## Phase 3 — Adoption + Learnability (Browser + Examples + Visualization)

**Outcome:** reduce friction for new users and make the library “tryable” immediately.

6. Browser build + CDN distribution (ESM + IIFE bundles; stable public surface)

- Plan: [Browser_Build_and_CDN_Distribution.md](completed/Browser_Build_and_CDN_Distribution.md) [DONE]

7. Interactive examples + learning path (Node + browser runnable examples, CI smoke checks)

- Plan: [Interactive_Examples_and_Learning_Path.md](completed/Interactive_Examples_and_Learning_Path.plans.md) [DONE]

8. NEATchat (tiny online sequence-learning chatbot demo)

- Plan: [completed/NEATchat.plans.md](completed/NEATchat.plans.md) [DONE]
- Current internal state: the NEATchat Phase 3 learnability lane is closed for current scope and archived as a completed baseline. The delivered surface remains intentionally toy-scale: near-zero start, short online adaptation, language-agnostic token-stream behavior, and browser or Node experimentation boundaries rather than web-scale training.
- Follow-up lane: [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE]. The post-toy conversational-systems lane closed. All six workstreams closed: W1 durable substrate, W2 stronger seed import, W3 episodic memory and retrieval, W4 background adaptation and candidate search, W5 hybrid routing, W6 evaluation harness, safety gate, and publishable product shape. Unrelated broad-run heap OOM remains outside NEATchat ownership.

9. Visualization export schema (JSON schema + optional DOT output)

- Plan: [Network_Visualization_Export_Schema.md](completed/Network_Visualization_Export_Schema.plans.md) [DONE]
- Current internal state: all three lanes are closed. `exportVisualizationGraph` + `toDot` schema/DOT export, shared canvas renderer (`renderNetworkView`), and Lane C documentation examples are complete and tested.

**Gate to Phase 4:** satisfied. Browser bundle distribution remains [PLANNED] but is not a gate blocker; interactive examples, NEATchat, and visualization export schema are all closed. Phase 4 is the current active stage.

## Phase 4 — Deployment + Parallel Evaluation (Inference Artifacts, Workers, Checkpoints)

**Outcome:** production and scale workflows: export models, evaluate quickly, resume long runs.

9. Standalone inference export (dependency-free runtime output)

- Plan: [Standalone_Inference_Export.md](completed/Standalone_Inference_Export.md) [DONE]
- Current internal state: the planning baseline for the standalone deployment lane is closed. The legacy `network.standalone()` generator now honors compiled activation traversal and explicit I/O role ordering, and the next implementation reopen point is to create `src/architecture/network/export/` and freeze `InferenceIRv1` when Phase 4 becomes active.

10. Worker-friendly serialization fastpath (clone/transfer payloads; predictor creation)

- Plan: [completed/Worker_Friendly_Network_Serialization_Fastpath.md](completed/Worker_Friendly_Network_Serialization_Fastpath.md) [DONE]
- Current internal state: Phase 0 through Phase 3 are complete and user-confirmed. `PortableInferencePayload`, `TransferableInferencePayload`, `InferenceChannel`, and `SharedInferenceWorker` are implemented, the worker transport substrate and turnkey extraction brief are complete, and the Step 4.8 user-host acceptance gate is closed. The accepted Flappy runtime policy is worker-local recurrent browser evaluation on the isolated host, while the shared-memory pool remains the reusable transport proof surface rather than the active recurrent browser default.

11. Turnkey multithread evaluation API (Node + browser workers)

- Plan: [completed/Turnkey_Multithread_Evaluation_API.md](completed/Turnkey_Multithread_Evaluation_API.md) [DONE]
- Current internal state: the turnkey extraction lane is complete through all six steps. The library now exposes reusable capability detection, automatic transport selection, a shared browser worker asset resolver, `ParallelInferencePool`, `evaluateInWorkers(...)`, and `createNeatParallelPopulationEvaluator(...)`, and Flappy exercises the full shared helper ladder while the docs explain the split between library-owned worker helpers and example-owned browser-worker policy.

12. Population save/resume + checkpointing (full vs light checkpoints, determinism contracts)

- Plan: [completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md) [DONE]
- Current internal state: the checkpointing lane is closed. The repo now ships a documented persistence ladder across population-only snapshots, light checkpoints, and strict full checkpoints, including the reserved downstream `extensions` bag across the `Neat` and `src/neat/export/` teaching surfaces.

13. Evolution–training interoperability contracts (parameter vectors, isolation, hybrid policies)

- Plan: [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md) [DONE]
- Current internal state: the hybrid-interoperability lane is archived as done. The closed baseline now covers deterministic `ParameterLayoutV1` ordering, parameter-vector export/import, `fineTuneVector(...)` isolation, explicit hybrid evaluation policy with Lamarckian opt-in, root-facade re-exports, and generated docs closure. The unchanged external ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains outside this lane.

**Why this ordering:**

- Standalone export and worker payloads share an “inference IR” concept; building that once reduces duplication.
- Worker transport should prove the substrate and the Flappy proof-of-concept first; the turnkey API should then extract the example-owned ergonomics into reusable public helpers.
- Checkpointing and hybrid evaluation benefit from deterministic scheduling and a clear parameter/vector mapping.
- Evolution-training parameter vectors are a later unification seam, not a blocker for standalone export or worker payloads unless that contract is deliberately split into an earlier mini-phase.

Next critical-path frontier:

- [completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md) now records the closed Phase 4 checkpointing baseline and the reopen point for future persistence work.
- The archived pre-NGE ONNX baseline now lives at [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md), and the archived hybrid baseline now lives at [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md). The NEATchat follow-up is now archived at [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE] with all six workstreams closed. The pre-NGE stop line is now satisfied; Phase 7 / NGE is the next frontier.

## Phase 5 — Scale & Performance (Memory Optimization Track)

**Outcome:** handle very large graphs and long runs on commodity hardware, across Node + browser.

This plan is large and can run as a **parallel lane** after Phase 1, but it should not destabilize correctness work.

- Memory & performance multi-layer strategy (Track 1: Phases 0–10; Track 2 gates Hyper work)
- Plan: [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]
- Current internal state: the pre-NGE memory lane is closed through Track 1 / Phase 10. The archived plan remains the authoritative memory-foundation baseline for later NGE work, while new Track 2 memory changes should reopen from the archive only when the NGE plans truly need foundational contract changes.

**Recommended sequencing guidance:**

- Start / continue Track 1 after Phase 1 is stable.
- Prioritize improvements that directly benefit the worker payload + inference export paths (typed arrays, slabs, reuse) so Phase 4 gets faster “for free”.
- If a detailed memory-plan subsection implies Hyper work can begin immediately after an intermediate Track 1 checkpoint, treat that as stale wording; Hyper remains gated by the Track 1 conditions below.

**Gate to Phase 7 (Hyper):** satisfied. The archived memory baseline in `plans/completed/Memory_Optimization.md` has closed its Track 1 stop line and no longer blocks Phase 7.

## Phase 6 — Interoperability Breadth (ONNX)

**Outcome:** standards-compliant ONNX serialization, checker-backed validation, runtime compatibility for the declared supported subset, and named external import subsets with honest rejection outside that boundary.

- ONNX export/import breadth and hardening
- Plan: [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE]
- Current internal state: the ONNX lane is archived as done through the current Phase 9 compliance target for the declared lower-opset same-family subset. The archived baseline now covers recurrent hardening, the conservative spatial contract, the same-family advanced-graph contract, the optimization and fidelity contract, the exporter-owned precision contract, deterministic binary `ModelProto` emission, runtime parity for the approved five-lane subset, and the first named external binary import subset through `importFromONNXBinary()`. Future ONNX work should reopen from the archive only through a narrower new amendment.
- NEATchat follow-up (persistent pretrained conversational system)
- Plan: [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE]
- Current internal state: all six workstreams closed. W1 snapshot v2 and durable substrate; W2 stronger default seed and honest external-seed contract; W3 episodic memory bank and token-overlap retrieval; W4 async-deferred background adaptation and explicit candidate lifecycle; W5 three-path hybrid routing with observability-only routingLog; W6 attribution-aware regression harness, structured safety gate, and publishable browser demo with explicit live/experimental/background-job lane labels.
- NEATchat live exchange safety red contract (live safety gate + snapshot quality loop)
- Plan: [completed/neatChat-live-safety-red.plans.md](completed/neatChat-live-safety-red.plans.md) [DONE]
- Current internal state: archived as the closed live safety baseline. `runNeatChatExchange` now screens routing candidates through `checkSafety`, rejects placeholder and incomplete-fragment outputs, and uses a vocabulary-aware bounded fallback floor that satisfies the requested shipped-browser four-turn bar. Reopen only for stricter extended-turn freshness.

**Recommended timing:**

- Treat the archived ONNX baseline as the reopen point for future interoperability widening rather than keeping the old root tracker active.
- Keep future ONNX scope honest: start from a narrower new amendment when runtime, external-import, or custom-domain claims need to change instead of reopening the archived baseline implicitly.
- Treat the NEATchat follow-up lane as an applied consumer of Phase 4 plus Phase 6 work, not as a shortcut around those foundations.

## Standalone Repo Cortex / Semantic Helping Lane [PLANNED]

**Outcome:** a no-compromise semantic helping system for the NeatapticTS repo, library, and
all demos — covering a SQLite corpus index with BM25 search and freshness proofs, MCP tools
exposing that index to AI agents, a browser-consumable JSON snapshot for all demos, the
archived 5-layer agent delegation tier enforcement prerequisite, and optional ONNX-backed
hybrid dense retrieval as the final advanced step. NeatChat also gains its own separate local
retrieval and memory layer for conversational quality.

This lane is **meta-workflow infrastructure**. It does not change `src/` library code and can
proceed in parallel with Phase 7 / NGE work. The six Repo Cortex layers execute sequentially;
Layers 1-5 are archived and Layer 6 remains planned. The archived Agentic Workflow Enforcement
Prerequisite ([completed/Delegation_Tier_Enforcement.plans.md](completed/Delegation_Tier_Enforcement.plans.md))
had no dependency on the SQLite corpus index and ran in parallel with Layers 1–3; it is now
[DONE] and no longer blocks Layer 4.
`NeatChat_Local_Retrieval_Memory.plans.md` can proceed in parallel with Layers 1–3.

### Repo Cortex layers (sequential)

1. Corpus index foundation (SQLite, BM25, freshness)

- Plan: [completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) [DONE]
- Artifacts: `rag-index/`, `rag-index/data/turso-replica.sqlite`

2. MCP tools (search, load, freshness, stats)

- Plan: [completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md) [DONE]
- Gate: Layer 1 [DONE] satisfied
- Artifacts: `scripts/mcp-semantic/`, `cortex` in `.vscode/mcp.json`

3. Browser snapshot and IndexedDB loader (all demos)

- Plan: [completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md) [DONE]
- Gate: Layers 1 and 2 [DONE] satisfied; Layer 3 archived
- Artifacts: `rag-index/snapshots/semantic-snapshot.json`, `examples/shared/semantic/`

4. Cortex MCP reliability hardening (Layer 4 — agents, skill, lifecycle gate, MCP enhancements)

- Plan: [completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md) [DONE]
- Gate: Layers 1, 2, and 3 [DONE] required; Delegation Tier Enforcement must be [DONE]; `neataptic-workflow-mcp.mjs` must be active
- Artifacts: `.github/agents/repo-cortex-scout.agent.md`, `.github/agents/cortex-embeddings-scout.agent.md`, `.github/skills/repo-cortex-workflow/SKILL.md`, `scripts/agent-customization/gates/cortex-index.gate.mjs`, `scripts/agent-customization/plan-session-redirect.mjs`, `scripts/agent-customization/validate-tsconfig-docs.mjs`

5. ONNX embeddings + hybrid BM25+dense ranking (Layer 5)

- Plan: [completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md) [DONE]
- Gate: Layers 1 and 2 [DONE] required
- Artifacts: `rag-index/embed-index.mjs`, `rag-index/data/embeddings.sqlite`, ONNX model cache

6. Embedding prewarm + default-on dense contract (Layer 6)

- Plan: [completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md) [DONE]
- Gate: Layer 5 [DONE] satisfied; Layer 6 archived after final prewarm, readiness, gate, MCP degradation, and tracker-closure validation
- Soft dependency: Layer 4 [DONE] improved MCP lifecycle management but was not required to close Layer 6
- Artifacts: `rag-index/prewarm-dense.mjs`, `rag-index/dense-readiness.mjs`, `scripts/agent-customization/gates/dense-readiness.gate.mjs`; MCP `search_corpus` now defaults `use_dense: true` with graceful cold-state degradation and warm-state `dense_state` provenance

7. Advanced RAG architecture (Layer 7+)

- Plan: [completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md](completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md) [DONE]
- Gate: Layers 1–6 [DONE] satisfied; builds on the existing BM25+dense hybrid
- Artifacts: semantic chunking, query classification, cross-encoder re-ranking, context window assembly, entity/relationship graphs, query expansion, relevance feedback, structured metadata filtering, multi-hop retrieval, ANN indexing, RAG eval suite

8. Premium primary search (Layer 7+ follow-up)

- Plan: [completed/Cortex_RAG_Premium_Primary_Search.plans.md](completed/Cortex_RAG_Premium_Primary_Search.plans.md) [DONE]
- Gate: Layer 7 [DONE] required
- Artifacts: smart freshness hooks, BM25/code-source fixes, reranker/graph hardening, exact symbol lookup, cold-start latency reduction, deterministic ANN, README noise filter, native search fallback, freshness transparency, feedback normalization, compact mode, single-call search-and-read, follow-up refs, explain_ranking, include_code_only, auto_fallback, tiered high-signal results

### Agentic Workflow Enforcement Prerequisite [DONE]

- 5-layer agent delegation tier enforcement
- Plan: [completed/Delegation_Tier_Enforcement.plans.md](completed/Delegation_Tier_Enforcement.plans.md) [DONE]
- Current internal state: archived baseline with 55 `tier:` frontmatter assignments, zero delegation-graph violations, the tier inventory or validator or gate or audit-report script surface, and `query_tier_graph` on `neataptic-gate-mcp`
- Layer 4 prerequisite: satisfied; Repo Cortex MCP Reliability may now add hidden specialist agents against the archived tier contract
- Artifacts: `scripts/agent-customization/tier-inventory.mjs`, `validate-agent-graph.mjs`, `tier-enforcement-gate.mjs`

### Parallel tracks (can run alongside Layers 1–3 and the prerequisite)

- NeatChat local retrieval and memory (separate from Repo Cortex)
- Plan: [completed/NeatChat_Local_Retrieval_Memory.plans.md](completed/NeatChat_Local_Retrieval_Memory.plans.md) [DONE]
- Depends on archived [completed/neatChat-live-safety-red.plans.md](completed/neatChat-live-safety-red.plans.md) [DONE] as the closed live safety baseline (no conflict; guarded integration)
- May reuse chunking and BM25 patterns from Layers 1–3 but must not import from `rag-index/`
- Artifacts: `examples/neatChat/memory/` (types, DB adapters, retrieval, services, tests)

**Coordination rules:**

- `completed/Delegation_Tier_Enforcement.plans.md` is the archived Agentic Workflow Enforcement Prerequisite baseline, not a Repo Cortex corpus layer. It has no "Layer N" designation.
- `rag-index/data/turso-replica.sqlite` and `rag-index/data/embeddings.sqlite` are gitignored generated artifacts.
- `rag-index/snapshots/semantic-snapshot.json` is a generated artifact; treat it as read-only.
- NeatChat memory (`examples/neatChat/memory/`) must never import from `rag-index/`.

## Standalone Turso RAG Migration Lane [WIP]

**Outcome:** migrate the Repo Cortex RAG system from the legacy synchronous local SQLite driver

- brute-force vector search to **Turso** (libSQL cloud database with
  native vector search, DiskANN ANN, FTS5, embedded replicas, and Platform API).
  This is the single largest infrastructure change the RAG layer has undergone:
  it replaces the database driver, the vector index, the hybrid ranking strategy,
  the deployment topology, and the connection model — all while keeping the 14+ MCP
  tools, the 26+ npm scripts, and the 58 skills / 65 agents that depend on Cortex
  search functioning. The goal is to make Turso-powered RAG the **unambiguous primary
  search mechanism** that every agent can rely on.

This lane is **meta-workflow infrastructure**. It does not change `src/` library
code and can proceed in parallel with Phase 7 / NGE work. It depends on all
archived Repo Cortex Layers 1–8 being [DONE] (they are). The 8 phases execute
sequentially:

- Turso RAG migration (8 phases, ~40 steps) [DONE]
- Plan: [completed/turso-rag-migration.plans.md](completed/turso-rag-migration.plans.md) [DONE]
- Final state: all phases complete, index warm and searchable, legacy SQLite artifacts removed.
- Phase 1: FTS5 compatibility, vector quantization, DiskANN recall, embedded
  replica topology, and full legacy sync SQLite driver import site audit.
- Phase 2: Schema migration (two SQLite DBs → one Turso DB, F8_BLOB embeddings,
  PRAGMA replacements, idempotent migration script).
- Phase 3: Core driver migration (legacy sync SQLite driver → @libsql/client, sync → async
  across 33+ files).
- Phase 4: Vector search migration (brute-force JS cosine → native
  vector_distance_cos() + DiskANN + metadata filtering).
- Phase 5: Search pipeline improvements (server-side RRF, parallel queries,
  batch transactions, server-side context assembly, SQL time-decay feedback).
- Phase 6: MCP tool updates (4 new tools: parallel_search, multi_hop_search,
  turso_branch, turso_pitr; update existing 14 tools for Turso-native features).
- Phase 7: Agent/skill/script documentation updates (copilot-instructions.md,
  implementation-standards skill, educational-docs skill, research-methodology
  skill, package.json scripts).
- Phase 8: Evaluation, optimization, and rollout (eval suite MRR@5 ≥ 0.350,
  latency optimization, final legacy sync SQLite driver cleanup, rollout signoff).

**Gate:** all archived Repo Cortex Layers 1–8 [DONE] satisfied.

**Coordination rule:** this lane is confined to `scripts/mcp-semantic/`,
`rag-index/`, `.mcp.json`, `.vscode/mcp.json`, `package.json` scripts,
and agent/skill documentation files. Do not modify `src/` library code. Treat the
archived Repo Cortex layer plans as the baselines this migration builds upon.

## Phase 7 — Advanced Research Features (Last)

**Outcome:** evo-devo / NGE capabilities and benchmark-driven validation that build on top of all prior infrastructure.

- NEAT Genesis EvoDevo (NGE) — core algorithm (computation motifs, lifecycle, DNA, reproduction, collective intelligence)
- Plan: [completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) [DONE]
- NGE Core Readiness — prior primitive-by-primitive readiness audit (computation motifs, lifecycle, DNA, reproduction, barriers, deterministic evaluation packs, experimental public API). Reopen-only baseline.
- Plan: [completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md](completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md) (`plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`) [DONE]
- Archive: [completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md](completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md) (`plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md`) [DONE]
- NGE Racing Curriculum — canonical long-form racing curriculum readiness plan: Phase 1 UI/behavior completion to Flappy Bird parity and inner-track centerline [DONE]; Phase 2 Tier 1 single-agent simple track [DONE] (worker-authoritative race-pack service, per-agent cyan/magenta guiding lines, browser visual confirmation, Tier 1 README contract); Phase 3 Tier 2 single-car-with-radio [DONE] (independent per-car NEAT agents, DR-011, 348 tests pass, lint clean, tsc clean); Phase 4 Tier 3 2v2 no pits [DONE] — all steps complete (4-car coevolution with independent genomes, shared-equal team fitness, 4-car browser rendering, worker-side continuous adaptation; Chrome DevTools MCP visual validation confirmed network growth N76/C288→N97/C372; 3 pre-existing race-pack test failures triaged as carry-forward debt); Phase 5 Tier 4 2v2 tires and pits [DONE] — all steps complete (95-channel observation, tire decay + grip multiplier wired into worker race-pack, pit lifecycle with 4-tick stops and 3 slots per team, DR-004/02-CORRECTION, 45 suites/385 tests pass, Chrome DevTools MCP visual confirmed tire markers and pit overlays, Tier 4 README contract with 3 Mermaid diagrams); Phase 6 Tier 5 3v3 full [DONE] — all steps complete (6-car coevolution with TIER_FIVE_CAR_COUNT=6, full 3-row radio population with self-broadcast, role-divergence observables with blockerDelta and inferredRole, 6-element pitStatus with layout-aware stride, renderer pit-overlay fix; 46 suites/394 tests pass, 3 skipped polyandric P1/P2; Chrome DevTools MCP visual confirmed Tier 5 N101/C388 STABLE 0 console errors; polyandric reproduction DEFERRED P1/P2 blockers nge-core-algorithm ownership; Tier 5 README contract with 3 Mermaid diagrams); Phase 7 Tier 6 3v3 advanced strategy [DONE] — all steps complete (FSM 5-bug fix DR-009, OpponentSnapshotPool wired with type adapter, strategy-divergence analytics module, multi-generation loop working, analytics-only fallback per DR-008; 68 suites/502 tests pass, 3 skipped polyandric P1/P2; modeIsEvolvable BLOCKED — nge-core-algorithm ownership; carry-forward blockers P1-P5 documented). Tier 1–6 green-gated ladder with promotion rules and carry/reset semantics from `examples/racing_curriculum/reference.plans.md`. Downstream trackers point here for tier semantics.
- Plan: [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) (plans/completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE]
- Phase 8 (Racing Curriculum v2) [DONE] — all 20 steps [DONE] and green-validated. Steps 01-17 archived in logs. Steps 18-19 fixed worker-authoritative demo evolution and Tier 1 follow-up defects. Step 20 fixed network growth blocker (evaluateRacingTrendScore void network; removed, 5 fixes: network-aware evaluator, tier promotion structure preservation, composite score signal, episodic slots, explicit config). Browser smoke: N109/C420 -> N523/C1524, ~60 FPS, 0 console errors.
- NGE Racing Curriculum Oscillation & Sub-Tier Fix [DONE] — core oscillation fix green-gated and complete (82/82 tests pass, NGE specialist APPROVED, all gates green). Remaining bundle rebuild and browser validation folded into Racing_Perception_Redesign.
- Plan: [completed/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md](completed/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md) [DONE]
- Racing Path-Tracking Debug and Quality Followup — pre-Phase-3 visual fix, geometry audit, and deferred quality cleanup
- Plan: [completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md](completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md) [DONE]
- Current internal state: archived after the shared-spline path-tracking repair, the
  user-confirmed rounded-lane visual pass, and the bounded folder-quality cleanup. The only
  remaining caveat is accepted static debt: `examples/racing_curriculum/browser-entry/browser-entry.ts`
  still lacks a sibling `browser-entry.test.ts`.
- NGE Ant Hive Ecosystem — multi-agent benchmark (stigmergy, role differentiation, collective intelligence). NGE Core Algorithm Workstream is now [DONE]; ant hive demo is unblocked.
- Plan: [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
- NGE Core Algorithm Workstream — complete the NGE core algorithm in `src/neat/` before any further demo work. Resolves carry-forward blockers P1 (NGE_DNA adoption & canonical envelope bridge), P2 (polyandric type exports), P3 (reproduction FSM integration), P4 (schema alignment), P5 (queenBias honoring), DR-008 (modeIsEvolvable activation), and the growth gap (agents stalled at 101 nodes / 388 connections vs. 8,000+ neuron target). All 7 phases are green-gated and [DONE]: NGE_DNA adoption, polyandric exports, modeIsEvolvable activation, growth engine diagnosis & fix, schema alignment, reproduction FSM integration, and end-to-end verification (seed → 8,000+ neurons with continuous adaptation; 139 suites / 1,709 tests pass, focused scale and polyandric suites pass, 100% coverage on touched production files). Workstream closed; plan/log pair archived to `plans/completed/`. Racing v2, ant hive, and predator/prey demo work is now unblocked. WebGPU acceleration remains a future performance lane after the core is closed.
- Plan: [completed/NGE_Core_Algorithm_Workstream.plans.md](completed/NGE_Core_Algorithm_Workstream.plans.md) (`plans/completed/NGE_Core_Algorithm_Workstream.plans.md`) [DONE]
- Archive: [completed/NGE_Core_Algorithm_Workstream.logs.md](completed/NGE_Core_Algorithm_Workstream.logs.md) (`plans/completed/NGE_Core_Algorithm_Workstream.logs.md`) [DONE]
- NGE GPU Acceleration — optional WebGPU inference fast path for slab-eligible acyclic NEAT networks. Builds on the completed NGE Core Algorithm Workstream and NGE Core Growth Engine Wiring, reuses the existing SoA/CSR slab layout, supports the worker activation registry, and falls back to CPU automatically. Primary first consumer is the racing-curriculum worker controller (6-car per-tick inference). [DONE]
- Plan: [completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md](completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md) (`plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`) [DONE]
- Archive: [completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md](completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md) (`plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.logs.md`) [DONE]
- NGE Core Growth Engine Wiring — bridge the two disconnected NGE growth systems (focus-scored morph planning and random runtime adaptation) into a single continuous pipeline. Fixes 6 audit gaps: missing morph applier (NgeMorphDelta[]→network.mutate), lifecycle never executing deltas, adaptOnTick using random operations instead of NGE focus-scoring, no morph-to-mutation mapping, limits too low (256→8k nodes / 1024→32k connections), commitGrowth never called after morph application. All 5 phases complete; Racing Curriculum unblocked.
- Plan: [completed/NGE_Core_Growth_Engine_Wiring.plans.md](completed/NGE_Core_Growth_Engine_Wiring.plans.md) (`plans/completed/NGE_Core_Growth_Engine_Wiring.plans.md`) [DONE]
- NGE Predator/Prey Co-evolution — co-evolutionary benchmark (sensory arms race, reproduction modes, non-stationary fitness). NGE Core Algorithm Workstream is now [DONE]; predator/prey demo is unblocked.
- Plan: [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]
- NGE Neon Shooter Demo (Neatenstein) — first-person shooter co-evolution demo with asymmetric main/enemy co-evolution, SWARM mode, and human-vs-evolved-enemy modes. NGE Core Algorithm Workstream is now [DONE]; Neon Shooter demo is unblocked.
- Plan: [Neon_Shooter_NGE_Demo.plans.md](Neon_Shooter_NGE_Demo.plans.md) [WIP]

## Standalone Public Library Demo-Agnostic Refactor Lane [DONE]

**Outcome:** remove demo-specific file names, exported symbols, JSDoc, internal identifiers, tests, and generated docs from `src/` so the public library expresses capabilities from the library point of view rather than any specific demo's perspective. This is a deliberate breaking public API refactor; a migration note in `RELEASE.md` maps every removed symbol to its generic replacement.

- Public library demo-agnostic refactor
- Plan: [completed/Public_Library_Demo_Agnostic_Refactor.plans.md](completed/Public_Library_Demo_Agnostic_Refactor.plans.md) (`plans\completed\Public_Library_Demo_Agnostic_Refactor.plans.md`) [DONE] — all 6 phases complete. Removed demo-specific names from `src/` public APIs, JSDoc, internal identifiers, tests, and generated docs; breaking public API refactor with migration notes in `RELEASE.md`.
- Archive: [completed/Public_Library_Demo_Agnostic_Refactor.logs.md](completed/Public_Library_Demo_Agnostic_Refactor.logs.md) (`plans\completed\Public_Library_Demo_Agnostic_Refactor.logs.md`) [DONE] — detailed done-state records for all 6 phases.
- Current internal state: Phases 1–6 [DONE]; plan/log pair archived to `plans/completed/`.

**Coordination rule:** this lane touches `src/architecture/network/gpu/network.gpu.racing.ts`, which is also being edited by the active WebGPU real-performance plan. Do not start Phase 3 implementation until the WebGPU plan's racing-file surface is stable. No `examples/` or `docs/browser-tests/` files should change; those surfaces are intentionally demo-specific and out of scope. Generated `src/**/README.md` files must be refreshed via `npm run docs`, never hand-edited.

**Why last:** this work depends heavily on the Memory Optimization track (Track 2 in that plan) and benefits from stable NEAT correctness, deterministic activation semantics, robust serialization/checkpointing, and a mature enough NGE core that benchmark results reflect the algorithm rather than unstable infrastructure.

## Summary: Critical Path vs Parallel Lanes

Current status: **The NGE Core Algorithm Workstream is [DONE] and archived to `plans/completed/`. Phase 7 verification demonstrated seed → 8,000+ neurons / 32,000+ edges with deterministic growth, continuous adaptation, and polyandric offspring validity (139 suites / 1,709 tests pass). Racing Curriculum Phase 8 is [WIP] — Step 08 [DONE] (Tier 4 start-line stall fixed and user-confirmed). Step 09-12 are [PLANNED] research-only steps for NGE growth stall, pit parity, pit-stop behavior, tire wear rate, Tier 5 activation, and dense network visualization; Step 13 [PLANNED] will synthesize findings into implementation packets. Racing v2 first slice (95→103 pit/strategy channels, all 8 steps green-gated) is [DONE]. Ant hive and predator/prey demo work remain unblocked. The racing curriculum Tier 1–6 ladder is [DONE] (68 suites/502 tests pass).** The proper-NEAT lane, stable activation-ordering lane, architecture-primitives lane, construct-from-parts lane, preconfigured architectures lane, examples and visualization lanes, worker and checkpointing lanes, the Phase 5 memory-foundation stop line, the full current ONNX compliance target, the hybrid-interoperability lane, and the NEATchat follow-up lane are all closed. **The archived ONNX baseline now includes recurrent hardening, the conservative Phase 4 spatial contract, the Phase 5 advanced-graph contract, the Phase 6 optimization contract, the Phase 7 exporter-owned precision contract, the Phase 8 binary contract, and the Phase 9 runtime-parity plus first external-import closure target for the declared lower-opset same-family subset. The archived hybrid-interoperability baseline now carries deterministic parameter vectors, isolation, explicit persistence policy, and the public docs surface needed by downstream consumers. The NEATchat follow-up lane is now archived as done: all six workstreams closed, including the evaluation harness, safety gate, and publishable browser demo.**

- **Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 memory stop line → Phase 6 ONNX → archived hybrid interoperability baseline → archived NEATchat follow-up → Phase 7 / NGE
- **Archived lane A (performance):** [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]
- **Archived lane B (interop):** [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE] — the current ONNX compliance target is closed through the declared Phase 9 stop line, including the binary-first runtime-parity seam and the first named external binary import subset for the approved lower-opset same-family boundary.
- **Archived lane C (hybrid interoperability):** [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md) [DONE] — deterministic parameter vectors, isolated fine-tuning, explicit persistence policy, root-facade re-exports, and generated docs closure.
- **Archived lane D (applied conversational systems):** [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE] — all six workstreams closed; persistent session identity, stronger seed, episodic memory, background adaptation, hybrid routing, attribution-aware regression harness, safety gate, and publishable browser demo.
- **Parallel lane E (quality):** [test-repair-and-coverage.plans.md](completed/test-repair-and-coverage.plans.md) [DONE] — 100% statement/branch/function/line coverage across all of `src/`. 331 suites / 3022 tests green.
- **Standalone meta-workflow lane F:** [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE] — numbered user-invocable agent architecture, hidden specialist delegation, skill-first customization, model routing, validators, evals, and the closed MCP runtime-visibility ownership baseline.
- **Pre-NGE stop line:** closed. NEATchat follow-up lane archived [DONE]; Phase 7 / NGE is now the next frontier.
- **Serial pre-NGE handoff:** after the archived Phase 5 memory stop line, the archived ONNX baseline, the archived hybrid-interoperability baseline, and the archived NEATchat follow-up baseline, the next lane is Phase 7 / NGE.
- **Final capstone:** [completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and its three benchmark demos ([Racing](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md), [Ant Hive](NEAT_Genesis_EvoDevo_AntHive_Demo.md), [Predator/Prey](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md))

## Plan Inventory in Roadmap Order

This is the full `plans/` inventory flattened into execution order so every plan
file has a visible place in the roadmap.

This inventory excludes [README.md](README.md), which is the plans index rather
than a roadmap-tracked plan file.

Completed entries below resolve into `plans/completed/`.

### Standalone meta-workflow inventory

M1. [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE]
M2. [completed/workspace-mcp-registration.plans.md](completed/workspace-mcp-registration.plans.md) [DONE]
M3. [completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md) [DONE]
M3b. [completed/Agent_Dispatch_MCP_Server.plans.md](completed/Agent_Dispatch_MCP_Server.plans.md) [DONE] — implemented `neataptic-dispatch-mcp`, registered in `.mcp.json` / `.vscode/mcp.json`, updated `execute` skill docs; Phase 1 green validation passed and tracker archived.

### Repo Cortex / Semantic Helping inventory (standalone meta-workflow lane)

M4. [completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) [DONE]
M5. [completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md) [DONE]
M6. [completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md) [DONE]
M6b. [completed/Delegation_Tier_Enforcement.plans.md](completed/Delegation_Tier_Enforcement.plans.md) [DONE]
M7. [completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md) [DONE]
M8. [completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md) [DONE]
M8b. [completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md) [DONE]
M8c. [completed/Cortex_RAG_Premium_Primary_Search.plans.md](completed/Cortex_RAG_Premium_Primary_Search.plans.md) [DONE]
M9. [completed/NeatChat_Local_Retrieval_Memory.plans.md](completed/NeatChat_Local_Retrieval_Memory.plans.md) [DONE]
M10. [completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md](completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md) [DONE]
M11. [Step_Packet_Goal_Redesign.plans.md](Step_Packet_Goal_Redesign.plans.md) [WIP]
M12. [completed/turso-rag-migration.plans.md](completed/turso-rag-migration.plans.md) [DONE]
M13. [Remove_Timestamps_From_Permanent_Logs.plans.md](Remove_Timestamps_From_Permanent_Logs.plans.md) [WIP] — remove wall-clock timestamps from agent/skill templates, writer scripts, and persisted artifacts.
M14. [Spec-Kit_Assimilation.plans.md](completed/Spec-Kit_Assimilation.plans.md) [DONE] — cherry-pick Spec Kit governance and workflow patterns (constitution, clarification cap, traceability IDs, task templates, spec checklist, bug triage, research artifacts, extension catalog, verbatim phrases) into the existing agent/gate/skill architecture.

### Phase 0 inventory

1. [Flappy_Bird_Folder_Documentation_Pass.md](completed/Flappy_Bird_Folder_Documentation_Pass.md) [DONE]
2. [architecture-solid-split.plans.md](completed/architecture-solid-split.plans.md) [DONE]
3. [methods-solid-split.plans.md](completed/methods-solid-split.plans.md) [DONE]
4. [methods-docs.plans.md](completed/methods-docs.plans.md) [DONE]
5. [neat-docs.plans.md](completed/neat-docs.plans.md) [DONE]
6. [readme-first-section-pass.plans.md](completed/readme-first-section-pass.plans.md) [DONE]
7. [utils-docs.plans.md](completed/utils-docs.plans.md) [DONE]
8. [neat-test-surface-repair.plans.md](completed/neat-test-surface-repair.plans.md) [DONE]
9. [asciiMaze-typescript-repair.plans.md](completed/asciiMaze-typescript-repair.plans.md) [DONE]
10. [generate-docs-solid-split.plans.md](completed/generate-docs-solid-split.plans.md) [DONE]
11. [render-docs-html-solid-split.plans.md](completed/render-docs-html-solid-split.plans.md) [DONE]
12. [analyze-trace-solid-split.plans.md](completed/analyze-trace-solid-split.plans.md) [DONE]
13. [src-no-explicit-any-cleanup.plans.md](completed/src-no-explicit-any-cleanup.plans.md) [DONE]
14. [ES2023 migration](completed/ES2023%20migration) [DONE]

### Phase 1 inventory

15. [neat.plans.md](completed/neat.plans.md) [DONE]
16. [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md) [DONE]

### Phase 2 inventory

17. [Architecture_Primitives_Node_Group_Layer.md](completed/Architecture_Primitives_Node_Group_Layer.md) [DONE]
18. [Construct_From_Parts_Graph_Assembly.md](completed/Construct_From_Parts_Graph_Assembly.md) [DONE]
19. [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md) [DONE]

### Phase 3 inventory

20. [Browser_Build_and_CDN_Distribution.md](completed/Browser_Build_and_CDN_Distribution.md) [DONE]
21. [Interactive_Examples_and_Learning_Path.plans.md](completed/Interactive_Examples_and_Learning_Path.plans.md) [DONE]
22. [completed/NEATchat.plans.md](completed/NEATchat.plans.md) [DONE]
23. [Network_Visualization_Export_Schema.plans.md](completed/Network_Visualization_Export_Schema.plans.md) [DONE]

### Phase 4 inventory

24. [Standalone_Inference_Export.md](completed/Standalone_Inference_Export.md) [DONE]
25. [completed/Worker_Friendly_Network_Serialization_Fastpath.md](completed/Worker_Friendly_Network_Serialization_Fastpath.md) [DONE]
26. [completed/Turnkey_Multithread_Evaluation_API.md](completed/Turnkey_Multithread_Evaluation_API.md) [DONE]
27. [completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md) [DONE]
28. [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md) [DONE]

### Phase 5 inventory

29. [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]

### Phase 6 inventory

30. [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE]
31. [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE]

### Phase 7 inventory

32. [completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) [DONE]
    32b. [completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md](completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md) (`plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`) [DONE] — prior core readiness audit
    32c. [completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md](completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md) (`plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md`) [DONE] — prior core audit archive
33. [completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) (plans/completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE] — archived canonical racing curriculum readiness plan; Phase 1 [DONE], Phase 2 [DONE], Phase 3 [DONE] (independent per-car NEAT agents; 348 tests pass), Phase 4 [DONE] (4-car coevolution, shared-equal team fitness, 4-car browser rendering, worker-side continuous adaptation; 3 pre-existing race-pack test failures as carry-forward debt), Phase 5 [DONE] (Tier 4 2v2 tires and pits; 45 suites/385 tests pass), Phase 6 [DONE] (Tier 5 3v3 full; 6-car coevolution, role-divergence observables, 46 suites/394 tests pass, 3 skipped polyandric P1/P2; polyandric reproduction DEFERRED), Phase 7 [DONE] (Tier 6 3v3 advanced strategy; FSM 5-bug fix, OpponentSnapshotPool wired, strategy-divergence analytics, analytics-only fallback; 68 suites/502 tests pass; modeIsEvolvable BLOCKED — nge-core-algorithm ownership; carry-forward blockers P1-P5 documented). Phase 8 [DONE] — all 20 steps [DONE] (controller input fix, growth stall research, pit parity, tire wear, Tier 5 auto-promotion, worker-authoritative evolution, defect hardening, network growth blocker fix). Browser smoke confirms N109/C420 -> N523/C1524 at ~60 FPS.
34. [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
35. [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]
36. [completed/racing-curriculum-parallel-variant-gap.plans.md](completed/racing-curriculum-parallel-variant-gap.plans.md) (`plans/completed/racing-curriculum-parallel-variant-gap.plans.md`) [DONE] — close the parallel-variant evaluation gap in the NGE juvenile grow-stabilize cycle; all 5 implementation slices green, 18 suites/435 tests, 100% coverage.
37. [completed/NGE_Core_Growth_Engine_Wiring.plans.md](completed/NGE_Core_Growth_Engine_Wiring.plans.md) (`plans/completed/NGE_Core_Growth_Engine_Wiring.plans.md`) [DONE] — bridges the two disconnected NGE growth systems into a single pipeline; all 5 phases complete. Racing Curriculum unblocked.\n38. [completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md](completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md) (`plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`) [DONE] — optional WebGPU inference fast path for slab-eligible acyclic NEAT networks; transparent CPU fallback; racing-curriculum worker is the first consumer.\n39. [completed/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md](completed/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md) (`plans/completed/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`) [DONE] — NGE WebGPU real-device performance: correct weighted forward kernel, buffer/pipeline caching, batched/parallel inference, and NGE-tier benchmarks past 32k hidden neurons; all phases complete, real-device benchmark 6×8k crossover 15.28× through 2×254k.

### Standalone public library cleanup inventory

- Plan: [completed/Public_Library_Demo_Agnostic_Refactor.plans.md](completed/Public_Library_Demo_Agnostic_Refactor.plans.md) (`plans\\completed\\Public_Library_Demo_Agnostic_Refactor.plans.md`) [DONE] — all 6 phases complete; `src/` public library is demo-agnostic, migration note added to `RELEASE.md`.
- Archive: [completed/Public_Library_Demo_Agnostic_Refactor.logs.md](completed/Public_Library_Demo_Agnostic_Refactor.logs.md) (`plans\\completed\\Public_Library_Demo_Agnostic_Refactor.logs.md`) [DONE] — full done-state record.
