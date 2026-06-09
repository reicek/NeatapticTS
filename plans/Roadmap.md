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
script + mandatory checklist entries in `.github/copilot-instructions.md` and `CLAUDE.md` for the
tooling half. Racing Curriculum Phase 3 (Tier 3: 2v2 Roles) is not part of this lane.

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
  - All six migration phases (A-F) complete: 30 named flows, 12 gates, 3 MCP servers, 61/61 gate tests green, all 9 acceptance criteria satisfied. Archived 2026-05-22.

**Coordination rule:** this lane is workflow-only infrastructure. Do not modify
`src/` while executing it, and treat
[completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md)
as the archived baseline for existing agent, skill, model-routing, validation,
and MCP ownership contracts.

## Standalone Meta-Workflow Lane — Orchestration System Optimization [PLANNED]

**Outcome:** enforce strict Tier 1→Tier 2/3/4 delegation by extracting durable policies into skills, creating targeted specialist agents, and updating flows to eliminate "God-agent" behavior from the eight numbered SDLC orchestrators.

- Orchestration system optimization (mini-agent transition)
  - Plan: [Orchestration_System_Optimization.plans.md](Orchestration_System_Optimization.plans.md) [WIP]
  - Current internal state: Phase 4 Step 03 [DONE] — All gates PASS (agent-graph: 61 agents/0 issues, tier-enforcement: 0 violations, plan-sync: 0 errors/0 warnings). Flow specialist references updated in 7 flow files. Step 04-07 remaining (flow selection red tests, tier-enforcement with flow awareness, docs, final validation).

**Coordination rule:** this lane is confined to `.github/agents/`, `.github/skills/`, `.github/flows/`, `scripts/agent-customization/`, and tracker/log files. Do not modify `src/` or MCP server implementations. Treat [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) and [completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md) as the archived baselines for agent architecture and flow/gates contracts.

## Standalone Meta-Workflow Lane — Step Packet Goal Redesign [WIP]

**Outcome:** replace the step packet YAML `agent` and `agent_file` fields with a `goal` field that declares what outcome a step needs rather than who does it, add an optional `tdd_sequence` field for multi-phase dispatch decomposition, update the orchestrator routing table in `copilot-instructions.md` §3, migrate all existing plan files, and remove backward compatibility.

- Step Packet Goal Redesign
  - Plan: [Step_Packet_Goal_Redesign.plans.md](Step_Packet_Goal_Redesign.plans.md) [WIP]
  - Current internal state: Phase 1 Step 01 [WIP] — planning the redesign. The step-packet gate currently requires `agent`; Phase 1 Step 02 must update it to accept `goal` before Phase 2 migration begins.

**Coordination rule:** this lane is confined to `.github/copilot-instructions.md`, `.github/skills/phase-handoff-workflow/SKILL.md`, `scripts/agent-customization/gates/step-packet.gate.mjs`, and plan file step packet YAML. Do not modify `src/` library code, flow YAML files (`.github/flows/*.flow.yml` use `agent:` for flow ownership, a different concern), or runtime enforcement scripts.

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
(`scripts/semantic-index/**`, `scripts/agent-customization/mcp/**`, gate scripts, package scripts,
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
- Follow-up lane: [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE]. The post-toy conversational-systems lane closed 2026-05-22. All six workstreams closed: W1 durable substrate, W2 stronger seed import, W3 episodic memory and retrieval, W4 background adaptation and candidate search, W5 hybrid routing, W6 evaluation harness, safety gate, and publishable product shape. Unrelated broad-run heap OOM remains outside NEATchat ownership.

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
- The archived pre-NGE ONNX baseline now lives at [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md), and the archived hybrid baseline now lives at [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md). The NEATchat follow-up is now archived at [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE] with all six workstreams closed 2026-05-22. The pre-NGE stop line is now satisfied; Phase 7 / NGE is the next frontier.

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
  - Current internal state: all six workstreams closed 2026-05-22. W1 snapshot v2 and durable substrate; W2 stronger default seed and honest external-seed contract; W3 episodic memory bank and token-overlap retrieval; W4 async-deferred background adaptation and explicit candidate lifecycle; W5 three-path hybrid routing with observability-only routingLog; W6 attribution-aware regression harness, structured safety gate, and publishable browser demo with explicit live/experimental/background-job lane labels.
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
- Artifacts: `scripts/semantic-index/`, `data/semantic-index.sqlite`

2. MCP tools (search, load, freshness, stats)

- Plan: [completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md) [DONE]
- Gate: Layer 1 [DONE] satisfied
- Artifacts: `scripts/mcp-semantic/`, `neataptic-cortex-mcp` in `.vscode/mcp.json`

3. Browser snapshot and IndexedDB loader (all demos)

- Plan: [completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md) [DONE]
- Gate: Layers 1 and 2 [DONE] satisfied; Layer 3 archived
- Artifacts: `docs/assets/semantic-snapshot.json`, `examples/shared/semantic/`

4. Cortex MCP reliability hardening (Layer 4 — agents, skill, lifecycle gate, MCP enhancements)

- Plan: [completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md) [DONE]
- Gate: Layers 1, 2, and 3 [DONE] required; Delegation Tier Enforcement must be [DONE]; `neataptic-workflow-mcp.mjs` must be active
- Artifacts: `.github/agents/repo-cortex-scout.agent.md`, `.github/agents/cortex-embeddings-scout.agent.md`, `.github/skills/repo-cortex-workflow/SKILL.md`, `scripts/agent-customization/gates/cortex-index.gate.mjs`, `scripts/agent-customization/plan-session-redirect.mjs`, `scripts/agent-customization/validate-tsconfig-docs.mjs`

5. ONNX embeddings + hybrid BM25+dense ranking (Layer 5)

- Plan: [completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md) [DONE]
- Gate: Layers 1 and 2 [DONE] required
- Artifacts: `scripts/semantic-index/embed-index.mjs`, `data/embeddings.sqlite`, ONNX model cache

6. Embedding prewarm + default-on dense contract (Layer 6)

- Plan: [completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md) [DONE]
- Gate: Layer 5 [DONE] satisfied; Layer 6 archived after final prewarm, readiness, gate, MCP degradation, and tracker-closure validation
- Soft dependency: Layer 4 [DONE] improved MCP lifecycle management but was not required to close Layer 6
- Artifacts: `scripts/semantic-index/prewarm-dense.mjs`, `scripts/semantic-index/dense-readiness.mjs`, `scripts/agent-customization/gates/dense-readiness.gate.mjs`; MCP `search_corpus` now defaults `use_dense: true` with graceful cold-state degradation and warm-state `dense_state` provenance

7. Advanced RAG architecture (Layer 7+)

- Plan: [Repo_Cortex_Advanced_RAG_Architecture.plans.md](Repo_Cortex_Advanced_RAG_Architecture.plans.md) [WIP]
- Gate: Layers 1–6 [DONE] satisfied; builds on the existing BM25+dense hybrid
- Artifacts: semantic chunking, query classification, cross-encoder re-ranking, context window assembly, entity/relationship graphs, query expansion, relevance feedback, structured metadata filtering, multi-hop retrieval, ANN indexing, RAG eval suite

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
  - May reuse chunking and BM25 patterns from Layers 1–3 but must not import from `scripts/semantic-index/`
  - Artifacts: `examples/neatChat/memory/` (types, DB adapters, retrieval, services, tests)

**Coordination rules:**

- `completed/Delegation_Tier_Enforcement.plans.md` is the archived Agentic Workflow Enforcement Prerequisite baseline, not a Repo Cortex corpus layer. It has no "Layer N" designation.
- `data/semantic-index.sqlite` and `data/embeddings.sqlite` are gitignored generated artifacts.
- `docs/assets/semantic-snapshot.json` is a generated artifact; treat it as read-only.
- NeatChat memory (`examples/neatChat/memory/`) must never import from `scripts/semantic-index/`.

## Phase 7 — Advanced Research Features (Last)

**Outcome:** evo-devo / NGE capabilities and benchmark-driven validation that build on top of all prior infrastructure.

- NEAT Genesis EvoDevo (NGE) — core algorithm (computation motifs, lifecycle, DNA, reproduction, collective intelligence)
  - Plan: [completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) [DONE]
- NGE Core Readiness Audit — primitive-by-primitive readiness matrix, gap classification by owner boundary, and first implementation tranche selection (core-first; demos are downstream e2e tests)
  - Plan: [NEAT_Genesis_EvoDevo_Core_Readiness.plans.md](NEAT_Genesis_EvoDevo_Core_Readiness.plans.md) (`plans\NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`) [PLANNED] — Phase 3 [DONE] (independent populations + generation barriers); Phase 4 [PLANNED] (deterministic evaluation packs); status changed to PLANNED per RAG architecture priority shift
- NGE Racing Curriculum — Team A/B benchmark (worker-streamed runtime authority, deterministic race packs, rolling opponent snapshots)
  - Plan: [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) (`plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`) [WIP]
- Racing Path-Tracking Debug and Quality Followup — pre-Phase-3 visual fix, geometry audit, and deferred quality cleanup
  - Plan: [completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md](completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md) [DONE]
  - Current internal state: archived after the shared-spline path-tracking repair, the
    user-confirmed rounded-lane visual pass, and the bounded folder-quality cleanup. The only
    remaining caveat is accepted static debt: `examples/racing_curriculum/browser-entry/browser-entry.ts`
    still lacks a sibling `browser-entry.test.ts`.
- NGE Ant Hive Ecosystem — multi-agent benchmark (stigmergy, role differentiation, collective intelligence)
  - Plan: [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
- NGE Predator/Prey Co-evolution — co-evolutionary benchmark (sensory arms race, reproduction modes, non-stationary fitness)
  - Plan: [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]

**Why last:** this work depends heavily on the Memory Optimization track (Track 2 in that plan) and benefits from stable NEAT correctness, deterministic activation semantics, robust serialization/checkpointing, and a mature enough NGE core that benchmark results reflect the algorithm rather than unstable infrastructure.

## Summary: Critical Path vs Parallel Lanes

Current status: **Phases 0, 1, 2, 3, and 4 are complete for the current roadmap scope, and the pre-NGE memory foundation stop line is archived as done through Track 1 / Phase 10.** The proper-NEAT lane, stable activation-ordering lane, architecture-primitives lane, construct-from-parts lane, preconfigured architectures lane, examples and visualization lanes, worker and checkpointing lanes, the Phase 5 memory-foundation stop line, the full current ONNX compliance target, the hybrid-interoperability lane, and the NEATchat follow-up lane are all closed. **The archived ONNX baseline now includes recurrent hardening, the conservative Phase 4 spatial contract, the Phase 5 advanced-graph contract, the Phase 6 optimization contract, the Phase 7 exporter-owned precision contract, the Phase 8 binary contract, and the Phase 9 runtime-parity plus first external-import closure target for the declared lower-opset same-family subset. The archived hybrid-interoperability baseline now carries deterministic parameter vectors, isolation, explicit persistence policy, and the public docs surface needed by downstream consumers. The NEATchat follow-up lane is now archived as done: all six workstreams closed 2026-05-22, including the evaluation harness, safety gate, and publishable browser demo.**

- **Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 memory stop line → Phase 6 ONNX → archived hybrid interoperability baseline → archived NEATchat follow-up → Phase 7 / NGE
- **Archived lane A (performance):** [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]
- **Archived lane B (interop):** [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE] — the current ONNX compliance target is closed through the declared Phase 9 stop line, including the binary-first runtime-parity seam and the first named external binary import subset for the approved lower-opset same-family boundary.
- **Archived lane C (hybrid interoperability):** [completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md) [DONE] — deterministic parameter vectors, isolated fine-tuning, explicit persistence policy, root-facade re-exports, and generated docs closure.
- **Archived lane D (applied conversational systems):** [completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md) [DONE] — all six workstreams closed 2026-05-22; persistent session identity, stronger seed, episodic memory, background adaptation, hybrid routing, attribution-aware regression harness, safety gate, and publishable browser demo.
- **Parallel lane E (quality):** [test-repair-and-coverage.plans.md](completed/test-repair-and-coverage.plans.md) [DONE] — 100% statement/branch/function/line coverage across all of `src/`. 331 suites / 3022 tests green.
- **Standalone meta-workflow lane F:** [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE] — numbered user-invocable agent architecture, hidden specialist delegation, skill-first customization, model routing, validators, evals, and the closed MCP runtime-visibility ownership baseline.
- **Pre-NGE stop line:** closed. NEATchat follow-up lane archived [DONE]; Phase 7 / NGE is now the next frontier.
- **Serial pre-NGE handoff:** after the archived Phase 5 memory stop line, the archived ONNX baseline, the archived hybrid-interoperability baseline, and the archived NEATchat follow-up baseline, the next lane is Phase 7 / NGE.
- **Final capstone:** [completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and its three benchmark demos ([Racing](NEAT_Genesis_EvoDevo_Racing_Curriculum.md), [Ant Hive](NEAT_Genesis_EvoDevo_AntHive_Demo.md), [Predator/Prey](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md))

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

### Repo Cortex / Semantic Helping inventory (standalone meta-workflow lane)

M4. [completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) [DONE]
M5. [completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md) [DONE]
M6. [completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md) [DONE]
M6b. [completed/Delegation_Tier_Enforcement.plans.md](completed/Delegation_Tier_Enforcement.plans.md) [DONE]
M7. [completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md) [DONE]
M8. [completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md) [DONE]
M8b. [completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md) [DONE]
M9. [completed/NeatChat_Local_Retrieval_Memory.plans.md](completed/NeatChat_Local_Retrieval_Memory.plans.md) [DONE]
M10. [completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md](completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md) [DONE]
M11. [Step_Packet_Goal_Redesign.plans.md](Step_Packet_Goal_Redesign.plans.md) [WIP]

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
    32b. [NEAT_Genesis_EvoDevo_Core_Readiness.plans.md](NEAT_Genesis_EvoDevo_Core_Readiness.plans.md) (`plans\NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`) [WIP]
33. [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) (`plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`) [WIP]
34. [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
35. [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]
