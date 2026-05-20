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
  - Plan: [src-no-explicit-any-cleanup.plans.md](completed/src-no-explicit-any-cleanup.plans.md)
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
- Follow-up lane: [NEATchat.plans.md](NEATchat.plans.md) [PLANNED]. The new lane owns the post-toy conversational-system push, but it remains planning-only until checkpointing, worker payloads, multithread evaluation, parameter-vector contracts, and recurrent ONNX hardening expose usable public seams. It should build on those foundations instead of widening the closed Phase 3 acceptance surface retroactively.

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

- Plan: [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) [WIP]

**Why this ordering:**

- Standalone export and worker payloads share an “inference IR” concept; building that once reduces duplication.
- Worker transport should prove the substrate and the Flappy proof-of-concept first; the turnkey API should then extract the example-owned ergonomics into reusable public helpers.
- Checkpointing and hybrid evaluation benefit from deterministic scheduling and a clear parameter/vector mapping.
- Evolution-training parameter vectors are a later unification seam, not a blocker for standalone export or worker payloads unless that contract is deliberately split into an earlier mini-phase.

Next critical-path frontier:

- [completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md) now records the closed Phase 4 checkpointing baseline and the reopen point for future persistence work.
- The archived pre-NGE ONNX baseline now lives at [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md), and the next non-chat foundation handoff moves to [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) while [NEATchat.plans.md](NEATchat.plans.md) remains dependency-gated behind those two stop lines.

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
  - Plan: [NEATchat.plans.md](NEATchat.plans.md) [PLANNED]
  - Current internal state: the Phase 3 NEATchat demo remains closed as a toy-scale learnability baseline, while the follow-up lane is a dependency-gated later plan for stronger pretrained seeds, checkpointed identity, retrieval-like memory, worker-backed background adaptation, and hybrid candidate routing. It should stay planning-only until checkpointing, worker payloads, multithread evaluation, parameter-vector contracts, and recurrent ONNX hardening are usable enough to support an honest external-seed target.

**Recommended timing:**

- Treat the archived ONNX baseline as the reopen point for future interoperability widening rather than keeping the old root tracker active.
- Keep future ONNX scope honest: start from a narrower new amendment when runtime, external-import, or custom-domain claims need to change instead of reopening the archived baseline implicitly.
- Treat the NEATchat follow-up lane as an applied consumer of Phase 4 plus Phase 6 work, not as a shortcut around those foundations.

## Phase 7 — Advanced Research Features (Last)

**Outcome:** evo-devo / NGE capabilities and benchmark-driven validation that build on top of all prior infrastructure.

- NEAT Genesis EvoDevo (NGE) — core algorithm (computation motifs, lifecycle, DNA, reproduction, collective intelligence)
  - Plan: [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) [PLANNED]
- NGE Racing Curriculum — single-agent benchmark (sensory specialization, neuromodulation, lifecycle staging)
  - Plan: [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) [PLANNED]
- NGE Ant Hive Ecosystem — multi-agent benchmark (stigmergy, role differentiation, collective intelligence)
  - Plan: [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
- NGE Predator/Prey Co-evolution — co-evolutionary benchmark (sensory arms race, reproduction modes, non-stationary fitness)
  - Plan: [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]

**Why last:** this work depends heavily on the Memory Optimization track (Track 2 in that plan) and benefits from stable NEAT correctness, deterministic activation semantics, robust serialization/checkpointing, and a mature enough NGE core that benchmark results reflect the algorithm rather than unstable infrastructure.

## Summary: Critical Path vs Parallel Lanes

Current status: **Phases 0, 1, 2, 3, and 4 are complete for the current roadmap scope, and the pre-NGE memory foundation stop line is archived as done through Track 1 / Phase 10.** The proper-NEAT lane, stable activation-ordering lane, architecture-primitives lane, construct-from-parts lane, preconfigured architectures lane, examples and visualization lanes, worker and checkpointing lanes, the Phase 5 memory-foundation stop line, and the full current ONNX compliance target are closed. **The archived ONNX baseline now includes recurrent hardening, the conservative Phase 4 spatial contract, the Phase 5 advanced-graph contract, the Phase 6 optimization contract, the Phase 7 exporter-owned precision contract, the Phase 8 binary contract, and the Phase 9 runtime-parity plus first external-import closure target for the declared lower-opset same-family subset. Hybrid interoperability is now the next non-chat foundation handoff, and the NEATchat follow-up stays dependency-gated behind that lane rather than opening directly from the archived ONNX baseline.**

- **Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 memory stop line → Phase 6 ONNX → hybrid interoperability → gated NEATchat follow-up → Phase 7 / NGE
- **Archived lane A (performance):** [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]
- **Archived lane B (interop):** [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE] — the current ONNX compliance target is closed through the declared Phase 9 stop line, including the binary-first runtime-parity seam and the first named external binary import subset for the approved lower-opset same-family boundary.
- **Active lane C (hybrid interoperability):** [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) [WIP]
- **Planned lane D (applied conversational systems):** [NEATchat.plans.md](NEATchat.plans.md) [PLANNED]
- **Parallel lane E (quality):** [test-repair-and-coverage.plans.md](completed/test-repair-and-coverage.plans.md) [DONE] — 100% statement/branch/function/line coverage across all of `src/`. 331 suites / 3022 tests green.
- **Standalone meta-workflow lane F:** [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE] — seven-phase user-invocable agent architecture, hidden specialist delegation, skill-first customization, model routing, validators, evals, and the closed MCP runtime-visibility ownership baseline.
- **Pre-NGE stop line:** not yet closed
- **Serial pre-NGE handoff:** after the archived Phase 5 memory stop line and the archived ONNX baseline, move to [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md), then the still-gated [NEATchat.plans.md](NEATchat.plans.md), and only then open Phase 7 / NGE work
- **Final capstone:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and its three benchmark demos ([Racing](NEAT_Genesis_EvoDevo_Racing_Curriculum.md), [Ant Hive](NEAT_Genesis_EvoDevo_AntHive_Demo.md), [Predator/Prey](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md))

## Plan Inventory in Roadmap Order

This is the full `plans/` inventory flattened into execution order so every plan
file has a visible place in the roadmap.

This inventory excludes [README.md](README.md), which is the plans index rather
than a roadmap-tracked plan file.

Completed entries below resolve into `plans/completed/`.

### Standalone meta-workflow inventory

M1. [completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md) [DONE]

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
28. [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) [WIP]

### Phase 5 inventory

29. [completed/Memory_Optimization.md](completed/Memory_Optimization.md) [DONE]

### Phase 6 inventory

30. [completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md) [DONE]
31. [NEATchat.plans.md](NEATchat.plans.md) [PLANNED]

### Phase 7 inventory

32. [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) [PLANNED]
33. [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) [PLANNED]
34. [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
35. [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]
