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
  - Scope note: ESM package wiring, the ES2023 TypeScript baseline, the lint scaffold, named-error rollout, targeted syntax cleanup, shared clone-helper normalization, narrow workflow or benchmark edge cleanup, and CI enforcement are now complete. Memory-management or performance-feature work remains owned by [Memory_Optimization.md](Memory_Optimization.md).

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
   - Plan: [Browser_Build_and_CDN_Distribution.md](Browser_Build_and_CDN_Distribution.md) [PLANNED]
7. Interactive examples + learning path (Node + browser runnable examples, CI smoke checks)
   - Plan: [Interactive_Examples_and_Learning_Path.md](Interactive_Examples_and_Learning_Path.md) [DONE]
8. NEATchat (tiny online sequence-learning chatbot demo)
  - Plan: [completed/NEATchat.plans.md](completed/NEATchat.plans.md) [DONE]
  - Current internal state: the NEATchat Phase 3 learnability lane is closed for current scope and archived as a completed baseline. The delivered surface remains intentionally toy-scale: near-zero start, short online adaptation, language-agnostic token-stream behavior, and browser or Node experimentation boundaries rather than web-scale training.
9. Visualization export schema (JSON schema + optional DOT output)
   - Plan: [Network_Visualization_Export_Schema.md](Network_Visualization_Export_Schema.md) [WIP]

**Notes:**

- Phase 3 examples are the starter set only: Node hello/evolve flows plus one minimal browser quickstart once the browser bundle exists.
- `NEATchat` is part of the learnability lane because it showcases the sequence builders and online adaptation at a toy scale; persistent memory files, worker-backed background training, or any heavier deployment surface remain follow-on work for later phases.
- Examples that depend on standalone export or worker execution are follow-on additions in Phase 4 after those capabilities land.
- Before expanding the examples catalog, choose the canonical examples home and decide whether `bench-browser/` is the browser-example host so demo work does not fragment.
- Visualization can be implemented slightly earlier, but it becomes much more valuable once primitives/builders provide stable labels/roles.

## Phase 4 — Deployment + Parallel Evaluation (Inference Artifacts, Workers, Checkpoints)

**Outcome:** production and scale workflows: export models, evaluate quickly, resume long runs.

9. Standalone inference export (dependency-free runtime output)
   - Plan: [Standalone_Inference_Export.md](completed/Standalone_Inference_Export.md) [DONE]
   - Current internal state: the planning baseline for the standalone deployment lane is closed. The legacy `network.standalone()` generator now honors compiled activation traversal and explicit I/O role ordering, and the next implementation reopen point is to create `src/architecture/network/export/` and freeze `InferenceIRv1` when Phase 4 becomes active.

10. Worker-friendly serialization fastpath (clone/transfer payloads; predictor creation)

- Plan: [Worker_Friendly_Network_Serialization_Fastpath.md](Worker_Friendly_Network_Serialization_Fastpath.md) [PLANNED]
- Current internal state: planning complete — four transport strategies defined (`PortableInferencePayload`, `TransferableInferencePayload`, `InferenceChannel`, `SharedInferenceWorker`). No implementation has started. Phase 0 (Shared Inference IR) is the first implementation step when this plan becomes active.

11. Turnkey multithread evaluation API (Node + browser workers)

- Plan: [Turnkey_Multithread_Evaluation_API.md](Turnkey_Multithread_Evaluation_API.md) [PLANNED]

12. Population save/resume + checkpointing (full vs light checkpoints, determinism contracts)

- Plan: [Population_Save_Resume_and_Checkpointing.md](Population_Save_Resume_and_Checkpointing.md) [PLANNED]

13. Evolution–training interoperability contracts (parameter vectors, isolation, hybrid policies)

- Plan: [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) [PLANNED]

**Why this ordering:**

- Standalone export and worker payloads share an “inference IR” concept; building that once reduces duplication.
- Multithread evaluation becomes straightforward after payloads exist.
- Checkpointing and hybrid evaluation benefit from deterministic scheduling and a clear parameter/vector mapping.
- Evolution-training parameter vectors are a later unification seam, not a blocker for standalone export or worker payloads unless that contract is deliberately split into an earlier mini-phase.

## Phase 5 — Scale & Performance (Memory Optimization Track)

**Outcome:** handle very large graphs and long runs on commodity hardware, across Node + browser.

This plan is large and can run as a **parallel lane** after Phase 1, but it should not destabilize correctness work.

- Memory & performance multi-layer strategy (Track 1: Phases 0–10; Track 2 gates Hyper work)
  - Plan: [Memory_Optimization.md](Memory_Optimization.md) [WIP]
  - Current internal state: Phases 0-3 are done, Phase 4 is next, and Track 2 Hyper work remains gated behind Track 1 stability.

**Recommended sequencing guidance:**

- Start / continue Track 1 after Phase 1 is stable.
- Prioritize improvements that directly benefit the worker payload + inference export paths (typed arrays, slabs, reuse) so Phase 4 gets faster “for free”.
- If a detailed memory-plan subsection implies Hyper work can begin immediately after an intermediate Track 1 checkpoint, treat that as stale wording; Hyper remains gated by the Track 1 conditions below.

**Gate to Phase 7 (Hyper):** Track 1 gates in `Memory_Optimization.md` are met (especially Phase 4–7 stability + variance/hardening).

## Phase 6 — Interoperability Breadth (ONNX)

**Outcome:** broader ecosystem compatibility and model portability.

- ONNX export/import breadth and hardening
  - Plan: [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md) [WIP]
  - Current internal state: Phase 0-2 are complete, recurrent groundwork is implemented and still being hardened, and convolutional/spatial groundwork is in progress.

**Recommended timing:**

- Continue ONNX work in parallel after Phase 2 (builders) so we have a clean way to produce supported architectures.
- Keep ONNX scope honest: focus on deterministic export/import behavior and clear “supported subset” docs.

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

Current status: **Phase 0 and Phase 1 are complete** and **Phase 2 is now the current roadmap stage**. The proper-NEAT lane in [neat.plans.md](completed/neat.plans.md) and the stable activation-ordering lane in [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md) are closed for their current scope. Both demos are solid split and documented, the example learnability pass materially strengthened across `examples`, the Flappy Bird documentation pass is closed, the main app is already solid split, the architecture split follow-through is closed, the broader educational documentation lane is closed including the repo-wide README first-section pass, the `src/` strict-typing cleanup is closed as a completed baseline, the ES2023 modernization lane is closed as a completed Phase 0 baseline, the architecture-primitives lane is closed as the first completed Phase 2 baseline, construct-from-parts is closed as the second completed Phase 2 baseline in [completed/Construct_From_Parts_Graph_Assembly.md](completed/Construct_From_Parts_Graph_Assembly.md), and preconfigured architectures (MLP, RandomSparse, NARX, GRU, LSTM) are now closed as the third and final completed Phase 2 baseline in [completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md). Phase 2 is now complete. The standalone-export planning baseline is now [DONE], which records the completed parity groundwork and the frozen Phase 4 implementation reopen point without changing the active roadmap priority away from Phase 3.

- **Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
- **Parallel lane A (performance):** [Memory_Optimization.md](Memory_Optimization.md) Track 1 after Phase 1 stabilizes
- **Parallel lane B (interop):** [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md) after Phase 2 (or earlier if scoped tightly)
- **Parallel lane C (quality):** [test-repair-and-coverage.plans.md](completed/test-repair-and-coverage.plans.md) [DONE] — 100% statement/branch/function/line coverage across all of `src/`. 331 suites / 3022 tests green.
- **Final capstone:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and its three benchmark demos ([Racing](NEAT_Genesis_EvoDevo_Racing_Curriculum.md), [Ant Hive](NEAT_Genesis_EvoDevo_AntHive_Demo.md), [Predator/Prey](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md))

## Plan Inventory in Roadmap Order

This is the full `plans/` inventory flattened into execution order so every plan
file has a visible place in the roadmap.

This inventory excludes [README.md](README.md), which is the plans index rather
than a roadmap-tracked plan file.

Completed entries below resolve into `plans/completed/`.

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

20. [Browser_Build_and_CDN_Distribution.md](Browser_Build_and_CDN_Distribution.md) [PLANNED]
21. [Interactive_Examples_and_Learning_Path.md](Interactive_Examples_and_Learning_Path.md) [WIP]
22. [completed/NEATchat.plans.md](completed/NEATchat.plans.md) [DONE]
23. [Network_Visualization_Export_Schema.md](Network_Visualization_Export_Schema.md) [WIP]

### Phase 4 inventory

24. [Standalone_Inference_Export.md](completed/Standalone_Inference_Export.md) [DONE]
25. [Worker_Friendly_Network_Serialization_Fastpath.md](Worker_Friendly_Network_Serialization_Fastpath.md) [PLANNED]
26. [Turnkey_Multithread_Evaluation_API.md](Turnkey_Multithread_Evaluation_API.md) [PLANNED]
27. [Population_Save_Resume_and_Checkpointing.md](Population_Save_Resume_and_Checkpointing.md) [PLANNED]
28. [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md) [PLANNED]

### Phase 5 inventory

29. [Memory_Optimization.md](Memory_Optimization.md) [WIP]

### Phase 6 inventory

30. [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md) [WIP]

### Phase 7 inventory

31. [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) [PLANNED]
32. [NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) [PLANNED]
33. [NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
34. [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]
