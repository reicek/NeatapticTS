# Roadmap (Order of Execution)

This roadmap orders every plan in `/plans` into a dependency-aware sequence optimized for **library growth**:

- **Correctness first** (trust + reproducibility)
- **Ergonomics next** (architecture building, deterministic execution)
- **Adoption next** (browser distribution + examples)
- **Scale + deployment** (standalone inference, workers, checkpointing)
- **Interop breadth** (ONNX)
- **Advanced research** last (HyperEvoDevoMorphoNEAT)

Where it helps, this roadmap uses **lanes** (things that can proceed in parallel) and **gates** (things that should be true before moving on).

## Phase 0 — Hygiene + Refactor Baseline

**Outcome:** keep iteration speed high, reduce refactor risk, and finish the structural cleanup needed before broad mechanical modernization.

- Demo refinement and learnability hardening [DONE]
  - Current focus: stabilize and refine the `flappy_bird` demo so it remains the reference quality bar for later example work.
- asciiMaze SOLID split before ES2023 modernization
  - Finish a maintainable split of `test/examples/asciiMaze` and any touched orchestration surfaces under `src/` so responsibilities are narrow, substitutable, and DRY.
  - The target shape is the stronger modular style already emerging in `test/examples/flappy_bird`: explicit boundaries, LSP-safe abstractions, and smaller units that make caching and later performance work easier to target precisely.
- ES2023 modernization (after the demo-structure pass; mechanical refactors + CI enforcement)
  - Plan: [ES2023 migration](ES2023%20migration)
  - Scope note: this phase is syntax/module modernization plus CI enforcement. Memory-management or performance-feature work remains owned by [Memory_Optimization.md](Memory_Optimization.md).

**Gate to Phase 1:** `flappy_bird` refinement is stable, the `asciiMaze` SOLID split is complete, and `npx tsc --noEmit -p tsconfig.json` plus `npm test` are green after the refactor pass.

## Phase 1 — Core Correctness + Determinism Foundations (Critical Path)

**Outcome:** deterministic semantics and “proper NEAT” correctness so everything built on top is reliable.

1. Proper NEAT (historical markings, correct crossover alignment, recurrent/self-connection policy, RNG determinism)
   - Plan: [neat.plans.md](neat.plans.md)
2. Explicit I/O roles + stable activation ordering (acyclic + recurrent mode semantics)
   - Plan: [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.md)

**Why this ordering:**

- Fixing NEAT correctness prevents “paper cuts” and broken invariants from leaking into every future feature.
- Stable execution semantics (I/O + scheduling) are required for safe builders, serialization, export, and workers.

**Gate to Phase 2:** deterministic runs are achievable end-to-end (seeded runs don’t “leak” randomness; scheduling is stable; recurrent behavior is documented and tested).

## Phase 2 — Architecture DX (Build Graphs Safely)

**Outcome:** a clear, typed, deterministic “build” story for users and for internal features.

3. Architecture primitives (Node/Group/Layer)
   - Plan: [Architecture_Primitives_Node_Group_Layer.md](Architecture_Primitives_Node_Group_Layer.md)
4. Deterministic construct-from-parts (validation + scheduling integration)
   - Plan: [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md)
5. Preconfigured architectures (MLP + sequence builders)
   - Plan: [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md)

**Gate to Phase 3:** builders use explicit I/O roles, and construction is deterministic with actionable diagnostics.

## Phase 3 — Adoption + Learnability (Browser + Examples + Visualization)

**Outcome:** reduce friction for new users and make the library “tryable” immediately.

6. Browser build + CDN distribution (ESM + IIFE bundles; stable public surface)
   - Plan: [Browser_Build_and_CDN_Distribution.md](Browser_Build_and_CDN_Distribution.md)
7. Interactive examples + learning path (Node + browser runnable examples, CI smoke checks)
   - Plan: [Interactive_Examples_and_Learning_Path.md](Interactive_Examples_and_Learning_Path.md)
8. Visualization export schema (JSON schema + optional DOT output)
   - Plan: [Network_Visualization_Export_Schema.md](Network_Visualization_Export_Schema.md)

**Notes:**

- Phase 3 examples are the starter set only: Node hello/evolve flows plus one minimal browser quickstart once the browser bundle exists.
- Examples that depend on standalone export or worker execution are follow-on additions in Phase 4 after those capabilities land.
- Before expanding the examples catalog, choose the canonical examples home and decide whether `bench-browser/` is the browser-example host so demo work does not fragment.
- Visualization can be implemented slightly earlier, but it becomes much more valuable once primitives/builders provide stable labels/roles.

## Phase 4 — Deployment + Parallel Evaluation (Inference Artifacts, Workers, Checkpoints)

**Outcome:** production and scale workflows: export models, evaluate quickly, resume long runs.

9. Standalone inference export (dependency-free runtime output)
   - Plan: [Standalone_Inference_Export.md](Standalone_Inference_Export.md)
10. Worker-friendly serialization fastpath (clone/transfer payloads; predictor creation)
    - Plan: [Worker_Friendly_Network_Serialization_Fastpath.md](Worker_Friendly_Network_Serialization_Fastpath.md)
11. Turnkey multithread evaluation API (Node + browser workers)
    - Plan: [Turnkey_Multithread_Evaluation_API.md](Turnkey_Multithread_Evaluation_API.md)
12. Population save/resume + checkpointing (full vs light checkpoints, determinism contracts)
    - Plan: [Population_Save_Resume_and_Checkpointing.md](Population_Save_Resume_and_Checkpointing.md)
13. Evolution–training interoperability contracts (parameter vectors, isolation, hybrid policies)
    - Plan: [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md)

**Why this ordering:**

- Standalone export and worker payloads share an “inference IR” concept; building that once reduces duplication.
- Multithread evaluation becomes straightforward after payloads exist.
- Checkpointing and hybrid evaluation benefit from deterministic scheduling and a clear parameter/vector mapping.
- Evolution-training parameter vectors are a later unification seam, not a blocker for standalone export or worker payloads unless that contract is deliberately split into an earlier mini-phase.

## Phase 5 — Scale & Performance (Memory Optimization Track)

**Outcome:** handle very large graphs and long runs on commodity hardware, across Node + browser.

This plan is large and can run as a **parallel lane** after Phase 1, but it should not destabilize correctness work.

- Memory & performance multi-layer strategy (Track 1: Phases 0–10; Track 2 gates Hyper work)
  - Plan: [Memory_Optimization.md](Memory_Optimization.md)

**Recommended sequencing guidance:**

- Start / continue Track 1 after Phase 1 is stable.
- Prioritize improvements that directly benefit the worker payload + inference export paths (typed arrays, slabs, reuse) so Phase 4 gets faster “for free”.
- If a detailed memory-plan subsection implies Hyper work can begin immediately after an intermediate Track 1 checkpoint, treat that as stale wording; Hyper remains gated by the Track 1 conditions below.

**Gate to Phase 7 (Hyper):** Track 1 gates in `Memory_Optimization.md` are met (especially Phase 4–7 stability + variance/hardening).

## Phase 6 — Interoperability Breadth (ONNX)

**Outcome:** broader ecosystem compatibility and model portability.

- ONNX export/import breadth and hardening
  - Plan: [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md)

**Recommended timing:**

- Continue ONNX work in parallel after Phase 2 (builders) so we have a clean way to produce supported architectures.
- Keep ONNX scope honest: focus on deterministic export/import behavior and clear “supported subset” docs.

## Phase 7 — Advanced Research Features (Last)

**Outcome:** evo-devo / hyper-scale capabilities that build on top of all prior infrastructure.

- HyperEvoDevo MorphoNEAT
  - Plan: [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md)

**Why last:** this work depends heavily on the Memory Optimization track (Track 2 in that plan) and benefits from stable NEAT correctness, deterministic activation semantics, and robust serialization/checkpointing.

## Summary: Critical Path vs Parallel Lanes

Current status: the project is still in **Phase 0**, with demo refinement active now and the `asciiMaze` SOLID split scheduled before repository-wide ES2023 modernization.

- **Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
- **Parallel lane A (performance):** [Memory_Optimization.md](Memory_Optimization.md) Track 1 after Phase 1 stabilizes
- **Parallel lane B (interop):** [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md) after Phase 2 (or earlier if scoped tightly)
- **Final capstone:** [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md)
