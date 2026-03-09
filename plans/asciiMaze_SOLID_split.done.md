# asciiMaze SOLID Split

Purpose

- Keep `test/examples/asciiMaze` aligned with the stronger modular example direction established by `test/examples/flappy_bird`.
- Track only the durable, high-level structural work still needed to reach a strict SOLID, DRY, maintainable end state.
- Keep this document resumable across sessions.
- Keep this document short; detailed split design belongs in the implementation step.

Progress rules

- Use `[]` for a step that is not yet completed.
- Use `[DONE]` immediately after finishing a step.
- Update this document whenever the step order changes, a step is added, a step is removed, or the overall plan shifts.
- Keep completed steps in place so progress is visible when resuming later.
- Do not expand this file with temporary implementation detail; only record durable high-level progress.

Current read

- `asciiMaze` is already meaningfully decomposed compared with an older flat layout.
- The remaining work is concentrated in a small number of large coordination and policy-heavy modules.
- Confirmed current runtime orchestration order: `browser-entry.ts` bootstraps host services and invokes `EvolutionEngine.runMazeEvolution()`, `evolutionEngine.ts` normalizes and prepares the run, `evolutionEngine/evolutionLoop.ts` owns generation orchestration, `mazeMovement.ts` executes per-agent simulation, and `dashboardManager.ts` emits telemetry back to browser listeners and host globals.
- The existing Step 2-8 order still matches that flow: split simulation policy first, then presentation, then browser host glue, then shared contracts, and only then collapse the remaining engine-side reporting and runtime adapter seams.
- Step 2 is now finalized around the dedicated `mazeMovement/` folder boundary: shared simulation types live in `mazeMovement.types.ts`, constants in `mazeMovement.constants.ts`, reusable helpers in `mazeMovement.utils.ts`, pooled mutable infrastructure in `mazeMovement.services.ts`, runtime primitives in `mazeMovement/runtime/mazeMovement.runtime.ts`, action policy in `mazeMovement/policy/mazeMovement.policy.ts`, reward shaping in `mazeMovement/shaping/mazeMovement.shaping.ts`, result assembly in `mazeMovement/finalization/mazeMovement.finalization.ts`, and the public class facade now lives in `mazeMovement/mazeMovement.ts` with the old top-level file reduced to a compatibility re-export.
- Step 3 is now finalized around the dedicated `dashboardManager/` folder boundary: shared dashboard types live in `dashboardManager.types.ts`, constants in `dashboardManager.constants.ts`, reusable formatting and calculation helpers live in `dashboardManager.utils.ts`, rendering/archive/telemetry orchestration now lives in `dashboardManager.services.ts`, and the public class facade now lives in `dashboardManager/dashboardManager.ts` with the old top-level file reduced to a compatibility re-export.
- Step 4 is now finalized around the dedicated `browser-entry/` folder boundary: shared browser host contracts live in `browser-entry.types.ts`, host and curriculum constants live in `browser-entry.constants.ts`, DOM resolution and curriculum helpers live in `browser-entry.utils.ts`, host bootstrap/resize wiring/abort composition/globals compatibility/runtime orchestration now live in `browser-entry.services.ts`, and the public browser facade now lives in `browser-entry/browser-entry.ts` with the old top-level file reduced to a compatibility re-export.
- Step 5 is now finalized around focused contract owners instead of the old `interfaces.ts` dependency bucket: evolution run/configuration and engine-internal helper contracts now live in `evolutionEngine/evolutionEngine.types.ts`, fitness evaluation contracts now live in `fitness.types.ts`, existing browser-entry/dashboardManager/mazeMovement module-owned `*.types.ts` files remain the primary owners for their split boundaries, and the top-level `interfaces.ts` file is now reduced to a narrow shared/core compatibility layer that re-exports moved contracts while retaining only the genuinely cross-cutting network, dashboard-abstraction, maze-run-result, and terminal-result shapes.
- Step 6 is now finalized around a narrow engine-owned host adapter boundary: `evolutionEngine/evolutionEngine.types.ts` owns the `EvolutionHostAdapter` and stop-event contracts, `evolutionEngine/setupHelpers.ts` now reads cooperative pause state through that adapter instead of polling browser globals directly, `evolutionEngine/evolutionLoop.ts` now reports solve and stop outcomes through the adapter instead of dispatching browser behavior itself, and `browser-entry/browser-entry.globals.services.ts` owns the browser implementation that maps those engine events back to browser globals and solved-event compatibility behavior.
- Step 7 is now finalized around owner-defined runtime and presentation contracts instead of browser-entry-local runtime shims: `dashboardManager/dashboardManager.types.ts` now owns the shared browser/non-browser presentation seam through `DashboardPresentationAdapter`, `DashboardTelemetryPayload`, and `AsciiMazeTelemetrySnapshot`; `browser-entry/browser-entry.types.ts` now consumes those dashboard-owned contracts for the public run handle and host services instead of defining its own `RuntimeDashboard`/`RuntimeEvolutionResult` adapters; and `evolutionEngine/evolutionEngine.types.ts` now owns the shared run-result and loose runtime helper contracts (`MazeEvolutionRunResult`, tracked network/genome helpers, species-history host) consumed by the engine facade, telemetry helpers, and population-dynamics helpers. The remaining compatibility-only runtime shapes are the browser platform shims in `browser-entry/browser-entry.types.ts` (`RuntimeWindow` and abort-signal helpers), not duplicated presentation or engine result seams.
- Step 8 is now finalized around an engine-owned curriculum/refinement boundary: `evolutionEngine/curriculumPhase.ts` now owns phase-outcome interpretation, curriculum solve-threshold evaluation, and refined winner carry-over resolution; `browser-entry/browser-entry.curriculum.services.ts` now focuses on browser-only dimension scheduling, animation-frame pacing, and lifecycle completion; and `asciiMaze.e2e.test.ts` now reuses the same engine-owned helper instead of re-implementing winner-refinement carry-over locally. The remaining compatibility surface is the underlying `networkRefinement.ts` implementation, which stays reusable behind the engine-owned curriculum helper rather than being called directly by browser-entry.
- Step 9 is now finalized around final-shape validation rather than further splitting: the folder-owned boundaries remain `mazeMovement/`, `dashboardManager/`, `browser-entry/`, and `evolutionEngine/`; the top-level compatibility files remain intentionally narrow for stable imports; the engine facade now explicitly documents `resolveMazeEvolutionPhaseOutcome` as a compatibility export owned by `evolutionEngine/curriculumPhase.ts`; and the required final validations succeeded with `npx tsc --noEmit -p tsconfig.json` and `npm run docs`.

Split standard

- Large or multi-responsibility example files should not be split into a growing pile of sibling files at the same level.
- When a main file becomes a real module boundary, split it into a dedicated folder that keeps orchestration, contracts, helpers, constants, errors, and services together.
- The exported main surface should remain easy to find and should keep the module's public orchestration flow.

Required naming pattern

- For a main module named `module`, prefer:
- `module/module.ts`
- `module/module.utils.ts`
- `module/module.types.ts`
- `module/module.errors.ts`
- `module/module.services.ts`
- `module/module.constants.ts`

- For a nested sub-module named `sub-module` inside `module`, prefer:
- `module/sub-module/module.sub-module.ts`
- `module/sub-module/module.sub-module.utils.ts`
- `module/sub-module/module.sub-module.types.ts`
- `module/sub-module/module.sub-module.errors.ts`
- `module/sub-module/module.sub-module.services.ts`
- `module/sub-module/module.sub-module.constants.ts`

Naming intent

- `*.ts`: main orchestration or primary public surface for that module boundary.
- `*.utils.ts`: pure or mostly pure helpers that do not define the module's main orchestration.
- `*.types.ts`: focused types, DTOs, interfaces, and narrow contracts.
- `*.errors.ts`: named error classes and error helpers local to the module.
- `*.services.ts`: side-effecting or stateful service helpers used by the orchestration layer.
- `*.constants.ts`: named constants and lookup tables local to the module.

Boundary rules

- Do not create dedicated folders for trivial files that are already small and single-purpose.
- Do create a dedicated folder once a file becomes a real subsystem with multiple helper categories.
- Keep the top-level `module/module.ts` file orchestration-first rather than turning it into another utility bucket.
- Prefer subfolders when a subsystem inside a module grows into its own concern, instead of continuing to append helper files to the parent module folder.
- Keep runtime-host glue, domain policy, rendering, telemetry, and shared contracts in separate files once they stop being tiny.

asciiMaze target shape

- `evolutionEngine.ts` should continue toward a dedicated `evolutionEngine/` orchestration-first module, which is already the correct direction.
- `mazeMovement.ts` should move toward a dedicated `mazeMovement/` folder rather than being split into multiple top-level siblings.
- `dashboardManager.ts` should move toward a dedicated `dashboardManager/` folder with rendering, archive, telemetry, and formatting boundaries.
- `browser-entry.ts` should move toward a dedicated `browser-entry/` folder that separates host bootstrap, runtime orchestration, globals compatibility, and resize behavior.
- `interfaces.ts` should be replaced by narrower module-local `*.types.ts` files plus a smaller shared contract surface where truly needed.

High-level gaps

- `mazeMovement.ts` is still a god module.
- It mixes simulation state, pooled buffers, action policy, exploration heuristics, reward shaping, saturation handling, and result finalization.
- Shared static mutable state in `mazeMovement.ts` weakens substitutability, test isolation, and future worker-safe reuse.
- When split, this should become a dedicated `mazeMovement/` folder instead of several new top-level files.

- `browser-entry.ts` still combines host bootstrapping with runtime policy.
- It currently handles DOM lookup, logger wiring, dashboard setup, resize behavior, abort composition, curriculum progression, best-network carry-over, compatibility globals, and auto-start behavior.
- This should be split into smaller host/runtime/bootstrap services so the browser entry becomes thin orchestration only.
- When split, prefer a dedicated `browser-entry/` folder with runtime, host, globals, and resize sub-areas.

- Runtime shape adapters are now mostly centralized in owning module boundaries.
- Engine-side loose runtime helper contracts now live in `evolutionEngine/evolutionEngine.types.ts`, and browser/non-browser presentation contracts now live in `dashboardManager/dashboardManager.types.ts`.
- Remaining runtime-specific shapes are limited to browser platform compatibility helpers rather than duplicated presentation or evolution-result adapters.

- Browser and non-browser presentation concerns are now more parallel.
- Host-specific browser wiring now depends on the shared dashboard-owned presentation model instead of re-declaring telemetry and redraw seams locally.
- Remaining follow-up work is concentrated in refinement, curriculum, and evolution orchestration boundaries rather than presentation-contract drift.

- Refinement, curriculum progression, and evolution orchestration now have clearer owners.
- The engine-owned curriculum helper interprets completed run results and refines carry-over winners, while browser-entry stays focused on host scheduling and maze-dimension progression.
- Future changes to winner refinement or curriculum carry-over policy should now land in the engine boundary instead of cross-cutting browser-entry and curriculum-style callers.

Execution steps

- [DONE] Step 1: Audit the remaining `asciiMaze` coordination-heavy surfaces and confirm the execution order across `mazeMovement`, `dashboardManager`, `browser-entry`, shared contracts, and engine-side reporting seams.
- [DONE] Step 2: Split `mazeMovement.ts` into a dedicated `mazeMovement/` module boundary so simulation state, pooled buffers, action policy, exploration heuristics, reward shaping, saturation handling, and result finalization stop accumulating in one file.
- [DONE] Step 3: Split `dashboardManager.ts` into a dedicated `dashboardManager/` module boundary so live rendering, solved-archive rendering, telemetry aggregation, bounded history storage, event emission, and formatting evolve behind focused files.
- [DONE] Step 4: Thin `browser-entry.ts` into a dedicated `browser-entry/` module boundary so host bootstrap, runtime orchestration, globals compatibility, and resize behavior evolve independently.
- [DONE] Step 5: Decompose `interfaces.ts` into focused module-owned `*.types.ts` files and leave behind only the smallest shared contract surface that is still truly cross-cutting.
- [DONE] Step 6: Remove browser-facing solve, stop, and pause side effects from engine internals so `evolutionEngine` reports through a narrower adapter or reporting boundary instead of touching host behavior directly.
- [DONE] Step 7: Consolidate scattered runtime shape adapters and presentation seams so browser and non-browser paths depend on clearer shared contracts rather than ad hoc local `Runtime*` compensating interfaces.
- [DONE] Step 8: Recheck refinement, curriculum, and evolution boundaries so follow-up changes do not reintroduce cross-cutting orchestration drift after the earlier splits.
- [DONE] Step 9: Validate the final shape by checking naming consistency, folder ownership, generated-doc expectations, and TypeScript/build health.

Execution expectation

- Each major split should create a dedicated folder first, then move one helper category at a time.
- Preserve the public entry surface in the main module file inside that folder.
- Avoid introducing one-off naming styles for each refactor; use the module and sub-module naming pattern above.

Done criteria

- No single demo file remains the obvious coordination sink for multiple unrelated responsibilities.
- Host glue, simulation policy, telemetry aggregation, and rendering each have clear module boundaries.
- Shared contracts are focused enough that runtime-specific adapters stop proliferating.
- The example remains educational, but the top-level flow becomes orchestration rather than implementation-heavy.
- Dedicated folders, file names, and subfolder names follow one predictable pattern instead of ad hoc split layouts.
