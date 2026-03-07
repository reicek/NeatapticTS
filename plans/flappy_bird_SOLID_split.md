# flappy_bird SOLID Split

Purpose

- Track the high-level structural gaps that still matter if `test/examples/flappy_bird` is to stay the reference demo for SOLID, modular example work.
- Keep this document intentionally shallow; detailed split decisions should be made during the implementation step.

Current read

- `flappy_bird` is already much closer to the desired modular shape than `asciiMaze`.
- Most remaining work is refinement-oriented rather than rescue-oriented.

Split standard

- Large or multi-responsibility demo files should not be split into an expanding pile of sibling files at the same folder level.
- When a file becomes a real subsystem, it should move into a dedicated folder that keeps orchestration, contracts, helpers, constants, errors, and services together.
- The main exported module file should remain the easiest entry point to find and should keep the top-level orchestration flow.

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
- `*.utils.ts`: helper logic that is not the main orchestration path.
- `*.types.ts`: focused types, DTOs, interfaces, and narrow contracts.
- `*.errors.ts`: named error classes and error helpers local to the module.
- `*.services.ts`: side-effecting or stateful service helpers used by the orchestration layer.
- `*.constants.ts`: named constants and lookup tables local to the module.

Boundary rules

- Do not create a dedicated folder for every tiny file that is already focused and stable.
- Do create a dedicated folder when a file becomes a real subsystem with multiple helper categories.
- Keep the top-level `module/module.ts` file orchestration-first rather than letting it become another broad utility bucket.
- Prefer subfolders once a module develops a clear internal subsystem instead of continuing to append more files at the parent level.
- Keep runtime-host glue, playback, visualization, worker protocol, trainer flow, and shared observation logic in clearly separated module boundaries.

flappy_bird target shape

- The `browser-entry/` area should continue to be the primary reference shape for dedicated folder-based splits.
- Larger browser-entry utility surfaces should continue moving toward narrower subfolders such as `host/`, `runtime/`, `playback/`, `network-view/`, `visualization/`, and `worker-channel/`.
- `trainer/trainer.ts` should stay or become an orchestration-first module backed by dedicated trainer-owned helper files rather than expanding as a single coordination file.
- `flappy-evolution-worker.ts` should continue toward a worker-owned folder boundary with protocol, playback, evolution, and simulation helpers grouped under that module.
- Observation and simulation-shared logic should stay grouped under dedicated shared folders rather than leaking back into broader top-level utility files.

High-level gaps

- A few browser-entry utility files are still too broad.
- `browser-entry.visualization.utils.ts`, `browser-entry.playback.utils.ts`, and `browser-entry.network-view.utils.ts` still concentrate multiple concerns inside single utility surfaces.
- These areas likely want another pass that separates orchestration, geometry/layout, rendering, and snapshot interpretation more sharply.
- When split further, this should happen inside dedicated subfolders rather than by adding more broad sibling utility files.

- The browser host layer is modular, but some DOM construction remains heavy in utility files.
- `browser-entry.host.utils.ts` still owns a large amount of host assembly.
- The host setup should stay compositional so layout creation, canvas creation, and panel wiring remain easy to swap or extend independently.
- Prefer a dedicated host module boundary over continued growth in one broad host utility file.

- Trainer orchestration is still a notable concentration point.
- `trainer/trainer.ts` appears to remain a large coordination surface for evaluation planning, reporting, stop behavior, and top-level evolution control.
- The trainer should stay as orchestration only, with policy and reporting details continuing to move into focused services.
- When split further, prefer trainer-owned `trainer/` files that follow the standard naming pattern instead of recreating a new catch-all helper file.

- Worker entry surfaces should remain thin and protocol-focused.
- `flappy-evolution-worker.ts` is already split better than many demos, but it is still a natural place for drift if playback, simulation, protocol, and evolution concerns start recombining.
- Keep worker bootstrapping separate from evaluation logic and playback-state assembly.
- When split further, keep worker logic inside a dedicated worker folder and its subfolders rather than scattering worker helpers across unrelated areas.

- Observation and simulation-shared logic should avoid becoming a second policy hub.
- `simulation-shared.observation.utils.ts` is one of the larger shared logic files.
- If more shaping or feature engineering accumulates there, it should be split by observation assembly responsibilities rather than growing into a broad shared utility bucket.
- Prefer `simulation-shared/` owned module files and subfolders over reintroducing broader top-level shared helpers.

- Network visualization is still a likely growth hotspot.
- The network-view/browser visualization path already spans several files and can easily drift into overlapping responsibilities across layout, color semantics, labeling, and draw-time interpretation.
- Keep those boundaries explicit so future visualization changes stay local.
- Prefer dedicated visualization and network-view module folders over mixed rendering files with overlapping scope.

- Compatibility globals and auto-start behavior should stay isolated.
- The runtime already has dedicated browser-global handling, which is good.
- The main gap here is to keep that compatibility surface from leaking back into runtime orchestration or host setup.
- Keep compatibility globals inside dedicated runtime-owned files instead of letting them bleed into broader runtime modules.

- Constants should continue to remain domain-scoped instead of re-centralizing.
- The constants split is one of the demo's strengths.
- Future refactors should resist collapsing constants back into broader files or recreating magic-number logic inside runtime utilities.
- Any new constant groups should remain module-owned and follow the same predictable naming and folder pattern.

Priority order

- First: keep splitting oversized browser-entry utility modules by responsibility.
- Second: keep `trainer.ts` thin and orchestration-only.
- Third: guard worker entrypoints against responsibility creep.
- Fourth: keep observation and visualization logic from reforming broad utility hubs.
- Fifth: preserve the current domain-scoped constants structure.

Execution expectation

- Each major split should create or continue a dedicated folder boundary first, then move one helper category at a time.
- Preserve the public entry surface in the main module file inside that folder.
- Avoid introducing one-off naming styles during cleanup; use the module and sub-module naming pattern above.

Done criteria

- Browser runtime files read as orchestration over focused services rather than large utility buckets.
- Trainer and worker entrypoints stay thin, with domain logic living behind explicit service boundaries.
- Visualization, observation, playback, and host setup each evolve independently without overlapping responsibility.
- The demo remains the stronger reference shape that other examples can copy.
- Dedicated folders, file names, and subfolder names continue following one predictable pattern instead of drifting into ad hoc split layouts.
