# Flappy Bird SOLID Refactor Plan (Folder-by-Folder)

## Scope and goals

This plan defines a **folder-by-folder split** for `test/examples/flappy_bird` to:

1. Reduce file size and cognitive complexity.
2. Separate pure logic from orchestration/side effects.
3. Improve SOLID compliance (especially SRP, OCP, DIP, ISP).
4. Keep naming consistent with current project style (`kebab-name.segment.ts`).
5. Produce educational, rich JSDoc and step-level inline comments.

We will execute this plan incrementally, one folder/domain at a time.

---

## Naming convention (agreed)

Use folder-rooted modules with descriptive file roles:

- `category/category.ts` (public entry / façade)
- `category/category.types.ts`
- `category/category.constants.ts`
- `category/category.errors.ts`
- `category/category.utils.ts` (pure functions only)
- `category/category.service.ts` (side effects/orchestration)

For subcategories:

- `category/category.subcategory.ts`
- `category/category.subcategory.types.ts`
- `category/category.subcategory.constants.ts`
- `category/category.subcategory.errors.ts`
- `category/category.subcategory.utils.ts`
- `category/category.subcategory.service.ts`

### Rules

- No default exports in new split modules unless required by existing API.
- Each module should have one clear reason to change.
- Keep methods small:
	- pure helpers: focused transforms,
	- orchestrators: linear flow (`Step 1`, `Step 2`, ...), delegate detail to helpers.
- Use long, descriptive names for locals and helpers.

---

## Execution strategy

### Pass model per folder

For each folder/domain split, follow this order:

1. Create `.types`, `.constants`, `.errors`.
2. Move pure logic into `.utils` first.
3. Move side effects/orchestration into `.service`.
4. Keep old file as compatibility façade (`category.ts`) and rewire imports.
5. Delete moved code from source file immediately after each move.
6. Run typecheck (`npx tsc --noEmit -p tsconfig.json`).

### Definition of done (per folder)

- Largest file in the folder is manageable (target: mostly < 250–350 lines).
- Public API unchanged unless explicitly approved.
- JSDoc present on all exported symbols.
- No broad barrel import where narrow import is possible.
- No mixed pure logic + side effects in same new module.

---

## Priority order (folder-by-folder)

1. `browser-entry/` (largest UX/runtime hotspot)
2. `flappyEvolution.worker.ts` (worker god-module)
3. `trainFlappyBird.ts` (trainer orchestration + analytics)
4. `flappy.simulation.shared.utils.ts` (shared domain decomposition)
5. `flappyEvaluation.ts` and `flappyEnvironment.ts` (evaluation/environment seams)
6. `constants/` normalization pass (only where needed)

---

## Folder 1 plan: `browser-entry/`

### 1A) Split `browser-entry.playback.utils.ts`

Create folder: `browser-entry/playback/`

Target files:

- `browser-entry/playback/playback.ts` (public entry)
- `browser-entry/playback/playback.types.ts`
- `browser-entry/playback/playback.constants.ts` (only local playback-specific constants)
- `browser-entry/playback/playback.errors.ts`
- `browser-entry/playback/playback.worker-channel.utils.ts` (pure request payload builders/parsers)
- `browser-entry/playback/playback.render.utils.ts` (pure geometry/color selection)
- `browser-entry/playback/playback.render.service.ts` (canvas drawing side effects)
- `browser-entry/playback/playback.starfield.utils.ts` (pure star layout math)
- `browser-entry/playback/playback.starfield.service.ts` (canvas/offscreen tile creation cache)
- `browser-entry/playback/playback.trail.utils.ts` (pure trail transformations)
- `browser-entry/playback/playback.loop.service.ts` (RAF loop + worker step orchestration)

Outcome:

- `animatePopulationEpisode` becomes a thin orchestrator.
- Render path and worker stepping are independent services.

### 1B) Split `browser-entry.visualization.utils.ts`

Create folder: `browser-entry/visualization/`

Target files:

- `browser-entry/visualization/visualization.ts`
- `browser-entry/visualization/visualization.types.ts`
- `browser-entry/visualization/visualization.constants.ts`
- `browser-entry/visualization/visualization.errors.ts`
- `browser-entry/visualization/visualization.colors.utils.ts`
- `browser-entry/visualization/visualization.legend.utils.ts`
- `browser-entry/visualization/visualization.topology.utils.ts`
- `browser-entry/visualization/visualization.draw.service.ts`

Outcome:

- Color-scale policy, legend layout, topology inference, and drawing become separate responsibilities.

### 1C) Split `browser-entry.network-view.utils.ts`

Create folder: `browser-entry/network-view/`

Target files:

- `browser-entry/network-view/network-view.ts`
- `browser-entry/network-view/network-view.types.ts`
- `browser-entry/network-view/network-view.constants.ts`
- `browser-entry/network-view/network-view.layout.utils.ts`
- `browser-entry/network-view/network-view.labels.utils.ts`
- `browser-entry/network-view/network-view.draw.service.ts`

Outcome:

- Network layout math and rendering orchestration are split.

### 1D) Split `browser-entry.host.utils.ts`

Create folder: `browser-entry/host/`

Target files:

- `browser-entry/host/host.ts`
- `browser-entry/host/host.types.ts`
- `browser-entry/host/host.constants.ts`
- `browser-entry/host/host.dom.service.ts` (DOM creation)
- `browser-entry/host/host.canvas.service.ts` (canvas context/config)
- `browser-entry/host/host.resize.service.ts` (resize observers/listeners)
- `browser-entry/host/host.stats.service.ts` (HUD table wiring)

Outcome:

- Host DOM composition and resize/canvas runtime concerns are isolated.

### 1E) Split `browser-entry.worker-channel.utils.ts`

Create folder: `browser-entry/worker-channel/`

Target files:

- `browser-entry/worker-channel/worker-channel.ts`
- `browser-entry/worker-channel/worker-channel.types.ts`
- `browser-entry/worker-channel/worker-channel.errors.ts`
- `browser-entry/worker-channel/worker-channel.url.service.ts` (worker URL resolution)
- `browser-entry/worker-channel/worker-channel.request.service.ts` (generic request/response)
- `browser-entry/worker-channel/worker-channel.generation.service.ts`
- `browser-entry/worker-channel/worker-channel.playback.service.ts`

Outcome:

- Remove duplicated message/error wiring and centralize protocol plumbing.

### 1F) Recompose `browser-entry.ts`

Create folder: `browser-entry/runtime/`

Target files:

- `browser-entry/runtime/runtime.ts`
- `browser-entry/runtime/runtime.types.ts`
- `browser-entry/runtime/runtime.errors.ts`
- `browser-entry/runtime/runtime.telemetry.service.ts`
- `browser-entry/runtime/runtime.evolution-loop.service.ts`
- `browser-entry/runtime/runtime.browser-globals.service.ts`

`browser-entry.ts` becomes compatibility façade exporting `start` from runtime.

### 1G) Replace broad barrel export

- Keep `browser-entry.utils.ts` only as temporary compatibility file.
- Stop importing it from worker/domain modules.
- Long-term: explicit imports from narrow modules only.

---

## Folder 2 plan: worker domain (`flappyEvolution.worker.ts`)

Create folder: `flappy-evolution-worker/`

Target files:

- `flappy-evolution-worker/flappy-evolution-worker.ts` (worker entry point)
- `flappy-evolution-worker/flappy-evolution-worker.types.ts`
- `flappy-evolution-worker/flappy-evolution-worker.constants.ts`
- `flappy-evolution-worker/flappy-evolution-worker.errors.ts`
- `flappy-evolution-worker/flappy-evolution-worker.protocol.service.ts` (message router/dispatcher)
- `flappy-evolution-worker/flappy-evolution-worker.runtime.service.ts` (init + lifecycle)
- `flappy-evolution-worker/flappy-evolution-worker.evolution.service.ts` (generation evolve/publish)
- `flappy-evolution-worker/flappy-evolution-worker.playback.service.ts` (start/step/finalize)
- `flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts` (structured-clone DTO mapping)
- `flappy-evolution-worker/flappy-evolution-worker.simulation.utils.ts` (pure simulation helpers)

Dependency inversion requirement:

- Remove dependency on `browser-entry.utils.ts`.
- Introduce focused shared imports from simulation-specific modules only.

---

## Folder 3 plan: trainer domain (`trainFlappyBird.ts`)

Create folder: `trainer/`

Target files:

- `trainer/trainer.ts` (CLI entry)
- `trainer/trainer.types.ts`
- `trainer/trainer.constants.ts`
- `trainer/trainer.errors.ts`
- `trainer/trainer.setup.service.ts` (neat config + runtime init)
- `trainer/trainer.evaluation-plan.utils.ts` (pure generation plan/schedules)
- `trainer/trainer.fitness.service.ts` (population evaluation orchestration)
- `trainer/trainer.reporting.utils.ts` (pure log/report composition)
- `trainer/trainer.loop.service.ts` (outer evolve loop)
- `trainer/trainer.signals.service.ts` (SIGINT/SIGTERM)
- `trainer/trainer.statistics.utils.ts` (mean/std/percentile)

Outcome:

- Trainer loop stays linear and delegates evaluation/report internals.

---

## Folder 4 plan: shared simulation (`flappy.simulation.shared.utils.ts`)

Create folder: `simulation-shared/`

Target files:

- `simulation-shared/simulation-shared.ts`
- `simulation-shared/simulation-shared.types.ts`
- `simulation-shared/simulation-shared.constants.ts`
- `simulation-shared/simulation-shared.errors.ts`
- `simulation-shared/simulation-shared.difficulty.utils.ts`
- `simulation-shared/simulation-shared.spawn.utils.ts`
- `simulation-shared/simulation-shared.observation.utils.ts`
- `simulation-shared/simulation-shared.memory.utils.ts`
- `simulation-shared/simulation-shared.control.utils.ts`
- `simulation-shared/simulation-shared.math.utils.ts`

Outcome:

- Cross-runtime pure domain logic is composable, testable, and independent of browser/worker side effects.

---

## Folder 5 plan: evaluation and environment

### 5A) `flappyEvaluation.ts` → `evaluation/`

- `evaluation/evaluation.ts`
- `evaluation/evaluation.types.ts`
- `evaluation/evaluation.constants.ts`
- `evaluation/evaluation.rollout.service.ts`
- `evaluation/evaluation.fitness.utils.ts`
- `evaluation/evaluation.statistics.utils.ts`
- `evaluation/evaluation.seed.utils.ts`

### 5B) `flappyEnvironment.ts` → `environment/`

- `environment/environment.ts`
- `environment/environment.types.ts`
- `environment/environment.constants.ts`
- `environment/environment.state.service.ts`
- `environment/environment.step.service.ts`
- `environment/environment.observation.utils.ts`
- `environment/environment.collision.utils.ts`

Outcome:

- Environment stepping and evaluation concerns become separate modules.

---

## Folder 6 plan: constants normalization

The `constants/` directory is already split well. Only perform targeted cleanup:

- move duplicated local literals from services into nearest `*.constants.ts`,
- preserve existing exported constant names unless there is explicit migration,
- ensure concise educational JSDoc for shared constants.

---

## Cross-cutting SOLID enforcement checklist

For every folder split:

1. **SRP**: each file has one role (types/constants/errors/utils/service/entry).
2. **OCP**: add new behavior by adding handlers/helpers, not editing long conditional blocks.
3. **LSP**: keep contracts stable when extracting services/helpers.
4. **ISP**: define narrow interfaces for each collaborator.
5. **DIP**: orchestration depends on interfaces/typed contracts, not broad utility barrels.

### Code quality requirements

- Rich educational JSDoc on exported symbols (`@param`, `@returns`, concise example where useful).
- Step-level inline comments in orchestrators.
- Long, descriptive names; avoid one-letter or cryptic locals.
- Prefer immutable transforms and ES2023 methods (`toSorted`, `toReversed`, `.at(-1)`, etc.) when behavior allows.

### Validation per completed folder

- `npx tsc --noEmit -p tsconfig.json`
- optional follow-up: targeted test/example execution after split is stable.

---

## Folder-by-folder execution log template

Use this checklist each time we start a folder:

- [ ] Define target file map for folder.
- [ ] Create skeleton files (`types/constants/errors/utils/service/entry`).
- [ ] Move code category-by-category (types → pure utils → services).
- [ ] Rewire imports and remove old in-place code immediately.
- [ ] Add/upgrade JSDoc and step-level comments.
- [ ] Run typecheck.
- [ ] Mark folder complete and prepare next folder plan delta.

---

## First execution target

Start with `browser-entry/playback/` as the first concrete split, then proceed to `browser-entry/visualization/`.

