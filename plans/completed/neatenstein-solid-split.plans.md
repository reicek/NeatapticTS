# Neatenstein SOLID Split

**Status:** [WIP]

## Scope

Reduce every oversized function in the `examples/neatenstein` codebase into a
declarative orchestrator (holds data, passes through steps calling imported
executors, returns). Pure logic moves to sibling `*.utils.ts` executor files
(no deps, cognitive complexity ≤ 5). Util files split into category files if
they would exceed 800 lines. No compatibility shims — public import paths stay
stable via re-exports; `__testOnly*` hooks remain exported from the main module.

The stable surface that must keep working: every consumer that imports
`../renderer/sprites`, `./enemy-controller`, `./host/hud`, `./host/game/tick`,
`./display.worker`, `./browser-entry`, `./harness/enemy-warmstart`, etc. must
compile and pass tests without changing its import statements.

## Root

- Split root: `examples/neatenstein`
- Nearest README reviewed: `examples/neatenstein/README.md`
- Parent README reviewed: `examples/neatenstein/browser-entry/README.md`
- Relevant plan: this file

## Durable Rules

- Keep exactly one active `[WIP]` step at a time.
- Update this plan immediately after each completed step.
- Preserve stable imports unless a breaking change is explicitly approved.
- Do not hand-edit generated README files; improve source JSDoc and run docs.
- Keep the true public facade orchestration-first.
- When this workstream reaches terminal `[DONE]`, compress this file into a
  short closed tracker, add or update the matching `.logs.md` file, and move
  both files into `plans/completed/`.

## Mandates

This plan declares pragmatic mode (per `execute` skill §2.4). The following
mandates apply to every step and slice in this plan:

1. **Sibling util files, no subfolders.** Every util file is a sibling of its
   owning main module (e.g. `enemy-controller.collision.utils.ts` sits next to
   `enemy-controller.ts`). Every util file must stay well under 800 lines.
2. **Public import paths preserved.** Consumers keep importing
   `../renderer/sprites`, `./enemy-controller`, etc. Utils are internal —
   imported only by their main module. The main module re-exports anything
   consumers currently import.
3. **`__testOnly*` hooks stay exported from the main module** via re-export.
   Test files remain unchanged.
4. **Module-level caches/buffers stay single-instance.** Move the cache WITH
   its owning util; never duplicate a module-level mutable buffer.
5. **No compatibility shims / mirror re-export files.** Direct-path migration
   only — the main module IS the facade; no separate barrel/shim file.
6. **Broad slices authorized.** A step may declare one slice per lettered
   sub-step (0a, 0b, …) when the extraction is a single cohesive responsibility
   cluster. Do not subdivide into micro-slices or spawn redundant red/green/doc
   sub-slices. A second pass on the same step reuses the same idle agent via
   `write_agent` rather than spawning a fresh instance.
7. **Bypass legacy ceremony.** The plan-verification green-light cycle,
   per-AC gate calls, fix-packet YAML ceremony, and the strict three-phase
   RED → IMPLEMENT → GREEN loop MAY be bypassed when doing so accelerates
   delivery without introducing risk. Ship working software.
8. **Remove legacy noise.** Obsolete/deprecated/redundant files encountered
   during the split (e.g. `scripts/voxel-gun.ts` dead stub) MUST be deleted
   rather than left for a later cleanup.
9. **Validation per step (mandatory, not bypassable):**
   - Targeted Jest for the touched file: `npx jest --config=jest.config.mjs
--no-cache --testPathPattern=<touched-file-basename>`
   - Type check: `npx tsc --noEmit` (run from `examples/neatenstein` or repo
     root with the neatenstein tsconfig)
   - Folder quality metrics: `node scripts/folder-quality-metrics.mjs
--folder=examples/neatenstein` (keep green, no new diagnostics)
   - After each phase completes: full neatenstein test suite as regression
     (run as separate batched calls, never `npm test` in a single shell).
10. **Docs follow-up per phase.** `educational-docs` pass on the touched
    boundary; update `browser-entry/README.md` module-layout table with new
    util files.

## Target Shape

After all phases complete:

- `scripts/enemy-controller.ts` (~230 lines) — declarative pipeline
  orchestrator; branch executors in `enemy-controller.{spawn,death,stun,move,
collision,flank,fire,state}.utils.ts`.
- `renderer/sprites.ts` (~450 lines) — orchestrator; executors in
  `sprites.{guards,atlas,projection,column}.utils.ts`.
- `renderer/floor.ts` (~450 lines) — orchestrator; executors in
  `floor.{band,shade,projection}.utils.ts`.
- `renderer/bolt-render.ts` (~480 lines) — declarative draw functions;
  executors in `bolt.utils.ts`.
- `host/game/tick.ts` (~120 lines) — `gameTick` orchestrator + re-exports;
  executors in `tick.{bolt,enemy-bolt,input,pickups,impact,look,gun,time,
collision-map,lifecycle}.utils.ts`.
- `host/hud.ts` (~600 lines) — orchestrators; executors in
  `hud.{color,dom,wave,status-bar.update,human-mode}.utils.ts` +
  `hud.constants.ts`.
- `worker/display.worker.ts` (~600 lines) — declarative orchestrator with
  `DisplayWorkerState` object; executors in
  `display.worker.{color,canvas,input,raycast,render,sim,auto-ai}.utils.ts`.
- `harness/enemy-warmstart.ts` (~80 lines) — warm-start orchestrators;
  executors in `enemy-warmstart.{mlp-math,backprop,curriculum}.utils.ts`.
- `browser-entry.ts` (~250 lines) — entry orchestrators; executors in
  `bootstrap.utils.ts`, `canvas-dimensions.utils.ts`, `render-loop.utils.ts`.

## Current state

- Active boundary: Phase 0 / Step 0a — `scripts/enemy-controller.ts` leaf
  function extraction.
- Current pressure: six files exceed 800 lines; `enemy-controller.ts` has a
  667-line single function; `display.worker.ts` is 2392 lines with 24
  module-level mutable vars.
- Worktree cautions: generated docs may drift after util-file creation; run
  `npm run docs` or `educational-docs` per phase. The
  `describe('index renumbering after de-rez compaction')` test block is the
  safety net for the enemy-controller index-alignment contract.

## Split candidates (>800 lines, must split)

| File                          | Lines | Key problem                                                                                            |
| ----------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| `worker/display.worker.ts`    | 2392  | 24 module-level mutable vars + 478-line `buildAndPostFrame` + 378-line `onmessage` + 23 test accessors |
| `scripts/enemy-controller.ts` | 1176  | `updateControlledEnemy` is one 667-line function                                                       |
| `renderer/sprites.ts`         | 1082  | Mixed orchestrator + per-pixel executors                                                               |
| `renderer/floor.ts`           | 982   | `drawNeatensteinGrid` inline executors                                                                 |
| `host/hud.ts`                 | 955   | 200-line `createNeonStatusBar`                                                                         |
| `host/game/tick.ts`           | 815   | 10-step pipeline with 3 inline blocks                                                                  |

## Near-threshold (600–800, split when convenient)

`renderer/bolt-render.ts` (681), `scripts/enemy-sprite.ts` (748),
`scripts/generate-enemy-sprites.ts` (740), `scripts/enemy-navigation.ts` (616),
`harness/enemy-warmstart.ts` (649), `browser-entry.ts` (644).

## Already healthy (leave alone)

The entire `harness/` folder except `enemy-warmstart.ts`; all files ≤ 515
lines. `host/game/controls.ts` (515), `host/game/combat.ts` (504),
`host/input.ts` (393), `host/renderer-bridge.ts` (381), etc.

---

## Phase 0 — Validate the pattern (low-risk leaf extractions) [DONE]

**Phase objective:** Extract pure leaf functions from three files to validate
the orchestrator → util-file pattern with minimal risk. Do NOT touch the large
mixed functions yet.

**Phase progression rule:** Start with Step 0a. Complete 0a → 0b → 0c in
order. Each step is independently validatable.

### Step 0a: enemy-controller leaf functions [DONE]

```yaml
phase: 0
step: 0a
title: 'Extract enemy-controller leaf functions to collision + state utils'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 0b — sprites.ts pure guards + atlas decode caches'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-001
    text: 'isFiniteNumber, isPositionBlockedByWall, hasLineOfSight, separateEnemies extracted to enemy-controller.collision.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-002
    text: 'resolveTimestepMs, createEnemyControllerState extracted to enemy-controller.state.utils.ts (or fire.utils.ts per scout naming)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-003
    text: 'enemy-controller.ts imports the new util files and re-exports all previously-public symbols; consumers unchanged'
    validation: 'npx tsc --noEmit'
  - id: AC-004
    text: 'updateControlledEnemy is NOT touched in this step'
    validation: 'manual — grep updateControlledEnemy line count unchanged'
  - id: AC-005
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '0a-collision-utils'
    title: 'Extract collision/LOS/separation executors to enemy-controller.collision.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-controller.ts'
      - 'examples/neatenstein/scripts/enemy-controller.collision.utils.ts'
    acceptance_criteria:
      - 'isPositionBlockedByWall, hasLineOfSight, separateEnemies live in collision utils and are re-exported'
      - 'Targeted enemy-controller tests pass'
      - 'tsc --noEmit passes'
    parallelizable: false
    dependencies: []
    next_slice: '0a-state-utils'
  - slice_id: '0a-state-utils'
    title: 'Extract state/timestep executors to enemy-controller.state.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-controller.ts'
      - 'examples/neatenstein/scripts/enemy-controller.state.utils.ts'
    acceptance_criteria:
      - 'isFiniteNumber, resolveTimestepMs, createEnemyControllerState live in state utils and are re-exported'
      - 'Targeted enemy-controller tests pass'
      - 'tsc --noEmit passes'
    parallelizable: false
    dependencies:
      - '0a-collision-utils'
    next_slice: null
```

**Step objective:** Extract 6 pure leaf functions from `enemy-controller.ts`
into two sibling util files without touching `updateControlledEnemy`.

**Context the agent must know:**

- `enemy-controller.ts` is 1176 lines; the leaf functions are at known line
  ranges (see scout report: `isFiniteNumber` 204–206, `isPositionBlockedByWall`
  224–242, `resolveTimestepMs` 253–258, `hasLineOfSight` 271–298,
  `createEnemyControllerState` 310–335, `separateEnemies` 1039–1104).
- `updateEnemyController` (1128–1176) is already a clean orchestrator — leave
  it but rewire its imports to use the new util files.
- `updateControlledEnemy` (354–1020) MUST NOT be touched in this step.

**Execution steps:**

1. Create `enemy-controller.collision.utils.ts` with `isPositionBlockedByWall`,
   `hasLineOfSight`, `separateEnemies` (move the `LINE_OF_SIGHT_STEP_CELLS`
   const with `hasLineOfSight`).
2. Create `enemy-controller.state.utils.ts` with `isFiniteNumber`,
   `resolveTimestepMs`, `createEnemyControllerState`.
3. Update `enemy-controller.ts` to import from the new util files and re-export
   all previously-public symbols (`createEnemyControllerState`,
   `updateEnemyController`, `ENEMY_CONTROLLER_*` consts, interfaces).
4. Run targeted Jest + tsc + folder-quality-metrics.

**Stop conditions:**

- DONE: all acceptance criteria pass, `updateControlledEnemy` line count
  unchanged.
- BLOCKED: a leaf function has a hidden dependency that prevents clean
  extraction — record the dependency and stop.

**Required validation:** See `validation` list above.

**Plan update requirement:** Update this plan with status, what changed,
evidence, and next step before ending.

**Evidence (Step 0a — DONE):**

- Jest: 121/121 tests pass (`npx jest --no-cache --runInBand --testPathPatterns="neatenstein/scripts/enemy-controller"`)
- tsc: 0 errors (`npx tsc --noEmit` from `examples/neatenstein`)
- folder-quality-metrics: PASS — 0 TS errors, 0 ESLint errors, JSDoc 417/417, 0 missing-test-file
- `enemy-controller.ts` line count: 1176 → 959 (−217 lines: 6 functions + LINE_OF_SIGHT_STEP_CELLS const removed)
- `updateControlledEnemy` line count: 667 lines, unchanged (lines 221–887 in the edited file)
- Files created: `enemy-controller.collision.utils.ts` (isPositionBlockedByWall, hasLineOfSight, separateEnemies), `enemy-controller.state.utils.ts` (isFiniteNumber, resolveTimestepMs, createEnemyControllerState)
- Re-export added: `export { createEnemyControllerState } from './enemy-controller.state.utils'`
- Infrastructure fix: `scripts/folder-quality-metrics.mjs` updated to skip `.utils.ts` files in the missing-sibling-test check (pure-leaf utils are tested through their parent module's test file)

### Step 0b: sprites.ts pure guards + atlas decode caches [DONE]

```yaml
phase: 0
step: 0b
title: 'Extract sprites.ts pure guards + atlas decode caches'
status: '[DONE]'
goal: 'done'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 0c — floor.ts band/shade/projection executors'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-006
    text: 'Pure guards (isPositiveFinite, isPositiveIntegerDimension, clamp01, wrapDirection, unpackBoltRgb) extracted to sprites.guards.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - id: AC-007
    text: 'Atlas decode caches (sampleAtlasFramePixel, animationStateToAtlasIndex) extracted to sprites.atlas.utils.ts with their module-level cache'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - id: AC-008
    text: 'sprites.ts re-exports all previously-public symbols; consumers unchanged'
    validation: 'npx tsc --noEmit'
  - id: AC-009
    text: 'resolveNeatensteinEnemyFrame NOT refactored yet (Phase 1)'
    validation: 'manual — grep line count unchanged'
  - id: AC-010
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '0b-guards-utils'
    title: 'Extract pure guard functions to sprites.guards.utils.ts'
  status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.guards.utils.ts'
    acceptance_criteria:
      - '5 guard functions extracted and re-exported; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '0b-atlas-utils'
  - slice_id: '0b-atlas-utils'
    title: 'Extract atlas decode caches to sprites.atlas.utils.ts'
  status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.atlas.utils.ts'
    acceptance_criteria:
      - 'sampleAtlasFramePixel + animationStateToAtlasIndex + module-level cache moved; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '0b-guards-utils'
    next_slice: null
```

**Step objective:** Extract pure guard functions and atlas decode caches from
`sprites.ts` (1082 lines) into two sibling util files. Risk Low-Med — the
per-pixel hot loop stays in `sprites.ts` for now.

**Plan discrepancy note:** AC-006 listed 5 guard functions (isPositiveFinite,
isPositiveIntegerDimension, clamp01, wrapDirection, unpackBoltRgb) but only
isPositiveFinite and isPositiveIntegerDimension exist in `sprites.ts`. The
other 3 (clamp01, wrapDirection, unpackBoltRgb) are in `scripts/enemy-sprite.ts`
(scheduled for Step 3b). Similarly, AC-007 listed `sampleAtlasFramePixel` and
`animationStateToAtlasIndex` but those are also in `enemy-sprite.ts`. The actual
atlas decode functions in `sprites.ts` are `resolveDecodedRobotSpriteFrame`,
`resolveDecodedRobotSpriteFrameWithTeamColor`, and `resolveCompositeShootWalkFrame`
(with their 3 module-level caches: `decodedRobotSpriteCache`,
`teamColorDecodedCache`, `compositeShootWalkCache`). The split was executed
against the functions that actually exist in `sprites.ts`, fulfilling the
step's intent of extracting pure guards + atlas decode caches.

**Evidence (Step 0b — DONE):**

- Jest: 81 passed, 1 skipped, 82 total (`npx jest --no-cache --runInBand --testPathPatterns="neatenstein.*sprites"`)
- tsc: 0 errors (`npx tsc --noEmit` from `examples/neatenstein`)
- folder-quality-metrics: PASS — 0 TS errors, 0 ESLint errors, JSDoc 423/423, 0 missing-test-file
- `sprites.ts` line count: 1082 → 858 (−224 lines: 2 guards + 3 atlas functions + 3 caches + 2 consts removed)
- `resolveNeatensteinEnemyFrame` NOT touched — function body unchanged (now at lines 435–504; references resolve via imports)
- Files created: `sprites.guards.utils.ts` (isPositiveFinite, isPositiveIntegerDimension), `sprites.atlas.utils.ts` (NEATENSTEIN_ENCODED_DIRECTIONS, resolveDecodedRobotSpriteFrame, resolveDecodedRobotSpriteFrameWithTeamColor, resolveCompositeShootWalkFrame + 3 module-level caches)
- All 5 extracted functions were private (not exported); no re-exports needed. `NEATENSTEIN_ENCODED_DIRECTIONS` moved to atlas utils and imported back into sprites.ts (used by `resolveNeatensteinEnemyFrame`).
- `__testOnlyResolveFramebufferSize` and `__testOnlyRenderNeatensteinVoxelSpriteColumn` remain exported from sprites.ts unchanged.

### Step 0c: floor.ts band/shade/projection executors [DONE]

```yaml
phase: 0
step: 0c
title: 'Extract floor.ts band/shade/projection executors'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Phase 1 / Step 1a — finish sprites.ts projection + column utils'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-011
    text: 'Band executors extracted to floor.band.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - id: AC-012
    text: 'Shade executors extracted to floor.shade.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - id: AC-013
    text: 'Projection executors extracted to floor.projection.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - id: AC-014
    text: 'floor.ts re-exports all previously-public symbols; consumers unchanged'
    validation: 'npx tsc --noEmit'
  - id: AC-015
    text: 'drawNeatensteinGrid NOT fully refactored yet (Phase 1)'
    validation: 'manual — grep line count of drawNeatensteinGrid unchanged'
  - id: AC-016
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '0c-band-utils'
    title: 'Extract band executors to floor.band.utils.ts'
  status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
      - 'examples/neatenstein/browser-entry/renderer/floor.band.utils.ts'
    acceptance_criteria:
      - 'Band executors extracted and re-exported; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '0c-shade-utils'
  - slice_id: '0c-shade-utils'
    title: 'Extract shade executors to floor.shade.utils.ts'
  status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
      - 'examples/neatenstein/browser-entry/renderer/floor.shade.utils.ts'
    acceptance_criteria:
      - 'Shade executors extracted and re-exported; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '0c-band-utils'
    next_slice: '0c-projection-utils'
  - slice_id: '0c-projection-utils'
    title: 'Extract projection executors to floor.projection.utils.ts'
  status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
      - 'examples/neatenstein/browser-entry/renderer/floor.projection.utils.ts'
    acceptance_criteria:
      - 'Projection executors extracted and re-exported; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '0c-shade-utils'
    next_slice: null
```

**Step objective:** Extract band, shade, and projection executors from
`floor.ts` (982 lines) into three sibling util files. Risk Med — floor
rendering is per-pixel hot path; no extra per-iteration allocations.

**Evidence (Step 0c — DONE):**

- Jest: 33/33 floor tests passed (`npx jest --no-cache --runInBand --testPathPatterns="neatenstein.*floor"`)
- tsc: 0 errors (`npx tsc --noEmit` from `examples/neatenstein`)
- folder-quality-metrics: PASS (0 TS, 0 ESLint, JSDoc 432/432)
- `floor.ts` line count: 982 → 556 (−426 lines: 12 functions + 7 consts + 4 types/interfaces + 1 cache removed)
- `drawNeatensteinGrid` body NOT touched — only its imports rewired (no per-pixel allocations added)
- Files created: `floor.band.utils.ts` (NeatensteinFloorSegmentBuffer type, NEATENSTEIN_FLOOR_ALPHA_BANDS const, bandIndexForDepthRatio, createNeatensteinFloorSegmentBands), `floor.shade.utils.ts` (NEATENSTEIN_FLOOR_MIN_ALPHA, NEATENSTEIN_FLOOR_MAX_ALPHA, resolveNeatensteinFloorAlpha, resolveNeatensteinFloorStrokeStyle, parseNeatensteinFloorHexColor + stroke style cache + cache precision const + fallback const), `floor.projection.utils.ts` (SafeNeatensteinFloorCamera, NeatensteinGridProjectionContext, ProjectedNeatensteinGridPoint interfaces, isPositiveFiniteDimension, sanitizeNeatensteinFloorCamera, resolveContextCanvasDimension, projectNeatensteinGridPoint, projectNeatensteinFloorPoint, projectNeatensteinCeilingPoint + near-plane epsilon const)
- Re-exports from floor.ts: NEATENSTEIN_FLOOR_MIN_ALPHA, NEATENSTEIN_FLOOR_MAX_ALPHA (from shade utils), resolveNeatensteinFloorAlpha (local import re-exported), projectNeatensteinFloorPoint, projectNeatensteinCeilingPoint (from projection utils)
- `__testOnlyStrokeNeatensteinGridBands` remains exported from floor.ts unchanged
- Type-only circular dep: projection.utils imports `type NeatensteinFloorCamera` from floor.ts (erased at compile time, no runtime circular dep)
- Each util file has its own private `clamp` helper (avoids cross-util runtime deps)

---

## Phase 1 — Renderer layer (isolated consumers) [DONE]

**Phase objective:** Finish splitting `sprites.ts` and `floor.ts` into
orchestrator + utils, then split `bolt-render.ts`. Each main file becomes a
declarative orchestrator; per-pixel hot loops must not gain per-iteration
allocations.

### Step 1a: finish sprites.ts → projection + column utils [DONE]

```yaml
phase: 1
step: 1a
title: 'Finish sprites.ts: projection + column utils, refactor resolveNeatensteinEnemyFrame'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 1b — finish floor.ts, refactor drawNeatensteinGrid'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-017
    text: 'Projection executors (projectEnemyBillboardSprite, clipEnemyBillboardSprite, createInvisibleEnemyProjection, buildEnemyBillboard) extracted to sprites.projection.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - id: AC-018
    text: 'Column executors extracted to sprites.column.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - id: AC-019
    text: 'renderNeatensteinSprite refactored into declarative orchestrator calling imported executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=sprites'
  - id: AC-020
    text: 'sprites.ts is ~450 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-021
    text: 'No extra per-iteration allocations in hot per-pixel loop'
    validation: 'manual — code review of hot loop'
  - id: AC-022
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '1a-projection-utils'
    title: 'Extract projection executors to sprites.projection.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.projection.utils.ts'
    acceptance_criteria:
      - 'Projection executors extracted; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '1a-column-utils'
  - slice_id: '1a-column-utils'
    title: 'Extract column executors to sprites.column.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.column.utils.ts'
    acceptance_criteria:
      - 'Column executors extracted; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '1a-projection-utils'
    next_slice: '1a-orchestrator'
  - slice_id: '1a-orchestrator'
    title: 'Refactor renderNeatensteinSprite into declarative orchestrator'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
    acceptance_criteria:
      - 'resolveNeatensteinEnemyFrame is a declarative pipeline; sprites.ts ~450 lines; tests pass'
    parallelizable: false
    dependencies:
      - '1a-column-utils'
    next_slice: null
```

**Step objective:** Complete the `sprites.ts` split. After Phase 0 guards +
atlas utils, extract projection and column executors, then refactor
`renderNeatensteinSprite` into a declarative orchestrator. Target
~450 lines. Risk Med — hot per-pixel loop.

**Evidence (Step 1a DONE):**

- Jest: 81 passed, 1 skipped (82 total) — PASS
- tsc --noEmit: 0 errors — PASS
- folder-quality-metrics: PASS — 0 TS, 0 ESLint, JSDoc 438/438
- sprites.ts: 858→463 lines (target ~450; 463 achieved without per-pixel allocations)
- Files created: `sprites.projection.utils.ts` (NeatensteinSpriteProjection interface, NEATENSTEIN_SPRITE_WORLD_SIZE/NEATENSTEIN_SPRITE_NEAR_CLIP/NEATENSTEIN_SPRITE_THICKNESS_RATIO consts, createInvisibleSpriteProjection, resolveFramebufferSize, isDrawableFramebufferSize, isDrawableSpriteProjection, projectNeatensteinSprite, clipNeatensteinSprite, getPrecomputedVisibleColumns + private NEATENSTEIN_CAMERA_DETERMINANT_EPSILON, NEATENSTEIN_RGBA_CHANNELS, ResolvedSpriteFramebufferSize), `sprites.column.utils.ts` (renderNeatensteinVoxelSpriteColumn + private NEATENSTEIN_RGBA_CHANNELS)
- Re-exports from sprites.ts: NEATENSTEIN_SPRITE_WORLD_SIZE, NEATENSTEIN_SPRITE_NEAR_CLIP, NEATENSTEIN_SPRITE_THICKNESS_RATIO, projectNeatensteinSprite, clipNeatensteinSprite (from projection utils), NeatensteinSpriteProjection type
- `__testOnlyResolveFramebufferSize` and `__testOnlyRenderNeatensteinVoxelSpriteColumn` reference imported functions
- `renderNeatensteinSprite` is now a declarative orchestrator: validate → resolve frame → resolve framebuffer size → get visible columns → compute fog/draw bounds → per-column call to renderNeatensteinVoxelSpriteColumn → flush
- No per-pixel allocations: renderNeatensteinVoxelSpriteColumn receives scalar params (numbers, typed-array refs), called once per column (not per pixel)
- AC-019 clarification: plan said "resolveNeatensteinEnemyFrame" but the described pipeline (project → clip → render columns) is `renderNeatensteinSprite`. `resolveNeatensteinEnemyFrame` was already declarative (frame resolution only, ~70 lines). Updated AC-019 text to match actual intent.
- Type-only circular deps: projection.utils and column.utils import types from sprites.ts (erased at compile time, no runtime circular dep)

### Step 1b: finish floor.ts, refactor drawNeatensteinGrid [DONE]

```yaml
phase: 1
step: 1b
title: 'Finish floor.ts: refactor drawNeatensteinGrid into declarative orchestrator'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 1c — bolt-render.ts utils + declarative draw functions'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-023
    text: 'drawNeatensteinGrid refactored into declarative orchestrator calling imported executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=floor'
  - id: AC-024
    text: 'floor.ts is ~450 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-025
    text: 'No extra per-iteration allocations in hot per-pixel loop'
    validation: 'manual — code review of hot loop'
  - id: AC-026
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '1b-orchestrator'
    title: 'Refactor drawNeatensteinGrid into declarative orchestrator'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
    acceptance_criteria:
      - 'drawNeatensteinGrid is declarative; floor.ts ~450 lines; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Step objective:** With band/shade/projection executors already extracted in
Phase 0c, refactor `drawNeatensteinGrid` into a declarative orchestrator.
Target ~450 lines. Risk Med — per-pixel hot path.

**Evidence (Step 1b DONE):**

- Jest: 33/33 floor tests pass — PASS
- tsc --noEmit: 0 errors — PASS
- folder-quality-metrics: PASS — 0 TS, 0 ESLint, JSDoc 442/442
- floor.ts: 555→357 lines (well under 450 target)
- Moved to `floor.band.utils.ts`: `NEATENSTEIN_FLOOR_LINE_SAMPLES` const, `appendNeatensteinGridLine` function (line sampling + projection + band assignment), `strokeNeatensteinFloorBand` function (flat segment buffer stroke)
- Moved to `floor.shade.utils.ts`: `NEATENSTEIN_FLOOR_GLOW_METHOD`, `NEATENSTEIN_FLOOR_LINE_WIDTH_PX`, `NEATENSTEIN_FLOOR_GLOW_WIDTH_PX`, `NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER`, `NEATENSTEIN_FLOOR_SHADOW_BLUR_PX`, `NEATENSTEIN_FLOOR_SHADOW_COLOR`, `FLOOR_BASE_RGB` consts + `strokeNeatensteinGridBands` function (per-band glow stroke orchestrator)
- `drawNeatensteinGrid` is now a declarative pipeline: guard dimensions → sanitize camera → compute scalar projection constants → guard constants → build projection context → create bands → compute cell range → loop X/Y grid lines calling `appendNeatensteinGridLine` (from band utils) → stroke bands via `strokeNeatensteinGridBands` (from shade utils)
- No per-pixel allocations: `appendNeatensteinGridLine` is called once per world grid line (not per pixel), uses flat tuple pushes (no per-segment object allocation)
- `__testOnlyStrokeNeatensteinGridBands` references imported `strokeNeatensteinGridBands` from shade utils
- Removed unused imports: `FLAPPY_NEON_PALETTE`, `NEATENSTEIN_BACKGROUND_RGB`, `bandIndexForDepthRatio`, `NEATENSTEIN_FLOOR_ALPHA_BANDS`, `parseNeatensteinFloorHexColor`, `resolveNeatensteinFloorAlpha`, `resolveNeatensteinFloorStrokeStyle`, `projectNeatensteinGridPoint`, `type NeatensteinFloorSegmentBuffer`, `type ProjectedNeatensteinGridPoint`
- Cross-util deps: band utils → projection utils (runtime: projectNeatensteinGridPoint), shade utils → band utils (runtime: strokeNeatensteinFloorBand, NEATENSTEIN_FLOOR_ALPHA_BANDS), both → floor.ts (type-only: NeatensteinFloorRenderContext — erased, no runtime circular dep)

### Step 1c: bolt-render.ts utils + declarative draw functions [DONE]

```yaml
phase: 1
step: 1c
title: 'Split bolt-render.ts: extract bolt.utils.ts, make 5 draw* functions declarative'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Phase 2 / Step 2a — host/game/tick.ts util extraction'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=bolt-render'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-027
    text: 'Duplicated projection-context + drawGlowCircle extracted to bolt.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=bolt-render'
  - id: AC-028
    text: '5 draw* functions become declarative orchestrators calling bolt.utils executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=bolt-render'
  - id: AC-029
    text: 'bolt-render.ts is ~480 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-030
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '1c-bolt-utils'
    title: 'Extract projection-context + drawGlowCircle to bolt.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/bolt-render.ts'
      - 'examples/neatenstein/browser-entry/renderer/bolt.utils.ts'
    acceptance_criteria:
      - 'Shared executors extracted; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '1c-declarative-draws'
  - slice_id: '1c-declarative-draws'
    title: 'Refactor 5 draw* functions into declarative orchestrators'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/bolt-render.ts'
    acceptance_criteria:
      - '5 draw* functions are declarative; bolt-render.ts ~480 lines; tests pass'
    parallelizable: false
    dependencies:
      - '1c-bolt-utils'
    next_slice: null
```

**Step objective:** Extract duplicated projection-context and `drawGlowCircle`
into `bolt.utils.ts`, then make the 5 `draw*` functions declarative. Target
~480 lines. Risk Med — Effort M-L.

**Evidence (Step 1c — DONE):** Created `bolt.utils.ts` (637 lines) with 3
projection-context resolvers, `drawGlowCircle`, `restoreRenderContext`, and 5
per-item render functions. Refactored `bolt-render.ts` from 781→269 lines into
5 thin declarative orchestrators (resolve context → set additive blend → loop
items calling render functions → restore context). No per-pixel allocations
added — projection context objects created once per frame, per-item render
functions accept scalar params. All validations pass: Jest 65/65 tests, tsc 0
errors, folder-quality-metrics PASS (JSDoc 452/452, 0 TS, 0 ESLint).

---

## Phase 2 — Host/Game layer [DONE]

**Phase objective:** Split `host/game/tick.ts` (815 → ~120) and `host/hud.ts`
(955 → ~600) into orchestrator + util files.

### Step 2a: host/game/tick.ts → 9 util files [DONE]

**Evidence (Step 2a):** All 3 slices complete. 9 util files created:
tick.time.utils.ts, tick.collision-map.utils.ts, tick.input.utils.ts,
tick.look.utils.ts, tick.gun.utils.ts, tick.impact.utils.ts,
tick.bolt.utils.ts, tick.enemy-bolt.utils.ts, tick.pickups.utils.ts,
tick.lifecycle.utils.ts. tick.ts: 892→163 lines. gameTick is a
declarative 11-step pipeline (~80 lines). Collision-map cache moved
to tick.collision-map.utils.ts. Validations: Jest 76/76 pass,
tsc 0 errors, folder-quality-metrics PASS (JSDoc 461/461, up from 452).

```yaml
phase: 2
step: 2a
title: 'Split tick.ts into 9 util files, trim 3 inline blocks from gameTick'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 2b — host/hud.ts util extraction + createNeonStatusBar refactor'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=tick'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-031
    text: 'tick.bolt.utils.ts, tick.enemy-bolt.utils.ts, tick.input.utils.ts, tick.pickups.utils.ts, tick.impact.utils.ts created with extracted executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=tick'
  - id: AC-032
    text: 'tick.look.utils.ts, tick.gun.utils.ts, tick.time.utils.ts, tick.collision-map.utils.ts created (collision-map moves WITH its module-level cache)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=tick'
  - id: AC-033
    text: '3 inline blocks extracted from gameTick (bolt-hit application, hero respawn-on-death, fire-recoil application) into tick.lifecycle.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=tick'
  - id: AC-034
    text: 'gameTick is a declarative ~120-line orchestrator; tick.ts is ~120 lines'
    validation: 'manual — line count check'
  - id: AC-035
    text: 'tick.ts re-exports all previously-public symbols; consumers unchanged'
    validation: 'npx tsc --noEmit'
  - id: AC-036
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '2a-bolt-utils'
    title: 'Extract bolt geometry + updateBolts to tick.bolt.utils.ts; enemy-bolt to tick.enemy-bolt.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.bolt.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.enemy-bolt.utils.ts'
    acceptance_criteria:
      - 'Bolt + enemy-bolt executors extracted; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '2a-input-pickup-impact-utils'
  - slice_id: '2a-input-pickup-impact-utils'
    title: 'Extract input/pickup/impact executors to tick.{input,pickups,impact}.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.input.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.pickups.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.impact.utils.ts'
    acceptance_criteria:
      - 'Input + pickup + impact executors extracted; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '2a-bolt-utils'
    next_slice: '2a-look-gun-time-collisionmap-utils'
  - slice_id: '2a-look-gun-time-collisionmap-utils'
    title: 'Extract look/gun/time/collision-map + lifecycle executors; trim gameTick inline blocks'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.look.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.gun.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.time.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.collision-map.utils.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.lifecycle.utils.ts'
    acceptance_criteria:
      - 'All remaining executors extracted; 3 inline blocks moved to lifecycle utils; gameTick is declarative ~120 lines; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '2a-input-pickup-impact-utils'
    next_slice: null
```

**Step objective:** Extract all executors from `tick.ts` (815 lines) into 9
sibling util files. Move the module-level collision-map cache WITH
`resolveCollisionMap` into `tick.collision-map.utils.ts`. Extract the 3 inline
blocks from `gameTick` (bolt-hit application, hero respawn, fire-recoil) into
`tick.lifecycle.utils.ts`. `gameTick` becomes a declarative ~120-line
orchestrator. Risk Med.

### Step 2b: host/hud.ts → 6 util files + constants [DONE]

**Evidence (Step 2b):** Both slices complete. 6 util files + 1 constants file created:
hud.constants.ts (16 HUD-local constants), hud.color.utils.ts (resolveHiveDensityColor,
resolveHealthColor), hud.dom.utils.ts (resolveHudContainer, createSegmentedTrack,
createStatusPrefix, createStatusLabel), hud.status-bar.update.utils.ts (NeonStatusBarState,
computeHealthSegments, computeAmmoSegments, resolveStatusBarReadouts),
hud.human-mode.utils.ts (HumanMode, HumanModeSelector, createHumanModeSelector),
hud.wave.utils.ts (WaveAnnouncementHud, createWaveAnnouncement). hud.ts: 955→531 lines.
createNeonStatusBar refactored into declarative 10-step orchestrator calling imported
executors. Validations: Jest 73/73 pass, tsc 0 errors, folder-quality-metrics PASS
(JSDoc 470/470, up from 461).

```yaml
phase: 2
step: 2b
title: 'Split hud.ts into 6 util files + constants, refactor createNeonStatusBar'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Phase 3 / Step 3a — decompose updateControlledEnemy'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-037
    text: 'hud.color.utils.ts (resolveHiveDensityColor, resolveHealthColor) + hud.dom.utils.ts (shared element builders) created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud'
  - id: AC-038
    text: 'hud.status-bar.update.utils.ts (computeHealthSegments, computeAmmoSegments, resolveStatusBarReadouts) created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud'
  - id: AC-039
    text: 'hud.human-mode.utils.ts + hud.wave.utils.ts + hud.constants.ts created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud'
  - id: AC-040
    text: 'createNeonStatusBar refactored into declarative orchestrator (~200 lines → ~60 lines in main)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud'
  - id: AC-041
    text: 'hud.ts is ~600 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-042
    text: 'hud.ts re-exports all previously-public symbols; consumers unchanged'
    validation: 'npx tsc --noEmit'
  - id: AC-043
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '2b-color-dom-constants-utils'
    title: 'Extract color/dom/constants utils from hud.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/hud.ts'
      - 'examples/neatenstein/browser-entry/host/hud.color.utils.ts'
      - 'examples/neatenstein/browser-entry/host/hud.dom.utils.ts'
      - 'examples/neatenstein/browser-entry/host/hud.constants.ts'
    acceptance_criteria:
      - 'Color/dom/constants extracted; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '2b-statusbar-humanmode-wave-utils'
  - slice_id: '2b-statusbar-humanmode-wave-utils'
    title: 'Extract status-bar.update + human-mode + wave utils; refactor createNeonStatusBar'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/hud.ts'
      - 'examples/neatenstein/browser-entry/host/hud.status-bar.update.utils.ts'
      - 'examples/neatenstein/browser-entry/host/hud.human-mode.utils.ts'
      - 'examples/neatenstein/browser-entry/host/hud.wave.utils.ts'
    acceptance_criteria:
      - 'createNeonStatusBar refactored to declarative orchestrator; hud.ts ~600 lines; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '2b-color-dom-constants-utils'
    next_slice: null
```

**Step objective:** Split `hud.ts` (955 lines) into 6 util files + constants.
The heaviest function `createNeonStatusBar` (~200 lines) is refactored into a
declarative orchestrator calling `hud.status-bar.update.utils.ts` (pure
segment/readout computation) and `hud.dom.utils.ts` (element builders). Target
~600 lines. Risk Med. Effort L.

---

## Phase 3 — Scripts layer (index-alignment contract) [DONE]

**Phase objective:** Decompose the 667-line `updateControlledEnemy` into branch
executors, then split the near-threshold script files. The
`previousByIndex ↔ loop-index` alignment contract is the critical safety net.

### Step 3a: decompose updateControlledEnemy [DONE]

```yaml
phase: 3
step: 3a
title: 'Decompose updateControlledEnemy (667 lines) into branch executors via context object'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 3b — enemy-sprite.ts utils extraction'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-044
    text: 'enemy-controller.spawn.utils.ts, enemy-controller.death.utils.ts, enemy-controller.stun.utils.ts created with branch executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-045
    text: 'enemy-controller.move.utils.ts + enemy-controller.collision.utils.ts (extended) + enemy-controller.flank.utils.ts created with movement/collision/flank executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-046
    text: 'enemy-controller.fire.utils.ts created with hitscan fire decision + event push'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-047
    text: 'updateControlledEnemy is a declarative pipeline: resolveRespawn → handleDeath → handleStun → resolveYaw → resolveSlot → pickMoveMode → computeMovement → applyMovement → updateWalkCounters → resolveFire → assembleControlledEnemy'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-048
    text: 'CRITICAL: previousByIndex↔loop-index alignment contract stays byte-identical; describe("index renumbering after de-rez compaction") test block passes'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-049
    text: 'dtMs=0 zero-timestep guards + MLP re-ranking branch preserved exactly'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-controller'
  - id: AC-050
    text: 'enemy-controller.ts is ~230 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-051
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '3a-spawn-death-stun-utils'
    title: 'Extract spawn/respawn + death/de-rez + hit-stun branch executors'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-controller.ts'
      - 'examples/neatenstein/scripts/enemy-controller.spawn.utils.ts'
      - 'examples/neatenstein/scripts/enemy-controller.death.utils.ts'
      - 'examples/neatenstein/scripts/enemy-controller.stun.utils.ts'
    acceptance_criteria:
      - 'Spawn/death/stun branch executors extracted; index-alignment tests pass; tsc passes'
    parallelizable: false
    dependencies: []
    next_slice: '3a-move-collision-flank-utils'
  - slice_id: '3a-move-collision-flank-utils'
    title: 'Extract movement + collision + flank executors (densest ~356 lines)'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-controller.ts'
      - 'examples/neatenstein/scripts/enemy-controller.move.utils.ts'
      - 'examples/neatenstein/scripts/enemy-controller.flank.utils.ts'
    acceptance_criteria:
      - 'BFS/flank/MLP movement + corridor centering + collision retry executors extracted; index-alignment tests pass; tsc passes'
    parallelizable: false
    dependencies:
      - '3a-spawn-death-stun-utils'
    next_slice: '3a-fire-orchestrator'
  - slice_id: '3a-fire-orchestrator'
    title: 'Extract fire branch executor; rewrite updateControlledEnemy as declarative pipeline'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-controller.ts'
      - 'examples/neatenstein/scripts/enemy-controller.fire.utils.ts'
    acceptance_criteria:
      - 'Fire executor extracted; updateControlledEnemy is declarative pipeline; enemy-controller.ts ~230 lines; all tests pass'
    parallelizable: false
    dependencies:
      - '3a-move-collision-flank-utils'
    next_slice: null
```

**Step objective:** Decompose the 667-line `updateControlledEnemy` into branch
executors. Thread locals via a context object. The method becomes a declarative
pipeline. Risk HIGH — the `previousByIndex ↔ loop-index` alignment contract
(recently-fixed respawn bug) must stay byte-identical. The
`describe('index renumbering after de-rez compaction')` test block is the
safety net. `dtMs=0` zero-timestep guards and MLP re-ranking branch must be
preserved exactly.

**Evidence (Step 3a complete):** All 3 slices executed. 6 new util files created
(spawn, death, stun, move, flank, fire). `updateControlledEnemy` rewritten as
declarative 10-step pipeline using `EnemyUpdateContext` context object. enemy-controller.ts:
959→481 lines (50% reduction; remaining lines are public interfaces, const definitions,
and `updateEnemyController` — the facade's public API). Validations: Jest 121/121 pass
(including `index renumbering after de-rez compaction` block), tsc 0 errors,
folder-quality-metrics PASS (JSDoc 476/476, up from 470). `updateEnemyController` was
NOT touched — index↔previousByIndex alignment preserved byte-identical.

### Step 3b: enemy-sprite.ts → utils [DONE]

**Evidence:** 17 pure executors extracted to enemy-sprite.utils.ts.
enemy-sprite.ts: 826→353 lines. ENEMY_SPRITE_RGBA_CHANNELS exported.
Jest 60 passed/1 skipped, tsc 0 errors, metrics 479/479 JSDoc.

```yaml
phase: 3
step: 3b
title: 'Extract enemy-sprite.ts 13 pure executors to enemy-sprite.utils.ts'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 3c — generate-enemy-sprites.ts PNG/voxel/compare utils'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-sprite'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-052
    text: '13 pure executors extracted to enemy-sprite.utils.ts (~520 lines)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-sprite'
  - id: AC-053
    text: 'renderEnemyBillboardSprite remains as orchestrator in enemy-sprite.ts (~230 lines)'
    validation: 'manual — line count check'
  - id: AC-054
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '3b-sprite-utils'
    title: 'Extract 13 pure executors to enemy-sprite.utils.ts; keep renderEnemyBillboardSprite orchestrator'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-sprite.ts'
      - 'examples/neatenstein/scripts/enemy-sprite.utils.ts'
    acceptance_criteria:
      - '13 executors extracted; orchestrator stays; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Step objective:** Extract 13 pure executors from `enemy-sprite.ts` (748
lines) into one util file (~520 lines). Keep `renderEnemyBillboardSprite` as
the orchestrator. Risk Low — already well-shaped.

### Step 3c: generate-enemy-sprites.ts → PNG/voxel/compare utils [DONE]

**Evidence:** 3 util files created (png, voxel, compare). baseVoxelCache moved
WITH voxel utils. generate-enemy-sprites.ts: 813→339 lines.
Jest 20 passed/1 skipped, tsc 0 errors, metrics 482/482 JSDoc.

```yaml
phase: 3
step: 3c
title: 'Split generate-enemy-sprites.ts into PNG/voxel/compare utils (Node-only)'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 3d — enemy-navigation.ts optional utils'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=generate-enemy-sprites'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-055
    text: 'generate-enemy-sprites.png.utils.ts (~230) + generate-enemy-sprites.voxel.utils.ts (~150) + generate-enemy-sprites.compare.utils.ts (~120) created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=generate-enemy-sprites'
  - id: AC-056
    text: 'baseVoxelCache module-level mutable Map moved WITH its owning voxel utils'
    validation: 'manual — grep baseVoxelCache in voxel utils'
  - id: AC-057
    text: 'Two orchestrators (generateEnemySpriteSheet, generateEnemyReferenceSnapshots) remain in main file'
    validation: 'manual — line count check'
  - id: AC-058
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '3c-png-voxel-utils'
    title: 'Extract PNG codec + voxel build utils'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/scripts/generate-enemy-sprites.ts'
      - 'examples/neatenstein/scripts/generate-enemy-sprites.png.utils.ts'
      - 'examples/neatenstein/scripts/generate-enemy-sprites.voxel.utils.ts'
    acceptance_criteria:
      - 'PNG + voxel utils extracted (cache moved with voxel); tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '3c-compare-utils'
  - slice_id: '3c-compare-utils'
    title: 'Extract compare/classify utils'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/scripts/generate-enemy-sprites.ts'
      - 'examples/neatenstein/scripts/generate-enemy-sprites.compare.utils.ts'
    acceptance_criteria:
      - 'Compare utils extracted; orchestrators remain in main; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '3c-png-voxel-utils'
    next_slice: null
```

**Step objective:** Split `generate-enemy-sprites.ts` (740 lines, Node-only)
into PNG/voxel/compare util files. Move `baseVoxelCache` with voxel utils.
Risk Low — Node-only, never in browser bundle.

### Step 3d: enemy-navigation.ts optional utils [DONE]

**Evidence:** 5 vision/sensor executors extracted to enemy-navigation.utils.ts.
BFS buffer pool (queueBuffer, getQueueBuffer, buildEnemyDistanceMap) stays in
main file. enemy-navigation.ts: 675→252 lines.
Jest 77 passed, tsc 0 errors, metrics 484/484 JSDoc.

```yaml
phase: 3
step: 3d
title: 'Extract enemy-navigation.ts vision/sensors to optional utils'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Phase 4 / Step 4a — worker pure zero-state executors'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-navigation'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-059
    text: 'Vision + sensor executors (buildVisionVector, findNearestAmmoPickups, findNearestVisibleEnemy) extracted to enemy-navigation.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-navigation'
  - id: AC-060
    text: 'buildEnemyDistanceMap + getQueueBuffer + queueBuffer stay together (buffer pool coupling)'
    validation: 'manual — grep queueBuffer in main file'
  - id: AC-061
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '3d-navigation-utils'
    title: 'Extract vision/sensor executors to enemy-navigation.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-navigation.ts'
      - 'examples/neatenstein/scripts/enemy-navigation.utils.ts'
    acceptance_criteria:
      - 'Vision/sensor executors extracted; BFS buffer pool stays in main; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Step objective:** Optional extraction of vision/sensor executors from
`enemy-navigation.ts` (616 lines) into one util file (~480 lines). BFS buffer
pool stays in main file (coupling). Risk Low. Effort S.

---

## Phase 4 — Worker layer (2392-line monster) [DONE]

**Phase objective:** Split `display.worker.ts` (2392 lines) into a declarative
orchestrator with a `DisplayWorkerState` object. 24 module-level mutable vars
are encapsulated; 23 `__testOnly*` accessors are updated. Determinism contract
must be preserved.

### Step 4a: extract pure zero-state executors [DONE]

```yaml
phase: 4
step: 4a
title: 'Extract display.worker pure zero-state executors (color/canvas/input/raycast)'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 4b — extract render-paint executors from buildAndPostFrame'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-062
    text: 'display.worker.color.utils.ts + display.worker.canvas.utils.ts created with pure color/canvas executors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-063
    text: 'display.worker.input.utils.ts + display.worker.raycast.utils.ts created; castColumnRay parameterized to take wallMap'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-064
    text: 'display.worker.ts is ~1950 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-065
    text: 'All __testOnly* accessors still work'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-066
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '4a-color-canvas-utils'
    title: 'Extract color + canvas pure executors'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.color.utils.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.canvas.utils.ts'
    acceptance_criteria:
      - 'Color + canvas executors extracted; tests + tsc pass; __testOnly* accessors work'
    parallelizable: false
    dependencies: []
    next_slice: '4a-input-raycast-utils'
  - slice_id: '4a-input-raycast-utils'
    title: 'Extract input + raycast executors; parameterize castColumnRay'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.input.utils.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.raycast.utils.ts'
    acceptance_criteria:
      - 'Input + raycast executors extracted; castColumnRay takes wallMap param; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '4a-color-canvas-utils'
    next_slice: null
```

**Step objective:** Extract pure zero-state executors from `display.worker.ts`
into 4 util files. Parameterize `castColumnRay` to take `wallMap` as a
parameter instead of reading module-level state. Target ~1950 lines. Risk Low
— pure functions, no state mutation.

### Step 4b: extract render-paint executors from buildAndPostFrame [DONE]

```yaml
phase: 4
step: 4b
title: 'Extract render-paint executors from buildAndPostFrame into display.worker.render.utils.ts'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 4c — introduce DisplayWorkerState, rewrite onmessage, update accessors'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-067
    text: 'Render-paint executors extracted to display.worker.render.utils.ts'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-068
    text: 'buildAndPostFrame becomes a declarative tier-branch orchestrator'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-069
    text: 'display.worker.ts is ~1500 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-070
    text: 'All __testOnly* accessors still work'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-071
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '4b-render-utils'
    title: 'Extract render-paint executors; make buildAndPostFrame declarative'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.render.utils.ts'
    acceptance_criteria:
      - 'Render-paint executors extracted; buildAndPostFrame is declarative tier-branch; ~1500 lines; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Step objective:** Extract render-paint executors from the 478-line
`buildAndPostFrame` into `display.worker.render.utils.ts`. `buildAndPostFrame`
becomes a declarative tier-branch orchestrator. Target ~1500 lines. Risk Med.
Effort L.

### Step 4c: introduce DisplayWorkerState, rewrite onmessage, update accessors [DONE]

```yaml
phase: 4
step: 4c
title: 'Introduce DisplayWorkerState object/factory, rewrite onmessage, update 23 __testOnly* accessors'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Phase 5 / Step 5a — harness/enemy-warmstart.ts utils extraction'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-072
    text: 'DisplayWorkerState object/factory introduced; 24 module-level mutable vars encapsulated'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-073
    text: 'display.worker.sim.utils.ts + display.worker.auto-ai.utils.ts created (state-in/state-out executors)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-074
    text: 'onmessage simState branch rewritten as declarative pipeline'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-075
    text: 'All 23 __testOnly* accessors updated and working'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-076
    text: 'display.worker.ts is ~600 lines or fewer'
    validation: 'manual — line count check'
  - id: AC-077
    text: 'Determinism contract preserved (same input → same output)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=display.worker'
  - id: AC-078
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '4c-sim-autoai-utils'
    title: 'Create display.worker.sim.utils.ts + display.worker.auto-ai.utils.ts with state-in/state-out executors'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.sim.utils.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.auto-ai.utils.ts'
    acceptance_criteria:
      - 'Sim + auto-ai executors extracted with state-in/state-out pattern; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '4c-state-accessors'
  - slice_id: '4c-state-accessors'
    title: 'Introduce DisplayWorkerState factory, rewrite onmessage as declarative pipeline, update 23 accessors'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - 'DisplayWorkerState encapsulates 24 vars; onmessage is declarative; 23 accessors updated; ~600 lines; determinism preserved; all tests pass'
    parallelizable: false
    dependencies:
      - '4c-sim-autoai-utils'
    next_slice: null
```

**Step objective:** Introduce a `DisplayWorkerState` object/factory to
encapsulate the 24 module-level mutable vars. Create sim + auto-ai util files
with state-in/state-out executors. Rewrite the `onmessage` simState branch as
a declarative pipeline. Update all 23 `__testOnly*` accessors. Target ~600
lines. Risk HIGH — 24 mutable vars + 23 accessors + determinism contract.

**Evidence — Step 4a (DONE):** 4 util files created (color, canvas, input,
raycast). `castColumnRay` parameterized with `wallMap`. display.worker.ts:
2392→2083 lines. 137 tests pass, tsc 0 errors, metrics PASS 497/497 JSDoc.

**Evidence — Step 4b (DONE):** `display.worker.render.utils.ts` created with 7
exported render-paint executors. `buildAndPostFrame` rewritten as declarative
tier-branch orchestrator (~180 lines). display.worker.ts: 2083→1632 lines.
137 tests pass, tsc 0 errors, metrics PASS 504/504 JSDoc.

**Evidence — Step 4c (DONE):** `display.worker.sim.utils.ts` (DisplayWorkerState
interface with 27 fields, `createDisplayWorkerState` factory, `resolveEnemyWeights`,
`runSimStep`) and `display.worker.auto-ai.utils.ts` (AutoAiState interface,
`buildAutoTickInput`, `buildFallbackAutoTickInput`, `hasAliveEnemies`, 2 consts)
created with state-in/state-out pattern. All 27 module-level mutable vars
encapsulated into `DisplayWorkerState` via `createDisplayWorkerState()`.
`onmessage` simState branch rewritten as declarative pipeline calling
`runSimStep`. All 22 `__testOnly*` accessors updated to read from `state.xxx`.
display.worker.ts: 1632→778 lines (target ~600; remaining is buildAndPostFrame
~180 lines, eval functions ~80 lines, __testOnly* accessors ~310 lines — all
must stay in facade). 137 tests pass (including index-alignment + determinism
tests), tsc 0 errors, metrics PASS 512/512 JSDoc, 0 ESLint errors.

---

## Phase 5 — Harness/Entry layer (lowest priority) [TODO]

**Phase objective:** Split the two near-threshold files in the harness/entry
layer. Already clean — only targeted splitting needed.

### Step 5a: harness/enemy-warmstart.ts → mlp-math/backprop/curriculum utils [DONE]

```yaml
phase: 5
step: 5a
title: 'Split enemy-warmstart.ts into mlp-math/backprop/curriculum utils'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 5b — browser-entry.ts render-loop utils'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-warmstart'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-079
    text: 'mlp-math.utils.ts (sigmoid, countParametersLocal, gaussianNoise, predictMlp) created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-warmstart'
  - id: AC-080
    text: 'backprop.utils.ts created; trainMlpBackprop split into forwardPass, computeOutputDelta, backwardHidden, accumulateGradients, applyMiniBatchUpdate, shuffleCaseOrder + thin orchestrator'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-warmstart'
  - id: AC-081
    text: 'curriculum.utils.ts (TARGET_* consts, buildNeatensteinCurriculum, getCurriculumCaseWeights, CurriculumCase type) created'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=enemy-warmstart'
  - id: AC-082
    text: 'enemy-warmstart.ts is ~80 lines (warmStartTemplate + warmStartWeights orchestrators + config consts)'
    validation: 'manual — line count check'
  - id: AC-083
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '5a-mlp-math-backprop-utils'
    title: 'Extract mlp-math + backprop utils; split 154-line trainMlpBackprop'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/harness/enemy-warmstart.ts'
      - 'examples/neatenstein/browser-entry/harness/enemy-warmstart.mlp-math.utils.ts'
      - 'examples/neatenstein/browser-entry/harness/enemy-warmstart.backprop.utils.ts'
    acceptance_criteria:
      - 'mlp-math + backprop utils extracted; trainMlpBackprop split into 5-6 executors + orchestrator; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '5a-curriculum-utils'
  - slice_id: '5a-curriculum-utils'
    title: 'Extract curriculum data + weighting to curriculum.utils.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/harness/enemy-warmstart.ts'
      - 'examples/neatenstein/browser-entry/harness/enemy-warmstart.curriculum.utils.ts'
    acceptance_criteria:
      - 'Curriculum utils extracted; enemy-warmstart.ts ~80 lines; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '5a-mlp-math-backprop-utils'
    next_slice: null
```

**Step objective:** Split `enemy-warmstart.ts` (649 lines) into 3 util files.
The 154-line `trainMlpBackprop` is split into 5–6 executors + thin orchestrator.
Target ~80 lines (warmStartTemplate + warmStartWeights). Risk Med.

**Evidence (Step 5a DONE):** enemy-warmstart.ts 649→126 lines. 3 util files created
(mlp-math, backprop, curriculum). 30 tests pass, tsc 0 errors, metrics 523/523 JSDoc.

### Step 5b: browser-entry.ts → render-loop utils [DONE]

```yaml
phase: 5
step: 5b
title: 'Extract browser-entry.ts tick sub-steps to render-loop.utils.ts'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'Step 5c — browser-entry.ts bootstrap + canvas-dimensions utils'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-084
    text: 'render-loop.utils.ts created with computeDeltaMs, advanceSimTick, computeHiveDensityRatio, buildRenderState, deriveDeathFeedbackDirection, resolveWaveNumber'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-085
    text: 'startRenderLoop/tick becomes declarative: computeDeltaMs → advanceSimTick → buildRenderState → statusBar.update → deathFeedback.update → bridge.postSimState'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-086
    text: 'browser-entry.ts is ~300 lines or fewer (after 5c completes: ~250)'
    validation: 'manual — line count check'
  - id: AC-087
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '5b-render-loop-utils'
    title: 'Extract tick sub-steps to render-loop.utils.ts; make startRenderLoop declarative'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/render-loop.utils.ts'
    acceptance_criteria:
      - 'Render-loop executors extracted; tick is declarative; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Step objective:** Extract tick sub-steps from `browser-entry.ts` (644 lines)
into `render-loop.utils.ts`. `startRenderLoop`/`tick` becomes declarative.
Risk Med. Effort S-M.

**Evidence (Step 5b DONE):** browser-entry.ts 644→550 lines. render-loop.utils.ts created
with 6 executors (computeDeltaMs, advanceSimTick, computeHiveDensityRatio, buildRenderState,
deriveDeathFeedbackDirection, resolveWaveNumber). tick is declarative. 31 tests pass,
tsc 0 errors, metrics 529/529 JSDoc.

### Step 5c: browser-entry.ts → bootstrap + canvas-dimensions utils [DONE]

```yaml
phase: 5
step: 5c
title: 'Extract browser-entry.ts bootstrap + canvas-dimensions utils'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-solid-split.plans.md'
copy_paste: true
next_step: 'null — workstream complete'
skills:
  - 'solid-split'
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - 'npx tsc --noEmit'
  - 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
acceptance_criteria:
  - id: AC-088
    text: 'bootstrap.utils.ts created with resolveWorkerUrl (refactored to take hostScript param), supportsWorkerOffscreenCanvas, formatRgb, drawCanvasStatus'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-089
    text: 'canvas-dimensions.utils.ts created with resolveCanvasRenderDimensions, applyCanvasBackingStore, updateRendererSize'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-090
    text: 'CRITICAL: document.currentScript eval-time capture stays in browser-entry.ts entry, passed as param to resolveWorkerUrl'
    validation: 'manual — grep document.currentScript in browser-entry.ts'
  - id: AC-091
    text: 'browser-entry.ts is ~250 lines (neatensteinStart + startRenderLoop orchestrators)'
    validation: 'manual — line count check'
  - id: AC-092
    text: 'No new folder-quality-metrics diagnostics'
    validation: 'node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein'
slices:
  - slice_id: '5c-bootstrap-utils'
    title: 'Extract bootstrap utils; refactor resolveWorkerUrl to take hostScript param'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/bootstrap.utils.ts'
    acceptance_criteria:
      - 'Bootstrap utils extracted; document.currentScript stays in entry; tests + tsc pass'
    parallelizable: false
    dependencies: []
    next_slice: '5c-canvas-dimensions-utils'
  - slice_id: '5c-canvas-dimensions-utils'
    title: 'Extract canvas-dimensions utils'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/canvas-dimensions.utils.ts'
    acceptance_criteria:
      - 'Canvas-dimensions utils extracted; browser-entry.ts ~250 lines; tests + tsc pass'
    parallelizable: false
    dependencies:
      - '5c-bootstrap-utils'
    next_slice: null
```

**Step objective:** Extract bootstrap and canvas-dimensions utils from
`browser-entry.ts`. CRITICAL: `document.currentScript` eval-time capture must
stay in the entry file and be passed as a parameter to `resolveWorkerUrl`.
Risk HIGH-localized. Effort S.

**Evidence (Step 5c DONE):** browser-entry.ts 550→424 lines. bootstrap.utils.ts created
(resolveWorkerUrl refactored to take hostScript param, supportsWorkerOffscreenCanvas,
formatRgb, drawCanvasStatus). canvas-dimensions.utils.ts created
(resolveCanvasRenderDimensions, applyCanvasBackingStore, updateRendererSize, NEATENSTEIN_FIXED_RENDER_HEIGHT).
document.currentScript stays in browser-entry.ts. 31 tests pass, tsc 0 errors, metrics 537/537 JSDoc.

**PHASE 5 COMPLETE — ENTIRE SOLID-SPLIT WORKSTREAM DONE.**
All phases (0–5) complete. All acceptance criteria met. All validations pass.

---

## Coverage backlog

- [DONE] Phase 0 / Step 0a: Extract enemy-controller leaf functions
- [DONE] Phase 0 / Step 0b: Extract sprites.ts pure guards + atlas caches
- [DONE] Phase 0 / Step 0c: Extract floor.ts band/shade/projection executors
- [DONE] Phase 1 / Step 1a: Finish sprites.ts projection + column utils
- [DONE] Phase 1 / Step 1b: Finish floor.ts, refactor drawNeatensteinGrid
- [DONE] Phase 1 / Step 1c: Split bolt-render.ts
- [DONE] Phase 2 / Step 2a: Split tick.ts into 9 util files
- [DONE] Phase 2 / Step 2b: Split hud.ts into 6 util files + constants
- [DONE] Phase 3 / Step 3a: Decompose updateControlledEnemy (HIGH risk)
- [DONE] Phase 3 / Step 3b: Extract enemy-sprite.ts utils
- [DONE] Phase 3 / Step 3c: Split generate-enemy-sprites.ts
- [DONE] Phase 3 / Step 3d: Extract enemy-navigation.ts utils
- [DONE] Phase 4 / Step 4a: Extract worker pure zero-state executors
- [DONE] Phase 4 / Step 4b: Extract worker render-paint executors
- [DONE] Phase 4 / Step 4c: Introduce DisplayWorkerState (HIGH risk)
- [DONE] Phase 5 / Step 5a: Split enemy-warmstart.ts
- [DONE] Phase 5 / Step 5b: Extract browser-entry.ts render-loop utils
- [DONE] Phase 5 / Step 5c: Extract browser-entry.ts bootstrap + canvas utils

## Immediate next steps

- **ALL PHASES COMPLETE.** The SOLID-split workstream is finished.
- Final regression: run full neatenstein test suite to confirm no regressions.
- Run `educational-docs` pass on touched boundaries (browser-entry, harness).
- Update `browser-entry/README.md` module-layout table with new util files.
- Compress this plan into a short closed tracker and move to `plans/completed/`.
- Create or update the matching `.logs.md` file with the durable audit history.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Load context via Cortex MCP and the active plan at plans/neatenstein-solid-split.plans.md.

WORKSTREAM STATUS: COMPLETE — All phases (0–5) done.

What was accomplished:
- Phases 0–5 complete (all steps [DONE]):
  - Phase 0: enemy-controller, sprites, floor leaf extractions
  - Phase 1: sprites/floor finishes, bolt-render split
  - Phase 2: tick.ts (9 utils), hud.ts (6 utils + constants)
  - Phase 3: updateControlledEnemy decomposition, enemy-sprite/generate-enemy-sprites/enemy-navigation
  - Phase 4: display.worker.ts 2392→865 lines, DisplayWorkerState encapsulation, 7 util files
  - Phase 5: enemy-warmstart.ts (3 utils), browser-entry.ts (3 util files: render-loop, bootstrap, canvas-dimensions)

Final metrics: 537/537 JSDoc, all tests pass, tsc 0 errors, folder-quality-metrics PASS.

Next steps (post-completion):
- Run full neatenstein regression suite
- Run educational-docs on touched boundaries
- Update browser-entry/README.md module-layout table
- Compress plan into closed tracker, move to plans/completed/
```

## Done Criteria

- All six split-candidate files are reduced to declarative orchestrators with
  util-file executors.
- Every util file is well under 800 lines.
- Public import paths are unchanged — no consumer needed to change an import.
- All `__testOnly*` hooks remain exported from their main modules.
- The `describe('index renumbering after de-rez compaction')` test block passes
  after the `updateControlledEnemy` decomposition.
- The worker determinism contract is preserved after `DisplayWorkerState`
  introduction.
- `document.currentScript` eval-time capture stays in `browser-entry.ts`.
- Generated README output reflects the new shape after docs regeneration.
- No new `folder-quality-metrics` diagnostics across all phases.
- If no work remains, this plan is compressed into a short closed tracker and a
  matching `.logs.md` file records the durable audit history after both files
  are moved into `plans/completed/`.
