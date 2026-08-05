# Neatenstein NGE Demo — Workstream Log

**Status:** [WIP] — Steps 01–10.2 compressed; Step 10.3 active in plan

**Source plan:** plans/Neon_Shooter_NGE_Demo.plans.md

---

## Phase 3 Step 10 final compression

#### Step 10: Replace failed enemy sprite renderer — encoded `examples/neatenstein/robot-sprite-data.js` pipeline, correct projection, restored raycast scene, no full-canvas framebuffer copy [DONE]

**Step objective:** Replace the failed enemy sprite renderer with the approved encoded sprite pipeline from `examples/neatenstein/robot-sprite-data.js`. (1) The runtime sprite renderer in `renderer/sprites.ts` draws flat neon vertical bars and/or samples the procedural voxel generator from Step 06; it must instead read the 8-direction × 4-pose encoded frames (`stand`, `walk1`, `walk2`, `shoot`) from `examples/neatenstein/robot-sprite-data.js`, operate on the 48×48 logical grid, and scale 4× only at final blit so enemies show animated robot sprites at the correct resolution. The `shoot` pose combines with lower-body walk frames using palette indices 7/8 for the semitransparent muzzle blast. (2) Correct the sprite projection formula so screen scale is derived from a constant focal length (`canvasHeight / perpDist * worldSize`) with horizontal/vertical consistency, not the old oversized `canvasHeight / transformY` style scaling. (3) Restore walls/floor/ceiling raycast rendering in the worker render loop and eliminate the full-canvas `getImageData`/`putImageData` copy by writing the entire scene into a persistent `Uint8ClampedArray` framebuffer that is committed once per frame.

**Boundary notes:**

- `examples/neatenstein/robot-sprite-data.js` is the source of truth for enemy sprite frames: `ROBOT_SPRITE_FRAMES` contains 8 directions × 4 poses (`stand`, `walk1`, `walk2`, `shoot`) on a 48×48 logical grid, `ROBOT_SPRITE_SCALE = 4` is applied only at final blit, and `ROBOT_SPRITE_PALETTE` uses indices 7/8 for the 50%-alpha muzzle blast. The runtime renderer must import this module directly and must not duplicate voxel generation logic or depend on `examples/neatenstein/generated/` or `robot-proposal-192*.png` reference frames.
- Public API of `renderer/sprites.ts` should remain stable where possible; `worker/display.worker.ts` still calls `renderNeatensteinSprite` and `clipNeatensteinSprite`, but their internals change to use the encoded sprite set.
- The z-buffer helpers in `renderer/sprites.ts` should be reused/extended rather than duplicated.
- Sprite math (projection, raycasting collision checks, per-pixel sampling) must run on the 48×48 logical grid and only scale 4× at the final canvas output.
- The worker must remain testable under Node/Jest with mocked `CanvasRenderingContext2D` / `OffscreenCanvasRenderingContext2D`; avoid browser-only APIs in the hot path.
- Dead code from the neon-bar renderer, the procedural voxel snapshot path, and the full-canvas snapshot must be removed in the same step that introduces the replacement (No Deferred Cleanup Policy).

**Step 10 packet:**

```yaml
phase: 3
step: 10
title: 'Replace enemy sprite renderer with encoded robot sprite set and restore raycast scene'
status: '[DONE]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 11 — Enhance cannon overlay: fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 10 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/robot-sprite-data.js,examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  - 'npm run lint'
acceptance_criteria:
  - id: 'AC-10-001'
    text: 'Enemy sprites render using the approved examples/neatenstein/robot-sprite-data.js encoded sprite set (8 directions × 4 poses on a 48×48 logical grid, final blit scaled 4× to 192×192), not flat neon bars or procedural voxel snapshots.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
  - id: 'AC-10-002'
    text: 'Raycast walls, floor, and ceiling are rendered into the scene before sprites so sprites appear correctly occluded and grounded.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  - id: 'AC-10-003'
    text: 'Sprite projection scale derives from the 48×48 logical sprite grid and correct camera transform; it is not oversized or based on canvasWidth / transformY hacks.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
  - id: 'AC-10-004'
    text: 'The render loop does not call getImageData or putImageData on the full canvas per frame; it writes to a persistent Uint8ClampedArray framebuffer and blits once.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  - id: 'AC-10-005'
    text: 'Dead neon-bar renderer code, procedural voxel snapshot imports/paths, and full-canvas snapshot helpers are removed in the same slices that add the replacement (No Deferred Cleanup Policy); touched source files build, lint, and have 100% coverage.'
    validation: 'npm run lint; npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry'
constitution_check:
  - 'principle-3-verbatim-binding'
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '10-red-encoded-sprite'
    title: 'Write red tests for encoded sprite rendering, projection, and framebuffer copy bugs'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10a-001'
        text: 'A failing assertion exists that renderNeatensteinSprite samples ROBOT_SPRITE_FRAMES pixel data (imported from examples/neatenstein/robot-sprite-data.js) instead of drawing a flat color bar.'
      - id: 'AC-10a-002'
        text: 'A failing assertion exists that projected sprite scale is computed from the 48×48 logical grid and proportional to canvasHeight / perpDist with the correct focal length.'
      - id: 'AC-10a-003'
        text: 'A failing assertion exists that the worker render loop does not call getImageData or putImageData on the full canvas during the sprite/wall pass.'
      - id: 'AC-10a-004'
        text: 'A failing assertion exists that walls/floor/ceiling raycast pixels are written to a persistent Uint8ClampedArray framebuffer before sprites are drawn.'
    parallelizable: false
    dependencies: []
    next_slice: '10-impl-encoded-sprite'
  - slice_id: '10-impl-encoded-sprite'
    title: 'Wire runtime sprite renderer to examples/neatenstein/robot-sprite-data.js and remove dead neon-bar/voxel code'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    acceptance_criteria:
      - id: 'AC-10b-001'
        text: 'sprites.ts imports ROBOT_SPRITE_FRAMES, ROBOT_SPRITE_SCALE, and ROBOT_SPRITE_PALETTE from examples/neatenstein/robot-sprite-data.js and decodes palette indices into RGBA.'
      - id: 'AC-10b-002'
        text: 'Flat neon-bar drawing code (renderNeatensteinSpriteColumnRgb and related helpers) and any procedural voxel snapshot imports/builders are deleted.'
      - id: 'AC-10b-003'
        text: 'Encoded sprite pixels are written to the framebuffer with z-buffer occlusion; muzzle-blast palette indices 7/8 produce semitransparent red/white pixels.'
    parallelizable: false
    dependencies:
      - '10-red-encoded-sprite'
    next_slice: '10-impl-projection'
  - slice_id: '10-impl-projection'
    title: 'Correct sprite projection from the 48×48 logical grid'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    acceptance_criteria:
      - id: 'AC-10c-001'
        text: 'projectNeatensteinSprite computes screen position and size from logical 48×48 sprite dimensions, applying ROBOT_SPRITE_SCALE = 4 only at final blit.'
      - id: 'AC-10c-002'
        text: 'Sprite screen size is verified against a known camera distance and does not overshoot the viewport.'
      - id: 'AC-10c-003'
        text: 'Direction selection (0..7) is exercised for the 8 encoded directions; pose selection cycles stand/walk1/walk2/shoot.'
    parallelizable: false
    dependencies:
      - '10-impl-encoded-sprite'
    next_slice: '10-impl-framebuffer'
  - slice_id: '10-impl-framebuffer'
    title: 'Restore raycast scene and eliminate full-canvas getImageData/putImageData'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10d-001'
        text: 'The worker allocates a persistent Uint8ClampedArray framebuffer and renders walls, floor, and ceiling before drawing sprites.'
      - id: 'AC-10d-002'
        text: 'putImageData is called at most once per frame to blit the persistent framebuffer; no getImageData call remains in the per-frame render path.'
      - id: 'AC-10d-003'
        text: 'Old full-canvas snapshot helpers and any dead raycast stubs are deleted in the same slice.'
    parallelizable: false
    dependencies:
      - '10-impl-projection'
    next_slice: '10-green'
  - slice_id: '10-green'
    title: 'Green validation: focused tests, build, lint, coverage guard, visible-browser smoke'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'coverage/lcov.info'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10e-001'
        text: 'Focused jest suites for sprites and worker pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry'
      - id: 'AC-10e-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry'
      - id: 'AC-10e-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-10e-004'
        text: 'Visible-browser smoke shows encoded robot sprites at correct size with walls/floor/ceiling visible and maintains smooth FPS (no full-canvas copy).'
        validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html; capture browserVisibility: visible-foreground evidence and approximate FPS.'
    parallelizable: false
    dependencies:
      - '10-impl-framebuffer'
    next_slice: null
```

**Validation evidence:**

- 2026-08-01: RED phase complete for slice `10-red-encoded-sprite`. `03-red-testing` authored four owner-local failing tests across `examples/neatenstein/browser-entry/renderer/sprites.test.ts` and `examples/neatenstein/browser-entry/worker/display.worker.test.ts`:
  - `samples ROBOT_SPRITE_FRAMES pixel data instead of drawing a flat color bar` → fails: `TypeError: Cannot read properties of undefined (reading 'NaN')` (renderer expects `VoxelSnapshot`, not encoded `number[][]` frame).
  - `computes projected sprite scale from the 48×48 logical grid` → fails: actual 13.85641 vs expected 27.71281 (world size 0.5 → needs 1.0 for 48×48 grid).
  - `worker sprite pass uses encoded robot frames without reading the canvas back` → fails: no encoded frame passed to `renderNeatensteinSprite`.
  - `flushes encoded robot sprite pixels from the persistent framebuffer` → fails: output never contains encoded robot red `[221,34,0,255]`.
  - Consolidated `slice-advancement` gate: PASS (TRIVIAL severity; red phase changed only test files).
  - `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → PASS (0 stale WIP plans).
  - fix-loop: 10-red-encoded-sprite iteration 1 status=passed
- 2026-08-03: Step 10 packet revised to use the approved encoded sprite set in `examples/neatenstein/robot-sprite-data.js` (relocated from `plans/`). The previous implementation evidence (tsc, lint, prettier, 80 focused tests, 100% coverage on `sprites.ts`/`display.worker.ts`, and per-slice `slice-advancement` passes for the old voxel pipeline slices) is superseded because the source of truth changed and the implementation slices have been reset to `[PLANNED]`.
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=Step-10 --changed-files=plans/Neon_Shooter_NGE_Demo.plans.md` → PASS (plan-only, TRIVIAL severity; 4/4 sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint).
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → PASS (0 stale plans).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md` → PASS (0 errors, 0 warnings).
- Source-level validations (tsc, lint, focused jest, coverage guard, visible-browser smoke) are pending execution-phase work and are recorded under the new red/implement/green slices.

### PlanUpdate for Step 10 packet revision to encoded `robot-sprite-data.js`

```yaml
PlanUpdate:
  boundary: 'Phase 3 / Step 10 packet revision'
  status: '[DONE]'
  what_changed:
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — revised Step 10 title, objective, boundary notes, and YAML packet to use examples/neatenstein/robot-sprite-data.js as source-of-truth encoded sprite set (8 directions × 4 poses, 48×48 logical grid, ROBOT_SPRITE_SCALE = 4, semitransparent muzzle-blast palette indices 7/8)'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — reset Step 10 implementation slices to [PLANNED]; red slice 10-red-encoded-sprite is [WIP]'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — updated Current state, Handoff query, Phase 3 status line, active frontier, and top-level status line'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — archived old voxel-pipeline validation evidence as superseded'
    - 'plans/README.md — created to register active WIP plan'
    - 'plans/Roadmap.md — created to register active WIP plan'
  evidence:
    - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=Step-10 --changed-files=plans/Neon_Shooter_NGE_Demo.plans.md → pass (4/4 structural sub-gates)'
    - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json → pass (0 stale WIP plans)'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md → pass (0 errors, 0 warnings)'
  removals:
    - 'Old Step 10 implementation slices (10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt) are superseded and reset to [PLANNED] under new IDs (10-impl-projection, 10-impl-encoded-sprite, 10-impl-framebuffer). Old voxel-pipeline validation evidence is archived as superseded.'
  next_boundary: '03-red-testing slice 10-red-encoded-sprite — author red tests for encoded sprite rendering, projection, and framebuffer copy bugs'

## Step 10 fix slices (reopened boundary)

**Boundary:** Phase 3 / Step 10 reopened for wall/floor/ceiling visibility, performance, and enemy-sprite polish fixes. The slices below were compressed from the active plan after reaching [DONE]; details are preserved here so the active tracker stays lean.

```yaml
slices:
  - slice_id: '10-fix-walls-fog'
    title: 'Raise/fix view distance so perimeter walls stay visible as neon with fog falloff; remove debug red square'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/framebuffer.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-10a-001'
        text: 'NEATENSTEIN_MAX_VIEW_DIST is raised/derived so walls at ~60 cells are still visible with a fog falloff.'
        validation: 'Code inspection of renderer/framebuffer.ts and visible-browser smoke'
      - id: 'AC-10a-002'
        text: 'No hard distance clip makes perimeter walls vanish at spawn.'
        validation: 'Visible-browser smoke of examples/neatenstein/index.html'
      - id: 'AC-10a-003'
        text: 'Debug red square at display.worker.ts:767-768 is removed.'
        validation: 'Code inspection of examples/neatenstein/browser-entry/worker/display.worker.ts'
    parallelizable: false
    dependencies: []
    next_slice: '10-fix-floorceiling-neon'
  - slice_id: '10-fix-floorceiling-neon'
    title: 'Correct floor/ceiling colors to intended neon and cache static fill on resize'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-10b-001'
        text: 'Floor and ceiling colors match the intended neon palette.'
        validation: 'Visible-browser smoke of examples/neatenstein/index.html'
      - id: 'AC-10b-002'
        text: 'Static floor/ceiling fill is computed once on resize and reused each frame.'
        validation: 'Code inspection and performance profile of display.worker.ts'
    parallelizable: false
    dependencies:
      - '10-fix-walls-fog'
    next_slice: '10-fix-perf-throttle'
  - slice_id: '10-fix-perf-throttle'
    title: 'Throttle host render posting to ≤30fps, reuse ImageData, eliminate per-frame allocations'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-10c-001'
        text: 'Host posts new render state at no more than 30fps while still inside requestAnimationFrame.'
        validation: 'Code inspection of browser-entry.ts startRenderLoop and frame-time measurement'
      - id: 'AC-10c-002'
        text: 'Worker reuses one ImageData buffer instead of allocating a full-canvas buffer each frame.'
        validation: 'Code inspection of display.worker.ts buildAndPostFrame'
      - id: 'AC-10c-003'
        text: 'Initial load backlog is gone; first visible frame appears well under the previous ~33s.'
        validation: 'Visible-browser smoke timing of examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '10-fix-floorceiling-neon'
    next_slice: '10-fix-collision-walk'
```

---

## Phase 3 Steps 01–09 final compression

#### Step 01 — Tech-debt cleanup and test/coverage repair [DONE]

5 slices (01-red-tests, 01-coverage-repair, 01-lint-types, 01-dead-code, 01-green) all [DONE] and green validated. Fixed test harness imports, repaired coverage gaps in src/ architecture files, removed dead code paths, established 100% coverage baseline for touched files. Detailed evidence in prior plan revisions (now compressed).

#### Step 02 — Lint-type follow-up for Neatenstein tests [DONE]

3 slices (01-lint-types-harness, 01-lint-types-host-src, 01-lint-types-green) all [DONE] and green validated. Resolved TypeScript strict-mode lint errors across Neatenstein test and host source files. All lint and tsc checks pass clean.

#### Step 03 — Center-screen DOOM-style plasma cannon [DONE]

5 slices (02-red-tests, 02-constants-types, 02-gun-render, 02-bolt-combat, 02-render-integration) all [DONE] and green validated. Implemented plasma cannon overlay (renderer/gun.ts), bolt combat system (host/game/combat.ts, tick.ts), and integrated into worker render loop. Cannon renders center-screen with bolt projectile firing. Coverage 100% on touched files.

#### Step 04 — Plasma cannon visual cleanup and volt visibility fix [DONE]

5 slices (04-red-tests, 04-halo, 04-gun-shadow, 04-bolt, 04-green) all [DONE] and green validated. Added muzzle halo, gun shadow, bolt trail visibility, and volt energy effects. Fixed transparency and z-ordering for overlay elements.

#### Step 05 — Enemy MLP evolution harness [DONE]

5 slices (05-red-mlp, 05-mlp-topology, 05-enemy-fitness, 05-enemy-barrier, 05-green) all [DONE] and green validated. Built enemy MLP evolution harness with fixed-topology weight-only mutation, fitness evaluation against hero combat, and generational barriers. Integrated with NGE lifecycle from Phase 1.

#### Step 06 — Enemy voxel-sprite asset pipeline [DONE]

8 slices (06-red-voxel, 06-voxel-descriptor, 06-snapshot-renderer, 06-animator, 06-coverage-config, 06-sprite-sheet, 06-reference-parity, 06-green-final) all [DONE] and green validated. Created 48×48 logical grid voxel sprite pipeline with 8-direction × 4-pose encoded frames (stand, walk1, walk2, shoot). Reference artwork parity verified. Sprite data encoded in `examples/neatenstein/robot-sprite-data.js`.

#### Step 07 — Wire enemies into live renderer [DONE]

5 slices (07-red-renderer, 07-renderer-bridge, 07-enemy-controller, 07-enemy-render, 07-wave-loop) all [DONE] and green validated. Connected enemy evolution harness to live raycast renderer via sprite projection, enemy controller, and wave-based spawning. Enemies render as billboard sprites in the raycast scene with z-buffer occlusion.

#### Step 08 — Canvas sizing fix: fixed 480px height with aspect-ratio width [DONE]

Fixed canvas backing store to 480px height with CSS aspect-ratio controlling width. Eliminated ultra-wide stretch artifacts. Single slice, green validated with visible-browser smoke test.

#### Step 09 — Bugfix: canvas horizontal stretch + missing enemy sprites [DONE]

5 slices (09-canvas-backing, 09-render-state-enemies, 09-worker-controller, 09-worker-sprite-pass, 09-green) all [DONE] and green validated. Fixed canvas backing store dimensions, restored enemy sprite pass through worker message protocol, and verified with visible-browser smoke on ultra-wide. Fix-packet-09-green-iteration-2 (canvas resize after transfer) visible-browser smoke passed.

---

## Phase 3 Step 10.2 final compression

#### Step 10.2 — FIX: 8 runtime issues from manual validation of bundle v=20260802-7 [DONE]

**Step objective:** Fix 8 runtime issues identified during manual validation: (1) sprite facing/sort, (2) walk animation speed + render cap, (3) collision sync, (4) bolt AI spawn, (5) depth sort, (6) render cap enforcement, (7) AI behavior, (8) spawn positions. Additionally fix cache-validation gaps (cache 4: bolt collision bugs, cache 7: AI/spawn gaps) and 4 user-reported runtime issues (enemies walk through player, walk animation too fast, enemies visible beyond 30-cell cap, performance).

**User-approved [DONE]** — all implementation slices complete, all cache gaps addressed, preflight + targeted jest pass (207/207 tests).

**Step 10.2 packet:**

```yaml
phase: 3
step: 10.2
title: 'FIX: 8 runtime issues from manual validation of bundle v=20260802-7'
status: '[DONE]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 10.3 — Render cap visual fixes, perf analysis, rAF clock [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
  - 'npm run lint'
  - 'npx tsc --noEmit -p tsconfig.json'
acceptance_criteria:
  - id: 'AC-10.2-001'
    text: 'All 8 runtime issues from manual validation are fixed and verified via focused jest suites.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
  - id: 'AC-10.2-002'
    text: 'Cache 4 (bolt collision) and cache 7 (AI/spawn) gaps are addressed.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/(tick|episode|waves)'
  - id: 'AC-10.2-003'
    text: '271/271 tests pass across 9 suites with 100% coverage on touched src files.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
slices:
  - slice_id: '10.2-fix-sprite-facing-sort'
    title: 'Fix sprite facing direction and depth sort'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-10.2a-001'
        text: 'Sprite facing direction matches movement vector with 8-way facing.'
      - id: 'AC-10.2a-002'
        text: 'Depth sort renders far-to-near for correct occlusion.'
    parallelizable: false
    dependencies: []
    next_slice: '10.2-fix-walk-anim-rendercap'
  - slice_id: '10.2-fix-walk-anim-rendercap'
    title: 'Fix walk animation speed and render cap enforcement'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/scripts/enemy-controller.ts'
    acceptance_criteria:
      - id: 'AC-10.2b-001'
        text: 'Walk animation cycle reduced to ~1/4 previous speed.'
      - id: 'AC-10.2b-002'
        text: 'Render cap at 30 cells enforced for wall rendering.'
    parallelizable: false
    dependencies:
      - '10.2-fix-sprite-facing-sort'
    next_slice: '10.2-fix-collision-sync'
  - slice_id: '10.2-fix-collision-sync'
    title: 'Fix collision sync and minimum enemy distance'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/collision.ts'
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
    acceptance_criteria:
      - id: 'AC-10.2c-001'
        text: 'Enemies maintain minimum distance from player (no walking through player).'
      - id: 'AC-10.2c-002'
        text: 'Collision sync between sim and render state is consistent.'
    parallelizable: false
    dependencies:
      - '10.2-fix-walk-anim-rendercap'
    next_slice: '10.2-fix-bolt-ai-spawn'
  - slice_id: '10.2-fix-bolt-ai-spawn'
    title: 'Fix bolt collision damage, AI spawn, and wave advancement'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
      - 'examples/neatenstein/browser-entry/host/game/episode.ts'
      - 'examples/neatenstein/browser-entry/host/game/waves.ts'
    acceptance_criteria:
      - id: 'AC-10.2d-001'
        text: 'Bolt collision detects enemies from current position and applies damage.'
      - id: 'AC-10.2d-002'
        text: 'outOfTime terminal condition removed; generations not time-bound.'
      - id: 'AC-10.2d-003'
        text: 'advanceWave wired into game loop when allEnemiesCleared returns true.'
    parallelizable: false
    dependencies:
      - '10.2-fix-collision-sync'
    next_slice: '10.2-green-visiblebrowser'
  - slice_id: '10.2-green-visiblebrowser'
    title: 'Green validation: focused tests, coverage, build, lint'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: 'AC-10.2e-001'
        text: '271/271 tests pass across 9 suites.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
      - id: 'AC-10.2e-002'
        text: '100% coverage on touched src files.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
      - id: 'AC-10.2e-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
    parallelizable: false
    dependencies:
      - '10.2-fix-bolt-ai-spawn'
    next_slice: null
```

**Validation evidence summary:**

- 271/271 tests PASS across 9 suites (combat, tick, waves, episode, enemy-controller, movement, collision, display.worker, sprites)
- tsc: OK (no errors), eslint: 0 issues
- Coverage: 100% statements/functions/lines on all touched files; collision.ts 96.42% branches (line 158 = dead constant guard)
- 8-cache validation: 6 GREEN (caches 1,2,3,5,6,8), 1 PARTIAL (cache 7), 1 GAPS (cache 4) — all gaps addressed in fix-packet-10.2-cache-gaps-iteration-1
- Fix-packet-10.2-cache-gaps-iteration-1: 7/9 observations fixed, 1 deferred (perf → Step 10.3), preflight+targeted jest pass (207/207)
- slice-advancement gate: 6/7 sub-gates PASS (code-coverage sub-gate PASS after focused coverage run)
- Bundle rebuilt: docs/assets/neatenstein.bundle.js + .map, neatenstein.worker.js + .map, cache-buster v=20260802-7

**Deferred to Step 10.3:**
- Observation 9 (performance): Chrome DevTools performance trace dispatched; analysis deferred to Step 10.3 slice `10.3-perf-cleanup` using `plans/Trace-20260804T200934.json`.

