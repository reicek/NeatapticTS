# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [WIP] — Phase 1 [DONE] · Phase 2 [DONE] · Phase 3 [WIP] · Steps 01–10.2 [DONE] (compressed to logs) · Step 10.3 [PLANNED] · Step 11 [PLANNED] · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17 · **Next step:** Step 10.3 — render cap visual fixes, perf analysis, rAF clock
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/Neon_Shooter_NGE_Demo.research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

**Sprite rendering resolution mandate:** All robot/enemy sprites are authored and rendered at a logical resolution of **48×48 pixels**, then scaled 4× to 192×192 for display. The renderer, sprite projection, raycasting collision checks, and voxel calculations must operate on the 48×48 logical grid and only scale at the final blit. This preserves the reference artwork exactly while keeping CPU cost ~16× lower than native 192×192 per-pixel operations. The `examples/neatenstein/robot-sprite-data.js` module is the source-of-truth encoded sprite set (8 directions × 4 poses: `stand`, `walk1`, `walk2`, `shoot`). Walk cycle: `stand → walk1 → stand → walk2`. The `shoot` frame uses semitransparent muzzle-blast palette indices 7/8 so the blast can be overlaid on any walk frame with a natural glow; recoil and cannon pixels remain opaque. Art is user-approved and locked.

**Model mandate:** All dispatches under this plan use `glm-5.2:cloud`. No Chrome MCP / browser DevTools MCP — validation is jest-based only (no visible-browser smoke tests via MCP specialists).

---

## Current state

Steps 01–10.2 [DONE] — all step packets, fix packets, and validation evidence compressed to `plans/Neon_Shooter_NGE_Demo.logs.md`.
Step 10.3 [PLANNED] — render cap visual fixes (fog, enemy cull, floor/ceiling tunnel), perf analysis via DevTools trace, rAF clock.
Step 11 [PLANNED] — cannon overlay enhancement (red + impl slices done, green slice pending).

### Deferred items

- **10.2 perf observation 9** (performance trace analysis) — deferred from Step 10.2 fix-packet-10.2-cache-gaps-iteration-1. Addressed in Step 10.3 slice `10.3-perf-cleanup` using `plans/Trace-20260804T200934.json`.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Context: Neatenstein NGE Demo — Phase 1 [DONE], Phase 2 [DONE], Phase 3 [WIP]. Steps 01–10.2 [DONE] and compressed to plans/Neon_Shooter_NGE_Demo.logs.md. Step 10.3 [PLANNED] — render cap visual fixes, perf analysis, rAF clock. Step 11 [PLANNED] — cannon overlay (red + impl done, green pending). All agents use glm-5.2:cloud. No Chrome MCP — jest-based validation only.

Current boundary: Phase 3 active frontier is Step 10.3 — 5 slices [PLANNED]:
  1. 10.3-perf-cleanup — parse plans/Trace-20260804T200934.json for bottlenecks, delete 8 stale research-cache .md files
  2. 10.3-fog-cap-30 — fog fully closes at 30-cell render cap (use RENDER_DISTANCE_CAP not MAX_VIEW_DIST in fog factor)
  3. 10.3-cull-enemies-30 — cull enemy sprites past 30-cell render cap
  4. 10.3-floor-ceiling-cap — fix floor/ceiling tunneling past 30-cell render cap
  5. 10.3-raf-clock — rAF-driven FPS-scaled delta-time clock (replace fixed timestep + throttle)

Next narrow task: Dispatch 04-implementing (glm-5.2:cloud) for Step 10.3 slices. Start with 10.3-perf-cleanup (analysis + cleanup, no src logic changes), then proceed through fog → cull → floor/ceiling → rAF clock in order. Each slice has red-green TDD except perf-cleanup (green-only). After all slices, run 05-green-testing with focused jest suites and coverage guard.

Key codebase context:
  - NEATENSTEIN_MAX_VIEW_DIST=140 (framebuffer.ts:37) — soft fog denominator, too large for 30-cell cap
  - NEATENSTEIN_RENDER_DISTANCE_CAP=30 (framebuffer.ts:47) — hard render cap, DDA stops here
  - resolveWallFogFactor (walls.ts:117) uses perpWallDist / MAX_VIEW_DIST → fog only ~21% at 30 cells
  - Floor/ceiling: renderer/floor.ts draws grids past 30-cell cap → tunnel effect
  - Sprite cull: sprites.ts projectNeatensteinSprite needs cap enforcement for perpWallDist > 30
  - Clock: browser-entry.ts uses requestAnimationFrame with NEATENSTEIN_HOST_POST_INTERVAL_MS=33 throttle + NEATENSTEIN_FIXED_TIMESTEP_MS=16 fixed timestep → convert to FPS-scaled delta-time
  - 8 research cache files: examples/neatenstein/shooter-research-cache-{1..8}.md (stale, delete)

Required validations:
  - neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry
  - npm run lint
  - npx tsc --noEmit -p tsconfig.json

Known worktree cautions: Reference art PNGs in examples/neatenstein/ required by parity tests. examples/neatenstein/generated/ is shared/transient. Coverage scoped to touched files only. All agents use glm-5.2:cloud for this session. No Chrome MCP.
```

**Phase 3 status:**

- Step 01 — Tech-debt cleanup and test/coverage repair — [DONE] (see logs §Phase 3 Steps 01–09).
- Step 02 — Lint-type follow-up for Neatenstein tests — [DONE] (see logs).
- Step 03 — Center-screen DOOM-style plasma cannon — [DONE] (see logs).
- Step 04 — Plasma cannon visual cleanup and volt visibility fix — [DONE] (see logs).
- Step 05 — Enemy MLP evolution harness — [DONE] (see logs).
- Step 06 — Enemy voxel-sprite asset pipeline — [DONE] (see logs).
- Step 07 — Wire enemies into live renderer — [DONE] (see logs).
- Step 08 — Canvas sizing fix: fixed 480px height with aspect-ratio width — [DONE] (see logs).
- Step 09 — Bugfix: canvas horizontal stretch + missing enemy sprites — [DONE] (see logs).
- Step 10 — Replace enemy sprite renderer with encoded `robot-sprite-data.js` set, restore raycast scene — [DONE] (see logs §Phase 3 Step 10 final compression).
- Step 10.2 — FIX: 8 runtime issues from manual validation — [DONE] (user-approved; see logs §Phase 3 Step 10.2 final compression).
- Step 10.3 — Render cap visual fixes, perf analysis, rAF clock — [PLANNED].
- Step 11 — Enhance cannon overlay — [PLANNED] (red + impl slices done, green pending).

**Active frontier:** Step 10.3 [PLANNED] — 5 slices.

## Implementation phases

### Phase 1 — Arena + Hero FPS controls (game-director-owned) [DONE]

[DONE] Phase 1 complete. Full step packet and validation evidence archived in `plans/Neon_Shooter_NGE_Demo.logs.md`.

### Phase 2 — Raycast Renderer + WebGL VFX (visualizer-owned) [DONE]

[DONE] Phase 2 complete. Full step packet and validation evidence archived in `plans/Neon_Shooter_NGE_Demo.logs.md`.

### Phase 3 — Live Enemy Rendering + Polish (visualizer + benchmark-owned) [WIP]

Steps 01–10.2 [DONE] — compressed to `plans/Neon_Shooter_NGE_Demo.logs.md`.

#### Step 10.3: Render cap visual fixes, perf analysis, rAF clock [PLANNED]

**Step objective:** Fix four visual issues related to the 30-cell render distance cap, analyze performance via DevTools trace, delete stale research cache files, and convert the game clock to rAF-driven FPS-scaled delta-time. (1) Fog currently uses `NEATENSTEIN_MAX_VIEW_DIST=140` as the denominator, so at the 30-cell render cap fog is only ~21% — fix so fog reaches 100% at the cap. (2) Enemy sprites beyond 30 cells still render — cull them at the cap. (3) Floor/ceiling grids extend past the 30-cell wall cap creating a "tunnel" effect — clamp floor/ceiling projection to the cap. (4) Replace fixed-timestep + throttle clock with rAF-driven FPS-scaled delta-time for smoother simulation. (5) Parse `plans/Trace-20260804T200934.json` for performance bottlenecks. (6) Delete 8 stale `shooter-research-cache-{1..8}.md` files.

**Boundary notes:**

- `NEATENSTEIN_RENDER_DISTANCE_CAP=30` (framebuffer.ts:47) is the hard render cap. Fog, sprite culling, and floor/ceiling projection must all terminate at this distance.
- `NEATENSTEIN_MAX_VIEW_DIST=140` (framebuffer.ts:37) is the soft fog denominator. The fog fix changes the fog factor denominator to `RENDER_DISTANCE_CAP` so fog reaches 100% at the cap wall. `MAX_VIEW_DIST` may be removed if no longer referenced.
- The rAF clock slice changes the host→worker timing contract. The worker remains message-driven (postMessage). The host converts from fixed-timestep + throttle to delta-time-based posting with FPS-scaled simulation stepping.
- All agents use `glm-5.2:cloud`. No Chrome MCP — validation is jest-based only.
- No Deferred Cleanup: remove old fixed-timestep/throttle constants and code in the same slice that introduces the rAF clock.

**Step 10.3 packet:**

```yaml
phase: 3
step: 10.3
title: 'Render cap visual fixes, perf analysis, rAF clock'
status: '[PLANNED]'
goal: 'implementing'
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
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step-10.3 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
  - 'npm run lint'
  - 'npx tsc --noEmit -p tsconfig.json'
acceptance_criteria:
  - id: 'AC-10.3-001'
    text: 'Fog reaches 100% opacity at the 30-cell render distance cap, not at 140 cells.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/walls.test.ts'
  - id: 'AC-10.3-002'
    text: 'Enemy sprites beyond 30-cell render cap are not rendered.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
  - id: 'AC-10.3-003'
    text: 'Floor and ceiling rendering terminates at the 30-cell render cap — no tunnel effect past walls.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/floor.test.ts'
  - id: 'AC-10.3-004'
    text: 'Game clock uses rAF-driven FPS-scaled delta-time instead of fixed timestep + throttle interval.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/browser-entry.test.ts'
  - id: 'AC-10.3-005'
    text: 'Performance trace analyzed and findings recorded in plan; 8 stale research-cache .md files deleted.'
    validation: 'Code inspection + file existence check'
  - id: 'AC-10.3-006'
    text: 'All touched source files build, lint, and have 100% coverage.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
constitution_check:
  - 'principle-3-verbatim-binding'
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '10.3-perf-cleanup'
    title: 'Perf analysis from DevTools trace + delete 8 stale research-cache .md files'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'plans/Neon_Shooter_NGE_Demo.plans.md'
      - 'examples/neatenstein/shooter-research-cache-1.md'
      - 'examples/neatenstein/shooter-research-cache-2.md'
      - 'examples/neatenstein/shooter-research-cache-3.md'
      - 'examples/neatenstein/shooter-research-cache-4.md'
      - 'examples/neatenstein/shooter-research-cache-5.md'
      - 'examples/neatenstein/shooter-research-cache-6.md'
      - 'examples/neatenstein/shooter-research-cache-7.md'
      - 'examples/neatenstein/shooter-research-cache-8.md'
    acceptance_criteria:
      - id: 'AC-10.3a-001'
        text: 'DevTools trace plans/Trace-20260804T200934.json is parsed and performance bottlenecks identified and recorded in the plan.'
        validation: 'Plan contains perf analysis findings section'
      - id: 'AC-10.3a-002'
        text: 'All 8 shooter-research-cache-{1..8}.md files are deleted from examples/neatenstein/.'
        validation: 'File existence check — none of the 8 files exist after slice'
      - id: 'AC-10.3a-003'
        text: 'No jest tests break from research-cache deletion.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein'
    parallelizable: false
    dependencies: []
    next_slice: '10.3-fog-cap-30'
  - slice_id: '10.3-fog-cap-30'
    title: 'Fix fog to fully close at 30-cell render distance cap'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/framebuffer.ts'
      - 'examples/neatenstein/browser-entry/renderer/walls.ts'
      - 'examples/neatenstein/browser-entry/renderer/walls.test.ts'
    acceptance_criteria:
      - id: 'AC-10.3b-001'
        text: 'resolveWallFogFactor uses NEATENSTEIN_RENDER_DISTANCE_CAP (30) as fog denominator so fog reaches 100% at the cap wall.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/walls.test.ts'
      - id: 'AC-10.3b-002'
        text: 'NEATENSTEIN_MAX_VIEW_DIST constant removed if no longer referenced (No Deferred Cleanup).'
        validation: 'npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-10.3b-003'
        text: '100% coverage on touched files.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/walls'
    parallelizable: false
    dependencies:
      - '10.3-perf-cleanup'
    next_slice: '10.3-cull-enemies-30'
  - slice_id: '10.3-cull-enemies-30'
    title: 'Cull enemy sprites past 30-cell render distance cap'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    acceptance_criteria:
      - id: 'AC-10.3c-001'
        text: 'projectNeatensteinSprite returns null/empty for sprites with perpWallDist > NEATENSTEIN_RENDER_DISTANCE_CAP.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - id: 'AC-10.3c-002'
        text: 'Worker sprite loop skips sprites culled by cap distance.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
      - id: 'AC-10.3c-003'
        text: '100% coverage on touched files.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/sprites'
    parallelizable: false
    dependencies:
      - '10.3-fog-cap-30'
    next_slice: '10.3-floor-ceiling-cap'
  - slice_id: '10.3-floor-ceiling-cap'
    title: 'Fix floor/ceiling tunneling past 30-cell render distance cap'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/renderer/floor.test.ts'
    acceptance_criteria:
      - id: 'AC-10.3d-001'
        text: 'Floor and ceiling projection terminates at NEATENSTEIN_RENDER_DISTANCE_CAP — no pixels drawn past 30 cells.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/floor.test.ts'
      - id: 'AC-10.3d-002'
        text: 'No visible tunnel/gap between wall cap and floor/ceiling termination.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
      - id: 'AC-10.3d-003'
        text: '100% coverage on touched files.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/floor'
    parallelizable: false
    dependencies:
      - '10.3-cull-enemies-30'
    next_slice: '10.3-raf-clock'
  - slice_id: '10.3-raf-clock'
    title: 'rAF-driven FPS-scaled delta-time clock (replace fixed timestep + throttle)'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/constants.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - id: 'AC-10.3e-001'
        text: 'Host render loop uses rAF delta-time to drive simulation stepping with FPS-scaled timing instead of fixed NEATENSTEIN_FIXED_TIMESTEP_MS.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/browser-entry.test.ts'
      - id: 'AC-10.3e-002'
        text: 'NEATENSTEIN_HOST_POST_INTERVAL_MS throttle replaced with per-frame delta-time posting (or FPS-scaled cadence).'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/browser-entry.test.ts'
      - id: 'AC-10.3e-003'
        text: 'Old fixed-timestep and throttle constants removed (No Deferred Cleanup).'
        validation: 'npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-10.3e-004'
        text: '100% coverage on touched files.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/browser-entry'
    parallelizable: false
    dependencies:
      - '10.3-floor-ceiling-cap'
    next_slice: null
```

#### Step 11: Enhance cannon overlay — fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]

**Step objective:** Improve the center-screen plasma cannon drawn by `renderer/gun.ts`. Fix the gun-local horizontal stretch on ultra-wide displays by deriving `gunWidth` from `gunHeight * GUN_BODY_ASPECT_RATIO` instead of from viewport width. Add visual detail (barrel bands, side vents, top sight, energy-core rings) so the cannon reads as a weapon. Add real 3D voxel depth through a dedicated `renderer/gun-sprite.ts` projection helper that projects a small voxel grid into screen space, without reusing the enemy billboard renderer.

**Status note:** Step 11 is [PLANNED] while Step 10.3 is active. Step 11 will become active again after Step 10.3 is closed.

**Boundary notes:**

- Public API must remain unchanged: `renderGunOverlay(ctx, gun, width, height)` and `createInitialGunState()` keep their current signatures; `worker/display.worker.ts` and `host/game/types.ts` do not change.
- Do **not** modify `renderer/sprites.ts` or the enemy voxel pipeline. The new `gun-sprite.ts` may reuse the inverse-camera math conceptually, but it is a separate overlay projection with its own near-camera clipping rules.
- New color/geometry constants should stay local to the gun boundary; do not add global constants unless reviewed.
- The 3D/voxel sprite projection is required to resolve the reported lack of cannon depth. It is delivered by the `11-voxel-sprite` slice; if projection complexity exceeds that slice budget, a follow-up slice completes it before the step is marked [DONE].

**Step 11 packet:**

```yaml
phase: 3
step: 11
title: 'Enhance cannon overlay — fix horizontal stretch, add detail, voxel 3D look via sprite projection'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 4 Step 01 — NGE Main Agent + Enemy MLPs red tests [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 11 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/gun.ts,examples/neatenstein/browser-entry/renderer/gun.test.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  - 'npm run lint'
acceptance_criteria:
  - id: 'AC-11-001'
    text: 'Plasma cannon is no longer horizontally stretched on ultra-wide displays; gun width is derived from gun height and a fixed body aspect ratio.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun.test.ts'
  - id: 'AC-11-002'
    text: 'Cannon overlay includes at least three new detail elements (e.g., barrel bands, side vents, top sight, energy-core rings) drawn by renderGunOverlay.'
    validation: 'Visual inspection of examples/neatenstein/index.html and focused gun tests'
  - id: 'AC-11-003'
    text: 'A dedicated gun-sprite.ts helper exists for voxel/3D projection and can render a small voxel grid into the overlay with consistent proportions.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - id: 'AC-11-004'
    text: 'All touched source files build, lint, and have 100% coverage on changed renderer files.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '11-red-gun'
    title: 'Write red tests for aspect-correct sizing, detail drawing, and gun-sprite projection'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-11a-001'
        text: 'A failing assertion exists that gun width equals gun height times a fixed aspect ratio for at least two aspect ratios.'
      - id: 'AC-11a-002'
        text: 'A failing assertion exists that at least one new detail path is called (e.g., ctx.fillRect for a barrel band) for a standard aspect ratio.'
      - id: 'AC-11a-003'
        text: 'A failing assertion exists that gun-sprite.ts exports a projectGunSprite function and a red test expects a non-empty projected polygon/pixel list.'
    parallelizable: false
    dependencies: []
    next_slice: '11-aspect-detail'
  - slice_id: '11-aspect-detail'
    title: 'Fix horizontal stretch and add cannon detail in gun.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
    acceptance_criteria:
      - id: 'AC-11b-001'
        text: 'gunWidth is computed as gunHeight * GUN_BODY_ASPECT_RATIO and no longer depends directly on viewport width.'
      - id: 'AC-11b-002'
        text: 'At least three new detail elements are drawn (barrel bands, side vents, top sight, energy-core rings).'
      - id: 'AC-11b-003'
        text: 'Public API renderGunOverlay and createInitialGunState are unchanged.'
    parallelizable: false
    dependencies:
      - '11-red-gun'
    next_slice: '11-voxel-sprite'
  - slice_id: '11-voxel-sprite'
    title: 'Add dedicated gun-sprite.ts for voxel/3D projection'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-11c-001'
        text: 'New gun-sprite.ts exports a helper that projects a small voxel grid using inverse-camera / screen-space math (does not reuse renderer/sprites.ts).'
      - id: 'AC-11c-002'
        text: 'renderGunOverlay integrates the projected voxel sprite as a detail layer without changing its public signature.'
      - id: 'AC-11c-003'
        text: 'The projected gun sprite preserves consistent screen-space height and width proportions across 16:9 and ultra-wide aspect ratios.'
    parallelizable: false
    dependencies:
      - '11-aspect-detail'
    next_slice: '11-green'
  - slice_id: '11-green'
    title: 'Green validation: focused tests, build, lint, coverage guard, visible-browser smoke'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: 'AC-11d-001'
        text: 'Focused jest suites for gun and gun-sprite pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-11d-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry/renderer/.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-11d-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-11d-004'
        text: 'Visible-browser smoke test shows the cannon without horizontal stretch, with new details, and with a voxel/3D look.'
        validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '11-voxel-sprite'
    next_slice: null
```

**Validation evidence (Step 11 red + impl slices):**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/gun` → PASS (14 tests, 2 suites).
- `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry/renderer/gun` → PASS; `gun.ts` and `gun-sprite.ts` 100/100/100/100.
- `npx tsc --noEmit -p tsconfig.json` → PASS.
- `npm run lint` → PASS (0 issues).
- `npx prettier --check` on gun files → PASS.
- slice-advancement gate: PASS (7/7 sub-gates) for slices 11-aspect-detail, 11-voxel-sprite.
- specialist review (api-contract-reviewer): APPROVE.

### Phase 4 — NGE Main Agent + Enemy MLPs (core + benchmark-owned) [PLANNED]

**Goal:** Full NGE main agent lifecycle + weight-only MLP co-evolution.

[PLANNED] Step 01 — NGE Main Agent + Enemy MLPs red tests (deferred until phase becomes active).

- Main agent: full NGE lifecycle (Embryo→Juvenile→Adult→Reproducing), tier-capped topology up to tier limit.
- **All motifs are EXISTING in `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE`** — no new motifs, no schema version bump. Motifs used: `AttentionHead` (threat prioritization), `GatedRecurrentCell` (aim/strafe state), `EpisodicSlot` (spawn-pattern memory).
- MLP enemies: fixed topology, weight-only mutation, no structural assimilation.
- **Assimilation is INTERNAL to the main agent lifecycle** — writes back structural priors derived from the main agent's own equilibrium candidate. The MLP enemy is the SELECTION PRESSURE, not an assimilation source. No weights or structure flow from MLP to main via assimilation. Priors are weak/decaying (defends against catastrophic forgetting).
- **Reproduction mode policy:** an external overlay that SELECTS a mode then writes the canonical `NgeReproductionPolicy.mode` field (only when `modeIsEvolvable: true`). Named `reproductionModeHysteresis` (distinct from `NgeHysteresisState` juvenile grow gate). Window: 3 generations, majority-vote. Mode selection: parthenogenesis (dominating) → polyandric (struggling) → sexual (stalemate).
- **New core-side primitives (core-owned):**
  - (a) Deterministic per-enemy substrate coordinate allocator for `WeightSharedCohort`: emits `NeatGenomeSubstrateCoordinate` within `NgeSubstrateConfig` (dimensions: 3, normalization: 'unit-cube'), produces stable `zoneId`s via existing zone-partition. Reproducible from `(swarmSize, enemyIndex, seed)` alone, no runtime allocation order dependency.
  - (b) Combat-pressure → reproduction-mode policy (inspectable, tested, in `src/neat/nge-evolution/`).

**Acceptance:**

- ARMS RACE mode runs at interactive rates.
- Main fitness computed against MLP snapshot, not live MLP.
- Assimilation writes internal priors, not enemy-derived weights/structure.
- Reproduction mode switches with `reproductionModeHysteresis` (3-gen window).
- Coordinate allocator: repeated-build hash test (same swarmSize + seed → identical coordinate set, stable ordering, unit-cube conformant).
- 100% coverage on touched `src/` files via `coverage-guard`.

### Phase 5 — SWARM Mode (core + benchmark-owned) [PLANNED]

**Goal:** WeightSharedCohort swarm + HIVE DENSITY legibility.

[PLANNED] Step 01 — SWARM Mode red tests (deferred until phase becomes active).

- WeightSharedCohort: one DNA, shared weight tensor, per-enemy coordinate injection (`receivesCoordinates: true`). Swarm motifs (all existing): `DenseFeedForward` (perception), `GatedRecurrentCell` (pursuit/evasion state), `ModulatorBroadcaster` (cohort alarm), `GatingRouter` (pursuit-vs-evasion switch), `EpisodicSlot` (hero position memory).
- Swarm fitness = collective damage + collective survival (one scalar). Swarm reproduces as one individual.
- **Full `NgeReproductionPolicy` for swarm:** `mode: 'parthenogenesis'`, `modeIsEvolvable: false` (size-ramped via density, not mode-switched), `parthenogenesisMutationRate: 0.1` (configurable via demo prop).
- **No hardcoded roles:** roles (if any emerge) are READ from coordinate injection, not hardwired by archetype. Ablation: coordinate-shuffle verifies role emergence is learned (shuffle coordinates → behavior should change).
- **HIVE DENSITY meter:** normalized 0–1 coordination budget (NOT headcount). Thresholds at 0.25/0.50/0.75/1.0: brighten → formation → flanking → lockstep single-organism. Swarm size stays ≤8; density = coordination quality. 100% = lockstep movement (single organism), not clustering. Thresholds survive any cap change (8→6).
- SWARM snapshot refresh: every 3 generations (explicit).

**Acceptance:**

- One DNA + shared weights + coordinate injection produces differentiated swarm behavior (focused test on coordinate-injection effect).
- Swarm fitness scalar; SWARM barrier deterministic.
- HIVE DENSITY (normalized 0–1) correlates with coordination behavior change.
- Coordinate-shuffle ablation: shuffling coordinates changes behavior (roles are learned, not hardcoded).

### Phase 6 — Human Modes + Replay Buffer (benchmark + game-director-owned) [PLANNED]

**Goal:** Replay-based per-death evolution + death feedback loop.

[PLANNED] Step 01 — Human Modes + Replay Buffer red tests (deferred until phase becomes active).

**Design pillars:**

- Human play as a mode: player death creates a replay entry, and that replay becomes selection pressure for the next enemy generation.
- Replay buffer: stores death contexts (hero pose, enemy state, damage source) for batch evaluation.
- Per-death evolution: each player death triggers a focused evolution pulse against the replay context.
- Death feedback loop: enemies visibly adapt to player tendencies within a session.
- No separate `src/` structural changes; leverages existing NGE lifecycle and MLP/Swarm harnesses.

**Acceptance:**

- Human mode is selectable from the demo UI.
- Player deaths are recorded in the replay buffer.
- Enemies show measurable adaptation to repeated player strategies within a single session.
- 100% coverage on touched `examples/neatenstein` files.

## Validation gates

- slice-advancement
- stale-wip-plans
- log-completion-marker
- phase-compression

---

## Consensus Record

| Round          | NGE Core        | NGE Benchmark   | Visualizer      | Game Director  |
| -------------- | --------------- | --------------- | --------------- | -------------- |
| 1 (propose)    | proposed        | proposed        | proposed        | proposed       |
| 2 (review)     | 10 observations | 10 observations | 11 observations | 9 observations |
| 3 (approve v2) | **APPROVED**    | **APPROVED**    | **APPROVED**    | **APPROVED**   |

## Design notes (high-level, retained)

- All evolution is headless/batch; visible enemies are rendered snapshots of the current population, not live training runs.
- No new core-side genome motifs until Phase 4; Phases 1–3 use existing `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE` motifs only.
- Demo scope: single arena, deterministic procedural wall grid, hero FPS controls, raycast renderer, WebGL overlay for voxel sprites.
- Coverage guard is scoped to files touched by the active step; full `src/` 100% coverage is deferred.