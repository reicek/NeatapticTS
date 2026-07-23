# Neatenstein NGE Demo — Phase Logs

**Status:** [DONE]

## Phase 1 — World & Renderer (visualizer-owned)

**Goal:** Raycasting neon renderer + frame protocol + audio.

### Step 01: World & Renderer scaffold and raycaster [DONE]

[DONE] All 13 Phase 1 slices completed and green validated.

- **01-red-phase1** [DONE]: Red tests for scaffold, frame protocol, raycaster, and audio contracts. 8 failed suites, 41 failed tests — all module-not-found/readme-not-found failures confirming no implementation existed yet.
- **01-scaffold** [DONE]: Created `examples/neatenstein/README.md`, `examples/neatenstein/browser-entry/constants.ts`, and `scripts/build-neatenstein.mjs` dual-entry esbuild stub. Green: `constants` 6/6 tests pass. Specialist review passed after README/build-stub fix cycle.
- **01-frame-protocol** [DONE]: Implemented `examples/neatenstein/browser-entry/renderer/frame.ts` with SoA typed arrays, transfer-list zero-copy, and incrementing `requestId`. Green: `frame` 5/5 tests pass.
- **01-raycaster-grid** [DONE]: Implemented `renderer/map.ts` (deterministic 24×24 wall grid) and `renderer/raycast.ts` (DDA ray cast). Green: `raycast` 5/5 tests pass.
- **01-neon-walls** [DONE]: Implemented `renderer/walls.ts` and `renderer/framebuffer.ts` for neon wall rendering + distance fog. Green: `walls` 4/4 tests pass.
- **01-floor-reuse** [DONE]: Implemented `renderer/floor.ts` reusing Flappy ground-grid helpers with camera-adapted yaw, runtime canvas dimensions, depth-aware alpha, and save/restore. Green: `floor` 11/11 tests pass after test-heuristic fix. Visible-browser smoke test passed.
- **01-sprites-zbuffer** [DONE]: Implemented `renderer/zbuffer.ts` and `renderer/sprites.ts` with per-column z-buffer occlusion. Green: `sprites` + `zbuffer` 19/19 tests pass. 100% coverage on both files.
- **01-pulse-system** [DONE]: Implemented `renderer/pulse.ts` with sim-tick emission, world-bearing continuity, event pulses, and z-buffer depth test. Green: `pulse` 6/6 tests pass. Browser smoke passed.
- **01-audio** [DONE]: Implemented `audio.ts` with 6 procedural WebAudio cues, `StereoPannerNode`, distance attenuation, and generation-up audio-visual pair. Green: `audio` 6/6 tests pass after mock and import-path fixes. Visible-browser smoke passed.
- **01-worker-offload** [DONE]: Implemented `worker/display.worker.ts` and `host/renderer-bridge.ts` with tier-gated render paths and OffscreenCanvas support. Green: `renderer-bridge` + `display.worker` 10/10 tests pass after `jest.resetModules()` test isolation fix. Visible-browser smoke passed.
- **01-interpolation-resize** [DONE]: Implemented `renderer/interpolate.ts` and `host/resize.ts`. Green: `interpolate` + `resize` 5/5 tests pass. Browser smoke deferred to `01-host-shell`.
- **01-host-shell** [DONE]: Created canonical `examples/neatenstein/index.html`, registered neatenstein in `scripts/copy-examples/copy-examples.definitions.ts`, and regenerated `docs/examples/neatenstein/index.html`. Green: `host-shell` 4/4 tests pass. Visible-browser smoke passed.
- **01-green-phase1** [DONE]: Full Phase 1 green validation and coverage guard. All 14 focused suites pass (81/81 tests). Lint passes. Visible-browser smoke at `docs/examples/neatenstein/index.html` passes with expected 404s for unbuilt bundle/worker assets.

**Plan gates:** plan-sync → pass; step-packet → pass; agent-graph → pass; specialist-review → pass; plan-slice-quality → pass; workflow-update-sync → pass; learning-event → pass; cortex-index → pass.

## Phase 2 — Game Logic & FPS State

**Status:** [DONE] — pending user manual browser review (EXPLICIT STOP before Phase 3).

**Goal:** FPS game state, controls, deterministic episode, hitscan combat, enemy waves.

### Completed slices

- **02-red-phase2** [DONE]: Red tests for all game domains. 10 test files authored, all failed for expected module-not-found reasons.
- **02-game-scaffold** [DONE]: host/game/ module layout (types.ts, state.ts, constants.ts). Deterministic reset with seedrandom. Green: state.test.ts + constants.test.ts (2 suites, 18 tests).
- **02-hero-state** [DONE]: Player health/ammo/dash with 200ms i-frames. Specialist review fixes: single-expect rule, applyDamage JSDoc, default-seed + contact-i-frame coverage tests. Green: state.test.ts passes.
- **02-enemy-waves** [DONE]: Continuous-trickle spawn, one per tick, 8-concurrent cap. Green: waves.test.ts passes.
- **02-controls** [DONE]: WASD (window-level listeners), mouse look, pointer lock, fire (pendingFire consume-on-read), arrow-key fallback, touch drag-to-look. Browser fix: keyboard listeners moved from canvas to window; mouseFireActive replaced with pendingFire flag. Green: controls.test.ts + input.test.ts pass.
- **02-projectiles** [DONE]: Hitscan neon beam, muzzle offset 0.2 cells, ImpactSpot on wall hits. Specialist fix: combat.test.ts pinned player.angleRad=0 for deterministic muzzle offset. Green: combat.test.ts passes.
- **02-collision** [DONE]: Wall-slide collision, contact damage with i-frames, normalized diagonal speed. Green: movement.test.ts (12 tests) + collision.test.ts (13 tests) + 15 suites/159 tests all pass. Browser smoke verified by user during iterative testing.
- **02-episode-loop** [DONE]: Episode lifecycle (15-25s), deterministic replay. Green: episode.test.ts + cadence.test.ts pass.
- **02-worker-game-sync** [DONE]: Game tick wired into display worker, renderer bridge forwarding. Green: tick.test.ts + display.worker.test.ts pass.
- **02-green-phase2** [DONE]: Full Phase 2 green validation. 43 suites, 354 tests all pass. Lint clean. Bundle rebuilt.

### Browser integration fixes (user-driven iterative testing)

During manual browser testing at http://localhost:8080/docs/examples/neatenstein/index.html, the user identified and drove fixes for:

1. **WASD not working** — Keyboard listeners on canvas (tabIndex=-1, not focused). Fix: attach to window.
2. **Fire not working** — mouseFireActive polled boolean lost quick clicks. Fix: pendingFire consume-on-read flag.
3. **Floor grid wrong** — Went through 4 rewrites: (a) yaw-rotated grid pointed to fixed horizon, (b) forced-perspective reticule, (c) per-pixel raycaster (dots + severe perf degradation at 115K+ iterations), (d) world-space line projection with double-stroke glow (~6.6K projections, batched strokes). Final: continuous glowing lines, depth-banded alpha.
4. **Floor-wall focal length mismatch** — Floor used (width/2)/tan(FOV/2) (~554px), walls used canvasHeight (360px). Fix: walls now use same wallFocalLength formula. Floor-wall aligned at horizon, same movement rate.
5. **Wall color** — Pink/magenta Y-side walls changed to dark neon blue {r:0, g:80, b:180} per user request. X-side walls remain cyan {r:0, g:191, b:255}.
6. **Projectiles not visible** — Tracer origin at depth 0 (near-clip rejects). Fix: muzzle offset 0.2 cells forward. Tracer never expired (permanent line). Fix: ageTracers helper in tick.ts decrements durationMs and filters expired.
7. **Pulse "flash" not traveling** — Static 8-column bars with bearing-tolerance fade (died in <0.5s). Rewritten as small dots traveling along integer grid lines with world-space velocity and lifetime-only fade. Max concurrent 11, interval 2000ms.
8. **Wall impact spots** — New feature: neon white spots at hit point, 3s lifetime fade, z-buffer depth test (<=). Initial scaling bug: radius * wallFocalLength / perpWallDist produced 4432px spots. Fix: radius / perpWallDist (4px at dist 1, 8px at dist 0.5, 2px at dist 2).
9. **Impact spot dynamic scaling** — Spots used stored perpWallDist (fixed size when player moves). Fix: recompute current perpendicular distance every frame using player's current position/angle. Re-project screen position using wall camera-plane math (lateral / (perpDist * planeScale)).
10. **Neon white colors** — Tracer and impact spots changed from #00bfff (cyan) to #f0f8ff (neon white) per user request.

### Key files modified during Phase 2

- `examples/neatenstein/browser-entry/host/game/types.ts` — Added ImpactSpot type.
- `examples/neatenstein/browser-entry/host/game/state.ts` — createGameState with impacts[], NEATENSTEIN_DEFAULT_SEED.
- `examples/neatenstein/browser-entry/host/game/combat.ts` — fireNeonBeam with muzzle offset, ImpactSpot creation.
- `examples/neatenstein/browser-entry/host/game/tick.ts` — ageTracers + ageImpacts helpers (immutable map+filter).
- `examples/neatenstein/browser-entry/host/game/movement.ts` — Wall-slide collision, normalized diagonal.
- `examples/neatenstein/browser-entry/host/game/collision.ts` — Contact damage with i-frames.
- `examples/neatenstein/browser-entry/host/game/controls.ts` — WASD, mouse look, pointer lock, fire.
- `examples/neatenstein/browser-entry/host/input.ts` — Window-level keyboard listeners, pendingFire flag.
- `examples/neatenstein/browser-entry/renderer/floor.ts` — World-space line projection with double-stroke glow.
- `examples/neatenstein/browser-entry/renderer/pulse.ts` — Traveling dots on grid lines, lifetime-only fade.
- `examples/neatenstein/browser-entry/worker/display.worker.ts` — Wall rendering (wallFocalLength), floor grid, pulse dots, tracer drawing, impact spots with dynamic scaling.
- `examples/neatenstein/browser-entry/constants.ts` — All renderer/game constants including neon white colors, pulse params, impact spot params.
- `docs/assets/neatenstein.bundle.js` (14.7 KB) + `docs/assets/neatenstein.worker.esm.js` (22.3 KB) — Rebuilt multiple times.

### Final validation

- 43 Jest suites, 354 tests — ALL PASS (32s).
- Lint: 0 errors, 113 pre-existing warnings (all in unrelated files).
- tsc: PASS.
- quality:folder: 184/184 JSDoc symbols documented.
- Bundle: 14.7 KB + 22.3 KB, build succeeds.
- Specialist reviews: ALL APPROVE (runtime, impl, viz across multiple review rounds).

### Step 01 YAML packet (archived)

```yaml
phase: 2
step: 1
title: 'Game Logic & FPS State red tests and implementation slices'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 3 Step 01 — Asymmetric Co-evolution Harness red tests'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/host/game'
  - 'npm run lint'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
acceptance_criteria:
  - id: AC-201
    text: 'The game-loop module lives under examples/neatenstein/browser-entry/host/game/ and exports a deterministic tick function that advances world state by one fixed timestep given an input snapshot'
  - id: AC-202
    text: 'Player health and ammo are initialized to documented constants, never go negative, never exceed their maximums, firing decrements ammo by exactly one, and damage events decrement health by the attackers configured damage value'
  - id: AC-203
    text: 'Enemy waves spawn as a continuous trickle with at most one new enemy per spawn tick, no simultaneous clumps, and the active enemy count is capped at 8 concurrent enemies'
  - id: AC-204
    text: 'The neon beam is hitscan; it intersects the nearest enemy or wall along the view center, applies damage only to the struck enemy, and renders a visible tracer in the frame it was fired'
  - id: AC-205
    text: 'WASD translates the player in world space, wall collision prevents entering solid map cells, and diagonal movement is normalized so combined keys do not increase speed'
  - id: AC-206
    text: 'Clicking the canvas requests pointer lock with unadjustedMovement: true; mouse deltas rotate the camera; arrow keys provide look fallback when pointer lock is unavailable; touch drag-to-look works on iOS Safari; on the Worker tier mouse deltas are forwarded to the display worker via postMessage'
  - id: AC-207
    text: 'Pressing Space triggers a dash that grants exactly 200 ms of invulnerability; the player takes zero damage during that window and cannot immediately re-dash (observable cooldown prevents indefinite chaining)'
  - id: AC-208
    text: 'A default episode ends within 15-25 seconds, and replaying the same seed with the same deterministic input sequence produces identical final health, ammo, kill count, and enemy roster'
  - id: AC-209
    text: 'The fixed-timestep game loop design supports a minimum cadence of at least 2 generations per minute in AI modes (episode length plus evaluation overhead stays ≤ 30 s per generation under default settings)'
  - id: AC-210
    text: 'Only the neon beam weapon exists; no weapon switching logic, state, or UI is added; left-click always fires the beam when ammo is greater than zero'
  - id: AC-215
    text: 'Red tests for all game domains exist and fail before implementation for the expected module-not-found or contract-missing reasons'
  - id: AC-216
    text: 'All Phase 2 focused Jest suites pass and npm run lint is clean for changed source files'
  - id: AC-217
    text: 'Visible-browser smoke test passes for movement, look, fire, dash, and touch look'
slices:
  - slice_id: '02-red-phase2'
    title: 'Red tests for game logic contracts'
    status: '[DONE]'
  - slice_id: '02-game-scaffold'
    title: 'Create host/game/ module layout and deterministic reset'
    status: '[DONE]'
  - slice_id: '02-hero-state'
    title: 'Player health, ammo, and dash invulnerability'
    status: '[DONE]'
  - slice_id: '02-enemy-waves'
    title: 'Continuous-trickle enemy spawn with 8-concurrent cap'
    status: '[DONE]'
  - slice_id: '02-controls'
    title: 'WASD, mouse look, pointer lock, fire, and worker-tier input forwarding'
    status: '[DONE]'
  - slice_id: '02-projectiles'
    title: 'Hitscan neon beam weapon and tracers'
    status: '[DONE]'
  - slice_id: '02-collision'
    title: 'Player/enemy movement, wall collision, and contact damage'
    status: '[DONE]'
  - slice_id: '02-episode-loop'
    title: 'Deterministic episode lifecycle and generation cadence'
    status: '[DONE]'
  - slice_id: '02-worker-game-sync'
    title: 'Wire game tick into the display worker and renderer bridge'
    status: '[DONE]'
  - slice_id: '02-green-phase2'
    title: 'Phase 2 green validation, lint, and visible-browser smoke'
    status: '[DONE]'
```
