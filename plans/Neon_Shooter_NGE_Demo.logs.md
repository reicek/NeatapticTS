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

### Phase 1 follow-up: halve vertical wall stripe width [DONE]

**Date:** 2026-07-24

**Goal:** Double tier column counts to halve vertical wall stripe width.

**Files changed:**

- `examples/neatenstein/browser-entry/constants.ts` — GPU 640, Worker 480, CPU 320 (was 320 / 240 / 160)
- `examples/neatenstein/browser-entry/host/resize.ts` — updated JSDoc CPU column-stride example (2 at 640px canvas, was 4)
- `examples/neatenstein/browser-entry/constants.test.ts`
- `examples/neatenstein/browser-entry/host/resize.test.ts`
- `examples/neatenstein/browser-entry/renderer/frame.test.ts`
- `examples/neatenstein/browser-entry/renderer/pulse.test.ts`
- `plans/Neon_Shooter_NGE_Demo.plans.md` — Phase 1 summary and risk note updated to new counts

**Validation evidence:**

- Focused Jest: 4 suites / 26 tests pass.
- Broad Neatenstein Jest: 44 suites / 380 tests pass.
- `npx tsc --noEmit -p tsconfig.json` — pass.
- `npx eslint examples/neatenstein/browser-entry/constants.ts` — pass.
- `npx prettier --check examples/neatenstein/browser-entry/constants.ts` — pass.
- `node scripts/build-neatenstein.mjs` — pass.
- Visible-browser smoke test — pass (canvas 3376×1235, `window.neatensteinStart` callable, no runtime JS errors except `/favicon.ico` 404, `browserVisibility: visible-foreground`).

**Notes:** Tier-to-column mappings in `display.worker.ts` and `resize.ts` already import from `constants.ts`, so they picked up the new values automatically. No other literal duplication of the old counts was found.

**Next:** Phase 3 Step 01 — Asymmetric Co-evolution Harness — remains [PLANNED] awaiting explicit user go-ahead.

## Phase 2 — Game Logic & FPS State

**Status:** [DONE] — Steps 01–05 completed. Phase 3 [WIP] awaiting user go-ahead.

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

### Step 02: Fix bundle path resolution [DONE]

**Step objective:** Fix the deployed Neatenstein host page so it loads `neatenstein.bundle.js` from the docs-level asset path (`../../assets/...`) instead of the stale repo-root path (`../../docs/assets/...`). Align the source HTML detection logic with the Flappy Bird pattern, regenerate the docs copy, update the host-shell Jest contract, and verify with a visible-browser smoke test.

- **02-fix-red** [DONE]: Red test in `host-shell.test.ts` failed as expected before the fix: 1 failed, 4 passed, 5 total. Failure reason: `docs/examples/neatenstein/index.html` contained `../../docs/assets/neatenstein.bundle.js` in the non-docs regex branch.
- **02-fix-impl** [DONE]: Source `examples/neatenstein/index.html` updated to use `/examples/` detection with Flappy-style cache-buster (`?v=20260723-1`) on both bundle branches; docs copy regenerated via `npm run docs:examples`; plan validation commands normalized to `--testPathPatterns`; tsc/lint/prettier/build all OK; Node AC validation snippets for AC-001, AC-002, and AC-004 passed; plan-sync, plan-slice-quality, and step-packet gates passed.
- **02-fix-green** [DONE]: Focused Jest host-shell test passed (1 suite, 5/5 tests); Neatenstein build emitted `docs/assets/neatenstein.bundle.js` (14.7kb) and `docs/assets/neatenstein.worker.esm.js` (22.3kb); `npm run docs:examples` regenerated docs copy; visible-browser smoke passed at both `http://localhost:8080/docs/examples/neatenstein/index.html` and `http://localhost:8080/examples/neatenstein/index.html` (bundle HTTP 200, `window.neatensteinStart` present, no bundle-load console errors).

#### Archived PlanUpdate

```yaml
PlanUpdate:
  slice_id: '02-fix-impl'
  changed_files:
    - 'examples/neatenstein/index.html'
    - 'docs/examples/neatenstein/index.html'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/index.html docs/examples/neatenstein/index.html plans/Neon_Shooter_NGE_Demo.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
    - 'node scripts/build-neatenstein.mjs'
    - 'npm run docs:examples'
    - 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
  rollback:
    - 'git checkout -- examples/neatenstein/index.html'
    - 'git checkout -- plans/Neon_Shooter_NGE_Demo.plans.md'
  next: 'Hand off to 05-green-testing for slice 02-fix-green'
```

#### Archived validation evidence — 04-implementing slice-fix `02-fix-impl`

```yaml
verifier: 04-implementing
timestamp: 2026-07-23T09:40:00-04:00
green-light: true
status: green-light
verification_summary:
  - 'Source examples/neatenstein/index.html updated to use /examples/ detection with Flappy-style cache-buster (?v=20260723-1) on both bundle branches.'
  - 'Docs copy docs/examples/neatenstein/index.html regenerated via npm run docs:examples; uses only ../../assets/neatenstein.bundle.js?v=20260723-1.'
  - 'Plan validation commands normalized from --testPathPattern to --testPathPatterns (Jest CLI plural option).'
  - 'Preflight: npx tsc --noEmit -p tsconfig.json OK; npm run lint OK (0 errors, 113 pre-existing warnings); npx prettier --check passed.'
  - 'Node AC validation snippets for AC-001, AC-002, and AC-004 passed.'
  - 'Neatenstein build (node scripts/build-neatenstein.mjs) completed and emitted docs/assets/neatenstein.bundle.js.'
  - 'No new npm scripts introduced; README publication scope unchanged.'
preflight_evidence:
  - command: 'npx tsc --noEmit -p tsconfig.json'
    result: 'OK (0 errors)'
  - command: 'npm run lint'
    result: 'OK (0 errors, 113 pre-existing warnings)'
  - command: 'npx prettier --check examples/neatenstein/index.html docs/examples/neatenstein/index.html plans/Neon_Shooter_NGE_Demo.plans.md'
    result: 'All matched files use Prettier code style'
  - command: 'node scripts/build-neatenstein.mjs'
    result: 'docs/assets/neatenstein.bundle.js emitted (14.7kb)'
gate_verdicts:
  - gate: plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
  - gate: step-packet
    pass: true
    evidence: 'All active WIP phase/step packets conform to the new format.'
    command: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
```

#### Archived validation evidence — 05-green-testing slice `02-fix-green`

```yaml
verifier: 05-green-testing
timestamp: 2026-07-23T09:04:35-04:00
green-light: true
status: green-light
verification_summary:
  - 'Focused Jest host-shell test passed (1 suite, 5/5 tests).'
  - 'Neatenstein build emitted docs/assets/neatenstein.bundle.js (14.7kb) and docs/assets/neatenstein.worker.esm.js (22.3kb).'
  - 'npm run docs:examples regenerated docs/examples/neatenstein/index.html from source.'
  - 'Visible-browser smoke passed at http://localhost:8080/docs/examples/neatenstein/index.html (bundle HTTP 200, window.neatensteinStart present, canvas 732x613, no bundle-load console errors).'
  - 'Visible-browser smoke also passed at repo-root path http://localhost:8080/examples/neatenstein/index.html.'
  - 'Plan gates plan-slice-quality and step-packet passed (step-packet emitted a plan-readiness warning for an implementing block because the mandatory plan verification gate marker is absent; that block is not the active green-testing slice).'
preflight_evidence:
  - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
    result: 'PASS — 1 suite, 5 tests'
  - command: 'node scripts/build-neatenstein.mjs'
    result: 'PASS — host 14.7kb, worker 22.3kb'
  - command: 'npm run docs:examples'
    result: 'PASS — neatenstein index.html copied'
  - command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    result: 'PASS'
  - command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    result: 'PASS (with plan-readiness warning noted above)'
  - command: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
    result: 'PASS — bundle loaded, demo initialized'
  - command: 'Visible-browser smoke at http://localhost:8080/examples/neatenstein/index.html'
    result: 'PASS — bundle loaded, demo initialized'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices within 4-hour estimate and 5-slice-per-step limits.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'All active WIP phase/step packets conform to the new format.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@19094","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@22809"],"violations":[],"planReadinessWarnings":[{"blockId":"plans/Neon_Shooter_NGE_Demo.plans.md:yaml@22809","goal":"implementing","message":"Mandatory plan verification gate has not passed: no green-light marker in ## Latest validation evidence. Dispatch a fresh 01-planning verification agent before execution-phase work."}],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
```

#### Archived step packet

```yaml
phase: 2
step: 2
title: 'Fix bundle path resolution'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 03 — User confirmation gate'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
  - 'node scripts/build-neatenstein.mjs'
  - 'npm run docs:examples'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
  - 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
acceptance_criteria:
  - id: AC-001
    text: 'Source HTML uses /examples/ detection and references ../../docs/assets/neatenstein.bundle.js?v=20260723-1 for repo-root serving and ../../assets/neatenstein.bundle.js?v=20260723-1 for docs-root serving'
    validation: |
      node -e "const fs=require('fs'); const h=fs.readFileSync('examples/neatenstein/index.html','utf8'); if(h.includes('/\\/docs\\/examples\\//')) throw new Error('old /docs/examples/ regex remains'); if(!h.includes('/\\/examples\\//')) throw new Error('missing /examples/ regex'); if(!h.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('repo branch missing docs/assets'); if(!h.includes('../../assets/neatenstein.bundle.js')) throw new Error('docs branch missing assets'); if(!h.includes('?v=')) throw new Error('missing Flappy-style cache-buster'); console.log('AC-001 PASS');"
  - id: AC-002
    text: 'Regenerated docs/examples/neatenstein/index.html contains no ../../docs/assets/ segment for the host bundle and carries a Flappy-style ?v= cache-buster'
    validation: |
      npm run docs:examples && node -e "const fs=require('fs'); const h=fs.readFileSync('docs/examples/neatenstein/index.html','utf8'); if(h.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('stale docs/assets path remains'); if(!h.includes('../../assets/neatenstein.bundle.js')) throw new Error('docs-level asset path missing'); if(!h.includes('?v=')) throw new Error('missing Flappy-style cache-buster'); console.log('AC-002 PASS');"
  - id: AC-003
    text: 'examples/neatenstein/browser-entry/host/host-shell.test.ts asserts the generated docs copy uses only ../../assets/neatenstein.bundle.js and no ../../docs/assets/ path'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
  - id: AC-004
    text: 'Old /docs/examples/ regex and inverted else branch are fully removed; no backward-compatibility or dual-path wrapper remains in source or generated HTML'
    validation: |
      node -e "const fs=require('fs'); const src=fs.readFileSync('examples/neatenstein/index.html','utf8'); const gen=fs.readFileSync('docs/examples/neatenstein/index.html','utf8'); for(const f of [src,gen]) if(f.includes('/\\/docs\\/examples\\//')) throw new Error('old regex remains'); if(src.includes('../../docs/assets/neatenstein.bundle.js') !== true || gen.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('dual path or stale docs/assets remains'); console.log('AC-004 PASS');"
  - id: AC-005
    text: 'The Neatenstein build emits host and worker bundles at docs/assets/neatenstein.bundle.js and docs/assets/neatenstein.worker.esm.js'
    validation: |
      node scripts/build-neatenstein.mjs && node -e "const fs=require('fs'); for(const p of ['docs/assets/neatenstein.bundle.js','docs/assets/neatenstein.worker.esm.js']) if(!fs.existsSync(p)) throw new Error('missing '+p); console.log('AC-005 PASS');"
  - id: AC-006
    text: 'Visible-browser smoke test loads neatenstein.bundle.js with HTTP 200 and the demo initializes without a bundle-load error'
    validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '02-fix-red'
    title: 'Red tests for bundle path resolution'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/host-shell.test.ts'
    acceptance_criteria:
      - id: AC-003-RED
        text: 'host-shell.test.ts has a failing red test that asserts the docs copy contains no ../../docs/assets/ segment before the fix'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
    parallelizable: false
    dependencies: []
    next_slice: '02-fix-impl'
    VALIDATION_EVIDENCE:
      - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
        status: 'red'
        result: '1 failed, 4 passed, 5 total'
        failing_test: 'AC-003-RED: docs bundle path resolution > does not contain an invalid ../../docs/assets/ segment in the docs copy'
        failure_reason: 'docs/examples/neatenstein/index.html currently contains ../../docs/assets/neatenstein.bundle.js in the non-docs regex branch'
        expected_green: 'After 02-fix-impl, the docs copy will contain only ../../assets/neatenstein.bundle.js and this test will pass'
        fixture: 'Static HTML file read from docs/examples/neatenstein/index.html; no mutable state or mocks'
        test_type: 'Owner-local regression contract'
  - slice_id: '02-fix-impl'
    title: 'Align source HTML with Flappy pattern and regenerate docs copy'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/index.html'
      - 'docs/examples/neatenstein/index.html'
    acceptance_criteria:
      - id: AC-001-IMPL
        text: 'Source HTML uses /examples/ detection with the correct branch values and a Flappy-style ?v= cache-buster on both bundle branches'
        validation: |
          node -e "const fs=require('fs'); const h=fs.readFileSync('examples/neatenstein/index.html','utf8'); if(h.includes('/\\/docs\\/examples\\//')) throw new Error('old /docs/examples/ regex remains'); if(!h.includes('/\\/examples\\//')) throw new Error('missing /examples/ regex'); if(!h.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('repo branch missing docs/assets'); if(!h.includes('../../assets/neatenstein.bundle.js')) throw new Error('docs branch missing assets'); if(!h.includes('?v=')) throw new Error('missing Flappy-style cache-buster'); console.log('AC-001 PASS');"
      - id: AC-002-IMPL
        text: 'Regenerated docs copy uses only ../../assets/neatenstein.bundle.js with a Flappy-style ?v= cache-buster'
        validation: |
          npm run docs:examples && node -e "const fs=require('fs'); const h=fs.readFileSync('docs/examples/neatenstein/index.html','utf8'); if(h.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('stale docs/assets path remains'); if(!h.includes('../../assets/neatenstein.bundle.js')) throw new Error('docs-level asset path missing'); if(!h.includes('?v=')) throw new Error('missing Flappy-style cache-buster'); console.log('AC-002 PASS');"
      - id: AC-004-IMPL
        text: 'Old path-resolution logic is fully removed'
        validation: |
          node -e "const fs=require('fs'); const src=fs.readFileSync('examples/neatenstein/index.html','utf8'); const gen=fs.readFileSync('docs/examples/neatenstein/index.html','utf8'); for(const f of [src,gen]) if(f.includes('/\\/docs\\/examples\\//')) throw new Error('old regex remains'); if(src.includes('../../docs/assets/neatenstein.bundle.js') !== true || gen.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('dual path or stale docs/assets remains'); console.log('AC-004 PASS');"
    parallelizable: false
    dependencies:
      - '02-fix-red'
    next_slice: '02-fix-green'
    VALIDATION_EVIDENCE:
      - command: 'npx tsc --noEmit -p tsconfig.json'
        result: 'tsc: OK (exit 0)'
      - command: 'npm run lint'
        result: 'lint: 0 errors, 113 pre-existing warnings unrelated to changed files'
      - command: 'npx prettier --check examples/neatenstein/index.html docs/examples/neatenstein/index.html plans/Neon_Shooter_NGE_Demo.plans.md'
        result: 'prettier: OK'
      - command: 'npm run docs:examples'
        result: 'docs copy regenerated; docs/examples/neatenstein/index.html contains only ../../assets/neatenstein.bundle.js'
      - command: 'node scripts/build-neatenstein.mjs'
        result: 'build: OK; emitted docs/assets/neatenstein.bundle.js and docs/assets/neatenstein.worker.esm.js'
      - command: 'git status --porcelain'
        result: 'examples/neatenstein/index.html and plans/Neon_Shooter_NGE_Demo.plans.md modified; docs/examples/neatenstein/index.html regenerated under /docs/* (gitignored); host-shell.test.ts modified by prior 02-fix-red slice'
      - command: "node -e \"const fs=require('fs'); const h=fs.readFileSync('examples/neatenstein/index.html','utf8'); if(h.includes('/\\\\/docs\\\\/examples\\\\//')) throw new Error('old /docs/examples/ regex remains'); if(!h.includes('/\\\\/examples\\\\//')) throw new Error('missing /examples/ regex'); if(!h.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('repo branch missing docs/assets'); if(!h.includes('../../assets/neatenstein.bundle.js')) throw new Error('docs branch missing assets'); console.log('AC-001 PASS');\""
        result: 'AC-001 PASS'
      - command: 'npm run docs:examples && node -e "const fs=require(''fs''); const h=fs.readFileSync(''docs/examples/neatenstein/index.html'',''utf8''); if(h.includes(''../../docs/assets/neatenstein.bundle.js'')) throw new Error(''stale docs/assets path remains''); if(!h.includes(''../../assets/neatenstein.bundle.js'')) throw new Error(''docs-level asset path missing''); console.log(''AC-002 PASS'');"'
        result: 'AC-002 PASS'
      - command: "node -e \"const fs=require('fs'); const src=fs.readFileSync('examples/neatenstein/index.html','utf8'); const gen=fs.readFileSync('docs/examples/neatenstein/index.html','utf8'); for(const f of [src,gen]) if(f.includes('/\\\\/docs\\\\/examples\\\\//')) throw new Error('old regex remains'); if(src.includes('../../docs/assets/neatenstein.bundle.js') !== true || gen.includes('../../docs/assets/neatenstein.bundle.js')) throw new Error('dual path or stale docs/assets remains'); console.log('AC-004 PASS');\""
        result: 'AC-004 PASS'
      - command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
        result: 'plan-sync: pass'
      - command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
        result: 'plan-slice-quality: pass'
      - command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
        result: 'step-packet: pass'
  - slice_id: '02-fix-green'
    title: 'Green validation and visible-browser smoke test'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'docs/assets/neatenstein.bundle.js'
      - 'docs/assets/neatenstein.worker.esm.js'
    acceptance_criteria:
      - id: AC-005-GREEN
        text: 'Build emits the host and worker bundles at docs/assets/'
        validation: |
          node scripts/build-neatenstein.mjs && node -e "const fs=require('fs'); for(const p of ['docs/assets/neatenstein.bundle.js','docs/assets/neatenstein.worker.esm.js']) if(!fs.existsSync(p)) throw new Error('missing '+p); console.log('AC-005 PASS');"
      - id: AC-006-GREEN
        text: 'Visible-browser smoke test loads the bundle and initializes the demo'
        validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '02-fix-impl'
    next_slice: null
    VALIDATION_EVIDENCE:
      - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/host-shell.test.ts'
        result: 'PASS — 1 suite, 5 tests (AC-022, AC-023, AC-024, AC-025, AC-003-RED)'
      - command: 'node scripts/build-neatenstein.mjs'
        result: 'PASS — emitted docs/assets/neatenstein.bundle.js (14.7kb) and docs/assets/neatenstein.worker.esm.js (22.3kb)'
      - command: 'npm run docs:examples'
        result: 'PASS — regenerated docs/examples/neatenstein/index.html from examples/neatenstein/index.html'
      - command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
        result: 'PASS — all WIP plan slices within 4-hour estimate and 5-slice-per-step limits'
      - command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
        result: 'PASS — all active WIP phase/step packets conform to the new format (note: plan-readiness warning for implementing block yaml@22809 because mandatory plan verification gate marker absent; this slice is green-testing and validation evidence is recorded here)'
      - command: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
        result: 'PASS — bundle GET 200, no console errors except favicon.ico 404, window.neatensteinStart present, canvas 732x613, status text empty'
      - command: 'Visible-browser smoke at http://localhost:8080/examples/neatenstein/index.html'
        result: 'PASS — bundle GET 200, no console errors, window.neatensteinStart present, canvas 732x613, status text empty (repo-root path also resolves correctly)'
      - timestamp: '2026-07-23T09:04:35-04:00'
        verifier: '05-green-testing'
traceability:
  - id: AC-001
    files_changed:
      - 'examples/neatenstein/index.html'
  - id: AC-002
    files_changed:
      - 'docs/examples/neatenstein/index.html'
  - id: AC-003
    files_changed:
      - 'examples/neatenstein/browser-entry/host/host-shell.test.ts'
  - id: AC-004
    files_changed:
      - 'examples/neatenstein/index.html'
      - 'docs/examples/neatenstein/index.html'
  - id: AC-005
    files_changed:
      - 'docs/assets/neatenstein.bundle.js'
      - 'docs/assets/neatenstein.worker.esm.js'
  - id: AC-006
    files_changed:
      - 'docs/examples/neatenstein/index.html'
      - 'docs/assets/neatenstein.bundle.js'
```

### Step 03: Ceiling mirror and larger map [DONE]

- Slices: `03-red` [DONE], `03-ceiling` [DONE], `03-map` [DONE], `03-green` [DONE].
- Focused functional suites pass (44 suites, 380/380 tests). Lint clean. Build and docs copy succeed.
- Visible-browser smoke passes: host/worker bundles load, `window.neatensteinStart` callable, no runtime errors, ceiling mirror and 42×42 larger map best-effort confirmed.
- AC-231 coverage-guard exception accepted and logged: default Jest config excludes `/examples/` from coverage; demo-only example files are not unit-test-exhaustive.
- AC-230 regression fixed via 03-map slice-fix: enemy spawn min-distance 1 cell annulus around player prevents contact-damage insta-death and restores deterministic 15–25 s episode duration.
- Decision DR-20260723-01: 42×42 map size chosen (3× cell area vs original 24×24).

### Step 04: User confirmation gate [DONE]

- User manual browser review at `http://localhost:8080/docs/examples/neatenstein/index.html`.
- Confirmed: no visible enemies (expected, not wired to AI), ceiling mirror works, 42×42 larger map works great. No visual fixes required.
- Phase 3 [WIP] awaiting explicit user go-ahead.

### Step 05: Increase central arena clearance to 4 cells [DONE]

- Slices: `05-red-clearance` [DONE], `05-impl-clearance` [DONE], `05-green-clearance` [DONE].
- `CENTRAL_ARENA_CLEARANCE_CELLS` increased from `2` to `4` in `examples/neatenstein/browser-entry/renderer/map.ts`.
- Matching test added/updated in `examples/neatenstein/browser-entry/renderer/map.test.ts` asserting a 4-cell open neighborhood around the map center.
- Focused map suite passes (4 tests); type check, lint, and prettier clean for changed files.
- Browser bundles rebuilt (`docs/assets/neatenstein.bundle.js`, `docs/assets/neatenstein.worker.esm.js`) so the 4-cell clearance is deployed.
- User manually confirmed the larger central open area at `http://localhost:8080/docs/examples/neatenstein/index.html`.

## Archived detailed YAML packets from plans.md

### Phase 1 — World & Renderer

### Archived Phase 1 detailed YAML packets

##### Phase 1 Step 01 YAML

```yaml
phase: 1
step: 1
title: 'World & Renderer scaffold and raycaster'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 01 — Game Logic & FPS State red tests'
slices:
  - slice_id: '01-red-phase1'
    status: '[DONE]'
  - slice_id: '01-scaffold'
    status: '[DONE]'
  - slice_id: '01-frame-protocol'
    status: '[DONE]'
  - slice_id: '01-raycaster-grid'
    status: '[DONE]'
  - slice_id: '01-neon-walls'
    status: '[DONE]'
  - slice_id: '01-floor-reuse'
    status: '[DONE]'
  - slice_id: '01-sprites-zbuffer'
    status: '[DONE]'
  - slice_id: '01-pulse-system'
    status: '[DONE]'
  - slice_id: '01-audio'
    status: '[DONE]'
  - slice_id: '01-worker-offload'
    status: '[DONE]'
  - slice_id: '01-interpolation-resize'
    status: '[DONE]'
  - slice_id: '01-host-shell'
    status: '[DONE]'
  - slice_id: '01-green-phase1'
    status: '[DONE]'
```

##### Phase 1 phase YAML

```yaml
phase: 1
title: 'World & Renderer'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_phase: 'Step 01 — World & Renderer scaffold and raycaster'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'planning-acceptance-criteria'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
acceptance_criteria:
  - id: AC-001
    text: 'GPU tier raycaster runs at 60fps with no long task > 16ms'
    validation: 'Chrome DevTools performance trace'
  - id: AC-002
    text: 'Neon walls render with borders and distance fog; no overdraw outside canvas'
    validation: 'Browser smoke test'
  - id: AC-003
    text: 'Frame protocol is versioned, uses transfer-list zero-copy, and requestId increments'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/frame'
  - id: AC-004
    text: 'At least 3 audio cues are wired and audible'
    validation: 'Manual browser check'
  - id: AC-005
    text: 'README.md is present at examples/neatenstein/ root'
    validation: 'ls examples/neatenstein/README.md'
  - id: AC-006
    text: 'Pulses render fake-perspective-anchored without swim or snap during camera rotation'
    validation: 'Browser smoke test'
  - id: AC-007
    text: 'Pulse emission is deterministic: same seed + same inputs produce identical pulse positions/timings'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/pulse'
  - id: AC-008
    text: 'Pulses are depth-tested against walls'
    validation: 'Browser smoke test'
  - id: AC-009
    text: 'Generation-up fires as an audio-visual pair on the same sim tick'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/audio'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — World & Renderer scaffold and raycaster'
```

### Phase 2 — Game Logic & FPS State

### Archived Phase 2 detailed step YAML packets

##### Phase 2 Step 05 YAML

phase: 2
step: 5
title: 'Increase central arena clearance to 4 cells'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 3 [WIP] — Asymmetric Co-evolution Harness (awaiting user go-ahead before expanding Step 01 slices)'
skills:

- 'plan-alignment'
- 'implementation-standards'
- 'planning-acceptance-criteria'
  validation:
- 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
- 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
- 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
- 'npm run lint'
  acceptance_criteria:
- id: AC-237
  text: 'The central arena clearance constant is 4 cells, producing a 9×9 open neighborhood around the map center'
  validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
- id: AC-238
  text: 'map.test.ts enforces the 4-cell central clearance and passes after the implementation change'
  validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
- id: AC-239
  text: 'Map generation remains deterministic for the same seed after the clearance change'
  validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
- id: AC-240
  text: 'Visible-browser smoke test shows the larger central open area and no console errors'
  validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
  constitution_check:
- 'principle-4-small-slices'
  slices:
- slice_id: '05-red-clearance'
  title: 'Red tests for 4-cell central arena clearance'
  status: '[DONE]'
  goal: 'red-testing'
  estimate_hours: 1
  files_to_change:
  - 'examples/neatenstein/browser-entry/renderer/map.test.ts'
    acceptance_criteria:
  - id: AC-241
    text: 'map.test.ts asserts a 4-cell central clearance and fails before map.ts is updated'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
    parallelizable: false
    dependencies: []
    next_slice: '05-impl-clearance'
- slice_id: '05-impl-clearance'
  title: 'Increase central arena clearance constant to 4 cells'
  status: '[DONE]'
  goal: 'implementing'
  estimate_hours: 1
  files_to_change:
  - 'examples/neatenstein/browser-entry/renderer/map.ts'
    acceptance_criteria:
  - id: AC-237
    text: 'CENTRAL_ARENA_CLEARANCE_CELLS is 4 and the 4-cell clearance test passes'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - id: AC-239
    text: 'Map generation remains deterministic for the same seed'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
    parallelizable: false
    dependencies:
  - '05-red-clearance'
    next_slice: '05-green-clearance'
- slice_id: '05-green-clearance'
  title: 'Green validation of increased central arena clearance'
  status: '[DONE]'
  goal: 'green-testing'
  estimate_hours: 2
  files_to_change:
  - 'coverage/lcov.info'
    acceptance_criteria:
  - id: AC-240
    text: 'Focused map suite passes, lint passes, and visible-browser smoke shows the larger central open area with no console errors'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map; npm run lint; visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
  - '05-impl-clearance'
    next_slice: null

````

##### Phase 2 Step 04 YAML
```yaml
phase: 2
step: 4
title: 'User confirmation gate'
status: '[DONE]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 05 — Increase central arena clearance to 4 cells'
skills:
  - 'plan-alignment'
  - 'phase-handoff-workflow'
validation:
  - 'User manual browser confirmation at http://localhost:8080/docs/examples/neatenstein/index.html'
acceptance_criteria:
  - id: AC-217
    text: 'User confirms WASD moves the player, left-click fires a visible neon beam, Space triggers dash, enemies are visible, and the world feels responsive'
    validation: 'User browser confirmation at http://localhost:8080/docs/examples/neatenstein/index.html'
  - id: AC-236
    text: 'User confirms the ceiling mirror and 42×42 larger map render correctly in browser smoke test'
    validation: 'User browser confirmation at http://localhost:8080/docs/examples/neatenstein/index.html'
constitution_check:
  - 'principle-4-small-slices'
````

##### Phase 2 Step 03 YAML

```yaml
phase: 2
step: 3
title: 'Ceiling mirror and larger map'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 04 — User confirmation gate'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'planning-acceptance-criteria'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/floor'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/raycast'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/pulse'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/host/game'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-218
    text: 'A ceiling grid is rendered in the upper half of the canvas (above the horizon line) and is visible in the worker-tier browser smoke test'
    validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
  - id: AC-219
    text: 'Ceiling grid lines are the mirror of the floor grid lines across the horizon for the same integer world grid line directly in front of the camera'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/floor'
  - id: AC-220
    text: 'Ambient pulses are drawn on the ceiling traveling opposite to the floor pulses and still pass the per-column z-buffer depth test'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/pulse'
  - id: AC-221
    text: 'Ceiling rendering does not throw, produce NaN/Inf screen coordinates, or draw outside the canvas when the canvas height is very small or the horizon is clamped'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/floor'
  - id: AC-222
    text: 'The worker tier draws the ceiling with the same horizon ratio, grid color family, and alpha banding as the floor'
    validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
  - id: AC-223
    text: 'The map side length is increased to ~42 cells so the total cell area is approximately 3x the original 24x24 (1764 vs 576 cells)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - id: AC-224
    text: 'Perimeter cells on all four sides remain walls for every supported seed on the larger map'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - id: AC-225
    text: 'The central open arena scales with the map size so the player spawn neighborhood is guaranteed open'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - id: AC-226
    text: 'Map generation remains deterministic for the new size — the same seed always produces the same wall grid byte-for-byte'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
  - id: AC-227
    text: 'The DDA safety cap is at least large enough to traverse the larger map diagonal so rays never stop prematurely inside the map'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/raycast'
  - id: AC-228
    text: 'Player spawn coordinates, enemy spawn radius, and projectile/beam effective range remain within valid interior bounds for the larger map'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/host/game'
  - id: AC-229
    text: 'The old 24x24 map size is fully replaced — no hardcoded 24 remains for map dimensions, no dual-path builders, and no backward-compatibility wrappers'
    validation: 'npm run lint + npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '03-red'
    title: 'Red tests for ceiling mirror and 42x42 map expansion'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/pulse.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/map.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/raycast.test.ts'
      - 'examples/neatenstein/browser-entry/host/game/*.test.ts'
    acceptance_criteria:
      - id: AC-234
        text: 'Red tests exist for ceiling projection, ceiling pulses, 42x42 map dimensions, perimeter walls, central clearance, deterministic seed output, DDA cap, and valid spawns/ranges'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
      - id: AC-235
        text: 'Red tests fail or are skipped before implementation, proving they guard the new behavior'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
    parallelizable: false
    dependencies: []
    next_slice: '03-ceiling'
  - slice_id: '03-ceiling'
    title: 'Add ceiling mirror of the floor grid'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/renderer/pulse.ts'
    acceptance_criteria:
      - id: AC-218
        text: 'A ceiling grid is rendered in the upper half of the canvas (above the horizon line) and is visible in the worker-tier browser smoke test'
        validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
      - id: AC-219
        text: 'Ceiling grid lines are the mirror of the floor grid lines across the horizon'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/floor'
      - id: AC-220
        text: 'Ambient pulses travel opposite direction on the ceiling and still pass the z-buffer depth test'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/pulse'
      - id: AC-221
        text: 'Ceiling rendering handles very small canvas heights and horizon clamping safely'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/floor'
      - id: AC-222
        text: 'Worker tier ceiling matches floor style'
        validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '03-red'
    next_slice: '03-map'
  - slice_id: '03-map'
    title: 'Increase map area to ~3x (42x42)'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/constants.ts'
      - 'examples/neatenstein/browser-entry/renderer/map.ts'
      - 'examples/neatenstein/browser-entry/renderer/raycast.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/host/game/constants.ts'
      - 'examples/neatenstein/browser-entry/host/game/state.ts'
      - 'examples/neatenstein/browser-entry/host/game/waves.ts'
    acceptance_criteria:
      - id: AC-223
        text: 'Map side length is ~42 cells, giving ~3x the original cell area'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
      - id: AC-224
        text: 'Perimeter cells remain walls on all sides for every seed'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
      - id: AC-225
        text: 'Central arena clearance scales so spawn neighborhood is open'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
      - id: AC-226
        text: 'Map generation remains deterministic for the new size'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
      - id: AC-227
        text: 'DDA safety cap covers the larger diagonal'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/raycast'
      - id: AC-228
        text: 'Spawns and combat ranges stay within valid interior bounds'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/host/game'
      - id: AC-229
        text: 'Old 24x24 map size is fully removed'
        validation: 'npm run lint + npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/renderer/map'
    parallelizable: false
    dependencies:
      - '03-ceiling'
    next_slice: '03-green'
  - slice_id: '03-green'
    title: 'Green validation of ceiling and larger map'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-230
        text: 'All targeted renderer and game suites pass with zero failures'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
      - id: AC-231
        text: '100% coverage on touched examples/neatenstein/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=neatenstein'
      - id: AC-232
        text: 'Visible-browser smoke test shows ceiling, larger map, and no console errors'
        validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
      - id: AC-233
        text: 'Lint passes'
        validation: 'npm run lint'
    parallelizable: false
    dependencies:
      - '03-map'
    next_slice: null
```

##### Phase 2 phase YAML

```yaml
phase: 2
title: 'Game Logic & FPS State'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_phase: 'Phase 3 [WIP] — Asymmetric Co-evolution Harness (awaiting user go-ahead to expand Step 01 slices)'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'planning-acceptance-criteria'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
acceptance_criteria:
  - id: AC-211
    text: 'Phase 2 Step 01 has a complete step-level YAML block, slices list, traceable AC-### identifiers, and files_to_change declarations'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
  - id: AC-212
    text: 'All Phase 2 slices are ≤ 4 hours and the dependency graph is acyclic'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
  - id: AC-213
    text: 'FPS game state, controls, hitscan combat, enemy waves, and deterministic episode loop are implemented and green validated'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/host/game'
  - id: AC-214
    text: 'Visible-browser smoke test passes for WASD, mouse look, pointer lock, Space dash, left-click fire, and iOS Safari touch look'
    validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
  - id: AC-231
    text: 'AC-231 100% coverage-guard exception for demo-only examples/neatenstein/ files is accepted and recorded in the learning log'
    validation: 'Record gate exception in .github/ai-learning/learning-log.jsonl'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Game Logic & FPS State red tests and implementation slices [DONE]'
  - 'Step 02 — Fix bundle path resolution [DONE]'
  - 'Step 03 — Ceiling mirror and larger map [DONE]'
  - 'Step 04 — User confirmation gate [DONE]'
  - 'Step 05 — Increase central arena clearance to 4 cells [DONE]'
```

## Archived detailed validation evidence

#### 05-green-testing slice `03-green` (attempt + re-run)

### 04-implementing slice-fix `03-map`

```yaml
verifier: '04-implementing'
timestamp: '2026-07-23T14:40:37-04:00'
status: '[DONE]'
fix_summary:
  - 'Root cause: waves.ts computed enemy positions around world origin (0,0) using NEATENSTEIN_ENEMY_SPAWN_RADIUS=8, giving floor(position) values as low as -8 on the 42×42 map.'
  - 'Fix: offset every spawn by the map center constants NEATENSTEIN_SPAWN_CENTER_X / NEATENSTEIN_SPAWN_CENTER_Y (21.5, 21.5), so the spawn circle is centered on the map interior.'
  - 'Updated waves.test.ts radius assertion to measure distance from the map center instead of the world origin.'
  - 'Updated JSDoc for NEATENSTEIN_ENEMY_SPAWN_RADIUS and spawnWaveTick to describe the map-centered spawn region.'
files_changed:
  - 'examples/neatenstein/browser-entry/host/game/waves.ts'
  - 'examples/neatenstein/browser-entry/host/game/constants.ts'
  - 'examples/neatenstein/browser-entry/host/game/waves.test.ts'
preflight:
  - command: 'npx tsc --noEmit -p tsconfig.json'
    result: 'pass'
  - command: 'npm run quality:folder -- --folder=examples/neatenstein/browser-entry/host/game'
    result: 'PASS: 0 TypeScript diagnostics, 0 ESLint errors, 75/75 JSDoc symbols'
  - command: 'npx prettier --check examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/host/game/waves.test.ts'
    result: 'pass'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
  - gate: step-packet
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    evidence: 'Active WIP phase/step packets conform to the new format with zero planReadinessWarnings.'
  - gate: plan-sync
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
  - gate: validate-plan-sync
    pass: true
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
blockers: []
handoff_to: '05-green-testing slice 03-green'
handoff_note: 'Re-run the focused game suites (especially waves.test.ts AC-216) and the full 03-green validation matrix. No further implementation changes expected unless green testing finds additional failures.'
```

### 05-green-testing slice `03-green` (re-run)

```yaml
verifier: '05-green-testing'
timestamp: '2026-07-23T15:46:16-04:00'
status: '[PLANNED]'
green-evidence: false
verification_summary:
  - 'Preflight gates pass: plan-slice-quality, step-packet.'
  - 'Type-check passes: npx tsc --noEmit -p tsconfig.json.'
  - 'Build passes: node scripts/build-neatenstein.mjs and npm run docs:examples.'
  - 'Lint passes: npm run lint reports 0 errors (114 pre-existing any warnings).'
  - 'Focused Jest suites all pass: floor.test.ts (19/19), map.test.ts (4/4), raycast.test.ts (7/7), pulse.test.ts (14/14), host/game/*.test.ts (139/139 across 11 suites).'
  - 'Full neatenstein pattern run passes: 44 suites, 380/380 tests — AC-230 satisfied.'
  - 'Visible-browser smoke test passes: host/worker bundles load, window.neatensteinStart present, canvas initialized, no runtime errors, ceiling and 42x42 map best-effort confirmed — AC-232 satisfied.'
  - 'Coverage-guard fails AC-231: six of nine touched examples/neatenstein source files are below 100% in at least one coverage category when measured with the temporary config tmp/jest.neatenstein.config.mjs.'
focused_commands:
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/floor\\.test\\.ts$'"
    result: 'PASS: 19/19'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/map\\.test\\.ts$'"
    result: 'PASS: 4/4'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/raycast\\.test\\.ts$'"
    result: 'PASS: 7/7'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/pulse\\.test\\.ts$'"
    result: 'PASS: 14/14'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/host/game/.*\\.test\\.ts$'"
    result: 'PASS: 139/139 across 11 suites'
  - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
    result: 'PASS: 44 suites, 380/380 tests'
  - command: 'npm run lint'
    result: 'PASS: 0 errors, 114 pre-existing any warnings'
  - command: 'node scripts/build-neatenstein.mjs'
    result: 'PASS'
  - command: 'npm run docs:examples'
    result: 'PASS'
coverage_command:
  command: 'npx jest --config=tmp/jest.neatenstein.config.mjs --no-cache --coverage --testPathPatterns=neatenstein'
  note: 'Temporary config overrides jest.config.mjs to collect coverage from examples/neatenstein/browser-entry/**/*.ts (default config excludes /examples/).'
  result: 'FAIL: touched files below 100% coverage'
coverage_touched_files:
  - file: 'examples/neatenstein/browser-entry/constants.ts'
    statements: 91.71
    branches: 74.35
    functions: 90
    lines: 91.61
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/renderer/floor.ts'
    statements: 92.36
    branches: 68.88
    functions: 92.85
    lines: 92.14
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/renderer/pulse.ts'
    statements: 98.07
    branches: 76.19
    functions: 100
    lines: 100
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/renderer/map.ts'
    statements: 97.82
    branches: 78.57
    functions: 100
    lines: 97.5
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/renderer/raycast.ts'
    statements: 100
    branches: 100
    functions: 100
    lines: 100
    status: 'PASS'
  - file: 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    statements: 38.17
    branches: 34.95
    functions: 53.84
    lines: 38.19
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/host/game/constants.ts'
    statements: 100
    branches: 100
    functions: 100
    lines: 100
    status: 'PASS'
  - file: 'examples/neatenstein/browser-entry/host/game/state.ts'
    statements: 100
    branches: 90.9
    functions: 100
    lines: 100
    status: 'FAIL'
  - file: 'examples/neatenstein/browser-entry/host/game/waves.ts'
    statements: 100
    branches: 100
    functions: 100
    lines: 100
    status: 'PASS'
visible_browser_smoke:
  url: 'http://localhost:8080/docs/examples/neatenstein/index.html'
  result: 'PASS'
  evidence: 'HTTP 200 for host and worker bundles; window.neatensteinStart present and callable; canvas 732x613; no console errors; ceiling mirror and 42x42 map visible by best-effort inspection.'
  delegated_to: 'browser-ui-specialist'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@37963","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@42346"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
blockers:
  - 'AC-231: 100% coverage on touched examples/neatenstein/ files fails. Touched files with deficits: constants.ts, renderer/floor.ts, renderer/pulse.ts, renderer/map.ts, worker/display.worker.ts, host/game/state.ts.'
  - 'AC-231 command npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=neatenstein reports 0% examples coverage because jest.config.mjs excludes /examples/ from collectCoverageFrom. A temporary config was used for meaningful measurement.'
learning_event:
  gate_id: 'code-coverage'
  session_id: 'green-03-20260723-154616'
  recorded: true
suggested_next_agent: '04-implementing'
handoff_to: '04-implementing (slice-fix for 03-green coverage gaps)'
handoff_note: 'Close coverage gaps on touched examples/neatenstein/ files. Priority: worker/display.worker.ts (large uncovered runtime), renderer/floor.ts (uncovered ceiling paths), renderer/pulse.ts (branch-only gap), renderer/map.ts (single line/branch), constants.ts, host/game/state.ts. Add owner-local tests for reachable paths; remove dead code for unreachable branches. Re-dispatch 05-green-testing afterward.'
```

#### 03-red-testing slice `03-red`

### 03-red-testing slice `03-red`

````yaml
verifier: '03-red-testing'
timestamp: '2026-07-23T18:24:00.000-04:00'
status: '[DONE]'
red-evidence: true
verification_summary:
  - 'Owner-local red tests authored and confirmed failing for ceiling mirror renderer, ceiling pulses, 42x42 map generation, DDA step-cap contract, and 42x42 spawn/combat range constants.'
  - 'All tests follow the single-expect rule and use deterministic seed 42 fixtures.'
  - 'Cortex index rebuilt and is fresh (1664 documents, 13 chunks).'
  - 'Step-packet gate passes; 03-red slice status advanced from [PLANNED] to [DONE].'
files_changed:
  - 'examples/neatenstein/browser-entry/renderer/map.test.ts'
  - 'examples/neatenstein/browser-entry/renderer/raycast.test.ts'
  - 'examples/neatenstein/browser-entry/renderer/pulse.test.ts'
  - 'examples/neatenstein/browser-entry/renderer/floor.test.ts'
  - 'examples/neatenstein/browser-entry/host/game/constants.test.ts'
  - 'examples/neatenstein/browser-entry/host/game/state.test.ts'
  - 'examples/neatenstein/browser-entry/host/game/waves.test.ts'
red_commands:
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/map\\.test\\.ts$'"
    expected_state: red
    result: 'FAIL: 3 failed, 1 passed (42x42 length, perimeter walls, central clearance fail against current 24x24 map).'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/raycast\\.test\\.ts$'"
    expected_state: red
    result: 'FAIL: 1 failed, 6 passed (NEATENSTEIN_DDA_MAX_STEPS export missing).'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/pulse\\.test\\.ts$'"
    expected_state: red
    result: 'FAIL: 4 failed, 10 passed (ceiling pulse layer constants, default layer tag, ceiling emission, and upward motion missing).'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/floor\\.test\\.ts$'"
    expected_state: red
    result: 'FAIL: TypeScript compile-time error — drawNeatensteinCeiling and renderNeatensteinCeiling are not exported from floor.ts.'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/host/game/.*\\.test\\.ts$'"
    expected_state: red
    result: 'FAIL: 4 failed across constants.test.ts/state.test.ts/waves.test.ts (beam range < 42*sqrt(2), spawn centers 12.5 not 21.5, enemy spawns outside 42x42 bounds).'
handoff_to: '03-ceiling (04-implementing)'
handoff_note: 'Red contracts are in place. The 03-ceiling implementation agent must add drawNeatensteinCeiling/renderNeatensteinCeiling to floor.ts and ceiling-layer exports/logic to pulse.ts. The 03-map implementation agent must change NEATENSTEIN_MAP_SIZE to 42, export NEATENSTEIN_DDA_MAX_STEPS >= 84, raise beam range, and fix waves.ts spawn math.'
gate_verdicts:
  - gate: step-packet
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@20333","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@24715"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: cortex-index
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=cortex-index'
    raw_json: '{"pass":true,"evidence":{"index_documents":1664,"index_fresh":true,"corpus_mcp_alive":true,"workflow_mcp_alive":true,"snapshot_age_seconds":5342,"snapshot_indexed_at":"2026-07-23T18:17:50.367Z"},"fixHint":null,"owner":"00-helping"}'

### 05-green-testing slice `03-green` (attempt)

```yaml
verifier: '05-green-testing'
timestamp: '2026-07-23T14:35:29-04:00'
status: '[PLANNED]'
green-evidence: false
verification_summary:
  - 'Preflight gates pass: plan-slice-quality, step-packet.'
  - 'Focused Jest suites for 03-ceiling changes pass: floor.test.ts (19/19), pulse.test.ts (14/14).'
  - 'Focused Jest suites for 03-map changes pass: map.test.ts (4/4), raycast.test.ts (7/7).'
  - 'Focused Jest suite for 03-map changes FAILS: host/game/*.test.ts (133/134 pass, 1 fail in waves.test.ts).'
  - 'Failure root cause: waves.ts still spawns enemies around world origin (0,0) using NEATENSTEIN_ENEMY_SPAWN_RADIUS=8, so floor(position) can be negative; AC-216 requires every enemy inside the 42x42 map bounds (0..42).'
  - 'Fix target: offset spawn positions by the map center (NEATENSTEIN_SPAWN_CENTER_X / NEATENSTEIN_SPAWN_CENTER_Y) or clamp to interior bounds in examples/neatenstein/browser-entry/host/game/waves.ts.'
  - 'Deferred until implementation fix: coverage-guard on touched examples/neatenstein/ files, npm run lint, visible-browser smoke test.'
focused_commands:
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/floor\\.test\\.ts$'"
    result: 'PASS: 19/19'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/map\\.test\\.ts$'"
    result: 'PASS: 4/4'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/raycast\\.test\\.ts$'"
    result: 'PASS: 7/7'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/renderer/pulse\\.test\\.ts$'"
    result: 'PASS: 14/14'
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/neatenstein/browser-entry/host/game/.*\\.test\\.ts$'"
    result: 'FAIL: 133/134 pass; waves.test.ts AC-216 spawn-bounds test fails (enemy outside 42x42 map).'
failing_test:
  file: 'examples/neatenstein/browser-entry/host/game/waves.test.ts:142'
  assertion: 'expect(outOfBounds).toBe(false)'
  expected: 'false (all enemies inside 42x42 bounds)'
  actual: 'true (at least one enemy outside 42x42 bounds)'
  fix_target: 'examples/neatenstein/browser-entry/host/game/waves.ts:69-72'
  fix_hint: 'Spawn enemies relative to map center (NEATENSTEIN_SPAWN_CENTER_X, NEATENSTEIN_SPAWN_CENTER_Y) or clamp to interior bounds so floor(position) is always in [0,42).'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@28749","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@33131"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
blockers:
  - 'AC-216 / 03-red: 42x42 map spawn bounds test in waves.test.ts fails because waves.ts spawns around origin instead of map center.'
suggested_next_agent: '04-implementing'
handoff_to: '04-implementing (slice-fix for 03-map)'
handoff_note: 'Fix enemy spawn math in waves.ts so every spawned enemy has floor(position) in [0,42) for both x and y. Re-run examples/neatenstein/browser-entry/host/game/.*\.test.ts$ until green, then re-dispatch 05-green-testing for 03-green.'
````

#### 01-planning green light (Step 03 patch)

```yaml
verifier: '01-planning (fresh context)'
timestamp: '2026-07-23T13:44:07.150-04:00'
green-light: true
status: green-light
verification_summary:
  - 'Step 03 YAML block is complete with phase/step/title/status/goal/tdd_sequence/expansion/skills/validation/acceptance_criteria/constitution_check/slices and four traceable slices (03-red, 03-ceiling, 03-map, 03-green).'
  - 'All Step 03 slices are within the 4-hour limit (2h, 3h, 2h, 2h) and the step contains 4 slices (≤5 limit); dependencies are linear and acyclic (03-red → 03-ceiling → 03-map → 03-green).'
  - 'Step 03 acceptance criteria (AC-218..AC-229) are observable, implementation-agnostic, and mapped to targeted Jest suites or visible-browser smoke; the 42x42 map decision is recorded as DR-20260723-01.'
  - 'Step 03 green validated and Step 04 (User confirmation gate) completed [DONE]; Phase 3 is [PLANNED] and awaiting explicit user go-ahead before Step 01 slices are authored.'
  - 'All required gates pass with zero warnings: plan-slice-quality, step-packet, plan-sync, and validate-plan-sync.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format with zero planReadinessWarnings after the green-light marker was recorded in ## Latest validation evidence.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@20557","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@24939"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: plan-sync
    pass: true
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    raw_json: '{"pass":true,"evidence":{"wipPlans":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md"],"missingFromReadme":[],"missingFromRoadmap":[],"plansChecked":7},"fixHint":"All WIP plans are correctly registered in README and Roadmap.","owner":"validate-plan-sync.mjs"}'
  - gate: validate-plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    raw_json: '{"name":"plan sync","ok":true,"issues":[],"counts":{"errors":0,"warnings":0},"summaryText":"PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)","plan":{"path":"plans/Neon_Shooter_NGE_Demo.plans.md","status":"WIP"},"downstreamTrackers":["plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md","plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md","plans/Racing_Perception_Redesign.plans.md","plans/mcp-active-binding.plans.md"]}'
blockers: []
execution_note: 'Execution-phase agents (03-red-testing, 04-implementing, 05-green-testing) may now be dispatched for Step 03 slices.'
```

#### 01-planning green light

```yaml
verifier: 01-planning (fresh context)
timestamp: 2026-07-23T08:23:29-04:00
green-light: true
status: green-light
verification_summary:
  - 'Phase 1 is [DONE] and satisfies the dependency precondition for Phase 2.'
  - 'Phase 2 implementation work is [DONE] and green validated; Phase 2 Step 02 is now a focused bundle-path resolution fix. Step 03 was restructured to the ceiling mirror and larger map; the user confirmation gate moved to Step 04.'
  - 'Root cause identified: source examples/neatenstein/index.html uses an inverted /docs/examples/ regex that the copy-examples rewriter does not normalize.'
  - 'Fix approach chosen: align source HTML with the Flappy pattern, regenerate docs copy, update host-shell.test.ts, and run a visible-browser smoke test.'
  - 'Step 02 step-level YAML block is complete with 3 slices (02-fix-red, 02-fix-impl, 02-fix-green), all ≤ 4 hours, sequential dependencies, and traceable AC identifiers.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@15830","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@19545"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: plan-sync
    pass: true
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    raw_json: '{"pass":true,"evidence":{"wipPlans":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md"],"missingFromReadme":[],"missingFromRoadmap":[],"plansChecked":7},"fixHint":"All WIP plans are correctly registered in README and Roadmap.","owner":"validate-plan-sync.mjs"}'
  - gate: validate-plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    raw_json: '{"name":"plan sync","ok":true,"issues":[],"counts":{"errors":0,"warnings":0},"summaryText":"PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)","plan":{"path":"plans/Neon_Shooter_NGE_Demo.plans.md","status":"WIP"},"downstreamTrackers":["plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md","plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md","plans/Racing_Perception_Redesign.plans.md","plans/mcp-active-binding.plans.md"]}'
```

#### 04-implementing slice 05-impl-clearance

### 04-implementing slice 05-impl-clearance

```yaml
implementer: '04-implementing'
slice_id: '05-impl-clearance'
timestamp: '2026-07-23T16:40:29-04:00'
status: done
changed_files:
  - 'examples/neatenstein/browser-entry/renderer/map.ts'
  - 'docs/assets/neatenstein.bundle.js'
  - 'docs/assets/neatenstein.bundle.js.map'
  - 'docs/assets/neatenstein.worker.esm.js'
  - 'docs/assets/neatenstein.worker.esm.js.map'
change_summary: 'CENTRAL_ARENA_CLEARANCE_CELLS increased from 2 to 4; rebuilt stale Neatenstein browser bundles so the 4-cell clearance is reflected in docs/assets/.'
preflight:
  - command: 'npx tsc --noEmit -p tsconfig.test.json'
    result: 'exit 2 due to pre-existing errors in examples/racing_curriculum/ (unrelated to this slice); examples/neatenstein/browser-entry/renderer/map.ts produced no type errors.'
  - command: 'npm run lint'
    result: 'exit 0 (114 pre-existing warnings, 0 errors); examples/neatenstein/browser-entry/renderer/map.ts produced no new issues.'
  - command: 'npx prettier --check examples/neatenstein/browser-entry/renderer/map.ts'
    result: 'exit 0'
  - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/map.test.ts --selectProjects default'
    result: 'PASS default examples/neatenstein/browser-entry/renderer/map.test.ts (4 tests passed)'
bundle_rebuild:
  command: 'node scripts/build-neatenstein.mjs'
  result: 'exit 0; regenerated docs/assets/neatenstein.bundle.js (14.7kb) + .map, docs/assets/neatenstein.worker.esm.js (22.9kb) + .map'
  source_mtime: '2026-07-23T16:40:52-04:00 (examples/neatenstein/browser-entry/renderer/map.ts)'
  bundle_mtime: '2026-07-23T16:50:44-04:00 (docs/assets/neatenstein.bundle.js, docs/assets/neatenstein.worker.esm.js)'
  verification: 'Bundle timestamps are newer than map.ts change, confirming the 4-cell central clearance is now included in the rebuilt browser artifacts.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format with zero planReadinessWarnings.'
    command: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md --slice=05-impl-clearance'
tests_not_run: 'Per 04-implementing contract, broad Jest execution deferred to 05-green-testing; a focused map-suite verification was run as part of this bundle-rebuild pass.'
next: 'User manually confirmed 4-cell clearance browser smoke; Phase 2 complete.'
PlanUpdate:
  changed_files:
    - 'examples/neatenstein/browser-entry/renderer/map.ts'
    - 'docs/assets/neatenstein.bundle.js'
    - 'docs/assets/neatenstein.bundle.js.map'
    - 'docs/assets/neatenstein.worker.esm.js'
    - 'docs/assets/neatenstein.worker.esm.js.map'
  preflight:
    - 'node scripts/build-neatenstein.mjs'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/map.test.ts --selectProjects default'
    - 'npm run lint'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/map.test.ts --selectProjects default'
  rollback:
    - 'node scripts/build-neatenstein.mjs'
  next: 'User manually confirmed 4-cell clearance browser smoke; Phase 2 complete.'
```

#### 01-planning Step 05 prior verification

### 01-planning Step 05 prior verification

```yaml
verifier: '01-planning (fresh context)'
timestamp: '2026-07-23T16:36:25.599-04:00'
green-light: true
status: green-light
verification_summary:
  - 'Step 05 YAML block is complete with phase/step/title/status/goal/tdd_sequence/expansion/skills/validation/acceptance_criteria/constitution_check/slices and three traceable slices (05-red-clearance, 05-impl-clearance, 05-green-clearance).'
  - 'All Step 05 slices are within the 4-hour limit (1h, 1h, 2h) and the step contains 3 slices (≤5 limit); dependencies are linear and acyclic (05-red-clearance → 05-impl-clearance → 05-green-clearance).'
  - 'Step 05 acceptance criteria (AC-237..AC-241) are observable, implementation-agnostic, and mapped to targeted Jest suites or visible-browser smoke.'
  - 'Execution-phase agents (03-red-testing, 04-implementing, 05-green-testing) may now be dispatched for Step 05 slices.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format with zero planReadinessWarnings.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@48768","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@67122"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: plan-sync
    pass: true
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    raw_json: '{"pass":true,"evidence":{"wipPlans":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"missingFromReadme":[],"missingFromRoadmap":[],"plansChecked":7},"fixHint":"All WIP plans are correctly registered in README and Roadmap.","owner":"validate-plan-sync.mjs"}'
  - gate: validate-plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    raw_json: '{"name":"plan sync","ok":true,"issues":[],"counts":{"errors":0,"warnings":0},"summaryText":"PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)","plan":{"path":"plans/Neon_Shooter_NGE_Demo.plans.md","status":"WIP"},"downstreamTrackers":["plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md","plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md","plans/Racing_Perception_Redesign.plans.md","plans/mcp-active-binding.plans.md"]}'
blockers: []
execution_note: 'Dispatch 03-red-testing for slice 05-red-clearance, then 04-implementing for slice 05-impl-clearance, then 05-green-testing for slice 05-green-clearance. Each agent loads context via Cortex MCP / get_slice_context.'
```

#### 01-planning Phase 2 finalization

### 01-planning Phase 2 finalization

```yaml
verifier: '01-planning (current session)'
timestamp: '2026-07-23T17:00:00-04:00'
green-light: true
status: green-light
verification_summary:
  - 'Phase 2 is [DONE]; Step 05 is [DONE] and compressed to a concise coverage note in plans/Neon_Shooter_NGE_Demo.plans.md and plans/Neon_Shooter_NGE_Demo.logs.md §Phase 2.'
  - 'CENTRAL_ARENA_CLEARANCE_CELLS changed from 2 to 4 in examples/neatenstein/browser-entry/renderer/map.ts; matching map.test.ts assertion passes.'
  - 'User manually confirmed the 4-cell central arena clearance at http://localhost:8080/docs/examples/neatenstein/index.html.'
  - 'Phase 3 is [WIP] with Step 01 [PLANNED] but NOT expanded; awaiting explicit user go-ahead before authoring slices.'
  - 'Plan gates pass with zero warnings: plan-sync, step-packet, plan-slice-quality.'
gate_verdicts:
  - gate: plan-sync
    pass: true
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    raw_json: '{"pass":true,"evidence":{"wipPlans":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md"],"missingFromReadme":[],"missingFromRoadmap":[],"plansChecked":7},"fixHint":"All WIP plans are correctly registered in README and Roadmap.","owner":"validate-plan-sync.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format with zero planReadinessWarnings.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@26746","plans/mcp-active-binding.plans.md:yaml@28199"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit and the 5-slice-per-step limit.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: validate-plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    raw_json: '{"name":"plan sync","ok":true,"issues":[],"counts":{"errors":0,"warnings":0},"summaryText":"PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)","plan":{"path":"plans/Neon_Shooter_NGE_Demo.plans.md","status":"WIP"},"downstreamTrackers":["plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md","plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md","plans/Racing_Perception_Redesign.plans.md","plans/mcp-active-binding.plans.md"]}'
blockers: []
execution_note: 'Phase 2 execution is complete. Do not dispatch Phase 3 execution-phase agents until the user explicitly requests Phase 3 work and a fresh 01-planning instance authors and verifies Phase 3 Step 01 slices.'
```

## Archived PlanUpdate packets

#### Slice-fix 03-map: enemy spawn separation from player spawn

### Slice-fix 03-map: enemy spawn separation from player spawn

```yaml
PlanUpdate:
  slice_id: '03-map'
  status: '[DONE]'
  fix_reason: 'AC-230 regression: the 03-map spawn-center fix placed enemies in a disk around the map center, which is also the player spawn point. Enemies could spawn inside NEATENSTEIN_CONTACT_RANGE_CELLS (0.5) and kill the player via contact damage in ~4.7 s instead of the required 15–25 s episode duration.'
  changed_files:
    - examples/neatenstein/browser-entry/host/game/constants.ts
    - examples/neatenstein/browser-entry/host/game/waves.ts
    - examples/neatenstein/browser-entry/host/game/waves.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run quality:folder -- --folder=examples/neatenstein/browser-entry/host/game'
    - 'npx prettier --check examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/game/waves.test.ts'
  preflight_results:
    tsc_base: 'pass'
    tsc_test: 'pass for touched files; pre-existing errors in unrelated src/**/*.test.ts files'
    quality_folder: 'PASS: 0 TypeScript diagnostics, 0 ESLint errors, 76/76 JSDoc symbols, 0 lcov entries below 100%'
    prettier: 'pass for all touched files'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/waves.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game'
  focused_jest_result:
    command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/waves.test.ts'
    result: '16/16 tests passed, 1 suite passed'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/constants.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/waves.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/waves.test.ts'
  next: 'Re-dispatch 05-green-testing for slice 03-green and attach coverage-guard evidence'
```

#### Slice 03-map implementation handoff

### Slice 03-map implementation handoff

```yaml
PlanUpdate:
  slice_id: '03-map'
  status: '[DONE]'
  changed_files:
    - examples/neatenstein/browser-entry/constants.ts
    - examples/neatenstein/browser-entry/renderer/map.ts
    - examples/neatenstein/browser-entry/renderer/raycast.ts
    - examples/neatenstein/browser-entry/host/game/constants.ts
    - examples/neatenstein/browser-entry/constants.test.ts
    - examples/neatenstein/browser-entry/renderer/raycast.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/constants.ts examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/constants.test.ts examples/neatenstein/browser-entry/renderer/raycast.test.ts'
  preflight_results:
    tsc_base: 'pass'
    tsc_test: 'pass'
    lint: '0 errors, 114 pre-existing warnings (none from edited files)'
    prettier: 'pass after npx prettier --write on edited files'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/map.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/raycast.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/constants.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/state.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/waves.test.ts'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/constants.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/map.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/raycast.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/constants.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/constants.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/raycast.test.ts'
  next: 'Run 05-green-testing slice 03-green and attach coverage-guard evidence'
```

#### Slice 03-ceiling implementation handoff

### Slice 03-ceiling implementation handoff

```yaml
PlanUpdate:
  slice_id: '03-ceiling'
  status: '[DONE]'
  changed_files:
    - examples/neatenstein/browser-entry/renderer/floor.ts
    - examples/neatenstein/browser-entry/renderer/pulse.ts
    - examples/neatenstein/browser-entry/worker/display.worker.ts
    - examples/neatenstein/browser-entry/renderer/pulse.test.ts
    - examples/neatenstein/browser-entry/renderer/floor.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/floor.ts examples/neatenstein/browser-entry/renderer/pulse.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/renderer/pulse.test.ts examples/neatenstein/browser-entry/renderer/floor.test.ts'
  preflight_results:
    tsc_base: 'pass'
    tsc_test: 'pass for touched files; pre-existing errors in unrelated src/**/*.test.ts files'
    lint: '0 errors, 114 pre-existing warnings'
    prettier: 'pass for all touched files'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/floor.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/pulse.test.ts'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/floor.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/pulse.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/pulse.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/floor.test.ts'
  next: 'Run 05-green-testing on the two focused renderer test suites, then hand off to 03-map implementation'
```
