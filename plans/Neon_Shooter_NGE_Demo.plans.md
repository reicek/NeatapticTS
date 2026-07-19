# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [WIP] · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/Neon_Shooter_NGE_Demo.research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

---

## Current state

Claim: 04-implementing @ 2026-07-18T19:44:59-04:00

**Active slice:** `01-floor-reuse` — Floor renderer (Flappy ground grid reuse, camera-adapted) ([WIP] · implementation complete, awaiting GREEN validation). Previous slice `01-neon-walls` is [DONE] with GREEN validation passed. Next slices after floor reuse are `01-sprites-zbuffer` [PLANNED] and `01-pulse-system` [PLANNED].

## Thesis

> **"Train your own killer — then survive it."**

The human's deaths train the enemies (act one: you are the teacher who gets killed), then the player fights to survive what they trained (act two: survival). The NGE-naive viewer sees "enemies learn from deaths" in AI modes (act one demo) and "I trained them, now I fight them" in human modes (act two).

---

## Confirmed Design Parameters (locked — do not change without re-running consensus)

- **Tier-aware preset table:** GPU = 2048 nodes / 2048 variants single-pass; WebWorker = 256/256 across 4 workers; CPU fallback = 128/128 synchronous.
- **Budget surfaces separated:** `NgeSubstrateBudgetOverride.maxNodes`/`maxEdges` owns DNA-level node/edge caps. `parallelVariantCount` on the acceleration/lifecycle-runner config owns runtime variant batching. `swarmMaxNodes` demo prop → `NgeSubstrateBudgetOverride`. `swarmMaxVariants` demo prop → runtime acceleration config. Two distinct config surfaces, never conflated.
- **4 rendered modes:**
  1. **ARMS RACE** (default, FLAGSHIP) — NGE main vs co-evolved MLP enemies.
  2. **SWARM** — NGE main vs WeightSharedCohort NGE swarm.
  3. **HUMAN vs MLP** — human player vs co-evolved MLP enemies.
  4. **HUMAN vs SWARM** — human player vs WeightSharedCohort NGE swarm.
- **2 test-only static-enemy modes** (not rendered) for deterministic baseline fitness.
- **Human modes evolve on every death:** record last 10 seconds of hero gameplay as the fitness replay buffer. If hero dies too fast (<10s), use last complete recording. If no recording exists yet, skip mutation this cycle (no evolution on empty input).
- **Human modes use CPU preset variant count (128 variants, not 2048)** for determinism. The "2048 variants" applies only to ARMS RACE and SWARM (AI vs AI) modes.
- **SWARM mode** uses WeightSharedCohort (one DNA, shared weights, coordinate injection). Swarm evolves via full NGE lifecycle, growing toward node cap.
- **Main agent:** full NGE, tier-capped topology growth (up to tier limit — every grow action obeys `NgeSubstrateBudgetOverride`, never "uncapped"), 2048 parallel weight variants per generation on GPU (AI modes).
- **MLP enemies:** fixed topology 8→6→4→2, weight-only co-evolution, throttled (every 5th gen, 32 variants).
- **Minimal co-evolution harness:** single main agent + enemy population (MLP) or enemy singleton (SWARM). Do NOT duplicate the PredatorPrey two-population symmetric harness.
- **8 concurrent enemy cap** (all modes including SWARM). SWARM ramps coordination density, not enemy count.

---

## Implementation phases

### Phase 1 — World & Renderer (visualizer-owned) [WIP]

**Goal:** Raycasting neon renderer + frame protocol + audio.

```yaml
phase: 1
title: 'World & Renderer'
status: '[WIP]'
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
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/frame'
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
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/pulse'
  - id: AC-008
    text: 'Pulses are depth-tested against walls'
    validation: 'Browser smoke test'
  - id: AC-009
    text: 'Generation-up fires as an audio-visual pair on the same sim tick'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/audio'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — World & Renderer scaffold and raycaster'
```

- Raycasting renderer (~800 lines): map grid, DDA ray cast, neon wall rendering (pure neon-line with optional line-pattern texture modulation, NOT sampled texels), enemy wireframe sprites, projectiles.
- **Floor: reuse Flappy Bird's synthwave ground grid** (camera-adapted). Reuse `FLAPPY_GROUND_GRID_*` constants, depth-curve/alpha/blur/thickness helpers, and `FLAPPY_NEON_PALETTE` ground colors. Adapt vertical rays to camera yaw rotation. **Pulse system is fake-perspective-anchored** (research §3.3.5): horizontal pulses reuse Flappy helpers unchanged; vertical pulses use world-bearing continuity (cache `worldBearingRad`, match by `Δθ` with 0.1 rad tolerance; off-screen bearings fade, never re-anchor). Pulses render on Layer 2 (dynamic), depth-tested against the z-buffer (§3.4.1). **Pulse emission is sim-tick-driven** (not wall-clock, not frameIndex) for Phase 2 determinism (§3.3.7). **Ambient density** `NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS=3000` (adapted from Flappy's 6000ms, §3.3.6); **event pulses** for generation-up ripple (white-hot expanding ring, 600ms, synced with generation-up sound §3.3.9), enemy death pulse (enemy-hue tint, 400ms), low-health dim (alpha × 0.5 when health < 30%). 8-concurrent-pulse ceiling. See research file §3.3.5–§3.3.9. This gives visual coherence with the Flappy demo and a secondary legibility channel for combat events.
- Tier-aware column count: GPU 320 cols, Worker 240, CPU 160. Glow passes skip on CPU. CPU fallback: lines only, no texture modulation, no glow.
- Worker offload: all NGE inference + enemy AI + projectile physics on workers; renderer reads packed `NeatensteinRenderFrame` (SoA typed arrays, transfer list, zero-copy, requestId-gated). Worker tier may use `OffscreenCanvas` via `transferControlToOffscreen()` for off-main-thread rendering (see research file §3.2.1). **Two render architectures by tier:** (a) CPU/GPU — display worker produces `NeatensteinRenderFrame`, main thread renders; (b) Worker — display worker renders directly via OffscreenCanvas, frame transfer bypassed. On Worker tier, display worker responsibilities = sim tick + NGE inference + OffscreenCanvas render.
- **Render path (tier-gated, see research file §3.2.1):** CPU tier → `ImageData` framebuffer + single `putImageData` (no per-column `fillRect`); Worker tier → `OffscreenCanvas`; GPU tier → stroke + `shadowBlur` (premium). All tiers: `getContext("2d", { alpha: false })`, integer-floored coordinates. Feature-detect `transferControlToOffscreen` and `ctx.filter`; fall back to CPU ImageData path if unavailable.
- **`transferControlToOffscreen()` is irreversible.** On tier downgrade from Worker → CPU/GPU, the host must create a fresh `<canvas>` element (old canvas is permanently worker-owned). `onBackendChange` handler accounts for canvas recreation + re-attach ResizeObserver + re-bind pointer lock.
- **Sprite occlusion:** per-column `Float32Array` z-buffer (not a boolean set) — handles partial occlusion. See research file §3.4.1.
- **Render interpolation:** `lerp(statePrev, stateCurr, alpha)` on each RAF to eliminate 30→60 Hz judder. See research file §4.1.2.
- **Raycaster is a shared build entry** — included in both host bundle (CPU/GPU tiers render on main thread) and worker bundle (Worker tier renders via OffscreenCanvas). Build script configures dual webpack entries. Worker instantiated as module worker: `new Worker(url, { type: 'module' })`.
- 60fps target on GPU tier, 30fps floor on CPU.
- **Audio (Phase 1 deliverable, same weight as renderer):** 6 sounds — fire, enemy hit, player damage, dash, kill, **generation-up** (rising arpeggio, the audio signal of learning) — via WebAudio procedural synthesis (oscillators, zero assets). Positional audio via `StereoPannerNode` + distance attenuation. `AudioContext.resume()` on first click (regardless of mode — AI modes need audio too). **AudioContext is main-thread only** — audio trigger events originate in the display worker and are `postMessage`d to the main thread for synthesis. The generation-up sound fires on the same sim tick as the generation-up floor ripple (research §3.3.9) as an audio-visual pair — audio punches in (200ms), ripple lingers (600ms). See research file §9.
- **Rendering invariant:** distance fog and glow use per-column/per-sprite explicit fill/stroke with layer opacity. NO global `ctx.globalAlpha` passes.
- **Canvas resize:** Phase 1 raycaster owns resize reaction (ResizeObserver → re-derive column stride + re-allocate SoA frame buffers). Phase 7 owns shell/sidebar layout.
- **Tier contract:** column count locks at session-start tier. `onBackendChange` observer re-evaluates tier caps + updates chip label on next RAF (not mid-frame). Re-draw-last-frame fallback coordinates with locked col count until next tier re-bind.
- **Module layout:** `examples/neatenstein/browser-entry/` (host, renderer/raycaster, renderer/sprites, renderer/camera, renderer/frame, ui, constants). `README.md` at `examples/neatenstein/` root (visualizer discovery anchor).
- Reuse: Flappy `WorkerPlaybackFrameSnapshot` SoA pattern, WeakMap buffer pool, `resolveWorkerPlaybackSnapshotTransferList`.

**Acceptance:**

- 60fps on GPU tier (Chrome DevTools trace, no long task > 16ms).
- Neon walls with borders + distance fog; no overdraw outside canvas.
- Frame protocol versioned + transfer-list zero-copy; `requestId` increments.
- 3 audio cues wired and audible.
- `README.md` present at example root.
- Pulses render fake-perspective-anchored: rotating the camera (mouse look) does not cause pulses to swim or snap; a pulse emitted in view remains continuous as it transits the FOV.
- Pulse emission is deterministic: same seed + same inputs → identical pulse positions/timings in a focused replay test (paired with Phase 2 determinism acceptance).
- Pulses are depth-tested against walls (no pulse shows through a wall).
- Generation-up fires as an audio-visual pair (sound + floor ripple on the same sim tick).

#### Step 01: World & Renderer scaffold and raycaster [WIP]

```yaml
phase: 1
step: 1
title: 'World & Renderer scaffold and raycaster'
status: '[WIP]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 01 — Game Logic & FPS State red tests'
skills:
  - 'implementation-standards'
  - 'red-test-contracts'
  - 'green-validation-gates'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-001
    text: 'GPU tier raycaster runs at 60fps with no long task > 16ms'
    validation: 'Chrome DevTools performance trace'
  - id: AC-002
    text: 'Neon walls render with borders and distance fog; no overdraw outside canvas'
    validation: 'Browser smoke test'
  - id: AC-003
    text: 'Frame protocol is versioned, uses transfer-list zero-copy, and requestId increments'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/frame'
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
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/pulse'
  - id: AC-008
    text: 'Pulses are depth-tested against walls'
    validation: 'Browser smoke test'
  - id: AC-009
    text: 'Generation-up fires as an audio-visual pair on the same sim tick'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/audio'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '01-red-phase1'
    title: 'Red tests for Phase 1 scaffold, frame protocol, raycaster, and audio contracts'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/constants.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/frame.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/raycast.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/walls.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/floor.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/pulse.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/interpolate.test.ts'
      - 'examples/neatenstein/browser-entry/audio.test.ts'
    acceptance_criteria:
      - id: AC-010
        text: 'Red tests assert README.md presence, constants exports, and frame protocol shape'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
      - id: AC-011
        text: 'Red tests fail before implementation exists'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
    VALIDATION_EVIDENCE:
      - 'Command: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
      - 'Result: 8 failed suites, 41 failed tests — all failures are module-not-found/readme-not-found, confirming no implementation exists yet.'
    parallelizable: false
    dependencies: []
    next_slice: '01-scaffold'
  - slice_id: '01-scaffold'
    title: 'Module scaffold + constants + README + build script stub'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/README.md'
      - 'examples/neatenstein/browser-entry/constants.ts'
      - 'scripts/build-neatenstein.mjs'
    acceptance_criteria:
      - id: AC-005
        text: 'README.md is present at examples/neatenstein/ root'
        validation: 'ls examples/neatenstein/README.md'
      - id: AC-012
        text: 'Constants file exports tier-aware column counts and versioned frame protocol constants'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/constants'
    VALIDATION_EVIDENCE:
      - 'Created examples/neatenstein/README.md'
      - 'Created examples/neatenstein/browser-entry/constants.ts with all required exports'
      - 'Created scripts/build-neatenstein.mjs dual-entry esbuild stub'
      - 'Preflight: npx tsc --noEmit --skipLibCheck --noResolve examples/neatenstein/browser-entry/constants.ts — OK'
      - 'Preflight: npx eslint examples/neatenstein/browser-entry/constants.ts — 0 issues'
      - 'Preflight: npx prettier --check <new files> — passed after --write'
      - 'Preflight: node --check scripts/build-neatenstein.mjs — syntax OK'
      - 'Note: full npx tsc --noEmit -p tsconfig.test.json fails in pre-existing node_modules/devtools-protocol .d.ts parse error (unrelated to this slice)'
      - "Specialist review (pre-green): implementation-pattern-scout — REQUEST_CHANGES; visualizer-scout — REQUEST_CHANGES. Observations compiled in this slice's tracker and handoff to orchestrator."
      - 'Fix packet applied per specialist review: README.md defers runnable browser demo until later Phase 1 slices, removes the missing `npm run build:neatenstein` script reference, and aligns the build instruction to `node scripts/build-neatenstein.mjs`. scripts/build-neatenstein.mjs guards missing dual entry files and exits 0 with a clear stub message instead of crashing.'
      - 'Preflight re-run post-fix: npx eslint — 0 errors; npx prettier --check — passed; node --check scripts/build-neatenstein.mjs — OK; node scripts/build-neatenstein.mjs — exited 0 with stub message.'
      - 'README-only fix (iteration 2) per visualizer-scout re-review: softened scaffold language, marked every deferred component in the Mermaid diagram as pending, added an explicit "What exists in this slice" section, and noted that renderer/ contains only test stubs. Source files constants.ts and build-neatenstein.mjs were not changed; pre-green review now APPROVED by implementation-pattern-scout and visualizer-scout.'
      - 'GREEN validation (05-green-testing) — 2026-07-18: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/constants → PASS (1 suite, 6/6 tests).'
      - 'GREEN preflight: npx prettier --check examples/neatenstein/browser-entry/constants.ts examples/neatenstein/README.md scripts/build-neatenstein.mjs → PASS (All matched files use Prettier code style!).'
      - 'GREEN preflight: npx eslint examples/neatenstein/browser-entry/constants.ts examples/neatenstein/README.md scripts/build-neatenstein.mjs → PASS (0 errors, 2 ignore warnings for README.md and build-neatenstein.mjs).'
      - 'GREEN preflight: node --check scripts/build-neatenstein.mjs → PASS; node scripts/build-neatenstein.mjs → exited 0 with stub message.'
      - 'RED contract preserved: 7 downstream test files (renderer/frame.test.ts, renderer/raycast.test.ts, renderer/walls.test.ts, renderer/floor.test.ts, renderer/pulse.test.ts, renderer/interpolate.test.ts, audio.test.ts) remain untouched and still fail with module-not-found (right reason).'
      - 'specialist-review gate (node scripts/agent-customization/gates/specialist-review.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md) → pass: true (evidence markers: specialist review, implementation-pattern-scout, visualizer-scout, APPROVE, REQUEST_CHANGES).'
      - 'Slice 01-scaffold confirmed [DONE]; next active slice is 01-frame-protocol [WIP].'
    parallelizable: false
    dependencies:
      - '01-red-phase1'
    next_slice: '01-frame-protocol'
  - slice_id: '01-frame-protocol'
    title: 'Frame protocol SoA + transfer list + requestId gating'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/frame.ts'
    acceptance_criteria:
      - id: AC-003
        text: 'Frame protocol is versioned, uses transfer-list zero-copy, and requestId increments'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/frame'
    VALIDATION_EVIDENCE:
      - 'Implemented examples/neatenstein/browser-entry/renderer/frame.ts with buildNeatensteinRenderFrame and resolveNeatensteinRenderFrameTransferList exports.'
      - 'Reused NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION from constants.ts for frame format/version identifiers.'
      - 'SoA typed arrays sized to columnCount: wallDistances (Float32Array), wallSides (Uint8Array), zBuffer (Float32Array), enemyScreenX (Float32Array), enemyScale (Float32Array), projectileScreenX (Float32Array).'
      - 'requestId increments via module-level counter across frame builds.'
      - 'Transfer list returns all 6 typed-array ArrayBuffers for zero-copy postMessage transfer.'
      - 'Preflight: npx tsc --noEmit --skipLibCheck --noResolve --target ES2023 --module ESNext --moduleResolution Bundler --strict --allowImportingTsExtensions examples/neatenstein/browser-entry/renderer/frame.ts examples/neatenstein/browser-entry/constants.ts → OK'
      - 'Preflight: npx eslint examples/neatenstein/browser-entry/renderer/frame.ts → 0 issues'
      - 'Preflight: npx prettier --check examples/neatenstein/browser-entry/renderer/frame.ts → passed'
      - "Import-convention fix per implementation-pattern-scout pre-green review: changed line 11 from '../constants.ts' to '../constants' (repo uses extensionless ESM imports). Re-ran eslint, prettier, and per-file tsc — all OK."
      - 'Red contract preserved: the other 6 downstream test files (renderer/raycast.test.ts, renderer/walls.test.ts, renderer/floor.test.ts, renderer/pulse.test.ts, renderer/interpolate.test.ts, audio.test.ts) remain untouched and still fail with module-not-found (right reason).'
      - 'GREEN validation (05-green-testing) — 2026-07-18: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/frame → PASS (1 suite, 5/5 tests).'
      - 'GREEN preflight: npx prettier --check examples/neatenstein/browser-entry/renderer/frame.ts → PASS (All matched files use Prettier code style!).'
      - 'GREEN preflight: npx eslint examples/neatenstein/browser-entry/renderer/frame.ts → PASS (0 errors, 0 warnings).'
      - 'Specialist review (pre-green): implementation-pattern-scout — APPROVE; worker-payload-scout — APPROVE (both iterations).'
      - 'specialist-review gate (node scripts/agent-customization/gates/specialist-review.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md) → pass: true.'
      - 'Plan gates post-update: plan-sync → pass: true; step-packet → pass: true; plan-slice-quality → pass: true.'
      - 'Slice 01-frame-protocol confirmed [DONE]; next active slice is 01-raycaster-grid [WIP].'
    parallelizable: false
    dependencies:
      - '01-scaffold'
    next_slice: '01-raycaster-grid'
  - slice_id: '01-raycaster-grid'
    title: 'Map grid + DDA ray cast'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/map.ts'
      - 'examples/neatenstein/browser-entry/renderer/raycast.ts'
    acceptance_criteria:
      - id: AC-013
        text: 'DDA ray cast returns per-column distance and wall side'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/raycast'
    VALIDATION_EVIDENCE:
      - 'Created examples/neatenstein/browser-entry/renderer/map.ts exporting buildNeatensteinMap(seed): deterministic 24x24 Uint8Array wall grid using Park-Miller LCG, perimeter walls, interior scatter density 0.22, central 5x5 clearance.'
      - 'Created examples/neatenstein/browser-entry/renderer/raycast.ts exporting castRayDDA(grid, gridWidth, gridHeight, posX, posY, dirX, dirY) and re-exporting buildNeatensteinMap from ./map.'
      - 'GREEN validation (05-green-testing) — 2026-07-18: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/raycast → PASS (1 suite, 5/5 tests).'
      - 'GREEN preflight: npx prettier --check examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts → PASS (All matched files use Prettier code style!).'
      - 'GREEN preflight: npx eslint examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts → PASS (0 errors, 0 warnings).'
      - 'Specialist review (pre-green): implementation-pattern-scout — APPROVE; determinism-scout — APPROVE.'
      - 'specialist-review gate (node scripts/agent-customization/gates/specialist-review.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md) → pass: true.'
      - 'Plan gates post-update: plan-sync → pass: true; step-packet → pass: true; plan-slice-quality → pass: true.'
      - 'Red contract preserved: the other 5 downstream test files (renderer/walls.test.ts, renderer/floor.test.ts, renderer/pulse.test.ts, renderer/interpolate.test.ts, audio.test.ts) remain untouched and still fail with module-not-found (right reason) — 5 failed suites, 25 failed tests.'
      - 'Slice 01-raycaster-grid confirmed [DONE]; next active slice is 01-neon-walls [WIP].'
    parallelizable: false
    dependencies:
      - '01-scaffold'
    next_slice: '01-neon-walls'
  - slice_id: '01-neon-walls'
    title: 'Neon wall render + CPU ImageData path + distance fog'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/walls.ts'
      - 'examples/neatenstein/browser-entry/renderer/walls.test.ts'
    acceptance_criteria:
      - id: AC-002
        text: 'Neon walls render with borders and distance fog; no overdraw outside canvas'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/walls'
    VALIDATION_EVIDENCE:
      - 'GREEN test: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/walls → PASS (1 suite, 4/4 tests).'
      - 'Preflight: npx prettier --check examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts → PASS (All matched files use Prettier code style!).'
      - 'Preflight: npx eslint examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts → PASS (0 errors, 0 warnings).'
      - 'Red contract preserved: downstream suites (floor.test.ts, pulse.test.ts, interpolate.test.ts, audio.test.ts) remain untouched and fail with module-not-found (right reason) — 4 failed suites, 21 failed tests.'
      - 'Specialist review (pre-green): implementation-pattern-scout — APPROVE; visualizer-scout — APPROVE after fix cycle.'
      - 'Specialist-review gate for target plan (node scripts/agent-customization/gates/specialist-review.gate.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md) → pass: true.'
      - 'Plan-sync gate → pass: true.'
      - 'Step-packet gate → pass: true.'
      - 'Plan-slice-quality gate → pass: true.'
      - 'Note: neataptic-gate-mcp:run_gate_check specialist-review defaults to plans/completed/Agentic_Workflow_Architecture.plans.md and reports pass:false for that unrelated completed plan; target-plan evidence was verified via the script with --plan.'
    parallelizable: false
    dependencies:
      - '01-raycaster-grid'
    next_slice: '01-floor-reuse'
  - slice_id: '01-floor-reuse'
    title: 'Floor renderer (Flappy ground grid reuse, camera-adapted)'
    status: '[WIP]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/floor.ts'
    acceptance_criteria:
      - id: AC-014
        text: 'Floor renderer reuses Flappy ground grid helpers and adapts to camera yaw'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/floor'
    VALIDATION_EVIDENCE:
      - 'Preflight (2026-07-18): targeted tsc with --noResolve cannot resolve relative Flappy imports when only floor.ts is passed; project-level `npx tsc --noEmit -p tsconfig.test.json` produced zero type errors for floor.ts (one unrelated node_modules devtools-protocol syntax error outside slice scope).'
      - 'Preflight: npx prettier --check examples/neatenstein/browser-entry/renderer/floor.ts → PASS (formatted with --write).'
      - 'Preflight: npx eslint examples/neatenstein/browser-entry/renderer/floor.ts → PASS (0 errors, 0 warnings).'
      - 'Red contract preserved: pulse.test.ts, interpolate.test.ts, audio.test.ts remain untouched and fail with module-not-found (right reason). walls.test.ts remains green.'
      - 'Implementation note: renderNeatensteinFloor uses default 320×240 canvas, horizon at height/2, vanishing point X = width*0.5 + sin(yaw)*width*0.25, so yaw=0 and yaw=PI/4 produce different average lineTo X.'
    parallelizable: false
    dependencies:
      - '01-neon-walls'
    next_slice: '01-sprites-zbuffer'
  - slice_id: '01-sprites-zbuffer'
    title: 'Sprite rendering + z-buffer occlusion'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/zbuffer.ts'
    acceptance_criteria:
      - id: AC-015
        text: 'Sprites project with per-column z-buffer occlusion'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/sprites'
    parallelizable: false
    dependencies:
      - '01-neon-walls'
    next_slice: '01-pulse-system'
  - slice_id: '01-pulse-system'
    title: 'Pulse system (sim-tick emission, world-bearing continuity, event pulses, depth test)'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/pulse.ts'
    acceptance_criteria:
      - id: AC-006
        text: 'Pulses render fake-perspective-anchored without swim or snap during camera rotation'
        validation: 'Browser smoke test'
      - id: AC-007
        text: 'Pulse emission is deterministic: same seed + same inputs produce identical pulse positions/timings'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/pulse'
      - id: AC-008
        text: 'Pulses are depth-tested against walls'
        validation: 'Browser smoke test'
    parallelizable: false
    dependencies:
      - '01-floor-reuse'
      - '01-sprites-zbuffer'
    next_slice: '01-audio'
  - slice_id: '01-audio'
    title: 'Audio (6 WebAudio sounds, StereoPanner, main-thread synthesis, generation-up pair)'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/audio.ts'
    acceptance_criteria:
      - id: AC-004
        text: 'At least 3 audio cues are wired and audible'
        validation: 'Manual browser check'
      - id: AC-009
        text: 'Generation-up fires as an audio-visual pair on the same sim tick'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/audio'
    parallelizable: false
    dependencies:
      - '01-frame-protocol'
    next_slice: '01-worker-offload'
  - slice_id: '01-worker-offload'
    title: 'Worker offload + OffscreenCanvas path + tier-gated render path'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    acceptance_criteria:
      - id: AC-016
        text: 'Worker tier renders via OffscreenCanvas; CPU/GPU tiers render from packed frame'
        validation: 'Browser smoke test'
    parallelizable: false
    dependencies:
      - '01-frame-protocol'
    next_slice: '01-interpolation-resize'
  - slice_id: '01-interpolation-resize'
    title: 'Render interpolation + canvas resize + tier contract'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/interpolate.ts'
      - 'examples/neatenstein/browser-entry/host/resize.ts'
    acceptance_criteria:
      - id: AC-017
        text: 'Renderer interpolates between previous and current state each RAF'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/interpolate'
      - id: AC-018
        text: 'Canvas resize re-derives column stride and reallocates SoA frame buffers'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/resize'
    parallelizable: false
    dependencies:
      - '01-worker-offload'
    next_slice: '01-host-shell'
  - slice_id: '01-host-shell'
    title: 'Host HTML shell at docs/examples/neatenstein/index.html'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'docs/examples/neatenstein/index.html'
    acceptance_criteria:
      - id: AC-022
        text: 'docs/examples/neatenstein/index.html exists and mirrors the docs/examples/racing_curriculum/index.html host-shell pattern'
        validation: 'ls docs/examples/neatenstein/index.html && diff -u <(sed "s/racing/neatenstein/g" docs/examples/racing_curriculum/index.html) docs/examples/neatenstein/index.html || true'
      - id: AC-023
        text: 'Host shell loads the host bundle from docs/assets/neatenstein.bundle.js'
        validation: 'grep -F "neatenstein.bundle.js" docs/examples/neatenstein/index.html'
      - id: AC-024
        text: 'Host shell mounts a canvas element for the renderer'
        validation: 'grep -E "<canvas|id=.*canvas" docs/examples/neatenstein/index.html'
      - id: AC-025
        text: 'Host shell instantiates the module worker from docs/assets/neatenstein.worker.esm.js'
        validation: 'grep -F "neatenstein.worker.esm.js" docs/examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '01-scaffold'
      - '01-worker-offload'
    next_slice: '01-green-phase1'
  - slice_id: '01-green-phase1'
    title: 'Green validation and coverage guard for Phase 1'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-019
        text: 'All Phase 1 focused suites pass'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
      - id: AC-020
        text: '100% coverage on touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein'
      - id: AC-021
        text: 'Lint passes'
        validation: 'npm run lint'
      - id: AC-026
        text: 'Browser smoke test: docs/examples/neatenstein/index.html opens at http://localhost:8080/docs/examples/neatenstein/index.html with no console errors'
        validation: 'Browser smoke test (browser-harness-specialist)'
    parallelizable: false
    dependencies:
      - '01-pulse-system'
      - '01-audio'
      - '01-interpolation-resize'
      - '01-host-shell'
    next_slice: null
```

### Phase 2 — Game Logic & FPS State (visualizer + benchmark-owned) [PLANNED]

**Goal:** FPS game state, controls, deterministic episode.

[PLANNED] Step 01 — Game Logic & FPS State red tests (deferred until phase becomes active).

- FPS game state: health, ammo, enemy waves (continuous trickle, not clumps), collision, projectiles (hitscan neon beam).
- Controls: WASD + mouse look (pointer lock with `unadjustedMovement: true`, see research file §4.2.6) + left-click fire + Space dash (200ms i-frames). Arrow-key look fallback if no pointer lock. Touch drag-to-look fallback for iOS Safari. **On Worker tier:** `mousemove` deltas forwarded from main thread to display worker via `postMessage` (pointer lock is on the canvas DOM element, which stays main-thread even with OffscreenCanvas).
- One weapon only (neon beam). No weapon switching.
- Wave cap: 8 concurrent enemies for legibility (all modes).
- **Target episode length:** 15–25s (short enough that generations fire frequently).
- **Minimum generation cadence:** ≥2 generations per minute in AI modes. First 60s = montage of visible change, not a wait.
- Game loop lives in `examples/neatenstein/browser-entry/host/game/` module.

**Acceptance:**

- Deterministic episode: same seed + same inputs → identical final world state (focused replay test, reuse racing `environment.step` determinism test pattern).
- Collision correct; projectiles render as tracers.
- Episode length and cadence within targets.

### Phase 3 — Asymmetric Co-evolution Harness (benchmark-owned, core-reviewed) [PLANNED]

**Goal:** Minimal single-main + enemy-population co-evolution harness.

[PLANNED] Step 01 — Asymmetric Co-evolution Harness red tests (deferred until phase becomes active).

- Single main agent lifecycle runner with combat telemetry adapters.
- Enemy population abstraction: MLP backend (fixed 8→6→4→2, weight-only, throttled every 5th gen, 32 variants) and WeightSharedCohort backend (singleton DNA, shared weights, coordinate injection).
- **Rolling opponent snapshot (asymmetric):**
  - Main evaluates against frozen enemy snapshot (hall-of-fame + recent sample).
  - Enemy evaluates against frozen main snapshot (main representative = current best variant from last generation).
  - Refresh every 5 gens (MLP), every 3 gens (SWARM — explicit).
- **Deterministic seed-pack fairness contract:** all variants (2048 main / 32 MLP / 128 human) evaluated against a fixed frozen seed pack per generation — same seeds for all variants in a generation.
- **Generation barrier:** main completes full variant batch + selection BEFORE enemy update gate fires.
- **Authoritative world owner:** one designated display worker owns the authoritative display world; batch-evaluation workers are stateless episode workers (reuse racing's worker-authoritative pattern).
- **GPU-tier determinism:** variant selection uses index-stable argmax with deterministic tie-break by variant id (lowest id wins ties).
- **Combat fitness composite:** `survivalTicks + damageDealt + kills - damageTaken - aimMissRate + performance-gated complexityBonus - parsimonyDensityPenalty` (reuse racing's `RacingQualitySignal` pattern, same 800–3000 syn/neuron parsimony band).

**Acceptance:**

- `genBarrier(seed=K)` reproduces identical `M_N` and enemy state on GPU tier (index-stable argmax + tie-break).
- Snapshot refresh cadence enforced (5 gens MLP, 3 gens SWARM).
- MLP throttle fires only on `N % 5 == 0`.
- Seed-pack fairness: all variants in a generation see identical seeds.

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

- Replay buffer: last 10s ring buffer of hero gameplay (deterministic fixed-timestep recording). On death, freeze as fitness replay buffer.
- **128 variants (CPU preset)** evaluated by replaying the recorded scenario. Human is hero; variants control enemies.
- Edge cases: death <10s → last complete recording; no recording → skip mutation; long survival → tail 10s; corrupted → fallback to last complete or skip.
- Human modes use CPU-only preset (128/128) for determinism (no worker ordering nondeterminism).
- **Focused determinism check:** same recording → same fitness within tolerance (focused test).
- **Death feedback:** freeze frame 400ms → death scrub (10s @ 4×, lethal moment highlighted) → "THEY LEARNED FROM THAT" banner → generation/wave tick → instant respawn (no menu, no "try again?" button).
- **Player-favoring rubber-band:** 0.3× enemy learn rate for first 3 deaths.
- Survival-time sparkline in HUD.
- **Human-mode entry moment:** pressing 3 or 4 triggers a 1.5s camera handoff — camera flies into the agent's POV, "NOW YOU" card (neon, 1s), then spawn.

**Acceptance:**

- All 128 variants see identical replay (fairness test).
- Edge cases each have a focused test: death <10s → last complete; no recording → skip mutation; long survival → tail 10s; corrupted → fallback.
- Same-recording → same-fitness determinism check passes.
- Death feedback loop feels responsive (no menu friction).
- Rubber-band prevents instant-quit (first 3 deaths feel winnable).

### Phase 7 — Mode Dial, Stats, UI Polish (visualizer + game-director-owned) [PLANNED]

**Goal:** Legible mode dial + stats + thesis delivery.

[PLANNED] Step 01 — Mode Dial, Stats, UI Polish red tests (deferred until phase becomes active).

- **Mode dial** (top-right): ARMS RACE (default, FLAGSHIP tag) → SWARM → HUMAN vs MLP → HUMAN vs SWARM. Keyboard shortcuts 1–4.
- **Human toggle** (sub-switch, only for HUMAN modes): "EVOLVE ON DEATH" on/off.
- **Acceleration chip:** "GPU (2048×2048)" / "WORKER (256×256)" / "CPU (128×128)" (reuse racing chip pattern, `resolveAccelerationChipPresentation` extended ADDITIVELY with `batchParallelCount` — parity-preserving, no racing regression).
- **Stats overlay** (top-left, persistent HUD, iconified/condensed): generation, best fitness, enemy adaptation Δ, swarm node count, tier, health, ammo, enemies alive, FPS, mode.
- **Generation counter** (top-center, large, pulsing) — most prominent HUD element.
- **Two independent behavior-diff signals:**
  - (a) Behavioral ghost replay: 2s ghost of previous gen's death in corner on each new gen (AI modes). De-risked with a Phase 1–2 spike (prove deterministic replay of last gen before building on it). Fallback: death-position marker if full replay too costly.
  - (b) "First time it did X" callout: neon text flash when the agent first exhibits a new behavior (first strafe, first pre-fire, first corner-camp). Requires a behavior taxonomy (strafe/pre-fire/corner-camp) defined in Phase 3–6.
- **Enemy color shift by generation:** dim red → hot orange → white-hot (reuse PredatorPrey Angel palette as terminal state).
- **Cross-mode state sharing:** within MLP family and within SWARM family only. Surfaces in onboarding: the 1→4 path tooltip explicitly says "Mode 1 trains the enemies. Mode 3 lets you fight them." Mode dial signals "these enemies remember Mode 1" with a small neon "TRAINED" badge on modes sharing state.
- **RESET EVOLUTION button** always visible.
- **5-second intro card** (once per session): "NEATENSTEIN / Train your own killer — then survive it. / Watching Mode 1. Press 1–4 to switch. Click for sound." The click resumes audio in ALL modes; in human modes (3/4) it also requests pointer lock. In AI modes (1/2) no pointer lock is needed (spectating).
- **On-screen text cap:** ≤20 words of TRANSIENT text at any time (intro card, banners, tooltips, callouts). Stats overlay is persistent HUD, exempt but iconified. Transient-stacking budget: max 2 transients simultaneously.

**Acceptance:**

- Mode switch posts `set-mode` and locks until `mode-ready`; chip shows tier; stats update each frame.
- Ghost replay legible (or fallback death-position marker functional).
- "First time it did X" callout fires on behavior taxonomy triggers.
- 3/3 NGE-naive viewers restate thesis (act one) after 30s of Mode 1 + intro card.
- 3/3 human playtesters restate "my death trained them" (act two) after one death in Mode 3.

### Phase 8 — Curriculum, Observables, Validation (benchmark-owned) [PLANNED]

**Goal:** Curriculum ramp + arms-race observables + ablations + browser smoke.

[PLANNED] Step 01 — Curriculum, Observables, Validation red tests (deferred until phase becomes active).

- **Curriculum ramp (ARMS RACE):** C0 (1 MLP, slow) → C1 (2 MLPs) → C2 (3 MLPs, cover-seeking) → C3 (2+1 sniper) → C4 (4 MLPs, full co-evolution). Promotion on reliable seed-pack median (5 episodes, deterministic seeds). Carry/reset: main phenotype CARRIES state across C-tiers (brain is the save file); arena state resets on promotion.
- **SWARM curriculum:** ramp coordination density 0→1, NOT swarm size (stays ≤8).
- **Arms-race chart (4 lines):** main fitness, enemy fitness/damage, main aim accuracy, enemy adaptation lag (`gen(enemyPeak) − gen(mainPeak)`).
- **SWARM mechanism observables:** cohort coordination index (synchronized-movement ratio), role-emergence entropy (variance of per-enemy recurrent state), effective-rank of shared weight head. Charted alongside HIVE DENSITY.
- **Reproduction-mode distribution chart:** mode per generation for main agent, correlated with arms-race phase transitions (parity with PredatorPrey lines 460–475).
- **Ablations:** no-snapshot (expect collapse), static-enemy (expect plateau), no-complexity-bonus (expect bloat), coordinate-shuffle (SWARM role emergence).
- **No-trivial-fixed-point acceptance:** adaptation-lag must oscillate (not converge to 0) over N=50 generations. Ablations must show predicted divergence. Non-convergence is a hard acceptance criterion.
- **Anti-frustration:** agent dies ≥1 per 3–5 gens; player win-rate floor ≥30%; challenge-spike waves every N gens; late-game mutation pressure.
- 2 static-enemy test modes (not rendered) for deterministic baseline.
- **Browser E2E smoke:** 0 console errors, ≥30fps, tier correctly reported.

**Acceptance:**

- Promotion gates require reliable performance over seed pack (not one lucky episode).
- Arms-race chart shows oscillation + ratchet; enemy adaptation lag observable stays positive and bounded.
- All four ablations produce predicted divergence.
- Browser smoke green.

---

## Validation gates

Active-step validation follows the Phase 1 Step 01 packet. Required gates:

- `neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`

---

## Build (browser-build)

- **Build script:** `scripts/build-neatenstein.mjs` → `docs/assets/neatenstein.bundle.js` + `docs/assets/neatenstein.worker.esm.js`. ESM, source maps (dev + prod, host + worker).
- **Smoke-test contract:** load bundle in browser-like env, instantiate `start('test-output')`, assert canvas + dial DOM nodes exist, assert `NeatensteinRenderFrame` produces finite typed-array values, assert worker returns valid frame, call `handle.stop()`. Verifiable gate.
- **Size budgets:** host bundle ≤200kB gz; worker bundle ≤150kB gz; combined ≤350kB gz. Core library surface excluded (covered by existing library budget).
- **Worker delivery** (webpack entry/path, dev/prod resolution) owned by build script. **SoA serialization layout** owned by worker-inference-transport boundary (Phase 3). Separate concerns, not merged.
- **COEP/COOP:** `Cross-Origin-Embedder-Policy: require-corp` + `Cross-Origin-Opener-Policy: same-origin` required for `SharedArrayBuffer` / worker transfer. Set by serving host, validated in smoke test.

---

## Reuse Summary

| Reused `src/` primitive                                                | Demo role                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NGE_DNA` envelope (`neat.nge-dna.ts`)                                 | Main agent + swarm DNA                                                                                                                                                                                                                                                                                 |
| `NgeSubstrateBudgetOverride` (`neat.nge-dna.types.ts`)                 | Tier caps (swarm + main)                                                                                                                                                                                                                                                                               |
| `NgeDnaModuleArchetype.weightSharedCohortId`                           | Swarm cohort                                                                                                                                                                                                                                                                                           |
| `NgeDnaModuleArchetype.receivesCoordinates`                            | Per-enemy coordinate injection                                                                                                                                                                                                                                                                         |
| `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE` (`genome.types.ts`)           | All combat motifs (no new motifs)                                                                                                                                                                                                                                                                      |
| `isValidWeightSharedCohortDescriptor` (`genome.utils.ts`)              | Swarm DNA validation                                                                                                                                                                                                                                                                                   |
| Lifecycle state machine (`neat.nge-lifecycle.ts`)                      | Main + swarm lifecycle                                                                                                                                                                                                                                                                                 |
| `assimilateEquilibriumCandidate` (`neat.nge-assimilation`)             | Structural prior write-back (internal)                                                                                                                                                                                                                                                                 |
| `NgeReproductionPolicy` (evolvable mode)                               | Combat-pressure mode selection                                                                                                                                                                                                                                                                         |
| Juvenile focus weights + hysteresis (`neat.nge-juvenile.constants.ts`) | Grow/prune gating                                                                                                                                                                                                                                                                                      |
| `accelerationConfig.parallelVariantCount`                              | Tier preset variant counts                                                                                                                                                                                                                                                                             |
| `RacingQualitySignal` composite fitness pattern                        | `CombatQualitySignal`                                                                                                                                                                                                                                                                                  |
| `OpponentSnapshotPool` / hall-of-fame (racing Tier 6)                  | Asymmetric rolling opponent snapshot                                                                                                                                                                                                                                                                   |
| Worker-authoritative deterministic episode runner (racing)             | Combat episode runner + seed-stamped snapshots                                                                                                                                                                                                                                                         |
| Flappy `WorkerPlaybackFrameSnapshot` SoA + transfer list               | `NeatensteinRenderFrame`                                                                                                                                                                                                                                                                               |
| **Flappy ground grid** (`playback/background/ground-grid/`)            | **Floor renderer** — depth-curve, alpha/blur/thickness helpers, palette. Adapt vertical rays to camera yaw. **Pulse system:** reuse lifetime/color/selection; adapt interval (6000→3000ms), emission (sim-tick), vertical continuity (world-bearing), + event pulses. See research file §3.3.5–§3.3.9. |
| Racing `resolveAccelerationChipPresentation`                           | Acceleration chip (extended additively)                                                                                                                                                                                                                                                                |

| New (core-side)                            | Location                  | Why core-owned                                 |
| ------------------------------------------ | ------------------------- | ---------------------------------------------- |
| Per-enemy substrate coordinate allocator   | `src/neat/nge-dna/`       | Determinism + hashability + unit-cube contract |
| Combat-pressure → reproduction-mode policy | `src/neat/nge-evolution/` | Inspectable policy surface, not a demo hack    |

---

## Risks (consolidated, 12)

1. **WeightSharedCohort behavioral diversity unproven.** Mitigate with diversity-metric test (variance of per-enemy recurrent state) in Phase 4 before swarm demo builds on it.
2. **Human replay buffer determinism vs float drift.** Deterministic serialization (sorted keys, canonical float encoding); CPU-only preset for human modes removes worker ordering nondeterminism. Focused determinism check in Phase 6.
3. **Assimilation vs weight-only opponent may over-fit structure.** Weak/decaying priors (internal to main, not enemy-derived).
4. **Reproduction mode oscillation.** `reproductionModeHysteresis` (3-gen window, majority-vote).
5. **Tier budget rollback under mid-episode growth.** Test rollback against `WeightSharedCohort` path (both `maxNodes` AND `maxEdges`).
6. **Raycaster visual noise at 8 enemies.** Cap at 8, drop to 6 if needed. Legibility > enemy count.
7. **Behavioral ghost replay technically hard.** De-risked with Phase 1–2 spike. Fallback: death-position marker. Second independent signal: "first time it did X" callout.
8. **Human-mode learning rate invisible.** Tune for obvious change first 3 deaths. If delta is sub-perceptual, the feature is dead.
9. **Mode 4 too punishing.** Density decay on kill-streak; asymptotic growth (slows as it approaches 100%).
10. **Demo faking intelligence.** Ablations (no-snapshot, static-enemy, no-complexity-bonus, coordinate-shuffle) + arms-race lag observable prove coevolution drives adaptation. Publish ablation results in demo UI.
11. **2048 variants single-pass on GPU may exceed `DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD` (1024) only for large nets.** Document fallback in UI.
12. **`shadowBlur` expensive at 320 cols × 32 sprites (8 enemies + 24 projectiles max).** Tier-gate glow; profile with `chrome-devtools-mcp`; offer "glow off" fallback.
13. **Fake-perspective-anchored pulses may alias at grazing angles / clutter during heavy combat.** Mitigate: 2px screen-size minimum for grazing pulses (§3.3.5); 8-concurrent-pulse ceiling with oldest-event-first drop (§3.3.6); ambient pulses never dropped mid-travel; depth-test against z-buffer (§3.4.1) prevents bleed-through. Profile pulse projection cost (≤8 sprites/frame, negligible vs 8 enemies + 24 projectiles).

---

## Acceptance Criteria (cross-phase, "fun and legible")

1. **30-second thesis test (act one):** 3/3 NGE-naive viewers restate "enemies learn from deaths" after 30s of Mode 1 + intro card.
2. **Act-two thesis test:** 3/3 human playtesters restate "my death trained them" after one death in Mode 3.
3. **Behavioral recognition:** 3/3 human-mode playtesters spontaneously report "the enemies learned from me" without being told.
4. **No-readme-required:** A viewer can operate all 4 modes and the RESET button without reading anything.
5. **Death is data:** In AI modes, the agent dies at least once every 5 generations. In human modes, the player wins ≥30% of early encounters.
6. **SWARM viscerality:** A viewer can identify that HIVE DENSITY (normalized 0–1) correlates with lockstep behavior change, unprompted.
7. **Stream-friendly:** HUD readable at 1080p stream compression. Generation counter and HIVE DENSITY meter survive bitrate loss.
8. **No dead air:** No 15-second stretch in any mode where nothing happens (no deaths, no counter movement, no behavior change). Generation cadence ≥2/min ensures this.
9. **No-trivial-fixed-point:** Adaptation-lag oscillates over N=50 generations; ablations show predicted divergence.
10. **Browser smoke:** 0 console errors, ≥30fps, tier correctly reported.
11. **Generation-up is viscerally a pair:** 3/3 NGE-naive viewers, asked "what happened just now?" within 2s of a generation-up event, mention either the sound or the floor ripple (ideally both). The pair is recognizable as a single "level-up" moment, not two unrelated effects.

---

## Consensus Record

| Round          | NGE Core        | NGE Benchmark   | Visualizer      | Game Director  |
| -------------- | --------------- | --------------- | --------------- | -------------- |
| 1 (propose)    | proposed        | proposed        | proposed        | proposed       |
| 2 (review)     | 10 observations | 10 observations | 11 observations | 9 observations |
| 3 (approve v2) | **APPROVED**    | **APPROVED**    | **APPROVED**    | **APPROVED**   |

All observations addressed in v2. Non-blocking notes:

- NGE Core: rollback test should cover `maxEdges`, not just `maxNodes`.
- Game Director: behavior taxonomy (strafe/pre-fire/corner-camp) must be defined in Phase 3–6 so the "first time it did X" callout has a trigger source.

---

## Next Steps

This plan is now [WIP] and structurally patched. The five verification blockers (B-001–B-005) are resolved, and the plan is registered in `plans/README.md` and `plans/Roadmap.md`.

1. Dispatch `01-planning` (verification mode) to independently validate slice sizes (≤4h, ideally 2–3), structural completeness, and run `plan-slice-quality` + `step-packet` gates.
2. Only after verification records `green-light: true` may the orchestrator proceed to RED/IMPLEMENT/GREEN.

---

## Latest validation evidence

green-light: true

```yaml
verifier: 01-planning verification agent (fresh context)
timestamp: 2026-07-18T18:30:52-04:00
green-light: true
status: green-light
verification_summary:
  - 'Read Phase 1 phase/step YAML packets and all 13 slice objects (including the newly added 01-host-shell).'
  - 'All 13 slices have required fields (slice_id, title, status, goal, estimate_hours, files_to_change, acceptance_criteria, parallelizable, dependencies, next_slice).'
  - 'Slice dependency graph is a coherent DAG: 01-red-phase1 → 01-scaffold → {01-frame-protocol, 01-raycaster-grid} → 01-neon-walls → {01-floor-reuse, 01-sprites-zbuffer} → 01-pulse-system; 01-frame-protocol → {01-audio, 01-worker-offload}; 01-worker-offload → 01-interpolation-resize and 01-worker-offload + 01-scaffold → 01-host-shell; 01-pulse-system + 01-audio + 01-interpolation-resize + 01-host-shell → 01-green-phase1.'
  - 'Slice 01-frame-protocol is the current [WIP] slice; 01-red-phase1 and 01-scaffold are [DONE]; remaining Phase 1 slices are [PLANNED].'
  - 'files_to_change paths are consistent with the documented module layout under examples/neatenstein/browser-entry/ and docs/examples/neatenstein/.'
  - 'Max slice estimate is 4h (01-worker-offload). It touches only 2 files and has a single acceptance criterion; within hard limit, though cross-cutting (worker + OffscreenCanvas + tier-gated path). Flagged as a watch item, not a blocker.'
  - 'New 01-host-shell slice is 2h, depends on 01-scaffold and 01-worker-offload, and feeds 01-green-phase1. AC-022..AC-025 are attached to 01-host-shell; AC-026 browser-smoke is attached to 01-green-phase1.'
  - 'Phases 2–8 remain [PLANNED] with placeholder Step 01 lines, which is acceptable for a plan at Phase 1 boundary.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'All active WIP phase/step packets conform to the new format. Plan-readiness warning for this plan resolved by this green-light marker.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@14636","plans/mcp-active-binding.plans.md:yaml@16089","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@3609","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@11673","plans/Racing_Perception_Redesign.plans.md:yaml@1944","plans/Racing_Perception_Redesign.plans.md:yaml@3122","plans/Racing_Perception_Redesign.plans.md:yaml@7758"],"violations":[],"planReadinessWarnings":[{"blockId":"plans/Racing_Perception_Redesign.plans.md:yaml@3122","goal":"implementing","message":"Mandatory plan verification gate has not passed: no green-light marker in ## Latest validation evidence. Dispatch a fresh 01-planning verification agent before execution-phase work."}],"plansScanned":3},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
blockers: []
watch_items:
  - id: W-001
    text: 'Slice 01-worker-offload is at the 4-hour hard limit and bundles worker display skeleton + OffscreenCanvas path + tier-gated render path + host renderer bridge. If implementation proves too large in practice, split into worker-frame production, OffscreenCanvas path, and renderer-bridge/tier-switch slices in the next planning patch.'
```

---

## Plan Update

```yaml
PlanUpdate:
  slice_id: 01-frame-protocol
  changed_files:
    - examples/neatenstein/browser-entry/renderer/frame.ts
  preflight:
    - 'npx tsc --noEmit --skipLibCheck --noResolve --target ES2023 --module ESNext --moduleResolution Bundler --strict --allowImportingTsExtensions examples/neatenstein/browser-entry/renderer/frame.ts examples/neatenstein/browser-entry/constants.ts'
    - 'npx eslint examples/neatenstein/browser-entry/renderer/frame.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/frame.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/frame'
  rollback:
    - 'git rm -f examples/neatenstein/browser-entry/renderer/frame.ts'
  next: 'Run 05-green-testing on frame.test.ts; then dispatch 04-implementing for slice 01-raycaster-grid'
```

```yaml
PlanUpdate:
  slice_id: 01-raycaster-grid
  changed_files:
    - examples/neatenstein/browser-entry/renderer/map.ts
    - examples/neatenstein/browser-entry/renderer/raycast.ts
  preflight:
    - 'npx tsc --noEmit --skipLibCheck --noResolve --target ES2023 --module ESNext --moduleResolution Bundler --strict examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts examples/neatenstein/browser-entry/constants.ts'
    - 'npx eslint examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/raycast'
  rollback:
    - 'git rm -f examples/neatenstein/browser-entry/renderer/map.ts examples/neatenstein/browser-entry/renderer/raycast.ts'
  next: 'Run 05-green-testing on raycast.test.ts; then dispatch 04-implementing for slice 01-neon-walls'
```

```yaml
PlanUpdate:
  slice_id: 01-neon-walls
  changed_files:
    - examples/neatenstein/browser-entry/renderer/walls.ts
    - examples/neatenstein/browser-entry/renderer/walls.test.ts
  preflight:
    - 'npx tsc --skipLibCheck --noResolve --target ES2023 --module ESNext --moduleResolution Bundler --strict --noEmit examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts'
    - 'npx eslint examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/walls.test.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/walls.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/walls'
  rollback:
    - 'git rm -f examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/walls.test.ts'
  next: 'Run 05-green-testing on walls.test.ts; then dispatch 04-implementing for slice 01-floor-reuse'
```

## Latest validation evidence

- Slice 01-floor-reuse preflight (2026-07-18): prettier/eslint passed; `npx tsc --noEmit -p tsconfig.test.json` reports zero type errors for `examples/neatenstein/browser-entry/renderer/floor.ts` (unrelated node_modules devtools-protocol syntax error outside slice scope). Tests not run — awaiting 05-green-testing.
- Pre-green specialist review (2026-07-18): implementation-pattern-scout — REQUEST_CHANGES; visualizer-scout — REQUEST_CHANGES.
- Fix packet applied to `examples/neatenstein/browser-entry/renderer/walls.ts` and `examples/neatenstein/browser-entry/renderer/walls.test.ts`.
- Preflight (2026-07-18):
  - `npx tsc --skipLibCheck --noResolve --target ES2023 --module ESNext --moduleResolution Bundler --strict --noEmit examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts` → exit 0.
  - `npx prettier --check examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/walls.test.ts` → exit 0.
  - `npx eslint examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/walls.test.ts` → exit 0, 1 pre-existing warning on `walls.test.ts:3:44` (`@typescript-eslint/no-explicit-any`).
- `framebuffer.ts` was not modified; `clampInt` is now imported from it.
- GREEN validation (2026-07-18): `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/walls` → PASS (1 suite, 4/4 tests).
- GREEN preflight: `npx prettier --check examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts` → PASS; `npx eslint examples/neatenstein/browser-entry/renderer/walls.ts examples/neatenstein/browser-entry/renderer/framebuffer.ts` → PASS (0 errors, 0 warnings).
- Red contract preserved: floor.test.ts, pulse.test.ts, interpolate.test.ts, audio.test.ts remain untouched and fail with module-not-found (right reason) — 4 failed suites, 21 failed tests.
- Plan gates: plan-sync → pass; step-packet → pass; plan-slice-quality → pass.
- Specialist-review evidence confirmed for target plan via direct script (`--plan=plans/Neon_Shooter_NGE_Demo.plans.md`). Note: `neataptic-gate-mcp:run_gate_check specialist-review` defaults to `plans/completed/Agentic_Workflow_Architecture.plans.md` (parseArgs default) and reports pass:false for that unrelated completed plan.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Active slice: 01-floor-reuse — Floor renderer (Flappy ground grid reuse, camera-adapted) ([WIP] · implementation complete, awaiting GREEN validation).
Files changed: examples/neatenstein/browser-entry/renderer/floor.ts.
Dependency: 01-neon-walls is [DONE] with GREEN validation passed (4/4 tests).
Implementation: floor.ts re-exports the four Flappy ground-grid curve helpers under Neatenstein names and renders a yaw-adapted synthwave floor grid with a default 320×240 canvas.
Preflight: prettier/eslint passed; `npx tsc --noEmit -p tsconfig.test.json` reports zero type errors for floor.ts.
Red contract preserved: pulse.test.ts, interpolate.test.ts, audio.test.ts remain untouched and still fail with module-not-found.
Next narrow task: Dispatch 05-green-testing with `npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/floor`.
If tests fail, return observations to 04-implementing for a slice-fix; if tests pass, mark slice [DONE] and advance to 01-sprites-zbuffer.
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 01-floor-reuse
  changed_files:
    - examples/neatenstein/browser-entry/renderer/floor.ts
  preflight:
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/floor.ts (passed after --write)'
    - 'npx eslint examples/neatenstein/browser-entry/renderer/floor.ts (0 errors, 0 warnings)'
    - 'npx tsc --noEmit -p tsconfig.test.json (zero floor.ts errors; unrelated node_modules devtools-protocol syntax error)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/floor'
  rollback:
    - 'git rm examples/neatenstein/browser-entry/renderer/floor.ts'
  next: 'Run 05-green-testing and attach coverage-guard evidence; mark slice [DONE] if tests pass.'
```
