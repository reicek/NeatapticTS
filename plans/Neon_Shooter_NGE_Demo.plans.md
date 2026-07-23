# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [WIP] — Phase 2 Step 02 [WIP]: User confirmation gate · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/Neon_Shooter_NGE_Demo.research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

---

## Current state

**Phase 1 — World & Renderer is [DONE].** All 13 Step 01 slices are green validated and compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 1. User confirmed (2026-07-22): 3D rendering works, mouse look works at `http://localhost:8080/docs/examples/neatenstein/index.html`.

**Phase 2 — Game Logic & FPS State is [WIP].** All 10 implementation slices completed and green validated (43 suites, 354 tests pass). Step 01 [DONE]. Step 02 (user confirmation gate) is [WIP] — the active step. Details compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2. Browser integration verified through iterative user-driven testing: WASD, mouse-look, fire, floor grid, pulses, impact spots, wall colors, collision slide.

**Active frontier:** Step 02 — User confirmation gate [WIP]. Awaiting user manual browser review at `http://localhost:8080/docs/examples/neatenstein/index.html`. User may request visual fixes or adjustments before approving Phase 3.

**USER CONFIRMATION GATE:** Before proceeding to Phase 3, the user must manually confirm in the browser:

1. WASD moves the player in the world.
2. Left-click fires the neon beam (visible tracer).
3. Space triggers dash.
4. Enemies are visible (at least placeholder sprites).
5. The world feels responsive and controllable.

Completed Phase 2 slice logs are in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2.

## Latest validation evidence

status: green-light
green-light: true

````yaml
verifier: 01-planning verification agent (fresh context)
timestamp: 2026-07-21T15:58:10-04:00
green-light: true
status: green-light
verification_summary:
  - 'Phase 1 is [DONE] and satisfies the dependency precondition for Phase 2.'
  - 'Phase 2 — Game Logic & FPS State is [DONE]. All 10 slices completed and green validated (43 suites, 354 tests pass).'
  - 'Phase 2 Step 01 step-level YAML block is complete with all slices marked [DONE].'
  - 'Browser integration verified through iterative user-driven testing during Phase 2.'
  - 'Detailed Phase 2 evidence compressed to plans/Neon_Shooter_NGE_Demo.logs.md §Phase 2.'
gate_verdicts:
  - gate: plan-slice-quality
    pass: true
    evidence: 'All WIP plan slices are within the 4-hour estimate limit. Phase 2 slices checked: 02-red-phase2 (2h), 02-game-scaffold (2h), 02-hero-state (2h), 02-enemy-waves (3h), 02-controls (3h), 02-projectiles (3h), 02-collision (3h), 02-episode-loop (3h), 02-worker-game-sync (4h), 02-green-phase2 (2h).'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
    raw_json: '{"pass":true,"evidence":{"plansChecked":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md","plans/Racing_Perception_Redesign.plans.md","plans/__gate-debug-1784660517090.plans.md","plans/__gate-debug-1784660579515.plans.md"],"violations":[],"limit":4},"fixHint":"All WIP plan slices are within the 4-hour estimate limit.","owner":"plan-slice-quality.gate.mjs"}'
  - gate: step-packet
    pass: true
    evidence: 'Active WIP phase/step packets conform to the new format and the plan-readiness green-light marker is detected.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
    raw_json: '{"pass":true,"evidence":{"blocksChecked":["plans/mcp-active-binding.plans.md:yaml@19927","plans/mcp-active-binding.plans.md:yaml@21380","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@16123","plans/Neon_Shooter_NGE_Demo.plans.md:yaml@19100"],"violations":[],"planReadinessWarnings":[],"preExecuteHooks":[],"plansScanned":5},"fixHint":"All active WIP phase/step packets conform to the new format.","owner":"step-packet.gate.mjs"}'
  - gate: plan-sync
    pass: true
    evidence: 'All WIP plans are correctly registered in README and Roadmap.'
    command: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
    raw_json: '{"pass":true,"evidence":{"wipPlans":["plans/mcp-active-binding.plans.md","plans/Neon_Shooter_NGE_Demo.plans.md"],"missingFromReadme":[],"missingFromRoadmap":[],"plansChecked":9},"fixHint":"All WIP plans are correctly registered in README and Roadmap.","owner":"validate-plan-sync.mjs"}'
  - gate: validate-plan-sync
    pass: true
    evidence: 'PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)'
    command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md'
    raw_json: '{"name":"plan sync","ok":true,"issues":[],"counts":{"errors":0,"warnings":0},"summaryText":"PASS plan sync: 0 errors, 0 warnings (plan: plans/Neon_Shooter_NGE_Demo.plans.md)","plan":{"path":"plans/Neon_Shooter_NGE_Demo.plans.md","status":"WIP"},"downstreamTrackers":["plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md","plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md","plans/Racing_Perception_Redesign.plans.md","plans/mcp-active-binding.plans.md"]}'

## Implementation phases

### Phase 1 — World & Renderer (visualizer-owned) [DONE]

**Goal:** Raycasting neon renderer + frame protocol + audio.

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
````

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

#### Step 01: World & Renderer scaffold and raycaster [DONE]

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

[DONE] All 13 Phase 1 slices completed and green validated. Detailed slice logs moved to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 1.

### Phase 2 — Game Logic & FPS State (visualizer + benchmark-owned) [WIP]

**Goal:** FPS game state, controls, deterministic episode.

[DONE] All 10 Phase 2 slices completed and green validated (43 suites, 354 tests). Browser integration verified through iterative user-driven testing. Detailed slice logs moved to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2.

**Slice summary:** 02-red-phase2 → 02-game-scaffold → 02-hero-state → 02-enemy-waves → 02-controls → 02-projectiles → 02-collision → 02-episode-loop → 02-worker-game-sync → 02-green-phase2. All [DONE].

```yaml
phase: 2
title: 'Game Logic & FPS State'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_phase: 'Step 02 — User confirmation gate'
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
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/host/game'
  - id: AC-214
    text: 'Visible-browser smoke test passes for WASD, mouse look, pointer lock, Space dash, left-click fire, and iOS Safari touch look'
    validation: 'Visible-browser smoke at http://localhost:8080/docs/examples/neatenstein/index.html'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Game Logic & FPS State red tests and implementation slices [DONE]'
  - 'Step 02 — User confirmation gate [WIP]'
```

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

#### Step 01: Game Logic & FPS State red tests and implementation slices [DONE]

[DONE] All 10 slices completed and green validated (43 suites, 354 tests). Browser integration verified through iterative user-driven testing. Full step packet with AC-201 through AC-217, traceability, and slice details archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2.

#### Step 02: User confirmation gate [WIP]

**Active step — awaiting user manual browser review.** All Phase 2 implementation is [DONE]. The user must manually verify the demo at `http://localhost:8080/docs/examples/neatenstein/index.html` before Phase 3 begins. User may request visual fixes or adjustments.

```yaml
phase: 2
step: 2
title: 'User confirmation gate'
status: '[WIP]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 3 Step 01 — Asymmetric Co-evolution Harness red tests'
skills:
  - 'plan-alignment'
  - 'phase-handoff-workflow'
validation:
  - 'User manual browser confirmation at http://localhost:8080/docs/examples/neatenstein/index.html'
acceptance_criteria:
  - id: AC-217
    text: 'User confirms WASD moves the player, left-click fires a visible neon beam, Space triggers dash, enemies are visible, and the world feels responsive'
    validation: 'User browser confirmation at http://localhost:8080/docs/examples/neatenstein/index.html'
constitution_check:
  - 'principle-4-small-slices'
```

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

Phase 2 validation is governed by the Step 01 slice-level acceptance criteria (AC-201..AC-210, AC-215..AC-217) plus the explicit browser-integration gates. Required automated gates for any active slice:

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein/host/game`
- `npm run lint`
- `neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md`

**Mandatory human gate before Phase 3:** USER CONFIRMATION GATE at `http://localhost:8080/docs/examples/neatenstein/index.html` (WASD, left-click fire, Space dash, enemy visibility, responsiveness). No agent may advance 02-collision to `[DONE]` or start Phase 3 without this confirmation.

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

Phase 2 implementation is complete. All 10 slices are [DONE] and compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2. The active step is Step 02 — User confirmation gate [WIP].

1. User serves the demo at `http://localhost:8080/docs/examples/neatenstein/index.html` (bundle already rebuilt v6).
2. User manually confirms the USER CONFIRMATION GATE checks (WASD, left-click fire, Space dash, enemy visibility, responsiveness).
3. User may request visual fixes or adjustments before approving.
4. Only after user confirmation may the orchestrator mark Step 02 [DONE], Phase 2 [DONE], and begin Phase 3.

## Prior validation evidence

- Prior verification at 2026-07-21T15:39:43-04:00 found blockers B-001..B-004 (missing step-level YAML, slices, traceable AC-###, and files_to_change). All four blockers are resolved by the current Phase 2 Step 01 packet; see the latest `## Latest validation evidence` section above.

---

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Context: Neatenstein NGE Demo — Phase 2 [WIP], Step 01 [DONE] (all 10 slices green validated, 43 suites, 354 tests). Active step is Step 02 — User confirmation gate [WIP]. EXPLICIT STOP for user manual browser review before Phase 3.
Current boundary: All Phase 2 implementation is [DONE]. Step 02 (user confirmation gate) is the active [WIP] step. User manual browser review is the only outstanding gate.
What is already covered: Phase 1 [DONE]. Phase 2 Step 01 [DONE] — all 10 slices completed. Detailed logs in plans/Neon_Shooter_NGE_Demo.logs.md §Phase 2. Bundle rebuilt (v6). 43 suites, 354 tests pass. Lint clean. tsc clean for neatenstein files.
Next narrow task: Wait for the user to manually review the demo at http://localhost:8080/docs/examples/neatenstein/index.html and provide visual feedback or confirmation. Do not start Phase 3 without user confirmation.
Required validations (USER CONFIRMATION GATE):
  - User confirms WASD moves the player
  - User confirms left-click fires a visible neon beam/tracer
  - User confirms Space triggers dash
  - User confirms enemies are visible (placeholder sprites)
  - User confirms the world feels responsive
  - User may request visual fixes or adjustments before approving
Known worktree cautions: Phase 2 code lives under examples/neatenstein/browser-entry/host/game/; do not modify Phase 3+ without a new plan step. Pre-existing repo-wide lint warnings and code-coverage drift are non-blocking.
```
