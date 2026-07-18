# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [PROPOSED] · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/references/neat-doom-engine-research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

---

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

## Phases

### Phase 1 — World & Renderer (visualizer-owned)

**Goal:** Raycasting neon renderer + frame protocol + audio.

- Raycasting renderer (~800 lines): map grid, DDA ray cast, neon wall rendering (pure neon-line with optional line-pattern texture modulation, NOT sampled texels), enemy wireframe sprites, projectiles.
- **Floor: reuse Flappy Bird's synthwave ground grid** (camera-adapted). Reuse `FLAPPY_GROUND_GRID_*` constants, depth-curve/alpha/blur/thickness helpers, pulse system, and `FLAPPY_NEON_PALETTE` ground colors. Adapt vertical rays to camera yaw rotation. See research file §3.3. This gives visual coherence with the Flappy demo — both share the same neon floor aesthetic.
- Tier-aware column count: GPU 320 cols, Worker 240, CPU 160. Glow passes skip on CPU. CPU fallback: lines only, no texture modulation, no glow.
- Worker offload: all NGE inference + enemy AI + projectile physics on workers; renderer reads packed `NeatensteinRenderFrame` (SoA typed arrays, transfer list, zero-copy, requestId-gated). Worker tier may use `OffscreenCanvas` via `transferControlToOffscreen()` for off-main-thread rendering (see research file §3.2.1). **Two render architectures by tier:** (a) CPU/GPU — display worker produces `NeatensteinRenderFrame`, main thread renders; (b) Worker — display worker renders directly via OffscreenCanvas, frame transfer bypassed. On Worker tier, display worker responsibilities = sim tick + NGE inference + OffscreenCanvas render.
- **Render path (tier-gated, see research file §3.2.1):** CPU tier → `ImageData` framebuffer + single `putImageData` (no per-column `fillRect`); Worker tier → `OffscreenCanvas`; GPU tier → stroke + `shadowBlur` (premium). All tiers: `getContext("2d", { alpha: false })`, integer-floored coordinates. Feature-detect `transferControlToOffscreen` and `ctx.filter`; fall back to CPU ImageData path if unavailable.
- **`transferControlToOffscreen()` is irreversible.** On tier downgrade from Worker → CPU/GPU, the host must create a fresh `<canvas>` element (old canvas is permanently worker-owned). `onBackendChange` handler accounts for canvas recreation + re-attach ResizeObserver + re-bind pointer lock.
- **Sprite occlusion:** per-column `Float32Array` z-buffer (not a boolean set) — handles partial occlusion. See research file §3.4.1.
- **Render interpolation:** `lerp(statePrev, stateCurr, alpha)` on each RAF to eliminate 30→60 Hz judder. See research file §4.1.2.
- **Raycaster is a shared build entry** — included in both host bundle (CPU/GPU tiers render on main thread) and worker bundle (Worker tier renders via OffscreenCanvas). Build script configures dual webpack entries. Worker instantiated as module worker: `new Worker(url, { type: 'module' })`.
- 60fps target on GPU tier, 30fps floor on CPU.
- **Audio (Phase 1 deliverable, same weight as renderer):** 6 sounds — fire, enemy hit, player damage, dash, kill, **generation-up** (rising arpeggio, the audio signal of learning) — via WebAudio procedural synthesis (oscillators, zero assets). Positional audio via `StereoPannerNode` + distance attenuation. `AudioContext.resume()` on first click (regardless of mode — AI modes need audio too). **AudioContext is main-thread only** — audio trigger events originate in the display worker and are `postMessage`d to the main thread for synthesis. See research file §9.
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

### Phase 2 — Game Logic & FPS State (visualizer + benchmark-owned)

**Goal:** FPS game state, controls, deterministic episode.

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

### Phase 3 — Asymmetric Co-evolution Harness (benchmark-owned, core-reviewed)

**Goal:** Minimal single-main + enemy-population co-evolution harness.

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

### Phase 4 — NGE Main Agent + Enemy MLPs (core + benchmark-owned)

**Goal:** Full NGE main agent lifecycle + weight-only MLP co-evolution.

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

### Phase 5 — SWARM Mode (core + benchmark-owned)

**Goal:** WeightSharedCohort swarm + HIVE DENSITY legibility.

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

### Phase 6 — Human Modes + Replay Buffer (benchmark + game-director-owned)

**Goal:** Replay-based per-death evolution + death feedback loop.

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

### Phase 7 — Mode Dial, Stats, UI Polish (visualizer + game-director-owned)

**Goal:** Legible mode dial + stats + thesis delivery.

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

### Phase 8 — Curriculum, Observables, Validation (benchmark-owned)

**Goal:** Curriculum ramp + arms-race observables + ablations + browser smoke.

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

## Build (browser-build)

- **Build script:** `scripts/build-neatenstein.mjs` → `docs/assets/neatenstein.bundle.js` + `docs/assets/neatenstein.worker.esm.js`. ESM, source maps (dev + prod, host + worker).
- **Smoke-test contract:** load bundle in browser-like env, instantiate `start('test-output')`, assert canvas + dial DOM nodes exist, assert `NeatensteinRenderFrame` produces finite typed-array values, assert worker returns valid frame, call `handle.stop()`. Verifiable gate.
- **Size budgets:** host bundle ≤200kB gz; worker bundle ≤150kB gz; combined ≤350kB gz. Core library surface excluded (covered by existing library budget).
- **Worker delivery** (webpack entry/path, dev/prod resolution) owned by build script. **SoA serialization layout** owned by worker-inference-transport boundary (Phase 3). Separate concerns, not merged.
- **COEP/COOP:** `Cross-Origin-Embedder-Policy: require-corp` + `Cross-Origin-Opener-Policy: same-origin` required for `SharedArrayBuffer` / worker transfer. Set by serving host, validated in smoke test.

---

## Reuse Summary

| Reused `src/` primitive | Demo role |
|---|---|
| `NGE_DNA` envelope (`neat.nge-dna.ts`) | Main agent + swarm DNA |
| `NgeSubstrateBudgetOverride` (`neat.nge-dna.types.ts`) | Tier caps (swarm + main) |
| `NgeDnaModuleArchetype.weightSharedCohortId` | Swarm cohort |
| `NgeDnaModuleArchetype.receivesCoordinates` | Per-enemy coordinate injection |
| `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE` (`genome.types.ts`) | All combat motifs (no new motifs) |
| `isValidWeightSharedCohortDescriptor` (`genome.utils.ts`) | Swarm DNA validation |
| Lifecycle state machine (`neat.nge-lifecycle.ts`) | Main + swarm lifecycle |
| `assimilateEquilibriumCandidate` (`neat.nge-assimilation`) | Structural prior write-back (internal) |
| `NgeReproductionPolicy` (evolvable mode) | Combat-pressure mode selection |
| Juvenile focus weights + hysteresis (`neat.nge-juvenile.constants.ts`) | Grow/prune gating |
| `accelerationConfig.parallelVariantCount` | Tier preset variant counts |
| `RacingQualitySignal` composite fitness pattern | `CombatQualitySignal` |
| `OpponentSnapshotPool` / hall-of-fame (racing Tier 6) | Asymmetric rolling opponent snapshot |
| Worker-authoritative deterministic episode runner (racing) | Combat episode runner + seed-stamped snapshots |
| Flappy `WorkerPlaybackFrameSnapshot` SoA + transfer list | `NeatensteinRenderFrame` |
| **Flappy ground grid** (`playback/background/ground-grid/`) | **Floor renderer** — depth-curve, alpha/blur/thickness helpers, pulse system, palette. Adapt vertical rays to camera yaw. See research file §3.3. |
| Racing `resolveAccelerationChipPresentation` | Acceleration chip (extended additively) |

| New (core-side) | Location | Why core-owned |
|---|---|---|
| Per-enemy substrate coordinate allocator | `src/neat/nge-dna/` | Determinism + hashability + unit-cube contract |
| Combat-pressure → reproduction-mode policy | `src/neat/nge-evolution/` | Inspectable policy surface, not a demo hack |

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

---

## Consensus Record

| Round | NGE Core | NGE Benchmark | Visualizer | Game Director |
|---|---|---|---|---|
| 1 (propose) | proposed | proposed | proposed | proposed |
| 2 (review) | 10 observations | 10 observations | 11 observations | 9 observations |
| 3 (approve v2) | **APPROVED** | **APPROVED** | **APPROVED** | **APPROVED** |

All observations addressed in v2. Non-blocking notes:
- NGE Core: rollback test should cover `maxEdges`, not just `maxNodes`.
- Game Director: behavior taxonomy (strafe/pre-fire/corner-camp) must be defined in Phase 3–6 so the "first time it did X" callout has a trigger source.

---

## Next Steps

This plan is [PROPOSED] and consensus-approved. Before implementation:
1. Dispatch `01-planning` (verification mode) to independently validate slice sizes (≤4h, ideally 2–3), structural completeness, and run `plan-slice-quality` + `step-packet` gates.
2. Only after verification records `green-light: true` may the orchestrator proceed to RED/IMPLEMENT/GREEN.