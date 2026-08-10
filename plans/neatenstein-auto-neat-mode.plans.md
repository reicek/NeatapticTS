# Neatenstein Auto / NEAT Mode Wiring

**Status:** [WIP]
**Plan ID:** NEATENSTEIN_AUTO_NEAT_MODE
**Created:** 2026-08-09
**Source of truth:** `plans/neatenstein-auto-neat-mode.plans.md`
**Research artifact:** `plans/neatenstein-auto-neat-mode.research.md`

## Mandates

- `model: glm-5.2:cloud` for all dispatches under this plan. Every dispatch under this plan overrides the frontmatter `model` with `glm-5.2:cloud`. Do NOT use kimi-k2.7-code:cloud or any other model.
- Pragmatic mode: broad slices (one per gap area), bypass legacy ceremony (skip plan-verification green-light cycle, skip per-AC gate calls, skip fix-packet YAML ceremony). Ship working software.
- Do not archive or supersede existing Neatenstein plans; this workstream is additive.
- Remove legacy noise: delete the RNG-based `runEpisode` stub in `main-runner.ts` when replaced by real gameplay (no dual-path code, no backward-compatibility wrappers).
- Human-played mode must remain fully functional — the `humanMode` selector is the switch point.

## Scope

Wire the Neatenstein evolution harness into the live game loop so that `humanMode: 'auto'` activates NEAT-driven gameplay:

1. **Worker reads `humanMode`** and branches to a neural-network controller instead of forwarding human input (Gap 1).
2. **`advanceWave()` is called from the live game** on wave-clear, evolving the enemy MLP population (Gap 2).
3. **Enemy weights are populated** from the MLP population via `createMlpEnemyPopulation()` / `activateMlp()` instead of always being `undefined` (Gap 3).
4. **The harness episode runs real gameplay** (or a headless fast-forward) instead of RNG-based fake metrics (Gap 4).
5. **A NEAT network controller produces `GameTickInputSnapshot`** from game-state sensors via `Network.activate()` (Gap 5).
6. **`GameState.generation` is incremented** on generation advance and synced from the harness counter (Gap 6).

## Determinism contract

**Rung:** Level 2 — Ordered deterministic (same seed + same genome + same fixed timestep → same trajectory).

1. **Seed propagation**: `gameState.seed` (set at game init, numeric) is the root seed. Evolution harness derives its seed as a deterministic numeric hash: `hashSeed(seed, generation) = ((seed * 100003 + generation) * 100003) >>> 0` — produces a 32-bit unsigned integer. Per-variant: `hashSeed(seed, generation, variantId) = ((seed * 100003 + generation) * 100003 + variantId) >>> 0`. This is NOT string-based mixing — Neat's `normalizeSeed` in `src/neat/rng/core/rng.utils.ts` only accepts finite numbers; non-numeric strings fall back to `Date.now()` which breaks determinism. `createSeedPack` must be updated to accept an optional `seed?: number` parameter: `createSeedPack({ generation, variantCount, seed })` — when `seed` is omitted, existing callers get the current behavior (no change needed). When provided, seeds are derived via the numeric hash. `harness/seed-pack.ts`, `harness/main-runner.ts` (caller at line 133), and `harness/barrier.ts` (caller at line 174) are all updated to pass the root seed.
2. **Fixed timestep in worker**: `display.worker.ts:1222-1223` currently uses host rAF `deltaMs` as simulation timestep. **Must clamp to `NEATENSTEIN_FIXED_TIMESTEP_MS = 16`** for deterministic simulation. The rAF delta only gates *when* a tick runs, not *how much* time advances. The Level 2 determinism claim applies to headless episode evaluation; the live worker path uses the rAF cadence for dispatch but clamps the timestep value.
3. **Episode reproducibility**: `runEpisode` in the harness uses `NEATENSTEIN_FIXED_TIMESTEP_MS` for all ticks. Same seed + same genome + same enemy snapshot → same fitness score.
4. **Async serialization**: Only one generation evaluation is in flight at a time. The worker's hoisted Neat evaluation (async `await neatPop.evaluate()` + `await neatPop.evolve()`) is guarded by a `pendingGeneration` number that stores the launched generation. Before starting a new evaluation, check `if (pendingGeneration !== null) return;` — skip if already evaluating. On completion, `runArmsRaceGeneration` returns `generation + 1` (the next generation number). The guard stores `pendingGeneration = launchedGeneration`; on completion, verify `result.generation === pendingGeneration + 1` before applying (prevents out-of-order application). Then clear `pendingGeneration = null`. If wave-clears arrive faster than the ~2-3s evaluation, the new request is dropped (no queueing). The synchronous `runArmsRaceGeneration` function is called only after the async evaluation completes and the champion Network is extracted. When `championNetwork` is provided, `runMainGeneration` is skipped (no duplicate evaluation).
5. **Seed propagation in all callers**: Both `main-runner.ts` (line 133) and `barrier.ts` (line 174) must pass the root seed to `createSeedPack`: `createSeedPack({ generation, variantCount, seed: rootSeed })`. In `main-runner.ts`, the root seed comes from the caller (the worker passes `gameState.seed`); the call site at line 133 changes from `createSeedPack({ generation })` to `createSeedPack({ generation, seed: rootSeed })`. In `barrier.ts`, `options.seed` is already available (declared in `GenBarrierOptions`); the one-line change is `createSeedPack({ generation: options.generation, seed: options.seed })`. The optional `seed?: number` API makes both changes backward-compatible. The per-variant seed derivation inside `createSeedPack` uses the numeric hash: `seeds[i] = hashSeed(seed, generation, i)` for each variant index `i`.
6. **humanMode type mapping**: Worker receives `humanMode: 'auto' | 'human'` (string) from `simState`. Must convert to boolean for `runArmsRaceGeneration({ humanMode: humanMode === 'auto' })` before the call.
7. **Versioned parameters**: `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS` and `NEATENSTEIN_MAIN_VARIANT_COUNT` are contract constants. Note: `NEATENSTEIN_MAIN_VARIANT_COUNT` is currently hard-coded as `8` in BOTH `main-runner.ts:46` AND `barrier.ts:87` — both must be removed and imported from `constants.ts` (value 4). This centralization happens in P5S2b (which already touches `constants.ts` and can add the export in the same slice, plus removes the local in barrier.ts in the same slice). P5S1 does NOT change the variant count — it only adds seed propagation to barrier.ts. `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS` (= 5000) derives a NEW fitness-specific tick count: `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = Math.floor(NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312. The existing `NEATENSTEIN_MAX_EPISODE_TICKS = 240` is NOT changed — it is used by `enemy-runner.ts` for the enemy barrier evaluation path. Introducing a separate `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` avoids the determinism contradiction of changing a shared constant. Note: `NEATENSTEIN_ENEMY_EVALUATION_DURATION_MS = 10000` (existing) is used by the enemy barrier evaluation path and is NOT changed by this plan. `NEATENSTEIN_FIXED_TIMESTEP_MS = 16` already exists in `host/game/constants.ts` (the authoritative source); `harness/constants.ts` must re-export it (not redefine) to avoid dual-definition drift. P2S1 imports `NEATENSTEIN_FIXED_TIMESTEP_MS` directly from `host/game/constants.ts` (the authoritative source), not from `harness/constants.ts` (which doesn't re-export it until P5S2b).

## Non-goals

- No changes to the core NeatapticTS library (`src/`) — only consume `Network.activate()`, `new Neat()`, and NEAT APIs from existing exports. The `NgeMainAgentEmbryo` → `NeatGenome` → `Network` converter is a documented follow-up, NOT part of this plan.
- No new enemy MLP topology changes — the existing 6→6→4→4 architecture is reused as-is.
- No changes to maze generation, wall collision, or rendering pipeline.
- No sound asset changes.
- No WebGPU tier changes.

## Open assumptions / decisions

1. **Enemy weight injection strategy**: Champion snapshot weights shared by all enemies (simplest), vs. per-variant distribution (`enemy.index % population.size`). Decision: start with champion snapshot (all enemies share the current champion's weights); per-variant can be a follow-up.
2. **Player network topology**: Use `buildMainAgentEmbryo()` from the NGE pipeline to construct the player's network, or build a simpler feed-forward network via `new Neat(inputCount, outputCount, fitnessFn)`. Decision: use the NGE embryo for the main-agent champion (consistent with the harness), but the initial auto-mode controller can use a simple NEAT network while the embryo pipeline is wired.
3. **Cadence compromise**: Real 20s episodes × 8 variants = 160s exceeds the 30s/generation budget. Decision: reduce episode duration for fitness evaluation to 5s (headless fast-forward), reduce variant count to 4, or run evaluation async in a separate worker. The initial implementation uses headless fast-forward at reduced duration.
4. **Sensor design**: The player controller needs game-state observations. Decision: start with a minimal sensor set (~12 inputs: player health, ammo, angle, position, nearest enemy bearing, nearest enemy distance, nearest enemy health, 4 wall raycasts) and expand if evolution stalls.
5. **Pitch control**: `GameTickInputSnapshot` only carries `lookDelta` (yaw), not pitch. Decision: keep yaw-only auto-aim for the initial implementation; pitch extension is a follow-up.

## Research findings summary

Five research specialists investigated the six gaps. Key findings:

### Gap 1 — Worker integration (humanMode dead field)
- `humanMode: 'auto'` is posted in `simState` to the worker at `display.worker.ts:1204-1205` but NEVER read.
- Worker builds `tickInput` from human input queue only (line 1212-1217).
- Branch insertion point: `display.worker.ts:1204-1217`, between `latestState` assignment and `tickInput` construction.
- `GameTickInputSnapshot` type (`host/game/tick.ts:73-85`): `{move: Vector2, lookDelta: number, fire: boolean, dash: boolean}` — network-output-friendly.
- Medium complexity; need to add NEAT network loading + observation extractor + output mapper to worker.

### Gap 2 — Arms-race wiring (harness never called)
- `advanceWave()` (`host/waves.ts:96`) — evolves MLP population, advances generation, clears arena, respawns. Returns `{state, snapshot, spawnedCount}`.
- `runArmsRaceGeneration()` (`harness/arms-race.ts:95`) — full arms race generation. Never called from live game.
- `spawnWaveTick()` (`host/game/waves.ts:161`) — live spawner, no evolution logic.
- Trigger point: wave-clear detection in worker after `gameTick`.
- `advanceWave()` already returns evolved `snapshot.weights` — just needs injection into enemy controllers.
- Enemy-side wiring is low-medium; main-agent side is higher (needs real episodes).

### Gap 3 — Enemy activation (weights always undefined)
- `createEnemyControllerState()` (`scripts/enemy-controller.ts:310`) initializes `weights: undefined`.
- `updateControlledEnemy()` (line 397): `weights = isRespawn ? undefined : previousOrDefault.weights` — always undefined.
- MLP re-ranking path (line 645-711) is fully implemented but dead code.
- MLP topology: 6→6→4→4, 90 params, tanh activation. Input: 6-element BFS vision vector. Output: move, strafe, turn, fire.
- **Low-to-moderate complexity** — the plumbing exists; just needs weight injection.
- Need to: instantiate population in worker, pass weights to `updateEnemyController`, fix `isRespawn` reset.

### Gap 4 — Episode fitness (RNG fake metrics)
- Harness `runEpisode` (`main-runner.ts:309-332`) generates all metrics via `seedrandom` RNG — no gameplay.
- Host `runEpisode` (`episode.ts:373-404`) plays real game with "deterministic damage bot" — no controller inputs.
- Fitness: `baseScore = survivalTicks*1 + damageDealt*2 + kills*5 - damageTaken*1 - aimMissRate*1 + complexityBonus*0.1`.
- `buildMainAgentEmbryo()` produces a topology descriptor, not an executable network.
- Cadence: 2 gen/min minimum → 30s/generation; real 20s × 8 variants = 160s — VIOLATES.
- Medium-high complexity; needs genome→controller materialization, `updateEpisode` input plumbing, telemetry extraction, cadence compromise.

### Gap 5 — Input handoff (no NEAT controller path)
- `GameTickInputSnapshot`: `{move: Vector2, lookDelta, fire, dash}` — only 4 fields.
- `Network.activate(input)` available from `src/architecture/network/network.ts:1100-1166`.
- Need sensor extraction from `GameState` + network activation → `GameTickInputSnapshot`.
- Cleanest insertion: add auto-mode controller in worker that reads `gameState`, calls `network.activate(sensors)`, writes `pendingTickInput` directly.
- ~200-400 LOC, medium complexity.

### Gap 6 — Generation counter (never incremented)
- `GameState.generation` initialized to 1 (`state.ts:107`), never incremented.
- Harness has own generation counters, never synced.
- Small complexity (~30-80 LOC).
- Best approach: sync from harness counter to `GameState.generation` on generation advance.

## Traceability

| Gap | Deliverable | Phase | Primary files |
|-----|-------------|-------|---------------|
| 6 | Generation counter sync | Phase 2 | `host/game/state.ts`, `host/game/tick.ts`, `worker/display.worker.ts`, `host/game/waves.ts` |
| 3 | Enemy MLP weight injection | Phase 2 | `scripts/enemy-controller.ts`, `harness/enemy-mlp.ts`, `worker/display.worker.ts` |
| 2 | Arms-race wiring (advanceWave) | Phase 3 | `host/waves.ts`, `worker/display.worker.ts`, `host/game/waves.ts` |
| 1 | Worker humanMode branching | Phase 4 | `worker/display.worker.ts`, `browser-entry.ts`, `host/renderer-bridge.ts`, `harness/arms-race.ts` |
| 5 | Player NEAT controller | Phase 4 | `worker/display.worker.ts`, `host/game/tick.ts`, `scripts/enemy-navigation.ts`, `src/architecture/network/network.ts` |
| 4 | Real episode fitness | Phase 5 | `harness/main-runner.ts`, `host/game/episode.ts`, `host/game/tick.ts`, `scripts/enemy-controller.ts`, `host/game/combat.ts`, `host/game/types.ts`, `harness/fitness.ts`, `harness/constants.ts` |

**Phase execution order:** 2 → 3 → 4 → 5 → 6 (dependency order). Each phase depends on the prior phase's deliverables.

## Implementation phases

### Phase 0 — Fix failing gun sprite / renderer tests [DONE]

**Phase objective:** Fix 8 failing tests across 3 test files (gun.test.ts, gun-sprite-data.test.ts, display.worker.test.ts) before starting implementation phases. Three root causes: (1) gun-sprite-data.js sprite grid is too narrow vertically — non-transparent bounds ratio is ~0.67, not the required ~1.6 width/height; (2) palette indices 5 and 6 have alpha 220 instead of 255, breaking the accent-palette alpha-preservation test; (3) gun.ts renderer uses `rgb(r,g,b)` strings for accent-colored pixels instead of the `NEATENSTEIN_GUN_ACCENT_COLOR` hex constant, failing the worker `fillStyle` assertion.

**Pragmatic mode:** This phase uses broad slices (2 slices, one per file) per the plan mandates. No red-phase ceremony — tests already exist and are failing (red). Fix-then-green directly.

```yaml
phase: 0
title: 'Fix failing gun sprite / renderer tests'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 1 — Planning & acceptance criteria'
skills:
  - implementation-standards
  - red-test-contracts
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test|display\.worker)'
acceptance_criteria:
  - id: AC-P0-001
    text: 'All 8 previously-failing neatenstein tests pass (0 failures)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test|display\.worker)'
  - id: AC-P0-002
    text: 'gun-sprite-data.js idle frame non-transparent bounds produce ~1.6 width/height aspect ratio'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-003
    text: 'Palette indices 5 and 6 have alpha 255 (not 220) so buildGunAccentPalette preserves alpha correctly'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-004
    text: 'gun.ts renderer sets fillStyle to NEATENSTEIN_GUN_ACCENT_COLOR (#00f0ff) for accent-colored pixels'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*display\.worker'
  - id: AC-P0-005
    text: 'No new test failures introduced (existing 1387 passing tests remain green)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
placeholder_steps:
  - 'Step 01 — Redesign gun-sprite-data.js sprite grid + fix palette alpha'
  - 'Step 02 — Fix gun.ts accent color rendering'
```

#### Step 01: Redesign gun-sprite-data.js sprite grid + fix palette alpha [DONE]

```yaml
phase: 0
step: 1
title: 'Redesign gun-sprite-data.js sprite grid + fix palette alpha'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Fix gun.ts accent color rendering'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test)'
acceptance_criteria:
  - id: AC-P0-006
    text: 'AC-04c-001: idle frame non-transparent bounds width/height ≈ 1.6 (toBeCloseTo 1.6, 0)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-007
    text: 'AC-04c-003: upper half (rows 0-11) majority index 4 or 8 (metallic/neon-white > 50%)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-008
    text: 'AC-04c-003: topmost row (row 0) has at least one cell with index 5 or 6 (teal muzzle ring)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-009
    text: 'AC-04c-004: half-widths have at least 3 distinct values AND a sharp drop >= 2 at barrel/receiver boundary'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-010
    text: 'AC-04c-011: row 4 col 19 in idle grid maps to palette index 4 (pure white [255,255,255,255])'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-011
    text: 'AC-04c-013: palette indices 5 and 6 have alpha 255; buildGunAccentPalette returns [255,0,128,255] for both'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun-sprite-data'
  - id: AC-P0-012
    text: 'AC-04c-002: measured body silhouette from fillRect calls ≈ 1.6 width/height at 640x360 and 2560x1080'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun\.test'
slices:
  - slice_id: 'P0S1-sprite-redesign'
    title: 'Redesign IDLE_GRID and FIRE_GRID for horizontal 1.6 ratio + fix palette alpha'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/gun-sprite-data.js'
    acceptance_criteria:
      - 'Non-transparent bounds of idle frame produce width/height ≈ 1.6'
      - 'Topmost row (row 0) has at least one index-5 or index-6 cell'
      - 'Upper half (rows 0-11) majority cells are index 4 or 8'
      - 'Row 4 col 19 is index 4 (pure white)'
      - 'Half-widths have >= 3 distinct values with a drop >= 2 at barrel/receiver boundary'
      - 'Palette indices 5 and 6 have alpha 255 (not 220)'
      - 'gun.test.ts AC-04c-002 measured body ratio ≈ 1.6 passes'
    parallelizable: false
    dependencies: []
    next_slice: 'P0S1-green'
  - slice_id: 'P0S1-green'
    title: 'Green validation: verify all sprite+gun tests pass and no regressions'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - 'All 7 previously-failing gun-sprite-data.test.ts and gun.test.ts tests pass (0 failures)'
      - 'No new failures in neatenstein test suite'
      - 'npm run build exit 0'
    parallelizable: false
    dependencies:
      - 'P0S1-sprite-redesign'
    next_slice: null
```

**Step objective:** Redesign the 40×24 palette-indexed sprite grid so the non-transparent content fills a wide horizontal area achieving ~1.6 width/height ratio, with correct material distribution (neon-white barrel, teal muzzle ring, stepped profile). Also fix palette alpha values for indices 5 and 6 from 220 to 255.

**Context the agent must know:**

Current state of `examples/neatenstein/gun-sprite-data.js`:
- Grid is 40 wide × 24 tall. Non-transparent content spans roughly columns 13-26 (width ~14) and rows 3-23 (height ~21), giving ratio ~0.67 — must be ~1.6.
- To achieve 1.6: if height stays ~21 rows, width must be ~34 cells. Or reduce height to ~14 rows with width ~22. The grid is 40×24, so aim for non-transparent bounds of roughly 32×20 (32/20 = 1.6) or 28×18 (28/18 ≈ 1.56) or 24×15 (24/15 = 1.6).
- The FIRE_GRID must be redesigned to match the new IDLE_GRID profile (same barrel/receiver shape, with muzzle flash burst added at top).
- Palette index 5 is `[0, 240, 255, 220]` — alpha must be 255. Index 6 is `[0, 200, 220, 220]` — alpha must be 255.
- Row 4, col 19 must be index 4 (`[255, 255, 255, 255]` — pure white). Currently it is index 6.
- Topmost row (row 0) must have at least one cell with index 5 or 6 (teal muzzle ring). Currently row 0 is all zeros.
- Upper half (rows 0-11) must be majority index 4 or 8 (metallic/neon-white). Currently many cells are index 6.
- Half-widths must have ≥ 3 distinct values with a drop ≥ 2 (barrel wider than receiver, or stepped profile).

Design approach:
- Make the barrel span roughly columns 4-35 (width ~32) in upper rows, narrowing to a receiver of width ~20 in lower rows. Height of ~20 rows gives 32/20 = 1.6.
- Row 0: teal muzzle ring cells (index 5 or 6) at the barrel tip center.
- Rows 1-3: barrel tip with teal accents.
- Rows 4-11: wide neon-white barrel (majority index 4, some index 8 for metallic shading, some index 5/6 for energy strips).
- Rows 12-23: narrower dark receiver body (indices 1, 2, 3), creating the stepped profile (drop >= 2 in half-width).
- Ensure row 4, col 19 is index 4.

**Execution steps:**

1. Read `examples/neatenstein/gun-sprite-data.js` to understand the current grid and palette.
2. Read the test file `examples/neatenstein/browser-entry/renderer/gun-sprite-data.test.ts` to confirm all failing assertions.
3. Redesign `IDLE_GRID`: widen non-transparent content to achieve ~1.6 width/height ratio. Ensure row 0 has teal, upper half is majority index 4, row 4 col 19 is index 4, and half-widths have a sharp drop ≥ 2.
4. Redesign `FIRE_GRID` to match the new IDLE shape with muzzle-flash burst (index 7) added at top rows.
5. Fix `GUN_SPRITE_PALETTE`: change index 5 alpha from 220 to 255, index 6 alpha from 220 to 255.
6. Run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test)` to verify all 7 sprite+gun failures are fixed.

**Stop conditions:** All 7 sprite-data and gun.test.ts failures pass. If a test still fails after redesign, adjust the grid until all constraints are met.

**Required validation:**
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test)` — 0 failures
- `npm run build` — exit 0
- `npm run lint` — exit 0 (only if gun-sprite-data.js is a .js file that eslint covers)

**Plan update requirement:** Update the plan with slice status, evidence, and next step before ending.

---

#### Step 02: Fix gun.ts accent color rendering [DONE]

```yaml
phase: 0
step: 2
title: 'Fix gun.ts accent color rendering'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 1 — Planning & acceptance criteria'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*display\.worker'
acceptance_criteria:
  - id: AC-P0-013
    text: 'display.worker.test.ts "draws the gun overlay in the worker tier" passes — fillStyle called with NEATENSTEIN_GUN_ACCENT_COLOR (#00f0ff)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*display\.worker'
  - id: AC-P0-014
    text: 'No regression in existing gun.test.ts tests (all previously passing tests remain green)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun\.test'
slices:
  - slice_id: 'P0S2-accent-color'
    title: 'Make gun.ts renderer use NEATENSTEIN_GUN_ACCENT_COLOR hex for accent pixels'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
    acceptance_criteria:
      - 'renderGunOverlay sets ctx.fillStyle to NEATENSTEIN_GUN_ACCENT_COLOR (#00f0ff) when drawing accent-colored pixels (palette index 5 or 6)'
      - 'display.worker.test.ts gun overlay test passes'
      - 'All existing gun.test.ts tests remain green'
    parallelizable: false
    dependencies:
      - 'P0S1-sprite-redesign'
    next_slice: 'P0S2-green'
  - slice_id: 'P0S2-green'
    title: 'Green validation: verify accent color test passes and no regressions'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - 'display.worker.test.ts gun overlay test passes (fillStyle called with #00f0ff)'
      - 'All existing gun.test.ts tests remain green'
      - 'npm run build exit 0'
    parallelizable: false
    dependencies:
      - 'P0S2-accent-color'
    next_slice: null
```

**Step objective:** Make the `renderGunOverlay` function in `gun.ts` set `ctx.fillStyle` to the `NEATENSTEIN_GUN_ACCENT_COLOR` hex constant (`#00f0ff`) when drawing accent-colored pixels (palette indices 5 and 6), instead of converting the palette RGBA to an `rgb()` string.

**Context the agent must know:**

Current state of `examples/neatenstein/browser-entry/renderer/gun.ts`:
- `renderGunOverlay` decodes the sprite frame via `decodeGunSpriteFrame`, then iterates over logical cells.
- For each non-transparent pixel, it sets `ctx.fillStyle` to either `rgba(r,g,b,a/255)` or `rgb(r,g,b)` based on the decoded RGBA values.
- The worker test (`display.worker.test.ts:960-962`) expects `setters.fillStyle` to have been called with `NEATENSTEIN_GUN_ACCENT_COLOR` which is `"#00f0ff"`.
- The current renderer never uses the hex constant — it only uses `rgb()`/`rgba()` strings — so the test fails.
- `NEATENSTEIN_GUN_ACCENT_COLOR` is already imported in `gun.ts` (line 15) and re-exported (line 36). It equals `#00f0ff`.
- The accent color corresponds to palette indices 5 and 6. After Step 01 fixes the palette alpha to 255, the decoded RGBA for accent pixels will be `[0, 240, 255, 255]` and `[0, 200, 220, 255]`.

Fix approach:
- When rendering a pixel whose logical palette index is 5 or 6, set `ctx.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR` (the hex string `#00f0ff`) instead of the `rgb()` string.
- To do this, the renderer needs to know the palette index for each cell. Currently it only has decoded RGBA. Options:
  - (A) Look up the palette index from the raw frame grid (before decode) alongside the decoded data.
  - (B) Compare the decoded RGBA to the accent palette colors and use the hex constant when they match.
  - (C) Pass the raw frame grid to the renderer loop and check `frame[row][col]` for indices 5/6.
- Option (C) is cleanest: the renderer already has access to `GUN_SPRITE_FRAMES` and selects `frame` at line 97. Use `frame[row][col]` to check for accent indices.
- When `frame[row][col]` is 5 or 6: `ctx.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR` (regardless of alpha — after the fix alpha is 255 so it's fully opaque).
- When `frame[row][col]` is 7 and alpha < 255: keep the `rgba()` path for the semi-transparent muzzle flash.
- All other non-transparent cells: keep the `rgb()` path.

**Execution steps:**

1. Read `examples/neatenstein/browser-entry/renderer/gun.ts` to understand the current rendering loop.
2. Read `examples/neatenstein/browser-entry/worker/display.worker.test.ts` lines 949-963 to confirm the test expectation.
3. Modify `renderGunOverlay` in `gun.ts`: inside the cell iteration loop, before setting `fillStyle`, check if the logical palette index (`frame[row][col]`) is 5 or 6. If so, set `ctx.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR`. Otherwise, keep the existing `rgb()`/`rgba()` logic.
4. Import `NEATENSTEIN_GUN_ACCENT_COLOR` is already available (line 15). Use it directly.
5. Run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*display\.worker` to verify the worker test passes.
6. Run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun\.test` to verify no regression in gun.test.ts.

**Stop conditions:** The display.worker.test.ts gun overlay test passes AND all gun.test.ts tests remain green.

**Required validation:**
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*display\.worker` — 0 failures for the gun overlay test
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*gun\.test` — 0 failures
- `npm run build` — exit 0
- `npm run lint` — exit 0

**Plan update requirement:** Update the plan with step status [DONE], evidence (all 8 tests green), and set Phase 0 to [DONE] before advancing to Phase 1.

```yaml
PlanUpdate:
  boundary: 'Phase 0 / Step 02 / slice P0S2-green'
  status: '[DONE]'
  what_changed:
    - 'examples/neatenstein/gun-sprite-data.js — redesigned IDLE_GRID and FIRE_GRID for horizontal 1.6 width/height ratio (32x20 non-transparent bounds); fixed palette indices 5 and 6 alpha from 220 to 255'
    - 'examples/neatenstein/browser-entry/renderer/gun.ts — renderGunOverlay now sets fillStyle to NEATENSTEIN_GUN_ACCENT_COLOR (#00f0ff) for palette index 5/6 cells'
  evidence:
    - 'npx jest --testPathPatterns="neatenstein.*(gun-sprite-data|gun\.test|display\.worker)" — 3 suites passed, 119 tests passed, 0 failures'
    - 'npx jest --testPathPatterns="neatenstein" — 74 suites passed, 1395 passed, 1 skipped, 0 failures (no regressions)'
    - 'npm run build — exit 0 (webpack + tsc)'
    - 'npx eslint on changed files — exit 0'
  removals:
    - 'Old vertical gun sprite grid (narrow 14-cell-wide barrel with ratio ~0.67) replaced by wide horizontal 32-cell barrel with ratio 1.6'
    - 'Old palette alpha 220 for indices 5 and 6 replaced by 255'
    - 'Old rgb() fillStyle path for accent pixels replaced by NEATENSTEIN_GUN_ACCENT_COLOR hex constant'
  next_boundary: 'Phase 1 — Planning & acceptance criteria [WIP]'
```

## Latest validation evidence

**Phase 0 verification (2025-01-24):**
- **status:** green-light
- **All 8 previously-failing tests now pass:**
  1. AC-04c-001: idle frame non-transparent bounds ratio = 32/20 = 1.6 ✓
  2. AC-04c-002: measured body silhouette from fillRect ≈ 1.6 at 640×360 and 2560×1080 ✓
  3. AC-04c-003: upper half (rows 0-11) majority index 4/8 = 96.4% ✓
  4. AC-04c-003: topmost row (row 0) has teal (index 5) ✓
  5. AC-04c-004: 9 distinct half-width values, max drop = 4.0 ✓
  6. AC-04c-011: row 4 col 19 = index 4 (pure white) ✓
  7. AC-04c-013: palette indices 5,6 alpha = 255; buildGunAccentPalette returns [255,0,128,255] ✓
  8. display.worker.test.ts: fillStyle called with NEATENSTEIN_GUN_ACCENT_COLOR (#00f0ff) ✓
- **No regressions:** 1395 passed, 1 skipped, 0 failed (was 1387 passed, 8 failed, 1 skipped before fix)
- **Build:** npm run build exit 0
- **Lint:** npx eslint on changed files exit 0
- **slice-advancement gate:** plan-sync ✓, step-packet ✓, plan-slice-quality ✓, plan-command-lint ✓ (shared-validation and code-coverage fail only due to pre-existing state; all Phase 0 tests now pass)

---

### Phase 1 — Planning & acceptance criteria [WIP]

**Phase objective:** Author all step packets for the five implementation phases from the research findings.

**Stop conditions:** Plan tracker is malformed, or a value-adding step lacks machine-readable acceptance criteria.

```yaml
phase: 1
title: 'Planning & acceptance criteria'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 2 — Foundation: generation sync + enemy MLP activation'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-auto-neat-mode.plans.md'
acceptance_criteria:
  - id: AC-001
    text: 'Step packets for Phases 2-6 are authored with machine-readable YAML blocks'
    validation: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-auto-neat-mode.plans.md'
  - id: AC-002
    text: 'Each phase has observable acceptance criteria mapped to validation commands'
    validation: 'manual review of plan YAML blocks'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Author step packets for Phases 2-6'
```

#### Step 01: Author step packets for Phases 2-6 [WIP]

```yaml
phase: 1
step: 1
title: 'Author step packets for Phases 2-6'
status: '[WIP]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 2 — Foundation: generation sync + enemy MLP activation'
skills:
  - plan-alignment
  - planning-acceptance-criteria
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-auto-neat-mode.plans.md'
acceptance_criteria:
  - id: AC-003
    text: 'Phase 2-6 step packets authored with slice definitions'
    validation: 'manual review of plan'
```

**Step objective:** Convert the five research findings into machine-readable step packets with slices, acceptance criteria, and traceability for downstream implementation agents.

**Context the agent must know:** The research artifact `plans/neatenstein-auto-neat-mode.research.md` contains the full gap analysis. Five research specialists investigated the six gaps and produced detailed file-level findings. This plan is built from those findings.

---

### Phase 2 — Foundation: generation sync + enemy MLP activation [PLANNED]

**Phase objective:** Wire the generation counter sync (Gap 6) and enemy MLP weight injection (Gap 3) — the two lowest-risk, highest-leverage changes that make enemy evolution functional in the live game.

```yaml
phase: 2
title: 'Foundation: generation sync + enemy MLP activation'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 3 — Arms-race wiring: advanceWave into live game'
skills:
  - implementation-standards
  - nge-core-algorithm
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(state|waves|enemy-controller|enemy-mlp)'
acceptance_criteria:
  - id: AC-010
    text: 'GameState.generation increments on wave-clear and syncs from harness counter'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*state'
  - id: AC-011
    text: 'Enemy controllers receive evolved MLP weights from the population snapshot'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
  - id: AC-012
    text: 'MLP re-ranking path is exercised (weights !== undefined) in live worker context'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-mlp'
  - id: AC-013
    text: '100% coverage on touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(state|enemy-controller)'
  - id: AC-013a
    text: 'Same-seed replay: identical seed + generation produces identical MLP population (Level 2 deterministic)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-mlp'
  - id: AC-013b
    text: 'Worker clamps timestep to NEATENSTEIN_FIXED_TIMESTEP_MS (not rAF deltaMs) for deterministic simulation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-013c
    text: 'Same-seed main-agent replay: identical numeric seed + generation produces identical Neat population (neatPop.getFittest() is deterministic with hashSeed)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
placeholder_steps:
  - 'Step 01 — Wire generation counter sync (Gap 6)'
  - 'Step 02 — Wire enemy MLP population + weight injection (Gap 3)'
```

#### Step 01: Wire generation counter sync (Gap 6) [PLANNED]

```yaml
phase: 2
step: 1
title: 'Wire generation counter sync (Gap 6)'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Wire enemy MLP population + weight injection (Gap 3)'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*state'
acceptance_criteria:
  - id: AC-014
    text: 'Wave-clear detection tracks allEnemiesCleared transition (detection mechanism for P3S1 advanceWave wiring)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*state'
  - id: AC-015
    text: 'Worker clamps timestep to NEATENSTEIN_FIXED_TIMESTEP_MS (fixed 16ms, not rAF deltaMs)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-016
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*state'
slices:
  - slice_id: 'P2S1-gen-sync'
    title: 'Wire generation counter increment on wave-clear + harness sync'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: AC-017
        text: 'Wave-clear detection triggers allEnemiesCleared flag (consumed by P3S1 advanceWave)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
      - id: AC-018
        text: 'Worker timestep clamped to NEATENSTEIN_FIXED_TIMESTEP_MS (16ms fixed, not rAF delta)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Slice details:**
- Add wave-clear detection in `display.worker.ts` after `gameTick` (line 1253): track `allEnemiesCleared` transition `false → true`.
- **Generation increment ownership**: P2S1 does NOT increment `gameState.generation` — that is owned by `advanceWave` in P3S1. P2S1 only detects wave-clear and stores the flag for P3S1 to consume. When advanceWave is wired (P3S1), it increments generation; the worker writes back `result.generation`.
- `GameState.generation` starts at 1 (existing). Harness `generation` starts at 1 (existing). Sync happens in P3S1 when advanceWave is called.
- **Fixed timestep clamping**: In `display.worker.ts:1222-1223`, replace `const dtMs = data.deltaMs` with `const dtMs = NEATENSTEIN_FIXED_TIMESTEP_MS`. Import `NEATENSTEIN_FIXED_TIMESTEP_MS` from `host/game/constants.ts` (the authoritative source — `harness/constants.ts` will later re-export it in P5S2b, but P2S1 imports directly from the source to avoid depending on a not-yet-created re-export). The rAF delta only gates *when* a tick runs, not *how much* time advances. This is critical for replay determinism (Determinism Contract §2).
- Add tests in `host/game/state.test.ts` and `host/game/waves.test.ts` verifying wave-clear detection and timestep clamping.

#### Step 02: Wire enemy MLP population + weight injection (Gap 3) [PLANNED]

```yaml
phase: 2
step: 2
title: 'Wire enemy MLP population + weight injection (Gap 3)'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 3 — Arms-race wiring: advanceWave into live game'
skills:
  - implementation-standards
  - nge-core-algorithm
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(enemy-controller|enemy-mlp)'
acceptance_criteria:
  - id: AC-019
    text: 'Worker instantiates MlpEnemyPopulation on init and tracks generation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-mlp'
  - id: AC-020
    text: 'Enemy controllers receive champion snapshot weights (not undefined) after population update'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
  - id: AC-021
    text: 'isRespawn re-derives weights from current snapshot instead of forcing undefined'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
  - id: AC-022
    text: 'MLP re-ranking path is live (weights !== undefined triggers activateMlp)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
  - id: AC-023
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(enemy-controller|enemy-mlp)'
slices:
  - slice_id: 'P2S2-enemy-mlp-inject'
    title: 'Instantiate MLP population in worker + inject champion weights into enemy controllers'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/scripts/enemy-controller.ts'
    acceptance_criteria:
      - id: AC-024
        text: 'Worker creates MlpEnemyPopulation on init with seed derived from gameState.seed'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-mlp'
      - id: AC-025
        text: 'updateEnemyController receives optional weights?: Float32Array (backward-compatible — undefined = existing BFS behavior)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
      - id: AC-026
        text: 'Respawned enemies get fresh weights from snapshot, not undefined'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
    parallelizable: false
    dependencies:
      - 'P2S1-gen-sync'
    next_slice: null
```

**Slice details:**
- In `display.worker.ts` init (near line 1181): `const enemyPopulation = createMlpEnemyPopulation({ seed: gameState.seed })`.
- Track generation counter; call `enemyPopulation.update({ generation })` to get `MlpSnapshot` on refresh generations.
- Extend `updateEnemyController` signature to accept an OPTIONAL `weights?: Float32Array | undefined` as the 5th parameter (backward-compatible — existing ~150 test call sites and 2 worker call sites continue to work with 4 args; `undefined` = existing BFS behavior). This is additive, not breaking.
- Fix `isRespawn` reset (line 397): on respawn, re-derive weights from the current snapshot rather than forcing `undefined`.
- The MLP re-ranking branch (line 645-711) is already fully implemented — it activates when `weights !== undefined`. This slice makes that branch live.
- **Determinism**: seed derivation is `gameState.seed` (root) → `createMlpEnemyPopulation({ seed })`. Same seed + same generation → same population.

---

### Phase 3 — Arms-race wiring: advanceWave into live game [PLANNED]

**Phase objective:** Wire `advanceWave()` into the live game's wave-clear path so the enemy MLP population evolves each generation, and wire `runArmsRaceGeneration()` on a generation cadence.

```yaml
phase: 3
title: 'Arms-race wiring: advanceWave into live game'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 4 — Player auto-mode controller'
skills:
  - implementation-standards
  - nge-core-algorithm
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(waves|arms-race|display.worker)'
acceptance_criteria:
  - id: AC-030
    text: 'advanceWave is called from the live worker on wave-clear, evolving the enemy population'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*waves'
  - id: AC-031
    text: 'Evolved snapshot weights flow from advanceWave to enemy controllers'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(waves|enemy-controller)'
  - id: AC-032
    text: 'runArmsRaceGeneration is called on generation cadence (async, off render thread)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
  - id: AC-033
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(waves|arms-race)'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Wire advanceWave into wave-clear detection'
  - 'Step 02 — Wire runArmsRaceGeneration on generation cadence'
```

#### Step 01: Wire advanceWave into wave-clear detection [PLANNED]

```yaml
phase: 3
step: 1
title: 'Wire advanceWave into wave-clear detection'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Wire runArmsRaceGeneration on generation cadence'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*waves'
acceptance_criteria:
  - id: AC-034
    text: 'Worker detects wave-clear (allEnemiesCleared false→true) and calls advanceWave'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*waves'
  - id: AC-035
    text: 'advanceWave result replaces worker gameState and feeds snapshot.weights to enemies'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(waves|enemy-controller)'
  - id: AC-036
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*waves'
slices:
  - slice_id: 'P3S1-advance-wave'
    title: 'Wire advanceWave into worker wave-clear detection + snapshot injection'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: AC-037
        text: 'Wave-clear edge detection triggers advanceWave with current generation'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*waves'
      - id: AC-038
        text: 'advanceWave returns evolved snapshot; worker injects weights into enemy controllers'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
    parallelizable: false
    dependencies:
      - 'P2S2-enemy-mlp-inject'
    next_slice: 'P3S2-arms-race-cadence'
```

**Slice details:**
- In `display.worker.ts` after `gameTick` (line 1253): detect `allEnemiesCleared(gameState.enemies)` transition `false → true`.
- On wave-clear: call `advanceWave(gameState, { population: enemyPopulation, spawnCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT })`.
- Replace `gameState` with `result.state`; inject `result.snapshot.weights` into enemy controllers (via the weight injection path from Phase 2).
- `advanceWave` already clears the arena, increments generation, and respawns enemies — replace the trickle `spawnWaveTick` path for wave *transitions*.

#### Step 02: Wire runArmsRaceGeneration on generation cadence [PLANNED]

```yaml
phase: 3
step: 2
title: 'Wire runArmsRaceGeneration on generation cadence'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 4 — Player auto-mode controller'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
acceptance_criteria:
  - id: AC-039
    text: 'Worker evaluates Neat population async (hoisted), then calls sync runArmsRaceGeneration with championNetwork param — does not block render'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
  - id: AC-040
    text: 'Arms-race result syncs generation back to GameState and updates enemy snapshot'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
  - id: AC-041
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*arms-race'
slices:
  - slice_id: 'P3S2-arms-race-cadence'
    title: 'Wire runArmsRaceGeneration on generation cadence with hoisted async Neat evaluation'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/harness/arms-race.ts'
      - 'examples/neatenstein/browser-entry/harness/types.ts'
      # Exception: hash-seed.ts (4th file) — new shared hash utility required to avoid circular harness import (arms-race.ts → main-runner.ts, main-runner.ts → hash-seed.ts). Documented exception per ≤3 files policy.
      - 'examples/neatenstein/browser-entry/harness/hash-seed.ts'
    acceptance_criteria:
      - id: AC-042
        text: 'runArmsRaceGeneration called with {seed, generation, enemySnapshot, humanMode: boolean, championNetwork?: Network}'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
      - id: AC-043
        text: 'Result generation syncs to GameState.generation; enemySnapshot updates enemy population'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
      - id: AC-043a
        text: 'Only one generation in flight at a time; results applied in generation order (not completion order)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
      - id: AC-060a
        text: 'createMainSnapshot returns MainVariant with optional network?: Network field (from championNetwork param, not placeholder null genome)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
      - id: AC-060b
        text: 'MainVariant type updated in harness/types.ts to include optional network?: Network field (with import type { Network })'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*types'
    parallelizable: false
    dependencies:
      - 'P3S1-advance-wave'
    next_slice: null
```

**Slice details:**
- **Hoisted evaluation model**: The Neat population evaluation is hoisted OUT of `runArmsRaceGeneration` into the worker. The worker:
  1. Creates and owns the Neat population: `const neatPop = new Neat(12, 5, fitnessFn, { popsize: 4, seed: hashSeed(seed, generation) })`.
  2. Calls `await neatPop.evaluate()` and `await neatPop.evolve()`. Note: `setTimeout(0)` chunking does NOT apply to Neat's `evaluate()`/`evolve()` internals — those are awaited as-is. The `setTimeout(0)` yielding applies to the headless episode tick loop INSIDE the fitness function (in `main-runner.ts`), which runs episode ticks in chunks of `NEATENSTEIN_EVAL_CHUNK_TICKS = 32` and yields via `setTimeout(0)` between chunks to allow `onmessage` to fire. Neat's async API is awaited normally; the chunking is in the episode simulation loop that the fitness function drives.
  3. Extracts the champion: `const championNetwork = neatPop.getFittest()` (returns `Network` directly — NOT `.network`).
  4. Passes the champion into a still-sync `runArmsRaceGeneration({ seed, generation, enemySnapshot, humanMode: humanModeBool, championNetwork })`.
- **hashSeed utility**: Define `hashSeed(seed, generation, variantId?)` as a small pure utility function in a new file `harness/hash-seed.ts`. This avoids a circular module dependency: `arms-race.ts` already imports `runMainGeneration` from `main-runner.ts`, so defining `hashSeed` in `arms-race.ts` would create a cycle when `main-runner.ts` (P5S1) imports `hashSeed` from `arms-race.ts`. The dedicated `harness/hash-seed.ts` module is imported by both `arms-race.ts` (P3S2 worker) and `main-runner.ts` (P5S1), breaking the cycle. Formula: `((seed * 100003 + generation) * 100003 + (variantId ?? 0)) >>> 0` — produces a 32-bit unsigned integer. P5S1 imports it from `hash-seed.ts` — no dependency on arms-race.ts or seed-pack.ts for the hash function.
- **runMainGeneration bypass**: When `championNetwork` is provided to `runArmsRaceGeneration`, the internal `runMainGeneration(...)` call is SKIPPED (short-circuited). The worker has already evaluated the Neat population; running `runMainGeneration` again would duplicate the ~1.2s evaluation work. The champion's quality signal is passed via a new optional `championQuality?: CombatQualitySignal` parameter to `runArmsRaceGeneration`. The worker extracts the champion's fitness score from `neatPop.getFittest().score` (Neat stores the best network's fitness as `.score`) and constructs the `CombatQualitySignal` from the episode telemetry. When `championNetwork` is undefined (existing test callers), `runMainGeneration` runs as before and produces the quality signal internally.
- This resolves the async/sync contradiction: `runArmsRaceGeneration` stays synchronous because the async Neat evaluation happens BEFORE the call, in the worker. The ~25 existing test callers don't pass `championNetwork` and get the placeholder behavior (backward-compatible).
- `createMainSnapshot` in `arms-race.ts` is updated to use the passed-in `championNetwork` parameter instead of the placeholder null genome. When `championNetwork` is undefined (existing callers), it falls back to the placeholder.
- `MainVariant` in `harness/types.ts` gets a new optional `network?: Network` field (with `import type { Network } from 'neataptic'` — the public entry point re-exports `Network` as a named export). The existing `genome` field stays for backward compatibility.
- **Type mapping**: `humanMode` from `simState` is `'auto' | 'human'` (string). Convert to boolean: `const humanModeBool = latestState.humanMode === 'auto'` before passing.
- **Scheduling model**: Worker runs full-speed headless episodes using `NEATENSTEIN_FIXED_TIMESTEP_MS=16`. Wall time per episode is ~300ms (312 ticks at full speed). 4 variants × ~300ms = ~1.2s wall time + evolution overhead ≈ 2-3s total. Well within the 30s/generation budget.
- **Non-blocking evaluation**: `requestIdleCallback` is NOT available in a Web Worker. Use `setTimeout(0)` micro-chunking — the async evaluation loop yields after every `NEATENSTEIN_EVAL_CHUNK_TICKS = 32` ticks (~512ms simulated), allowing `onmessage` to fire between chunks. Chunk size is a configurable constant in `harness/constants.ts`.
- **Cleanup rule**: The worker replaces the previous Neat population with the new one each generation. Only one population is in flight at a time. No unbounded retention.
- **Launch guard**: Before starting a new `await neatPop.evaluate()/evolve()` cycle, check `if (pendingGeneration !== null) return;` — skip if already evaluating. This prevents overlapping generation launches. Store `pendingGeneration = currentGeneration` (the generation being evaluated). On completion, `runArmsRaceGeneration` returns `generation + 1`; verify `result.generation === pendingGeneration + 1` before applying. Then clear `pendingGeneration = null`.
- **Async serialization**: Only one generation evaluation is in flight at a time. A `pendingGeneration` number guard stores the launched generation; on completion, `result.generation` must equal `pendingGeneration + 1` before applying (prevents out-of-order application). This accounts for `runArmsRaceGeneration`'s `generation + 1` return value.
- **Concurrent wave-clear handling**: If wave-clears arrive faster than the ~2-3s evaluation, the launch guard drops the new request (the next evaluation will pick up the latest generation). No queueing.
- Sync result: `gameState.generation = result.generation`; update `enemySnapshot` from `result.enemySnapshot`.
- The arms-race result's `mainSnapshot` (champion `network` field) is stored for Phase 4's player controller.

---

### Phase 4 — Player auto-mode controller [PLANNED]

**Phase objective:** Wire the worker to read `humanMode` and branch to a NEAT network controller that produces `GameTickInputSnapshot` from game-state sensors (Gaps 1 + 5).

```yaml
phase: 4
title: 'Player auto-mode controller'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 5 — Real episode Fitness'
skills:
  - implementation-standards
  - nge-core-algorithm
  - architecture-builder
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(display.worker|tick|controls)'
acceptance_criteria:
  - id: AC-050
    text: 'Worker reads humanMode from simState and branches to NEAT controller when auto'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-051
    text: 'NEAT controller extracts sensors from GameState and produces GameTickInputSnapshot'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*tick'
  - id: AC-052
    text: 'Network.activate() is called with sensor vector and outputs map to move/lookDelta/fire/dash'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(tick|display.worker)'
  - id: AC-053
    text: 'Human mode remains fully functional (humanMode: human uses InputRouter path)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-054
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(display.worker|tick)'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Worker humanMode branching + NEAT network loading (Gap 1)'
  - 'Step 02 — Sensor extraction + network activation → GameTickInputSnapshot (Gap 5)'
```

#### Step 01: Worker humanMode branching + NEAT network loading (Gap 1) [PLANNED]

```yaml
phase: 4
step: 1
title: 'Worker humanMode branching + NEAT network loading (Gap 1)'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Sensor extraction + network activation → GameTickInputSnapshot (Gap 5)'
skills:
  - implementation-standards
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
acceptance_criteria:
  - id: AC-055
    text: 'Worker reads simState.humanMode and branches: auto → NEAT controller, human → InputRouter'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-056
    text: 'Champion Network stored in worker scope from P3S2 hoisted Neat evaluation; used for auto-mode ticks'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-057
    text: 'In auto mode, tickInput comes from NEAT controller, not pendingTickInput (human queue)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-058
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*display.worker'
slices:
  - slice_id: 'P4S1-worker-branch'
    title: 'Worker reads humanMode, branches to NEAT controller, uses champion network from worker scope'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: AC-059
        text: 'simState handler branches on humanMode at line 1204-1217'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
      - id: AC-060
        text: 'Champion Network stored in worker scope from P3S2 hoisted evaluation; used for auto-mode ticks'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
    parallelizable: false
    dependencies:
      - 'P3S2-arms-race-cadence'
    next_slice: 'P4S2-sensor-activation'
```

**Slice details:**
- In `display.worker.ts:1204-1217`, inside the `simState` handler, after `latestState = data.state`:
  - Read `const humanMode = (data.state as any).humanMode`.
  - If `humanMode === 'auto'`: build `tickInput` from the NEAT controller (Phase 4 Step 02).
  - If `humanMode === 'human'` (or undefined): use `pendingTickInput` as before (existing path).
- The champion Network is already stored in worker scope from P3S2's hoisted Neat evaluation (`neatPop.getFittest()` returns a `Network` directly — NOT `.network`). No need to receive it via a separate message; it's already in the worker's local variable.
- The `createMainSnapshot` and `MainVariant` type updates were done in P3S2 (arms-race.ts and types.ts). P4S1 only touches the worker to add the humanMode branch.
- Store the champion Network reference in worker scope; on each `simState` tick in auto mode, call `network.activate(sensors)` → map to `GameTickInputSnapshot`.
- **Worker message protocol**: No new `setNetwork` message type is needed — the worker owns the Neat population and evaluates it locally. The champion Network is already in scope.

#### Step 02: Sensor extraction + network activation → GameTickInputSnapshot (Gap 5) [PLANNED]

```yaml
phase: 4
step: 2
title: 'Sensor extraction + network activation → GameTickInputSnapshot (Gap 5)'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 5 — Real episode fitness'
skills:
  - implementation-standards
  - architecture-builder
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(tick|display.worker|enemy-navigation)'
acceptance_criteria:
  - id: AC-061
    text: 'Sensor extractor (extractSensors) in scripts/enemy-navigation.ts builds 12-element observation vector from GameState'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-navigation'
  - id: AC-062
    text: 'Network.activate(sensors) returns output array mapped to GameTickInputSnapshot fields'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-063
    text: 'move.x, move.y ∈ [-1,1] (tanh); lookDelta = tanh(out)*maxTurnRate; fire = out > threshold'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-064
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(tick|display.worker)'
slices:
  - slice_id: 'P4S2-sensor-activation'
    title: 'Build sensor extractor from GameState + Network.activate → GameTickInputSnapshot mapper'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/scripts/enemy-navigation.ts'
    acceptance_criteria:
      - id: AC-065
        text: 'Sensor vector includes: player health, ammo, angle, x/y, nearest enemy bearing, distance, health, 4 wall raycasts'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
      - id: AC-066
        text: 'Network output mapped: out[0]→move.x, out[1]→move.y, out[2]→lookDelta, out[3]→fire, out[4]→dash'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
    parallelizable: false
    dependencies:
      - 'P4S1-worker-branch'
    next_slice: null
```

**Slice details:**
- Build `extractSensors(gameState: GameState): number[]` in `scripts/enemy-navigation.ts` (alongside existing `buildVisionVector` and `buildEnemyDistanceMap`):
  - Player: `health/maxHealth`, `ammo`, `angleRad`, `position.x`, `position.y` (5)
  - Nearest enemy: relative bearing (`atan2(dy,dx) - playerAngle`), distance (`hypot(dx,dy)`), `health` (3)
  - Wall raycasts: 4 cardinal directions, distance to nearest wall using existing `castRayDDAFromFlatMap` (4)
  - Total: 12 sensors (minimal set, expandable)
  - Placing the helper in `scripts/enemy-navigation.ts` (not `display.worker.ts`) allows the harness to reuse it for episode fitness evaluation in Phase 5 without importing worker code.
- Build `networkOutputToTickInput(outputs: number[]): GameTickInputSnapshot` in `display.worker.ts`:
  - `move.x = tanh(outputs[0])` (strafe)
  - `move.y = tanh(outputs[1])` (forward/back)
  - `lookDelta = tanh(outputs[2]) * MAX_TURN_RATE` (radians)
  - `fire = outputs[3] > 0` (threshold)
  - `dash = outputs[4] > 0.5` (threshold)
- Use `Network.activate(sensors)` or `noTraceActivate(sensors)` from `src/architecture/network/network.ts`.
- The network is the champion stored in worker scope from P3S2's hoisted evaluation (`neatPop.getFittest()` — returns `Network` directly). The embryo-to-NeatGenome converter is a follow-up (see Open Assumption #2).

---

### Phase 5 — Real episode fitness [PLANNED]

**Phase objective:** Replace the RNG-based `runEpisode` stub in `main-runner.ts` with real gameplay evaluation (headless fast-forward), add telemetry extraction, and solve the cadence constraint.

```yaml
phase: 5
title: 'Real episode fitness'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Phase 6 — Integration testing & browser validation'
skills:
  - implementation-standards
  - nge-core-algorithm
  - reproducibility-contracts
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(main-runner|episode|fitness)'
acceptance_criteria:
  - id: AC-070
    text: 'Harness runEpisode uses real gameplay (headless fast-forward) instead of RNG'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
  - id: AC-071
    text: 'Telemetry extraction maps GameState → CombatQualitySignal (damageDealt, aimMissRate tracked)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*fitness'
  - id: AC-072
    text: 'Cadence target met: generations-per-minute >= 2 with real episodes (full-speed headless, not wall-clock)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*cadence'
  - id: AC-073
    text: 'RNG-based runEpisode stub removed (no dual-path code)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
  - id: AC-074
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(main-runner|episode|fitness)'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Bridge harness episode to real gameplay (Gap 4)'
  - 'Step 02 — Telemetry extraction + cadence optimization'
```

#### Step 01: Bridge harness episode to real gameplay (Gap 4) [PLANNED]

```yaml
phase: 5
step: 1
title: 'Bridge harness episode to real gameplay (Gap 4)'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Telemetry extraction + cadence optimization'
skills:
  - implementation-standards
  - nge-core-algorithm
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
acceptance_criteria:
  - id: AC-075
    text: 'runEpisode in main-runner.ts calls host/game/episode.ts runEpisode (or headless fast-forward)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
  - id: AC-076
    text: 'gameTick (existing single input-handling path) receives NEAT controller output — no dual-path in updateEpisode'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
  - id: AC-077
    text: 'RNG-based runEpisode stub deleted (no dual-path code)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
  - id: AC-078
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*main-runner'
slices:
  - slice_id: 'P5S1-real-episode'
    title: 'Replace RNG runEpisode with real gameplay via gameTick (existing single input path)'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/harness/main-runner.ts'
      - 'examples/neatenstein/browser-entry/host/game/episode.ts'
      - 'examples/neatenstein/browser-entry/harness/seed-pack.ts'
      # Exception: barrier.ts (4th file) — one-line change: pass options.seed to createSeedPack for end-to-end seed determinism. The NEATENSTEIN_MAIN_VARIANT_COUNT centralization is deferred to P5S2b (which already touches constants.ts) to avoid a missing-export compile error. Documented exception per ≤3 files policy.
      - 'examples/neatenstein/browser-entry/harness/barrier.ts'
    acceptance_criteria:
      - id: AC-079
        text: 'main-runner runEpisode calls host episode with NEAT controller inputs via gameTick (no new parameters on updateEpisode)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
      - id: AC-080
        text: 'gameTick receives NEAT controller output as tickInput; worker keeps explicit updateEnemyController pass (gameTick does NOT call updateEnemyController internally)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
      - id: AC-081
        text: 'Deterministic damage bot removed; NEAT network drives player in evaluation'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
      - id: AC-081a
        text: 'createSeedPack accepts optional seed?: number parameter: createSeedPack({ generation, variantCount, seed? }) — numeric hash for per-variant seeds (backward-compatible: omitted seed = existing behavior)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*seed-pack'
    parallelizable: false
    dependencies:
      - 'P4S2-sensor-activation'
    next_slice: 'P5S2a-telemetry'
```

**Slice details:**
- Delete the RNG-based `runEpisode` in `main-runner.ts:309-332` (no dual-path code per Mandates).
- Replace with a call to the host `runEpisode` (or a headless fast-forward variant):
  - Build an `Episode` from `(seed, generation, variantId, enemySnapshot)`.
  - Construct a single `Network` from the variant's genome (NOT a full `Neat` population — that would allocate/speciate a whole evolutionary controller per variant per generation, blowing the ~300ms/episode budget). Use `Network.fromJSON(variant.genome)` or the equivalent genome constructor to get a single executable network. Then call `network.activate(sensors)` each tick. The `Neat` population is created ONCE in the worker (P3S2) and its member networks are extracted per-variant for episode evaluation. The seed for the variant's network is derived via `hashSeed(seed, generation, variantId)` — but this seeds the network's RNG state (if any), not a population.
  - For each step: `extractSensors(state)` (from `scripts/enemy-navigation.ts`) → `network.activate(sensors)` → `networkOutputToTickInput(outputs)` → `gameTick(state, tickInput, collisionMap, dtMs)` (correct parameter order: state, snapshot, collisionMap?, dtMs?).
  - **Key correction**: `gameTick` in `host/game/tick.ts` accepts `Partial<GameTickInputSnapshot>` as its 2nd parameter and handles player movement/fire. However, `gameTick` does NOT call `updateEnemyController` internally — the worker must keep the explicit `updateEnemyController` pass (as it does today at `display.worker.ts:1227`). Route controller inputs through `gameTick` for the player, and keep the separate `updateEnemyController` call for enemies. Do NOT add a `controllerInput` parameter to `updateEpisode` — `updateEpisode` is called by `gameTick` internally and doesn't need modification.
  - Remove the "deterministic damage bot" — the NEAT network drives the player via `gameTick`.
  - **Determinism**: Episode uses `NEATENSTEIN_FIXED_TIMESTEP_MS = 16` for all ticks. Same seed + same genome → same fitness. `createSeedPack` must fold in `seed` (not just `generation`).
  - **Episode tick count**: `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS = 5000` → 312 ticks at 16ms. The fitness evaluation path uses a NEW constant `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = Math.floor(NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312 (added in P5S2b). The existing shared `NEATENSTEIN_MAX_EPISODE_TICKS = 240` is NOT changed — it remains used by `enemy-runner.ts` for the enemy barrier evaluation path. The local `NEATENSTEIN_MAX_EPISODE_TICKS = 240` in `main-runner.ts:54` and `barrier.ts:95` are replaced with `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` (imported from `constants.ts`) in P5S2b so the fitness path uses 312 ticks while the enemy barrier path stays at 240.
  - **barrier.ts update**: One-line change: `createSeedPack({ generation: options.generation })` → `createSeedPack({ generation: options.generation, seed: options.seed })` — ensures end-to-end seed determinism in the barrier path. (The NEATENSTEIN_MAIN_VARIANT_COUNT centralization for barrier.ts is deferred to P5S2b, which already touches constants.ts and can add the export in the same slice.)
  - **main-runner.ts seed propagation**: Line 133 changes from `createSeedPack({ generation })` to `createSeedPack({ generation, seed: rootSeed })` where `rootSeed` is passed from the worker's `gameState.seed`. Per-variant seeds are derived inside `createSeedPack` as `seeds[i] = hashSeed(seed, generation, i)`.

#### Step 02: Telemetry extraction + cadence optimization [PLANNED]

```yaml
phase: 5
step: 2
title: 'Telemetry extraction + cadence optimization'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Phase 6 — Integration testing & browser validation'
skills:
  - implementation-standards
  - reproducibility-contracts
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(fitness|cadence)'
acceptance_criteria:
  - id: AC-082
    text: 'Telemetry extractor maps final GameState → CombatQualitySignal with real metrics'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*fitness'
  - id: AC-083
    text: 'damageDealt accumulated from applyEnemyDamage calls during episode'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
  - id: AC-084
    text: 'aimMissRate = (shotsFired - shotsHit) / shotsFired (tracked in fireBolt)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
  - id: AC-085
    text: 'Cadence: episode duration = 5s simulated at full-speed headless (312 ticks, ~300ms wall); NEATENSTEIN_MAIN_VARIANT_COUNT=4; meets 2 gen/min'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*cadence'
  - id: AC-086
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(fitness|cadence)'
slices:
  - slice_id: 'P5S2a-telemetry'
    title: 'Add telemetry tracking to combat system + GameState types'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/combat.ts'
      - 'examples/neatenstein/browser-entry/host/game/types.ts'
    acceptance_criteria:
      - id: AC-087
        text: 'EpisodeTelemetry object (or GameState fields) tracks damageDealt, shotsFired, shotsHit'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*combat'
      - id: AC-087a
        text: 'fireBolt increments shotsFired; applyEnemyDamage increments damageDealt and shotsHit'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*combat'
    parallelizable: false
    dependencies:
      - 'P5S1-real-episode'
    next_slice: 'P5S2b-fitness-cadence'
  - slice_id: 'P5S2b-fitness-cadence'
    title: 'Fitness signal extraction from real telemetry + cadence constants'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/harness/fitness.ts'
      - 'examples/neatenstein/browser-entry/harness/constants.ts'
      - 'examples/neatenstein/browser-entry/harness/main-runner.ts'
      # Exception: barrier.ts (4th file) — remove local NEATENSTEIN_MAIN_VARIANT_COUNT=8 and import from constants.ts. Must land in same slice that adds the export to avoid missing-import. Documented exception per ≤3 files policy.
      - 'examples/neatenstein/browser-entry/harness/barrier.ts'
    acceptance_criteria:
      - id: AC-088
        text: 'extractCombatQualitySignal(gameState, telemetry) produces CombatQualitySignal from real metrics'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*fitness'
      - id: AC-089
        text: 'NEATENSTEIN_MAIN_VARIANT_COUNT added to harness/constants.ts (value 4); local constants in main-runner.ts:46 AND barrier.ts:87 removed and imported from constants.ts; NEATENSTEIN_FITNESS_EPISODE_DURATION_MS = 5000; NEATENSTEIN_EVAL_CHUNK_TICKS = 32 (configurable); NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS derived from duration/timestep (new constant, does NOT change existing NEATENSTEIN_MAX_EPISODE_TICKS=240 used by enemy-runner.ts); NEATENSTEIN_FIXED_TIMESTEP_MS re-exported from host/game/constants.ts (not redefined)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(constants|main-runner)'
    parallelizable: false
    dependencies:
      - 'P5S2a-telemetry'
    next_slice: 'P5S2c-bfs-pooling'
  - slice_id: 'P5S2c-bfs-pooling'
    title: 'BFS buffer pooling for buildEnemyDistanceMap'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-navigation.ts'
    acceptance_criteria:
      - id: AC-090
        text: 'buildEnemyDistanceMap accepts optional distances?: Int32Array; callers pass reused buffer (worker-scoped for live, per-episode for headless)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-navigation'
      - id: AC-091
        text: '100% coverage on touched files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*enemy-navigation'
    parallelizable: false
    dependencies:
      - 'P5S2b-fitness-cadence'
    next_slice: null
```

**Slice details (P5S2a — telemetry tracking):**
- Add `EpisodeTelemetry` interface to `host/game/types.ts`:
  - `damageDealt: number`, `shotsFired: number`, `shotsHit: number`, `aimMissRate: number`
- Add telemetry accumulation to `combat.ts`:
  - `damageDealt`: accumulate in `applyEnemyDamage`.
  - `shotsFired` / `shotsHit`: track in `fireBolt` — increment `shotsFired` on each bolt, `shotsHit` on enemy hit.
  - `aimMissRate = (shotsFired - shotsHit) / max(1, shotsFired)`.
- Add an `EpisodeTelemetry` field to `GameState` (or pass as a parallel object to `runEpisode`).

**Slice details (P5S2b — fitness extraction + cadence):**
- Build `extractCombatQualitySignal(gameState, telemetry): CombatQualitySignal` in `harness/fitness.ts`:
  - `survivalTicks = episodeTimeMs / NEATENSTEIN_FIXED_TIMESTEP_MS`
  - `damageDealt = telemetry.damageDealt`
  - `kills = gameState.kills`
  - `damageTaken = (deaths * maxHealth) + (maxHealth - player.health)`
  - `aimMissRate = telemetry.aimMissRate`
- Cadence: update `harness/constants.ts`, `harness/main-runner.ts`, and `harness/barrier.ts`:
  - `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS = 5000` (5s simulated time).
  - `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = Math.floor(NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312 — a NEW constant for the fitness evaluation path. The existing `NEATENSTEIN_MAX_EPISODE_TICKS = 240` is NOT changed — it is used by `enemy-runner.ts` for the enemy barrier evaluation path and stays at 240. This avoids the determinism contradiction of changing a shared constant. The local `NEATENSTEIN_MAX_EPISODE_TICKS = 240` in `main-runner.ts:54` and `barrier.ts:95` are replaced with `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` (imported from `constants.ts`) so the fitness path uses 312 ticks while the enemy barrier path stays at 240.
  - `NEATENSTEIN_MAIN_VARIANT_COUNT` from 8 to 4 — remove the hard-coded constant in `main-runner.ts:46` AND `barrier.ts:87`, import from `constants.ts`. Both removals and the export addition happen in this slice.
  - `NEATENSTEIN_EVAL_CHUNK_TICKS = 32` (configurable constant in `harness/constants.ts`).
  - `NEATENSTEIN_FIXED_TIMESTEP_MS` re-exported from `host/game/constants.ts` (not redefined).
  - At full-speed headless: 4 variants × ~300ms wall = ~1.2s + evolution overhead ≈ 2-3s total → ~20+ gen/min.
  - If 4 variants produce too noisy a fitness signal, add a configurable knob (`NEATENSTEIN_FITNESS_VARIANT_COUNT`).
  - **Non-blocking**: evaluation uses `setTimeout(0)` chunking (NOT `requestIdleCallback` — that is a Window API, not available in Web Workers). Yield after every `NEATENSTEIN_EVAL_CHUNK_TICKS = 32` ticks (configurable constant in `harness/constants.ts`) to allow `onmessage` to fire between chunks.
  - **barrier.ts constant removal**: Remove local `NEATENSTEIN_MAIN_VARIANT_COUNT = 8` (line 87) and import from `constants.ts` (value 4). This centralizes the variant count and eliminates drift between barrier and main-runner paths.

**Slice details (P5S2c — BFS buffer pooling):**
- `buildEnemyDistanceMap` allocates a fresh `Int32Array(120*120)` (57KB) per tick. For 312 ticks × 4 variants = ~71MB allocation/generation in headless mode; ~3.4MB/s at 60fps in the live worker path. Pool the buffer — allocate once per episode (headless) and reuse a worker-scoped buffer (live path). This change touches `scripts/enemy-navigation.ts` to accept an optional pre-allocated `distances?: Int32Array` parameter. Callers pass a reused buffer; the function fills it in-place when provided, or allocates fresh when undefined (backward-compatible).

---

### Phase 6 — Integration testing & browser validation [PLANNED]

**Phase objective:** End-to-end validation that auto mode works in the browser: enemies evolve, player is NEAT-controlled, generation advances, fitness reflects real gameplay.

```yaml
phase: 6
title: 'Integration testing & browser validation'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_phase: 'Archive'
skills:
  - green-validation-gates
  - browser-testing-harness
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
  - 'npm run build'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-090
    text: 'Full Neatenstein test suite passes (no regressions in human mode)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
  - id: AC-091
    text: 'Build and lint pass'
    validation: 'npm run build && npm run lint'
  - id: AC-092
    text: 'Browser smoke test: auto mode loads, enemies use MLP, player is NEAT-driven'
    validation: 'browser-harness-specialist visible-window validation'
  - id: AC-093
    text: 'Generation counter advances in browser HUD during auto mode'
    validation: 'browser-harness-specialist visible-window validation'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Full test suite + build + lint validation'
  - 'Step 02 — Browser smoke test (visible window)'
```

#### Step 01: Full test suite + build + lint validation [PLANNED]

```yaml
phase: 6
step: 1
title: 'Full test suite + build + lint validation'
status: '[PLANNED]'
goal: 'green-testing'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: 'Step 02 — Browser smoke test (visible window)'
skills:
  - green-validation-gates
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
  - 'npm run build'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-094
    text: 'All Neatenstein tests pass with zero failures'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein'
  - id: AC-095
    text: 'Build succeeds with no errors'
    validation: 'npm run build'
  - id: AC-096
    text: 'Lint passes with no errors'
    validation: 'npm run lint'
```

#### Step 02: Browser smoke test (visible window) [PLANNED]

```yaml
phase: 6
step: 2
title: 'Browser smoke test (visible window)'
status: '[PLANNED]'
goal: 'green-testing'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: fresh-session
source_of_truth: plans/neatenstein-auto-neat-mode.plans.md
copy_paste: true
next_step: null
skills:
  - browser-testing-harness
  - chrome-devtools-mcp
validation:
  - 'browser-harness-specialist visible-window smoke test'
acceptance_criteria:
  - id: AC-097
    text: 'Auto mode loads in browser, no console errors'
    validation: 'browser-harness-specialist visible-window validation'
  - id: AC-098
    text: 'Enemies navigate using MLP (not pure BFS) — visible movement pattern change'
    validation: 'browser-harness-specialist visible-window validation'
  - id: AC-099
    text: 'Player moves and fires autonomously in auto mode'
    validation: 'browser-harness-specialist visible-window validation'
  - id: AC-100
    text: 'Generation counter increments in HUD after wave-clear'
    validation: 'browser-harness-specialist visible-window validation'
  - id: AC-101
    text: 'Human mode remains fully functional (toggle back, keyboard/mouse work)'
    validation: 'browser-harness-specialist visible-window validation'
```

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.
Workstream: Neatenstein Auto / NEAT Mode Wiring.
Active plan: plans/neatenstein-auto-neat-mode.plans.md.
Research artifact: plans/neatenstein-auto-neat-mode.research.md.
Current boundary: Phase 0 [DONE] — all 8 failing gun sprite/renderer tests fixed. Phase 1 [WIP] planning review.
Next: Resume Phase 1 planning review and dispatch implementation agents for Phase 2 (generation sync + enemy MLP activation).
Validations: npx jest --testPathPattern=neatenstein, npm run build, npm run lint.
Caution: Do not run full test suite in a single shell invocation — use targeted patterns.
```

## Latest validation evidence

- Research complete: 5 specialists investigated 6 gaps, findings recorded in research artifact and plan.
- Plan authored: Phases 1-6 with step packets, slices, acceptance criteria, and traceability.
- First review cycle: 5 reviewers dispatched (boundary-mapper, api-contract-reviewer, determinism-reviewer, performance-reviewer, implementation-pattern-scout). All 5 returned FAIL/REQUEST_CHANGES with 13 blockers (1 critical, 7 high, 5 medium).
- Plan revised to address all 13 blockers:
  - B1 (CRITICAL): Added `new Neat(12, 5, fitnessFn)` direct NEAT population; documented embryo-to-NeatGenome converter as follow-up; `createMainSnapshot` placeholder to be replaced with real Neat population in Phase 4.
  - B2-B3: Added explicit scheduling model — full-speed headless at `NEATENSTEIN_FIXED_TIMESTEP_MS=16` (312 ticks in ~300ms wall, not 5s wall); `requestIdleCallback` chunking for non-blocking; documented async serialization guard.
  - B4: Added `NEATENSTEIN_MAIN_VARIANT_COUNT` update from 8→4 in `harness/constants.ts` to P5S2 files_to_change.
  - B5: Added humanMode string→boolean conversion: `humanMode === 'auto'` before `runArmsRaceGeneration` call.
  - B6: Changed `updateEnemyController` weights parameter to optional (`weights?: Float32Array | undefined`); backward-compatible with ~150 existing call sites.
  - B7: Routed controller inputs through `gameTick` (existing single input-handling path), NOT `updateEpisode`. Eliminated dual-path conflict.
  - B8: Added Determinism Contract section with seed propagation (`gameState.seed` → `seed + generation`), fixed timestep clamping in worker, episode reproducibility, async serialization.
  - B9: Added missing files to P5S1 (`host/game/tick.ts`, `scripts/enemy-controller.ts`) and P5S2 (`host/game/types.ts`, `harness/constants.ts`).
  - B10: Removed non-existent `ControllerInput` type reference; using `GameTickInputSnapshot` (existing type) throughout.
  - B11: Placed `extractSensors` in `scripts/enemy-navigation.ts` (alongside existing `buildVisionVector`); added to P4S2 files_to_change.
  - B12: Named determinism rung (Level 2 — Ordered deterministic); added same-seed replay ACs (AC-013a, AC-013b).
  - B13: Eliminated `updateEpisode` dual-path; `gameTick` is the single input-handling path for both human and auto modes.
- Awaiting: Second review cycle (5 reviewers) on revised plan.
- Second review cycle: 5 reviewers dispatched. All 5 returned REQUEST_CHANGES or FAIL with more specific issues:
  - boundary-mapper: FAIL — P4S2 over-lists tick.ts. FIXED: removed tick.ts from P4S2.
  - api-contract-reviewer: REQUEST_CHANGES — runArmsRaceGeneration sync signature (~25 callers); createMainSnapshot needs MainVariant type update. FIXED: documented sync signature preservation; added AC-060b for MainVariant network field; added harness/types.ts to P4S1 files_to_change.
  - determinism-reviewer: REQUEST_CHANGES — createSeedPack doesn't accept seed param; numeric seed mixing collision-prone. FIXED: changed to string-based seed mixing `${seed}:${generation}`; added AC-081a for createSeedPack seed param; added harness/seed-pack.ts to P5S1 files_to_change.
  - performance-reviewer: REQUEST_CHANGES — requestIdleCallback not in Worker; NEATENSTEIN_MAIN_VARIANT_COUNT hard-coded in main-runner.ts:46; BFS allocation churn. FIXED: replaced requestIdleCallback with setTimeout(0) chunking; added main-runner.ts to P5S2b for local constant removal; documented BFS buffer pooling (allocate once per episode, reuse).
  - implementation-pattern-scout: FAIL — Neat.evaluate()/evolve() async but harness sync; MainVariant type mismatch; createSeedPack API; default Neat popsize is 50. FIXED: documented Neat lifecycle (async evaluate/evolve, worker awaits, sync function preserved); added MainVariant network field; documented createSeedPack API change; documented { popsize: 4, seed } in Neat constructor.
- Second revision pass complete. All second-cycle blockers addressed.
- Third review cycle: 5 reviewers dispatched. All 5 returned REQUEST_CHANGES with deeper issues:
  - boundary-mapper: REQUEST_CHANGES — P2S1 over-lists state.ts/waves.ts; P2S2 over-lists enemy-mlp.ts; P3S2 over-lists arms-race.ts; P4S1 4-file exception incomplete (renderer-bridge.ts no edits); P4S2 path typo; P5S2b missing enemy-navigation.ts for BFS pooling.
  - api-contract-reviewer: REQUEST_CHANGES — createSeedPack seed param specified as required (breaks barrier.ts); AC-060a says network: Network (required) vs AC-060b says network?: Network (optional); types.ts needs Network import.
  - determinism-reviewer: REQUEST_CHANGES — string-based seed incompatible with Neat's numeric normalizeSeed (falls back to Date.now()); missing main-agent same-seed AC.
  - performance-reviewer: REQUEST_CHANGES — sync runArmsRaceGeneration incompatible with setTimeout(0) chunking (sync loop can't yield); BFS pooling only for headless not live worker.
  - implementation-pattern-scout: REQUEST_CHANGES — async lifecycle contradiction (Neat.evaluate() is async but called inside sync runArmsRaceGeneration); Neat.getFittest() returns Network directly (not .network); gameTick does NOT call updateEnemyController internally.
- Third revision pass — addressed all third-cycle blockers:
  - Hoisted Neat evaluation OUT of runArmsRaceGeneration into worker: worker creates Neat pop, calls await evaluate()/evolve(), extracts champion via getFittest() (returns Network directly), passes to still-sync runArmsRaceGeneration as championNetwork? param. Resolves async/sync contradiction.
  - Moved createMainSnapshot and MainVariant type updates from P4S1 to P3S2 (3 files: display.worker.ts, arms-race.ts, types.ts). P4S1 simplified to 1 file (display.worker.ts only).
  - Fixed Neat.getFittest() → returns Network directly (not .network).
  - Fixed gameTick claim: does NOT call updateEnemyController internally; worker keeps explicit updateEnemyController pass.
  - Changed seed mixing from string-based to numeric hash: hashSeed(seed, generation) = ((seed * 100003 + generation) * 100003) >>> 0. Neat-compatible (numeric).
  - Made createSeedPack seed optional (seed?: number). barrier.ts unchanged (omitted seed = existing behavior).
  - Harmonized AC-060a to network?: Network (optional, matching AC-060b).
  - Removed over-listed files: P2S1 (removed state.ts, waves.ts → 1 file), P2S2 (removed enemy-mlp.ts → 2 files), P4S1 (removed renderer-bridge.ts, arms-race.ts, types.ts → 1 file).
  - Fixed P4S2 path typo (trailing quote).
  - Added enemy-navigation.ts to P5S2b for BFS buffer pooling (4th file with exception).
  - Extended BFS pooling to live worker path (not just headless).
  - Documented cleanup rule for Neat populations (replace each generation, one in flight).
  - Added AC-013c for main-agent same-seed determinism.
  - Removed setNetwork message type (worker owns Neat pop, no external network injection needed).
- Awaiting: Fourth review cycle (5 reviewers) on thrice-revised plan.
- Fourth review cycle: 2/5 APPROVED (api-contract-reviewer, implementation-pattern-scout), 3/5 REQUEST_CHANGES:
  - boundary-mapper: REQUEST_CHANGES — P2S1/P3S1 double-increment; hashSeed defined in P5S1 but used in P3S2.
  - determinism-reviewer: REQUEST_CHANGES — barrier.ts drops root seed; no launch guard for overlapping generations.
  - performance-reviewer: REQUEST_CHANGES — runMainGeneration not bypassed when championNetwork provided (duplicate evaluation); episode tick count inconsistency (312 vs 240); chunk size not configurable.
  - api-contract-reviewer: APPROVE ✅ (2 low-severity observations: Network import path, MainVariant JSDoc)
  - implementation-pattern-scout: APPROVE ✅ (all 3 blockers verified fixed against source)
- Fourth revision pass — addressed all fourth-cycle blockers:
  - Generation increment: P2S1 no longer increments generation (owned by advanceWave in P3S1). Updated ACs.
  - hashSeed: defined in harness/arms-race.ts (P3S2's files_to_change), not in seed-pack.ts. P5S1 imports from arms-race.ts.
  - barrier.ts: added to P5S1 files_to_change (4th file with exception). One-line change to pass options.seed.
  - Launch guard: documented in P3S2 — `if (pendingGeneration !== null) return;` before starting evaluation.
  - runMainGeneration bypass: documented — when championNetwork is provided, runMainGeneration is skipped.
  - Episode tick count: NEATENSTEIN_MAX_EPISODE_TICKS derived from NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / NEATENSTEIN_FIXED_TIMESTEP_MS.
  - Chunk size: NEATENSTEIN_EVAL_CHUNK_TICKS = 32 as configurable constant in harness/constants.ts.
  - gameTick parameter order: fixed to (state, snapshot, collisionMap?, dtMs?).
  - Network import: documented as `import type { Network } from 'neataptic'` (public entry point).
  - Concurrent wave-clear: documented drop behavior (no queueing).
- Awaiting: Fifth review cycle (5 reviewers) on four-times-revised plan.
- Fifth review cycle: 3/5 APPROVED (api-contract-reviewer ✅, implementation-pattern-scout ✅, — both consecutive), 3/5 REQUEST_CHANGES:
  - boundary-mapper: REQUEST_CHANGES — (1) Circular module dependency: hashSeed in arms-race.ts creates cycle (arms-race.ts → main-runner.ts, main-runner.ts → arms-race.ts). (2) host/waves.ts over-listed in P3S1 — no concrete edit.
  - determinism-reviewer: REQUEST_CHANGES — (1) barrier.ts local NEATENSTEIN_MAIN_VARIANT_COUNT=8 not reconciled. (2) P5S1 doesn't explicitly require main-runner.ts to pass seed. (3) pendingGeneration invariant inconsistent with generation+1 return. Plus: setTimeout(0) chunking scope unclear; NEATENSTEIN_FIXED_TIMESTEP_MS dual definition; createSeedPack per-variant formula unspecified; NEATENSTEIN_ENEMY_EVALUATION_DURATION_MS role unclear.
  - performance-reviewer: REQUEST_CHANGES — (1) NEATENSTEIN_MAIN_VARIANT_COUNT must be centralized. (2) NEATENSTEIN_MAX_EPISODE_TICKS derivation. (3) BFS buffer pooling. (4) barrier.ts constant drift.
  - api-contract-reviewer: APPROVE ✅ (2nd consecutive — all additions additive, backward-compatible)
  - implementation-pattern-scout: APPROVE ✅ (2nd consecutive — all 6 additions verified against source)
- Fifth revision pass — addressed all fifth-cycle blockers:
  - Circular import: Created `harness/hash-seed.ts` as a new shared module. hashSeed defined there, imported by both arms-race.ts (P3S2) and main-runner.ts (P5S1). Breaks the arms-race.ts ↔ main-runner.ts cycle. P3S2 now has 4 files with documented exception.
  - host/waves.ts removed from P3S1 files_to_change — advanceWave is already implemented and exported; the concrete edit is in display.worker.ts only.
  - barrier.ts NEATENSTEIN_MAIN_VARIANT_COUNT=8: P5S1 now documents two barrier.ts changes: (1) seed propagation, (2) remove local constant and import from constants.ts (value 4).
  - main-runner.ts seed propagation: P5S1 slice details now explicitly document that main-runner.ts line 133 passes rootSeed to createSeedPack.
  - pendingGeneration invariant: Fixed to account for generation+1 return. Guard stores `pendingGeneration = launchedGeneration`; completion check is `result.generation === pendingGeneration + 1`.
  - setTimeout(0) chunking: Clarified — applies to episode tick loop inside fitness function (main-runner.ts), NOT to Neat.evaluate()/evolve() internals.
  - NEATENSTEIN_FIXED_TIMESTEP_MS: harness/constants.ts must re-export from host/game/constants.ts (not redefine).
  - createSeedPack per-variant formula: Explicitly stated `seeds[i] = hashSeed(seed, generation, i)`.
  - NEATENSTEIN_ENEMY_EVALUATION_DURATION_MS: Clarified as governing the enemy barrier path, NOT changed by this plan (different from NEATENSTEIN_FITNESS_EPISODE_DURATION_MS which governs fitness evaluation).
  - AC-089 updated to include barrier.ts constant removal and NEATENSTEIN_FIXED_TIMESTEP_MS re-export.
- Awaiting: Sixth review cycle (3 remaining reviewers: boundary-mapper, determinism-reviewer, performance-reviewer).
- Sixth review cycle: 3/5 APPROVED (api-contract-reviewer ✅, implementation-pattern-scout ✅ — both 3rd consecutive), 3/5 REQUEST_CHANGES:
  - boundary-mapper: REQUEST_CHANGES — P5S1 barrier.ts imports NEATENSTEIN_MAIN_VARIANT_COUNT from constants.ts, but the export is added in P5S2b (missing-export compile error).
  - determinism-reviewer: REQUEST_CHANGES — Same ordering issue; P2S1 timestep import path not explicit (should be host/game/constants.ts).
  - performance-reviewer: REQUEST_CHANGES — (1) P5S1 uses `new Neat(...)` per-variant episode (should be single Network). (2) P3S2 championNetwork bypass lacks quality-signal conduit.
- Sixth revision pass — addressed all sixth-cycle blockers:
  - NEATENSTEIN_MAIN_VARIANT_COUNT ordering: Moved barrier.ts constant centralization to P5S2b (which already touches constants.ts). P5S1 barrier.ts only gets seed propagation (one-line change). P5S2b adds the export to constants.ts AND removes local constants in both main-runner.ts and barrier.ts.
  - P2S1 timestep import: Explicitly documented as importing from `host/game/constants.ts` (authoritative source), not harness/constants.ts.
  - P5S1 per-variant network: Replaced `new Neat(12, 5, fitnessFn, { popsize: 4, ... })` with `Network.fromJSON(variant.genome)` — single executable network per variant, NOT a full Neat population. The Neat population is created ONCE in the worker (P3S2).
  - P3S2 quality signal: Added `championQuality?: CombatQualitySignal` parameter to runArmsRaceGeneration. Worker extracts champion's fitness from `neatPop.getFittest().score` and constructs CombatQualitySignal from episode telemetry. When championNetwork is undefined, runMainGeneration produces quality internally as before.
  - AC-089 updated to reflect P5S2b adding the export to constants.ts.
- Awaiting: Seventh review cycle (3 remaining reviewers).
- Seventh review cycle: 4/5 APPROVED (api-contract-reviewer ✅, implementation-pattern-scout ✅, performance-reviewer ✅ — first approval), 2/5 REQUEST_CHANGES:
  - boundary-mapper: REQUEST_CHANGES — P5S2b files_to_change omits barrier.ts but AC-089 requires editing it. Adding it would make 5 files.
  - determinism-reviewer: REQUEST_CHANGES — Same barrier.ts issue. Also: NEATENSTEIN_MAX_EPISODE_TICKS change affects enemy-runner.ts (shared constant). P2S1 timestep import path only in contract §7, not in slice details.
- Seventh revision pass — addressed all seventh-cycle blockers:
  - P5S2b split: Moved BFS buffer pooling (enemy-navigation.ts) to new P5S2c slice (1 file, 2h). P5S2b now has 4 files (fitness.ts, constants.ts, main-runner.ts, barrier.ts) with documented exception. barrier.ts added for constant removal + import. P5S2c handles BFS pooling independently.
  - NEATENSTEIN_MAX_EPISODE_TICKS: Introduced NEW `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = 312` for the fitness evaluation path. Existing `NEATENSTEIN_MAX_EPISODE_TICKS = 240` is NOT changed — it stays for enemy-runner.ts. Eliminates the determinism contradiction of changing a shared constant.
  - P2S1 timestep import: Added explicit import path (`host/game/constants.ts`) to P2S1 slice details.
  - AC-089 updated to reflect NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS (new) instead of modifying NEATENSTEIN_MAX_EPISODE_TICKS.
- Awaiting: Eighth review cycle (2 remaining reviewers: boundary-mapper, determinism-reviewer).
- Eighth review cycle: 5/5 APPROVED (boundary-mapper ✅ — first approval), 1/5 REQUEST_CHANGES:
  - boundary-mapper: APPROVE ✅ — P5S2b/P5S2c split verified, 4-file exception justified, linkage correct.
  - determinism-reviewer: REQUEST_CHANGES — P5S1 line 842 stale text still says "NEATENSTEIN_MAX_EPISODE_TICKS must be updated to 312" (contradicts cycle-7 fix). Also: local 240s in main-runner.ts:54 and barrier.ts:95 should be replaced with NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS.
- Eighth revision pass:
  - Fixed P5S1 line 842: Replaced stale text with correct guidance — fitness path uses NEW NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = 312; existing NEATENSTEIN_MAX_EPISODE_TICKS = 240 NOT changed.
  - Added explicit instructions in P5S2b slice details: local NEATENSTEIN_MAX_EPISODE_TICKS = 240 in main-runner.ts:54 and barrier.ts:95 replaced with NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS (imported from constants.ts).
- Awaiting: Ninth review cycle (1 remaining reviewer: determinism-reviewer).
- Ninth review cycle: 5/5 APPROVED ✅✅✅✅✅
  - determinism-reviewer: APPROVE ✅ — "Both cycle-8 determinism fixes are correctly reflected in the plan. P5S1 line 842 and P5S2b lines 962/AC-089 unambiguously split fitness-specific constants from the shared enemy-runner constant and document the local-constant replacements. The Level 2 ordered-determinism contract remains consistent."

## ALL REVIEWERS APPROVED — PLAN READY FOR IMPLEMENTATION

All 5 reviewers have approved the plan after 8 revision passes:
1. **api-contract-reviewer**: APPROVE ✅ (stable since cycle 4)
2. **implementation-pattern-scout**: APPROVE ✅ (stable since cycle 4)
3. **performance-reviewer**: APPROVE ✅ (approved cycle 7)
4. **boundary-mapper**: APPROVE ✅ (approved cycle 8)
5. **determinism-reviewer**: APPROVE ✅ (approved cycle 9)

Plan status: [WIP], Phase 1 Step 01 in progress. Plan remains OPEN per user request — follow-up phases (fix failing tests, restore 100% coverage) will be dispatched separately after user confirmation.