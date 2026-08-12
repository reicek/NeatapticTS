**Status:** [DONE]
**Plan ID:** NEATENSTEIN_AUTO_NEAT_MODE
**Created:** 2026-08-09
**Source of truth:** `plans/neatenstein-auto-neat-mode.plans.md`
**Research artifact:** `plans/neatenstein-auto-neat-mode.research.md`

**Latest research update (2026-08-14):** Fallback AI no-enemy exploration
behavior analyzed. The current `buildFallbackAutoTickInput()` spins almost in
place (radius ≈ 0.37 cells) because it combines constant forward movement
with a fixed `NEATENSTEIN_FALLBACK_TURN_RATE`. The recommended replacement
is direction-persistence with wall-bounce, optionally followed by a
deterministic seeded Levy walk. See the research artifact addendum
"Fallback AI no-enemy exploration behavior" for full evidence, options,
and risks.

- `model: glm-5.2:cloud` for all dispatches under this plan. Every dispatch under this plan overrides the frontmatter `model` with `glm-5.2:cloud`. Do NOT use kimi-k2.7-code:cloud or any other model.
- Pragmatic mode: broad slices (one per gap area), bypass legacy ceremony (skip plan-verification green-light cycle, skip per-AC gate calls, skip fix-packet YAML ceremony). Ship working software.
- Do not archive or supersede existing Neatenstein plans; this workstream is additive.
- Remove legacy noise: delete the RNG-based `runEpisode` stub in `main-runner.ts` when replaced by real gameplay (no dual-path code, no backward-compatibility wrappers).
- Human-played mode must remain fully functional — the `humanMode` selector is the switch point.
  Wire the Neatenstein evolution harness into the live game loop so that `humanMode: 'auto'` activates NEAT-driven gameplay:

1. **Worker reads `humanMode`** and branches to a neural-network controller instead of forwarding human input (Gap 1).
2. **`advanceWave()` is called from the live game** on wave-clear, evolving the enemy MLP population (Gap 2).
3. **Enemy weights are populated** from the MLP population via `createMlpEnemyPopulation()` / `activateMlp()` instead of always being `undefined` (Gap 3).
4. **The harness episode runs real gameplay** (or a headless fast-forward) instead of RNG-based fake metrics (Gap 4).
5. **A NEAT network controller produces `GameTickInputSnapshot`** from game-state sensors via `Network.activate()` (Gap 5).
6. **`GameState.generation` is incremented** on generation advance and synced from the harness counter (Gap 6).
   **Rung:** Level 2 — Ordered deterministic (same seed + same genome + same fixed timestep → same trajectory).
   **Rung:** Level 2 — Ordered deterministic (same seed + same genome + same fixed timestep → same trajectory).

7. **Seed propagation**: `gameState.seed` (set at game init, numeric) is the root seed. Evolution harness derives its seed as a deterministic numeric hash: `hashSeed(seed, generation) = ((seed * 100003 + generation) * 100003) >>> 0` — produces a 32-bit unsigned integer. Per-variant: `hashSeed(seed, generation, variantId) = ((seed * 100003 + generation) * 100003 + variantId) >>> 0`. This is NOT string-based mixing — Neat's `normalizeSeed` in `src/neat/rng/core/rng.utils.ts` only accepts finite numbers; non-numeric strings fall back to `Date.now()` which breaks determinism. `createSeedPack` must be updated to accept an optional `seed?: number` parameter: `createSeedPack({ generation, variantCount, seed })` — when `seed` is omitted, existing callers get the current behavior (no change needed). When provided, seeds are derived via the numeric hash. `harness/seed-pack.ts`, `harness/main-runner.ts` (caller at line 133), and `harness/barrier.ts` (caller at line 174) are all updated to pass the root seed.
8. **Fixed timestep in worker**: `display.worker.ts:1222-1223` currently uses host rAF `deltaMs` as simulation timestep. **Must clamp to `NEATENSTEIN_FIXED_TIMESTEP_MS = 16`** for deterministic simulation. The rAF delta only gates _when_ a tick runs, not _how much_ time advances. The Level 2 determinism claim applies to headless episode evaluation; the live worker path uses the rAF cadence for dispatch but clamps the timestep value.
9. **Episode reproducibility**: `runEpisode` in the harness uses `NEATENSTEIN_FIXED_TIMESTEP_MS` for all ticks. Same seed + same genome + same enemy snapshot → same fitness score.
10. **Async serialization**: Only one generation evaluation is in flight at a time. The worker's hoisted Neat evaluation (async `await neatPop.evaluate()` + `await neatPop.evolve()`) is guarded by a `pendingGeneration` number that stores the launched generation. Before starting a new evaluation, check `if (pendingGeneration !== null) return;` — skip if already evaluating. On completion, `runArmsRaceGeneration` returns `generation + 1` (the next generation number). The guard stores `pendingGeneration = launchedGeneration`; on completion, verify `result.generation === pendingGeneration + 1` before applying (prevents out-of-order application). Then clear `pendingGeneration = null`. If wave-clears arrive faster than the ~2-3s evaluation, the new request is dropped (no queueing). The synchronous `runArmsRaceGeneration` function is called only after the async evaluation completes and the champion Network is extracted. When `championNetwork` is provided, `runMainGeneration` is skipped (no duplicate evaluation).
11. **Seed propagation in all callers**: Both `main-runner.ts` (line 133) and `barrier.ts` (line 174) must pass the root seed to `createSeedPack`: `createSeedPack({ generation, variantCount, seed: rootSeed })`. In `main-runner.ts`, the root seed comes from the caller (the worker passes `gameState.seed`); the call site at line 133 changes from `createSeedPack({ generation })` to `createSeedPack({ generation, seed: rootSeed })`. In `barrier.ts`, `options.seed` is already available (declared in `GenBarrierOptions`); the one-line change is `createSeedPack({ generation: options.generation, seed: options.seed })`. The optional `seed?: number` API makes both changes backward-compatible. The per-variant seed derivation inside `createSeedPack` uses the numeric hash: `seeds[i] = hashSeed(seed, generation, i)` for each variant index `i`.
12. **humanMode type mapping**: Worker receives `humanMode: 'auto' | 'human'` (string) from `simState`. Must convert to boolean for `runArmsRaceGeneration({ humanMode: humanMode === 'auto' })` before the call.
13. **Versioned parameters**: `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS` and `NEATENSTEIN_MAIN_VARIANT_COUNT` are contract constants. Note: `NEATENSTEIN_MAIN_VARIANT_COUNT` is currently hard-coded as `8` in BOTH `main-runner.ts:46` AND `barrier.ts:87` — both must be removed and imported from `constants.ts` (value 4). This centralization happens in P5S2b (which already touches `constants.ts` and can add the export in the same slice, plus removes the local in barrier.ts in the same slice). P5S1 does NOT change the variant count — it only adds seed propagation to barrier.ts. `NEATENSTEIN_FITNESS_EPISODE_DURATION_MS` (= 5000) derives a NEW fitness-specific tick count: `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = Math.floor(NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312. The existing `NEATENSTEIN_MAX_EPISODE_TICKS = 240` is NOT changed — it is used by `enemy-runner.ts` for the enemy barrier evaluation path. Introducing a separate `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` avoids the determinism contradiction of changing a shared constant. Note: `NEATENSTEIN_ENEMY_EVALUATION_DURATION_MS = 10000` (existing) is used by the enemy barrier evaluation path and is NOT changed by this plan. `NEATENSTEIN_FIXED_TIMESTEP_MS = 16` already exists in `host/game/constants.ts` (the authoritative source); `harness/constants.ts` must re-export it (not redefine) to avoid dual-definition drift. P2S1 imports `NEATENSTEIN_FIXED_TIMESTEP_MS` directly from `host/game/constants.ts` (the authoritative source), not from `harness/constants.ts` (which doesn't re-export it until P5S2b).

- No changes to the core NeatapticTS library (`src/`) — only consume `Network.activate()`, `new Neat()`, and NEAT APIs from existing exports. The `NgeMainAgentEmbryo` → `NeatGenome` → `Network` converter is a documented follow-up, NOT part of this plan.
- No new enemy MLP topology changes — the existing 6→6→4→4 architecture is reused as-is.
- No changes to maze generation, wall collision, or rendering pipeline.
- No sound asset changes.
- No WebGPU tier changes.

1. **Enemy weight injection strategy**: Champion snapshot weights shared by all enemies (simplest), vs. per-variant distribution (`enemy.index % population.size`). Decision: start with champion snapshot (all enemies share the current champion's weights); per-variant can be a follow-up.
2. **Player network topology**: Use `buildMainAgentEmbryo()` from the NGE pipeline to construct the player's network, or build a simpler feed-forward network via `new Neat(inputCount, outputCount, fitnessFn)`. Decision: use the NGE embryo for the main-agent champion (consistent with the harness), but the initial auto-mode controller can use a simple NEAT network while the embryo pipeline is wired.
3. **Cadence compromise**: Real 20s episodes × 8 variants = 160s exceeds the 30s/generation budget. Decision: reduce episode duration for fitness evaluation to 5s (headless fast-forward), reduce variant count to 4, or run evaluation async in a separate worker. The initial implementation uses headless fast-forward at reduced duration.
4. **Sensor design**: The player controller needs game-state observations. Decision: start with a minimal sensor set (~12 inputs: player health, ammo, angle, position, nearest enemy bearing, nearest enemy distance, nearest enemy health, 4 wall raycasts) and expand if evolution stalls.
5. **Pitch control**: `GameTickInputSnapshot` only carries `lookDelta` (yaw), not pitch. Decision: keep yaw-only auto-aim for the initial implementation; pitch extension is a follow-up.
   Five research specialists investigated the six gaps. Key findings:

- `humanMode: 'auto'` is posted in `simState` to the worker at `display.worker.ts:1204-1205` but NEVER read.
- Worker builds `tickInput` from human input queue only (line 1212-1217).
- Branch insertion point: `display.worker.ts:1204-1217`, between `latestState` assignment and `tickInput` construction.
- `GameTickInputSnapshot` type (`host/game/tick.ts:73-85`): `{move: Vector2, lookDelta: number, fire: boolean, dash: boolean}` — network-output-friendly.
- Medium complexity; need to add NEAT network loading + observation extractor + output mapper to worker.
- `advanceWave()` (`host/waves.ts:96`) — evolves MLP population, advances generation, clears arena, respawns. Returns `{state, snapshot, spawnedCount}`.
- `runArmsRaceGeneration()` (`harness/arms-race.ts:95`) — full arms race generation. Never called from live game.
- `spawnWaveTick()` (`host/game/waves.ts:161`) — live spawner, no evolution logic.
- Trigger point: wave-clear detection in worker after `gameTick`.
- `advanceWave()` already returns evolved `snapshot.weights` — just needs injection into enemy controllers.
- Enemy-side wiring is low-medium; main-agent side is higher (needs real episodes).
- `createEnemyControllerState()` (`scripts/enemy-controller.ts:310`) initializes `weights: undefined`.
- `updateControlledEnemy()` (line 397): `weights = isRespawn ? undefined : previousOrDefault.weights` — always undefined.
- MLP re-ranking path (line 645-711) is fully implemented but dead code.
- MLP topology: 6→6→4→4, 90 params, tanh activation. Input: 6-element BFS vision vector. Output: move, strafe, turn, fire.
- **Low-to-moderate complexity** — the plumbing exists; just needs weight injection.
- Need to: instantiate population in worker, pass weights to `updateEnemyController`, fix `isRespawn` reset.
- Harness `runEpisode` (`main-runner.ts:309-332`) generates all metrics via `seedrandom` RNG — no gameplay.
- Host `runEpisode` (`episode.ts:373-404`) plays real game with "deterministic damage bot" — no controller inputs.
- Fitness: `baseScore = survivalTicks*1 + damageDealt*2 + kills*5 - damageTaken*1 - aimMissRate*1 + complexityBonus*0.1`.
- `buildMainAgentEmbryo()` produces a topology descriptor, not an executable network.
- Cadence: 2 gen/min minimum → 30s/generation; real 20s × 8 variants = 160s — VIOLATES.
- Medium-high complexity; needs genome→controller materialization, `updateEpisode` input plumbing, telemetry extraction, cadence compromise.
- `GameTickInputSnapshot`: `{move: Vector2, lookDelta, fire, dash}` — only 4 fields.
- `Network.activate(input)` available from `src/architecture/network/network.ts:1100-1166`.
- Need sensor extraction from `GameState` + network activation → `GameTickInputSnapshot`.
- Cleanest insertion: add auto-mode controller in worker that reads `gameState`, calls `network.activate(sensors)`, writes `pendingTickInput` directly.
- ~200-400 LOC, medium complexity.
- `GameState.generation` initialized to 1 (`state.ts:107`), never incremented.
- Harness has own generation counters, never synced.
- Small complexity (~30-80 LOC).
- Best approach: sync from harness counter to `GameState.generation` on generation advance.
  | Gap | Deliverable                    | Phase   | Primary files                                                                                                                                                                                   |
  | --- | ------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 6   | Generation counter sync        | Phase 2 | `host/game/state.ts`, `host/game/tick.ts`, `worker/display.worker.ts`, `host/game/waves.ts`                                                                                                     |
  | 3   | Enemy MLP weight injection     | Phase 2 | `scripts/enemy-controller.ts`, `harness/enemy-mlp.ts`, `worker/display.worker.ts`                                                                                                               |
  | 2   | Arms-race wiring (advanceWave) | Phase 3 | `host/waves.ts`, `worker/display.worker.ts`, `host/game/waves.ts`                                                                                                                               |
  | 1   | Worker humanMode branching     | Phase 4 | `worker/display.worker.ts`, `browser-entry.ts`, `host/renderer-bridge.ts`, `harness/arms-race.ts`                                                                                               |
  | 5   | Player NEAT controller         | Phase 4 | `worker/display.worker.ts`, `host/game/tick.ts`, `scripts/enemy-navigation.ts`, `src/architecture/network/network.ts`                                                                           |
  | 4   | Real episode fitness           | Phase 5 | `harness/main-runner.ts`, `host/game/episode.ts`, `host/game/tick.ts`, `scripts/enemy-controller.ts`, `host/game/combat.ts`, `host/game/types.ts`, `harness/fitness.ts`, `harness/constants.ts` |

**Phase execution order:** 2 → 3 → 4 → 5 → 6 (dependency order). Each phase depends on the prior phase's deliverables.

**Phase objective:** Fix 8 failing tests across 3 test files (gun.test.ts, gun-sprite-data.test.ts, display.worker.test.ts) before starting implementation phases. Three root causes: (1) gun-sprite-data.js sprite grid is too narrow vertically — non-transparent bounds ratio is ~0.67, not the required ~1.6 width/height; (2) palette indices 5 and 6 have alpha 220 instead of 255, breaking the accent-palette alpha-preservation test; (3) gun.ts renderer uses `rgb(r,g,b)` strings for accent-colored pixels instead of the `NEATENSTEIN_GUN_ACCENT_COLOR` hex constant, failing the worker `fillStyle` assertion.

**Pragmatic mode:** This phase uses broad slices (2 slices, one per file) per the plan mandates. No red-phase ceremony — tests already exist and are failing (red). Fix-then-green directly.
NT_COLOR`hex constant, failing the worker`fillStyle` assertion.

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

g gun-sprite-data.test.ts and gun.test.ts tests pass (0 failures)' - 'No new failures in neatenstein test suite' - 'npm run build exit 0'
parallelizable: false
dependencies: - 'P0S1-sprite-redesign'
next_slice: null

````

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
thPatterns=neatenstein.*(gun-sprite-data|gun\.test)` to verify all 7 sprite+gun failures are fixed.

**Stop conditions:** All 7 sprite-data and gun.test.ts failures pass. If a test still fails after redesign, adjust the grid until all constraints are met.

**Required validation:**
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein.*(gun-sprite-data|gun\.test)` — 0 failures
- `npm run build` — exit 0
- `npm run lint` — exit 0 (only if gun-sprite-data.js is a .js file that eslint covers)

**Plan update requirement:** Update the plan with slice status, evidence, and next step before ending.

---
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
````

er.test.ts gun overlay test passes (fillStyle called with #00f0ff)' - 'All existing gun.test.ts tests remain green' - 'npm run build exit 0'
parallelizable: false
dependencies: - 'P0S2-accent-color'
next_slice: null

````

**Step objective:** Make the `renderGunOverlay` function in `gun.ts` set `ctx.fillStyle` to the `NEATENSTEIN_GUN_ACCENT_COLOR` hex constant (`#00f0ff`) when drawing accent-colored pixels (palette indices 5 and 6), instead of converting the palette RGBA to an `rgb()` string.

**Context the agent must know:**

Current state of `examples/neatenstein/browser-entry/renderer/gun.ts`:
- `renderGunOverlay` decodes the sprite frame via `decodeGunSpriteFrame`, then iterates over logical cells.
- For each non-transparent pixel, it sets `ctx.fillStyle` to either `rgba(r,g,b,a/255)` or `rgb(r,g,b)` based on the decoded RGBA values.
- The worker test (`display.worker.test.ts:960-962`) expects `setters.fillStyle` to have been called with `NEATENSTEIN_GUN_ACCENT_COLOR` which is `"#00f0ff"`.
- The current renderer never uses the hex constant — it only uses `rgb()`/`rgba()` strings — so the test fails.
- `NEATENSTEIN_GUN_ACCENT_COLOR` is already imported in `gun.ts` (line 15) and re-exported (line 36). It equals `#00f0ff`.
- The accent color corresponds to palette indices 5 and 6. After Step 01 fixes the palette alpha to 255, the decoded RGBA for accent pixels will be `[0, 240, 255, 255]` and `[0, 200, 220, 255]`.
It equals `#00f0ff`.
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
ess of alpha — after the fix alpha is 255 so it's fully opaque).
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
atterns=neatenstein.*gun\.test` — 0 failures
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
````

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

**Phase objective:** Author all step packets for the five implementation phases from the research findings.

**Stop conditions:** Plan tracker is malformed, or a value-adding step lacks machine-readable acceptance criteria.

```yaml
phase: 1
title: 'Planning & acceptance criteria'
status: '[DONE]'
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

```yaml
phase: 1
step: 1
title: 'Author step packets for Phases 2-6'
status: '[DONE]'
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

**Phase objective:** Wire the generation counter sync (Gap 6) and enemy MLP weight injection (Gap 3) — the two lowest-risk, highest-leverage changes that make enemy evolution functional in the live game.
**Phase objective:** Wire the generation counter sync (Gap 6) and enemy MLP weight injection (Gap 3) — the two lowest-risk, highest-leverage changes that make enemy evolution functional in the live game.

```yaml
phase: 2
title: 'Foundation: generation sync + enemy MLP activation'
status: '[DONE]'
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

=neatenstein.*arms-race'
constitution_check:

- 'principle-4-small-slices'
- 'principle-5-unique-ids'
  placeholder_steps:
- 'Step 01 — Wire generation counter sync (Gap 6)'
- 'Step 02 — Wire enemy MLP population + weight injection (Gap 3)'

````
```yaml
phase: 2
step: 1
title: 'Wire generation counter sync (Gap 6)'
status: '[DONE]'
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
    status: '[DONE]'
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
````

mped to NEATENSTEIN_FIXED_TIMESTEP_MS (16ms fixed, not rAF delta)'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
parallelizable: false
dependencies: []
next_slice: null

````

**Slice details:**
- Add wave-clear detection in `display.worker.ts` after `gameTick` (line 1253): track `allEnemiesCleared` transition `false → true`.
- **Generation increment ownership**: P2S1 does NOT increment `gameState.generation` — that is owned by `advanceWave` in P3S1. P2S1 only detects wave-clear and stores the flag for P3S1 to consume. When advanceWave is wired (P3S1), it increments generation; the worker writes back `result.generation`.
- `GameState.generation` starts at 1 (existing). Harness `generation` starts at 1 (existing). Sync happens in P3S1 when advanceWave is called.
- **Fixed timestep clamping**: In `display.worker.ts:1222-1223`, replace `const dtMs = data.deltaMs` with `const dtMs = NEATENSTEIN_FIXED_TIMESTEP_MS`. Import `NEATENSTEIN_FIXED_TIMESTEP_MS` from `host/game/constants.ts` (the authoritative source — `harness/constants.ts` will later re-export it in P5S2b, but P2S1 imports directly from the source to avoid depending on a not-yet-created re-export). The rAF delta only gates *when* a tick runs, not *how much* time advances. This is critical for replay determinism (Determinism Contract §2).
- Add tests in `host/game/state.test.ts` and `host/game/waves.test.ts` verifying wave-clear detection and timestep clamping.
```yaml
phase: 2
step: 2
title: 'Wire enemy MLP population + weight injection (Gap 3)'
status: '[DONE]'
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
    status: '[DONE]'
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
````

resh weights from snapshot, not undefined'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
parallelizable: false
dependencies: - 'P2S1-gen-sync'
next_slice: null

````

**Slice details:**
- In `display.worker.ts` init (near line 1181): `const enemyPopulation = createMlpEnemyPopulation({ seed: gameState.seed })`.
- Track generation counter; call `enemyPopulation.update({ generation })` to get `MlpSnapshot` on refresh generations.
- Extend `updateEnemyController` signature to accept an OPTIONAL `weights?: Float32Array | undefined` as the 5th parameter (backward-compatible — existing ~150 test call sites and 2 worker call sites continue to work with 4 args; `undefined` = existing BFS behavior). This is additive, not breaking.
- Fix `isRespawn` reset (line 397): on respawn, re-derive weights from the current snapshot rather than forcing `undefined`.
- The MLP re-ranking branch (line 645-711) is already fully implemented — it activates when `weights !== undefined`. This slice makes that branch live.
- **Determinism**: seed derivation is `gameState.seed` (root) → `createMlpEnemyPopulation({ seed })`. Same seed + same generation → same population.

---
**Phase objective:** Wire `advanceWave()` into the live game's wave-clear path so the enemy MLP population evolves each generation, and wire `runArmsRaceGeneration()` on a generation cadence.

```yaml
phase: 3
title: 'Arms-race wiring: advanceWave into live game'
status: '[DONE]'
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
````

```yaml
phase: 3
step: 1
title: 'Wire advanceWave into wave-clear detection'
status: '[DONE]'
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
    status: '[DONE]'
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

y controllers'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-controller'
parallelizable: false
dependencies: - 'P2S2-enemy-mlp-inject'
next_slice: 'P3S2-arms-race-cadence'

````

**Slice details:**
- In `display.worker.ts` after `gameTick` (line 1253): detect `allEnemiesCleared(gameState.enemies)` transition `false → true`.
- On wave-clear: call `advanceWave(gameState, { population: enemyPopulation, spawnCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT })`.
- Replace `gameState` with `result.state`; inject `result.snapshot.weights` into enemy controllers (via the weight injection path from Phase 2).
- `advanceWave` already clears the arena, increments generation, and respawns enemies — replace the trickle `spawnWaveTick` path for wave *transitions*.
```yaml
phase: 3
step: 2
title: 'Wire runArmsRaceGeneration on generation cadence'
status: '[DONE]'
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
    status: '[DONE]'
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
````

k?: Network field (with import type { Network })'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*types'
parallelizable: false
dependencies: - 'P3S1-advance-wave'
next_slice: null

```
k?: Network field (with import type { Network })'
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

- Sync result: `gameState.generation = result.generation`; update `enemySnapshot` from `result.enemySnapshot`.
- The arms-race result's `mainSnapshot` (champion `network` field) is stored for Phase 4's player controller.

---

**Phase objective:** Wire the worker to read `humanMode` and branch to a NEAT network controller that produces `GameTickInputSnapshot` from game-state sensors (Gaps 1 + 5).

```yaml
phase: 4
title: 'Player auto-mode controller'
status: '[DONE]'
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

```yaml
phase: 4
step: 1
title: 'Worker humanMode branching + NEAT network loading (Gap 1)'
status: '[DONE]'
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
    status: '[DONE]'
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

uto-mode ticks'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
parallelizable: false
dependencies: - 'P3S2-arms-race-cadence'
next_slice: 'P4S2-sensor-activation'

````

**Slice details:**
- In `display.worker.ts:1204-1217`, inside the `simState` handler, after `latestState = data.state`:
  - Read `const humanMode = (data.state as any).humanMode`.
  - If `humanMode === 'auto'`: build `tickInput` from the NEAT controller (Phase 4 Step 02).
  - If `humanMode === 'human'` (or undefined): use `pendingTickInput` as before (existing path).
- The champion Network is already stored in worker scope from P3S2's hoisted Neat evaluation (`neatPop.getFittest()` returns a `Network` directly — NOT `.network`). No need to receive it via a separate message; it's already in the worker's local variable.
- The `createMainSnapshot` and `MainVariant` type updates were done in P3S2 (arms-race.ts and types.ts). P4S1 only touches the worker to add the humanMode branch.
- Store the champion Network reference in worker scope; on each `simState` tick in auto mode, call `network.activate(sensors)` → map to `GameTickInputSnapshot`.
- **Worker message protocol**: No new `setNetwork` message type is needed — the worker owns the Neat population and evaluates it locally. The champion Network is already in scope.
```yaml
phase: 4
step: 2
title: 'Sensor extraction + network activation → GameTickInputSnapshot (Gap 5)'
status: '[DONE]'
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
    status: '[DONE]'
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
````

2]→lookDelta, out[3]→fire, out[4]→dash'
validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
parallelizable: false
dependencies: - 'P4S1-worker-branch'
next_slice: null

````

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
## Phase 5 - Real episode fitness [DONE]
> Detailed step/slice/VALIDATION_EVIDENCE blocks compressed to plans/neatenstein-auto-neat-mode.logs.md.
> Steps: 01 (real episode bridge) [DONE], 02 (telemetry + cadence, slices P5S2a/P5S2b/P5S2c) [DONE].

---
## Phase 6 - Integration testing & browser validation [DONE]
> Detailed step/slice/VALIDATION_EVIDENCE blocks compressed to plans/neatenstein-auto-neat-mode.logs.md.
> Steps: 01 (full test suite + build + lint) [DONE], 02 (browser smoke test, 6 rounds) [DONE].

---

## Phase 7 - Test quality: robot-sprite-data capability tests [DONE]
> Detailed step/slice/VALIDATION_EVIDENCE blocks compressed to plans/neatenstein-auto-neat-mode.logs.md.
> Steps: 01 (refactor robot-sprite-data tests, slice 07-robot-sprite-capability-tests) [DONE].

---

## Phase 8 — Coverage gap closure [DONE]
> Detailed step/slice/VALIDATION_EVIDENCE blocks compressed to plans/neatenstein-auto-neat-mode.logs.md.
> Steps: 01 (write targeted tests for 14 coverage gap files + dead code removal, slices P8S1-coverage-closure + P8S1-green) [DONE].
> All 14 neatenstein source files at 100/100/100/100 coverage; 1492 tests pass, 1 skipped; build exit 0; lint exit 0.

---

## Latest validation evidence

Claim: 04-implementing @ 2026-08-11T13:27:00Z — P9S2-fallback-hunter (implemented, coverage green, awaiting 05-green-testing)

```yaml
PlanUpdate:
  slice_id: neatenstein-hud-humanMode-default
  changed_files:
    - examples/neatenstein/browser-entry/host/hud.ts
    - examples/neatenstein/browser-entry/host/hud-human-mode.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx prettier --check <changed-files>'
    - 'npx eslint <changed-files>'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=hud-human-mode'
  preflight_results:
    - 'tsc: OK'
    - 'prettier: OK'
    - 'eslint: 0 issues'
    - 'jest: 4/4 pass (hud-human-mode.test.ts)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=hud-human-mode'
  rollback:
    - 'Revert hud.ts line 337 to auto and line 333 to AUTO label; revert test expectations'
  next: 'Run 05-green-testing for hud-human-mode boundary'
```
````

```yaml
PlanUpdate:
  slice_id: P9S2-fallback-hunter
  changed_files:
    - examples/neatenstein/browser-entry/host/game/constants.ts
    - examples/neatenstein/browser-entry/worker/display.worker.ts
    - examples/neatenstein/browser-entry/worker/display.worker.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'npx jest --config=jest.config.mjs --selectProjects neatenstein --testPathPatterns=display.worker.test.ts --no-cache'
  preflight_results:
    - 'tsc: OK'
    - 'lint: 0 issues'
    - 'prettier: OK'
    - 'jest: 124/124 pass (display.worker.test.ts)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --selectProjects neatenstein --no-cache --coverage --testPathPatterns=display.worker.test.ts'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  next: 'Run 05-green-testing and attach coverage-guard evidence for the three changed files'
```

```yaml
PlanUpdate:
  slice_id: P9S2-fallback-hunter
  changed_files:
    - examples/neatenstein/browser-entry/host/game/constants.ts
    - examples/neatenstein/browser-entry/worker/display.worker.ts
    - examples/neatenstein/browser-entry/worker/display.worker.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'npx jest --config=jest.config.mjs --selectProjects neatenstein --testPathPatterns=display.worker.test.ts --no-cache --coverage'
    - 'node scripts/agent-customization/gates/merge-coverage-summaries.mjs'
  preflight_results:
    - 'tsc: OK'
    - 'lint: 0 issues'
    - 'prettier: OK'
    - 'jest: 132/132 pass (display.worker.test.ts)'
    - 'coverage: display.worker.ts 100/100/100/100; host/game/constants.ts 100/100/100/100'
    - 'merge-coverage-summaries: OK'
    - 'slice-advancement: PASS (shared-validation errored/spawnSync ETIMEDOUT and was skipped)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --selectProjects neatenstein --no-cache --coverage --testPathPatterns=display.worker.test.ts'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/constants.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  next: 'Run 05-green-testing and attach coverage-guard evidence for the three changed files'
```

Research note (fire gate, 2026-08-10): `buildAutoTickInput` in `display.worker.ts` correctly passes sensors and `fireGateState` to `networkOutputToTickInput`; the fire gate is NOT bypassed in live auto-mode. It IS bypassed in the two fitness-evaluation paths (`harness/main-runner.ts:334` and `worker/eval.worker.ts:81`), which call `networkOutputToTickInput(out)` without the gate config. See `plans/neatenstein-auto-neat-mode.research.md` addendum for full evidence. If the gate is meant to be part of the learned policy, the next slice should thread the gate through those fitness functions; otherwise the train/runtime mismatch will persist.

Research note (old shooter dead-code audit, 2026-08-10): A four-scout read-only audit of `examples/neatenstein` found the requested old/dead shooter code categories. Key results: (1) `extractSensors` is not duplicated but is misplaced in `scripts/enemy-navigation.ts` and used by three live consumers; two sensor-based PBRS helpers there are dead, and `sensorHistory` in `host/game/tick.ts` is collected but not consumed. (2) No legacy `NEATENSTEIN_MAIN_NEAT_INPUTS = 12` constant remains; the only literal "12" is `ENEMY_VISIBLE_SENSOR_INDEX = 12`, valid in the 15-input layout. Backward-compat test hooks in `display.worker.ts`/`display.worker.test.ts` and the active plan prose still refer to "12 inputs". (3) Enemies in `scripts/enemy-controller.ts` remain omniscient: yaw/flanking/BFS distance map are derived directly from `gameState.player.position`. (4) Placeholder fitness persists in `barrier.ts` (RNG episode), `arms-race.ts` (zero-quality fallback), `fitness.ts` (zero `complexityBonus`/`parsimonyDensityPenalty`), and `tick.ts` (unread `sensorHistory`). (5) Fire-without-vision is present in headless fitness paths (`main-runner.ts:334`, `eval.worker.ts:81`) and the dead `enemy-runner.ts`, while the live display-worker path is correctly gated. Two standalone orchestration modules (`enemy-runner.ts`, `main-agent.ts`) and several dead helper/interface/constant exports were also identified. Full file/line inventory and risks are in `plans/neatenstein-auto-neat-mode.research.md` Addendum 2.

Research note (15-input vision bypass, 2026-08-10): The 15-input `extractSensors` vector **is** used in live auto-mode whenever a champion network exists (`display.worker.ts:1671-1678` → `buildAutoTickInput` → `extractSensors` → `network.activate` → `networkOutputToTickInput` with fire gate). The call-path direction is producer/consumer: `buildAutoTickInput` produces the `GameTickInputSnapshot` that `gameTick` consumes; there is no downstream edge from `gameTick` to `buildAutoTickInput`. The only live bypass is `buildFallbackAutoTickInput` (`display.worker.ts:1497-1553`), used when `humanMode === 'auto'` but `championMainNetwork` is still `null`. That fallback uses raw Euclidean distance to the nearest active enemy, ignores `VISION_RANGE_CELLS` and wall line-of-sight, and fires by bearing/cooldown — i.e., the old omniscient 12-input-style behavior. No production 12-input champion path remains thanks to the genome-extinction guard (`display.worker.ts:1656-1669`). The remaining "12" references are `ENEMY_VISIBLE_SENSOR_INDEX = 12` (valid 15-input index), backward-compat test hooks, and the stale "~12 inputs" prose in this plan. Full evidence is in `plans/neatenstein-auto-neat-mode.research.md` Addendum 3.

Research note (sensor support for exploration and kiting, 2026-08-11): The 15-input `extractSensors` layout supports basic reactive kiting (distance `[6]`, relative bearing `[5]`, fire-arc `[13]`) and rudimentary wall-following exploration when no enemy is visible (`enemyVisible = 0`, cardinal wall rays `[8-11]`). It is missing predictive kiting signals (enemy velocity, dash readiness, player-relative forward obstacle distance) and exploration memory (time-since-enemy-seen, last-known-bearing decay, scan-state timer). Highest-value additions would be `sin/cos` bearing encoding, a `dashReady`/`dashCooldown` sensor, and a `forwardWallDistance` raycast. If the input budget must remain 15, candidates for displacement are `lastShotHit` `[14]` and one absolute position sensor `[3]`/`[4]`. Full evidence and alternatives are in `plans/neatenstein-auto-neat-mode.research.md` Addendum 4.

---

## Phase 9 — Hunter behavior: fallback AI fixes (exploration + kiting) [DONE]
> Detailed step/slice/VALIDATION_EVIDENCE blocks compressed to plans/neatenstein-auto-neat-mode.logs.md.
> Steps: 01 (plan exploration + kiting behavior) [DONE], 02 (implement exploration + kiting, slice P9S2-fallback-hunter) [DONE], 03 (validate fallback AI hunter behavior + browser smoke) [DONE].
> 134/134 targeted tests pass; touched files `display.worker.ts` and `host/game/constants.ts` at 100/100/100/100 coverage; browser smoke PASS.

---
