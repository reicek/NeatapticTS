# Neatenstein Auto / NEAT Mode Wiring

**Status:** [WIP]
**Plan ID:** NEATENSTEIN_AUTO_NEAT_MODE
**Created:** 2026-08-09
**Source of truth:** `plans/neatenstein-auto-neat-mode.plans.md`
**Research artifact:** `plans/neatenstein-auto-neat-mode.research.md`

## Mandates

- `model: kimi-k2.7-code:cloud` for all dispatches under this plan.
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

## Non-goals

- No changes to the core NeatapticTS library (`src/`) — only consume `Network.activate()` and NEAT APIs from existing exports.
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

| Gap | Deliverable | Research section | Primary files |
|-----|-------------|-----------------|---------------|
| 6 | Generation counter sync | §Gap 6 | `host/game/state.ts`, `host/game/tick.ts`, `worker/display.worker.ts`, `host/game/waves.ts` |
| 3 | Enemy MLP weight injection | §Gap 3 | `scripts/enemy-controller.ts`, `harness/enemy-mlp.ts`, `worker/display.worker.ts` |
| 2 | Arms-race wiring (advanceWave) | §Gap 2 | `host/waves.ts`, `worker/display.worker.ts`, `host/game/waves.ts` |
| 1 | Worker humanMode branching | §Gap 1 | `worker/display.worker.ts`, `browser-entry.ts`, `host/renderer-bridge.ts` |
| 5 | Player NEAT controller | §Gap 5 | `worker/display.worker.ts`, `host/game/tick.ts`, `host/game/types.ts`, `src/architecture/network/network.ts` |
| 4 | Real episode fitness | §Gap 4 | `harness/main-runner.ts`, `host/game/episode.ts`, `harness/fitness.ts` |

## Implementation phases

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
    text: 'GameState.generation increments when a wave is cleared (allEnemiesCleared transition)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*state'
  - id: AC-015
    text: 'Generation counter is read by the worker and synced from harness advanceWave result'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(state|waves)'
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
      - 'examples/neatenstein/browser-entry/host/game/state.ts'
      - 'examples/neatenstein/browser-entry/host/game/waves.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: AC-017
        text: 'GameState.generation increments on wave-clear transition'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*state'
      - id: AC-018
        text: 'Worker reads generation from advanceWave result and writes to GameState'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*waves'
    parallelizable: false
    dependencies: []
    next_slice: null
```

**Slice details:**
- Add wave-clear detection in `display.worker.ts` after `gameTick` (line 1253): track `allEnemiesCleared` transition `false → true`.
- On wave-clear, increment `gameState.generation` and store it for the next `advanceWave` call.
- `GameState.generation` starts at 1 (existing). Harness `generation` starts at 1 (existing). Sync: when `advanceWave` is called, pass `gameState.generation`; when it returns, write back `result.generation`.
- Add tests in `host/game/state.test.ts` and `host/game/waves.test.ts` verifying generation increments.

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
      - 'examples/neatenstein/browser-entry/harness/enemy-mlp.ts'
    acceptance_criteria:
      - id: AC-024
        text: 'Worker creates MlpEnemyPopulation on init with seed from game config'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*enemy-mlp'
      - id: AC-025
        text: 'updateEnemyController receives champion weights and sets them on ControlledEnemy'
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
- In `display.worker.ts` init (near line 1181): `const enemyPopulation = createMlpEnemyPopulation({ seed })`.
- Track generation counter; call `enemyPopulation.update({ generation })` to get `MlpSnapshot` on refresh generations.
- Extend `updateEnemyController` signature to accept a `weights: Float32Array | undefined` parameter (or assign `controller.enemies[i].weights` after creation and after each respawn).
- Fix `isRespawn` reset (line 397): on respawn, re-derive weights from the current snapshot rather than forcing `undefined`.
- The MLP re-ranking branch (line 645-711) is already fully implemented — it activates when `weights !== undefined`. This slice makes that branch live.

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
      - 'examples/neatenstein/browser-entry/host/waves.ts'
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
    text: 'runArmsRaceGeneration is called on generation advance (async, does not block render)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
  - id: AC-040
    text: 'Arms-race result syncs generation back to GameState and updates enemy snapshot'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
  - id: AC-041
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*arms-race'
slices:
  - slice_id: 'P3S2-arms-race-cadence'
    title: 'Wire runArmsRaceGeneration on generation cadence with async evaluation'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/harness/arms-race.ts'
    acceptance_criteria:
      - id: AC-042
        text: 'runArmsRaceGeneration called with {seed, generation, enemySnapshot, humanMode}'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
      - id: AC-043
        text: 'Result generation syncs to GameState.generation; enemySnapshot updates enemy population'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*arms-race'
    parallelizable: false
    dependencies:
      - 'P3S1-advance-wave'
    next_slice: null
```

**Slice details:**
- On each generation advance (after `advanceWave`), call `runArmsRaceGeneration({ seed, generation: gameState.generation, enemySnapshot: currentSnapshot, humanMode })`.
- Run asynchronously (does not block the render loop) — queue the result and apply on next tick.
- Sync result: `gameState.generation = result.generation`; update `enemySnapshot` from `result.enemySnapshot`.
- `humanMode` comes from the `simState` message (already posted by host).
- The arms-race result's `mainSnapshot` (champion genome) is stored for Phase 4's player controller.

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
    text: 'NEAT network is loaded into worker via init message or new setNetwork message'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-057
    text: 'In auto mode, tickInput comes from NEAT controller, not pendingTickInput (human queue)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
  - id: AC-058
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*display.worker'
slices:
  - slice_id: 'P4S1-worker-branch'
    title: 'Worker reads humanMode, branches to NEAT controller, loads network'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    acceptance_criteria:
      - id: AC-059
        text: 'simState handler branches on humanMode at line 1204-1217'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*display.worker'
      - id: AC-060
        text: 'Network received via message and stored in worker scope'
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
- Add a new message type (or extend `init`) to receive a serialized NEAT network from the host.
- The network is the champion genome from `runArmsRaceGeneration` (Phase 3) or a fresh `new Neat(inputCount, outputCount, fitnessFn)`.
- Store the network in worker scope; on each `simState` tick in auto mode, call `network.activate(sensors)` → map to `GameTickInputSnapshot`.

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
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*(tick|display.worker)'
acceptance_criteria:
  - id: AC-061
    text: 'Sensor extractor builds observation vector from GameState (player + nearest enemy + walls)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*tick'
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
      - 'examples/neatenstein/browser-entry/host/game/tick.ts'
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
- Build `extractSensors(gameState: GameState): number[]` in the worker (or a new helper file):
  - Player: `health/maxHealth`, `ammo`, `angleRad`, `position.x`, `position.y` (5)
  - Nearest enemy: relative bearing (`atan2(dy,dx) - playerAngle`), distance (`hypot(dx,dy)`), `health` (3)
  - Wall raycasts: 4 cardinal directions, distance to nearest wall (4)
  - Total: 12 sensors (minimal set, expandable)
- Build `networkOutputToTickInput(outputs: number[]): GameTickInputSnapshot`:
  - `move.x = tanh(outputs[0])` (strafe)
  - `move.y = tanh(outputs[1])` (forward/back)
  - `lookDelta = tanh(outputs[2]) * MAX_TURN_RATE` (radians)
  - `fire = outputs[3] > 0` (threshold)
  - `dash = outputs[4] > 0.5` (threshold)
- Use `Network.activate(sensors)` or `noTraceActivate(sensors)` from `src/architecture/network/network.ts`.
- The network is constructed from the NGE embryo (`buildMainAgentEmbryo`) or a simple `new Neat(12, 5, fitnessFn)`.

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
    text: 'Cadence target met: generations-per-minute >= 2 with real episodes'
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
    text: 'updateEpisode accepts controller inputs from the NEAT network (new parameter)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
  - id: AC-077
    text: 'RNG-based runEpisode stub deleted (no dual-path code)'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
  - id: AC-078
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*main-runner'
slices:
  - slice_id: 'P5S1-real-episode'
    title: 'Replace RNG runEpisode with real gameplay + controller input plumbing'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/harness/main-runner.ts'
      - 'examples/neatenstein/browser-entry/host/game/episode.ts'
    acceptance_criteria:
      - id: AC-079
        text: 'main-runner runEpisode uses host episode with NEAT controller inputs'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*main-runner'
      - id: AC-080
        text: 'updateEpisode signature accepts optional controllerInput: GameTickInputSnapshot'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
      - id: AC-081
        text: 'Deterministic damage bot removed; NEAT network drives player in evaluation'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
    parallelizable: false
    dependencies:
      - 'P4S2-sensor-activation'
    next_slice: 'P5S2-telemetry-cadence'
```

**Slice details:**
- Delete the RNG-based `runEpisode` in `main-runner.ts:309-332` (no dual-path code per Mandates).
- Replace with a call to the host `runEpisode` (or a headless fast-forward variant):
  - Build an `Episode` from `(seed, generation, variantId, enemySnapshot)`.
  - Construct a NEAT network from the variant's genome (`buildMainAgentEmbryo` → Network).
  - For each step: `extractSensors(state)` → `network.activate(sensors)` → `networkOutputToTickInput(outputs)` → `updateEpisode(state, dtMs, collisionMap, controllerInput)`.
  - Remove the "deterministic damage bot" — the NEAT network drives the player.
- Extend `updateEpisode` (`episode.ts:244`) to accept an optional `controllerInput?: GameTickInputSnapshot`:
  - If provided: route movement/fire through `fireBolt` (combat.ts) and movement step.
  - If absent: existing behavior (for backward compat with human-mode tests — but per Mandates, no dual-path; all evaluation uses controller inputs).

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
    text: 'Cadence: episode duration reduced to 5s for fitness eval; meets 2 gen/min target'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*cadence'
  - id: AC-086
    text: '100% coverage on touched files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=neatenstein.*(fitness|cadence)'
slices:
  - slice_id: 'P5S2-telemetry-cadence'
    title: 'Telemetry extraction from real gameplay + cadence optimization'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/game/episode.ts'
      - 'examples/neatenstein/browser-entry/host/game/combat.ts'
      - 'examples/neatenstein/browser-entry/harness/fitness.ts'
    acceptance_criteria:
      - id: AC-087
        text: 'GameState or parallel telemetry object tracks damageDealt, shotsFired, shotsHit'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*episode'
      - id: AC-088
        text: 'extractCombatQualitySignal(gameState, telemetry) produces CombatQualitySignal from real metrics'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*fitness'
      - id: AC-089
        text: 'Episode duration for fitness eval = 5s (NEATENSTEIN_FITNESS_EPISODE_DURATION_MS)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=neatenstein.*cadence'
    parallelizable: false
    dependencies:
      - 'P5S1-real-episode'
    next_slice: null
```

**Slice details:**
- Add telemetry tracking to `GameState` (or a parallel `EpisodeTelemetry` object):
  - `damageDealt`: accumulate in `applyEnemyDamage` (combat.ts).
  - `shotsFired` / `shotsHit`: track in `fireBolt` (combat.ts) — increment `shotsFired` on each bolt, `shotsHit` on enemy hit.
  - `aimMissRate = (shotsFired - shotsHit) / max(1, shotsFired)`.
- Build `extractCombatQualitySignal(gameState, telemetry): CombatQualitySignal`:
  - `survivalTicks = episodeTimeMs / NEATENSTEIN_FIXED_TIMESTEP_MS`
  - `damageDealt = telemetry.damageDealt`
  - `kills = gameState.kills`
  - `damageTaken = (deaths * maxHealth) + (maxHealth - player.health)`
  - `aimMissRate = telemetry.aimMissRate`
- Cadence: reduce fitness evaluation episode duration to 5s (`NEATENSTEIN_FITNESS_EPISODE_DURATION_MS = 5000`).
  - 4 variants × 5s = 20s + overhead ≈ 22s → ~2.7 gen/min (meets 2 gen/min target).
  - If 4 variants insufficient, reduce to 3 or run evaluation async.

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
Current boundary: Phase 1 Step 01 (planning) — authoring step packets for Phases 2-6.
Next: Dispatch implementation agents for Phase 2 (generation sync + enemy MLP activation) after plan review approval.
Validations: npx jest --testPathPattern=neatenstein, npm run build, npm run lint.
Caution: Do not run full test suite in a single shell invocation — use targeted patterns.
```

## Latest validation evidence

- Research complete: 5 specialists investigated 6 gaps, findings recorded in research artifact and plan.
- Plan authored: Phases 1-6 with step packets, slices, acceptance criteria, and traceability.
- Awaiting: 5-plan review approval before dispatching implementation agents.