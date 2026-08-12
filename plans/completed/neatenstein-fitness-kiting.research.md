# Neatenstein Fitness: Exploration & Kiting Reward-Shaping Research

## Question

Does `computeCombatQualitySignal` in `examples/neatenstein/browser-entry/harness/fitness.ts` currently reward:

1. **Exploration / productive movement** when no enemy is visible?
2. **Distance maintenance (kiting)** around a 15–20 cell band when an enemy is visible?

If not, what is the smallest, source-grounded reward-shaping design that adds these signals without breaking the existing combat-quality composite?

## Evidence

### Current fitness formula has no exploration or distance terms

`examples/neatenstein/browser-entry/harness/fitness.ts:111-161`

```ts
export function computeCombatQualitySignal(
  signal: CombatQualitySignal,
  complexity?: number,
): FitnessScore {
  const baseScore =
    signal.survivalTicks * NEATENSTEIN_WEIGHT_SURVIVAL_TICKS +
    signal.damageDealt * NEATENSTEIN_WEIGHT_DAMAGE_DEALT +
    signal.kills * NEATENSTEIN_WEIGHT_KILLS +
    killEfficiency * NEATENSTEIN_WEIGHT_KILL_EFFICIENCY +
    hitRate * NEATENSTEIN_WEIGHT_HIT_RATE +
    killRate * NEATENSTEIN_WEIGHT_KILL_RATE +
    fireRate * NEATENSTEIN_WEIGHT_FIRE_RATE -
    signal.damageTaken * NEATENSTEIN_WEIGHT_DAMAGE_TAKEN -
    signal.aimMissRate * NEATENSTEIN_WEIGHT_AIM_MISS_RATE -
    (signal.shotsBlindFire ?? 0) * NEATENSTEIN_WEIGHT_BLIND_FIRE_PENALTY -
    (signal.shotsWallHit ?? 0) * NEATENSTEIN_WEIGHT_WALL_HIT_PENALTY +
    signal.complexityBonus * NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS -
    signal.parsimonyDensityPenalty *
      NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY;
  // ... parsimony band only
}
```

The signal rewards survival, damage, kills, shot efficiency/rates, complexity and penalizes damage taken, misses, blind fire and wall hits. There is **no positive incentive to move** while no enemy is visible, and **no distance-to-enemy term** at all.

### `CombatQualitySignal` and `EpisodeTelemetry` lack the needed fields

- `harness/types.ts:56-81` defines `CombatQualitySignal`: optional fields cover shot taxonomy (`shotsFired`, `shotsHit`, `shotsBlindFire`, `shotsWallHit`, `ticksElapsed`) but nothing about enemy-distance history or no-enemy-visible ticks.
- `host/game/types.ts:231-272` defines `EpisodeTelemetry`: also shot/combat counters only.

### Per-tick sensor data is available but thrown away

`examples/neatenstein/browser-entry/harness/main-runner.ts:352` extracts the 15-element sensor vector every tick:

```ts
const sensors = extractSensors(state, flatMap, NEATENSTEIN_MAP_SIZE);
```

The runner already consumes `sensors[ENEMY_VISIBLE_SENSOR_INDEX]` (index 12, `neat-io-config.ts:54`) for the fire gate (`main-runner.ts:359`). The same vector also carries the nearest visible enemy distance at sensor index 6, normalized by `VISION_RANGE_CELLS` (`enemy-navigation.ts:475`). However, once the tick is advanced, that distance/visibility information is **not accumulated** into the episode summary.

### Vision range vs. the requested 15–20 cell kiting band

`examples/neatenstein/scripts/enemy-navigation.ts:36`:

```ts
const VISION_RANGE_CELLS = 15;
```

`findNearestVisibleEnemy` (`enemy-navigation.ts:382-407`) only returns an enemy when:

1. Euclidean distance `<= 15`, and
2. There is clear line-of-sight.

Therefore a 15–20 cell band is **at or beyond the current vision edge**. If kiting reward is computed from the visible-enemy sensor only, the agent would be rewarded right at the moment the enemy disappears (sensor = 1.0) or not at all. Supporting the requested band requires either:

- **Option A**: compute distance to the nearest _active_ enemy in the runner directly from `state.enemies`, regardless of visibility/LOS.
- **Option B**: raise `VISION_RANGE_CELLS` to at least 20 and normalize sensor[6] accordingly; this also changes the fire-gate and sensor semantics.

### Existing combat controls already reduce blind firing

- The soft fire gate (`neat-io-config.ts:114-134`) suppresses fire when no enemy is visible (`enemyVisible < 0.15`).
- The fitness composite penalizes `shotsBlindFire` (`constants.ts:150`) and `shotsWallHit` (`constants.ts:158`).

So the agent is already discouraged from shooting at nothing; the missing piece is a **positive pressure to do something useful** (move/explore) during those periods.

### Relevant reference constants

- Contact damage range: `NEATENSTEIN_CONTACT_RANGE_CELLS = 0.5` (`host/game/constants.ts:132`).
- Bolt max range: `NEATENSTEIN_BOLT_MAX_RANGE_CELLS = 30` (`host/game/constants.ts:370`).
- Max episode ticks: `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = 312` (`harness/constants.ts:40`).

## Decision

Add two small, additive reward-shaping terms to the combat-quality composite, plus the per-tick accumulation needed to feed them.

### 1. Exploration reward — active when no enemy is visible

In `main-runner.ts:runEpisode`, count ticks where `sensors[12] < 0.5` (or use the same `enemyVisible` value passed to the fire gate). Reward each such tick with a small positive weight so standing still during search is strictly worse than moving, without the agent needing to farm kills.

Recommended initial values (to be tuned empirically):

- `NEATENSTEIN_WEIGHT_EXPLORATION = 0.05` per tick with no visible enemy.
- Optional, higher-fidelity variant: track **unique grid cells visited** while no enemy is visible (a `Set<string>` of `x|y` grid keys), using the asciiMaze `recordMazeMovementVisitAndPenalties` pattern (`examples/asciiMaze/mazeMovement/runtime/mazeMovement.runtime.ts`). This rewards covering new ground rather than just moving in circles.

### 2. Distance-maintenance / kiting reward — active around 15–20 cells

Compute the raw Euclidean distance from `state.player.position` to the nearest active enemy in `state.enemies` every tick inside `runEpisode` (do not rely on the visible-enemy sensor because of the 15-cell cap). Derive a distance-quality scalar:

```ts
function computeKitingQuality(distanceCells: number): number {
  if (distanceCells < NEATENSTEIN_CONTACT_RANGE_CELLS) return -1; // too close
  if (
    distanceCells >= NEATENSTEIN_KITING_MIN_CELLS &&
    distanceCells <= NEATENSTEIN_KITING_MAX_CELLS
  )
    return +1; // sweet spot
  if (distanceCells > NEATENSTEIN_BOLT_MAX_RANGE_CELLS) return -0.5; // too far to engage
  return 0; // acceptable but not rewarded
}
```

Add a running sum `kitingScore` over the episode.

Recommended initial constants:

- `NEATENSTEIN_KITING_MIN_CELLS = 15`
- `NEATENSTEIN_KITING_MAX_CELLS = 20`
- `NEATENSTEIN_WEIGHT_KITING = 0.5` per tick in the band.
- `NEATENSTEIN_WEIGHT_TOO_CLOSE = 1.0` (multiplier on the -1 region).
- `NEATENSTEIN_WEIGHT_TOO_FAR = 0.2` (multiplier on the beyond-bolt-range region).

### 3. Type and formula changes

Extend `CombatQualitySignal` in `harness/types.ts`:

```ts
export interface CombatQualitySignal {
  // ... existing fields ...
  /** Ticks spent with no visible enemy (exploration-mode proxy). */
  explorationTicks?: number;
  /** Accumulated distance-maintenance quality score. */
  kitingScore?: number;
}
```

Add terms in `computeCombatQualitySignal`:

```ts
const baseScore =
  // ... existing terms ...
  +(signal.explorationTicks ?? 0) * NEATENSTEIN_WEIGHT_EXPLORATION +
  (signal.kitingScore ?? 0) * NEATENSTEIN_WEIGHT_KITING;
```

Add the new weights to `harness/constants.ts`.

### 4. Where to accumulate

The smallest insertion point is the existing tick loop in `main-runner.ts:351-367`. After extracting sensors, before `gameTick`, compute and accumulate the new metrics using the already-extracted `sensors` and the current `state.enemies`. Then pass the accumulated values into `extractCombatQualitySignal` or directly into the returned `CombatQualitySignal`.

If a richer “unique cells visited” exploration reward is desired, also record `tickInput.move` and the player grid cell after `gameTick` returns the next `state`.

### 5. Test additions

- `fitness.test.ts`: sign-convention tests that fitness increases with `explorationTicks` and `kitingScore` and stays backward-compatible when the new optional fields are omitted.
- `main-runner.test.ts`: deterministic smoke assertions that the returned signal contains the new numeric fields and that `kitingScore` is non-negative/zero for an all-zero/mock sensor case.

## Risks

1. **Vision-range mismatch** — the requested 15–20 cell band sits at/above the current `VISION_RANGE_CELLS = 15` cap. Using only the visible-enemy sensor for kiting would create a perverse reward right at the vision boundary or none at all. Compute raw enemy distance in the runner (Option A) to avoid changing sensor/fire-gate semantics.
2. **Over-rewarding passive circling** — if `NEATENSTEIN_WEIGHT_KITING` is too high, agents may learn to maintain distance indefinitely rather than dealing damage. Keep kiting weight smaller than `NEATENSTEIN_WEIGHT_DAMAGE_DEALT` (2) and `NEATENSTEIN_WEIGHT_KILLS` (5), and tune with champion replays.
3. **Exploration vs. combat trade-off** — a strong exploration bonus could pull the agent away from fights. Weight should be small (≪ survival tick weight) and ideally tied to _new_ cells visited, not just movement.
4. **Determinism** — any per-tick accumulator (Set, running sums) must be local to `runEpisode` and derived only from deterministic state; no shared mutable state.
5. **Cortex RAG index was stale** during this research; direct file reads were used as fallback. Rebuild the index (`node rag-index/build-index.mjs`) before relying on semantic search for the implementation slice.

## Provenance

| Finding                                                      | Confidence | Source                                                                        |
| ------------------------------------------------------------ | ---------- | ----------------------------------------------------------------------------- |
| Fitness formula lacks exploration/distance terms             | 0.98       | static code `fitness.ts:111-161`                                              |
| Sensor vector carries enemyVisible[12] and enemyDistance[6]  | 0.98       | static code `neat-io-config.ts:54`, `enemy-navigation.ts:475`                 |
| Runner discards per-tick distance/visibility after fire gate | 0.95       | static code `main-runner.ts:352-360`                                          |
| VISION_RANGE_CELLS = 15, limiting visible distance to ≤15    | 0.98       | static code `enemy-navigation.ts:36`                                          |
| Bolt max range = 30, contact range = 0.5                     | 0.98       | static code `host/game/constants.ts:132,370`                                  |
| Prior art for visit-based exploration exists in asciiMaze    | 0.90       | static code `examples/asciiMaze/mazeMovement/runtime/mazeMovement.runtime.ts` |
