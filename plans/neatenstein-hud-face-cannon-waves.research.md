# Neatenstein HUD / Face Cannon / Waves — Vision Filtering Live Gameplay Investigation

> Research artifact for active plan `neatenstein-hud-face-cannon-waves.plans.md`.
> Related plans/research:
> - `plans/neatenstein-auto-neat-mode.plans.md` (Auto / NEAT mode wiring, Phases 1–8 [DONE])
> - `plans/neatenstein-firing-sensor-system.research.md` (sensor/LOS/fire-gating redesign, Phases 1–5 [DONE])

## Question

1. Why do `findNearestVisibleEnemy` and vision-range filtering appear to fail in live Neatenstein gameplay?
2. Is `extractSensors` in `scripts/enemy-navigation.ts` actually called by the real game loop, or only by tests?
3. Where does old omniscient enemy-detection code still run?

## Evidence

### `extractSensors` is called by real runtime paths, not just tests

| Consumer | File | Line | Context |
|----------|------|------|---------|
| Live champion auto-AI | `examples/neatenstein/browser-entry/worker/display.worker.ts` | 1451 | `buildAutoTickInput` when `humanMode === 'auto' && championMainNetwork` is set |
| Fitness evaluation worker | `examples/neatenstein/browser-entry/worker/eval.worker.ts` | 76 | `runFitnessEpisode` evaluates every network during evolution |
| Headless main-agent runner | `examples/neatenstein/browser-entry/harness/main-runner.ts` | 329 | `runEpisode` headless fitness episode |
| Tests | `examples/neatenstein/scripts/enemy-navigation.test.ts` | many | `extractSensors` + `findNearestVisibleEnemy` coverage |

**Confidence: 0.96** | Provenance: static-code

### `findNearestVisibleEnemy` is implemented correctly and unit tests pass

`examples/neatenstein/scripts/enemy-navigation.ts:382` defines `findNearestVisibleEnemy` with:

- Skip `active === false` enemies.
- Euclidean distance `<= VISION_RANGE_CELLS` (15 cells).
- `hasLineOfSight` from `examples/neatenstein/browser-entry/renderer/raycast.ts:226`.

Validation command:

```bash
npx jest --testPathPatterns="enemy-navigation.test.ts" --testNamePattern="findNearestVisibleEnemy|extractSensors" --no-coverage
```

Result:

```text
Tests:       55 skipped, 25 passed, 80 total
Test Suites: 1 passed, 1 total
```

All vision-range, line-of-sight, inactive-enemy, and nearest-selection cases pass.

**Confidence: 0.94** | Provenance: static-code + runtime validation

### The omniscient fallback auto-AI bypasses all vision filtering

`examples/neatenstein/browser-entry/worker/display.worker.ts:1497` defines `buildFallbackAutoTickInput`. Target selection uses raw Euclidean distance only:

```typescript
for (const enemy of state.enemies) {
  if (enemy.active === false) continue;
  const dx = enemy.position.x - px;
  const dy = enemy.position.y - py;
  const distSq = dx * dx + dy * dy;
  if (distSq < nearestDistSq) { ... }
}
```

There is **no `VISION_RANGE_CELLS` check** and **no `hasLineOfSight` check**. The fallback can detect enemies through walls and across the entire map.

The fallback runs whenever `humanMode === 'auto'` and `championMainNetwork === null`:

```typescript
if (humanMode === 'auto' && championMainNetwork) {
  tickInput = buildAutoTickInput(...);              // vision-filtered champion path
} else if (humanMode === 'auto') {
  tickInput = buildFallbackAutoTickInput(gameState); // omniscient fallback
}
```

`championMainNetwork` starts as `null` and is only populated after `handleEvalComplete` receives a champion from the eval worker (`display.worker.ts:1384-1390`). The eval worker is delegated only after `advanceWave` is triggered by wave-clear. Prior logs (`plans/neatenstein-auto-neat-mode.logs.md:808`) identified that the worker's wave-clear detection uses `enemies.length` instead of `allEnemiesCleared()`, so the champion may never be produced, leaving the omniscient fallback active indefinitely.

**Confidence: 0.97** | Provenance: static-code

### Enemy→player detection is separate and already range/LOS-gated

`examples/neatenstein/scripts/enemy-controller.ts:960-974` gates enemy firing with:

- `distToPlayer <= ENEMY_CONTROLLER_FIRE_RANGE_CELLS`
- `hasLineOfSight(position, gameState.player.position, collisionMap)`

This is enemy AI shooting at the player and is not the source of the reported player-vision bug.

**Confidence: 0.95** | Provenance: static-code

## Decision

The reported symptom is **not caused by a bug in `findNearestVisibleEnemy` or `extractSensors`**. Both are implemented correctly and are used by the champion path, the eval worker, and the headless runner.

The actual cause is the **fallback auto-AI path** (`buildFallbackAutoTickInput`), which:

1. Runs whenever there is no champion network (the default initial state, and can persist indefinitely due to the wave-clear detection bug).
2. Selects the nearest active enemy using raw Euclidean distance with no vision-range or line-of-sight check.
3. Therefore appears omniscient in live gameplay.

## Recommended Fix

Wire `buildFallbackAutoTickInput` to the same visibility primitive as the champion path:

- Use `findNearestVisibleEnemy(state, wallMap, NEATENSTEIN_MAP_SIZE, VISION_RANGE_CELLS)` to pick the fallback target.
- Only steer toward and fire at enemies that pass range + LOS.
- This removes the "enemy detected through walls" behavior and aligns the fallback with the NEAT controller's vision model.

A smaller alternative is to add a manual range/LOS check inside `buildFallbackAutoTickInput`, but reusing `findNearestVisibleEnemy` avoids duplicating the visibility logic.

## Risks

- The fallback AI currently relies on knowing enemy positions through walls to steer. Capping it to vision range may make the fallback less effective at clearing the first wave, which could delay or prevent the first champion from being produced.
- If the wave-clear detection bug (`enemies.length` vs `allEnemiesCleared()`) is not fixed separately, the champion may never arrive regardless of fallback changes.
- Changing fallback target selection does not affect the sensor-vector size or network input count, so it is safe for existing champion networks.

## Confidence

Overall root-cause confidence: **0.95**.

## Active Plan Mismatch

The workflow MCP reports the active plan as `plans/neatenstein-hud-face-cannon-waves.plans.md`, but that file does not exist in `plans/`. `plans/README.md` does not list it either. The most relevant existing plan is `plans/neatenstein-auto-neat-mode.plans.md` (Phases 1–8 [DONE]). This research artifact is materialized alongside the reported active plan name; the matching `.plans.md` tracker must be created or the active plan reference must be corrected by `01-planning` before handoff to implementation.
