# Neatenstein hunter bugfix triad — research notes

## Question

Why do enemy waves stop spawning after approximately 14 kills in the Neatenstein demo hunter?

## Evidence

### 1. `spawnWaveTick` pauses after every 8 total spawns while any enemy is alive

In `examples/neatenstein/browser-entry/host/game/waves.ts`, `spawnWaveTick` increments a monotonic `spawnCount` and computes a `batchComplete` flag:

```ts
const cap = NEATENSTEIN_ENEMY_MAX_CONCURRENT; // 8
const batchComplete = state.spawnCount > 0 && state.spawnCount % cap === 0;

if (batchComplete && !allEnemiesCleared(state)) {
  return { state, spawned: 0 };
}
```

Once `spawnCount` hits a multiple of 8, **no further enemies are spawned until every enemy in the current arena is dead or inactive** (`allEnemiesCleared` checks `enemy.health <= 0 || enemy.active === false`).

### 2. `advanceWave` does not reset `spawnCount`

`examples/neatenstein/browser-entry/host/waves.ts` defines `advanceWave`:

```ts
let next = { ...state, enemies: [], bolts: [], impacts: [] };
next.generation += 1;
```

It clears projectiles and increments the generation, but it **does not reset `spawnCount`**. Because `spawnWaveTick` is called inside `advanceWave`, the next wave adds another 8 to the running total.

### 3. Wave-clear detection in the display worker is correct

`examples/neatenstein/browser-entry/worker/display.worker.ts` detects a wave clear by comparing `hasAliveEnemies` before and after the tick:

```ts
const hasAliveEnemiesBefore = hasAliveEnemies(prevState);
...
const hasAliveEnemiesAfter = hasAliveEnemies(nextState);
if (hasAliveEnemiesBefore && !hasAliveEnemiesAfter) {
  // trigger advanceWave
}
```

`hasAliveEnemies` checks `health > 0 && active !== false`, matching the definition used by `allEnemiesCleared`. The detection fires exactly once on the alive→dead transition and calls `advanceWave(state, { spawnCount: 8 })`.

### 4. Why the symptom appears around 14 kills

- The first batch of 8 enemies is spawned and killed: `spawnCount` = 8.
- Wave clear triggers `advanceWave`; it does not reset `spawnCount`.
- The second batch of 8 enemies is spawned: `spawnCount` = 16.
- At 14 kills, 6 of the second batch are dead and 2 are still alive.
- `spawnWaveTick` now sees `spawnCount = 16` (`16 % 8 === 0`) and `allEnemiesCleared` is false, so it returns without spawning.
- The spawner remains paused until the player kills the remaining 2 enemies.

The pause is not a wave-clear detection failure; it is the intended behavior of the current batch-based gating logic.

### 5. Existing tests encode the batch model

`examples/neatenstein/browser-entry/host/game/waves.test.ts` includes expectations such as:

> "starts a new batch once all current enemies are dead"

This confirms the current design intentionally waits for a full clear before the next batch.

## Decision

- The root cause is the `batchComplete` gate in `spawnWaveTick` combined with `advanceWave` not resetting `spawnCount`.
- The wave-clear detection in `display.worker.ts` is working correctly and should not be modified for this symptom.
- The fix should change the spawn model so that new enemies are spawned continuously as long as the number of **currently alive** enemies is below `NEATENSTEIN_ENEMY_MAX_CONCURRENT`, rather than pausing until an entire 8-enemy batch is cleared.
- Implementation should touch:
  - `examples/neatenstein/browser-entry/host/game/waves.ts` (rework `spawnWaveTick` batch gate)
  - `examples/neatenstein/browser-entry/host/waves.ts` (decide whether `advanceWave` resets `spawnCount` or uses a live-count cap)
  - `examples/neatenstein/browser-entry/worker/display.worker.ts` (likely no change, unless the wave-clear trigger needs adjustment for continuous spawning)

## Risks

- **Test churn:** `host/game/waves.test.ts` currently expects batch gating. Changing to continuous capped spawning will require rewriting those tests.
- **Wave UI drift:** `examples/neatenstein/browser-entry/browser-entry.ts` derives the on-screen wave number from `spawnCount`. If `spawnCount` is reset or its semantics change, the UI calculation must be updated to avoid showing wave 0 or an incorrect wave number.
- **Index alignment:** Dead enemies are intentionally kept in arrays for index alignment. Any new live-count cap must use the same `health > 0 && active !== false` definition that `hasAliveEnemies` and `allEnemiesCleared` use.
- **Scope creep:** The symptom is confined to the two wave modules and the display worker, so the stop condition in the plan is not triggered. No redesign of the eval worker or fire gate is needed.

## 03-los — `findNearestVisibleEnemy` returns enemies behind walls

### Question

Why does `findNearestVisibleEnemy` return enemies behind walls in slice 03-los of `plans/neatenstein-hunter-bugfix.plans.md`?

### Evidence

- `findNearestVisibleEnemy` (`examples/neatenstein/scripts/enemy-navigation.ts:382`) filters enemies by `VISION_RANGE_CELLS` and then calls `hasLineOfSight(from, to, flatMap, mapSize)`.
- `hasLineOfSight` (`examples/neatenstein/browser-entry/renderer/raycast.ts:226`) normalizes the direction vector, runs `castRayDDAFromFlatMap`, and returns:
  ```ts
  return wallDist >= dist;
  ```
- `castRayDDAFromFlatMap` returns `perpWallDist`, which equals the Euclidean distance to the first wall face because the direction is normalized.
- **Boundary-tie false-positive:** when an enemy center lies exactly on the first wall face, `wallDist === dist`. The `>=` comparison then reports the enemy as visible.
  - Concrete example: player at `(3.5, 3.5)`, wall column at `x = 4`, enemy at `(4.0, 3.5)`. The ray hits the wall face at distance `0.5`; the target distance is also `0.5`; `0.5 >= 0.5` evaluates to `true`.
- Both the champion-network path (`extractSensors` → `findNearestVisibleEnemy`) and the fallback-AI path in `display.worker.ts` use the same live `gameState.map` / module-level `wallMap` built from the same seed. No stale-grid or coordinate-system mismatch was found.
- The DDA step cap of `NEATENSTEIN_RENDER_DISTANCE_CAP = 30` is not a factor within `VISION_RANGE_CELLS = 15`.
- Existing targeted tests for "enemy behind a wall" pass; none exercise the exact wall-face boundary.

### Decision

The high-confidence root cause is the inclusive comparison in `hasLineOfSight`. A target whose center is on or beyond the first wall face should not be treated as visible. The minimal fix is to change:

```ts
return wallDist >= dist;
```

to:

```ts
return wallDist > dist;
```

at `examples/neatenstein/browser-entry/renderer/raycast.ts:252`.

### Risks

- No consumer was found that depends on "target exactly at wall face is visible", but changing the boundary semantics could affect other callers of `hasLineOfSight` if any exist.
- The integration-level `display.worker.test.ts` tests mock `findNearestVisibleEnemy`, so a real end-to-end LOS test is missing. Adding one may require a deterministic wall seed or a test-only hook to set `wallMap` directly.
- If the user's observed symptom involves enemies clearly *behind* a wall (not exactly on the face), there may be an additional reproduction not yet captured; the boundary tie is the only concrete false-positive identified under current code and tests.

## 03-los wall-vision — `extractSensors` sensor source audit

### Question

Does `extractSensors` in `enemy-navigation.ts` read raw `state.enemies` directly, or does it use `findNearestVisibleEnemy`? Are the `enemyVisible` flag and enemy position/health/bearing sensors driven by a vision-filtered (range + line-of-sight) source, or by an omniscient source?

### Evidence

- `extractSensors` (`examples/neatenstein/scripts/enemy-navigation.ts:448`) does **not** iterate `gameState.enemies`.
- It obtains the enemy target in a single call:
  ```ts
  const visibleEnemy = findNearestVisibleEnemy(gameState, flatMap, mapSize);
  ```
  (`examples/neatenstein/scripts/enemy-navigation.ts:466`).
- All enemy-derived sensor values are gated by the `if (visibleEnemy !== null)` block (`examples/neatenstein/scripts/enemy-navigation.ts:467`):
  - `sensors[5]` — nearest visible enemy bearing (`examples/neatenstein/scripts/enemy-navigation.ts:474`).
  - `sensors[6]` — nearest visible enemy distance (`examples/neatenstein/scripts/enemy-navigation.ts:475`).
  - `sensors[7]` — nearest visible enemy health ratio (`examples/neatenstein/scripts/enemy-navigation.ts:476-480`).
  - `sensors[12]` — `enemyVisible` binary flag (`examples/neatenstein/scripts/enemy-navigation.ts:482`).
  - `sensors[13]` — `enemyInFiringArc` binary flag (`examples/neatenstein/scripts/enemy-navigation.ts:483`).
- `findNearestVisibleEnemy` (`examples/neatenstein/scripts/enemy-navigation.ts:382`) filters active enemies by:
  1. Euclidean distance `<= VISION_RANGE_CELLS` (`examples/neatenstein/scripts/enemy-navigation.ts:398`).
  2. `hasLineOfSight(flatMap, mapSize, p.position, e.position)` (`examples/neatenstein/scripts/enemy-navigation.ts:400`).
- The fallback-AI target selection in `display.worker.ts` (`examples/neatenstein/browser-entry/worker/display.worker.ts:1532`) also calls `findNearestVisibleEnemy`, using the same vision-filtered source.
- `extractSensors` is consumed by `main-runner.ts` (`examples/neatenstein/browser-entry/harness/main-runner.ts:352`) and `display.worker.ts` (`examples/neatenstein/browser-entry/worker/display.worker.ts:1462`) without any post-processing that would reintroduce omniscient enemy data.

### Decision

- `extractSensors` **uses `findNearestVisibleEnemy`, not raw `state.enemies`**.
- The `enemyVisible` flag and all enemy-derived sensors (position, bearing, distance, health, firing-arc) come from the **vision-filtered** source: only enemies within `VISION_RANGE_CELLS` and with a clear `hasLineOfSight` are exposed to the network.
- There is **no omniscient enemy feed** in the sensor path.
- The wall-vision bug described in the step objective cannot be caused by `extractSensors` reading raw enemies; the remaining suspect is the boundary semantics of `hasLineOfSight` (the `>=` vs `>` tie case already documented above) or stale/incorrect `flatMap` data.

### Risks

- If `findNearestVisibleEnemy` is mocked in tests, the sensor path may not exercise real line-of-sight logic. `display.worker.test.ts` and `enemy-navigation.test.ts` mock `findNearestVisibleEnemy` in several places, so end-to-end wall-vision coverage depends on tests that supply real `flatMap` values.
- `extractSensors` initializes all 15 sensors to `0` and only populates enemy fields when a visible enemy exists; this is safe, but any future change that reads `gameState.enemies` directly inside `extractSensors` would silently reintroduce omniscience.
- The vision-filtered contract is implicit in the code and JSDoc but is not enforced by a type-level boundary. A regression could be caught only by tests, not by the compiler.

## 03-los wall-vision — renderer sprite visibility vs hunter enemy visibility

### Question

Does the renderer's sprite visibility use the same `wallMap` as the hunter's enemy visibility? Is there a stale `wallMap` or map-reference mismatch that could cause the hero to "see" enemies through walls?

### Evidence

1. **Both vision paths receive the same module-level `wallMap` in `display.worker.ts`.**
   - `display.worker.ts` declares a single module-level `let wallMap: Uint8Array | null = null` (`examples/neatenstein/browser-entry/worker/display.worker.ts:145`).
   - It is built exactly once in the `init` handler:
     ```ts
     wallMap = buildNeatensteinMap(seed);
     collisionMap = createCollisionMap(wallMap, NEATENSTEIN_MAP_SIZE);
     ```
     (`examples/neatenstein/browser-entry/worker/display.worker.ts:1711-1712`).
   - The wall raycaster consumes it via `castColumnRay`:
     ```ts
     return castRayDDAFromFlatMap(
       wallMap!,
       NEATENSTEIN_MAP_SIZE,
       ...
     );
     ```
     (`examples/neatenstein/browser-entry/worker/display.worker.ts:592-599`).
   - The AI path consumes the same variable when building the champion-network tick input:
     ```ts
     tickInput = buildAutoTickInput(
       championMainNetwork,
       gameState,
       wallMap,
       NEATENSTEIN_MAP_SIZE,
     );
     ```
     (`examples/neatenstein/browser-entry/worker/display.worker.ts:1775-1780`).
   - `buildAutoTickInput` calls `extractSensors(state, flatMap, mapSize)` (`examples/neatenstein/browser-entry/worker/display.worker.ts:1462`), and `extractSensors` calls `findNearestVisibleEnemy(gameState, flatMap, mapSize)` (`examples/neatenstein/scripts/enemy-navigation.ts:466`).

2. **Renderer sprite visibility is z-buffer-based, not a direct wallMap read.**
   - The renderer fills `zBuffer[column]` with `perpWallDist` from `castColumnRay` for every column (`examples/neatenstein/browser-entry/worker/display.worker.ts:720-740`).
   - `clipNeatensteinSprite` in `sprites.ts` clips each sprite span against that `zBuffer`:
     ```ts
     const clip = clipNeatensteinSpriteSpan(
       zBuffer,
       projection.left,
       projection.right,
       projection.perpDist,
     );
     ```
     (`examples/neatenstein/browser-entry/renderer/sprites.ts:809-814`).
   - A sprite column is visible only when its perpendicular distance is less than the wall distance stored for that screen column. This is the classic "painter's algorithm" wall-occlusion test, derived from the same `wallMap` but expressed as a depth buffer.

3. **Hunter enemy visibility is a direct LOS raycast against `wallMap`.**
   - `findNearestVisibleEnemy` filters active enemies by range and then calls `hasLineOfSight(flatMap, mapSize, p.position, e.position)` (`examples/neatenstein/scripts/enemy-navigation.ts:400`).
   - `hasLineOfSight` normalizes the direction, calls `castRayDDAFromFlatMap`, and returns `wallDist >= dist` (`examples/neatenstein/browser-entry/renderer/raycast.ts:226-253`).
   - The boundary-tie false-positive (`>=` vs `>`) is documented in the previous section: a target whose center lies exactly on the first wall face is reported as visible.

4. **No map-reference mismatch was found inside the worker session.**
   - There is one canonical `wallMap` variable per worker and one canonical `collisionMap` derived from it.
   - `collisionMap` wraps the same flat grid (`examples/neatenstein/browser-entry/renderer/map.ts:281-299`), so enemy navigation, combat caches, and raycasting all operate on the same wall data for a given seed.

5. **A stale `wallMap` risk exists if the host changes `mapSeed` after `init`.**
   - `display.worker.ts` builds `wallMap` only on the `init` message; the `simState` handler never rebuilds it.
   - The `init` seed comes from `data.mapSeed`, while `latestState` also carries a `mapSeed` field, but `simState` only updates `latestState` and does not call `buildNeatensteinMap` again.
   - If the host ever sends a different `mapSeed` in `simState`, the renderer's z-buffer and the AI's `hasLineOfSight` will both use the stale original map, so they will remain consistent with each other but wrong relative to the host's current seed.

### Decision

- The renderer and the hunter AI **do use the same `wallMap`** within a `display.worker.ts` session. There is no separate renderer wall grid vs AI wall grid.
- The two systems express visibility differently (z-buffer clipping for sprites, LOS raycast for enemies), but both derive from the same flat map.
- There is **no evidence** of a map-reference mismatch causing "hero sees enemies through walls" in the current code path.
- The remaining concrete wall-vision bug is the inclusive comparison in `hasLineOfSight` (`>=`), not a stale or mismatched `wallMap`.
- The **stale-map risk** is latent: if the host supports mapSeed mutation after init, `display.worker.ts` should rebuild `wallMap` and `collisionMap` on any `simState` whose `mapSeed` differs from the one used at init. That is outside the scope of the current bug unless reproduction confirms seed changes.

### Risks

- **Z-buffer vs LOS semantic drift:** Sprite clipping uses per-column perpendicular wall distance, while `hasLineOfSight` uses Euclidean ray length after normalizing the direction. These are mathematically equivalent for axis-aligned wall faces and normalized rays, but any future change to one formula (e.g., adding a bias or a different distance cap) could make sprites and AI disagree on visibility even with the same `wallMap`.
- **Stale `wallMap` on seed change:** If `mapSeed` is ever changed at runtime, both renderer and AI will silently use the old map. Fixing this requires rebuilding `wallMap`/`collisionMap`/`gameState` in the `simState` handler when `latestState.mapSeed !== initSeed`, which is a non-trivial lifecycle change.
- **Boundary-tie still the active suspect:** The `>=` comparison remains the most concrete, source-grounded explanation for a target exactly at a wall face being treated as visible. Any broader "through walls" symptom needs a separate reproduction that is not explained by the boundary tie alone.

## 03-los wall-vision — DDA traversal and coordinate-transform audit

### Question

Is `castRayDDAFromFlatMap` in `examples/neatenstein/browser-entry/renderer/raycast.ts` a correct DDA grid traversal implementation? Can rays skip walls because of coordinate-transform bugs, missing bounds checks, or non-axis-aligned edge cases?

### Evidence

1. **DDA core is a classic Wolfenstein-style grid walker.**
   - The function starts in the cell `mapX = Math.floor(posX)`, `mapY = Math.floor(posY)` (`examples/neatenstein/browser-entry/renderer/raycast.ts:157-158`).
   - Step signs are derived from `Math.sign(dirX)` / `Math.sign(dirY)` (`examples/neatenstein/browser-entry/renderer/raycast.ts:159-160`).
   - `deltaDistX` / `deltaDistY` are `sqrt(1 + (dirY/dirX)^2)` and `sqrt(1 + (dirX/dirY)^2)`, which are the reciprocal absolute direction components for a unit-length ray (`examples/neatenstein/browser-entry/renderer/raycast.ts:163-169`).
   - `sideDistX` / `sideDistY` are initialized to the parametric distance to the first crossed grid line in each axis (`examples/neatenstein/browser-entry/renderer/raycast.ts:170-185`).
   - The loop advances the nearest axis (`if (sideDistX < sideDistY)`) and checks the newly-entered cell for a wall (`examples/neatenstein/browser-entry/renderer/raycast.ts:187-207`).
   - This matches the canonical DDA algorithm and is correct for solid 1×1 wall cells.

2. **World/grid coordinate transform is consistent.**
   - World units and grid units are 1:1: each cell occupies `[floor(x), floor(x)+1) × [floor(y), floor(y)+1)`.
   - The flat map is row-major: `flatMap[mapY * mapSize + mapX]` (`examples/neatenstein/browser-entry/renderer/raycast.ts:201`).
   - `buildNeatensteinMap` and `createCollisionMap` use the same row-major convention (`examples/neatenstein/browser-entry/renderer/map.ts:128-130`, `examples/neatenstein/browser-entry/renderer/map.ts:281-299`).
   - No coordinate-scale mismatch was found between the raycaster, the collision map, and the game state positions.

3. **`hasLineOfSight` correctly normalizes the ray direction.**
   - It computes `dist = Math.hypot(dx, dy)` and then `dirX = dx / dist`, `dirY = dy / dist` (`examples/neatenstein/browser-entry/renderer/raycast.ts:234-238`).
   - Because the direction is unit length, the DDA's parametric distance equals the Euclidean ray distance, so comparing `perpWallDist` against `dist` is geometrically valid.

4. **No reproducible wall-skipping was found in constructed corner cases.**
   - A temporary runtime audit script (deleted after findings were materialized) exercised the actual `castRayDDAFromFlatMap` implementation via `tsx` with scenarios for:
     - Diagonal thin-wall tunneling (two single-cell walls arranged diagonally with an open gap).
     - Origin exactly on a grid boundary.
     - Concave-corner tie-breaks where `sideDistX === sideDistY`.
     - Closed-grid sanity (ray inside a sealed box returns finite wall distance).
   - In all constructed cases the ray stopped at a wall rather than passing through it.
   - The map generator's `INTERIOR_WALL_DENSITY = 0.12` scatter pass can produce single-cell walls and diagonal arrangements in principle, but the probability of a ray exactly aligning with the required diagonal gap is low for continuous player/enemy positions.

5. **Boundary-tie false-positive is the concrete wall-vision bug.**
   - `hasLineOfSight` returns `wallDist >= dist` (`examples/neatenstein/browser-entry/renderer/raycast.ts:252`).
   - When a target lies exactly on the first wall face, `wallDist === dist`, so the target is reported as visible.
   - Runtime reproduction: player `(0.5, 0.5)`, wall at cell `(1,0)`, target `(1.0, 0.5)`:
     - `hasLineOfSight(...) === true` (bug).
     - Target `(1.0000001, 0.5)` returns `false` (correct).
   - This is the only concrete, source-grounded false-positive identified; it is not a DDA wall-skip but a visibility-classification error at the equality boundary.

6. **Out-of-bounds read risk if perimeter precondition is violated.**
   - `castRayDDAFromFlatMap` does not check `mapX` / `mapY` against `[0, mapSize)` before indexing `flatMap`.
   - If a caller passes a map without a sealed perimeter, the ray walks off the grid and reads out-of-bounds indices until the `NEATENSTEIN_RENDER_DISTANCE_CAP` step limit is reached.
   - Runtime reproduction on an all-open 4×4 grid:
     - Ray from `(1.5, 1.5)` in `+x` returned `mapX = 12`, well beyond the 0–3 valid range.
     - `Uint8Array` out-of-bounds reads return `0`, so the function silently returns a bogus result instead of throwing.
   - In the Neatenstein demo this is mitigated by `buildNeatensteinMap`, which always writes a solid perimeter, but the primitive itself is not defensive.

7. **Tie-break is deterministic but potentially surprising.**
   - The comparison `if (sideDistX < sideDistY)` means that when both axes are equidistant, the Y-axis step is always taken first (`examples/neatenstein/browser-entry/renderer/raycast.ts:193`).
   - This determines which wall face is reported at exact grid corners; it does not, by itself, cause wall skipping in the tested cases.

### Decision

- The DDA traversal in `castRayDDAFromFlatMap` is correct for the intended solid-cell grid model.
- Rays do **not** skip walls under the normal generated-map constraints (sealed perimeter, continuous positions).
- The actual wall-vision bug is in `hasLineOfSight` (`>=` should be `>`), not in the DDA traversal.
- The raycast primitive should remain unchanged for the wall-vision slice; the fix is a one-character comparison change in `hasLineOfSight`.
- Optionally, add defensive bounds checking to `castRayDDAFromFlatMap` in a separate hardening slice, because the current implicit reliance on a sealed perimeter is a latent reliability risk.

### Risks

- **Out-of-bounds reads on malformed input:** Any future caller that builds a map without a sealed perimeter can get undefined behavior. Tests or documentation should reinforce the precondition.
- **Boundary-tie fix could mask a different bug:** If the observed "through walls" symptom happens for enemies clearly behind a wall (not exactly on the face), the `>` change alone will not fix it and a new reproduction will be required.
- **Tie-break semantics:** Changing `<` to `<=` or vice-versa in the DDA would alter which wall face is reported at corners. This is not required for the wall-vision slice, but any future change should be regression-tested.
- **Test gaps:** Existing raycast tests do not cover diagonal thin-wall configurations, exact grid-boundary origins, or maps without perimeter walls. These gaps are acceptable for the current slice but should be closed if the primitive is promoted to a shared utility.
