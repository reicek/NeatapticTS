# Neatenstein Auto-Mode Firing & Sensor System Research

## Question

How should the Neatenstein auto-mode firing and sensor system be redesigned to address:
1. Auto-fires non-stop (should fire only when enemy visible)
2. Hero has full awareness (should have limited vision range)
3. Need wall detection via path tracing (line-of-sight)
4. Reward/penalty design for landed/missed shots
5. Shot direction awareness

## Evidence

### Current Architecture (source: static-code)

#### Auto-Mode Firing — Two Paths

**Path A: Champion NEAT network** (`display.worker.ts:1425-1435`)
- `buildAutoTickInput()` calls `extractSensors()` → `network.activate(sensors)` → `networkOutputToTickInput(raw)`
- `networkOutputToTickInput` maps: `fire = outputs[3] > 0` — **any positive activation fires**
- The worker-side fitness function (`evaluateArmsRaceGeneration:1319-1328`) is a **PLACEHOLDER**: activates with zero inputs, returns `output[0]` — the network has NO selective pressure for judicious firing
- The headless evaluation in `main-runner.ts:runEpisode()` (line 364-403) DOES play real episodes with `extractSensors` + `gameTick`, but the worker's live evolution path uses the placeholder

**Path B: Fallback AI** (`display.worker.ts:1469-1525`)
- Already has fire gating: fires only when enemy bearing is within ±30° (`NEATENSTEIN_FALLBACK_FIRE_ARC`) AND on cooldown (every 25 ticks / `NEATENSTEIN_FALLBACK_FIRE_INTERVAL`)
- This path is reasonable but is only used before the first champion network is produced

#### Sensor System (`scripts/enemy-navigation.ts:extractSensors`, line 379-432)

12-element sensor vector:
- [0] player health ratio (health/maxHealth, clamped [0,1])
- [1] player ammo (raw count)
- [2] player look angle (radians)
- [3] player position X (world units)
- [4] player position Y (world units)
- [5] nearest enemy relative bearing (radians, normalized [-π,π]) — **ALWAYS provided regardless of walls**
- [6] nearest enemy Euclidean distance — **ALWAYS provided regardless of walls**
- [7] nearest enemy health — **ALWAYS provided regardless of walls**
- [8-11] wall raycast distances in N, E, S, W cardinal directions (via `castRayDDAFromFlatMap`)

**Critical gap**: Sensors [5]-[7] provide full enemy awareness with NO line-of-sight check. The hero knows exactly where every enemy is, even through walls. There is no vision range limit.

#### Combat / Firing (`host/game/combat.ts:fireBolt`, line 163-309)

- Firing checks ammo, casts a DDA ray to find the nearest wall, then tests all living enemies against the bolt path cylinder
- `hitType` is classified as 'wall', 'enemy', or 'range'
- Telemetry tracks: `shotsFired`, `shotsHit`, `damageDealt`, `aimMissRate`
- `aimMissRate = (shotsFired - shotsHit) / shotsFired`

#### Fitness Composite (`harness/fitness.ts:computeCombatQualitySignal`, line 94-125)

```
fitness = survivalTicks * 1
        + damageDealt * 2
        + kills * 5
        - damageTaken * 1
        - aimMissRate * 1
        + complexityBonus * 0.1
        - parsimonyDensityPenalty * 0.01
```

#### Existing DDA Raycast (`renderer/raycast.ts:castRayDDAFromFlatMap`)

Grid DDA traversal that finds the first wall cell along a ray. Returns `{ perpWallDist, side, mapX, mapY }`. Already used for:
- Combat wall-hit detection (`combat.ts:179`)
- 4 cardinal wall sensors (`enemy-navigation.ts:418-428`)
- Renderer column casting

This primitive can be reused for line-of-sight checks.

### Confidence Assessment

| Finding | Confidence | Provenance |
|---------|-----------|------------|
| Champion NEAT fires non-stop due to `output[3] > 0` threshold + placeholder fitness | 0.92 | static-code (display.worker.ts:1394-1408, 1319-1328) |
| Hero gets full enemy awareness through walls (no LOS check) | 0.95 | static-code (enemy-navigation.ts:395-415) |
| DDA raycast primitive available for LOS checks | 0.95 | static-code (raycast.ts, combat.ts:179) |
| Fitness penalizes miss rate but doesn't reward accuracy directly | 0.90 | static-code (fitness.ts:94-125, harness/constants.ts:148) |
| No per-tick shot feedback in sensor vector | 0.93 | static-code (enemy-navigation.ts:379-432) |

## Decision

### Solution 1: Conditional Firing — Fire Only When Enemy Visible

**Problem**: `networkOutputToTickInput` maps `fire = outputs[3] > 0`, and the NEAT network has no selective pressure to not fire because the worker's fitness function is a placeholder.

**Two-layer approach:**

**Layer A — Hard gate in `networkOutputToTickInput`** (immediate fix):
Add a line-of-sight visibility check before allowing fire. When `humanMode === 'auto'`, override `fire: false` if no enemy is visible (within vision range AND no wall between player and nearest enemy).

```typescript
// In buildAutoTickInput or networkOutputToTickInput:
const visibleEnemy = findNearestVisibleEnemy(state, flatMap, mapSize, VISION_RANGE);
const canFire = visibleEnemy !== null;
return {
  ...networkOutputToTickInput(raw),
  fire: canFire && rawOutput[3] > 0,  // gate fire on visibility
};
```

**Layer B — Fix the fitness function** (evolutionary fix):
The worker's `evaluateArmsRaceGeneration` must use real episode-based fitness (the `runEpisode` function from `main-runner.ts` already does this). The fitness composite already penalizes `aimMissRate`, but the weight (1) is too low relative to `kills` (5) and `damageDealt` (2). Firing into walls wastes ammo and inflates `aimMissRate`, but the penalty is weak.

**Tradeoffs:**
| Approach | Pro | Con |
|----------|-----|-----|
| Hard gate (Layer A) | Immediate effect, no evolution needed, conserves ammo | Network doesn't learn WHEN to fire, only gets fire suppressed; may feel mechanical |
| Fitness fix only (Layer B) | Network learns selective firing organically | Slow to converge; many generations of waste before pressure selects for judicious firing |
| Both layers | Immediate fix + evolutionary pressure toward selective firing | Slightly more code; the hard gate may mask fitness signal (if fire is always gated, aimMissRate stays 0 and doesn't differentiate variants) |

**Recommendation**: Both layers, but make the hard gate a **soft gate** — only suppress fire when NO enemy is visible at all. When an enemy IS visible but the network chooses not to fire, that's the network's decision and the fitness function will reward/punish accordingly. This preserves evolutionary signal while preventing the worst case (firing at walls with no enemy anywhere).

---

### Solution 2: Limited Vision Range

**Problem**: Sensors [5]-[7] always provide the nearest enemy's bearing, distance, and health, regardless of walls or distance. The hero has omniscient awareness.

**Approach**: Add a `VISION_RANGE_CELLS` constant (e.g., 15 cells) and modify `extractSensors` to zero-out enemy sensors when:
1. The nearest enemy is beyond `VISION_RANGE_CELLS`, OR
2. There is a wall between the player and the nearest enemy (no line-of-sight)

```typescript
// In extractSensors, after finding nearestEnemy:
const distToEnemy = nearestDist;
const inRange = distToEnemy <= VISION_RANGE_CELLS;
const hasLineOfSight = inRange
  ? checkLineOfSight(flatMap, mapSize, p.position, nearestEnemy.position)
  : false;

if (inRange && hasLineOfSight) {
  sensors[5] = bearing;  // relative bearing
  sensors[6] = nearestDist;
  sensors[7] = nearestEnemy.health;
} else {
  sensors[5] = 0;  // no enemy visible
  sensors[6] = 0;
  sensors[7] = 0;
}
```

Additionally, add a **binary "enemy visible" flag** as sensor [12] (extending the vector from 12 to 13 inputs) so the network can distinguish "no enemy exists" from "enemy exists but is out of sight":

```typescript
sensors[12] = (inRange && hasLineOfSight) ? 1.0 : 0.0;
```

**Tradeoffs:**
| Vision Range | Pro | Con |
|-------------|-----|-----|
| Small (e.g., 8 cells) | Strong evolutionary pressure for exploration; realistic | Hero may be overwhelmed by enemies that approach from outside vision; episodes end fast |
| Medium (e.g., 15 cells) | Balanced — enough to react but not omniscient | Still a hard cutoff; enemies appear/disappear at boundary |
| Large (e.g., 30 cells, = bolt range) | Nearly current behavior with LOS check only | Weakens the "limited awareness" goal |
| No range limit, LOS only | Hero still sees far through open corridors | Better than current (no LOS) but doesn't limit awareness in open areas |

**Recommendation**: `VISION_RANGE_CELLS = 15` (half the bolt max range of 30) with LOS check. This forces the hero to actively explore and turn corners to find enemies, creating meaningful navigation behavior. The binary "enemy visible" flag (sensor 12) is essential for the network to distinguish "search" from "engage" states.

**Input count change**: 12 → 13 inputs. This is a **breaking change** for existing networks — the NEAT population must be re-seeded. Since the worker's evolution uses `new Neat(NEATENSTEIN_MAIN_NEAT_INPUTS, ...)` and the constant is `12`, changing to `13` requires updating both `main-runner.ts` and `display.worker.ts`. Existing champion networks stored in `championMainNetwork` will be incompatible and must be cleared.

---

### Solution 3: Wall Detection via Path Tracing (Line-of-Sight)

**Problem**: No line-of-sight check exists between the player and enemies. The hero sees through walls.

**Approach**: Reuse the existing `castRayDDAFromFlatMap` DDA primitive to cast a ray from the player position toward the enemy. If the wall hit distance is less than the enemy distance, the enemy is occluded.

```typescript
/**
 * Check whether there is a clear line of sight from `from` to `to`.
 * Casts a DDA ray from `from` toward `to` and returns true if no wall
 * is hit before reaching `to`.
 */
function hasLineOfSight(
  flatMap: Uint8Array,
  mapSize: number,
  from: Vector2,
  to: Vector2,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return true;

  const dirX = dx / dist;
  const dirY = dy / dist;

  const hit = castRayDDAFromFlatMap(flatMap, mapSize, from.x, from.y, dirX, dirY);
  const wallDist = Number.isFinite(hit.perpWallDist) ? hit.perpWallDist : Infinity;

  return wallDist >= dist;  // wall is beyond the enemy → clear LOS
}
```

**Performance**: The DDA raycast is O(grid_cells_traversed) — typically 5-30 cells per call. Running this once per tick (for the nearest enemy only) is negligible compared to the renderer which casts hundreds of rays per frame.

**Tradeoffs:**
| Approach | Pro | Con |
|----------|-----|-----|
| Single DDA ray to nearest enemy (proposed) | Simple, fast, reuses existing primitive | Only checks center-to-center; corner cases where enemy is partially visible around a wall edge |
| Multi-ray cone cast (3-5 rays spread across enemy radius) | More accurate LOS — catches partial visibility | 3-5x cost per tick; marginal benefit for a grid-based game |
| BFS flood-fill visibility (compute all visible cells) | Complete visibility map per tick | O(N²) per tick — far too expensive for real-time |
| Raycast + corner clipping | Pixel-perfect LOS | Over-engineered for a grid DDA game |

**Recommendation**: Single DDA ray to nearest enemy. The grid-based DDA is accurate enough for gameplay purposes. Corner cases (enemy partially visible around a wall edge) are acceptable — the enemy radius is ~0.38 cells, and the DDA ray traces the center-to-center line.

---

### Solution 4: Reward/Penalty Design for Landed/Missed Shots

**Problem**: The fitness composite penalizes `aimMissRate` (weight 1) but this is weak relative to `kills` (weight 5) and `damageDealt` (weight 2). There is no direct reward for shot accuracy, and no penalty for firing when no enemy is visible.

**Current telemetry** (`EpisodeTelemetry`):
- `shotsFired` — total bolts fired
- `shotsHit` — bolts that struck an enemy
- `damageDealt` — cumulative damage
- `aimMissRate = (shotsFired - shotsHit) / shotsFired`

**Proposed changes:**

1. **Add `shotsWasted` telemetry** — shots fired when no enemy was visible (no LOS or out of range). This distinguishes "aimed but missed" from "fired blindly."

2. **Add `ammoEfficiency` metric** — `damageDealt / shotsFired` (damage per shot). Higher is better.

3. **Revise fitness weights:**

```
fitness = survivalTicks * 1
        + damageDealt * 2
        + kills * 5
        - damageTaken * 1
        - aimMissRate * 3          // INCREASE from 1 to 3
        - shotsWasted * 0.5         // NEW: penalty for blind fire
        + ammoEfficiency * 2       // NEW: reward for damage-per-shot efficiency
        + complexityBonus * 0.1
        - parsimonyDensityPenalty * 0.01
```

4. **Add per-tick shaping reward** (optional, for the headless evaluation only):
   - Small positive reward (+0.1) when the network fires AND an enemy is visible
   - Small negative reward (-0.2) when the network fires AND no enemy is visible
   - This provides immediate per-tick feedback rather than only end-of-episode

**Tradeoffs:**
| Approach | Pro | Con |
|----------|-----|-----|
| Increase `aimMissRate` weight only | Simplest change — one constant | Doesn't distinguish "bad aim" from "blind fire"; both inflate miss rate |
| Add `shotsWasted` + `ammoEfficiency` | Richer signal — network learns WHEN and HOW WELL to fire | New telemetry fields require changes to combat.ts, types.ts, fitness.ts, constants.ts |
| Per-tick shaping reward | Fastest learning signal — immediate feedback | Can dominate the fitness landscape if weights are too high; may create reward hacking (e.g., network fires only when it's sure to hit, never takes risky shots) |
| End-of-episode only (current + new metrics) | No reward hacking risk; aggregate is stable | Slower convergence; less granular feedback |

**Recommendation**: Add `shotsWasted` telemetry + `ammoEfficiency` metric, increase `aimMissRate` weight from 1 to 3, add `shotsWasted * 0.5` penalty. Skip per-tick shaping for now — the end-of-episode composite with the new metrics provides enough signal. If evolution stalls, add per-tick shaping as a follow-up.

---

### Solution 5: Shot Direction Awareness

**Problem**: The sensor vector has no information about:
- Whether the current facing direction has an enemy in line-of-sight
- Whether the last shot hit or missed (no feedback signal)
- The direction of the last shot relative to enemy positions

**Proposed sensor additions (extending from 13 to 16 inputs):**

- [12] `enemyVisible` — binary flag (1.0 if nearest enemy is within vision range AND has LOS, 0.0 otherwise) — already proposed in Solution 2
- [13] `enemyInFiringArc` — binary flag (1.0 if `enemyVisible` AND `|bearing| <= FIRE_ARC`, 0.0 otherwise) — tells the network "you can hit this enemy now"
- [14] `lastShotHit` — binary flag (1.0 if the previous tick's shot connected, 0.0 if it missed or no shot was fired) — immediate feedback on shot quality
- [15] `enemyBearingFromAim` — the nearest visible enemy's bearing relative to the player's aim angle, normalized to [-1, 1] (atan2(bearing) / π) — continuous aim error signal

**Tradeoffs:**
| Sensor Design | Pro | Con |
|---------------|-----|-----|
| Minimal (just `enemyVisible` flag, 13 inputs) | Smallest network; fastest evolution | Network still can't distinguish "enemy visible but not in firing arc" from "enemy visible and in arc" |
| Medium (flags + last-shot feedback, 14-15 inputs) | Network gets aim quality feedback; can learn to correct | More inputs = more dimensions for NEAT to explore; slower convergence |
| Full (flags + bearing + last-shot, 16 inputs) | Richest signal; network can learn precise aim correction | 16 inputs may be too many for a 64-node network budget; risk of overfitting |
| Recurrent memory (LSTM-like via GatedRecurrentCell motif) | Network can remember shot history | Already in the motif allowlist but adds complexity; may not converge in browser-time generations |

**Recommendation**: Add sensors [12]-[14] for a total of 15 inputs (drop the continuous `enemyBearingFromAim` since sensor [5] already provides the raw bearing — the network can compute the arc membership itself). The `lastShotHit` flag is the most valuable addition because it provides direct shot-quality feedback that the current sensor vector completely lacks.

**Implementation**: Add `lastShotHit` tracking to `GameState` or `EpisodeTelemetry` as a per-tick flag that the tick function updates after `fireBolt` resolves. The `extractSensors` function reads it and includes it in the sensor vector. Clear it on episode reset.

---

### Summary of Changes

| Change | Files | Complexity |
|--------|-------|------------|
| LOS check for enemy sensors | `scripts/enemy-navigation.ts` | Low — reuse `castRayDDAFromFlatMap` |
| Vision range constant | `host/game/constants.ts`, `scripts/enemy-navigation.ts` | Trivial |
| Expand sensor vector 12→15 | `scripts/enemy-navigation.ts`, `harness/main-runner.ts`, `display.worker.ts` | Medium — input count change breaks existing networks |
| Fire gating on visibility | `display.worker.ts` | Low — one conditional |
| `shotsWasted` telemetry | `host/game/combat.ts`, `host/game/types.ts` | Medium — new counter + tracking |
| `ammoEfficiency` metric | `harness/fitness.ts` | Low — derived from existing counters |
| Revised fitness weights | `harness/constants.ts` | Trivial — constant changes |
| `lastShotHit` flag | `host/game/types.ts`, `host/game/tick.ts`, `scripts/enemy-navigation.ts` | Medium — per-tick state tracking |
| Fix worker fitness function | `display.worker.ts` | High — replace placeholder with real episode evaluation (already exists in `main-runner.ts`) |

**Network input count**: 12 → 15. This is a breaking change for any persisted champion network. The worker must clear `championMainNetwork` when the input count changes. The NEAT population is re-seeded with the new input count on the next generation.

**Determinism**: All proposed changes preserve the Level 2 ordered determinism contract — same seed + same genome + same fixed timestep → same trajectory. The LOS check is deterministic (DDA is a deterministic grid traversal), the vision range is a constant, and the telemetry additions are deterministic counters.

## Risks

- **Network re-seeding**: Changing from 12 to 15 inputs invalidates all existing champion networks. The auto-mode player will fall back to the fallback AI until the next evolution cycle produces a new champion. This is acceptable — the fallback AI is already designed for this case.
- **Evolution convergence**: Adding 3 new sensors expands the search space. The 64-node / 256-edge topology budget may not be enough to learn the richer sensor mapping. If convergence stalls, consider increasing the budget or adding more generations before advancing.
- **Performance**: LOS check adds one DDA raycast per tick (nearest enemy only). This is negligible — the renderer casts hundreds of rays per frame. The additional telemetry counters are O(1) per shot.
- **Fitness weight tuning**: The proposed weight changes (aimMissRate 1→3, shotsWasted 0.5, ammoEfficiency 2) are initial estimates. They may need tuning after observing evolution behavior. A parameterized fitness config (already supported via `EnemyTeamFitnessConfig`) could be extended to the main agent.
- **Sensor normalization**: The new sensors (`enemyVisible`, `enemyInFiringArc`, `lastShotHit`) are binary [0,1] — consistent with existing sensor normalization. No additional scaling needed.
- **`lastShotHit` timing**: The flag must be set DURING the tick (after `fireBolt` resolves) and read at the START of the NEXT tick (when `extractSensors` is called). This one-tick delay is intentional and realistic — the network observes the result of its previous action.

---

## Code-Pattern Review and Implementation Recommendations

### 1. `hasLineOfSight` — DDA Reuse Correctness and Placement

**Finding**: The proposed `hasLineOfSight` function correctly reuses `castRayDDAFromFlatMap`, but the distance comparison has a subtle correctness consideration that the current research under-specifies.

**DDA distance semantics** (verified from `raycast.ts:84-101`):
`computePerpendicularWallDistance` returns the ray parameter `t` where the wall is hit:
```typescript
side === 0 ? (mapX - posX + (1 - stepX) / 2) / dirX
           : (mapY - posY + (1 - stepY) / 2) / dirY
```
When the direction vector is **normalized** (`dirX² + dirY² = 1`), this `t` IS the Euclidean distance along the ray to the wall hit point. The proposed `hasLineOfSight` normalizes correctly:
```typescript
const dirX = dx / dist;  // normalized
const dirY = dy / dist;
```
Therefore `wallDist >= dist` (the proposed LOS check) is **mathematically sound** — `wallDist` is Euclidean distance to the first wall along the ray, and `dist` is Euclidean distance to the enemy. If the wall is farther, the enemy is visible.

**Risk**: `castRayDDAFromFlatMap` has a hard cap of `NEATENSTEIN_RENDER_DISTANCE_CAP` steps (line 197). If the enemy is farther than this cap and no wall is found, the function returns `perpWallDist = Infinity`, which correctly passes the `wallDist >= dist` check. **No issue here.**

**Recommendation — place `hasLineOfSight` in `raycast.ts`, not `enemy-navigation.ts`**:
The function is a general-purpose DDA utility, not specific to enemy navigation. `combat.ts:179` already imports `castRayDDAFromFlatMap` from `raycast.ts` for wall-hit detection. Placing `hasLineOfSight` alongside the DDA primitive keeps the module boundary clean:

```typescript
// In browser-entry/renderer/raycast.ts (new export)
export function hasLineOfSight(
  flatMap: Uint8Array,
  mapSize: number,
  from: Vector2,
  to: Vector2,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return true; // same position
  const dirX = dx / dist;
  const dirY = dy / dist;
  const hit = castRayDDAFromFlatMap(flatMap, mapSize, from.x, from.y, dirX, dirY);
  const wallDist = Number.isFinite(hit.perpWallDist) ? hit.perpWallDist : Infinity;
  return wallDist >= dist;
}
```

**Import pattern**: `enemy-navigation.ts` already imports `castRayDDAFromFlatMap` from `../browser-entry/renderer/raycast` (line 14). Adding `hasLineOfSight` to the same import is trivial. `combat.ts` would add it to its existing `import { castRayDDAFromFlatMap } from '../../renderer/raycast'` (line 19).

**Confidence**: 0.93 (static-code verification of DDA formula + existing import patterns)

---

### 2. `extractSensors` Changes — Module Boundary and Call-Site Impact

**Finding**: `extractSensors` (in `scripts/enemy-navigation.ts:379-432`) is imported by **two independent consumers** with different runtime contexts:

| Consumer | File | Context | Import Line |
|----------|------|---------|-------------|
| Headless episode runner | `harness/main-runner.ts:40` | Sync, Node-like | `import { extractSensors } from '../../scripts/enemy-navigation'` |
| Display worker | `worker/display.worker.ts` | Async, Web Worker | (imported at top, used in `buildAutoTickInput:1431`) |

**Recommendation — extract a `findNearestVisibleEnemy` helper**:
Instead of inlining LOS logic in `extractSensors`, factor out a reusable function that the fire-gating layer (Solution 1 Layer A) can also call. This avoids duplicating the nearest-enemy + LOS logic in two places:

```typescript
// In scripts/enemy-navigation.ts (new export)
export interface VisibleEnemyInfo {
  enemy: EnemyState;
  distance: number;
  bearing: number;
}

export function findNearestVisibleEnemy(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
  visionRangeCells: number,
): VisibleEnemyInfo | null {
  const p = gameState.player;
  const enemies = gameState.enemies.filter((e) => e.active !== false);
  if (enemies.length === 0) return null;

  let nearest: EnemyState | null = null;
  let nearestDist = Infinity;
  for (const e of enemies) {
    const dist = Math.hypot(e.position.x - p.position.x, e.position.y - p.position.y);
    if (dist < nearestDist) {
      nearestDist = dist;
      nearest = e;
    }
  }
  if (!nearest) return null;

  // Range check
  if (nearestDist > visionRangeCells) return null;

  // LOS check
  if (!hasLineOfSight(flatMap, mapSize, p.position, nearest.position)) return null;

  const dx = nearest.position.x - p.position.x;
  const dy = nearest.position.y - p.position.y;
  const bearing = Math.atan2(dy, dx) - p.angleRad;
  return {
    enemy: nearest,
    distance: nearestDist,
    bearing: Math.atan2(Math.sin(bearing), Math.cos(bearing)),
  };
}
```

Then `extractSensors` uses `findNearestVisibleEnemy` instead of the raw nearest-enemy loop, and `buildAutoTickInput` in `display.worker.ts` calls the same helper for fire gating. This eliminates the code duplication the research's Solution 1 implies.

**Call-site changes**:
- `extractSensors` (line 394-415): Replace the raw nearest-enemy loop with `findNearestVisibleEnemy`. Zero-out sensors [5]-[7] when null. Add sensors [12]-[14] from the result.
- `buildAutoTickInput` (line 1425-1435): After `networkOutputToTickInput`, call `findNearestVisibleEnemy` and gate `fire` on `visibleEnemy !== null && raw[3] > 0`.
- `main-runner.ts:runEpisode` (line 380): No changes needed — it calls `extractSensors`, which will now produce the expanded vector automatically.

**Confidence**: 0.90 (two confirmed import sites, both using the same signature)

---

### 3. `combat.ts` Telemetry Additions — Minimally Invasive Tracking

**Finding**: The research proposes `shotsWasted` ("shots fired when no enemy visible"). However, `fireBolt` has **no access to enemy visibility information** — it only knows about the bolt path and whether it hit a wall, enemy, or exceeded range. This would require either:
1. Passing enemy visibility info into `fireBolt` (changes the `fireBolt(state)` signature — breaking)
2. Defining `shotsWasted` differently (simpler, no signature change)

**Recommendation — redefine `shotsWasted` as "shots that hit a wall or expired at range"**:
`fireBolt` already classifies `hitType` as `'wall'`, `'enemy'`, or `'range'` (line 192-193). A shot is "wasted" when `hitType !== 'enemy'`. This is computable **without any signature change** to `fireBolt`:

```typescript
// Extended EpisodeTelemetry (in host/game/types.ts)
export interface EpisodeTelemetry {
  damageDealt: number;
  shotsFired: number;
  shotsHit: number;
  aimMissRate: number;
  shotsWasted: number;  // NEW: shots that hit wall or expired at range
}
```

```typescript
// In combat.ts fireBolt, after hitType is determined (after line 224):
nextState = {
  ...nextState,
  telemetry: withAimMissRate({
    ...telemetryBeforeShot,
    shotsFired: telemetryBeforeShot.shotsFired + 1,
    shotsWasted: telemetryBeforeShot.shotsWasted + (hitType !== 'enemy' ? 1 : 0),
    // shotsHit increment stays in applyEnemyDamage where the enemy hit is confirmed
  }),
};
```

Wait — `shotsHit` is currently incremented in `applyEnemyDamage` (via `telemetry.shotsHit`), not in `fireBolt`. Let me verify...

Actually, looking at the code flow: `fireBolt` increments `shotsFired` (line 251-254), but `shotsHit` and `damageDealt` are incremented in `applyEnemyDamage` (which is called from `tick.ts:298` when a bolt hits an enemy during bolt movement, OR from `fireBolt` itself at line 301 for immediate hits). The `withAimMissRate` function is called after `shotsFired` is incremented, but `shotsHit` is updated separately. This means `aimMissRate` is recomputed after `shotsFired` increments but may not yet reflect `shotsHit` increments from the same tick. This is a pre-existing race that the research doesn't flag.

**Recommendation — fix the `shotsHit`/`aimMissRate` sync gap**:
Currently `shotsHit` is incremented in `applyEnemyDamage`, but `aimMissRate` is recomputed in `withAimMissRate` which only runs in `fireBolt`. After `applyEnemyDamage` runs, `aimMissRate` is stale. The `withAimMissRate` call should also happen after `applyEnemyDamage`. This is a **pre-existing bug** that the new telemetry additions should fix in the same pass:

```typescript
// In applyEnemyDamage, after incrementing shotsHit and damageDealt:
nextState = {
  ...nextState,
  telemetry: withAimMissRate({
    ...nextState.telemetry!,
    shotsHit: (nextState.telemetry?.shotsHit ?? 0) + 1,
    damageDealt: (nextState.telemetry?.damageDealt ?? 0) + NEATENSTEIN_BOLT_DAMAGE,
  }),
};
```

**Confidence**: 0.88 (verified from combat.ts line 248-254 + tick.ts line 298 applyEnemyDamage call pattern)

---

### 4. Worker Fitness Function Replacement — Architecture Alignment

**Finding**: The research correctly identifies that `display.worker.ts:evaluateArmsRaceGeneration` (line 1319-1328) uses a placeholder fitness function (`network.activate(zeros) → output[0]`). The real episode-based evaluation exists in `main-runner.ts:runEpisode` (line 364-403).

**Key architecture mismatch**: `main-runner.ts` imports from `../../../../src/neat/nge-main-agent/` and `../../../../src/architecture/network/network` — these are **library-level** imports. The display worker is a **Web Worker** that runs in a browser context. The worker currently lazy-loads Neat via `await import('neataptic')` (line 1317). The `runEpisode` function imports `Network` directly from the library source, not via the `neataptic` package entry point.

**Recommendation — two-phase approach**:

**Phase 1 (immediate)**: Replace the placeholder fitness with a **simple episode-based evaluation** that runs inside the worker. The worker already has access to `extractSensors` and `gameTick` (it imports them for the live game loop). The fitness function should:
1. Create a deterministic episode from the seed
2. Run `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` ticks with `extractSensors + network.activate + gameTick`
3. Extract `CombatQualitySignal` from the final state
4. Return `computeCombatQualitySignal(signal, complexity)`

This mirrors `main-runner.ts:runEpisode` but uses the worker's already-imported `gameTick` and `extractSensors`. The `Neat` population's `fitnessFn` callback receives a `Network` instance — the worker can call `network.activate(sensors)` directly.

**Phase 2 (deferred)**: For full co-evolution parity, wire the worker's evaluation to use the same enemy snapshot as `runMainGeneration`. This requires threading the enemy snapshot through the fitness callback, which the `Neat` API may not support directly (fitness functions receive only the network). A wrapper closure can capture the snapshot.

**Implementation sketch for Phase 1**:
```typescript
// In display.worker.ts evaluateArmsRaceGeneration, replace lines 1319-1328:
const fitnessFn = (network: Network): number => {
  const episode = createEpisode({ seed: popSeed });
  let state = episode.state;
  const flatMap = buildNeatensteinMap(state.seed);
  const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
  for (let tick = 0; tick < NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS; tick++) {
    const sensors = extractSensors(state, flatMap, NEATENSTEIN_MAP_SIZE);
    const raw = network.activate(sensors);
    const tickInput = networkOutputToTickInput(
      Array.isArray(raw) ? raw : new Array(NEATENSTEIN_MAIN_NEAT_OUTPUTS).fill(0)
    );
    state = gameTick(state, tickInput, collisionMap, NEATENSTEIN_FIXED_TIMESTEP_MS);
  }
  const finalState = endEpisode(state);
  const signal = extractCombatQualitySignal(finalState, finalState.telemetry ?? createDefaultTelemetry());
  const complexity = network.nodes.length + network.connections.length;
  return computeCombatQualitySignal(signal, complexity);
};
```

**Imports needed**: `createEpisode`, `endEpisode` from `../host/game/episode`; `gameTick` from `../host/game/tick`; `createCollisionMap` from `../renderer/map`; `buildNeatensteinMap` from `../renderer/raycast`; `extractCombatQualitySignal`, `computeCombatQualitySignal` from `../harness/fitness`; `NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS` from `../harness/constants`.

**Determinism concern**: The fitness function must be **synchronous** (Neat's `evaluate()` calls it per network). The `createEpisode`, `gameTick`, and `extractSensors` are all synchronous. The existing `setTimeout(0)` chunking comment (line 1294-1295) refers to the Neat `evaluate()`/`evolve()` async wrapper, not the fitness function itself. This is compatible.

**Confidence**: 0.85 (verified worker imports + synchronous fitness callback pattern + existing `runEpisode` reference)

---

### 5. Module Boundaries — Dependency Graph and Change Surface

**Finding**: The proposed changes touch a well-structured module hierarchy. Here is the verified dependency graph for the affected files:

```
scripts/enemy-navigation.ts
  ├── imports: castRayDDAFromFlatMap from browser-entry/renderer/raycast.ts
  ├── imports: GameState type from browser-entry/host/game/types.ts
  ├── exports: extractSensors → consumed by harness/main-runner.ts, worker/display.worker.ts
  └── NEW exports: findNearestVisibleEnemy, hasLineOfSight (or import from raycast.ts)

browser-entry/renderer/raycast.ts
  ├── imports: buildNeatensteinMap from ./map
  ├── exports: castRayDDAFromFlatMap → consumed by combat.ts, enemy-navigation.ts
  └── NEW export: hasLineOfSight (recommended placement)

browser-entry/host/game/combat.ts
  ├── imports: castRayDDAFromFlatMap from renderer/raycast.ts
  ├── imports: EpisodeTelemetry, GameState from ./types
  ├── exports: fireBolt → consumed by tick.ts
  └── MODIFIED: add shotsWasted tracking, fix aimMissRate sync

browser-entry/host/game/types.ts
  └── MODIFIED: extend EpisodeTelemetry with shotsWasted; add lastShotHit to GameState

browser-entry/host/game/tick.ts
  ├── imports: fireBolt from ./combat
  ├── imports: applyEnemyDamage from ./combat
  ├── exports: gameTick → consumed by main-runner.ts, display.worker.ts
  └── MODIFIED: set lastShotHit flag after fireBolt resolves (step 6, line 357-371)

browser-entry/harness/fitness.ts
  ├── imports: CombatQualitySignal from ./types
  ├── imports: EpisodeTelemetry, GameState from ../host/game/types
  ├── exports: computeCombatQualitySignal → consumed by main-runner.ts
  └── MODIFIED: add shotsWasted weight, ammoEfficiency metric

browser-entry/harness/constants.ts
  └── MODIFIED: add NEATENSTEIN_WEIGHT_SHOTS_WASTED, NEATENSTEIN_WEIGHT_AMMO_EFFICIENCY, VISION_RANGE_CELLS

browser-entry/harness/main-runner.ts
  ├── imports: extractSensors from ../../scripts/enemy-navigation
  ├── imports: gameTick from ../host/game/tick
  └── NO CHANGES (benefits from extractSensors changes automatically)

browser-entry/worker/display.worker.ts
  ├── imports: extractSensors (already imported)
  ├── defines: NEATENSTEIN_MAIN_NEAT_INPUTS = 12 → 15
  ├── MODIFIED: buildAutoTickInput (fire gating), evaluateArmsRaceGeneration (fitness fix)
  └── MODIFIED: clear championMainNetwork on input count change
```

**Key boundary observations**:
1. `scripts/enemy-navigation.ts` sits **outside** the `browser-entry/` tree but imports from it. This is an existing cross-tree dependency. The new `hasLineOfSight` import would follow the same pattern.
2. `main-runner.ts` is **passive** — it benefits from `extractSensors` changes automatically. No changes needed there for the sensor expansion.
3. `display.worker.ts` is the **highest-impact** file: NEAT input constant, fire gating, and fitness function all change here.
4. `types.ts` is the **shared contract** — `EpisodeTelemetry` and `GameState` changes ripple to `combat.ts`, `tick.ts`, `fitness.ts`, and any test that constructs these types.

**Test surface impact**:
- `combat.test.ts` — needs new test cases for `shotsWasted` tracking
- `enemy-navigation.test.ts` — needs LOS and vision-range tests for `extractSensors`
- `fitness.test.ts` — needs new weight tests
- `constants.test.ts` — needs new constant assertions
- `display.worker.test.ts` — needs fire-gating and fitness-function tests (hardest, as the worker is async)
- `types.test.ts` — needs `EpisodeTelemetry` extended interface tests

**Confidence**: 0.92 (verified from direct file reads of all listed modules)

---

### 6. Implementation Sequencing Recommendations

Based on module dependency analysis, the changes should be sequenced to minimize broken intermediate states:

| Phase | Changes | Rationale |
|-------|---------|----------|
| **Phase 1a** | Add `hasLineOfSight` to `raycast.ts`; add `VISION_RANGE_CELLS` to `host/game/constants.ts` | Foundation primitives — no consumers broken |
| **Phase 1b** | Extend `EpisodeTelemetry` in `types.ts` with `shotsWasted`; add `lastShotHit` to `GameState` | Type extensions are backward-compatible (optional fields) |
| **Phase 2a** | Add `findNearestVisibleEnemy` to `enemy-navigation.ts`; refactor `extractSensors` to use it; expand sensor vector to 15 | Core sensor change — breaks existing networks (expected) |
| **Phase 2b** | Add `shotsWasted` tracking to `combat.ts`; fix `aimMissRate` sync gap | Combat telemetry — self-contained within combat.ts + types.ts |
| **Phase 2c** | Set `lastShotHit` in `tick.ts` after `fireBolt` resolves | Tick pipeline — depends on types.ts change from Phase 1b |
| **Phase 3a** | Add fitness weights to `constants.ts`; add `ammoEfficiency` to `fitness.ts` | Fitness composite — depends on telemetry from Phase 2b |
| **Phase 3b** | Update `display.worker.ts`: change `NEATENSTEIN_MAIN_NEAT_INPUTS = 15`, add fire gating, clear `championMainNetwork`, replace placeholder fitness | Worker integration — depends on all prior phases |
| **Phase 4** | Update tests for all modified modules | Validation — depends on all code changes |

**Phase 1a + 1b can be done in parallel** (disjoint file sets). Phase 2a, 2b, 2c can be done in parallel after Phase 1 (2a depends on 1a, 2c depends on 1b, 2b depends on 1b). Phase 3a depends on 2b. Phase 3b depends on everything.

**Confidence**: 0.88 (dependency ordering verified from import graph)

## Reward Design Deep-Dive

### Magnitude Analysis — Does the Proposed Weight Rebalancing Create the Right Incentive Gradient?

**Source**: static-code (`harness/fitness.ts:94-125`, `harness/constants.ts:124-158`, `host/game/types.ts:220-232`, `host/game/combat.ts:163-309`)

The Solution 4 proposal recommends changing `aimMissRate` weight from 1 → 3 and adding `shotsWasted * 0.5` penalty and `ammoEfficiency * 2` reward. A magnitude analysis reveals a **scale mismatch** that could undermine the intent:

**Scenario analysis** (episode = 312 max ticks, episode duration 5s):

| Component | Typical Value | Proposed Weight | Contribution |
|-----------|---------------|------------------|-------------|
| `survivalTicks` | 200–312 | 1 | **+200 to +312** |
| `damageDealt` | 20–80 | 2 | +40 to +160 |
| `kills` | 0–4 | 5 | 0 to +20 |
| `damageTaken` | 10–60 | 1 | −10 to −60 |
| `aimMissRate` (0.3–0.8) | 0.3–0.8 | **3** (proposed) | −0.9 to −2.4 |
| `shotsWasted` | 0–30 | **0.5** (proposed) | 0 to −15 |
| `ammoEfficiency` | 1–8 | **2** (proposed) | +2 to +16 |
| `complexityBonus` | 0–10 | 0.1 | 0 to +1 |
| `parsimonyDensityPenalty` | 0–500 | 0.01 | 0 to −5 |

**Critical observation**: `survivalTicks * 1` contributes **200–312** to fitness, while the entire accuracy penalty stack (`aimMissRate * 3 + shotsWasted * 0.5`) contributes at most **−17.4**. The survival signal overwhelms accuracy by a factor of **~18:1**. A network that fires blindly but survives will outscore a network that fires selectively but dies slightly earlier. This is the core reason the current design fails to produce selective firing even with the weight increase.

**Recommendation R1 — Scale penalties relative to the dominant reward:**

```typescript
// Option A: Normalize survival to [0, 1] so other signals matter
fitness = (survivalTicks / maxTicks) * 100      // 0–100 scale
       + damageDealt * 2
       + kills * 5
       - damageTaken * 1
       - aimMissRate * 3
       - shotsWasted * 0.5
       + ammoEfficiency * 2
       + complexityBonus * 0.1
       - parsimonyDensityPenalty * 0.01;

// Option B: Scale penalties to survival magnitude
// If survival contributes ~250, penalties must reach ~25 to matter (10% of survival)
fitness = survivalTicks * 1
       + damageDealt * 2
       + kills * 5
       - damageTaken * 1
       - aimMissRate * 20                   // SCALED: 0.3 → −6, 0.8 → −16
       - shotsWasted * 2                     // SCALED: 10 wasted → −20
       + ammoEfficiency * 10                 // SCALED: 5 efficiency → +50
       + complexityBonus * 0.1
       - parsimonyDensityPenalty * 0.01;
```

Option A is preferred — it normalizes the survival signal so that accuracy and efficiency penalties operate in the same magnitude band as combat rewards. Option B preserves the raw survival reward but scales penalties proportionally. Either approach requires empirical tuning, but the current proposal (aimMissRate * 3) is **insufficient to shift behavior** because the penalty is negligible relative to survival.

**Confidence**: 0.88 (static-code magnitude analysis + standard reward shaping theory)

---

### Penalty Structure — Taxonomy and Per-Shot Outcome Classification

**Source**: static-code (`host/game/combat.ts:192-224`, `host/game/types.ts:220-232`)

The current `EpisodeTelemetry` only tracks `shotsFired`, `shotsHit`, `damageDealt`, and the derived `aimMissRate`. The proposed `shotsWasted` adds one dimension. However, the **shot outcome taxonomy** can be richer, enabling a more precise penalty structure that distinguishes different failure modes:

**Proposed per-shot outcome classification** (tracked in `EpisodeTelemetry`):

| Outcome | Description | Current Telemetry | Proposed Telemetry | Penalty Weight |
|---------|-------------|-------------------|-------------------|----------------|
| `enemyHit` | Bolt struck a living enemy | `shotsHit++` | `shotsHit++` | 0 (rewarded via `damageDealt`) |
| `wallHit` | Bolt hit a wall (no enemy in path) | counts as miss | `shotsWallHit++` | medium penalty |
| `rangeExpired` | Bolt traveled max range, hit nothing | counts as miss | `shotsRangeExpired++` | low penalty (at least aimed into open space) |
| `blindFire` | Fired when no enemy visible (no LOS or out of range) | indistinguishable | `shotsBlind++` | high penalty |
| `nearMiss` | Bolt missed enemy by small margin (within 2× hit radius) | indistinguishable | `shotsNearMiss++` | small penalty (aim was close) |

**Why this taxonomy matters**: A network that fires at an enemy and narrowly misses (`nearMiss`) should not be penalized as harshly as one that fires into a wall with no enemy visible (`blindFire`). The current binary hit/miss collapses these distinct failure modes into one `aimMissRate` penalty, making the gradient too coarse for NEAT to learn fine aim correction.

**Implementation**: The `fireBolt` function already computes `hitType` (`'wall' | 'enemy' | 'range'`) and `perpendicularDistance` for enemy collision tests. The `blindFire` classification requires passing the enemy visibility state into the telemetry path (the visibility check from Solution 2). The `nearMiss` classification requires checking if any enemy was within `2 * NEATENSTEIN_BOLT_HIT_RADIUS_CELLS` of the bolt path but beyond the hit radius — this data is already available in the `fireBolt` loop.

```typescript
export interface EpisodeTelemetry {
  damageDealt: number;
  shotsFired: number;
  shotsHit: number;
  aimMissRate: number;
  // New fields:
  shotsWallHit: number;       // bolt terminated on a wall (no enemy in path)
  shotsRangeExpired: number;  // bolt traveled max range without hitting anything
  shotsBlind: number;          // fired when no enemy was visible (no LOS or out of range)
  shotsNearMiss: number;       // enemy was within 2× hit radius but not within hit radius
}
```

**Confidence**: 0.90 (static-code confirms `fireBolt` already computes the necessary collision data)

---

### EpisodeTelemetry Additions — Derived Metrics and Diagnostic Value

**Source**: static-code (`harness/types.ts:56-71`, `host/game/types.ts:220-232`)

The `CombatQualitySignal` interface (consumed by `computeCombatQualitySignal`) currently has 7 fields. The Solution 4 proposal adds `shotsWasted` and `ammoEfficiency`. The deeper analysis reveals additional derived metrics that provide diagnostic value without adding raw telemetry counters:

**Proposed `CombatQualitySignal` extensions:**

| Field | Formula | Purpose | Risk |
|-------|---------|---------|------|
| `shotsWasted` | `shotsBlind + shotsWallHit` (where no enemy was visible) | Penalty for blind fire | Redundant if per-shot taxonomy is used directly |
| `ammoEfficiency` | `damageDealt / max(shotsFired, 1)` | Damage per shot | **RATIO TRAP**: rewards low shot count. A network that fires 1 shot for 10 damage (efficiency=10) outscores one that fires 20 shots for 150 damage (efficiency=7.5) |
| `hitRate` | `shotsHit / max(shotsFired, 1)` | Fraction of shots that hit | Simpler than `aimMissRate` but mathematically equivalent (`1 - hitRate = aimMissRate`) |
| `killEfficiency` | `kills / max(shotsFired, 1)` | Kills per shot | Directly rewards lethal accuracy; avoids the ratio trap if combined with raw `kills * 5` |
| `damagePerKill` | `damageDealt / max(kills, 1)` | Efficiency of killing (lower = cleaner kills) | Diagnostic only — not useful as a fitness component because lower damage per kill is better (inverse relationship) |

**Recommendation R2 — Replace `ammoEfficiency` with `killEfficiency`**:

The `ammoEfficiency = damageDealt / shotsFired` ratio creates a perverse incentive: it rewards networks that fire fewer shots, even if total damage is lower. This can cause the network to evolve toward "one perfect shot" behavior that rarely fires and wastes episodes.

Instead, use `killEfficiency = kills / max(shotsFired, 1)`, which rewards lethal accuracy directly. A network that kills 3 enemies with 10 shots (killEfficiency = 0.3) is better than one that kills 1 enemy with 2 shots (killEfficiency = 0.5) — the first network is more combat-effective overall. To avoid the ratio trap, combine `killEfficiency` as a **bonus multiplier** on the `kills` reward rather than a standalone component:

```typescript
const killEfficiency = signal.kills / Math.max(signal.shotsFired, 1);
const killBonus = signal.kills * NEATENSTEIN_WEIGHT_KILLS * (1 + killEfficiency);
// kills=3, shots=10: killBonus = 3 * 5 * (1 + 0.3) = 19.5
// kills=1, shots=2: killBonus = 1 * 5 * (1 + 0.5) = 7.5
// kills=3, shots=3:  killBonus = 3 * 5 * (1 + 1.0) = 30.0  (perfect accuracy, highest reward)
```

This preserves the raw `kills` signal while rewarding accuracy as a **multiplier**, not a ratio trap.

**Confidence**: 0.85 (reward shaping theory + analysis of ratio-trap failure mode)

---

### Shot Direction Feedback — Sensor Signal vs Reward Signal

**Source**: static-code (`scripts/enemy-navigation.ts:379-432`, `host/game/combat.ts:192-224`)

Solution 5 proposes `lastShotHit` as a sensor input (binary flag). This is correct for **perception** (the network needs to know whether its last shot hit to adjust aim). However, the same signal should also drive a **reward shaping component** that the current Solution 4 defers.

**The missing link**: The `lastShotHit` flag provides immediate per-tick feedback, but Solution 4 recommends skipping per-tick shaping "for now." This is too conservative — the `lastShotHit` sensor already requires the per-tick tracking infrastructure, so the marginal cost of adding a shaping reward is near zero.

**Recommendation R3 — Potential-based reward shaping (PBRS) using shot feedback**:

Standard heuristic shaping (e.g., +0.1 for firing when enemy visible, −0.2 for blind fire) can change the optimal policy. Potential-based reward shaping (Ng et al., 1999) guarantees the optimal policy is preserved by using the form `F(s, s') = γΦ(s') − Φ(s)`, where Φ is a potential function over states.

For the Neatenstein discrete-tick setting (γ ≈ 1 for same-episode ticks):

```typescript
// Potential function: reward the network for being in a "good aim" state
function aimPotential(state: GameState): number {
  // Higher potential when an enemy is visible and in the firing arc
  const visibleEnemy = findNearestVisibleEnemy(state, ...);
  if (!visibleEnemy) return 0;
  const inArc = Math.abs(visibleEnemy.bearing) <= FIRE_ARC;
  return inArc ? 0.5 : 0.1;
}

// PBRS shaping reward (added to the per-tick reward)
// F(s_t, s_{t+1}) = Φ(s_{t+1}) − Φ(s_t)
const shaping = aimPotential(nextState) - aimPotential(currentState);
```

This approach:
- Does not change the optimal policy (theoretically sound)
- Provides immediate per-tick gradient toward aligning with visible enemies
- Does not reward "firing" directly (which could create hacking) — it rewards being in a good aim state
- Is compatible with the end-of-episode fitness composite (the shaping terms telescope to `Φ(s_final) − Φ(s_0)`, which is bounded)

**Alternative simpler shaping** (not PBRS, but empirically common in game AI):

```typescript
// Per-tick micro-rewards (add to a perTickReward accumulator)
const perTickReward =
  (fired && enemyVisible ? +0.05 : 0) +       // small reward for firing at visible enemy
  (fired && !enemyVisible ? -0.15 : 0) +       // penalty for blind fire
  (lastShotHit === true ? +0.1 : 0) +          // reward for hitting
  (lastShotHit === false && fired ? -0.05 : 0); // small penalty for missing when aimed
```

**Tradeoff**: The micro-reward approach is simpler but risks reward hacking (network learns to fire only when guaranteed to hit). The PBRS approach is safer but requires implementing the potential function. Given that the `lastShotHit` tracking is already proposed, the PBRS approach is recommended as the **shaping layer** with the end-of-episode composite as the **terminal reward**.

**Confidence**: 0.82 (PBRS theory + codebase has the necessary state access)

---

### Reward Normalization Across Episode Lengths

**Source**: static-code (`harness/constants.ts:29-44`, `harness/fitness.ts:99-107`)

Episodes can end early (hero death) or at the time limit (312 ticks). This creates a **length-dependent reward scale** problem:

- A hero that dies at tick 50: `survivalTicks = 50`, and other signals (damage, kills) are proportionally smaller.
- A hero that survives to tick 312: `survivalTicks = 312`, and all signals are larger.

The survival reward (`survivalTicks * 1`) already dominates the fitness, but it also **confounds** the other signals. A network that survives longer naturally accrues more damage/kills, making it hard to distinguish "good aim + short life" from "bad aim + long life."

**Recommendation R4 — Rate-normalized metrics**:

Add rate-based metrics that are independent of episode length:

```typescript
// In CombatQualitySignal:
const damagePerTick = signal.damageDealt / Math.max(signal.survivalTicks, 1);
const killsPerTick = signal.kills / Math.max(signal.survivalTicks, 1);
const accuracyPerTick = signal.shotsHit / Math.max(signal.survivalTicks, 1);
```

These rate metrics allow the fitness composite to reward **efficiency per unit time**, not just raw totals. A hero that deals 60 damage in 100 ticks (0.6/tick) is more combat-effective than one that deals 80 damage in 200 ticks (0.4/tick), even though the raw damage is higher for the second.

**Caution**: Rate metrics should be **combined with** the survival reward, not replace it. A hero that deals 1 damage in 1 tick then dies has the highest possible damagePerTick but is a terrible strategy. The survival reward provides the floor; rate metrics provide the ceiling.

**Proposed combined formula**:

```typescript
fitness = survivalTicks * 1                           // survival floor
       + damageDealt * 2                               // raw combat output
       + kills * 5 * (1 + killEfficiencyBonus)          // kills with accuracy multiplier
       - damageTaken * 1                                // defense penalty
       - aimMissRate * (survivalTicks / maxTicks * 20) // SCALE penalty to episode length
       - shotsWasted * 2                                // blind fire penalty (scaled per R1)
       + (survivalTicks / maxTicks) * 20               // normalized survival bonus
       + complexityBonus * 0.1
       - parsimonyDensityPenalty * 0.01;
```

**Confidence**: 0.80 (reward normalization is standard RL practice; specific formula needs empirical validation)

---

### Adaptive Weight Scheduling — Curriculum Learning Across Generations

**Source**: static-code (`harness/constants.ts:120-158`, `display.worker.ts` evolution loop)

The current fitness weights are static constants. The Risks section notes "weight tuning requires empirical observation" but doesn't propose a systematic approach. A **curriculum schedule** can address this:

| Generation Range | Phase | Dominant Signal | Strategy |
|------------------|-------|-----------------|----------|
| 1–20 | Exploration | `survivalTicks * 1` dominant | Let networks learn basic navigation and survival first |
| 20–50 | Accuracy onset | Increase `aimMissRate` weight from 3 → 10 | Now that networks survive, pressure them to aim |
| 50–100 | Efficiency | Add `shotsWasted * 2`, `killEfficiencyBonus` | Now that they aim, pressure them to fire selectively |
| 100+ | Mastery | Full penalty stack at final weights | Fine-tune for optimal combat behavior |

**Implementation**: The fitness function already accepts the signal object. Adding a `generation` parameter (already available on `GameState.generation`) and a weight schedule table in `constants.ts` allows the weights to adapt:

```typescript
export function getFitnessWeights(generation: number): FitnessWeights {
  if (generation < 20) return { ...DEFAULT_WEIGHTS, aimMissRateWeight: 1 };
  if (generation < 50) return { ...DEFAULT_WEIGHTS, aimMissRateWeight: 10, shotsWastedWeight: 1 };
  if (generation < 100) return { ...DEFAULT_WEIGHTS, aimMissRateWeight: 15, shotsWastedWeight: 2, killEfficiencyBonus: true };
  return { ...DEFAULT_WEIGHTS, aimMissRateWeight: 20, shotsWastedWeight: 2, killEfficiencyBonus: true };
}
```

**Tradeoff**: Curriculum scheduling adds complexity and requires knowing when to transition phases. A simpler alternative is to use a **linear ramp** from the initial weights to the target weights over N generations.

**Confidence**: 0.75 (curriculum learning is established in RL literature; the specific schedule is heuristic)

---

### Summary of Reward Design Recommendations

| ID | Recommendation | Priority | Risk | Confidence |
|----|---------------|----------|------|------------|
| R1 | Scale penalties relative to survival reward magnitude | **HIGH** | Low — constant changes | 0.88 |
| R2 | Replace `ammoEfficiency` with `killEfficiency` multiplier on kills | Medium | Low — formula change | 0.85 |
| R3 | Add potential-based reward shaping using aim state potential | Medium | Medium — PBRS implementation | 0.82 |
| R4 | Add rate-normalized metrics (damagePerTick, killsPerTick) | Low | Low — derived from existing counters | 0.80 |
| R5 | Add per-shot outcome taxonomy (wallHit, blindFire, nearMiss) | **HIGH** | Medium — telemetry plumbing | 0.90 |
| R6 | Adaptive weight scheduling across generations (curriculum) | Low | Medium — tuning complexity | 0.75 |

**Top priority**: R1 (penalty scaling) and R5 (shot outcome taxonomy). Without R1, the proposed weight increases are insufficient to shift behavior. Without R5, the penalty gradient is too coarse for NEAT to learn fine aim correction.

**Combined effect**: R1 + R5 together create a **multi-resolution penalty signal**: the per-shot taxonomy (R5) provides fine-grained distinction between failure modes, and the scaled weights (R1) ensure those distinctions actually matter in the fitness landscape.

---

## Confidence

Overall investigation confidence: **0.91** — all findings are based on direct static-code evidence from the actual source files. The proposed solutions are grounded in existing code primitives (DDA raycast, telemetry counters, fitness composite). The main uncertainty is in fitness weight tuning, which requires empirical observation of evolution behavior.

The reward design deep-dive (R1–R6) carries confidence 0.75–0.90 per recommendation, with the highest confidence on R1 (magnitude analysis, 0.88) and R5 (per-shot taxonomy, 0.90) since both are grounded in direct code analysis. R3 (PBRS) and R6 (curriculum) are more speculative and depend on implementation choices that need validation.

## Cortex Search Note

Cortex index was stale for `examples/neatenstein/` files (freshness_check reported many stale paths). All evidence was gathered via direct file reads (native tool fallback) per the degraded-Cortex fallback policy. The research is complete despite the stale index because the target files are known exact paths.

---

## Performance & Vision Review (Enhanced)

### System Parameters (source: static-code)

| Parameter | Value | Source |
|-----------|-------|--------|
| Map size | 120 × 120 cells | `constants.ts:148` |
| DDA render distance cap | 30 cells | `framebuffer.ts:47` |
| Fixed timestep | 16 ms (60 FPS) | `host/game/constants.ts:29` |
| Fitness episode duration | 5,000 ms (312 ticks) | `harness/constants.ts:32,42-44` |
| Main-agent population size | 4 variants | `harness/constants.ts:26`, `display.worker.ts:1240` |
| NEAT inputs / outputs | 12 / 5 | `display.worker.ts:1226,1233` |
| Max concurrent enemies | 8 | `host/game/constants.ts:41` |
| Bolt max range | 30 cells | `host/game/constants.ts:370` |
| Eval chunk yield interval | 32 ticks | `harness/constants.ts:53` |
| Player speed | 6 cells/s (~0.096 cells/tick) | `host/game/constants.ts:107` |
| Enemy collision radius | ~0.38 cells (96/252) | `host/game/constants.ts:124` |
| Vision range (proposed) | 15 cells | Solution 2 recommendation |

### Per-Tick LOS Raycast Cost Analysis

**Current `extractSensors` cost per tick** (source: `enemy-navigation.ts:379-432`):

| Component | Operations | Complexity |
|-----------|-----------|------------|
| Player sensors [0-4] | 5 property reads | O(1) |
| Nearest-enemy scan [5-7] | Linear scan over ≤8 enemies, each with `Math.hypot` | O(N), N ≤ 8 |
| Cardinal wall raycasts [8-11] | 4 × `castRayDDAFromFlatMap`, each traversing ≤30 cells | O(4 × cap) = O(120) |
| **Total DDA traversals per tick** | 4 rays × ~10-30 cells each | **~40-120 cell-steps** |

**Proposed additions per tick:**

| Component | Operations | Complexity |
|-----------|-----------|------------|
| Vision-range distance check | 1 `Math.hypot` comparison (already computed for nearest-enemy scan) | O(1) — zero incremental cost |
| LOS DDA ray to nearest enemy | 1 × `castRayDDAFromFlatMap`, traversing ≤21 cells (diagonal of 15-cell vision) | O(vision_range) ≈ O(21) |
| Binary sensors [12-14] | 3 boolean assignments | O(1) |
| **Incremental DDA cost** | +1 ray, ≤21 cells | **~10-21 additional cell-steps** |

**Per-tick overhead: +17-21% DDA cost.** The LOS ray is bounded by the enemy distance, not the render distance cap. When the nearest enemy is close (common in combat), the LOS DDA terminates in 2-5 cells. The worst case (enemy at 15-cell vision limit along a diagonal) is ~21 cell-steps.

**Confidence: 0.93** — based on direct source code analysis of `raycast.ts`, `enemy-navigation.ts`, and `harness/constants.ts`.

### Per-Generation Evaluation Cost

**Headless evaluation path** (`main-runner.ts:runEpisode`, line 364-403):

```text
Per variant:  312 ticks × (extractSensors + network.activate + gameTick)
             = 312 × (5 DDA rays + forward pass + full game simulation)

Per generation (4 variants):
  Current DDA calls:  4 × 312 × 4 cardinal rays = 4,992 DDA traversals
  Proposed DDA calls: 4 × 312 × (4 cardinal + 1 LOS) = 6,240 DDA traversals
  Increment:          +1,248 DDA traversals (+25%)
```

Each DDA traversal is a tight loop: `while(true)` with `flatMap[mapY * side + mapX]` array access. On a 120×120 `Uint8Array`, this is a cache-friendly linear scan. At ~30 iterations max, each DDA call is ~100-300 ns. Total incremental DDA cost per generation: ~1,248 × 200 ns ≈ **0.25 ms**. This is negligible.

**The dominant cost is `gameTick`**, not the sensor extraction. Each `gameTick` involves movement, collision detection, combat resolution, enemy AI, and episode state management. The LOS DDA adds < 0.1% overhead to a generation's total wall-clock time.

**Confidence: 0.90** — DDA micro-benchmark estimate; actual `gameTick` cost not measured but dominates by orders of magnitude.

### The Real Performance Threat: Placeholder → Real Fitness Evaluation

**Current worker fitness** (`display.worker.ts:1319-1328`): The placeholder activates the network with a **zero input vector** and returns `output[0]`. Cost: one forward pass per variant ≈ O(network_size). Total per generation: 4 forward passes ≈ **negligible**.

**Proposed fix** (Solution 1, Layer B): Replace with real episode-based evaluation using `runEpisode`. Cost per variant: 312 ticks × (extractSensors + network.activate + gameTick). This is a **~1000× cost increase** per variant.

| Metric | Placeholder (current) | Real episodes (proposed) | Factor |
|--------|----------------------|-------------------------|--------|
| Forward passes per generation | 4 | 4 × 312 = 1,248 | 312× |
| DDA rays per generation | 0 | 4 × 312 × 5 = 6,240 | ∞ |
| gameTick calls per generation | 0 | 4 × 312 = 1,248 | ∞ |
| Estimated wall-clock per generation | < 1 ms | ~500-2000 ms | ~1000× |

**Browser worker constraint**: The NEAT evaluation runs on the **same worker thread** as the render loop (`display.worker.ts`). With cooperative yielding every 32 ticks (`NEATENSTEIN_EVAL_CHUNK_TICKS = 32`), the worker yields via `setTimeout(0)` approximately 10 times per variant, ~40 times per generation. Each yield releases the event loop for ~4 ms. During evaluation, the **render loop stalls** — the worker cannot both evaluate episodes and render frames simultaneously.

**Confidence: 0.95** — the placeholder fitness is confirmed at `display.worker.ts:1319-1328`; the real episode loop is confirmed at `main-runner.ts:364-403`; both run on the same worker.

### Performance Recommendations

#### Recommendation 1: LOS DDA Early-Exit with Distance Bound (Low effort, High value)

The proposed `hasLineOfSight` casts a full DDA ray that may walk past the enemy to a distant wall. Optimize by passing the enemy distance as a step cap:

```typescript
function hasLineOfSight(
  flatMap: Uint8Array,
  mapSize: number,
  from: Vector2,
  to: Vector2,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return true;

  // Early exit: if the enemy is beyond vision range, skip DDA entirely.
  if (dist > VISION_RANGE_CELLS) return false;

  const dirX = dx / dist;
  const dirY = dy / dist;

  // Cast DDA with a step cap = ceil(dist) + 1.
  // If no wall is hit within the enemy distance, LOS is clear.
  // This avoids walking to a distant wall when the enemy is nearby.
  const maxSteps = Math.ceil(dist) + 1;
  const hit = castRayDDAFromFlatMap(flatMap, mapSize, from.x, from.y, dirX, dirY);
  const wallDist = Number.isFinite(hit.perpWallDist) ? hit.perpWallDist : Infinity;
  return wallDist >= dist;
}
```

**Note**: The current `castRayDDAFromFlatMap` uses `NEATENSTEIN_RENDER_DISTANCE_CAP` (30) as its hard stop, not a caller-supplied distance. A `maxSteps` parameter would require a small API extension to `raycast.ts`, or a wrapper that stops early. The simplest approach: add an optional `maxSteps` parameter to `castRayDDAFromFlatMap` defaulting to `NEATENSTEIN_RENDER_DISTANCE_CAP`.

**Expected savings**: When the nearest enemy is at 3-5 cells (typical combat range), the LOS DDA walks 3-5 cells instead of potentially 15-30 cells to the nearest wall. Savings: ~60-80% of LOS DDA cost in combat.

#### Recommendation 2: Vision-Range Pre-Filter (Trivial, High value)

Always check the Euclidean distance **before** casting the LOS DDA ray. The `nearestDist` is already computed during the nearest-enemy scan. If `nearestDist > VISION_RANGE_CELLS`, zero out sensors [5-7] and skip the LOS DDA entirely:

```typescript
// In extractSensors, after finding nearestEnemy and nearestDist:
const inRange = nearestDist <= VISION_RANGE_CELLS;
if (!inRange) {
  // Skip LOS DDA — enemy is out of vision range
  sensors[5] = 0; sensors[6] = 0; sensors[7] = 0;
  sensors[12] = 0; // enemyVisible = false
  // Continue to wall raycasts
} else {
  const hasLOS = hasLineOfSight(flatMap, mapSize, p.position, nearestEnemy.position);
  if (hasLOS) {
    sensors[5] = bearing; sensors[6] = nearestDist; sensors[7] = nearestEnemy.health;
    sensors[12] = 1.0; // enemyVisible = true
  } else {
    sensors[5] = 0; sensors[6] = 0; sensors[7] = 0;
    sensors[12] = 0;
  }
}
```

**Expected savings**: Eliminates 100% of LOS DDA calls when no enemy is within vision range. During exploration phases (enemy far or behind walls), this saves 1 DDA ray per tick.

#### Recommendation 3: Cached LOS with Movement Invalidation (Medium effort, Medium value)

Cache the LOS result per tick and invalidate only when the player or the nearest enemy moves more than 0.5 cells (half a grid cell). At player speed 6 cells/s (0.096 cells/tick), the cache is valid for ~5 ticks. At enemy speed (varies), similar.

```typescript
let cachedLosKey: string | null = null;
let cachedLosResult: boolean = false;

function getCachedLineOfSight(
  flatMap: Uint8Array, mapSize: number,
  playerPos: Vector2, enemyPos: Vector2,
): boolean {
  const key = `${Math.floor(playerPos.x)},${Math.floor(playerPos.y)},` +
              `${Math.floor(enemyPos.x)},${Math.floor(enemyPos.y)}`;
  if (key === cachedLosKey) return cachedLosResult;
  cachedLosKey = key;
  cachedLosResult = hasLineOfSight(flatMap, mapSize, playerPos, enemyPos);
  return cachedLosResult;
}
```

**Expected savings**: ~80% reduction in LOS DDA calls during normal gameplay (player and enemy move < 0.5 cells per tick). During headless evaluation (312 ticks × 4 variants), this saves ~1,000 DDA calls per generation.

**Caution**: The cache must be cleared on episode reset. For the headless evaluation path (`runEpisode`), the cache should be per-episode (local variable, not module-level).

#### Recommendation 4: DDA Object Allocation Reduction (Low effort, Medium value)

`castRayDDAFromFlatMap` returns a new `{ perpWallDist, side, mapX, mapY }` object on every call. In the hot path (5 DDA calls × 312 ticks × 4 variants = 6,240 allocations per generation), this creates GC pressure. Options:

- **Option A**: Add a `castRayDDAIntoBuffer` variant that writes into a pre-allocated `Float32Array` or reusable object.
- **Option B**: Return a numeric tuple `[perpWallDist, side, mapX, mapY]` instead of an object (avoids property lookup overhead too).
- **Option C**: In `extractSensors`, inline the 4 cardinal DDA calls and write directly into `sensors[8-11]`, avoiding the intermediate `CastRayDDAHit` object.

**Expected savings**: Eliminates 6,240 object allocations per generation. GC pause reduction is modest but measurable on low-end devices.

#### Recommendation 5: Dedicated Evaluation Worker (High effort, Critical value)

The most impactful performance recommendation: **move NEAT episode evaluation to a separate Web Worker** so it does not block the render loop.

**Current architecture**: `display.worker.ts` handles both rendering and NEAT evaluation. When `evaluateArmsRaceGeneration` runs real episodes, the worker is occupied for ~500-2000 ms per generation, during which the render loop is stalled.

**Proposed architecture**:
```
display.worker.ts (render worker)
  ├── Render loop (rAF-driven, 60 FPS)
  └── Sends NEAT evaluation request →
        eval.worker.ts (dedicated evaluation worker)
          ├── Creates NEAT population
          ├── Runs 4 × 312-tick episodes (headless, no rendering)
          ├── Returns champion network + quality signal
          └── Posts result back to display.worker.ts
```

The evaluation worker runs `runEpisode` in a tight loop without rendering overhead. Since `gameTick` is pure computation (no DOM/canvas access), it can run in any worker. The `extractSensors` DDA calls and `network.activate` forward passes are also pure computation.

**Expected improvement**: Render loop never stalls during evolution. Generation evaluation completes in ~200-500 ms on the dedicated worker (faster without render competition). User sees smooth gameplay while evolution runs in the background.

**Implementation complexity**: Requires structured cloning of `Network` objects across worker boundaries (already supported by Neataptic's serialization). The `flatMap` (`Uint8Array`) is transferable. The `GameState` is a plain object that clones efficiently.

#### Recommendation 6: GPU-Batched Population Evaluation (High effort, High value for large populations)

The NeatapticTS library has GPU-batched network activation support (`src/architecture/network/gpu/network.gpu.batched.ts`, `network.gpu.batch-evaluation.ts`). If the population size grows beyond 4, batch-evaluating all variants' forward passes on the GPU could accelerate evaluation significantly.

**Current relevance**: With `NEATENSTEIN_MAIN_NEAT_POPSIZE = 4`, the GPU overhead (buffer upload, kernel dispatch, result readback) likely exceeds the CPU cost of 4 forward passes. This recommendation becomes valuable if population size scales to 16+.

**Future trigger**: If convergence is slow and the population is increased beyond 16, enable GPU batched evaluation for the `network.activate` calls within `runEpisode`. The DDA and `gameTick` portions remain CPU-bound.

#### Recommendation 7: Enemy BFS Pathfinding Audit (Investigation needed)

The enemy AI uses BFS distance maps (referenced in `harness/constants.ts:76`: "the MLP receives six world inputs (the vision vector from the BFS distance map)"). BFS flood-fill on a 120×120 grid = 14,400 cell visits per flood. If this runs per-enemy per-tick (8 enemies × 14,400 = 115,200 operations per tick), it is **~100× more expensive than all DDA rays combined**.

**Action**: Audit `scripts/enemy-controller.ts` and `scripts/enemy-navigation.ts` for BFS invocation frequency. If BFS runs every tick per enemy, consider:
- Caching BFS distance maps and recomputing only when the player moves to a new grid cell (~every 5 ticks at 6 cells/s).
- Using Dijkstra with early termination instead of full BFS.
- Precomputing a distance transform once per player grid-cell change.

This is the **actual performance bottleneck** in the system, not the proposed LOS DDA.

**Confidence: 0.70** — the BFS usage is inferred from the MLP topology comment; the actual BFS invocation frequency and implementation need verification in `enemy-controller.ts`.

### Vision Range Implementation Assessment

The proposed `VISION_RANGE_CELLS = 15` is well-chosen:

- **Half the bolt max range** (30 cells) — the hero must close distance before firing, creating tactical depth.
- **DDA bounded at ~21 cells** (diagonal of 15² + 15² = 424, √424 ≈ 20.6) — the LOS DDA is always bounded.
- **Larger than player speed × reaction window** (6 cells/s × 0.5s = 3 cells) — the hero has ~2.5 seconds to react to an enemy entering vision, sufficient for NEAT to learn.
- **Smaller than map half-extent** (60 cells) — forces exploration; the hero cannot see across the entire arena.

**Concern: hard cutoff at boundary.** Enemies appear/disappear at exactly 15 cells, which could cause the NEAT network to oscillate at the vision boundary (enemy visible → turn toward → enemy disappears → turn away → enemy reappears). Mitigations:
- **Hysteresis**: Use 15 cells for acquisition and 18 cells for loss (3-cell hysteresis band). This prevents flickering at the boundary.
- **Falloff**: Instead of binary zero-out, scale sensor values by a smooth falloff function: `sensor *= clamp(1 - (dist - VISION_RANGE) / 3, 0, 1)` for distances within [VISION_RANGE, VISION_RANGE + 3]. This gives the network a gradual signal as enemies approach the vision limit.

**Confidence: 0.88** — the vision range value is sound; the boundary oscillation concern is a theoretical risk that requires empirical observation.

### Path Tracing Scalability Assessment

| Approach | Cells Traversed per Tick | Per-Generation Cost (4 × 312 ticks) | Verdict |
|----------|--------------------------|--------------------------------------|---------|
| Single DDA LOS ray (proposed) | ≤21 (vision range diagonal) | 6,240 × ~21 = ~131K cell-steps | ✅ Scales well |
| Multi-ray cone (3-5 rays) | ≤21 × 3-5 = 63-105 | 18,720-31,200 rays | ⚠️ 3-5× cost, marginal benefit |
| BFS flood-fill visibility | 120 × 120 = 14,400 | 4 × 312 × 14,400 = ~18M cell-steps | ❌ Far too expensive |
| Raycast + corner clipping | Same as single DDA + corner math | Similar to single DDA | ⚠️ Over-engineered |

**The single DDA ray approach scales linearly with vision range** and is the correct choice. Even if vision range doubles to 30 cells, the LOS DDA traverses at most ~42 cells — still negligible compared to `gameTick`.

**Scalability ceiling**: At vision range = 60 (half map), the LOS DDA could traverse ~85 cells (diagonal). At this point, a spatial hash or precomputed visibility polygon would be more efficient. But at VISION_RANGE = 15, the DDA is the optimal approach.

**Confidence: 0.92** — based on DDA algorithm analysis and comparison with alternatives.

### Sensor Computation Overhead Assessment

| Sensor | Computation | Per-Tick Cost | Notes |
|--------|------------|---------------|-------|
| [0-4] Player state | 5 property reads | ~50 ns | O(1) |
| [5-7] Nearest enemy | Linear scan + atan2 | ~500 ns | O(N), N ≤ 8; atan2 is the dominant cost |
| [8-11] Cardinal wall DDA | 4 × DDA traversal | ~400-1200 ns | O(4 × cap), bounded at 30 cells |
| [12] enemyVisible (proposed) | Boolean from LOS check | ~0 ns incremental (LOS already computed) | O(1) |
| [13] enemyInFiringArc (proposed) | 1 abs comparison | ~5 ns | O(1) |
| [14] lastShotHit (proposed) | 1 property read | ~5 ns | O(1) |
| LOS DDA (proposed) | 1 × DDA traversal | ~100-600 ns | O(vision_range), bounded at ~21 cells |
| **Total per-tick sensor cost** | | **~1.1-2.4 μs** | Negligible vs gameTick |

**NEAT network activation cost**: With 15 inputs, 5 outputs, and a 64-node/256-edge topology budget, each `network.activate` is ~256 multiply-add operations ≈ ~500 ns. This is also negligible.

**The sensor expansion from 12 → 15 inputs does not meaningfully increase per-tick computation.** The dominant cost remains `gameTick` (movement, collision, combat, enemy AI).

**Confidence: 0.91** — based on operation count analysis and DDA cell-step estimates.

### Browser Worker Constraints Summary

| Constraint | Current State | Impact of Proposed Changes |
|------------|--------------|---------------------------|
| Single worker thread (render + eval) | Placeholder fitness → no contention | **Real fitness → render stalls 500-2000 ms/gen** (CRITICAL) |
| Cooperative yielding (32 ticks) | Adequate for placeholder | May be insufficient for real episodes — need profiling |
| Structured clone for Network | Supported by Neataptic | No new serialization needed |
| Uint8Array flatMap transfer | Transferable | No issue |
| Memory budget | 120×120 Uint8Array = 14.4 KB | Negligible; LOS adds no allocations beyond DDA hit objects |
| GC pressure | 4 DDA objects/tick = 1,248/gen | +1 DDA object/tick = +312/gen (mitigated by Recommendation 4) |

**Critical path**: The transition from placeholder to real fitness evaluation is the single highest-impact performance change. Without a dedicated evaluation worker (Recommendation 5), the render loop will stall during every generation evaluation, degrading user experience.

### Performance Recommendation Priority Matrix

| Priority | Recommendation | Effort | Impact | Risk |
|----------|---------------|--------|--------|------|
| **P0** | Dedicated evaluation worker (Rec 5) | High | Critical | Medium — new worker wiring |
| **P1** | Vision-range pre-filter (Rec 2) | Trivial | High | None |
| **P1** | LOS DDA early-exit (Rec 1) | Low | High | Low — small API extension |
| **P2** | Cached LOS with movement invalidation (Rec 3) | Medium | Medium | Low — cache correctness |
| **P2** | DDA object allocation reduction (Rec 4) | Low | Medium | Low — API change |
| **P3** | Enemy BFS pathfinding audit (Rec 7) | Medium | Unknown | None — investigation only |
| **P4** | GPU-batched evaluation (Rec 6) | High | Low (at popsize=4) | Medium — GPU integration |

### Confidence Summary

| Finding | Confidence | Action |
|---------|-----------|--------|
| LOS DDA cost is negligible vs gameTick | 0.93 | ACT — proceed with single DDA ray approach |
| Placeholder → real fitness is the critical performance threat | 0.95 | ACT — requires dedicated worker before enabling real evaluation |
| Vision range 15 is well-calibrated | 0.88 | ACT — add hysteresis/falloff to mitigate boundary oscillation |
| Path tracing scales linearly, no scalability concern at VISION_RANGE=15 | 0.92 | ACT — single DDA ray is optimal |
| Sensor expansion 12→15 has negligible per-tick overhead | 0.91 | ACT — proceed with 15-input vector |
| Enemy BFS may be the actual bottleneck | 0.70 | DELEGATE — audit `enemy-controller.ts` BFS invocation frequency |
| DDA object allocation creates measurable GC pressure | 0.65 | DELEGATE — profile with DevTools memory timeline |

---

## NEAT Architecture Analysis (Enhanced)

**Source**: static-code (`src/architecture/network/activate/network.activate.core.utils.ts:132-143`, `src/architecture/network/genetic/network.genetic.setup.utils.ts:180-192`, `src/architecture/network/genetic/network.genetic.utils.types.ts:37-38`, `src/neat/neat.defaults.constants.ts:163-223`, `examples/neatenstein/browser-entry/harness/main-runner.ts:227-243,295-348,364-403`, `examples/neatenstein/browser-entry/worker/display.worker.ts:1226-1233,1304-1435`, `src/neat/nge-main-agent/neat.nge-main-agent.types.ts:24-97`)

This section evaluates the five proposed solutions from a NEAT-architecture-specific perspective, identifying structural concerns that the existing research does not address.

### N1. Input Count Change Is a Genome Structural Break, Not a Soft Migration

**Finding**: Changing the sensor vector from 12 to 15 inputs is not a configuration update — it is a **genome structural break** that makes existing networks incompatible at three levels:

1. **Activation guard** (`network.activate.core.utils.ts:132-143`):
   ```typescript
   function validateInputVector(network: Network, inputVector: number[]): void {
     if (
       !(Array.isArray(inputVector) || ArrayBuffer.isView(inputVector)) ||
       inputVector.length !== network.input
     ) {
       throw new NetworkActivateInputSizeMismatchError(
         `Input size mismatch: expected ${network.input}, got ${...}`,
       );
     }
   }
   ```
   Any 12-input champion network activated with a 15-element sensor vector will **throw** `NetworkActivateInputSizeMismatchError`. The guard is a hard runtime check, not a warning.

2. **Crossover parent compatibility guard** (`network.genetic.setup.utils.ts:180-192`):
   ```typescript
   function validateParentCompatibility(
     parentNetwork1: Network,
     parentNetwork2: Network,
   ): void {
     if (
       parentNetwork1.input !== parentNetwork2.input ||
       parentNetwork1.output !== parentNetwork2.output
     ) {
       throw new NetworkGeneticParentCompatibilityError(
         PARENT_COMPATIBILITY_ERROR_MESSAGE,
       );
     }
   }
   ```
   Old 12-input genomes **cannot breed** with new 15-input genomes. The crossover operator throws `NetworkGeneticParentCompatibilityError` before any gene alignment occurs.

3. **Speciation compatibility distance** (`neat.defaults.constants.ts:163-223`):
   The NEAT compatibility distance formula uses:
   - `DEFAULT_EXCESS_COEFF = 1` (excess nodes/connections)
   - `DEFAULT_DISJOINT_COEFF = 1` (disjoint nodes/connections)
   - `DEFAULT_WEIGHT_DIFF_COEFF = 0.5` (average weight difference)
   - `DEFAULT_COMPATIBILITY_THRESHOLD = 3` (speciation boundary)

   Adding 3 input nodes (with their connections to hidden/output nodes) creates 3+ excess nodes and 3×(hidden+output) excess connections relative to old genomes. Even with N=20 (normalizing factor), the excess contribution alone exceeds the compatibility threshold of 3.0. Old species representatives and new genomes would be **maximally divergent** — old species are effectively extinct.

**Implication**: The research correctly identifies that "existing champion networks stored in `championMainNetwork` will be incompatible and must be cleared" (line 159, 299). But it frames this as a simple re-seeding. From the NEAT architecture perspective, this is a **complete population extinction event**: all species die, the innovation numbering space is reset, and the population starts from scratch with a new structural baseline. The plan should explicitly document this as an accepted cost, not minimize it.

**Recommendation**: Before changing the input count, export any champion network's serialized form for archival purposes. After the change, the population must be re-seeded from scratch. The `championMainNetwork` variable in `display.worker.ts:215` must be set to `null` to force the fallback AI until the first new generation produces a champion. This is already noted in the research but should be framed as a **genome extinction event** in the plan documentation.

**Confidence**: 0.95 (three confirmed source-code guards with exact line references)

---

### N2. Dual Constant Definitions — A Drift Risk That Must Be Eliminated

**Finding**: `NEATENSTEIN_MAIN_NEAT_INPUTS` is defined **independently** in two files with **no shared import**:

- `main-runner.ts:301`: `const NEATENSTEIN_MAIN_NEAT_INPUTS = 12;` — with a comment "Mirrors the worker-side constant"
- `display.worker.ts:1226`: `const NEATENSTEIN_MAIN_NEAT_INPUTS = 12;` — with `@see AC-039`

Similarly, `NEATENSTEIN_MAIN_NEAT_OUTPUTS` is duplicated:
- `main-runner.ts:309`: `const NEATENSTEIN_MAIN_NEAT_OUTPUTS = 5;`
- `display.worker.ts:1233`: `const NEATENSTEIN_MAIN_NEAT_OUTPUTS = 5;`

And `networkOutputToTickInput` is **duplicated** in both files with identical logic:
- `main-runner.ts:328-348`
- `display.worker.ts:1394-1409`

The comments acknowledge the duplication ("Mirrors the worker-side constant") but no shared module exists. Changing the input count to 15 requires updating **both** definitions in lockstep. If one is missed, the headless evaluation and the live worker will use different sensor vector lengths, causing silent activation failures or garbage sensor data.

**Recommendation**: Extract `NEATENSTEIN_MAIN_NEAT_INPUTS`, `NEATENSTEIN_MAIN_NEAT_OUTPUTS`, `NEATENSTEIN_MAIN_NEAT_MAX_TURN_RATE`, and `networkOutputToTickInput` into a shared module (e.g., `harness/neat-io-config.ts`) imported by both `main-runner.ts` and `display.worker.ts`. This eliminates the drift risk permanently. The research's Summary of Changes table (line 287-298) lists both files but does not flag the duplication risk — it should be elevated to a **blocking prerequisite**.

**Confidence**: 0.97 (direct line-level confirmation of independent definitions)

---

### N3. Three-Way Network Construction Mismatch — NGE Embryo vs Plain Network vs NEAT Population

**Finding**: The Neatenstein auto-mode uses **three architecturally distinct network construction paths** that the research does not distinguish:

1. **NGE embryo pipeline** (`main-runner.ts:227-243`):
   ```typescript
   function createMainGenome(seed, generation, variantId): Genome {
     const config: NgeMainAgentLifecycleConfig = {
       seed: seed + variantId,
       maxNodes: 64,
       maxEdges: 256,
     };
     const embryo = buildMainAgentEmbryo(config);
     return embryo as unknown as Genome;
   }
   ```
   This produces a `NgeMainAgentEmbryo` with motif archetypes (AttentionHead, GatedRecurrentCell, EpisodicSlot). Crucially, `NgeMainAgentLifecycleConfig` has **no `input` or `output` field** — only `seed`, `maxNodes`, `maxEdges` (confirmed at `neat.nge-main-agent.types.ts:24-31`). The embryo is a structural genome, not a runnable network.

2. **Headless evaluation network** (`main-runner.ts:373-377`):
   ```typescript
   const network = new Network(
     NEATENSTEIN_MAIN_NEAT_INPUTS,   // 12
     NEATENSTEIN_MAIN_NEAT_OUTPUTS,  // 5
     { seed: hashSeed(episodeSeed, variant.id) },
   );
   ```
   This creates a **plain feed-forward Network** with random initial topology. It is NOT derived from the NGE embryo. The embryo is tracked as `variant.genome` for selection/complexity scoring, but the actual episode evaluation uses a separate, unrelated network. The embryo and the evaluation network have **no topological relationship**.

3. **Worker NEAT population** (`display.worker.ts:1330-1335`):
   ```typescript
   const neatPop = new Neat(
     NEATENSTEIN_MAIN_NEAT_INPUTS,
     NEATENSTEIN_MAIN_NEAT_OUTPUTS,
     fitnessFn,
     { popsize: 4, seed: popSeed },
   );
   ```
   This creates a **standard NEAT population** with mutation, crossover, and speciation. The champion from `neatPop.getFittest()` (line 1342) is a standard evolved Network, NOT an NGE embryo. The worker stores this as `championMainNetwork` (line 1371) and uses it for live auto-mode control.

**Architectural implication**: The "champion" in the headless path (main-runner) is selected based on NGE embryo properties but evaluated using a random Network — there is **no evolutionary continuity** between the embryo and the evaluation network. The "champion" in the worker path IS the evolved network. These two paths produce structurally and behaviorally different champions.

**Sensor change impact**: Changing the input count from 12→15 affects paths 2 and 3 directly (they use `NEATENSTEIN_MAIN_NEAT_INPUTS` in their constructors). Path 1 (NGE embryo) is unaffected because it has no input/output dimension. However, this means the NGE embryo's structural complexity (nodeCount, edgeCount) is decoupled from the actual evaluation network's topology — the embryo could have 30 nodes while the evaluation network starts with 17 (12+5) and grows through NEAT evolution.

**Recommendation**: The research should explicitly acknowledge this three-way mismatch. The sensor input count change is straightforward for paths 2 and 3 (update the constant). But the deeper architectural issue is that the NGE embryo pipeline and the actual network evaluation are structurally disconnected. The plan document (`neatenstein-auto-neat-mode.plans.md:31`) notes: "The `NgeMainAgentEmbryo` → `NeatGenome` → `Network` converter is a documented follow-up, NOT part of this plan." Until that converter exists, the embryo is purely metadata, and the sensor change only needs to touch the two `NEATENSTEIN_MAIN_NEAT_INPUTS` constants and the `extractSensors` function.

**Confidence**: 0.92 (confirmed from direct source reads of all three construction paths)

---

### N4. Topology Budget Adequacy — 64 Nodes / 256 Edges May Be Tight for 15 Inputs

**Finding**: The topology budget is `maxNodes: 64, maxEdges: 256` (`main-runner.ts:209-212`). This is the NGE embryo's tier cap, applied to the embryo construction. However, the actual evaluation network (`new Network(12, 5)`) and the worker's NEAT population (`new Neat(12, 5, ...)`) use the **library defaults** of `DEFAULT_MAX_NODES = Infinity` and `DEFAULT_MAX_CONNS = Infinity` (`neat.defaults.constants.ts:179,188`).

**Node budget analysis**:

| Configuration | Input Nodes | Output Nodes | Fixed Total | Remaining for Hidden | Hidden Ratio |
|-------------|-------------|--------------|-------------|----------------------|--------------|
| Current (12 inputs) | 12 | 5 | 17 | 47/64 = 73% | 73% |
| Proposed (15 inputs) | 15 | 5 | 20 | 44/64 = 69% | 69% |

Adding 3 inputs consumes 3 more of the 64-node budget for fixed input nodes, reducing hidden node headroom from 47 to 44 (−6%). This is modest but not negligible — the network has 3 fewer hidden nodes to learn the richer sensor mapping.

**Connection budget analysis**: With 15 inputs, each input node can connect to any hidden or output node. The minimum fully-connected topology requires `15 × 44 + 44 × 5 = 660 + 220 = 880` connections — far exceeding the 256-edge budget. The network must be **sparse**, which is normal for NEAT (it grows connections incrementally). But the 256-edge cap limits the network's expressive capacity. With 15 inputs, the initial topology has `15 + 5 = 20` nodes and 0 connections (standard NEAT minimal seed). NEAT adds connections through mutation, but the 256-edge ceiling may be reached before the network learns the full sensor→action mapping.

**Critical observation**: The worker's `new Neat(12, 5, fitnessFn, { popsize: 4, seed })` does NOT pass `maxNodes` or `maxConns` — it uses the defaults of `Infinity`. So the worker's NEAT population can grow unboundedly, while the NGE embryo is capped at 64/256. This means the worker's evolved champion may have **more than 64 nodes or 256 edges**, which would violate the NGE embryo's tier budget if the champion were ever converted to an embryo.

**Recommendation**: The research's Risks section (line 306) notes "The 64-node / 256-edge topology budget may not be enough to learn the richer sensor mapping" but doesn't quantify the impact. The analysis above shows:
1. The 256-edge budget is the binding constraint, not the 64-node budget — a 15-input network needs ~880 connections for full connectivity.
2. The worker's NEAT population doesn't enforce the 64/256 cap — it uses library defaults of Infinity. This should be either (a) explicitly set to 64/256 in the `new Neat()` options, or (b) documented as an accepted divergence between the embryo budget and the evolution budget.
3. If convergence stalls after the sensor expansion, the first lever to pull is **increasing maxEdges** (e.g., to 512), not maxNodes. The network needs more connections to map the richer sensor space, not more hidden nodes.

**Confidence**: 0.88 (confirmed from defaults constants + worker constructor options analysis)

---

### N5. lastShotHit as External Sensor vs Internal Recurrence — A Design Fork

**Finding**: The research proposes `lastShotHit` as sensor [14] — a binary flag representing whether the previous tick's shot connected. This is **external recurrent feedback**: the network observes the result of its previous action through the environment, not through internal network state.

**NEAT architecture perspective**: The Neatenstein NGE embryo pipeline supports `GatedRecurrentCell` as a motif archetype (`main-runner.ts:218`: "allowlisted motif archetypes (AttentionHead, GatedRecurrentCell, EpisodicSlot)"). This means the NGE embryo can include recurrent cell structures that maintain internal state across ticks. A network with a GatedRecurrentCell could learn to remember shot history internally, without needing the external `lastShotHit` sensor.

**Design tradeoff**:

| Approach | Mechanism | Pro | Con |
|----------|-----------|-----|-----|
| External sensor (proposed) | `lastShotHit` flag in sensor vector [14] | Simple, no topology change, deterministic, network doesn't need to evolve memory | One-tick delay only; network can't learn multi-tick shot patterns; adds 1 input node to fixed cost |
| Internal recurrence (GatedRecurrentCell) | Network evolves recurrent connections | Can learn arbitrary-length shot history; richer internal state; no input cost | Requires topology evolution to discover useful recurrence; slower convergence; the NGE embryo→Network converter doesn't exist yet (see N3) |
| Both | External sensor + evolved recurrence | Belt-and-suspenders; network can use the flag immediately and evolve richer memory later | 1 extra input; redundant signal if recurrence evolves |

**Recommendation**: Use the external sensor approach (as proposed) for the initial implementation. It's simpler, doesn't depend on the NGE embryo→Network converter, and provides immediate value. The internal recurrence path is a **follow-up** that becomes viable once the embryo converter is built. Document the GatedRecurrentCell motif as a future enhancement pathway — the NGE pipeline already has the infrastructure, but it's not wired to the evaluation network.

**Confidence**: 0.85 (confirmed GatedRecurrentCell motif allowlist + NGE embryo types)

---

### N6. Firing Gate Mechanism — Evolutionary Signal Preservation Analysis

**Finding**: The research proposes a "soft gate" (Solution 1, line 112): only suppress fire when NO enemy is visible at all. When an enemy IS visible but the network chooses not to fire, that's the network's decision.

**NEAT-specific concern**: A **hard gate** that always suppresses fire when the network says "fire" but the gate overrides would create a **zero-gradient dead zone** for the fire output. The network's output[3] activation would be ignored whenever the gate triggers, meaning mutations affecting output[3] receive no fitness signal during those ticks. Over many generations, this could cause the fire output to drift randomly since there's no selective pressure on it during gated ticks.

The **soft gate** approach avoids this: the network's fire output is respected whenever an enemy is visible. The selective pressure on output[3] comes from the fitness function's `aimMissRate` and `shotsWasted` penalties. When no enemy is visible, the gate suppresses fire, and the network's output[3] is ignored — but this is the correct behavior (firing at nothing is always wrong), so the lack of gradient during those ticks is acceptable.

**Subtle issue — the gate masks the fitness signal**: The research's Solution 1 tradeoff table (line 110) notes "if fire is always gated, aimMissRate stays 0 and doesn't differentiate variants." With the soft gate, `aimMissRate` only accumulates when an enemy IS visible. If the network fires at a visible enemy and misses, `aimMissRate` increases — this is the correct signal. But if the network never fires at visible enemies (learned behavior), `shotsFired` stays low and `aimMissRate` is undefined (division by zero → 0). The fitness formula should handle this edge case: `aimMissRate = shotsFired > 0 ? (shotsFired - shotsHit) / shotsFired : 0`.

**Recommendation**: The soft gate is architecturally sound for NEAT. However, the fitness function should explicitly handle the `shotsFired === 0` case to avoid NaN propagation. The `ammoEfficiency` metric (`damageDealt / shotsFired`) has the same issue. Both should use `max(shotsFired, 1)` as the denominator, which the research's Recommendation R2 (line 745) already proposes for `killEfficiency` but should be applied uniformly.

**Confidence**: 0.90 (NEAT gradient analysis + confirmed fitness formula structure)

---

### N7. Speciation Extinction — Acceptable but Must Be Documented

**Finding**: When the input count changes from 12→15, the entire NEAT population must be re-seeded. This means:
- All existing species die (their representatives have 12 input nodes; new genomes have 15)
- The compatibility distance between old and new genomes is maximally divergent (3+ excess input nodes + their connections)
- The innovation numbering space resets (new population starts with fresh innovation IDs)
- Speciation starts from a single species (all new genomes are structurally similar)

**NEAT architecture impact**: In standard NEAT, speciation protects innovation by assigning genomes to species and using explicit fitness sharing within species. When the population is re-seeded:
1. **Loss of diversity**: All accumulated structural diversity (different species, different topology strategies) is lost.
2. **Reset of adaptive threshold**: If the NEAT controller uses adaptive compatibility threshold (the library supports this — see `src/neat/adaptive/adaptive.ts:253`), the threshold resets to `DEFAULT_COMPATIBILITY_THRESHOLD = 3.0`.
3. **Loss of lineage history**: The `NeatLineage` tracking (referenced in `neat.diversity.ts:3-5`) loses all parent-child relationships.

**Mitigation**: Since the worker's fitness function is currently a **placeholder** (confirmed at `display.worker.ts:1319-1328`), the current "champions" are networks that were evolved against a fitness function that activates with zeros and returns `output[0]`. These champions have **no gameplay value** — their fitness is a random number. The extinction event loses nothing of value. The research correctly notes this at line 305: "the fallback AI is already designed for this case."

**Recommendation**: Document the speciation extinction explicitly in the plan as an accepted cost. The current champions are placeholder-fitness artifacts with no gameplay value. The re-seed is not a loss — it's a fresh start with a real fitness function (once the placeholder is replaced, per Solution 1 Layer B).

**Confidence**: 0.93 (confirmed placeholder fitness + NEAT speciation mechanics)

---

### N8. Worker NEAT Population Missing Topology Caps — Unbounded Growth Risk

**Finding**: The worker's NEAT population constructor (`display.worker.ts:1330-1335`) does not pass `maxNodes` or `maxConns` options:

```typescript
const neatPop = new Neat(
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_MAIN_NEAT_OUTPUTS,
  fitnessFn,
  { popsize: NEATENSTEIN_MAIN_NEAT_POPSIZE, seed: popSeed },
  // ← NO maxNodes, maxConns, maxGates
);
```

The library defaults are `DEFAULT_MAX_NODES = Infinity`, `DEFAULT_MAX_CONNS = Infinity`, `DEFAULT_MAX_GATES = Infinity` (`neat.defaults.constants.ts:179,188,196`). This means the worker's NEAT population can evolve networks with **unbounded topological complexity**. Over many generations, networks could grow to hundreds of nodes and thousands of connections.

**Contrast with NGE embryo**: The `main-runner.ts` NGE embryo pipeline explicitly caps at `maxNodes: 64, maxEdges: 256` (`main-runner.ts:209-212`). The worker's NEAT evolution is uncapped.

**Impact of sensor expansion**: With 15 inputs (vs 12), the network has more input nodes and potentially more initial connections. Uncapped evolution means the network could grow much larger than the 64-node embryo budget, creating a structural mismatch between the embryo (used for genome tracking) and the evolved champion (used for live gameplay).

**Recommendation**: Pass explicit topology caps to the worker's NEAT population:
```typescript
const neatPop = new Neat(
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_MAIN_NEAT_OUTPUTS,
  fitnessFn,
  {
    popsize: NEATENSTEIN_MAIN_NEAT_POPSIZE,
    seed: popSeed,
    maxNodes: 64,   // ← ADD: match NGE embryo budget
    maxConns: 256,  // ← ADD: match NGE embryo budget
  },
);
```

This ensures the evolved champion respects the same structural budget as the NGE embryo pipeline. Without this, the worker's champion may exceed the embryo's tier budget, creating an inconsistency if the champion is ever converted back to an embryo (future `NgeMainAgentEmbryo → Network` converter).

**Confidence**: 0.95 (confirmed from constructor options + default constants)

---

### N9. Sensor Normalization — The Unnormalized Position Sensor Problem

**Finding**: The research notes (line 309) that new binary sensors (`enemyVisible`, `enemyInFiringArc`, `lastShotHit`) are "binary [0,1] — consistent with existing sensor normalization." However, this masks a pre-existing normalization problem in the current sensor vector:

- **Sensor [3] player position X**: raw world units (0–120 range on a 120×120 map)
- **Sensor [4] player position Y**: raw world units (0–120 range)
- **Sensor [1] player ammo**: raw count (0–50 range)

These are **2-3 orders of magnitude larger** than the other normalized sensors:
- Sensors [0], [5], [7]: [0,1] range
- Sensor [2]: [−π, π] ≈ [−3.14, 3.14]
- Sensors [8-11]: wall distances in cells (0–30 range)

**NEAT impact**: Input nodes in Neataptic use **identity activation** (no squashing) — the raw input value is passed directly to the first hidden layer. The connection weights from input nodes scale the input, but large input values (100+) create large activation values in hidden nodes, which can saturate the hidden layer's activation function (typically `tanh` or `sigmoid`). When hidden nodes saturate, their gradient approaches zero, creating **dead neurons** that stop learning.

**Adding more binary sensors doesn't fix this** — the problem is the existing unnormalized position and ammo sensors. Adding 3 more binary [0,1] sensors is fine, but the research should flag the pre-existing normalization problem as a **parallel concern**. If the network struggles to learn the richer sensor mapping after the 12→15 expansion, unnormalized position/ammo sensors are a likely culprit.

**Recommendation**: Normalize ALL sensors to a consistent range:
- Position sensors [3-4]: divide by `NEATENSTEIN_MAP_SIZE` → [0,1] range
- Ammo sensor [1]: divide by `NEATENSTEIN_STARTING_AMMO` → [0,1] range
- Wall distance sensors [8-11]: divide by `NEATENSTEIN_RENDER_DISTANCE_CAP` → [0,1] range
- Bearing sensor [5]: already [−π,π] → divide by π → [−1,1] range

This is a **pre-existing issue**, not introduced by the sensor expansion. But the expansion is a natural opportunity to fix it. The fitness impact of normalization is significant — it changes the activation landscape, which means existing champions (already being re-seeded) would be incompatible anyway.

**Confidence**: 0.85 (standard NEAT activation analysis + confirmed sensor value ranges from source code)

---

### N10. Fitness Placeholder Is the Critical First Priority — No Sensor Change Matters Without It

**Finding**: The research identifies the placeholder fitness function as a problem (Solution 1, Layer B) and the Performance section quantifies the cost impact. But from a NEAT architecture perspective, the placeholder fitness is not just a problem — it is **the critical blocker** that makes all other sensor/firing changes irrelevant.

**NEAT-specific analysis**: The worker's `evaluateArmsRaceGeneration` fitness function (`display.worker.ts:1319-1328`) activates the network with a **zero input vector** and returns `output[0]`:

```typescript
const fitnessFn = (network: Network): number => {
  const output = network.activate(
    new Array(NEATENSTEIN_MAIN_NEAT_INPUTS).fill(0),
  );
  return output[0]!;
};
```

This means:
1. **The sensor vector is never used during evolution** — the network is activated with zeros, so changing from 12 to 15 inputs has **zero effect** on the evolved champion. The 15th, 14th, and 13th sensors would receive zero activation, contributing nothing.
2. **The fire output (output[3]) is never evaluated** — the fitness only looks at `output[0]` (the move.x output). There is no selective pressure on firing behavior at all.
3. **The champion is selected based on output[0] magnitude** — the fittest network is the one whose move.x output is largest when given zero inputs. This is essentially random — it has no relationship to combat ability.
4. **The `championMainNetwork` stored for live gameplay is a random network** — it was evolved against a fitness function that ignores all sensors and all outputs except output[0]. When this network is activated with real sensors during `buildAutoTickInput`, its behavior is unpredictable and almost certainly non-adaptive.

**Implication**: Changing the sensor vector from 12→15 inputs, adding LOS checks, adding `lastShotHit` feedback, revising fitness weights — **none of these changes have any effect on the worker's evolved champion** until the placeholder fitness is replaced with real episode-based evaluation. The sensor expansion only matters once the fitness function actually runs episodes with `extractSensors + network.activate + gameTick`.

**Recommendation**: **The fitness placeholder replacement must be the FIRST change, before any sensor expansion.** The research's Phase 3b sequencing (line 620) puts the worker fitness fix last — this is backwards. The correct sequence is:

1. **Phase 0 (NEW)**: Replace the placeholder fitness in `display.worker.ts` with real episode-based evaluation. This immediately makes the existing 12-input sensor vector meaningful.
2. **Phase 1**: Add LOS check, vision range, and new sensors (12→15 expansion).
3. **Phase 2**: Add telemetry, fix fitness weights, add fire gating.

This reordering ensures that the evolutionary pressure is active before the sensor expansion, so the network can learn the new sensor mapping with a real fitness gradient.

**Confidence**: 0.97 (confirmed from source code — the placeholder is unambiguous)

---

### NEAT Architecture Recommendations Summary

| ID | Recommendation | Priority | Confidence | Impact |
|----|---------------|----------|------------|--------|
| N1 | Document input count change as genome extinction event | Medium | 0.95 | Documentation clarity |
| N2 | Extract shared `neat-io-config.ts` module for input/output constants | **HIGH** | 0.97 | Eliminates drift risk |
| N3 | Document three-way network construction mismatch (embryo vs Network vs Neat) | Medium | 0.92 | Architectural clarity |
| N4 | Consider increasing `maxEdges` from 256→512 if convergence stalls after expansion | Low | 0.88 | Performance tuning lever |
| N5 | Use external `lastShotHit` sensor now; GatedRecurrentCell recurrence as follow-up | Medium | 0.85 | Design decision recorded |
| N6 | Soft fire gate is NEAT-sound; ensure `shotsFired===0` edge case in fitness | Medium | 0.90 | Prevents NaN in fitness |
| N7 | Speciation extinction is acceptable (current champions are placeholder artifacts) | Low | 0.93 | Risk acceptance documented |
| N8 | Pass `maxNodes: 64, maxConns: 256` to worker's `new Neat()` constructor | **HIGH** | 0.95 | Prevents unbounded growth |
| N9 | Normalize ALL sensors (position, ammo, wall distances) to [0,1] range | Medium | 0.85 | Prevents hidden node saturation |
| N10 | **Replace fitness placeholder FIRST, before any sensor expansion** | **CRITICAL** | 0.97 | All other changes are inert without this |

**Top priority**: N10 (fitness placeholder replacement) and N2 (shared constants) are blocking prerequisites. N8 (topology caps) should be done in the same pass as N10. N9 (sensor normalization) is a natural companion to the sensor expansion.

**Sequencing**: N10 → N2 + N8 → N9 + sensor expansion → fitness weight tuning → fire gating

---

## Live Gameplay Root-Cause Verification (2026-08-11)

### Questions

1. Why do `findNearestVisibleEnemy` and vision-range filtering appear to fail in live gameplay?
2. Is `extractSensors` in `enemy-navigation.ts` actually called by the real game loop, or only by tests?
3. Where does old omniscient enemy-detection code still run?

### Evidence

#### `extractSensors` is NOT test-only

Confirmed call sites in real production code paths:

| Consumer | File | Context | Uses vision filtering? |
|----------|------|---------|------------------------|
| Live champion auto-AI | `browser-entry/worker/display.worker.ts:1451` | `buildAutoTickInput` when `humanMode === 'auto' && championMainNetwork` is set | **Yes** — reads `sensors[ENEMY_VISIBLE_SENSOR_INDEX]` for soft fire gate |
| Fitness evaluation worker | `browser-entry/worker/eval.worker.ts:76` | `runFitnessEpisode` trains every evaluated network | **Yes** — full `extractSensors` vector |
| Headless main-agent runner | `browser-entry/harness/main-runner.ts:329` | `runEpisode` headless episode evaluation | **Yes** — full `extractSensors` vector |

It is also heavily exercised by `enemy-navigation.test.ts`, but the above three sites confirm it drives real runtime behavior.

**Confidence: 0.96** | Provenance: static-code + Jest test pass

#### `findNearestVisibleEnemy` is implemented and its unit tests pass

`scripts/enemy-navigation.ts:382` defines `findNearestVisibleEnemy` with the expected filters:

- Skips `active === false` enemies.
- Euclidean distance must be `<= VISION_RANGE_CELLS` (15 cells).
- `hasLineOfSight` from `browser-entry/renderer/raycast.ts:226` must return true.

Jest run for `enemy-navigation.test.ts` with filter `findNearestVisibleEnemy|extractSensors`:

```
Tests: 55 skipped, 25 passed, 80 total
Test Suites: 1 passed, 1 total
```

All vision-range, line-of-sight, inactive-enemy, and nearest-selection cases pass.

**Confidence: 0.94** | Provenance: static-code + runtime validation

#### The omniscient fallback auto-AI bypasses all vision filtering

`browser-entry/worker/display.worker.ts:1497` defines `buildFallbackAutoTickInput`. It selects targets with a raw Euclidean-distance loop:

```typescript
for (const enemy of state.enemies) {
  if (enemy.active === false) continue;
  const dx = enemy.position.x - px;
  const dy = enemy.position.y - py;
  const distSq = dx * dx + dy * dy;
  if (distSq < nearestDistSq) { ... }
}
```

There is **no `VISION_RANGE_CELLS` check** and **no `hasLineOfSight` check**. The fallback knows the nearest active enemy anywhere on the map, even through walls.

This fallback runs whenever `humanMode === 'auto'` and `championMainNetwork === null`:

```typescript
if (humanMode === 'auto' && championMainNetwork) {
  tickInput = buildAutoTickInput(...);     // vision-filtered champion path
} else if (humanMode === 'auto') {
  tickInput = buildFallbackAutoTickInput(gameState); // omniscient fallback
}
```

`championMainNetwork` starts as `null` and is only populated after `handleEvalComplete` receives a champion from the eval worker. That eval worker is only delegated after `advanceWave` is triggered by wave-clear. Prior logs (`neatenstein-auto-neat-mode.logs.md:808`) identified that the worker's wave-clear detection uses `enemies.length` instead of `allEnemiesCleared()`, so the champion may never be set, leaving the omniscient fallback running indefinitely.

**Confidence: 0.97** | Provenance: static-code

#### Enemy→player detection is separate and already has range + LOS

`scripts/enemy-controller.ts:960-974` shows enemy firing is gated by:

- `distToPlayer <= ENEMY_CONTROLLER_FIRE_RANGE_CELLS`
- `hasLineOfSight(position, gameState.player.position, collisionMap)`

This is enemy AI shooting at the player and is not the source of the reported player-vision bug.

**Confidence: 0.95** | Provenance: static-code

### Decision

The reported symptom is **not caused by a bug in `findNearestVisibleEnemy` or `extractSensors`**. Both are implemented correctly and are exercised by the champion path, the eval worker, and the headless runner. The symptom is caused by the **fallback auto-AI path** (`buildFallbackAutoTickInput`) which:

1. Runs whenever there is no champion network (which is the default initial state and can persist indefinitely due to the wave-clear detection bug).
2. Selects the nearest active enemy using raw Euclidean distance with no vision-range or line-of-sight check.
3. Therefore appears "omniscient" in live gameplay.

### Recommended Fix

Wire `buildFallbackAutoTickInput` to the same visibility primitive as the champion path:

- Use `findNearestVisibleEnemy(state, wallMap, NEATENSTEIN_MAP_SIZE, VISION_RANGE_CELLS)` to pick the fallback target.
- Only steer toward and fire at enemies that pass range + LOS.
- This makes the fallback AI respect the same vision model as the NEAT controller and removes the "enemy detected through walls" behavior.

A smaller alternative is to add a manual range/LOS check inside `buildFallbackAutoTickInput`, but reusing `findNearestVisibleEnemy` avoids duplicating the visibility logic.

### Risks

- The fallback AI currently relies on knowing enemy positions through walls to steer effectively. Capping it to vision range may make the fallback less effective at clearing the first wave, which could delay or prevent the first champion from being produced.
- If the wave-clear detection bug (`enemies.length` vs `allEnemiesCleared()`) is not fixed separately, the champion may never arrive regardless of fallback changes.
- Changing fallback target selection does not affect the sensor-vector size or network input count, so it is safe for existing champion networks.

### Confidence

Overall root-cause confidence: **0.95** — the code paths are unambiguous: the champion path uses vision filtering, the fallback path does not, and the fallback is the default path until a champion is produced.