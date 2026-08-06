## Slice 11-death-effects Visual Quality Validation (2026-08-06)

### Question

Validate the visual quality of slice `11-death-effects` (Tron-style pixel-by-pixel derez death animation) against the actual sprite data and render pipeline. Check: pixel-by-pixel dissolution of 48×48 sprite array will look good, scatter pattern (not row-by-row), neon gray tint subsumes item 6, 700ms timing, ROBOT_SPRITE_SCALE=4 means 4×4 screen blocks removed.

### Evidence

#### Sprite data (robot-sprite-data.js)

- `ROBOT_SPRITE_SCALE = 4` (line 6) — confirmed.
- `ROBOT_SPRITE_PALETTE` has 9 entries (line 8): index 0 = fully transparent, indices 1–3 = dark grays, index 4 = white, indices 5–6 = red team colors, index 7 = semi-transparent orange, index 8 = semi-transparent white.
- Frame `front.stand` has **551 non-zero pixels** out of ~2304 total → **23.4% fill rate**. The robot is mostly empty space (76.6% transparent).
- Each non-zero logical pixel maps to a **4×4 screen block** via `decodeRobotSpriteFrame` (sprites.ts:421–432), producing a 192×192 RGBA `VoxelSnapshot`.

#### Render pipeline (sprites.ts)

- `NEATENSTEIN_ANIMATION_TO_POSE` maps `death → 'stand'` (line 379). Currently the worker render path shows NO death visual — death enemies display the `stand` pose with no overlay.
- `renderNeatensteinVoxelSpriteColumn` (line 904) operates on the decoded 192×192 `VoxelSnapshot`, NOT the 48×48 logical grid. Its signature does NOT include `animationState`, `deRezElapsedMs`, `deRezDurationMs`, or `seed`.
- `activeEnemySprites` map (display.worker.ts:494–506) does NOT propagate `deRezElapsedMs` or `deRezDurationMs` — AC-11e-002 requires adding these.

#### Death color (AC-11e-005)

- `NEATENSTEIN_ENEMY_DEATH_COLOR = [180, 190, 210]` — cool light gray, slight cyan tint. NOT a saturated neon color.
- Lerp factor = `t * 0.5` (AC-11e-004), so maximum tint at t=1 (700ms end) is 50%.
- Example: dark gray pixel [18,20,24] lerp 50% → [99,105,117] = medium gray. Not luminous.

#### Timing (AC-11e-001)

- Current: `ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 4000` (enemy-controller.ts:188).
- Planned: 700ms. At 60fps = 42 frames, ~13 pixels/frame dissolve. At 30fps = 21 frames, ~26 pixels/frame.

#### Existing CPU/billboard death effect (enemy-sprite.ts)

- 4000ms duration, teal/orange bolt light tint (line 48, 84). Orange = `0xff8c00`.
- This is a color tint on the billboard renderer, NOT a pixel dissolution. The new slice replaces this with a pixel dissolve + gray tint in the worker voxel path.

### Decision

The slice is **implementable as designed** but has **five visual quality concerns**:

1. **Function signature gap (IMPLEMENTATION RISK).** `renderNeatensteinVoxelSpriteColumn` lacks the death-state parameters. The derez mask needs `animationState`, `deRezElapsedMs`, `deRezDurationMs`, and `seed` threaded through the call chain. Alternatively, the mask could be pre-applied at decode time.

2. **Mask resolution mapping.** The 48×48 hash mask operates on logical coordinates, but the column function works at 192×192 display resolution. Each screen pixel must map back to its logical pixel (`floor(frameY / 4)`, `floor(frameX / 4)`). AC-11e-007 claims O(2304) but the column function iterates O(192 × visibleColumns) — the hash is recomputed per screen row unless cached.

3. **Hash quality is the primary visual risk.** AC-11e-006 requires "visually scattered (seeded spatial noise, not row-by-row)." A naive hash produces visible diagonal/structured artifacts. A proper integer hash with prime mixing is required. The AC does not specify hash quality — implementer must choose a good hash.

4. **Neon-gray tint is subtle, not luminous.** `[180,190,210]` is muted gray. At max 50% lerp, produces medium gray, not a neon glow. For true Tron aesthetic, consider increasing lerp cap, brightening the color, or adding edge glow. Accepted trade-off: dissolution IS the Tron effect.

5. **Sparse sprite dissolution.** At 23.4% fill rate, the robot is mostly empty space. As pixels dissolve, the remaining structure becomes increasingly skeletal. At t≈0.5, only ~275 pixels remain scattered across 48×48 — may read as "vanishing noise" rather than "dissolving robot." Consider biasing interior pixels to dissolve first to preserve outline longer (optional).

### Risks

- **Hash quality** is the single biggest visual risk. Poor hash → structured artifacts.
- **Signature threading** — column function needs new parameters or pre-applied decode mask.
- **Subtle tint** may not satisfy "neon gray" expectation — 50% lerp produces medium gray.
- **Sparse fill rate** — dissolve may look noisy rather than shaped mid-animation.
- **No visual validation** — plan mandates jest-only validation. Visual quality (hash scatter, tint visibility, timing feel) requires human/browser review.

---

# Research: Enemy Pushback + Stun on Shot

## Question (original — pushback/stun)

How to implement shot enemy pushback + stun in the Neatenstein demo (`examples/neatenstein/`).
When an enemy is shot, it should be pushed back a bit and stunned for ~0.2s — invincible
during stun but doesn't move. The enemy AI (MLP activation) should be skipped during stun.

## Evidence

### 1. Cortex Search Results

- **freshness_check**: PASSED (index fresh, timestamp 1786049125497).
- **search_corpus**: ALL calls FAILED with `SQLITE_BUSY: database is locked`. Degraded
  to native tools (view, Select-String) per Cortex-first fallback policy.
- **Native grep** for `stun`, `pushback`, `knockback`, `invincible`, `invulnerable`,
  `freeze`, `immobile`, `hitReact`, `flash` across key enemy files: **NONE found**.
  These concepts do NOT yet exist in the enemy system.

### 2. Current Enemy State Model

#### EnemyState (authoritative game state)
**File:** `examples/neatenstein/browser-entry/host/game/types.ts` (lines 51-60)

```typescript
interface EnemyState {
  position: Vector2;
  health: number;
  active?: boolean;
  controllerPosition?: Vector2;
}
```

Minimal — no stun, knockback, or invincibility fields. This is the state that
`applyEnemyDamage` modifies and that persists across game ticks.

#### ControlledEnemy (controller working state)
**File:** `examples/neatenstein/scripts/enemy-controller.ts` (lines 50-90)

```typescript
interface ControlledEnemy {
  index: number;
  position: Vector2;
  health: number;
  yawRad: number;
  animationState: 'idle' | 'move' | 'fire' | 'death';
  ammo: number;
  fireCooldownMs: number;
  deRezElapsedMs: number;
  active: boolean;
  walkTick: number;
  shootBlinkTicks: number;
  flankStallTicks: number;
  bfsStallTicks: number;
  weights: number[] | undefined;
  variantId: number;
  previousStepDistance: number;
}
```

Richer per-enemy AI/animation state, but NO stun/pushback/invincible/knockback fields.
Note: `animationState` type is `'idle' | 'move' | 'fire' | 'death'` — does NOT include
`'damage'` (which exists in the animator type but is never set on ControlledEnemy).

#### EnemyAnimationState (animator type)
**File:** `examples/neatenstein/scripts/enemy-animator.ts` (line 17)

```typescript
type EnemyAnimationState = 'idle' | 'move' | 'fire' | 'death' | 'damage';
```

A `'damage'` state ALREADY EXISTS — described as "two-frame overlay used for hit-flash
feedback" with 2 frames (`ENEMY_ANIMATION_FRAME_COUNTS.damage = 2`). However,
`ControlledEnemy.animationState` restricts to the first four states only.

#### State Machine
There is no explicit state machine enum. State transitions are implicit in
`updateControlledEnemy`:
- `health <= 0` → death/de-rez animation path (line 398)
- Moving → `animationState = 'move'`
- Firing → `animationState = 'fire'`, `shootBlinkTicks = 3`
- Otherwise → `animationState = 'idle'`

No "stunned" or "hit" state exists.

### 3. Enemy Movement System

**File:** `examples/neatenstein/scripts/enemy-controller.ts`

- **`updateControlledEnemy()`** (lines 344-966): Per-tick enemy update.
  - **Death path** (line 398): If `health <= 0`, runs de-rez animation, returns early.
  - **Alive path** (line 422+): Computes `dxToPlayer`, `dyToPlayer` (lines 423-425).
    BFS distance map navigation + flanking toward player.
  - **Movement speed**: `ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND = 2.5` cells/sec.
    Step distance = speed × dtMs/1000.
  - **Movement directions**: Cardinal [0,-1],[1,0],[0,1],[-1,0] sorted by BFS distance.
  - **MLP re-ranking** (lines 604-635): When `weights !== undefined && dtMs > 0`,
    calls `activateMlp(weights, visionVector)` to re-rank BFS candidates.
  - **Fire logic** (lines 700+): Hitscan fire toward player.
  - **Walk tick update** (lines 900+): `walkTick++` when moving.

- **`updateEnemyController()`** (lines 1067-1113): Main entry. Builds BFS distance map,
  calls `updateControlledEnemy` per enemy, then `separateEnemies()`.

- **`separateEnemies()`** (lines 985-1050): Pushes overlapping enemies apart with
  wall re-check. **This is the closest existing pattern to pushback** — it does
  position offsets with wall collision checks.

- **`createEnemyControllerState()`** (lines 305-329): Initializes fresh controller state.

### 4. Bullet → Enemy Collision / Damage Pipeline

TWO damage application paths exist, both calling `applyEnemyDamage`:

#### Path 1: Immediate Hitscan (`fireBolt`)
**File:** `examples/neatenstein/browser-entry/host/game/combat.ts` (lines 233-235)
```typescript
if (hitType === 'enemy' && hitEnemyIndex >= 0) {
  nextState = applyEnemyDamage(nextState, hitEnemyIndex);
}
```

#### Path 2: Traveling Bolt (`updateBolts` → `gameTick`)
**File:** `examples/neatenstein/browser-entry/host/game/tick.ts` (lines 252-267)
```typescript
const updatedBolts = updateBolts(...);
for (const bolt of updatedBolts) {
  if (!bolt.active && bolt.hitEnemyIndex !== undefined && bolt.hitEnemyIndex >= 0) {
    next = applyEnemyDamage(next, bolt.hitEnemyIndex);
  }
}
```

#### applyEnemyDamage (the key modification point)
**File:** `examples/neatenstein/browser-entry/host/game/combat.ts` (lines 296-313)
```typescript
export function applyEnemyDamage(state: GameState, enemyIndex: number): GameState {
  const enemy = state.enemies[enemyIndex];
  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BOLT_DAMAGE);
  const killedByThisShot = newHealth === 0;
  const newEnemies = state.enemies.map((existing, index) =>
    index === enemyIndex ? { ...existing, health: newHealth } : existing,
  );
  return { ...state, enemies: newEnemies, kills: ... };
}
```

Currently only modifies health and kills. **This is where stun timer initialization
and knockback direction computation should go.** The function has access to
`state.player.position` and `enemy.position`, so pushback direction =
`normalize(enemy.position - player.position)`.

### 5. Pushback Direction

- Player position: `state.player.position` (Vector2)
- Enemy position: `enemy.position` (Vector2)
- `dxToPlayer` / `dyToPlayer` computation already exists in `updateControlledEnemy`
  (lines 423-425): `dxToPlayer = player.x - position.x`, `dyToPlayer = player.y - position.y`
- Pushback direction = **normalize(enemy.position - player.position)** = negation of
  the player-to-enemy vector. The enemy is pushed AWAY from the player.

### 6. Player Invulnerability Pattern (reference for enemy stun)

**File:** `examples/neatenstein/browser-entry/host/game/state.ts` (lines 118-123)
```typescript
function isInvulnerable(state): boolean {
  return state.player.dashTimeRemainingMs > 0 || state.player.contactIFrameMs > 0;
}
```

Timer-based invulnerability. Timers are decremented each tick via `tickTimerMs()`.
This pattern is directly applicable to enemy stun: add a `stunTimerMs` field, decrement
each tick, skip movement while > 0.

### 7. MLP AI During Stun

**File:** `examples/neatenstein/browser-entry/harness/enemy-mlp.ts` (lines 167-202)

`activateMlp()` is called inside `updateControlledEnemy()` at line 620 during BFS
re-ranking. During stun, the entire movement/AI block should be skipped (before line 523
where `shouldMoveByBfs || shouldMoveByFlank` is evaluated), which naturally skips
`activateMlp` as well. No changes needed to `enemy-mlp.ts` itself.

### 8. Enemy Rendering

#### Sprite Mapping (display.worker.ts)
**File:** `examples/neatenstein/browser-entry/worker/display.worker.ts` (lines 494-506)
```typescript
activeEnemySprites = enemyControllerState!.enemies
  .filter((enemy) => enemy.active)
  .map((enemy) => ({
    worldX: enemy.position.x,
    worldY: enemy.position.y,
    facing: enemy.yawRad,
    animationState: enemy.animationState,  // ← forwarded to renderer
    frameIndex: 0,
    type: enemy.index,
    walkTick: enemy.walkTick,
    shootBlinkTicks: enemy.shootBlinkTicks,
    teamColor: resolveEnemyTeamColor(enemy.index),
  }));
```

The `animationState` from `ControlledEnemy` is forwarded directly to the
`NeatensteinSprite` for rendering.

#### Animation → Pose Map (sprites.ts)
**File:** `examples/neatenstein/browser-entry/renderer/sprites.ts` (lines 372-381)
```typescript
const NEATENSTEIN_ANIMATION_TO_POSE: Record<EnemyAnimationState, ...> = {
  idle: 'stand',
  move: 'walk1',
  fire: 'shoot',
  death: 'stand',
  damage: 'stand',   // ← already mapped, renders as 'stand' (freeze frame)
};
```

The `'damage'` state already maps to `'stand'` pose — a stunned enemy would freeze
in the standing pose. The 2-frame damage overlay (red/white flash) is already defined
in `ENEMY_ANIMATION_FRAME_COUNTS.damage = 2`.

#### Visual Approach Options
1. **Reuse `'damage'` state**: Set `animationState = 'damage'` during stun. The enemy
   freezes in 'stand' pose with the existing 2-frame hit-flash overlay. Minimal
   rendering changes — just update the `ControlledEnemy.animationState` type to
   include `'damage'` and set it during stun.
2. **Add new `'stunned'` state**: Add to `EnemyAnimationState`, add to
   `NEATENSTEIN_ANIMATION_TO_POSE`, add frame count. More work but allows distinct
   visual (e.g., stars/dizzy effect).

### 9. Episode Runner (enemy-runner.ts)

**File:** `examples/neatenstein/browser-entry/harness/enemy-runner.ts` (lines 221-351)

`simulateEnemyEpisode()` is a headless evaluation rollout for enemy evolution. It
uses local variables (enemyX, enemyY) and does NOT use the full `gameTick` pipeline.
The static player in the evaluation does NOT shoot at enemies — so enemies are never
hit during evaluation.

**Stun/pushback does NOT need to be added to the enemy-runner** unless the evaluation
is extended to include player combat against the evaluated enemy. Currently, the
runner only evaluates navigation and enemy-to-player firing.

### 10. Constants

**File:** `examples/neatenstein/browser-entry/host/game/constants.ts`

New constants needed:
- `NEATENSTEIN_ENEMY_STUN_DURATION_MS = 200` — stun duration (~12-13 ticks at 16ms)
- `NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS` — instant pushback distance (e.g., 0.3-0.5
  cells), OR `NEATENSTEIN_ENEMY_PUSHBACK_SPEED_CELLS_PER_SECOND` for velocity-based
  decaying pushback
- `NEATENSTEIN_ENEMY_STUN_INVINCIBLE = true` — flag for invincibility during stun

`NEATENSTEIN_FIXED_TIMESTEP_MS = 16` — at 60fps, 200ms stun ≈ 12-13 ticks.

### 11. Architecture: State Flow

```
GameState.enemies (EnemyState[])     ← authoritative: health, position, active
    ↓ applyEnemyDamage (combat.ts)   ← SET stun timer + knockback here
    ↓ gameTick (tick.ts)             ← traveling bolt path also calls applyEnemyDamage
    ↓
EnemyControllerState.enemies         ← controller working state
  (ControlledEnemy[])                ← READ stun, SKIP movement + MLP + fire
    ↓ updateControlledEnemy          ← decrement stun timer, apply knockback decay
    ↓
Display Worker                       ← maps ControlledEnemy → NeatensteinSprite
    ↓                                ← animationState = 'damage' for stun visual
Renderer (sprites.ts)                ← renders freeze frame + hit-flash
```

**Key Insight**: The display worker calls `updateEnemyController` BEFORE `gameTick`
(line 1147), then syncs controlled positions back to `gameState.enemies` (line 1157),
then calls `gameTick` which may apply damage via `applyEnemyDamage`. After gameTick,
it calls `updateEnemyController` again with dtMs=0 (line 1195) to sync health changes.

This means:
- `applyEnemyDamage` sets stun on `EnemyState` (authoritative).
- The post-tick controller pass (dtMs=0) reads the new health and should also read
  the stun timer from `EnemyState`.
- The next frame's pre-tick controller pass (with real dtMs) decrements the stun
  timer and skips movement.

## Decision

### Proposed Implementation

#### Pushback: Instantaneous Position Offset
- On hit in `applyEnemyDamage`, compute pushback direction =
  `normalize(enemy.position - player.position)`.
- Apply immediate position offset: `enemy.position += pushbackDir * PUSHBACK_DISTANCE`.
- Wall-check the new position (reuse the `isWallAt` pattern from `separateEnemies`).
- This is simpler than velocity-based decay and matches the "pushed back a bit"
  requirement.

#### Stun: Timer-based, mirrors player invulnerability
- Add `stunTimerMs?: number` to `EnemyState` (set by `applyEnemyDamage`).
- Add `stunTimerMs: number` to `ControlledEnemy` (synced from `EnemyState`,
  decremented each tick).
- In `updateControlledEnemy`, before the movement block (before line 523):
  - If `stunTimerMs > 0`: skip movement, skip MLP activation, skip fire.
  - Set `animationState = 'damage'` (reuse existing hit-flash state).
  - Decrement `stunTimerMs` by `dtMs`.
- Invincibility during stun: in `applyEnemyDamage`, check `enemy.stunTimerMs > 0`
  and skip damage application (early return) — OR check in the collision detection
  before calling `applyEnemyDamage`.

#### Rendering
- Reuse the existing `'damage'` animation state (freeze frame + 2-frame hit-flash).
- Update `ControlledEnemy.animationState` type to include `'damage'`.
- No sprite renderer changes needed — `NEATENSTEIN_ANIMATION_TO_POSE.damage` already
  maps to `'stand'`.

### All Impacted Files

| # | File | Change |
|---|------|--------|
| 1 | `host/game/types.ts` | Add `stunTimerMs?: number` to `EnemyState` |
| 2 | `host/game/constants.ts` | Add `NEATENSTEIN_ENEMY_STUN_DURATION_MS`, `NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS` |
| 3 | `host/game/combat.ts` | Modify `applyEnemyDamage`: compute pushback direction, set `stunTimerMs`, apply position offset with wall check, skip if already stunned (invincibility) |
| 4 | `scripts/enemy-controller.ts` | Add `stunTimerMs` to `ControlledEnemy` interface; in `updateControlledEnemy`: stun check before movement block (skip movement + MLP + fire), decrement timer, set `animationState = 'damage'`; update `createEnemyControllerState` defaults |
| 5 | `scripts/enemy-animator.ts` | No change needed — `'damage'` state already exists. Optionally add `'stunned'` state if distinct visual desired. |
| 6 | `browser-entry/renderer/sprites.ts` | No change needed — `NEATENSTEIN_ANIMATION_TO_POSE.damage` already maps to `'stand'`. Optionally add stun-specific visual effect. |
| 7 | `browser-entry/worker/display.worker.ts` | Update `__testOnlyInjectTestEnemies` to include `stunTimerMs: 0` in test enemy construction. No functional change needed — `animationState` already forwarded. |
| 8 | `host/game/tick.ts` | No direct change — `applyEnemyDamage` is already called for both hit paths. Stun/pushback handled inside `applyEnemyDamage`. |
| 9 | `browser-entry/harness/enemy-runner.ts` | No change needed — evaluation doesn't model player shooting at enemies. |
| 10 | `browser-entry/harness/enemy-mlp.ts` | No change — MLP skip is handled at the call site in `enemy-controller.ts`. |

### Test Files Needing Updates

| File | Reason |
|------|--------|
| `host/game/combat.test.ts` | Test `applyEnemyDamage` now sets stun + pushback |
| `host/game/tick.test.ts` | Test traveling bolt path applies stun |
| `host/game/types.test.ts` | Test `EnemyState` with stun field |
| `host/game/constants.test.ts` | Test new constants exist |
| `scripts/enemy-controller.test.ts` | Test stunned enemy doesn't move, MLP skipped, timer decremented |
| `browser-entry/worker/display.worker.test.ts` | Test stunned enemy renders with 'damage' state |
| `scripts/enemy-animator.test.ts` | Test 'damage' animation state for stun (may already pass) |
| `browser-entry/renderer/sprites.test.ts` | Test 'damage' state renders as 'stand' pose (may already pass) |

## Risks

1. **Position desync**: `applyEnemyDamage` modifies `EnemyState.position` but the
   `EnemyControllerState` maintains its own `ControlledEnemy.position`. The display
   worker syncs controlled → gameState BEFORE gameTick, then gameState → controller
   AFTER. If pushback is applied to `EnemyState.position` in `applyEnemyDamage` (during
   gameTick), the next controller pass needs to read and adopt the pushed-back
   position. **Mitigation**: The post-tick controller pass (dtMs=0, line 1195) should
   sync the pushed-back position from `EnemyState` to `ControlledEnemy`.

2. **Stun timer persistence**: The stun timer must survive the controller → gameState →
   controller round-trip each tick. If it lives only on `ControlledEnemy`, it resets
   when the controller state is rebuilt. **Mitigation**: Store `stunTimerMs` on
   `EnemyState` (authoritative), sync to `ControlledEnemy` each tick.

3. **Wall collision during pushback**: Pushing an enemy into a wall could clip it
   through geometry. **Mitigation**: Reuse the wall-check pattern from
   `separateEnemies()` — check `isWallAt` at the target position and clamp.

4. **SeparateEnemies interaction**: `separateEnemies()` runs after all enemies
   update. A stunned enemy that doesn't move could be pushed by separation forces
   from other enemies. **Mitigation**: Skip stunned enemies in `separateEnemies()`
   or let them be pushed (minor, only affects visual).

5. **Double-tick stun**: The display worker calls `updateEnemyController` twice per
   frame (once with real dtMs before gameTick, once with dtMs=0 after). The dtMs=0
   pass won't decrement the stun timer (dtMs=0 → no decrement), so no double-tick
   issue.

6. **Bolt damage = 50, enemy health = 100**: Currently one shot kills. With stun +
   invincibility, the first shot stuns (and deals damage), the enemy survives, and
   the second shot (after stun ends) kills. This changes gameplay balance. The
   bolt damage or stun duration may need tuning. Alternatively, stun + invincibility
   only triggers when the enemy survives the hit (health > 0 after damage).

## Confidence

- **0.95** — Current state model, movement system, collision pipeline, rendering path.
  All verified by direct file reads.
- **0.90** — Proposed implementation approach. The architecture is clear and the
  `applyEnemyDamage` + `updateControlledEnemy` modification points are well-defined.
- **0.80** — Pushback position sync between EnemyState and ControlledEnemy. The
  display worker's dual-pass controller update is complex and the exact sync point
  for pushback position needs careful implementation.

---

# Research: Bullet Explosion + Enemy Impact Marks

## Question

How to make bolts (plasma bullets) explode on enemy impact and leave a visual mark
on the enemy, similar to how wall impact spots (bullet holes) work. Currently bolts
hit enemies (damage IS applied) but produce NO visual effect at the hit point.

## Evidence

### 1. Cortex Search

- **freshness_check**: FAILED with `SQLITE_BUSY: database is locked`.
- **search_corpus**: ALL calls FAILED with `SQLITE_BUSY`. Degraded to native tools
  (view, Select-String) per Cortex-first fallback policy.

### 2. Current Bullet (Bolt) System Architecture

The weapon system uses "plasma bolts" (`BoltState`), not "bullets".

#### BoltState type (`host/game/types.ts` lines 69-88)
```typescript
interface BoltState {
  position: Vector2;
  direction: Vector2;
  speedCellsPerSecond: number;
  active: boolean;
  createdAtMs: number;
  origin?: Vector2;
  targetDistance?: number;
  radius?: number;
  hitEnemyIndex?: number;  // ← set when bolt hits an enemy
}
```

#### Firing (`host/game/combat.ts: fireBolt()` lines 121-242)
- Ammo check → raycast from muzzle along yaw → DDA wall hit → enemy cylinder test
- Enemy collision IS implemented (lines 157-182): tests each living enemy via
  `projectOntoRay` + `perpendicularDistance` within `NEATENSTEIN_BOLT_HIT_RADIUS_CELLS`.
- `ImpactSpot` created ONLY when `hitType === 'wall'` (lines 207-231).
- When `hitType === 'enemy'`: `applyEnemyDamage()` called, NO visual effect created.

#### Traveling (`host/game/tick.ts: updateBolts()` lines 439-511)
- Bolts move at 36 cells/s, 300ms fixed visual travel duration.
- `findBoltEnemyImpact()` checks enemies against bolt path per tick.
- On enemy hit: bolt deactivates, `hitEnemyIndex` set, damage applied in `gameTick()`.

#### Damage model (`combat.ts: applyEnemyDamage()` lines 296-313)
- `NEATENSTEIN_BOLT_DAMAGE` = 50, enemy default health = 100.
- 2 hits to kill. Kill counter increments on health→0 transition.

### 3. Wall Impact Mark Mechanism (the reference behavior)

#### ImpactSpot type (`host/game/types.ts` lines 90-117)
```typescript
interface ImpactSpot {
  wallHit: { mapX: number; mapY: number; side: 0 | 1; wallX: number };
  position: Vector2;
  createdAtMs: number;
  lifetimeMs: number;       // 3000ms
  perpWallDist: number;
  boltTravelTimeMs: number; // spot invisible until bolt arrives
}
```

#### Rendering (`renderer/bolt-render.ts: drawImpactSpots()` lines 105-178)
- Neon circles at wall positions using `globalCompositeOperation = 'lighter'`
- Shadow blur glow, radius scales with distance, alpha fades with lifetime
- Z-buffer depth tested
- Only visible after bolt travel time completes (`travelRatio >= 1`)

#### Constants (`browser-entry/constants.ts` lines 72-93)
- `NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS` = 3000ms
- `NEATENSTEIN_IMPACT_SPOT_RADIUS_PX` = 4px
- `NEATENSTEIN_IMPACT_SPOT_COLOR` = '#f0f8ff'
- `NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR` = 'rgba(240,248,255,0.5)'
- `NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX` = 2

#### Paint order (`display.worker.ts` line 443)
```
clear → floor → ceiling → walls → sprites → sprite flush → pulses → impact spots → bolts → gun
```

#### Lifecycle
- Created in `fireBolt()` only for wall hits.
- Aged by `ageImpacts()` in `gameTick()` — lifetime decremented per tick.
- Removed when lifetime ≤ 0.

### 4. Enemy Collision Gap

**The core gap**: When a bolt hits an enemy, NO `ImpactSpot` or visual effect is
created. `fireBolt()` creates `ImpactSpot` ONLY for `hitType === 'wall'`. The
`hitEnemyIndex` on `BoltState` is used by `drawBolts()` to position the bolt's
visual endpoint at the enemy, but there is no explosion, burst, or persistent mark.

No explosion or particle system exists anywhere in the codebase. The only impact
visuals are wall impact spots.

### 5. Enemy Rendering Pipeline

#### Worker tier (`display.worker.ts` lines 679-735)
- `getImageData` → `renderNeatensteinSprite` per sorted sprite → `putImageData`
- Pre-rendered voxel frames from `robot-sprite-data.js`
- Team colors applied via palette swap (indices 5/6/7)

#### Sprite type (`renderer/sprites.ts: NeatensteinSprite` lines 128-147)
```typescript
interface NeatensteinSprite {
  worldX: number; worldY: number;
  facing?: number;
  animationState?: EnemyAnimationState;
  walkTick?: number;
  shootBlinkTicks?: number;
  teamColor?: readonly [number, number, number];
}
```

#### Animation states (`scripts/enemy-animator.ts`)
- `'idle' | 'move' | 'fire' | 'death' | 'damage'`
- `damage` state has 2 frames (red/white hit-flash overlay) but maps to `'stand'`
  pose in `NEATENSTEIN_ANIMATION_TO_POSE` (sprites.ts line 380) and is NEVER triggered.

#### Robot sprite data (`robot-sprite-data.js`)
- 8 directions × 4 poses (stand, walk1, walk2, shoot)
- `ROBOT_SPRITE_PALETTE` with swappable indices 5/6/7
- Muzzle-blast uses semitransparent indices 7/8

### 6. Audio

- `enemy-hit` sound IS defined in `audio.ts` (square wave, 330Hz) but is NOT
  wired to any game logic trigger. Audio system is standalone, not integrated.

## Decision

### Proposed Implementation: Enemy Impact Spots + Burst Effect

Mirror the wall impact spot system for enemy hits, adding two visual layers:

#### Layer 1: Enemy Impact Spot (persistent mark)
- Add `EnemyImpactSpot` type (or extend `ImpactSpot` with optional `enemyIndex`).
  - Fields: `position: Vector2`, `createdAtMs`, `lifetimeMs` (~1000ms, shorter
    than wall spots since enemy moves), `boltTravelTimeMs`, `enemyIndex`.
- In `fireBolt()`, when `hitType === 'enemy'`, create an enemy impact spot at the
  enemy's world position.
- Render via new `drawEnemyImpactSpots()` in `bolt-render.ts` — neon circle at
  enemy position, projected same as bolts/pulses, z-buffer tested, additive blend.
- Age in `gameTick()` similar to `ageImpacts()`.

#### Layer 2: Explosion Burst (short-lived visual)
- A brief expanding neon circle/burst at the impact point, larger than the wall
  spot. Could be a separate `EnemyBurst` type or a phase of the impact spot
  (expanding radius over first ~200ms, then fading).
- Rendered in the same additive-blend pass as impact spots.
- Uses a burst-specific color (e.g., the bolt's teal `#00f0ff` for plasma
  explosion) and larger initial radius.

#### Layer 3 (optional): Damage Flash on Sprite
- Wire the existing `damage` animation state when an enemy is hit.
- Set `animationState = 'damage'` for a few ticks on the `ControlledEnemy`.
- Modify `NEATENSTEIN_ANIMATION_TO_POSE` or add damage-specific overlay frames.
- This requires coordinating with `enemy-controller.ts` which manages animation
  state transitions.

### Impacted Files

| # | File | Change |
|---|------|--------|
| 1 | `host/game/types.ts` | Add `EnemyImpactSpot` type; add `enemyImpacts` to `GameState` |
| 2 | `host/game/constants.ts` | Add enemy impact constants (lifetime, burst radius, burst duration) |
| 3 | `host/game/combat.ts` | In `fireBolt()`, create enemy impact spot when `hitType === 'enemy'` |
| 4 | `host/game/tick.ts` | Add `ageEnemyImpacts()` and call in `gameTick()`; optionally wire `damage` state |
| 5 | `renderer/bolt-render.ts` | Add `drawEnemyImpactSpots()` rendering function |
| 6 | `worker/display.worker.ts` | Call enemy impact rendering in paint order (after sprites, with/after wall impacts) |
| 7 | `browser-entry/constants.ts` | Add enemy impact visual constants (color, glow, radius) |
| 8 | `renderer/sprites.ts` | Optionally: wire `damage` animation state for hit-flash overlay |
| 9 | `host/game/combat.test.ts` | Test enemy impact spot creation on hit |
| 10 | `host/game/tick.test.ts` | Test enemy impact spot aging and removal |
| 11 | `renderer/bolt-render.test.ts` | Test enemy impact spot rendering |
| 12 | `worker/display.worker.test.ts` | Test enemy impact rendering in frame |

## Risks

1. **Determinism**: All new visual state must be deterministic and seedable. Impact
   spots must use `simTimeMs` for creation/aging, not `Date.now()`.
2. **Performance**: Additional rendering passes add per-frame cost. Enemy impact
   spots should be capped (like `NEATENSTEIN_PULSE_MAX_CONCURRENT = 40`).
3. **Moving target alignment**: Enemy impact spots are at enemy world positions at
   time of hit, but enemies move. The spot should be a brief screen-space effect,
   not permanently attached to the enemy sprite.
4. **Paint order**: Enemy impact spots should render after sprites (appear on top
   of enemy) but before bolts (bolt's final frame shouldn't obscure explosion).
5. **Two damage paths**: Both `fireBolt()` (immediate) and `updateBolts()` (traveling)
   can trigger enemy hits. Impact spots should be created in both paths.
6. **Damage animation coordination**: If wiring the `damage` animation state, the
   enemy controller may override it on the next tick. Stun timer (from prior
   research) may be needed to persist the damage visual.

## Confidence

- **0.95** — Current bolt system, wall impact mechanism, enemy rendering pipeline.
  All verified by direct file reads.
- **0.90** — Proposed enemy impact spot approach. The wall impact spot pattern is
  a direct template; the main work is creating the enemy-position variant.
- **0.75** — Damage flash animation wiring. The `damage` state exists but has never
  been triggered; controller coordination is uncertain.

---

# Research: Enemy Return Fire (Item 3 — Enemies Shoot Back)

## Question

How to make enemies shoot visible projectiles at the player in the Neatenstein demo
(`examples/neatenstein/`). Enemy shots should explode on player impact and reduce
player health by 10%. Research only — no file edits.

## Evidence

### 1. Cortex Search Results

- **freshness_check**: Index fresh (timestamp 1786044125497).
- **search_corpus**: First query (broad "shoot fire projectile bullet") SUCCEEDED —
  returned `plans/Neon_Shooter_NGE_Demo.plans.md`. Five parallel follow-up queries ALL
  failed with `SQLITE_BUSY: database is locked`.
- **Fallback**: Native tools (view, Select-String) used for known file paths —
  documented as degraded-Cortex fallback per Cortex-first policy.

### 2. Player Shooting System (Template for Enemy Bolts)

**File:** `examples/neatenstein/browser-entry/host/game/combat.ts` (lines 121-160)

`fireBolt()` creates `BoltState` objects from the player's gun muzzle:
- Origin: player position + gun muzzle offset (screen ratios 0.5, 0.82)
- Direction: player yaw
- Speed: `NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND = 36`
- Max range: `NEATENSTEIN_BOLT_MAX_RANGE_CELLS = 30`
- Damage: `NEATENSTEIN_BOLT_DAMAGE = 50` (currently one-hit overkill)
- Lifetime: 300ms fixed screen-travel duration

**BoltState** (`types.ts` lines 68-88):
```typescript
interface BoltState {
  id: number;
  originCell: Vector2;
  direction: Vector2;
  damage: number;
  alive: boolean;
  travelTimeMs: number;
  maxRangeCells: number;
}
```

### 3. Player Health System

**File:** `examples/neatenstein/browser-entry/host/game/state.ts`

---

## Slice 11-death-effects Rendering Mechanics Validation (2026-08-06)

### Question

Validate the derez rendering mechanics described in slice `11-death-effects` of
`plans/Neon_Shooter_NGE_Demo.plans.md` against the actual source code in
`examples/neatenstein/browser-entry/renderer/sprites.ts`. Check: (1) seeded noise
dissolution approach, (2) hash(frameX,frameY,seed) threshold, (3)
renderNeatensteinVoxelSpriteColumn integration, (4) performance O(2304).

### Evidence

#### 1. Slice status: [PLANNED] — not yet implemented

- `derez.ts` and `derez.test.ts` do NOT exist.
- `NEATENSTEIN_ENEMY_DEATH_COLOR` does NOT exist in `host/game/constants.ts`.
- `ENEMY_CONTROLLER_DE_REZ_DURATION_MS` is still 4000 (not 700).
- `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS` is also 4000 in `scripts/enemy-sprite.ts`.

#### 2. VoxelSnapshot dimensions: 192×192, not 48×48

The decoded VoxelSnapshot is scaled by `ROBOT_SPRITE_SCALE=4`:

```ts
// sprites.ts lines 419-422
const logicalHeight = frame.length;        // 48
const logicalWidth = (frame[0] as number[]).length;  // 48
const width = logicalWidth * ROBOT_SPRITE_SCALE;     // 192
const height = logicalHeight * ROBOT_SPRITE_SCALE;   // 192
```

`renderNeatensteinVoxelSpriteColumn` (line 904) receives this 192×192
VoxelSnapshot as `frame: VoxelSnapshot`. The `frameX` parameter (line 912) is a
column index in the 192-pixel-wide buffer, NOT a 48×48 logical grid coordinate.
The `frameY` is computed as `Math.floor(v * (frameHeight - 1))` (line 937) where
`frameHeight` = 192.

#### 3. renderNeatensteinSprite → renderNeatensteinVoxelSpriteColumn call chain

The worker calls (display.worker.ts lines 725-732):

```ts
renderNeatensteinSprite(
  spriteSnapshot.data,
  zBuffer,
  projection,
  frame,           // EncodedRobotSpriteFrame (the resolved frame)
  spriteContext,
  sprite.teamColor, // team color only
);
```

`renderNeatensteinSprite` (line 982) receives `source: NeatensteinSpriteSource`
(VoxelSnapshot | EncodedRobotSpriteFrame | string), NOT the `NeatensteinSprite`
(with `animationState`, `deRezElapsedMs`). It calls
`renderNeatensteinVoxelSpriteColumn` (line 1045) with:

```ts
renderNeatensteinVoxelSpriteColumn(
  framebuffer, width, height, column,
  drawStart, drawEnd, frame, frameX, fogFactor,
);
```

No `animationState`, `deRezElapsedMs`, `deRezDurationMs`, or `seed` parameter
exists in either function signature.

#### 4. NeatensteinSprite interface (line 128)

```ts
export interface NeatensteinSprite {
  worldX: number;
  worldY: number;
  facing?: number;
  animationState?: EnemyAnimationState;  // includes 'death'
  frameIndex?: number;
  type?: number;
  walkTick?: number;
  shootBlinkTicks?: number;
  teamColor?: readonly [number, number, number];
}
```

No `deRezElapsedMs` or `deRezDurationMs` field.

#### 5. activeEnemySprites map (display.worker.ts lines 494-506)

```ts
activeEnemySprites = enemyControllerState!.enemies
  .filter((enemy) => enemy.active)
  .map((enemy) => ({
    worldX: enemy.position.x,
    worldY: enemy.position.y,
    facing: enemy.yawRad,
    animationState: enemy.animationState,
    frameIndex: 0,
    type: enemy.index,
    walkTick: enemy.walkTick,
    shootBlinkTicks: enemy.shootBlinkTicks,
    teamColor: resolveEnemyTeamColor(enemy.index),
  }));
```

No `deRezElapsedMs` or `deRezDurationMs` propagated. The enemy controller state
DOES track `deRezElapsedMs` (enemy-controller.ts line 66, 385, 399, 410, 956).

#### 6. CPU path death visual (enemy-sprite.ts)

- `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS = 4000` (line 48) — separate constant
  from `ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 4000` (enemy-controller.ts line 188).
- CPU billboard path applies orange bolt-light tint during death (lines 749-752).
- `enemy-sprite.ts` is NOT in the slice's `files_to_change` list.

#### 7. NEATENSTEIN_ANIMATION_TO_POSE map (sprites.ts line 372)

```ts
death: 'stand',   // line 379 — death maps to 'stand' pose
```

Worker render path shows no death visual — death enemies render as standing.

### Decision

#### GAP 1 (CRITICAL): VoxelSnapshot coordinate mismatch with hash(frameX, frameY, seed)

**Plan claims:** hash(frameX, frameY, seed) operates on the 48×48 sprite array.
**Reality:** `renderNeatensteinVoxelSpriteColumn` receives a 192×192 VoxelSnapshot.
`frameX` and `frameY` are indices in the 192-pixel buffer, not the 48×48 logical
grid.

**Required:** The derez module must map VoxelSnapshot coordinates back to the
48×48 logical grid before hashing:
```ts
const logicalX = Math.floor(frameX / ROBOT_SPRITE_SCALE);
const logicalY = Math.floor(frameY / ROBOT_SPRITE_SCALE);
const noise = hash(logicalX, logicalY, seed);
```
This maps to the same 4×4 VoxelSnapshot block per logical pixel, ensuring whole
blocks dissolve together (matching the plan's "each removed pixel = 4×4 screen
block" description).

**Plan gap type:** `partial` — the hash approach is sound but the coordinate
mapping from 192×192 VoxelSnapshot to 48×48 logical grid is unspecified.

#### GAP 2 (CRITICAL): Parameter threading from worker to renderNeatensteinVoxelSpriteColumn

**Plan claims:** renderNeatensteinVoxelSpriteColumn applies derez mask when
animationState===death (AC-11e-004).

**Reality:** The call chain is:
1. Worker calls `renderNeatensteinSprite(frame, ...)` — passes only the frame,
   not the sprite metadata (animationState, deRezElapsedMs).
2. `renderNeatensteinSprite` calls `renderNeatensteinVoxelSpriteColumn(frame, frameX, fogFactor)` — no death parameters.
3. `renderNeatensteinVoxelSpriteColumn` has no animationState, deRezElapsedMs,
   deRezDurationMs, or seed parameter.

**Required:** Both `renderNeatensteinSprite` and `renderNeatensteinVoxelSpriteColumn`
need new parameters (animationState, deRezElapsedMs, deRezDurationMs, seed)
threaded through. The worker must pass these from the `NeatensteinSprite` to
`renderNeatensteinSprite`. AC-11e-002 correctly identifies the
`NeatensteinSprite` interface and `activeEnemySprites` map gaps, but does not
specify the `renderNeatensteinSprite` → `renderNeatensteinVoxelSpriteColumn`
parameter additions.

**Plan gap type:** `partial` — the ACs identify the data propagation gaps but
don't specify the renderer function signature changes.

#### GAP 3 (MODERATE): enemy-sprite.ts missing from files_to_change

**Reality:** `scripts/enemy-sprite.ts` has `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS = 4000`
(line 48), a separate constant from `ENEMY_CONTROLLER_DE_REZ_DURATION_MS`. The
CPU billboard path uses this for its orange bolt-light death visual (lines
749-752). The plan's `files_to_change` list does NOT include `enemy-sprite.ts`.

**Risk:** If only `ENEMY_CONTROLLER_DE_REZ_DURATION_MS` is changed to 700
(AC-11e-001), the CPU billboard path will still use 4000ms, creating a duration
mismatch between worker/voxel path (700ms) and CPU/billboard path (4000ms).
Either:
- Add `enemy-sprite.ts` to `files_to_change` and update
  `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS` to 700, OR
- Document that the CPU billboard path is being retired/replaced and the old
  death visual is being removed (No Deferred Cleanup mandate, plan line 171).

**Plan gap type:** `missing` — file not listed for a change it requires.

#### GAP 4 (LOW): Performance claim O(2304) is inaccurate

**Plan claims:** O(48×48) = O(2304) hash lookups per frame.

**Reality:** The hash is called per VoxelSnapshot pixel in
`renderNeatensteinVoxelSpriteColumn` (one call per screen row per column). For a
192×192 VoxelSnapshot, this is up to 192×192 = 36,864 hash calls per sprite per
frame, not 2,304. The 2,304 figure counts unique 48×48 logical pixels, but the
plan says "no per-frame array allocation; mask computed on-the-fly per pixel" —
meaning no caching, so each VoxelSnapshot pixel triggers a redundant hash call
for the same logical cell (16x redundancy: 4×4 VoxelSnapshot pixels per logical
pixel).

**Practical impact:** Negligible — the hash is trivial integer arithmetic and
the existing RGBA writes already iterate the same pixel count. The claim is
optimistic but the conclusion ("negligible vs existing per-pixel RGBA writes")
is correct.

**Plan gap type:** `contradicts` — the O(2304) figure conflicts with the
"on-the-fly per pixel" implementation approach (which gives O(192×192)).

#### GAP 5 (LOW): Seed derivation for "death start tick" underspecified

**Plan claims:** Seed = enemy type index + death start tick.

**Reality:** `NeatensteinSprite.type` provides the enemy type index. But "death
start tick" is not a field on `NeatensteinSprite` or the worker's
`activeEnemySprites` map. The enemy controller tracks `deRezElapsedMs` (elapsed
since death), but the absolute death start tick must be derived as
`currentSimTick - (deRezElapsedMs / msPerTick)` or stored separately.

The seed MUST be fixed for the duration of the death animation — if it changes
per frame, the noise pattern shifts and dissolved pixels could reappear
(non-monotonic dissolution). Using `deRezElapsedMs` in the seed would cause
flickering. The plan correctly identifies this need ("seeded RNG, no
Math.random()") but doesn't specify how the death start tick is derived or
stored.

**Plan gap type:** `partial` — seed composition is stated but derivation path
is unspecified.

### Risks

1. **GAP 1 + GAP 2 combined:** Without specifying both the coordinate mapping
   (192→48) and the parameter threading (worker→renderer→column), the
   implementer may apply the hash on 192×192 coordinates (wrong visual — 16x
   more noise cells, not matching the 4×4 block dissolution intent) or may
   not thread death parameters at all (no derez visual).

2. **GAP 3:** Duration mismatch between CPU and worker paths if enemy-sprite.ts
   is not updated.

3. **GAP 4:** Performance claim is inaccurate but practically harmless.
   Implementer should be aware the hash is called ~16x more than claimed.

4. **GAP 5:** If seed is not fixed per death event, dissolution will flicker
   rather than monotonically progress.

- `PlayerState.health` initialized to `NEATENSTEIN_PLAYER_MAX_HEALTH = 100`
- `applyDamage(state, amount)` (line 149): reduces health, respects invulnerability
- `isInvulnerable(state)` (line 118): true if `dashTimeRemainingMs > 0` OR
  `contactIFrameMs > 0` (500ms i-frame after contact damage)
- **No health HUD**: `browser-entry.ts` (lines 32, 127) notes "reserved for future
  HUD output." Player has no visible health feedback currently.

### 4. Enemy AI Controller — Fire Logic

**File:** `examples/neatenstein/scripts/enemy-controller.ts`

**CRITICAL FINDING:** The controller ALREADY produces `HitscanEvent` objects on
every fire tick, but `display.worker.ts` NEVER consumes `controlled.hitscanEvents`
— they are discarded entirely.

**HitscanEvent** (enemy-controller.ts):
```typescript
interface HitscanEvent {
  enemyIndex: number;
  origin: Vector2;
  direction: Vector2;
  damage: number;  // ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10
}
```

**Fire conditions** (lines 908-920):
- `fireCooldownMs <= 0` (ENEMY_CONTROLLER_FIRE_COOLDOWN_MS = 1000ms)
- `ammo > 0` (ENEMY_CONTROLLER_STARTING_AMMO = 3)
- `distToPlayer <= 8 cells` (ENEMY_CONTROLLER_FIRE_RANGE_CELLS)
- `hasLineOfSight` (BFS raycast)
- On fire: ammo decremented, fireCooldownMs reset, shootBlinkTicks = 4 (muzzle blink)

**MLP fire output**: Topology 6→6→4→4, outputs labeled
`['move','strafe','turn','fire']`. The "fire" output (outputs[3]) ALREADY EXISTS.
- **Harness rollout** (`enemy-runner.ts` line 322): USES fire output
  (`if (fire > 0 && fireCooldown <= 0)`)
- **Live demo controller** (`enemy-controller.ts` lines 620-623): only uses
  outputs[0] (move) and outputs[1] (strafe); outputs[3] (fire) is IGNORED.
  Firing is rule-based, not MLP-driven.

**Two-system divergence risk**: Live demo controller (rule-based hitscan) vs
evolution harness rollout (MLP-fire-output-gated abstract damage) model enemy fire
differently. Must keep them conceptually aligned or evolved behavior won't
transfer to live demo.

### 5. Enemy Fire Constants

| Constant | Value | File |
|----------|-------|------|
| `ENEMY_CONTROLLER_FIRE_COOLDOWN_MS` | 1000ms | enemy-controller.ts |
| `ENEMY_CONTROLLER_FIRE_RANGE_CELLS` | 8 | enemy-controller.ts |
| `ENEMY_CONTROLLER_STARTING_AMMO` | 3 | enemy-controller.ts |
| `ENEMY_CONTROLLER_SHOOT_BLINK_TICKS` | 4 | enemy-controller.ts |
| `ENEMY_CONTROLLER_HITSCAN_DAMAGE` | 10 | enemy-controller.ts |
| `NEATENSTEIN_PLAYER_MAX_HEALTH` | 100 | constants.ts |
| `ROLLOUT_FIRE_COOLDOWN_TICKS` | 63 (~1 sec) | enemy-runner.ts |
| `ROLLOUT_FIRE_DAMAGE` | 10 | enemy-runner.ts |
| `ROLLOUT_FIRE_RANGE_CELLS` | 8 | enemy-runner.ts |

**Damage alignment**: 10 damage = exactly 10% of maxHealth(100). The requested
feature aligns perfectly with existing constants.

### 6. Rendering System for Bolts

**File:** `examples/neatenstein/browser-entry/renderer/bolt-render.ts`

`drawBolts()` (line 194) renders player bolts:
- Hardwired to interpolate from player's gun muzzle (screen ratios 0.5, 0.82)
- Teal color (#00f0ff)
- Uses `projectNeatensteinFloorPoint()` (already imported) for world→screen

**Enemy bolt rendering gap**: `drawBolts` is player-only. Enemy bolts need a
separate `drawEnemyBolts()` interpolating from the enemy's projected world
position with a red/orange color to distinguish from teal player bolts.

`drawImpactSpots()` (line 105) renders explosion/impact effects with additive
blend, glow, distance-scaling, 3000ms lifetime. Template for player-impact
explosion effect on enemy bolt hit.

### 7. Collision Detection

**File:** `examples/neatenstein/browser-entry/host/game/tick.ts`

- `updateBolts()` (line 439): moves player bolts, checks wall collision
- `findBoltEnemyImpact()` (line 366): tests bolt proximity to enemies
  (hit radius = `NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS`)
- Enemy bolts need `updateEnemyBolts()`: move bolts, check wall collision, test
  player proximity, apply damage via `applyDamage()` on hit.

**i-frame semantics**: `isInvulnerable()` checks `dashTimeRemainingMs > 0` OR
`contactIFrameMs > 0` (500ms). Enemy bolts could reuse `contactIFrameMs` (one
bolt per 500ms) or get a separate shorter projectile i-frame. Design decision
required.

### 8. Display Worker — Where hitscanEvents Are Discarded

**File:** `examples/neatenstein/browser-entry/worker/display.worker.ts`

- Lines 1147-1200: `updateEnemyController()` is called but
  `controlled.hitscanEvents` are NEVER read.
- Lines 788-795: `drawBolts()` called with `gameState.bolts` (player bolts only).
- This is where hitscanEvents must be consumed to spawn enemy bolts and where
  `drawEnemyBolts()` must be called.

### 9. Determinism Requirement

The codebase is explicitly deterministic-replay-oriented (seed-stable, pure/
immutable state transitions). Enemy bolt creation must use the already-computed
`HitscanEvent` origin/direction to avoid introducing extra RNG. No `Math.random()`
in any bolt creation or movement code.

## Decision

### Proposed Enemy Fire Implementation

1. **Entity**: Add `EnemyBoltState` to `types.ts` (mirror `BoltState` with enemy-
   specific fields: `enemyIndex`, `spawnCell`, `direction`, `damage`, `alive`,
   `travelTimeMs`, `maxRangeCells`).
2. **State**: Add `enemyBolts: EnemyBoltState[]` to `GameState` in `types.ts`;
   initialize `enemyBolts: []` in `createGameState()` in `state.ts`.
3. **Constants**: Add enemy-bolt speed/radius/lifetime/damage constants to
   `constants.ts`. Reuse `ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10` for damage.
4. **Spawning**: In `display.worker.ts`, consume `controlled.hitscanEvents` after
   `updateEnemyController()`. For each event, create `EnemyBoltState` and append
   to `gameState.enemyBolts`. Use `fireEnemyBolt()` helper in `combat.ts`.
5. **Movement + Collision**: Add `updateEnemyBolts()` to `tick.ts`:
   - Move bolts along direction at enemy-bolt speed
   - Check wall collision (remove on wall hit, create impact spot)
   - Test player proximity (hit radius)
   - On player hit: call `applyDamage(state, bolt.damage)`, create explosion
     `ImpactSpot`, remove bolt
   - Remove expired bolts (travelTimeMs > lifetime)
6. **Rendering**: Add `drawEnemyBolts()` to `bolt-render.ts`:
   - Interpolate from enemy projected world position (use
     `projectNeatensteinFloorPoint`)
   - Red/orange color to distinguish from teal player bolts
   - Explosion effect on player hit (reuse `drawImpactSpots` pattern)
7. **Pipeline**: Call `updateEnemyBolts()` in `gameTick()` after
   `updateEnemyController()`. Call `drawEnemyBolts()` in `display.worker.ts`
   after `drawBolts()`.

### All Impacted Files

| # | File | Change |
|---|------|--------|
| 1 | `host/game/types.ts` | Add `EnemyBoltState`; add `enemyBolts` to `GameState` |
| 2 | `host/game/state.ts` | Init `enemyBolts: []` in `createGameState()` |
| 3 | `host/game/constants.ts` | Add enemy-bolt speed/radius/lifetime constants |
| 4 | `host/game/combat.ts` | Add `fireEnemyBolt()` helper |
| 5 | `host/game/tick.ts` | Add `updateEnemyBolts()`; call in `gameTick()` |
| 6 | `worker/display.worker.ts` | Consume `hitscanEvents`; call `drawEnemyBolts()` |
| 7 | `renderer/bolt-render.ts` | Add `drawEnemyBolts()` with red/orange color |
| 8 | `scripts/enemy-controller.ts` | (Optional) No changes needed — hitscanEvents already produced |
| 9 | `harness/enemy-mlp.ts` | (Optional future) Wire MLP fire output in live controller |
| Test files | `types.test.ts`, `tick.test.ts`, `combat.test.ts`, `display.worker.test.ts`, `bolt-render.test.ts` | New tests for enemy bolt lifecycle |

## Risks

1. **Two-system divergence**: Live demo controller (rule-based hitscan) vs
   harness rollout (MLP-fire-output-gated abstract damage) model enemy fire
   differently. Evolved behavior won't transfer to live demo unless kept aligned.
2. **i-frame design**: Reuse `contactIFrameMs` (500ms, shared with contact damage)
   or dedicated enemy-bolt i-frame? If shared, enemy bolt hit grants 500ms
   invulnerability to ALL damage. Decision needed.
3. **No health HUD**: Player has no visible health feedback. applyDamage works
   but player can't see damage taken. Consider adding HUD in a separate slice.
4. **Determinism**: Enemy bolt creation must use HitscanEvent origin/direction
   (already computed) — no extra RNG. No `Math.random()` anywhere in bolt code.
5. **Bolt speed vs hitscan**: HitscanEvent is instant, but spawning a traveling
   bolt means damage arrives after travel time. Player can dodge by moving.
   This is intentional (visible projectiles) but changes damage timing vs
   the current (discarded) hitscan model.
6. **Ammo depletion**: Enemies start with 3 ammo. With 1000ms cooldown, they fire
   3 times then stop. May need ammo replenishment or infinite ammo for sustained
   combat.

## Confidence

- **0.95** — Current shooting architecture, player health model, enemy AI fire
  logic, hitscanEvent production/discard gap. All verified by direct file reads.
- **0.92** — Proposed enemy fire implementation. Reuses existing patterns
  (fireBolt template, applyDamage, drawImpactSpots, projectNeatensteinFloorPoint).
- **0.80** — Two-system divergence risk and i-frame design choice. These are
  architectural decisions with moderate uncertainty.