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
  return (
    state.player.dashTimeRemainingMs > 0 || state.player.contactIFrameMs > 0
  );
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
    animationState: enemy.animationState, // ← forwarded to renderer
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

| #   | File                                     | Change                                                                                                                                                                                                                                      |
| --- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `host/game/types.ts`                     | Add `stunTimerMs?: number` to `EnemyState`                                                                                                                                                                                                  |
| 2   | `host/game/constants.ts`                 | Add `NEATENSTEIN_ENEMY_STUN_DURATION_MS`, `NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS`                                                                                                                                                       |
| 3   | `host/game/combat.ts`                    | Modify `applyEnemyDamage`: compute pushback direction, set `stunTimerMs`, apply position offset with wall check, skip if already stunned (invincibility)                                                                                    |
| 4   | `scripts/enemy-controller.ts`            | Add `stunTimerMs` to `ControlledEnemy` interface; in `updateControlledEnemy`: stun check before movement block (skip movement + MLP + fire), decrement timer, set `animationState = 'damage'`; update `createEnemyControllerState` defaults |
| 5   | `scripts/enemy-animator.ts`              | No change needed — `'damage'` state already exists. Optionally add `'stunned'` state if distinct visual desired.                                                                                                                            |
| 6   | `browser-entry/renderer/sprites.ts`      | No change needed — `NEATENSTEIN_ANIMATION_TO_POSE.damage` already maps to `'stand'`. Optionally add stun-specific visual effect.                                                                                                            |
| 7   | `browser-entry/worker/display.worker.ts` | Update `__testOnlyInjectTestEnemies` to include `stunTimerMs: 0` in test enemy construction. No functional change needed — `animationState` already forwarded.                                                                              |
| 8   | `host/game/tick.ts`                      | No direct change — `applyEnemyDamage` is already called for both hit paths. Stun/pushback handled inside `applyEnemyDamage`.                                                                                                                |
| 9   | `browser-entry/harness/enemy-runner.ts`  | No change needed — evaluation doesn't model player shooting at enemies.                                                                                                                                                                     |
| 10  | `browser-entry/harness/enemy-mlp.ts`     | No change — MLP skip is handled at the call site in `enemy-controller.ts`.                                                                                                                                                                  |

### Test Files Needing Updates

| File                                          | Reason                                                          |
| --------------------------------------------- | --------------------------------------------------------------- |
| `host/game/combat.test.ts`                    | Test `applyEnemyDamage` now sets stun + pushback                |
| `host/game/tick.test.ts`                      | Test traveling bolt path applies stun                           |
| `host/game/types.test.ts`                     | Test `EnemyState` with stun field                               |
| `host/game/constants.test.ts`                 | Test new constants exist                                        |
| `scripts/enemy-controller.test.ts`            | Test stunned enemy doesn't move, MLP skipped, timer decremented |
| `browser-entry/worker/display.worker.test.ts` | Test stunned enemy renders with 'damage' state                  |
| `scripts/enemy-animator.test.ts`              | Test 'damage' animation state for stun (may already pass)       |
| `browser-entry/renderer/sprites.test.ts`      | Test 'damage' state renders as 'stand' pose (may already pass)  |

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
  hitEnemyIndex?: number; // ← set when bolt hits an enemy
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
  lifetimeMs: number; // 3000ms
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
  worldX: number;
  worldY: number;
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

| #   | File                            | Change                                                                              |
| --- | ------------------------------- | ----------------------------------------------------------------------------------- |
| 1   | `host/game/types.ts`            | Add `EnemyImpactSpot` type; add `enemyImpacts` to `GameState`                       |
| 2   | `host/game/constants.ts`        | Add enemy impact constants (lifetime, burst radius, burst duration)                 |
| 3   | `host/game/combat.ts`           | In `fireBolt()`, create enemy impact spot when `hitType === 'enemy'`                |
| 4   | `host/game/tick.ts`             | Add `ageEnemyImpacts()` and call in `gameTick()`; optionally wire `damage` state    |
| 5   | `renderer/bolt-render.ts`       | Add `drawEnemyImpactSpots()` rendering function                                     |
| 6   | `worker/display.worker.ts`      | Call enemy impact rendering in paint order (after sprites, with/after wall impacts) |
| 7   | `browser-entry/constants.ts`    | Add enemy impact visual constants (color, glow, radius)                             |
| 8   | `renderer/sprites.ts`           | Optionally: wire `damage` animation state for hit-flash overlay                     |
| 9   | `host/game/combat.test.ts`      | Test enemy impact spot creation on hit                                              |
| 10  | `host/game/tick.test.ts`        | Test enemy impact spot aging and removal                                            |
| 11  | `renderer/bolt-render.test.ts`  | Test enemy impact spot rendering                                                    |
| 12  | `worker/display.worker.test.ts` | Test enemy impact rendering in frame                                                |

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
const logicalHeight = frame.length; // 48
const logicalWidth = (frame[0] as number[]).length; // 48
const width = logicalWidth * ROBOT_SPRITE_SCALE; // 192
const height = logicalHeight * ROBOT_SPRITE_SCALE; // 192
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
  frame, // EncodedRobotSpriteFrame (the resolved frame)
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
  framebuffer,
  width,
  height,
  column,
  drawStart,
  drawEnd,
  frame,
  frameX,
  fogFactor,
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
  animationState?: EnemyAnimationState; // includes 'death'
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
  damage: number; // ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10
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

| Constant                             | Value       | File                |
| ------------------------------------ | ----------- | ------------------- |
| `ENEMY_CONTROLLER_FIRE_COOLDOWN_MS`  | 1000ms      | enemy-controller.ts |
| `ENEMY_CONTROLLER_FIRE_RANGE_CELLS`  | 8           | enemy-controller.ts |
| `ENEMY_CONTROLLER_STARTING_AMMO`     | 3           | enemy-controller.ts |
| `ENEMY_CONTROLLER_SHOOT_BLINK_TICKS` | 4           | enemy-controller.ts |
| `ENEMY_CONTROLLER_HITSCAN_DAMAGE`    | 10          | enemy-controller.ts |
| `NEATENSTEIN_PLAYER_MAX_HEALTH`      | 100         | constants.ts        |
| `ROLLOUT_FIRE_COOLDOWN_TICKS`        | 63 (~1 sec) | enemy-runner.ts     |
| `ROLLOUT_FIRE_DAMAGE`                | 10          | enemy-runner.ts     |
| `ROLLOUT_FIRE_RANGE_CELLS`           | 8           | enemy-runner.ts     |

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

| #          | File                                                                                               | Change                                                        |
| ---------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1          | `host/game/types.ts`                                                                               | Add `EnemyBoltState`; add `enemyBolts` to `GameState`         |
| 2          | `host/game/state.ts`                                                                               | Init `enemyBolts: []` in `createGameState()`                  |
| 3          | `host/game/constants.ts`                                                                           | Add enemy-bolt speed/radius/lifetime constants                |
| 4          | `host/game/combat.ts`                                                                              | Add `fireEnemyBolt()` helper                                  |
| 5          | `host/game/tick.ts`                                                                                | Add `updateEnemyBolts()`; call in `gameTick()`                |
| 6          | `worker/display.worker.ts`                                                                         | Consume `hitscanEvents`; call `drawEnemyBolts()`              |
| 7          | `renderer/bolt-render.ts`                                                                          | Add `drawEnemyBolts()` with red/orange color                  |
| 8          | `scripts/enemy-controller.ts`                                                                      | (Optional) No changes needed — hitscanEvents already produced |
| 9          | `harness/enemy-mlp.ts`                                                                             | (Optional future) Wire MLP fire output in live controller     |
| Test files | `types.test.ts`, `tick.test.ts`, `combat.test.ts`, `display.worker.test.ts`, `bolt-render.test.ts` | New tests for enemy bolt lifecycle                            |

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

## Slice 11 Performance Trace Root Cause Analysis (2026-08-06)

### Question

Analyze Chrome DevTools Performance trace (Trace-20260806T191754.json, 42003 lines) to find the root cause of severe performance degradation. Two suspects: (1) Step 11 view-distance slice increased NEATENSTEIN_RENDER_DISTANCE_CAP 30-to-40 and doubled NEATENSTEIN_FLOOR_LINE_SAMPLES 80-to-160; (2) Step 10.5 added real MLP neural network enemy AI. Which system is the bottleneck?

### Evidence

**Trace Metrics:** Worker onmessage 197 calls avg 23ms max 63ms total 4539ms. 70 calls at 40-50ms, 19 at 50-60ms. Frame cadence: 98 heavy frames avg interval 47.9ms. Effective FPS 14-21 (target 60). Main thread trivially fast (357 calls avg 64us). No V8 sampling profiler data. 0 forced reflows.

**Finding 1 - Floor/Ceiling Grid Rendering PRIMARY Bottleneck (~70-80pct frame time):** RENDER_DISTANCE_CAP 40 (was 30) plus FLOOR_LINE_SAMPLES 160 (was 80) = 2.64x increase in projection calls per frame (52164 vs 19764). 16 ctx.stroke() calls with shadowBlur per frame. Estimated 30-35ms per frame.

**Finding 2 - Redundant Zero-Timestep updateEnemyController SECONDARY:** Second call dtMs=0 still rebuilds BFS 14400-cell map. ~2-4ms waste. Validated by performance-trace-specialist.

**Finding 3 - MLP Enemy AI NEGLIGIBLE:** Topology [6,6,4,4] 90 weights, ~76 multiply-adds per enemy, microseconds. Step 10.5 EXONERATED.

**Finding 4 - Raycasting minor:** 480 columns DDA up to 40 cells, ~3-5ms.

### Decision

Root cause: floor/ceiling grid rendering increase from Step 11. Recommendations: (1) reduce FLOOR_LINE_SAMPLES to 100-120 (biggest lever), (2) reduce RENDER_DISTANCE_CAP to 35, (3) optimize shadowBlur strokes, (4) skip BFS in zero-timestep pass, (5) MLP needs NO optimization.

### Risks

No before/after trace. Visual banding risk from reduced samples. Pop-in risk from reduced distance. shadowBlur refactor is larger work.

### Confidence

0.92 floor/ceiling primary bottleneck. 0.88 redundant zero-timestep pass. 0.95 MLP negligible. 0.80 specific ms estimates approximate (no V8 profiler data).
---

## Phase 7 Research: HUD Health and Ammo Display

### HUD instantiation pattern

- `examples/neatenstein/browser-entry/host/hud.ts` is the single public HUD surface.
- `createHiveDensityHud(outputId)` resolves `#outputId`, creates a 160x8 px meter track, a fill bar, and a label, appends them to the container, and returns `{ container, meter, fill, label, update }`.
- `createHumanModeSelector(outputId)` uses the same container-resolution pattern, appends a `<select>` with `auto`/`human` options, and exposes `{ container, select, mode, setMode, onToggle }`.
- Both factories are instantiated once in `examples/neatenstein/browser-entry/browser-entry.ts#neatensteinStart` with `outputId = 'neatenstein-output'`.

### Update lifecycle

- The host render loop (`browser-entry.ts#startRenderLoop`) currently updates the HIVE DENSITY HUD on every rAF tick by calling `hud.update({ hiveDensity })`.
- The authoritative `GameState` lives inside `examples/neatenstein/browser-entry/worker/display.worker.ts`. The worker advances `gameState` via `gameTick(...)` on every `simState` message and then posts a `frame` message back.
- Worker-tier `frame` messages carry only `{ requestId }`. CPU/GPU-tier frames carry `gun` and `bolts` but no player vitals.
- To render live health/ammo, the worker must send `player.health` and `player.ammo` back to the host. The recommended seam is to extend `NeatensteinRenderFrame` in `renderer/frame.ts` with optional `playerHealth`, `playerAmmo`, `playerMaxHealth`, and `playerMaxAmmo`, populate them from `gameState.player` in `display.worker.ts`, and consume them in `browser-entry.ts` via `bridge.setFrameConsumer`.

### Player-state source of truth

- `PlayerState` (`host/game/types.ts`) already has `health`, `maxHealth`, `ammo`, and `maxAmmo`.
- `createGameState` initializes both to `NEATENSTEIN_PLAYER_MAX_HEALTH` (100) and `NEATENSTEIN_PLAYER_MAX_AMMO` (50) from `host/game/constants.ts`.
- `consumeAmmo` and `applyDamage` in `host/game/state.ts` mutate these values deterministically. No `restoreAmmo` exists yet (reserved for Phase 7 Step 04).

### Visual design proposal

- Health bar: horizontal 160x8 px bar, bottom-left of the HUD container, color-coded by fraction:
  - > = 70% -- `rgb(0, 240, 255)` (neon cyan / healthy)
  - 30-69% -- `rgb(240, 160, 0)` (amber / caution)
  - < 30% -- `rgb(255, 0, 85)` (magenta / critical)
- Numeric label beside the bar: `HEALTH ${health}/${maxHealth}`.
- Ammo counter below or to the right: `AMMO ${ammo}/${maxAmmo}`, rendered in the accent cyan.
- Updates are instant per frame; resources are discrete, so smoothing is unnecessary.
- Layout stacks inside `#neatenstein-output` after the existing HIVE DENSITY meter and human-mode selector. If stacking pushes the canvas, add an absolutely-positioned HUD wrapper in `examples/neatenstein/index.html`.

### DOM structure for new factory

Recommended `createHealthAmmoHud(outputId)` returns:

```
{
  container: HTMLElement;
  healthTrack: HTMLElement;
  healthFill: HTMLElement;
  healthLabel: HTMLElement;
  ammoLabel: HTMLElement;
  update: (state: HealthAmmoHudState) => void;
}
```

It appends:

- a track `div` (160x8 px, dark/neon border),
- a fill `div` (height 100%, width = health%),
- a health-label `div` (`HEALTH ${health}/${maxHealth}`),
- an ammo-label `div` (`AMMO ${ammo}/${maxAmmo}`).

### Step-by-step implementation plan

1. Extend `renderer/frame.ts` `NeatensteinRenderFrame` with optional `playerHealth`, `playerAmmo`, `playerMaxHealth?`, `playerMaxAmmo?`.
2. In `worker/display.worker.ts`, set these fields from `gameState.player` before `self.postMessage({ type: 'frame', frame })` in both worker and CPU/GPU paths.
3. Add `HealthAmmoHudState`, `HealthAmmoHud`, and `createHealthAmmoHud()` to `host/hud.ts`. Add a local `resolveHealthColor(health / maxHealth)` helper. Import new design tokens from `../constants.ts`.
4. Add health/ammo design tokens (bar width/height, colors, labels) to `examples/neatenstein/browser-entry/constants.ts`.
5. In `browser-entry.ts`:
   - import and instantiate `createHealthAmmoHud(outputId)`,
   - pass the instance to `startRenderLoop`,
   - register `bridge.setFrameConsumer(frame => healthAmmoHud.update(frame))`.
6. Add red-phase tests in `examples/neatenstein/browser-entry/host/hud-health-ammo.test.ts` covering container resolution, health fill width, color thresholds, and ammo label.
7. Validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/hud`, `npx tsc --noEmit -p tsconfig.json`, and `npm run quality:folder -- --folder=examples/neatenstein/browser-entry/host` after the implementation edits.

### Files that will change in the implementation step

- `examples/neatenstein/browser-entry/host/hud.ts` -- new factory and types.
- `examples/neatenstein/browser-entry/constants.ts` -- health/ammo HUD design tokens.
- `examples/neatenstein/browser-entry/renderer/frame.ts` -- optional player vitals on `NeatensteinRenderFrame`.
- `examples/neatenstein/browser-entry/worker/display.worker.ts` -- populate player vitals before posting frame.
- `examples/neatenstein/browser-entry/browser-entry.ts` -- instantiate HUD and wire frame consumer.
- `examples/neatenstein/browser-entry/host/hud-health-ammo.test.ts` -- red-phase contract tests.
- `examples/neatenstein/browser-entry/README.md` -- generated README will need an `educational-docs` pass after implementation.

---

## Phase 7 Research: Ammo Drops from Dying Enemies

### Question

When an enemy dies, a white plasma ball pickup should spawn at the enemy's death position. The player can collect it to restore ammo. This research investigates: (1) the new `AmmoPickupState` type design, (2) `GameState` storage changes, (3) `combat.ts` spawn hook on kill, (4) `tick.ts` collision/lifecycle management, (5) `state.ts` `restoreAmmo` helper, (6) the white plasma ball rendering approach, (7) pickup lifetime/despawn policy, and (8) ammo restoration amount.

### Evidence

Evidence was gathered from direct source-file reads of all target files and validated by two dispatched specialists (boundary-mapper and implementation-pattern-scout). The Cortex index was stale (`index_fresh: false`), so native file reads were used as the primary discovery path.

**Source files read:**

- `examples/neatenstein/browser-entry/host/game/types.ts` — GameState, EnemyState, PlayerState, BoltState interfaces (230 lines). GameState has optional array fields (`bolts?`, `enemyBolts?`, `enemyImpacts?`) as the established pattern for backward-compatible additions.
- `examples/neatenstein/browser-entry/host/game/combat.ts` — `fireBolt()` and `applyEnemyDamage()` (440 lines). `applyEnemyDamage()` at line 334 computes `killedByThisShot = newHealth === 0` (line 346). The kill branch (lines 390-394) currently only increments `state.kills` and maps the enemy health to zero. It does NOT set `enemy.active = false` or spawn any pickup.
- `examples/neatenstein/browser-entry/host/game/tick.ts` — `gameTick()` pipeline (23.4 KB, 750+ lines). Pipeline steps: Step 1 (episode), Step 2 (look), Step 3 (dash), Step 4 (movement), Step 5 (bolts + enemy damage), Step 5b (enemy bolts), Step 6 (fire). No pickup-related step exists. `gameTick` returns at line 336.
- `examples/neatenstein/browser-entry/host/game/state.ts` — `createGameState()`, `consumeAmmo()`, `applyDamage()`, `applyDash()` (206 lines). `consumeAmmo()` at line 170 decrements ammo by 1, clamped at 0. No `restoreAmmo()` exists.
- `examples/neatenstein/browser-entry/host/game/constants.ts` — `NEATENSTEIN_PLAYER_MAX_AMMO = 50`, `NEATENSTEIN_BOLT_DAMAGE = 20`, `NEATENSTEIN_ENEMY_MAX_HEALTH = 100`, `NEATENSTEIN_CONTACT_RANGE_CELLS = 0.5` (478 lines). No ammo-pickup constants exist.
- `examples/neatenstein/browser-entry/renderer/bolt-render.ts` — `drawImpactSpots()`, `drawEnemyImpactSpots()`, `drawBolts()` (20+ KB). Uses additive blending (`globalCompositeOperation = 'lighter'`), floor projection via `projectNeatensteinFloorPoint`, depth testing via `depthTestPulse`, and shadow-blur glow. Impact spot colors: `NEATENSTEIN_IMPACT_SPOT_COLOR = '#f0f8ff'`, `NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR = 'rgba(240,248,255,0.5)'`, `NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX = 2`, `NEATENSTEIN_IMPACT_SPOT_RADIUS_PX = 4`.
- `examples/neatenstein/browser-entry/constants.ts` — visual constants including impact spot colors and lifetimes (confirmed at lines 72-93).

**Specialist findings:**

- **boundary-mapper** (confidence: 0.92): Mapped four lifecycle seams — (1) types.ts: add `AmmoPickupState` + `ammoPickups?: AmmoPickupState[]` on GameState; (2) state.ts: initialize `ammoPickups: []` in `createGameState` + add `restoreAmmo()`; (3) combat.ts: spawn pickup inside `applyEnemyDamage` when `killedByThisShot`; (4) tick.ts: add `updateAmmoPickups` stage in `gameTick` pipeline. Flagged tick.ts as a coordination sink (23 KB) — recommends keeping pickup logic as a focused helper. Confirmed no existing pickup code anywhere in the repo. Provenance: direct source reads + Select-String import/export enumeration.
- **implementation-pattern-scout** (confidence: 0.90): Discovered the renderer pattern for the white plasma ball — reuse the bolt/impact-spot additive-glow floor-projection pattern from `bolt-render.ts` rather than the full voxel sprite path. Key pattern: save composite → set `'lighter'` → project floor point → depth-test against zBuffer → draw arc with shadow blur → restore. Recommended either extending `bolt-render.ts` or creating a new `renderer/ammo-pickup.ts`. Confirmed `display.worker.ts` orchestrates all draw calls and would need a new pickup draw call inserted at the correct z-order. Provenance: direct reads of bolt-render.ts, sprites.ts, display.worker.ts, floor.ts, pulse.ts.

### Decision

#### 1. AmmoPickupState type design

Add to `host/game/types.ts`:

```typescript
/** A white plasma ball ammo pickup spawned when an enemy dies. */
export interface AmmoPickupState {
  /** World-space position where the pickup spawned (enemy death position). */
  position: Vector2;
  /** Amount of ammo restored when collected. */
  amount: number;
  /** `true` while the pickup is still active (not yet collected or expired). */
  active: boolean;
  /** Simulation time at which the pickup was spawned, in milliseconds. */
  createdAtMs: number;
  /** Optional lifetime in milliseconds; pickup despawns after this elapses. */
  lifetimeMs?: number;
}
```

This mirrors the existing `BoltState` / `EnemyBoltState` pattern: plain objects with position, active flag, and createdAtMs. The optional `lifetimeMs` follows the `ImpactSpot.lifetimeMs` pattern.

#### 2. GameState changes

Add an optional `ammoPickups?: AmmoPickupState[]` field to `GameState` in `types.ts`:

```typescript
/** Active ammo pickups in the world. */
ammoPickups?: AmmoPickupState[];
```

This follows the backward-compatible pattern already used for `enemyImpacts?`, `bolts?`, and `enemyBolts?`. Callers should use `state.ammoPickups ?? []` when reading, matching the `enemyImpacts` convention documented at types.ts line 199-203.

Initialize as `ammoPickups: []` in `createGameState()` in `state.ts`, mirroring the `bolts: []` and `enemyBolts: []` initialization at line 103-104.

#### 3. combat.ts spawn hook

In `applyEnemyDamage()` (line 334), the `killedByThisShot` branch (line 346, true when `newHealth === 0`) currently only increments `state.kills`. Add pickup spawning:

```typescript
// Inside applyEnemyDamage, after computing newEnemies and before the return:
if (killedByThisShot) {
  const pickup: AmmoPickupState = {
    position: { ...enemy.position },
    amount: NEATENSTEIN_AMMO_PICKUP_AMOUNT,
    active: true,
    createdAtMs: state.simTimeMs,
    lifetimeMs: NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS,
  };
  return {
    ...state,
    enemies: newEnemies,
    kills: state.kills + 1,
    ammoPickups: [...(state.ammoPickups ?? []), pickup],
  };
}
```

The spawn happens at the enemy's death position (`enemy.position`). The existing `killedByThisShot` check is the correct seam — it already distinguishes lethal from non-lethal hits and is the only place kills are counted.

#### 4. tick.ts collision/lifecycle

Add a new Step 5c in `gameTick()` between Step 5b (enemy bolts) and Step 6 (fire), at approximately line 318:

```typescript
// Step 5c: Update ammo pickups — age lifetime, check player proximity,
// collect if within radius, restore ammo, and cull inactive pickups.
const pickupResult = updateAmmoPickups(
  next.ammoPickups ?? [],
  resolvedDtMs,
  next.simTimeMs,
  next.player,
);
next = pickupResult.state;
next = {
  ...next,
  ammoPickups: pickupResult.pickups,
};
```

The `updateAmmoPickups` helper should:

1. For each active pickup, compute distance from player: `Math.hypot(player.position.x - pickup.position.x, player.position.y - pickup.position.y)`.
2. If distance <= `NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS` (recommended 0.5, matching `NEATENSTEIN_CONTACT_RANGE_CELLS`), mark the pickup inactive and call `restoreAmmo(state, pickup.amount)`.
3. If `lifetimeMs` is set and `simTimeMs - createdAtMs >= lifetimeMs`, mark inactive.
4. Filter out inactive pickups.
5. Return updated state and filtered pickups array.

Keep this as a focused helper function (like `updateBolts` and `updateEnemyBolts`) to preserve the declarative gameTick pipeline. Do not inline the logic into `gameTick` itself.

#### 5. state.ts restoreAmmo

Add `restoreAmmo()` symmetric to `consumeAmmo()` (line 170):

```typescript
/**
 * Restore ammo to the player, clamped at maxAmmo.
 *
 * @param state - Snapshot before restoration.
 * @param amount - Amount to restore; clamped to maxAmmo.
 * @returns New snapshot with ammo incremented, clamped at maxAmmo.
 */
export function restoreAmmo(state: GameState, amount: number): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      ammo: Math.min(state.player.maxAmmo, state.player.ammo + amount),
    },
  };
}
```

This mirrors `consumeAmmo` exactly but with `Math.min(maxAmmo, ammo + amount)` instead of `Math.max(0, ammo - 1)`.

#### 6. Rendering approach (white plasma ball)

Reuse the bolt/impact-spot additive-glow pattern from `bolt-render.ts`. The plasma ball needs no animated sprite atlas — it is a static glowing circle on the floor plane.

**Recommended approach:** Create a new `drawAmmoPickups()` export in `bolt-render.ts` (or a new `renderer/ammo-pickup.ts` if the file grows). Follow the `drawImpactSpots()` pattern exactly:

1. Save `context.globalCompositeOperation`.
2. Set `globalCompositeOperation = 'lighter'` (additive blending).
3. For each active pickup: project world position to screen using `projectNeatensteinFloorPoint` with `BOLT_PROJECTED_CAMERA_HEIGHT_WORLD = 0` (floor level). Depth-test against zBuffer via `depthTestPulse`. Compute distance-based radius and alpha. Draw `context.arc()` with `shadowColor` glow and `fillStyle` white core.
4. Restore composite, alpha, and shadow.

**Visual constants** (add to `browser-entry/constants.ts`):

- `NEATENSTEIN_AMMO_PICKUP_COLOR = '#ffffff'` — white core
- `NEATENSTEIN_AMMO_PICKUP_GLOW_COLOR = 'rgba(240,248,255,0.5)'` — cool-white halo (same family as impact spots)
- `NEATENSTEIN_AMMO_PICKUP_GLOW_BLUR_PX = 4` — slightly larger glow than impact spots
- `NEATENSTEIN_AMMO_PICKUP_RADIUS_PX = 5` — slightly larger than impact spot radius (4px)
- `NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS = 10000` — pickup despawns after 10 seconds if uncollected

**Render frame integration:** Extend `NeatensteinRenderFrame` in `renderer/frame.ts` with optional `ammoPickups?: readonly AmmoPickupState[]`. Populate from `gameState.ammoPickups` in `display.worker.ts` alongside bolts and impacts. Insert the `drawAmmoPickups()` call in the worker's render sequence after sprites and impacts, before or alongside bolts.

**Z-order:** Pickups should be drawn after floor/walls/sprites but before or alongside bolts, matching the impact-spot paint order ("after sprites, before bolts" per bolt-render.ts line 202).

#### 7. Pickup lifetime/despawn policy

**Decision:** Pickups have a finite lifetime of 10 seconds (`NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS = 10000`). After lifetime expires, the pickup is marked inactive and culled in `updateAmmoPickups`. This prevents uncollected pickups from accumulating indefinitely and keeps the world clean. The lifetime is optional on the type (`lifetimeMs?: number`) — if omitted, the pickup persists until collected.

Rationale: The episode duration is 20 seconds (`NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS = 20000`). A 10-second lifetime means a pickup spawned early in the episode has a 50% chance of being collected before expiry, creating natural urgency without being punishing.

#### 8. Ammo restoration amount

**Decision:** Fixed amount of 5 ammo per pickup (`NEATENSTEIN_AMMO_PICKUP_AMOUNT = 5`). The player starts with 50 max ammo and each shot consumes 1 ammo. With 5 ammo per pickup, each kill effectively refunds 5 shots. Since enemies require 5 hits to kill (`NEATENSTEIN_ENEMY_MAX_HEALTH = 100`, `NEATENSTEIN_BOLT_DAMAGE = 20`), collecting every pickup would recover the 5 shots spent to kill that enemy — making pickups a 1:1 ammo recovery loop.

Alternative: 10 ammo per pickup (2:1 recovery, more generous). This is a gameplay tuning decision that can be adjusted via the constant without code changes.

#### Gameplay constants (add to `host/game/constants.ts`)

```typescript
/** Amount of ammo restored by a single ammo pickup. */
export const NEATENSTEIN_AMMO_PICKUP_AMOUNT = 5;

/** Lifetime of an ammo pickup in milliseconds before despawn. */
export const NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS = 10000;

/** Radius in world cells within which the player collects an ammo pickup. */
export const NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS = 0.5;
```

### Step-by-step implementation plan

1. **types.ts** — Add `AmmoPickupState` interface and `ammoPickups?: AmmoPickupState[]` field on `GameState`.
2. **state.ts** — Add `restoreAmmo(state, amount)` helper (clamped to `maxAmmo`). Initialize `ammoPickups: []` in `createGameState`.
3. **constants.ts** (host/game) — Add `NEATENSTEIN_AMMO_PICKUP_AMOUNT`, `NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS`, `NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS`.
4. **combat.ts** — In `applyEnemyDamage()`, when `killedByThisShot` is true, append a new `AmmoPickupState` at `enemy.position` to `state.ammoPickups`.
5. **tick.ts** — Add `updateAmmoPickups()` helper and insert Step 5c in `gameTick()` pipeline (between Step 5b and Step 6). Handle proximity collision, ammo restoration via `restoreAmmo()`, lifetime expiry, and inactive filtering.
6. **constants.ts** (browser-entry) — Add visual constants: `NEATENSTEIN_AMMO_PICKUP_COLOR`, `NEATENSTEIN_AMMO_PICKUP_GLOW_COLOR`, `NEATENSTEIN_AMMO_PICKUP_GLOW_BLUR_PX`, `NEATENSTEIN_AMMO_PICKUP_RADIUS_PX`.
7. **bolt-render.ts** — Add `drawAmmoPickups()` following the `drawImpactSpots()` pattern with white core + cool-white halo, additive blending, floor projection, and depth testing.
8. **frame.ts** — Extend `NeatensteinRenderFrame` with optional `ammoPickups?: readonly AmmoPickupState[]`.
9. **display.worker.ts** — Populate `frame.ammoPickups` from `gameState.ammoPickups` before posting frame. Insert `drawAmmoPickups()` call in render sequence.
10. **Tests** — Red-phase tests in: `state.test.ts` (restoreAmmo clamping, createGameState ammoPickups init), `combat.test.ts` (applyEnemyDamage spawns pickup on kill), `tick.test.ts` (updateAmmoPickups proximity collection + lifetime expiry), `types.test.ts` (AmmoPickupState shape), `bolt-render.test.ts` (drawAmmoPickups rendering).
11. **Validation** — `npx tsc --noEmit -p tsconfig.json`, `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/host/game/(state|combat|tick|types)`, `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/bolt-render`, `npm run build`, `npm run lint`.

### Files that will change in the implementation step

- `examples/neatenstein/browser-entry/host/game/types.ts` — AmmoPickupState interface + GameState field
- `examples/neatenstein/browser-entry/host/game/state.ts` — restoreAmmo helper + createGameState init
- `examples/neatenstein/browser-entry/host/game/constants.ts` — pickup gameplay constants
- `examples/neatenstein/browser-entry/host/game/combat.ts` — spawn hook in applyEnemyDamage
- `examples/neatenstein/browser-entry/host/game/tick.ts` — updateAmmoPickups + Step 5c
- `examples/neatenstein/browser-entry/constants.ts` — pickup visual constants
- `examples/neatenstein/browser-entry/renderer/bolt-render.ts` — drawAmmoPickups function
- `examples/neatenstein/browser-entry/renderer/frame.ts` — NeatensteinRenderFrame extension
- `examples/neatenstein/browser-entry/worker/display.worker.ts` — frame population + draw call
- Test files: `state.test.ts`, `combat.test.ts`, `tick.test.ts`, `types.test.ts`, `bolt-render.test.ts`

### Risks

1. **tick.ts coordination sink:** tick.ts is already 23 KB with 7 pipeline steps. Adding Step 5c increases its surface. Mitigation: keep `updateAmmoPickups` as a focused helper function, not inline logic. If pickup logic grows (multiple pickup types, VFX), extract to `host/game/pickup.ts`.
2. **Determinism:** Ammo pickup spawning must remain deterministic. `applyEnemyDamage` already uses `state.simTimeMs` for timing. No RNG is needed for pickup spawning (position is the enemy's death position, amount is a constant). This preserves replay determinism.
3. **Render frame extension:** Extending `NeatensteinRenderFrame` with `ammoPickups` is backward-compatible (optional field), but `display.worker.ts` must be updated to populate it. Missing this step would result in pickups not rendering despite existing in the game state.
4. **Z-order ambiguity:** The exact z-order for pickup rendering (after sprites, before/alongside bolts) needs validation in the implementation phase. The impact-spot pattern ("after sprites, before bolts") is the recommended starting point.
5. **Ammo balance:** 5 ammo per pickup creates a 1:1 recovery loop (5 shots to kill, 5 ammo recovered). This may make ammo management trivial. Consider reducing to 3 or increasing enemy health in a later tuning pass.

### Confidence

- 0.95 — AmmoPickupState type design and GameState field addition (established pattern, no ambiguity).
- 0.93 — combat.ts spawn hook (killedByThisShot seam is well-defined and already exists).
- 0.92 — tick.ts pipeline insertion point (Step 5c between 5b and 6 is clearly the right slot).
- 0.91 — state.ts restoreAmmo (exact mirror of consumeAmmo, trivial).
- 0.88 — Rendering approach (pattern is well-established in bolt-render.ts, but z-order and visual sizing need runtime validation).
- 0.85 — Ammo balance (5 per pickup is a reasonable default but requires playtesting to confirm).

## Phase 7 Research: Enemy Death Derez Animation

### Question

What is needed to fully implement the enemy death derez animation? Specifically: (1) Is `shouldDissolvePixel` actually called in the render path? (2) Does `EnemyState` need new death animation fields? (3) Does `applyEnemyDamage()` need to change to initiate death animation? (4) How is `NEATENSTEIN_ENEMY_DEATH_COLOR` used for gray tinting? (5) What changes to `robot-sprite-data.js` are needed? (6) Does the tick loop need to advance the death animation timer?

### Evidence

Three Tier-3 specialists were dispatched in parallel (boundary-mapper, implementation-pattern-scout, docs-scout) with model `glm-5.2:cloud`. All completed successfully. Findings synthesized below with confidence levels and provenance.

#### 1. Derez renderer is FULLY WIRED (confidence: 0.95)

**Source:** `examples/neatenstein/browser-entry/renderer/derez.ts` (runtime, 90 lines)

- Three pure functions: `shouldDissolvePixel(x, y, t, seed)`, `derezHash(x, y, seed)`, `voxelToLogical(vx, vy)`.
- `shouldDissolvePixel` returns `true` when `derezHash(x, y, seed) < t`, producing a deterministic scattered dissolution pattern.
- All functions are implemented and tested (`derez.test.ts` passes: determinism, [0,1) range, t=0 no dissolution, t=1 full dissolution, scattered pattern).

**Source:** `examples/neatenstein/browser-entry/renderer/sprites.ts` (runtime)

- `NeatensteinDerezState` interface (lines 164-171): `{ derezActive: boolean, derezT: number, seed: number }`.
- `renderNeatensteinVoxelSpriteColumn` (lines 936-1024): when `derezActive` is true, calls `shouldDissolvePixel(x, y, derezT, seed)` per-pixel (line 994). Dissolved pixels are skipped (not rendered). Surviving pixels are tinted toward `NEATENSTEIN_ENEMY_DEATH_COLOR [180, 190, 210]` with `tintFactor = derezT * 0.5` (lines 1013-1021), capped at 50% gray blend.
- `renderNeatensteinSprite` (lines 1049-1128): accepts optional `derezState` as 7th argument and passes it through to the column renderer.

**Source:** `examples/neatenstein/browser-entry/worker/display.worker.ts` (runtime)

- Lines 743-753: constructs `NeatensteinDerezState` from `sprite.deRezElapsedMs`, `sprite.deRezDurationMs`, and `sprite.seed`. Maps `derezActive = sprite.deRezElapsedMs > 0`, `derezT = deRezElapsedMs / deRezDurationMs` (clamped to [0, 1]).

**Conclusion:** The entire render path from controller state → sprite state → derez state → pixel dissolution + gray tinting is fully wired. No changes needed to `derez.ts`, `sprites.ts`, or the derez state construction in `display.worker.ts`.

#### 2. CRITICAL GAP-1: Post-tick pruning resets death animation timer (confidence: 0.92)

**Source:** `examples/neatenstein/browser-entry/worker/display.worker.ts` (static code, lines ~1252-1264)

The post-tick pruning logic removes any enemy with `health <= 0` from the controller roster. The zero-timestep pass (which follows pruning) then recreates the enemy with `deRezElapsedMs = 0`, resetting the animation timer every frame.

**Root cause chain:**

1. `applyEnemyDamage()` reduces enemy `health` to 0 on kill — correctly leaves `active = true`.
2. Controller's death path activates: `animationState = 'death'`, `deRezElapsedMs` begins advancing.
3. Post-tick pruning detects `health <= 0`, removes the enemy from the controller roster.
4. Zero-timestep pass recreates the enemy from game state — `deRezElapsedMs` resets to 0.
5. Next frame: controller death path activates again, `deRezElapsedMs` starts from 0 again.
6. Net result: `deRezElapsedMs` never exceeds a single frame's `dtMs` (~16ms), far below the 700ms duration. The death animation never visibly progresses. No pixels dissolve, no gray tint is applied.

**Fix options (ranked by minimal change):**

- **Option A (recommended):** Change the pruning in `display.worker.ts` to check `deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS` before removing dead enemies. Only prune enemies whose death animation has completed.
- **Option B:** Move the pruning pass after the zero-timestep pass, so recreated enemies retain their `deRezElapsedMs` from the controller's internal state.
- **Option C:** Add `deathTimerMs` / `deathDurationMs` to `EnemyState` (types.ts) so game state tracks death progress. The controller reads these instead of maintaining its own `deRezElapsedMs`. Larger change but makes death animation replay-safe.

#### 3. Controller death path works in isolation (confidence: 0.90)

**Source:** `examples/neatenstein/scripts/enemy-controller.ts` (runtime, lines 405-428)

- `ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 700` (line 193).
- On `health <= 0`: `animationState = 'death'`, `deRezElapsedMs += dtMs`, `active = deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS`.
- After 700ms: `active = false`, enemy is fully removed.
- Tested in `enemy-controller.test.ts` (lines 290-367): animationState='death', active=true while animating, active=false after duration, deRezElapsedMs > 0 after tick. All tests pass.

**Conclusion:** The controller death path is correct. The bug is in the display worker's integration with the controller, NOT in the controller itself.

#### 4. EnemyState does NOT need new fields (confidence: 0.88)

**Source:** `examples/neatenstein/browser-entry/host/game/types.ts` (static code, lines 52-65)

- `EnemyState` has: `health`, `active?`, `stunTimerMs?` — no death animation timer fields.
- Research conclusion: `EnemyState` does NOT need new fields under Option A or B. The death timer correctly lives in `ControlledEnemy` (enemy-controller.ts internal state).
- Under Option C (game-state-driven), `EnemyState` would gain `deathTimerMs` and `deathDurationMs` fields. This is optional for replay safety.

#### 5. combat.ts does NOT need changes (confidence: 0.85)

**Source:** `examples/neatenstein/browser-entry/host/game/combat.ts` (static code, lines 334-395)

- `applyEnemyDamage()` detects kills via `killedByThisShot = newHealth === 0`.
- Does NOT set `active = false` on kill — this is CORRECT behavior, as the death animation needs the enemy to remain active.
- The controller infers death from `health <= 0` and activates the death path.

**Conclusion:** No changes needed to `combat.ts`. The death detection is already correct.

#### 6. tick.ts does NOT need changes (confidence: 0.85)

**Source:** `examples/neatenstein/browser-entry/host/game/tick.ts` (static code)

- The tick loop does NOT advance any death animation timer. The controller handles death timing internally via `deRezElapsedMs += dtMs` in its own update method.
- No search results for "death" or "derez" in `tick.ts`.

**Conclusion:** No changes needed to `tick.ts`. The death timer advancement is controller-side.

#### 7. robot-sprite-data.js does NOT need changes (confidence: 0.92)

**Source:** `examples/neatenstein/robot-sprite-data.js` (static code)

- 9-entry `[R, G, B, A]` palette (indices 0-8). `ROBOT_SPRITE_SCALE = 4`.
- Indices 5, 6, 7 are team-color swappable (set per-game via `NEATENSTEIN_ENEMY_TEAM_COLORS`).
- Gray death tint is applied at RENDER TIME as a per-pixel lerp toward `NEATENSTEIN_ENEMY_DEATH_COLOR [180, 190, 210]` in `sprites.ts` lines 1013-1021. It is NOT a palette-level change.
- The tint factor is `derezT * 0.5` (capped at 50% gray blend), applied uniformly across all palette indices.

**Conclusion:** No changes needed to `robot-sprite-data.js`. The gray mapping is render-time, not palette-level.

#### 8. Secondary gap: parallel billboard render path (confidence: 0.70)

**Source:** `examples/neatenstein/scripts/enemy-sprite.ts` (static code)

- The billboard renderer reads `deRezElapsedMs` but only applies orange death-bolt lighting.
- Does NOT call `shouldDissolvePixel` or tint toward `NEATENSTEIN_ENEMY_DEATH_COLOR`.
- This is a secondary gap if the billboard render path is used alongside the voxel sprite path.
- Confidence is 0.70 — needs validation of whether the billboard path is active in the current game.

#### 9. Tint factor cap at 50% (confidence: 0.75)

**Source:** `examples/neatenstein/browser-entry/renderer/sprites.ts` (static code, line 1015)

- `tintFactor = derezT * 0.5` means maximum 50% gray blend at full dissolution (t=1).
- Applied uniformly across all palette indices — no differential desaturation for team-color pixels vs chassis pixels.
- A more dramatic TRON-style effect could use a higher cap (e.g., `tintFactor = derezT * 0.8`) or differential desaturation (team colors desaturate faster than chassis).
- This is an aesthetic enhancement, not a functional gap.

### Decision

**Primary fix (GAP-1):** Implement Option A — change the post-tick pruning in `display.worker.ts` (lines ~1252-1264) to respect the controller's death animation window. Only prune enemies where `deRezElapsedMs >= ENEMY_CONTROLLER_DE_REZ_DURATION_MS` (700ms). This is the minimal change that unblocks the death animation.

**Files that do NOT need changes:**

- `derez.ts` — fully implemented and tested.
- `sprites.ts` — derez integration fully wired.
- `types.ts` — `EnemyState` does not need new fields (Option A).
- `combat.ts` — death detection is correct.
- `tick.ts` — timer advancement is controller-side.
- `robot-sprite-data.js` — gray tint is render-time, not palette-level.

**File that MUST change:**

- `display.worker.ts` — fix the post-tick pruning to respect the death animation window.

**Optional secondary enhancements:**

- `scripts/enemy-sprite.ts` — add dissolution + gray tinting to the billboard render path (if active).
- `sprites.ts` — increase tint factor cap for more dramatic TRON effect (aesthetic).

### Step-by-step implementation plan

1. **Fix GAP-1 in `display.worker.ts`** (lines ~1252-1264): Add a guard to the post-tick pruning that checks the controller's `deRezElapsedMs` before removing dead enemies. Only prune when `deRezElapsedMs >= ENEMY_CONTROLLER_DE_REZ_DURATION_MS` (700ms). This allows the death animation to play fully before the enemy is removed.

2. **Red-phase test for GAP-1 fix**: Write a test in a new `display-worker-derez.test.ts` that verifies:
   - An enemy with `health = 0` and `deRezElapsedMs < 700` is NOT pruned.
   - An enemy with `health = 0` and `deRezElapsedMs >= 700` IS pruned.
   - The death animation progresses (deRezElapsedMs increases across frames).

3. **Optional: Billboard render path**: If `scripts/enemy-sprite.ts` is used in the current game, add `shouldDissolvePixel` call and gray tinting to match the voxel sprite path. This is secondary to GAP-1.

4. **Optional: Tint factor enhancement**: Increase `tintFactor = derezT * 0.5` to `derezT * 0.7` or `derezT * 0.8` for a more dramatic gray blend at full dissolution. Aesthetic choice.

5. **Validation**: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/derez`, `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/scripts/enemy-controller`, `npx tsc --noEmit -p tsconfig.json`.

### Files that will change in the implementation step

- `examples/neatenstein/browser-entry/worker/display.worker.ts` — fix post-tick pruning to respect death animation window (GAP-1).
- `examples/neatenstein/browser-entry/worker/display-worker-derez.test.ts` — new red-phase test for pruning guard (optional, recommended).
- `examples/neatenstein/scripts/enemy-sprite.ts` — optional: add dissolution + gray tinting to billboard render path.
- `examples/neatenstein/browser-entry/renderer/sprites.ts` — optional: increase tint factor cap (aesthetic).

### Risks

- **GAP-1 fix risk:** The pruning logic may have other dependencies (e.g., roster size limits, spawn timing) that assume immediate removal of dead enemies. The fix must ensure that keeping dead enemies in the roster for 700ms does not break spawn logic or enemy count limits.
- **Option C (game-state-driven) trade-off:** If the team later wants replay-safe death animation (deterministic from game state alone), Option C (adding `deathTimerMs` to `EnemyState`) would be needed. Option A is simpler but relies on controller internal state surviving across frames.
- **Billboard render path uncertainty:** Confidence 0.70 that `scripts/enemy-sprite.ts` is an active code path. Needs validation before implementing dissolution there.
- **Tint factor aesthetic:** The 50% cap is a design choice. Increasing it changes the visual identity of the death effect. User approval recommended.

### Confidence

| Finding                                    | Confidence | Action                                                    |
| ------------------------------------------ | ---------- | --------------------------------------------------------- |
| Derez renderer fully wired                 | 0.95       | ACT — no renderer changes needed                          |
| GAP-1: post-tick pruning resets timer      | 0.92       | ACT — fix pruning in display.worker.ts                    |
| Controller death path works in isolation   | 0.90       | ACT — no controller changes needed                        |
| EnemyState does NOT need new fields        | 0.88       | ACT — no types.ts changes needed (Option A)               |
| combat.ts does NOT need changes            | 0.85       | ACT — no combat.ts changes needed                         |
| tick.ts does NOT need changes              | 0.85       | ACT — no tick.ts changes needed                           |
| robot-sprite-data.js does NOT need changes | 0.92       | ACT — no palette changes needed                           |
| Billboard render path gap                  | 0.70       | DELEGATE — validate if path is active before implementing |
| Tint factor cap at 50%                     | 0.75       | DELEGATE — aesthetic choice, user approval needed         |
