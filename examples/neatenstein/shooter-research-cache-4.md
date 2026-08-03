# Research Cache 4: Plasma Bolt Collision

## Root Cause Analysis
Plasma bolts do NOT collide with enemies while traveling. Enemy hits are an instant hitscan-style ray test performed ONCE at fire time in `fireBolt()` (combat.ts). The traveling `BoltState` projectile is moved every tick by `updateBolts()` (tick.ts) without ever checking enemies again. Bolts visually pass through enemies.

Key issues:
1. Bolt is not stopped/deactivated at the enemy — continues to `targetDistance` and beyond
2. Enemies that move into bolt path AFTER firing are ignored
3. Fire-time test uses `NEATENSTEIN_BOLT_HIT_RADIUS_CELLS = 0.4` instead of actual enemy radius `96/252 ≈ 0.381`
4. No continuous swept collision — bolt moves 0.576 cells/tick, enemy radius only 0.381, so could tunnel

## Code Locations
- `examples/neatenstein/browser-entry/host/game/combat.ts:121-240` — `fireBolt()` instant ray-enemy test
- `examples/neatenstein/browser-entry/host/game/tick.ts:323-373` — `updateBolts()` moves bolts, checks walls but NEVER enemies
- `examples/neatenstein/browser-entry/host/game/types.ts:65-80` — `BoltState` definition
- `examples/neatenstein/browser-entry/host/game/constants.ts:293-333` — bolt constants (speed 36 cells/s, damage 50, hit radius 0.4)
- `examples/neatenstein/browser-entry/renderer/bolt-render.ts:194-344` — bolt rendering

## Proposed Fix
1. Add continuous enemy collision to `updateBolts()` using segment-vs-circle test
2. When bolt segment passes within enemy radius, stop bolt at impact point, apply damage via `applyEnemyDamage()`, deactivate bolt
3. Use actual `NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS` instead of arbitrary hit radius
4. Add swept collision to prevent tunneling (bolt step 0.576 > enemy radius 0.381)
5. Optionally shorten bolt visual travel to enemy hit distance in bolt-render.ts

## Files to Change
1. `examples/neatenstein/browser-entry/host/game/combat.ts` — use enemy collision radius, store nearest enemy index on bolt
2. `examples/neatenstein/browser-entry/host/game/types.ts` — add `hitEnemyIndex?` and `enemyHitDistance?` to BoltState
3. `examples/neatenstein/browser-entry/host/game/tick.ts` — add enemy collision to `updateBolts()`
4. `examples/neatenstein/browser-entry/host/game/constants.ts` — align hit radius with enemy radius
5. `examples/neatenstein/browser-entry/renderer/bolt-render.ts` — stop bolt visual at enemy hit distance
6. Test files: `combat.test.ts`, `tick.test.ts`