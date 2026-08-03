# Research Cache 1: Collision Inconsistency

## Root Cause Analysis
The hero's collision code checks `gameState.enemies` positions, but those positions are **never updated after spawning**. The actual moving enemy positions live in the worker-side `enemyControllerState` and are emitted as `activeEnemySprites` for rendering — they are NOT written back to `gameState.enemies`.

Result: `isPositionBlocked` tests the hero against original spawn points. On-screen enemies walk away from spawn, so the hero walks through them. Collision "works" only when spawn point still coincides with enemy's current location (e.g., right after spawn).

Secondary: `isPositionBlocked` does not skip dead/inactive enemies, so killed enemies' spawn points remain permanent invisible obstacles.

## Code Locations
- `examples/neatenstein/browser-entry/host/game/movement.ts:268-315` — `isPositionBlocked` (enemy distance loop)
- `examples/neatenstein/browser-entry/host/game/movement.ts:172-201` — `resolveWallCollision` calls `isPositionBlocked`
- `examples/neatenstein/browser-entry/host/game/movement.ts:216-255` — `updatePlayerMovement`
- `examples/neatenstein/browser-entry/host/game/constants.ts:98` — player speed (6 cells/s, step 0.096/tick)
- `examples/neatenstein/browser-entry/host/game/constants.ts:107` — player radius (0.25)
- `examples/neatenstein/browser-entry/host/game/constants.ts:115` — enemy radius (96/252 ≈ 0.381)
- `examples/neatenstein/browser-entry/worker/display.worker.ts:478-502` — `updateEnemyController` runs, positions NOT synced back to `gameState.enemies`
- `examples/neatenstein/browser-entry/worker/display.worker.ts:1001-1027` — `simState` handler runs `gameTick` before `buildAndPostFrame`
- `examples/neatenstein/browser-entry/host/game/waves.ts:81-86` — enemy positions created here, never moved again
- `examples/neatenstein/browser-entry/host/game/types.ts:51-56` — `EnemyState` shape (only position + health)

## Key Findings
1. `isPositionBlocked` tests desired player position (correct approach) against `state.enemies`
2. Collision array is STALE — positions come from `spawnWaveTick` and are never refreshed from AI controller
3. Hero speed (0.096 cells/tick) is NOT high enough to tunnel — combined radius 0.631 cells, step is 15% of that
4. No cross-thread race — enemy AI and player movement both run on worker, but `gameTick` runs before controller output is synced back
5. `isPositionBlocked` checks every enemy with no spatial filter — issue is wrong positions, not skipped enemies
6. Collision math is correct (squared Euclidean, combined radius)
7. Dead/inactive enemies NOT skipped — killed enemies' spawn points remain solid obstacles

## Proposed Fix
1. **Sync controller positions back into `gameState.enemies`** — after `updateEnemyController`, copy each controlled enemy's `position`, `health`, `active` into corresponding `gameState.enemies` entry
2. **Add `active` field to `EnemyState`** — optional, default `true`; `waves.ts` sets `active: true` on spawn
3. **Skip inactive/dead enemies in `isPositionBlocked`** — `if (enemy.active === false || enemy.health <= 0) continue;`
4. **Respect `active` in combat and contact damage** — skip `active === false` in `fireBolt` and `resolveContactDamage`
5. **(Recommended) Reorder worker `simState` path** — run `updateEnemyController` before `gameTick` so movement collision uses same-tick enemy positions

## Files to Change
1. `examples/neatenstein/browser-entry/worker/display.worker.ts` — sync controller positions back to gameState
2. `examples/neatenstein/browser-entry/host/game/movement.ts` — skip dead/inactive enemies
3. `examples/neatenstein/browser-entry/host/game/types.ts` — add `active?` field to EnemyState
4. `examples/neatenstein/browser-entry/host/game/waves.ts` — set `active: true` on spawn
5. `examples/neatenstein/browser-entry/host/game/combat.ts` — skip inactive in fireBolt
6. `examples/neatenstein/browser-entry/host/game/collision.ts` — skip inactive in contact damage