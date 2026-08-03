# Research Cache 7: Enemy AI Stopping and Spawn Logic

## Root Cause Analysis
Enemies stop moving for TWO independent reasons:

1. **Hard stop distance** — `ENEMY_CONTROLLER_STOP_DISTANCE_CELLS = 0.3` causes every living enemy to stop walking once within 0.3 cells of player. Still fires but no longer pushes forward, so groups freeze at stand-off distance.

2. **Ammunition-driven "death"** — each enemy gets only `ENEMY_CONTROLLER_STARTING_AMMO = 3` hitscan shots. After 3rd shot, enemy enters 4-second de-rez death animation and becomes inactive, even at full health. Makes enemies appear to stop and vanish.

Spawning is also wrong:
- **No wave/batch gate** — `spawnWaveTick` adds one enemy per tick as long as `state.enemies.length < 8`. Does NOT check if existing enemies are alive or dead.
- **Dead enemies never removed** from `GameState.enemies` — health set to 0 but remain in array. Once 8 enemies ever spawned, spawner thinks arena is full forever, even if all dead.
- **Generations not wired** — `advanceWave` exists in `host/waves.ts` but is NOT called by the live game loop. Worker only calls `gameTick → updateEpisode → spawnWaveTick`.

## Code Locations
- `examples/neatenstein/scripts/enemy-controller.ts:116` — stop distance constant (0.3)
- `examples/neatenstein/scripts/enemy-controller.ts:434` — movement guard using stop distance
- `examples/neatenstein/scripts/enemy-controller.ts:131,396-421` — ammo depletion death trigger
- `examples/neatenstein/scripts/enemy-controller.ts:423-452` — movement/seek logic
- `examples/neatenstein/browser-entry/worker/display.worker.ts:478-502` — per-frame enemy update
- `examples/neatenstein/browser-entry/host/game/waves.ts:55-97` — trickle spawner (one per tick, count cap only)
- `examples/neatenstein/browser-entry/host/game/episode.ts:237-251` — episode update invokes spawner
- `examples/neatenstein/browser-entry/host/game/episode.ts:265-281` — terminal conditions (includes time limit)
- `examples/neatenstein/browser-entry/host/waves.ts:96-122` — `advanceWave` (unused in live loop)

## Proposed Fix

### 1. Make enemies chase until death
- Set `ENEMY_CONTROLLER_STOP_DISTANCE_CELLS = 0` (or remove the distance guard) so enemies always step toward player while alive
- Remove ammo depletion as death trigger — enemies should only die when `health <= 0` (killed by player)
- Keep fire cooldown and line-of-sight checks, but enemies push forward while attacking

### 2. Only spawn next batch after all destroyed
- Change `spawnWaveTick` to check `state.enemies.some(e => e.health > 0)` — refuse to spawn while any alive
- When arena has no living enemies, spawn entire next batch at once (up to 8)
- Remove dead enemies from `GameState.enemies` after de-rez completes
- Wire `advanceWave` into worker loop, triggered only when previous batch fully cleared
- Remove episode time-limit terminal condition if generations should not be time-bound

## Files to Change
1. `examples/neatenstein/scripts/enemy-controller.ts` — stop distance, ammo death path
2. `examples/neatenstein/browser-entry/host/game/waves.ts` — alive-check and batch-spawn
3. `examples/neatenstein/browser-entry/host/game/episode.ts` — filter de-rezzed enemies, remove time limit
4. `examples/neatenstein/browser-entry/worker/display.worker.ts` — wire wave advancement
5. `examples/neatenstein/browser-entry/host/game/constants.ts` — batch size constant if needed
6. Test files: `enemy-controller.test.ts`, `waves.test.ts`, `episode.test.ts`