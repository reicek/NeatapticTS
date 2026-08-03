# Research Cache 8: Enemy Spawn Positions at Map Edges

## Root Cause Analysis
All enemies currently spawn in an annulus around the **map center**. `spawnWaveTick()` computes positions as:
```ts
position: {
  x: NEATENSTEIN_SPAWN_CENTER_X + Math.cos(angle) * distance,
  y: NEATENSTEIN_SPAWN_CENTER_Y + Math.sin(angle) * distance,
}
```
- `NEATENSTEIN_SPAWN_CENTER_X/Y = floor(120/2) + 0.5 = 60.5` — same place the player starts
- Spawn radius only 8 cells (`NEATENSTEIN_ENEMY_SPAWN_RADIUS = 8`)
- Every enemy lands within a small ring around the hero

## Code Locations
- `examples/neatenstein/browser-entry/host/game/waves.ts:55-97` — `spawnWaveTick()` center-based spawn
- `examples/neatenstein/browser-entry/host/game/constants.ts:35,45,57,147-157` — enemy cap, spawn radius, center
- `examples/neatenstein/browser-entry/host/game/state.ts:69-107` — `createGameState()` places player at center
- `examples/neatenstein/browser-entry/host/game/episode.ts:237-251` — `updateEpisode()` calls spawner
- `examples/neatenstein/browser-entry/renderer/map.ts:242-299` — `buildNeatensteinMap()` and `createCollisionMap()`

## Map Structure
- `NEATENSTEIN_MAP_SIZE = 120` (120×120 cells)
- Hero starts at center (60.5, 60.5)
- Perimeter walls at x=0, x=119, y=0, y=119
- Interior walls: 12% random density
- Central arena: 9×9 cleared region around center
- Walkability: `collisionMap.isSolid(x, y)` returns true for walls/out-of-bounds

## Proposed Fix
Replace center-annulus spawn with edge-based spawn. Place each enemy at one of 8 map edges (N, NW, W, SW, S, SE, E, NE), one per edge.

### Edge positions (one cell inside perimeter):
| Compass | Cell (x,y) | World pos |
|---------|-----------|-----------|
| N       | (60, 1)   | (60.5, 1.5) |
| NW      | (1, 1)    | (1.5, 1.5) |
| W       | (1, 60)   | (1.5, 60.5) |
| SW      | (1, 118)  | (1.5, 118.5) |
| S       | (60, 118) | (60.5, 118.5) |
| SE      | (118, 118)| (118.5, 118.5) |
| E       | (118, 60) | (118.5, 60.5) |
| NE      | (118, 1)  | (118.5, 1.5) |

### Implementation:
1. Add `EDGE_DIRECTIONS` table with base cell + scan direction for each of 8 edges
2. Add `findEdgeSpawnCell(directionIndex, collisionMap)` — scans inward from edge until finding open cell
3. Replace center-annulus math in `spawnWaveTick` with `findEdgeSpawnCell(state.spawnCount % 8, map)`
4. Pass existing `collisionMap` to avoid rebuilding map each spawn
5. Fallback: center arena if edge fully blocked (should never happen)

## Files to Change
1. `examples/neatenstein/browser-entry/host/game/waves.ts` — replace spawn math, add edge helpers
2. `examples/neatenstein/browser-entry/host/game/episode.ts` — pass collisionMap to spawnWaveTick
3. `examples/neatenstein/browser-entry/worker/display.worker.ts` — pass collisionMap through
4. `examples/neatenstein/browser-entry/host/game/waves.test.ts` — update tests for edge positions
5. `examples/neatenstein/browser-entry/host/waves.test.ts` — update batch wave tests