# Research Cache 6: Render Distance Cap at 30 Cells

## Root Cause Analysis
The raycasting DDA loop runs `while(true)` with NO distance limit (`raycast.ts:159-191`). It only stops when hitting a wall cell. The map is 120×120 with a closed perimeter, so open corridors can step 50-60 cells before hitting a wall. The fog constant `NEATENSTEIN_MAX_VIEW_DIST = 140` (`framebuffer.ts:37`) is a soft fade, not a hard cap. Floor/ceiling already cap at 30 cells (`floor.ts:119`).

## Code Locations
- `examples/neatenstein/browser-entry/renderer/raycast.ts:129-192` — DDA loop (`while(true)`, no max distance)
- `examples/neatenstein/browser-entry/renderer/framebuffer.ts:37` — `NEATENSTEIN_MAX_VIEW_DIST = 140` (soft fog only)
- `examples/neatenstein/browser-entry/renderer/framebuffer.ts:45-49` — background color `#060b14`
- `examples/neatenstein/browser-entry/worker/display.worker.ts:525-527` — canvas clear to background color
- `examples/neatenstein/browser-entry/worker/display.worker.ts:542-578` — wall stripe rendering loop
- `examples/neatenstein/browser-entry/renderer/sprites.ts:736-805` — sprite projection (no far-clip)
- `examples/neatenstein/browser-entry/renderer/floor.ts:113-119` — floor/ceiling already capped at 30 cells

## Proposed Fix
1. Add `NEATENSTEIN_RENDER_DISTANCE_CAP = 30` constant in `framebuffer.ts`
2. Add step/distance limit to DDA loop in `raycast.ts` — return `perpWallDist: Infinity` when cap exceeded
3. In worker, skip wall stripe drawing for distances ≥ cap (canvas already cleared to background color)
4. Set `zBuffer[column] = Infinity` for missed columns so distant walls don't occlude sprites
5. Add far-clip to `projectNeatensteinSprite` — return invisible projection if `perpDist >= 30`
6. Optionally lower `NEATENSTEIN_MAX_VIEW_DIST` from 140 to 30

## Performance Impact
- Worst-case ray steps cut roughly in half (60 cells → 30 cells)
- ~19,200 cell steps instead of ~38,400 in worst case
- Columns beyond 30 skip fillRect/fog computation entirely
- Far enemies rejected before projection/z-buffer work
- Floor/ceiling already bounded — this aligns wall rendering with existing floor contract

## Files to Change
1. `examples/neatenstein/browser-entry/renderer/framebuffer.ts` — add render distance cap constant
2. `examples/neatenstein/browser-entry/renderer/raycast.ts` — add max distance to DDA loop
3. `examples/neatenstein/browser-entry/worker/display.worker.ts` — skip far walls, set z-buffer to Infinity
4. `examples/neatenstein/browser-entry/renderer/sprites.ts` — add far-distance cull
5. `examples/neatenstein/browser-entry/renderer/zbuffer.ts` — verify Infinity handling (likely no change needed)
6. `examples/neatenstein/browser-entry/renderer/walls.ts` — ensure far walls treated as empty