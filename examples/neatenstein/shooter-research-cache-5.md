# Research Cache 5: Depth Sorting — Enemy Behind Enemy

## Root Cause Analysis
The wall pass writes a correct per-column z-buffer, and each sprite column is correctly clipped against the wall z-buffer. **However, the sprite pass does NOT sort enemies by distance and never updates the z-buffer with sprite depths.** Sprites are rendered in `activeEnemySprites` array order (controller order), so a farther enemy drawn after a nearer one overwrites its pixels.

The z-buffer comparison is correct for walls but useless for sprite-vs-sprite occlusion because it only contains wall distances.

## Code Locations
- `examples/neatenstein/browser-entry/renderer/frame.ts:78-82` — z-buffer structure (per-column Float32Array)
- `examples/neatenstein/browser-entry/worker/display.worker.ts:522-578` — wall z-buffer fill
- `examples/neatenstein/browser-entry/renderer/zbuffer.ts:248-284` — `clipNeatensteinSpriteSpan` (per-column wall test)
- `examples/neatenstein/browser-entry/renderer/sprites.ts:736-805` — sprite projection (uses perpendicular distance `perpDist`)
- `examples/neatenstein/browser-entry/worker/display.worker.ts:580-625` — sprite render loop (NO SORTING)

## Proposed Fix
Sort enemies **far-to-near** (descending `perpDist`) before rendering. This is the standard painter's algorithm for sprite occlusion.

```ts
const spriteDrawList = activeEnemySprites
  .map((sprite) => {
    const frame = resolveNeatensteinEnemyFrame(sprite, spriteCamera);
    if (!frame) return null;
    const projection = clipNeatensteinSprite(sprite, spriteCamera, canvasWidth, canvasHeight, zBuffer);
    return projection.visible ? { sprite, frame, projection } : null;
  })
  .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
  .sort((a, b) => b.projection.perpDist - a.projection.perpDist);
```

Optional hardening: update z-buffer with sprite depth per column after drawing each sprite:
```ts
for (const column of projection.visibleColumns) {
  zBuffer[column] = Math.min(zBuffer[column], projection.perpDist);
}
```

## Files to Change
1. `examples/neatenstein/browser-entry/worker/display.worker.ts` — add far-to-near sprite sort in render loop (~lines 580-625)
2. `examples/neatenstein/browser-entry/worker/display.worker.test.ts` — add regression test for sprite ordering
3. `examples/neatenstein/browser-entry/renderer/sprites.ts` (optional) — expose `perpDist` if needed for tests