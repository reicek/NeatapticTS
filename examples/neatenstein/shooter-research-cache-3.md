# Research Cache 3: Enemy Perspective From All Angles

## Root Cause Analysis
Enemies always look "paper-thin" because **every enemy's `facing` (yawRad) is forced to point directly at the player** in the enemy controller, which cancels out the relative-yaw math in the renderer.

The frame resolver computes:
```
cameraRelativeYaw = atan2(camera.posY - sprite.worldY, camera.posX - sprite.worldX)  // angle from ENEMY → HERO
relativeYaw = cameraRelativeYaw - sprite.facing
```

The enemy controller sets `sprite.facing = atan2(toPlayer.direction.y, toPlayer.direction.x)` — exactly `cameraRelativeYaw`. Therefore `relativeYaw ≈ 0` for every enemy every frame, so `yawIndexFromRelativeYaw(0) === 0` → **`front`** atlas frame for ALL enemies regardless of screen position.

Result: center enemy looks correct (front view), side enemies look "paper thin" (flat front billboard viewed obliquely).

The angle math and 45° sector quantizer are correct in isolation — the bug is the coupling between controller (forces facing = direction-to-player) and renderer (subtracts that same direction).

## Code Locations
- `examples/neatenstein/browser-entry/renderer/sprites.ts:592-602` — `yawIndexFromRelativeYaw` (8-step quantizer, 45° each)
- `examples/neatenstein/browser-entry/renderer/sprites.ts:614-685` — `resolveNeatensteinEnemyFrame` (key calc at `:635-640`)
- `examples/neatenstein/browser-entry/renderer/sprites.ts:317-328` — yaw-step constants (8 steps, 45° each)
- `examples/neatenstein/browser-entry/renderer/sprites.ts:383-392` — `NEATENSTEIN_ENCODED_DIRECTIONS` order: front, frontRight, right, backRight, back, backLeft, left, frontLeft
- `examples/neatenstein/browser-entry/renderer/sprites.ts:736-800` — `projectNeatensteinSprite`
- `examples/neatenstein/scripts/enemy-controller.ts:424-429` — alive path: `yawRad = atan2(toPlayer...)` (always faces player)
- `examples/neatenstein/scripts/enemy-controller.ts:404` — death path: same
- `examples/neatenstein/browser-entry/worker/display.worker.ts:462-472` — camera dir/plane derivation; `planeScale` widens with aspect
- `examples/neatenstein/browser-entry/worker/display.worker.ts:410-435` — `castColumnRay` (per-column ray direction)
- `examples/neatenstein/browser-entry/renderer/floor.ts:57` — `NEATENSTEIN_FLOOR_FOV_RADIANS = Math.PI / 3` (vertical FOV)
- `examples/neatenstein/robot-sprite-data.js:24-803` — 8 direction keys

## Current Angle Calculation (WRONG)
```ts
// sprites.ts:635-640
const cameraRelativeYaw = Math.atan2(camera.posY - sprite.worldY, camera.posX - sprite.worldX);
const relativeYaw = cameraRelativeYaw - sprite.facing;  // facing == cameraRelativeYaw → relativeYaw ≈ 0
const yawIndex = yawIndexFromRelativeYaw(relativeYaw);  // always 0 = front
```

```ts
// enemy-controller.ts:424-429 (alive enemies)
yawRad = Math.atan2(toPlayer.direction.y, toPlayer.direction.x);  // = cameraRelativeYaw
```

## Correct Angle Calculation (FIX)
Use the **hero's view angle** to the enemy (hero→enemy bearing minus hero facing), NOT the enemy's facing:

```ts
// Hero's view angle to the enemy (world-space), then subtract hero facing.
const cameraYaw = Math.atan2(camera.dirY, camera.dirX); // dir == (cos yaw, sin yaw)
const viewAngle = Math.atan2(sprite.worldY - camera.posY, sprite.worldX - camera.posX) - cameraYaw;

// Negate for handedness: engine plane=(−dirY, dirX)*scale, so +Y world delta = RIGHT of screen.
// Enemy on LEFT of screen → negative viewAngle → negate → positive → right/frontRight/backRight.
// Enemy on RIGHT of screen → positive viewAngle → negate → negative → left/frontLeft/backLeft.
const yawIndex = yawIndexFromRelativeYaw(-viewAngle);
```

Handedness verification (yaw=0, dir=(1,0), plane=(0,1)·scale):
- Enemy directly ahead → viewAngle ≈ 0 → index 0 `front` ✓
- Enemy far left (dy<0) → viewAngle ≈ −90° → negate → +90° → index 2 `right` ✓ (hero sees enemy's right side)
- Enemy far right (dy>0) → viewAngle ≈ +90° → negate → −90° → 270° → index 6 `left` ✓ (hero sees enemy's left side)
- Enemy behind → |viewAngle| ≈ 180° → `back` ✓

## Ultra-Wide Considerations
The renderer keeps vertical FOV fixed at π/3 and widens horizontal FOV with aspect ratio via `planeScale`. The proposed view-angle approach handles this **automatically** because:
1. `viewAngle` is the true world-space bearing — an enemy at the far edge of an ultra-wide screen genuinely sits at a large bearing, so an extreme side sprite is correct
2. `yawIndexFromRelativeYaw` normalizes any angle (including ±π) into the 8 sectors
3. No assumption of narrow FOV is baked in — unlike a naive "screenX → sector" lookup

## Key Findings
1. Enemies always face the player (`enemy-controller.ts:428`) — no independent facing
2. Renderer subtracts that exact angle (`sprites.ts:635-640`) → `relativeYaw ≈ 0` always
3. `relativeYaw ≈ 0` always maps to `front` (index 0) — all enemies show front frame
4. The angle used is world-position based (not screen column) — correct approach, but defeated by facing coupling
5. Correct angle should be `atan2(enemyY − heroY, enemyX − heroX) − heroYaw` (hero→enemy bearing)
6. 8 sectors are 45° each — quantization is correct
7. Per-column ray angle is `atan2(rayDirY, rayDirX)` — same math, can reuse
8. Must **negate** view angle before quantization for correct left/right handedness
9. Ultra-wide: world-space bearing is FOV-agnostic, handles edge enemies automatically
10. Unit tests mask the bug — they use artificial `facing` values that differ from camera bearing

## Proposed Fix
**Primary fix** — in `sprites.ts`, replace the relative-yaw calc in `resolveNeatensteinEnemyFrame` (`:635-640`):

```ts
// OLD (enemy-facing-relative; collapses to ~0)
const cameraRelativeYaw = Math.atan2(camera.posY - sprite.worldY, camera.posX - sprite.worldX);
const relativeYaw = cameraRelativeYaw - sprite.facing;
const yawIndex = yawIndexFromRelativeYaw(relativeYaw);

// NEW (hero-perspective view angle)
const cameraYaw = Math.atan2(camera.dirY, camera.dirX);
const viewAngle = Math.atan2(sprite.worldY - camera.posY, sprite.worldX - camera.posX) - cameraYaw;
const yawIndex = yawIndexFromRelativeYaw(-viewAngle);
```

Relax the `facing` guard at `:618-624` — the new calculation no longer needs `sprite.facing`. `NeatensteinSprite.facing` can stay optional for backward compatibility.

Update tests in `sprites.test.ts` to assert hero-perspective behavior (camera facing +X, enemy at (+1,0) → front; (+1,-1) → frontRight; (0,-1) → right; (-1,0) → back). Add ultra-wide regression test.

## Files to Change
1. `examples/neatenstein/browser-entry/renderer/sprites.ts` — rewrite relative-yaw calc in `resolveNeatensteinEnemyFrame` (`:635-640`); relax `facing` guard
2. `examples/neatenstein/browser-entry/renderer/sprites.test.ts` — update direction-resolution tests; add ultra-wide and edge-bearing regression tests
3. `examples/neatenstein/browser-entry/worker/display.worker.ts` — no change strictly required (spriteCamera already carries dirX/dirY); optionally pass cameraYaw explicitly
4. `examples/neatenstein/browser-entry/worker/display.worker.test.ts` — add integration regression for side-of-screen enemies