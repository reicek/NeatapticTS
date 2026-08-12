# Neatenstein Tron-Style Derez Death Animation — Research

> Research-only artifact for implementing a Tron-like disintegration ("derez")
> death animation in the Neatenstein demo (`examples/neatenstein/`).
> No file edits were made during this investigation.

## Question

How can a Tron-style "derez" (disintegration) death animation be implemented for
enemies in the Neatenstein demo, replacing the current death behavior? The effect
should show the enemy breaking into glowing particles/shards that fly outward and
fade over a short duration (~0.5-1s) so gameplay is not slowed.

## Evidence

### 1. Existing animation system

- **Walk cycle:** `stand → walk1 → stand → walk2`, indexed by
  `Math.floor(walkTick / 4) % 4` (`sprites.ts:354-360`,
  `NEATENSTEIN_WALK_CYCLE_POSES`).
- **Frame selection:** `resolveNeatensteinEnemyFrame()` (`sprites.ts:619-688`)
  maps `EnemyAnimationState` → pose via `NEATENSTEIN_ANIMATION_TO_POSE`
  (`sprites.ts:372-381`). States: `idle→stand`, `move→walk1`, `fire→shoot`,
  `death→stand`, `damage→stand`.
- **Animator:** `enemy-animator.ts` — `MS_PER_FRAME = 100ms`; frame counts:
  idle=6, move=12, fire=3, death=12, damage=2. `getEnemyAnimationFrame()` is
  deterministic from (state, elapsedMs, seed).
- **Critical gap:** The `death` state maps to the **`stand` pose** — the worker
  render path renders NO death-specific visual. The enemy simply stands still
  during the entire death window.

### 2. Existing "de-rez" death animation (ALREADY EXISTS, but is NOT Tron-style)

- **Controller (`enemy-controller.ts:397-420`):** When `health <= 0`,
  `deRezElapsedMs += dtMs`, `animationState = 'death'`, and
  `active = deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS` (4000ms).
  Enemy stays active (rendered) for 4 seconds, then becomes inactive and is
  filtered out (`display.worker.ts:1188` `.filter(c => c.active)`).
- **Billboard/CPU render path (`enemy-sprite.ts:42-48,540-568,700-823`):**
  Has `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS = 4000`, applies an orange bolt
  light tint (`ENEMY_BOLT_LIGHT_ORANGE = 0xff8c00`) via `applyBoltLight()` —
  a linear fade + sine pulse flicker (period 200ms). This is the only visual
  death effect, and it only exists on the **billboard/CPU render path**, NOT the
  worker-tier render path used for the primary demo.
- **Worker render path (`sprites.ts`):** `death → 'stand'` pose. No bolt light,
  no tint, no disintegration. The enemy is visually static for 4 seconds then
  vanishes.

### 3. Render pipeline & frame loop

- **Loop (`browser-entry.ts:329-451`):** Worker-paced rAF. `deltaMs` derived from
  consecutive rAF timestamps, clamped to `MAX_DELTA_MS = 64ms`. First frame
  falls back to 16ms. The worker derives `timestepMs = deltaMs > 0 ? deltaMs : 16`
  (`display.worker.ts:1142-1143`).
- **simState handler (`display.worker.ts:1124-1206`):**
  1. Advance enemy controller BEFORE gameTick (synced positions/health/active).
  2. Map controller state → `gameState.enemies` (position, health, active).
  3. `gameTick()`.
  4. Filter dead enemies: `.map(...)` sets `active=false,health=0` for dead,
     then `.filter(c => c.active)`.
  5. Zero-timestep re-sync pass.
  6. `buildAndPostFrame()`.
- **Worker-tier render order (`display.worker.ts`):** clear → floor grid →
  ceiling grid → fogged wall stripes (DDA raycast) → **enemy sprites**
  (`getImageData` snapshot → per-sprite `renderNeatensteinSprite` →
  `putImageData` flush) → ambient pulses → impact spots → bolts → gun overlay
  → `commit()`.
- **Sprite render path (`display.worker.ts:679-736`):** Single `getImageData`
  for entire canvas, per-sprite `renderNeatensteinSprite` writes RGBA into the
  shared buffer with z-buffer occlusion + distance fog, single `putImageData`
  flush. Active enemy sprite list built at `:494-506` filtered by `.active`.

### 4. Canvas capabilities

- **Context:** `OffscreenCanvasRenderingContext2D` on the worker tier.
- **Available:** `putImageData`, `getImageData`, `fillRect`,
  `createLinearGradient`, `fillStyle`, `shadowColor`/`shadowBlur` (used by
  pulses), `globalAlpha`, `translate`/`rotate`/`scale`, `commit()`.
- **Sprite rendering is manual RGBA writes** — NOT using context transforms or
  `globalAlpha`. Pixel-level manipulation is fully possible via `ImageData`.
- **Pulses (`display.worker.ts:739`):** Existing effect system using
  `shadowBlur`/gradient fills — a reference pattern for a new effect layer.

### 5. Sprite data format

- **`robot-sprite-data.js`:** 48×48 logical grid, palette indices 0-8,
  `ROBOT_SPRITE_SCALE = 4` (→ 192×192 output). Palette: transparent(0),
  dark grays(1-3), white(4), team-red(5-6), semitransparent(7-8). Frames
  organized by direction × pose.
- **`renderNeatensteinSprite` (`sprites.ts:982-1059`):** Decodes encoded frames
  to RGBA `VoxelSnapshot` via nearest-neighbor scaling, writes per-column with
  z-buffer occlusion + distance fog + alpha skip. Per-pixel alpha manipulation
  is feasible here for a dissolve effect.

### 6. Enemy removal flow

- **Combat (`combat.ts:296-313`):** `applyEnemyDamage` clamps health to 0,
  increments `kills` counter. Does NOT set `active = false` — that is the
  controller's job via `deRezElapsedMs` timing.
- **`EnemyState` (`types.ts:51-60`):** `position`, `health`, `active?`,
  `controllerPosition?`. No `deRezElapsedMs` field on the host-side type —
  that field lives only on the controller-side enemy
  (`enemy-controller.ts` `EnemyControllerEnemy`).
- **Filtering (`display.worker.ts:1177-1189`):** After gameTick, dead enemies
  get `active=false` then are `.filter()`-ed out of the controller roster. The
  render list is built from `.filter(enemy => enemy.active)` at `:494-495`.

### 7. No existing particle system

- There is no dedicated particle/visual-effect system. The closest analogues:
  - **Pulses** (`updateNeatensteinPulses`) — ambient expanding rings.
  - **Impact spots** — wall decals with lifetime.
  - **Bolts** — traveling projectiles.
- A true Tron derez needs a NEW particle-burst or dissolve effect layer.

## Decision: Proposed Implementation Approach

### Recommended: Hybrid particle-burst + pixel-dissolve (3-phase)

A true Tron derez should combine a pixel dissolve of the sprite body with a
glowing particle burst. The existing 4s duration should be reduced to **~700ms**
(0.7s) per the user's ~0.5-1s target.

**Phase 1 — Spawn (0ms):** On `health→0`, capture the enemy's last projected
sprite rectangle (screen x/y/w/h) and the decoded RGBA frame. Initialize a
derez state: `{ startMs, durationMs: 700, particles[], dissolveProgress: 0 }`.

**Phase 2 — Dissolve + Burst (0-700ms):**

- **Pixel dissolve:** In `renderNeatensteinSprite` (or a new
  `renderDerezSprite`), decrease per-pixel alpha as `t = elapsed/duration`
  advances. Use a noise/seeded threshold so pixels vanish in a scattered
  pattern (top-to-bottom or radial) rather than uniformly — Tron-style.
  Tint surviving pixels toward the enemy's team color or white-cyan glow.
- **Particle burst:** Spawn N particles (e.g. 24-40) at the sprite center with
  random outward velocities. Render as small glowing quads using the existing
  `shadowBlur`/gradient fill pattern from the pulse system. Particles fade
  alpha → 0 and shrink over the 700ms window. Glowing team-colored trails.

**Phase 3 — Expire (700ms+):** `active = false`, enemy filtered out as today.

### Alternative approaches (ranked)

1. **Particle burst only** (simplest): Keep the sprite as-is (or freeze on last
   frame), spawn a glowing particle burst from the sprite center. Less
   visually "Tron" but minimal sprite-path changes. Render particles in the
   post-sprite effect layer (before bolts).
2. **Pixel dissolve only**: Modify `renderNeatensteinSprite` to apply an
   alpha-dissolve mask during the death state. No separate particle system.
   Cheap, deterministic, but lacks the "shards flying outward" feel.
3. **Shard/fragment effect**: Slice the decoded sprite frame into N×M tiles,
   displace each tile outward with rotation + gravity, fade alpha. Most
   visually impressive but highest complexity (requires per-tile transform
   tracking and render).

### Timing recommendation

- **Duration: 700ms** (within the user's 0.5-1s range). Current 4000ms is far
  too long and would slow gameplay.
- **Frame budget at 60fps:** ~42 frames. At 30fps: ~21 frames. Sufficient for
  a smooth dissolve + particle fade.
- The `deRezElapsedMs` infrastructure already exists and is passed through the
  controller — only the duration constant needs changing, and `deRezElapsedMs`
  needs to be propagated to the render path (currently it is NOT on the
  `NeatensteinSprite` interface or `activeEnemySprites` map).

## Impacted Files

| File                                                            | Change                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/neatenstein/scripts/enemy-controller.ts`              | Reduce `ENEMY_CONTROLLER_DE_REZ_DURATION_MS` from 4000 → 700. Expose `deRezElapsedMs` on the controller enemy so the render path can read it.                                                                                                                                      |
| `examples/neatenstein/browser-entry/worker/display.worker.ts`   | Pass `deRezElapsedMs` (and derez duration) into `activeEnemySprites` map (`:494-506`). Add a derez-effect render pass in the painter order (after enemy sprites, before/with pulses). Particle spawn on death transition.                                                          |
| `examples/neatenstein/browser-entry/renderer/sprites.ts`        | Add `deRezElapsedMs`/`deRezDurationMs` to `NeatensteinSprite` interface (`:136`). Add a `renderDerezSprite` or modify `renderNeatensteinSprite` to apply alpha-dissolve + glow tint when in death state. Update `NEATENSTEIN_ANIMATION_TO_POSE` death mapping (currently `stand`). |
| `examples/neatenstein/browser-entry/renderer/frame.ts`          | If `deRezElapsedMs` must cross the worker/host boundary via the render frame protocol, add a typed-array field. (Likely NOT needed if the worker owns rendering.)                                                                                                                  |
| `examples/neatenstein/scripts/enemy-sprite.ts`                  | If the billboard/CPU render path is still used, update its de-rez from orange tint → Tron dissolve/particles, and reduce its `ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS` to match (700ms).                                                                                             |
| `examples/neatenstein/scripts/enemy-animator.ts`                | If death uses 12 atlas frames over the new 700ms (→ ~58ms/frame), update the death frame timing. May not need changes if the animator's death row is abandoned in favor of the dissolve.                                                                                           |
| `examples/neatenstein/robot-sprite-data.js`                     | No change needed for particle/dissolve (operates on decoded RGBA). Only needed if a dedicated "derez" sprite atlas row is desired.                                                                                                                                                 |
| `examples/neatenstein/browser-entry/host/game/types.ts`         | Optionally add `deRezElapsedMs?` to `EnemyState` if the host needs to know death progress. Currently not required.                                                                                                                                                                 |
| `examples/neatenstein/browser-entry/host/game/combat.ts`        | No change — damage/kill logic stays the same. The animation timing is the controller's responsibility.                                                                                                                                                                             |
| `examples/neatenstein/browser-entry/host/game/constants.ts`     | Optionally add a shared `NEATENSTEIN_DEREZ_DURATION_MS = 700` constant if both host and worker need it.                                                                                                                                                                            |
| **NEW: `examples/neatenstein/browser-entry/renderer/derez.ts`** | Recommended new module: derez particle system + dissolve mask logic. Keeps the effect isolated and testable.                                                                                                                                                                       |

## Risks

- **Determinism:** The demo emphasizes deterministic simulation (seeded RNG,
  `getEnemyAnimationFrame` is deterministic). Any particle RNG must be seeded
  by the enemy index + death tick, not `Math.random()`.
- **z-buffer interaction:** Particles drawn in the post-sprite layer would not
  be z-buffer occluded by closer walls. They should either be occluded (sample
  zBuffer at their screen x) or drawn as a screen-space overlay (simpler but
  may show through walls). Recommend zBuffer occlusion check for correctness.
- **Performance:** Per-pixel dissolve on a 192×192 sprite each frame is fine
  (already doing per-pixel writes). Particle count should be bounded (~40).
- **Two render paths:** The worker path (`sprites.ts`) and billboard path
  (`enemy-sprite.ts`) must both be updated or the effect will only appear on
  one tier. The worker path is the primary demo path.
- **Test coverage:** `combat.test.ts`, `enemy-controller` tests, and sprite
  tests will need updating for the new duration constant and death-frame
  selection logic.

## Confidence

- **0.92** — All findings verified by direct source reads. The existing
  de-rez infrastructure (controller timing, `deRezElapsedMs`, `active` flag,
  death animation state) is well-established and reusable. The render pipeline
  and canvas capabilities are fully understood. The only uncertainty is whether
  the billboard/CPU path (`enemy-sprite.ts`) is actively used in the demo or
  is a legacy alternate — but the worker path is confirmed primary.
