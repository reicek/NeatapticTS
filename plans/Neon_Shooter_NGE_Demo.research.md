# Neatenstein Engine Research Notes

**Aliases:** Neatenstein · "Neat Shooter" · plan ID `NEATENSTEIN_NGE_DEMO`

**Purpose:** Reference notes for implementing the Neatenstein raycasting renderer in the neon aesthetic. Summarizes how classic DOOM and grid-based raycasting work, with concrete guidance for a browser canvas implementation. No code is copied from external sources; algorithms are described in original words.

**License & Attribution:**

- Classic DOOM engine source (id Software): GNU GPL v2-or-later. Algorithm descriptions below are paraphrased from public documentation (Wikipedia "Doom engine", Fabien Sanglard's _Game Engine Black Book: DOOM_, Doom Wiki). No DOOM source code is reproduced.
- `carlini/js13k2019-yet-another-doom-clone`: GNU GPL v3.0. This repo uses WebGL polygon rendering (NOT grid raycasting) and a turtle-graphics map format. We do NOT copy its code. We reference its _structural ideas_ (polygon sectors, floor/ceiling heights, light levels, sprite billboarding) where they inform our design, but our renderer uses 2D-grid DDA raycasting on canvas 2D, which is a different rendering path.
- Lode Vandevenne's raycasting tutorial (lodev.org/cgtutor/raycasting.html): Copyright 2004-2020 Lode Vandevenne, all rights reserved. The DDA algorithm description below is paraphrased in original words; no code or text is reproduced.

**Attribution placement:** This file is the canonical attribution home. Do not scatter license notes into generated READMEs or source files.

---

## 1. Rendering Lineage — Lineage B (grid DDA raycasting, locked)

Neatenstein uses **Lineage B (grid DDA raycasting)** — the Wolfenstein 3D / Lode's tutorial lineage. The alternative (Lineage A, classic DOOM's BSP + polygon sectors) was rejected because BSP precomputation is a large subsystem with no benefit for a single-arena NGE combat demo. We borrow only three conceptual ideas from DOOM (not its engine): per-region light levels, billboarded sprite clipping, and a top-down map mode for debugging.

### Lineage B characteristics

- **Map:** 2D square grid. Each cell is 0 (empty) or a positive integer (wall variant/texture id). All walls are the same height, axis-aligned, on the grid.
- **Visibility:** For each vertical screen column, cast a ray from the player through the camera plane and walk the grid with **DDA** until a wall cell is hit. The perpendicular distance determines wall height on screen.
- **Walls:** One vertical line per screen column. Textured by sampling a texture column based on where the ray hit the cell edge.
- **Floors/ceilings:** Solid color (or simple floor-casting texture projection in advanced versions). Neatenstein uses a world-space floor grid-line projection (§3.3).
- **Sprites:** Billboarded, depth-sorted, drawn after walls.
- **Look up/down:** Not supported (vertical-wall limitation). Horizontal mouse look only (§4.2.6).
- **Strengths:** Trivial to implement, very fast, perfect for canvas 2D, no BSP precomputation.
- **Weaknesses:** No variable heights, no stairs, no rooms-over-rooms, grid-aligned walls only. None of these matter for the Neatenstein demo.

### Conceptual borrowings from DOOM (without its engine)

- Per-region light levels → we vary neon brightness per map region, encoded as a per-cell "light level" in the grid.
- Billboarded sprites clipped against walls → our enemy wireframes use the same depth-sort + clip approach.
- Top-down "map mode" → useful as a debug overlay and potentially as the network-view's spatial context.

**Abandoned approaches (superseded):** multi-hue wall variants and pink-tinted walls were dropped in favor of the fixed two-tone cyan/blue side-shading scheme, which is cheaper, avoids color-blindness problems, and reads clearly under stream compression.

---

## 2. The DDA Raycasting Algorithm (Lineage B, paraphrased)

This is the core renderer. The algorithm, in original words:

### 2.1 Camera model

The player is a 2D position `(posX, posY)` with a **direction vector** `(dirX, dirY)` and a **camera plane vector** `(planeX, planeY)`. The camera plane is perpendicular to the direction and its length relative to the direction sets the FOV:

- `|plane| / |dir| = 0.66` → 66° FOV (classic Wolf3D value).
- Direction and plane are both rotated together when the player turns (apply the 2D rotation matrix to both).

### 2.2 Per-column ray setup

For each screen column `x` (0 to width-1):

1. Compute `cameraX = 2 * x / width - 1` (ranges from -1 at left edge to +1 at right edge).
2. The ray direction is `rayDir = dir + plane * cameraX` (vector addition).

### 2.3 DDA grid traversal

The ray walks the grid one cell at a time, always jumping to the next grid line crossing:

1. `mapX, mapY` = current cell (integer floor of player position).
2. `deltaDistX = abs(1 / rayDirX)` (distance to cross one cell in X; use `1e30` if `rayDirX == 0`).
3. `deltaDistY = abs(1 / rayDirY)` (same for Y).
4. `stepX, stepY` = ±1 based on ray direction sign.
5. `sideDistX, sideDistY` = distance to the first X-side and Y-side grid line from the player position.
6. **Loop:** step to whichever side is closer (`sideDistX < sideDistY` → step X, else step Y). Increment the stepped `sideDist` by its `deltaDist`. Check if the new cell is a wall. Stop when a wall is hit. Track which side (X or Y) was hit — this is the `side` variable.

### 2.4 Perpendicular distance (fisheye avoidance)

Use the **perpendicular distance** to the wall, NOT the Euclidean distance, to avoid the fisheye effect:

- If `side == 0` (X-side hit): `perpWallDist = sideDistX - deltaDistX`.
- If `side == 1` (Y-side hit): `perpWallDist = sideDistY - deltaDistY`.

This is the distance projected onto the camera direction, which keeps walls straight instead of curved.

### 2.5 Wall column height

- `lineHeight = screenHeight / perpWallDist` (inverse projection).
- `drawStart = -lineHeight / 2 + screenHeight / 2` (clamp to 0).
- `drawEnd = lineHeight / 2 + screenHeight / 2` (clamp to screenHeight - 1).
- Draw a vertical line from `drawStart` to `drawEnd` at column `x`.

### 2.6 Texture coordinate (for textured version)

- `wallX = posY + perpWallDist * rayDirY` (if side==0) or `posX + perpWallDist * rayDirX` (if side==1), then `wallX -= floor(wallX)` → fractional position along the wall [0,1).
- `texX = floor(wallX * texWidth)`, with a flip depending on ray direction and side to avoid mirroring.
- For each screen pixel `y` in the column, interpolate `texY` and sample the texture.

### 2.7 Side shading

Walls hit on Y-sides are drawn darker than X-sides (divide RGB by 2). This gives a cheap directional-light effect that reads as depth.

---

## 3. Neatenstein Renderer Design (neon adaptation)

### 3.1 Map representation

- 2D `Uint8Array` grid (`gridW × gridH`). Cell values:
  - `0` = empty (walkable).
  - any positive value = wall cell (the renderer uses a fixed two-tone side-shading palette; no per-wall hue lookup).
- Optional: a parallel `Uint8Array` of "light levels" per cell (0-15, DOOM-style) to vary neon brightness per region. This is a cheap way to get visual variety without variable floor heights.
- The grid is transferred to the worker once at episode start; the renderer caches it locally.

### 3.2 Neon wall rendering

The implemented renderer is deliberately simple: vertical wall stripes with distance fog and side-based hue variation. No texture sampling, no neon-grid wall pattern, no continuous horizon glow.

- **CPU tier (`renderer/walls.ts`):** writes each wall column directly into a `Uint8ClampedArray` framebuffer and flushes once per frame with `putImageData`. No per-column `fillRect`, no `globalAlpha` mutations, no `shadowBlur`.
- **Worker tier (`worker/display.worker.ts`):** draws the same fogged vertical stripes through `CanvasRenderingContext2D`.
- **Wall colors:**
  - X-side hits (east/west walls): `#00bfff` (neon cyan).
  - Y-side hits (north/south walls): `#0050b4` (darker blue).
  - This replaces the earlier idea of multiple wall hue variants / pink-tinted walls; the two-tone cyan/blue scheme reads clearly against the dark background and keeps the CPU path stateless.
- **Distance fog:** linearly interpolates the wall color toward the background `#060b14` as `perpWallDist` approaches `NEATENSTEIN_MAX_VIEW_DIST`.
- **Side shading:** Y-sides are rendered with the darker blue constant; the brightness split is baked into the two base colors rather than computed per column.

**Superseded:** per-wall-variant hue palettes, neon-grid mullion/scanline wall patterns, top/bottom edge glow strokes, continuous horizon-glow paths, and pre-rendered glow sprites were all cut in favor of the two-tone fogged-stripe approach.

### 3.3 Floor and ceiling

- **Background:** solid `#060b14` for both ceiling and the empty upper screen. No starfield, no ceiling grid.
- **Horizon:** a horizontal divider at the vanishing point; the floor renderer draws below it.

#### 3.3.1 World-space floor grid projection (implemented)

The floor is rendered by projecting **world-space integer grid lines** into screen space (`renderer/floor.ts`):

1. For each integer X and Y grid line within a bounded range around the camera, sample points along the line are transformed by the camera yaw and perspective-projected onto the canvas.
2. Visible projected segments are batched into a single path per alpha band and stroked twice: a wide, low-alpha halo first, then the core 1px line. This gives a subtle neon glow without `shadowBlur` cost.
3. Alpha is depth-graded from 0.12 near the horizon to 0.58 near the bottom edge, replacing the Flappy curve helpers with a local linear mapping.
4. The grid is drawn every frame on the dynamic canvas layer; no separate static floor layer is kept.

This is **not** the earlier fake-perspective Flappy grid or a per-pixel floor-caster; it is a world-fixed line projection that rotates correctly with the camera and matches the DDA wall geometry.

#### 3.3.2 Pulses — dots on grid lines (implemented)

Pulses are small yellow dots that travel along the integer world grid lines (`renderer/pulse.ts`):

- **Shape:** screen-space dots (`NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX = 1.5`) with a small glow blur, rendered above the floor grid.
- **Movement:** world-space velocity along the chosen grid line, driven by the fixed simulation tick.
- **Emission:** deterministic Park-Miller LCG seeded with the episode seed and `simTick`; ambient interval is `2000` ms (lifetime `2700` ms), max `11` concurrent pulses.
- **Depth test:** each pulse is projected to a screen column and compared against the per-column z-buffer; pulses behind closer walls are hidden.
- **Event pulses:** generation-up and enemy-death pulses are driven by the same deterministic sim-tick events.

**Superseded:** the earlier fake-perspective pulse design (bearing-tolerance re-matching, screen-space anchoring, Flappy depth-curve helpers) was replaced by this world-space grid-line approach because it stays consistent with the rotating floor and needs no continuity cache or grazing-angle clamp.

#### 3.3.3 Ceiling

Solid `#060b14`; no ceiling grid in the current build.

#### 3.3.4 Scanline-coherent floor-casting (superseded)

A true per-pixel/per-scanline floor-caster (Lode Part 2) was considered as a Phase 8 stretch goal. It was abandoned in favor of the cheaper world-space grid-line projection in §3.3.1, which is fast enough for the CPU tier and visually consistent with the wall DDA.

### 3.4 Enemy wireframe sprites

- Each enemy is a **wireframe quad** (4 neon line segments + 2 internal cross-brace lines), NOT a textured billboard.
- Projection: transform enemy world position by the camera matrix → screen X, scale by `1 / perpDist`.
- Color by enemy type from `NEATENSTEIN_PALETTE.enemyHues[type]` (reuse `FLAPPY_NEON_BIRD_PALETTE` pattern).
- **Depth sort:** painter's algorithm — sort all enemies + projectiles by `perpDist` descending, draw far-to-near. This is the DOOM vissprite approach, simplified.
- **Glow:** `shadowBlur` on the wireframe stroke; skip on CPU tier.

#### 3.4.1 Wall clipping — per-column z-buffer (recommended)

- Use a `Float32Array(width)` z-buffer: one `perpWallDist` write per wall column, one compare per sprite column. Cheaper than a JS `Set` and handles partial occlusion.
- **Partial occlusion:** A sprite column partially behind a wall (wall covers top half, sprite visible in bottom half) cannot be handled by a boolean flag. The z-buffer handles it: sprite pixel drawn only if `spriteDist < zBuffer[x]`.
- For wireframe sprites (~6 line segments), per-column cost is trivial — each line rasterizes to a few columns, each needing one `zBuffer[x]` compare.
- DOOM lacked a z-buffer (used depth-sorted painter's + seg clipping because memory was tight). A modern Canvas 2D renderer has the memory.

#### 3.4.2 DOOM's actual structure: solidsegs span array (not linked list)

- Correction: some sources say segs are "stored in a linked list." Sanglard's source review says it's actually `solidsegs` — a sorted array of `{start, end}` screen-x spans, against which new segs are clipped. The span approach is more memory-efficient than a per-column boolean and handles partial occlusion cleanly.
- If we keep a non-z-buffer approach for CPU tier, use `Uint8Array(width)` + parallel `Int16Array(width)` of wall `drawEnd` (not a JS `Set`).

#### 3.4.3 FOV culling before projection

- Before projecting an enemy, compute the angle from player to enemy; if outside `[dirAngle - fov/2, dirAngle + fov/2]` (with margin), skip. Mirrors DOOM's sector-linked thing collection. Avoids projecting off-screen enemies.

#### 3.4.4 Transparent walls edge case

- If transparent neon barriers (force-fields) are ever added, they must be drawn with sprites in a masked pass (after walls + flats), not as solid columns. The solid-column / z-buffer approach assumes opaque walls.

#### 3.4.5 Silhouette readability (tiered)

- **Tiered silhouette:** close range → articulated wireframe (diamond head + tapered body, or triangular "shard" — sharp angles read as hostile at small sizes); mid range → diamond/triangle alone; far range → single bright neon dot with hue-coded ring. A 6-line quad reads as a crate, not a threat.
- **2-state pose:** "idle" (slow vertical bob, dim) and "alert/aggro" (brighter, expanded outline, slight forward lean). State transition on detecting player.
- **Health indication:** hue shift along `FLAPPY_REGULAR_NEON_RAMP` (green→yellow→red) for health, keep opacity purely for distance fog. Decouples the two. Pair with shape encoding for color-blind safety (full-HP = solid diamond, wounded = diamond + inner cross, critical = diamond + pulsing ring).

### 3.5 Beam tracers and wall-impact spots

The weapon is a hitscan neon beam, not a slow projectile.

- **Tracer:** a world-space line from the gun origin to the hit point, projected to screen space in the worker tier.
  - Color: `#f0f8ff` (bright white with a cool tint).
  - Glow: `rgba(240,248,255,0.5)` with `shadowBlur = 6`.
  - Line width: 2px.
  - Duration: 80ms.
  - Distance scaling: the projected screen length and position follow perspective projection; far endpoints shrink toward the vanishing point.
- **Wall-impact spot:** a small glowing circle at the world-space wall hit point.
  - Color: `#f0f8ff`, glow `rgba(240,248,255,0.5)`, radius 4px, glow blur 6px.
  - Lifetime: 3000ms.
  - Projected with the same camera transform as sprites; scale falls with distance.

**Superseded:** slow-moving colored projectile lines (blue player, pink enemy) were replaced by the instant white beam + persistent white impact spot, which reads more clearly as a laser weapon and is cheaper to render.

### 3.6 Player weapon overlay

- Bottom-center neon line (the "gun"): a simple neon rectangle/line at the bottom of the screen. No 3D weapon model — keep it abstract and on-theme.
- **Gun bob + kickback:** the gun moves up-down and side-to-side with the walk cycle (synced to camera bob, §4.3). Kicks back on fire. Optional: spin (chaingun aesthetic). Brightens on fire (muzzle flash = brief `shadowBlur` burst). The "brightens on fire" is the minimum; bob + kickback are the satisfying-shooting additions.
- **Dash i-frames visualization:** dash grants 200ms invulnerability, visualized by a brightness flash on the player weapon overlay + a brief afterimage trail (3 fading copies at 60ms spacing). This makes dash a mechanic, not just animation.

### 3.7 CRT overlay & neon "breath" (optional synthwave finishers)

- **CRT overlay:** Pre-rendered scanline sprite (1px-on, 2px-off horizontal lines) blitted with `globalAlpha ≈ 0.08` and `globalCompositeOperation = "multiply"` over the final frame. Plus optional vignette via radial-gradient `fillRect`. One `drawImage` per frame.
- **Neon "breath" cycle:** Global neon brightness modulated by `0.85 + 0.15 * sin(t * 0.5Hz)` on the glow pass only (not the body). Slow, subtle. Reduces visual fatigue on long streams and reads as "alive."

---

## 4. Game Loop & State (DOOM-inspired, simplified)

### 4.1 Fixed-timestep simulation & interpolation (DOOM ran at 35 Hz logic, uncapped render)

DOOM separated the game world tick (35 Hz fixed) from rendering (as fast as possible). We follow the same split:

- **Simulation tick:** fixed timestep (e.g., 30 Hz or 60 Hz). All game logic (movement, collision, enemy AI, projectiles, NGE inference) runs here, deterministically.
- **Render:** `requestAnimationFrame`, as fast as possible. Reads the latest simulation state.

#### 4.1.1 Vanilla DOOM judder at 60Hz

DOOM ran logic at 35 Hz with no interpolation between gametics. On a 60 Hz display, the same state is shown ~1.7 times per render, producing visible judder. A naive "re-render the last state if no new state" approach has this problem.

#### 4.1.2 Render-side interpolation (recommended)

Keep `statePrev` and `stateCurr`; on each RAF, render `lerp(statePrev, stateCurr, alpha)` where `alpha = (now - lastTickTime) / tickDuration`. Eliminates 30→60 Hz judder. This is what modern DOOM source ports do (Crispy Doom, GZDoom). If no new state and no interpolation, re-render the last (keeps UI responsive, matches Flappy's `requestId`-gated pattern).

#### 4.1.3 ticcmd_t-style input snapshot

Pack player input per tick into a small struct `(forwardmove: int8, sidemove: int8, angleturn: int16, buttons: uint8)` — DOOM's `ticcmd_t` layout. Enables demo recording / replay (reuse `WorkerPlaybackFrameSnapshot` patterns) and is cheaper than recording full state. Directly relevant to the repo's `reproducibility-contracts` skill.

### 4.2 Player movement, collision & input

#### 4.2.1 Movement model

- Position `(posX, posY)`, direction `(dirX, dirY)`, plane `(planeX, planeY)`.
- **Move forward/back:** `pos += dir * moveSpeed`. Check the target cell in the grid; if it's a wall, don't move (or slide — check X and Y separately for wall-sliding).
- **Strafe:** `pos += plane * strafeSpeed` (perpendicular to direction).
- **Turn:** rotate both `dir` and `plane` by the rotation matrix with `rotSpeed`.

#### 4.2.2 Collision — grid-cell check (Wolf3D/Lode lineage)

- Before moving, check if `grid[floor(newX)][floor(posY)]` and `grid[floor(posX)][floor(newY)]` are walls. This gives wall-sliding for free (you slide along walls instead of stopping dead).
- **Attribution correction:** The original draft stated "DOOM cast 8 rays around the player to push the camera out of walls." This is **wrong**. DOOM used the **blockmap** (128×128-unit grid) plus `P_TryMove`: the mobj has a radius, and collision tests all linedefs in overlapping blockmap blocks via line-side tests. The 8-ray approach is carlini's ("a poor-man's sphere collision detection"). For Neatenstein's grid raycaster, the per-axis cell check is fine but should be attributed to Lode/Wolf3D lineage, not DOOM.

#### 4.2.3 Diagonal corner-pop guard

The per-axis sliding approach (check X and Y independently) has a known failure: moving diagonally into an inside corner can pop through the diagonal wall cell.

- **Fix:** after resolving X and Y independently, do a final `if (grid[floor(newX)][floor(newY)] is wall) reject both` check. Alternatively, give the player a small radius (0.2 cell) and check the 4 corner cells of the player's bounding box.

#### 4.2.4 DOOM's actual wall-sliding mechanism

DOOM does NOT do per-axis sliding. `P_TryMove` attempts the full move; if blocked, the move is rejected entirely. Sliding emerges because `forwardmove` and `sidemove` are independent inputs — pressing forward into a wall while strafing moves you along the wall because only the strafe component is unblocked. The per-axis approach is a Wolf3D/Lode technique.

#### 4.2.5 Blockmap-style spatial index (future)

For 8 enemies on a 24×24 grid, O(n²) collision is fine. If enemy count grows, a per-cell linked list of entities (DOOM's `blocklinks`) avoids O(n²).

#### 4.2.6 Pointer Lock API

- `canvas.requestPointerLock({ unadjustedMovement: true })` on first canvas click. Disables OS-level mouse acceleration for raw input. Feature-detect `unadjustedMovement` (Chrome 88+, FF 152+, Safari 18.4+); fall back to plain `requestPointerLock()`.
- **Engagement gesture requirement:** `requestPointerLock()` requires a user gesture (click). The game can _run_ from frame 1 (keyboard movement works), but mouse-look activates on first canvas click. The 5-second intro card's click doubles as the pointer-lock engagement gesture.
- **ESC exit:** ESC exits pointer lock (browser-level, unavoidable). On `pointerlockchange` → unlocked, show a "click to resume" overlay; on click, re-request lock.
- **iOS Safari:** Pointer Lock is unsupported. Touch fallback: drag-to-look (`touchstart`/`touchmove` deltas → `yaw`).

#### 4.2.7 Mouse rotation model

- In `mousemove` handler (while locked): `pendingYaw += e.movementX * SENSITIVITY` (e.g., `SENSITIVITY = 0.0022` rad/px, a common FPS value). User-adjustable multiplier.
- In the sim tick (not the event handler — keep sim deterministic): apply rotation matrix to `dir`/`plane` using `pendingYaw`; reset `pendingYaw` to 0. Cap at ±0.5 rad to prevent spin exploits.
- **`movementX` unit quirk:** Browsers use inconsistent units (physical/logical/CSS pixels). `unadjustedMovement: true` normalizes this; if cross-browser consistency is critical, fall back to manual `screenX`/`screenY` delta.

### 4.3 Game state & game feel

- `health` (0-100, or DOOM-style 0-100+), `ammo` (count), `armor` (optional, skip for v1).
- **Enemy waves:** continuous trickle (not discrete clumps) — spawn 1 enemy every N ticks, up to the 8-concurrent cap. This reads as "an ecosystem," not "level design" (per the Game Director plan).
- **Projectiles:** hitscan (instant ray-hit, like DOOM's hitscan weapons) for the player's neon beam; enemy projectiles can be slow-moving tracers for visual legibility.
- **Death:** health ≤ 0 → trigger the death feedback loop (freeze, scrub, banner, respawn). No menu.

#### 4.3.1 Camera bob (biggest game-feel gap)

- Player maintains a bob-phase variable (mod 2π) that **resets to 0 when movement stops** — so starting/stopping feels clean. Apply as a small vertical offset + optional roll oscillation synced to the walk cycle.
- This is the single biggest contributor to DOOM's distinctive movement feel. The carlini clone uses this (attributed).

#### 4.3.2 Hit feedback bundle (largest game-feel hole)

- **Hit marker:** 4-stroke neon "X" at crosshair for 80ms on enemy hit.
- **Hit-stop:** Freeze simulation for 40–60ms on kill (skip 2–3 ticks). The canonical "art of screenshake" technique (Jan Willem Nijman, GDC talk "The Art of Screenshake").
- **Screen shake:** `ctx.translate(rand()*amp, rand()*amp)` with `amp` decaying from 4px on hit, 8px on damage taken.
- **Damage flash:** Full-canvas red `fillRect` with `globalAlpha = 0.25` decaying over 200ms.
- **Enemy death burst:** 8–12 neon line fragments radiating outward with additive blend, lifetime ~300ms (exploding-cube death, carlini-attributed).
- **Screen shake on fire:** small shake on firing (carlini-attributed). Cheap game-feel addition.

#### 4.3.3 Crosshair

- 2px neon center dot + 4 6px ticks in `FLAPPY_NEON_PALETTE.pipeEdgeInner` (`#bfffd4`), drawn on the UI canvas layer. No FPS feels right without one.

#### 4.3.4 Enemy AI structure (carlini-attributed, for baseline behavior)

- Pace → wake on LOS or ally-shot (alert propagation) → beeline → random-direction on obstacle → shoot occasionally.
- Alert propagation on ally death prevents corner-camping. This is the baseline AI; NGE evolves over it.

### 4.4 No intro screen

Per the user's instruction: go straight to game. The 5-second intro card (from the plan) is a transient overlay, not a menu — it fades and never blocks input. The game is running behind it from frame 1. The intro card's click doubles as the pointer-lock engagement gesture (§4.2.6).

---

## 5. The Map (single arena, reused across modes)

Per the user's instruction: use the same map for all modes. This simplifies the demo and makes cross-mode state sharing meaningful (enemies trained on the same geometry).

### 5.1 Map design

- **Size:** 24×24 grid (576 cells). Locked — 32×32 is too sparse for 8 enemies in 15–25s episodes (too much wandering, not enough combat).
- **Sightline constraint:** maximum straight-line sightline ≤ 12 cells. Place pillars/wall stubs in long corridors to break sightlines and force mid-range encounters where silhouettes read (diamond/triangle LOD, not the dot LOD).
- **Structure:** a central arena with surrounding corridors and a few interior rooms/pillars. Provides cover, sightlines, and flanking routes — the spatial variety the NGE agents need to develop interesting behavior.
- **Generation:** seeded procedural (recursive backtracker + extra connections, like the PredatorPrey maze) OR hand-authored. Seeded procedural is better for the demo (different seeds = different arenas, same code). Reuse the PredatorPrey quarter-symmetric generation if symmetry is desired, or a simpler random-walk for organic layouts.
- **Wall cells:** wall cells are positive values in the grid; the renderer applies the fixed two-tone side-shading palette (§3.2). Optional per-cell light levels can still modulate brightness.

### 5.2 Map transfer

- Grid is a `Uint8Array` transferred to the worker once at episode start (not per frame).
- On mode switch (within the same enemy family), the grid is NOT re-sent — only the enemy state changes. On RESET, a new seed generates a new grid.

---

## 6. Performance Budget (tier-aware)

| Tier           | Columns | Enemies | Glow                                                  | Target fps       |
| -------------- | ------- | ------- | ----------------------------------------------------- | ---------------- |
| GPU 2048/2048  | 320     | 8       | full shadowBlur                                       | 60               |
| Worker 256/256 | 240     | 8       | limited shadowBlur / `ctx.filter` drop-shadow         | 60               |
| CPU 128/128    | 160     | 8       | no shadowBlur, no glow sprites, no texture modulation | 60 (fallback 30) |

- Column count is the primary render cost lever (DDA per column). 320 columns at 60fps is trivial on any modern CPU; the bottleneck is `shadowBlur`, not DDA.
- Enemy count is capped at 8 across all tiers (legibility constraint, not performance).
- `shadowBlur` is the expensive operation; tier-gate it aggressively. CPU tier uses the ImageData framebuffer path (§3.2); Worker tier draws fogged vertical stripes through `CanvasRenderingContext2D` (§3.2); GPU tier keeps `shadowBlur` for beam/impact effects (§3.5).

---

## 7. What We Do NOT Take from the carlini Clone

To be explicit about the boundary (license safety + design clarity):

- **No WebGL.** The carlini clone uses WebGL with shadow maps, GLSL shaders, and 3D polygon rendering. We use canvas 2D with DDA raycasting. Completely different rendering path.
- **No turtle-graphics map format.** The carlini clone compresses maps as turtle commands. We use a plain `Uint8Array` grid.
- **No polygon sectors with variable floor/ceiling heights.** We use a flat-grid (Lineage B). Variable heights are a Lineage A feature we explicitly skip.
- **No shadow mapping.** The carlini clone computes shadow maps for lights. We use distance fog + side shading for depth cues.
- **No code copied.** All algorithms are reimplemented in original TypeScript from the paraphrased descriptions above.

What we _do_ take from the carlini clone conceptually (game-feel, not code):

- The _idea_ of light levels per region (we express this as per-cell brightness variation, not hue).
- The _idea_ of billboarded sprites clipped against walls (we use wireframe sprites, not textured billboards, but the depth-sort + clip approach is the same).
- The _idea_ of a muzzle flash light (we express this as a brief `shadowBlur` burst, not a dynamic light source).
- Game-feel techniques: camera bob, gun bob, hit-stop, screen shake, exploding-cube death, alert-propagation enemy AI (all attributed to carlini's design notes; no code reproduced).

---

## 8. Reuse from Flappy Bird / Existing Repo

| Flappy pattern                                              | Neatenstein reuse                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FLAPPY_NEON_PALETTE`                                       | Extended as `NEATENSTEIN_PALETTE` (enemy hues, horizon, beam/impact colors). The fixed wall side-shading palette in §3.2 is independent of per-wall hue variants.                                                                                                                                                                                                                                      |
| **Flappy ground grid** (`playback/background/ground-grid/`) | **Inspiration only.** The implemented floor uses a new world-space grid-line projection (`renderer/floor.ts`) rather than the Flappy fake-perspective helpers. Only the palette color `groundGridLine` and the yellow pulse accent color `groundGridPulseFill` were reused visually. Pulse timing is local to Neatenstein (`NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 2000`, lifetime `2700`). See §3.3. |
| `FLAPPY_GROUND_GRID_*` constants                            | **Not reused.** Neatenstein defines its own floor constants (`NEATENSTEIN_FLOOR_*`) and pulse timing (`NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 2000`, lifetime `2700`). The projection math is local to `renderer/floor.ts`. See §3.3.                                                                                                                                                                 |
| `WorkerPlaybackFrameSnapshot` SoA + transfer list           | `NeatensteinRenderFrame` (player, enemies, projectiles as typed arrays)                                                                                                                                                                                                                                                                                                                                |
| `resolveWorkerPlaybackSnapshotTransferList`                 | `resolveNeatensteinSnapshotTransferList`                                                                                                                                                                                                                                                                                                                                                               |
| WeakMap buffer pool                                         | Snapshot buffer reuse to avoid per-frame allocation                                                                                                                                                                                                                                                                                                                                                    |
| `requestId`-gated playback step                             | `request-render-step` / `render-step` protocol                                                                                                                                                                                                                                                                                                                                                         |
| Fixed-timestep RAF loop                                     | Same loop structure, extended for FPS controls + interpolation                                                                                                                                                                                                                                                                                                                                         |
| `autoEnableAcceleration` + `AccelerationStatus`             | Startup tier detection + chip label                                                                                                                                                                                                                                                                                                                                                                    |
| Racing `resolveAccelerationChipPresentation`                | Extended additively with `batchParallelCount`                                                                                                                                                                                                                                                                                                                                                          |
| Two-region layout + `ResizeObserver`                        | Canvas + sidebar, responsive                                                                                                                                                                                                                                                                                                                                                                           |

---

## 9. Audio — WebAudio Procedural Synthesis

### 9.1 Synthesis approach

- WebAudio oscillator-based synthesis (zero asset weight, fits "neon = synthetic" aesthetic). No sampled audio.
- Graph: `OscillatorNode` → `BiquadFilterNode` → `GainNode` (ADSR envelope) → `DynamicsCompressorNode` → `destination`.

### 9.2 Sound inventory (6 sounds)

1. **Fire:** square wave, 220→110 Hz sweep, 80ms, lowpass 2kHz.
2. **Enemy hit:** sawtooth, 440→880 Hz rise, 120ms, `WaveShaperNode` distortion for grit.
3. **Player damage:** triangle, 110→55 Hz fall, 200ms, + noise burst via `AudioBufferSourceNode` (200ms white noise through bandpass).
4. **Dash:** noise sweep through sweeping bandpass 1kHz→4kHz, 150ms (whoosh).
5. **Kill:** pitched noise burst with downward pitch glide + short reverb tail via `ConvolverNode` (generated impulse).
6. **Generation-up:** rising arpeggio (330→660→990 Hz, 200ms, sine + triangle layer). The audio signal of learning — fires on the generation counter pulse and the "THEY LEARNED FROM THAT" banner. Intentionally contrasted with kill (downward = death; generation-up = upward = evolution).

### 9.3 Positional audio (DOOM DMX-inspired)

- `StereoPannerNode` (pan = relative angle of source to player direction) + `GainNode` (gain ∝ 1/distance) for each enemy sound. Cheap, browser-native, mirrors DOOM's positional audio.

### 9.4 AudioContext lifecycle

- `AudioContext.resume()` on the first click **regardless of mode** (AI modes need audio too — a viewer spectating Mode 1 who never clicks gets silence). In human modes (3/4), the same click requests pointer lock. In AI modes (1/2), no pointer lock is needed.
- **AudioContext is main-thread only** (not available in workers). Audio trigger events (fire/hit/damage/dash/kill/generation-up) originate in the display worker and are `postMessage`d to the main thread, which owns the AudioContext and performs synthesis. Add audio-trigger events to the worker→host message protocol.
- Master `GainNode` + `DynamicsCompressorNode` at chain end to prevent clipping when multiple sounds overlap.

### 9.5 Audio tiering

- CPU tier → oscillator-only (no `ConvolverNode` reverb, no `WaveShaper`). GPU/Worker tier → full graph.
- Optional ambient drone: two detuned `OscillatorNode`s (55 Hz + 55.5 Hz sine) through a slow LFO on a lowpass filter — the synthwave "background hum." ~2 nodes, zero assets.

---

## 10. Stream Legibility & Color Safety

### 10.1 WCAG contrast verification

- Target: ≥ 4.5:1 for HUD text, ≥ 3:1 for gameplay-critical elements (enemy hues, projectiles, crosshair) against `#060b14`.
- `FLAPPY_NEON_PALETTE.hudText` `#9fdcff` on `#060b14` — ~9.8:1, fine.
- `horizonLine` `#0a8ea0` on `#060b14` — likely **below 3:1**. Brighten to `#1ac4d8` or reuse `groundGridPulseFill` `#fff14a` for the horizon.
- Verify all palette colors before implementation; flag any below-threshold.

### 10.2 Protanopia / color-blind safety

- WCAG 1.4.3 flags "predominantly long wavelength colors against darker colors" as problematic for protanopia. Avoid pure red `#ff1a1a`/`#ff3300` for critical state on black; shift the ramp's low end toward `#ff6a00`/`#ffaa00` (orange) which maintains luminance contrast for protan viewers.
- Never rely on color alone for state: pair hue with shape/pattern encoding (§3.4.5).

### 10.3 Stream compression survival

- Minimum stroke width **2px** for any gameplay-critical neon line (enemy outlines, projectiles, crosshair); 1px only for decorative grid. 1080p H.264 crushes 1px neon lines into shimmer.

### 10.4 Neon fatigue mitigation

- The §3.7 "breath" cycle reduces sustained high-saturation fatigue. Consider a dim "rest" state when no enemies are visible for > 5 seconds.

---

## 11. Resolved and remaining questions

1. **Floor rendering — RESOLVED:** world-space grid-line projection (§3.3.1) superseded both the fake-perspective Flappy grid and the per-pixel floor-caster idea.
2. **Pulse anchoring — RESOLVED:** world-space grid-line dots (§3.3.2) superseded the bearing-tolerance, fake-perspective pulse design.
3. **Wall colors — RESOLVED:** fixed two-tone cyan/blue side-shading (§3.2) superseded hue-varied / pink walls.
4. **Y-shearing (fake look up/down):** The plan specifies mouse look. True 3D look up/down is impossible in raycasting (walls are vertical). Y-shearing (moving the horizon line) fakes it but distorts. Recommend: horizontal mouse look only (rotate), no vertical look. The dash mechanic provides the "dodge" feel without vertical aim.
5. **Minimap:** DOOM had a map mode. A small top-down minimap in the corner (neon grid + player dot + enemy dots) would help spatial awareness and is cheap to draw. Recommend as a future UI element, toggleable with `M`.
6. **Sprite rendering method:** Direct line drawing for the neon aesthetic; the wireframe is only a few lines per enemy.

---

## 12. NGE Integration Surface (Round 2 gap — NGE integration specialist)

The renderer and game-feel sections above describe the _engine_. This section describes the _seam_ between the engine and the NGE evolution layer — the contracts Phase 4–6 depend on.

### 12.1 Sensory encoding (NGE input vector)

The renderer's DDA rays (§2.2–2.4, 160–320 per frame) are _rendering_ rays — their output is `perpWallDist` + `side` for wall-height drawing. The NGE agent needs a separate, smaller set of **sensory rays** (~8–16 angular samples) reporting per-ray:

- `wallDist` — distance to nearest wall along this bearing (reuse DDA traversal, early-out on wall hit).
- `enemyVisible` — boolean; true if an enemy is visible along this bearing before the wall hit (check via z-buffer or FOV-cull angle test, §3.4.1/§3.4.3).
- `enemyDist` — distance to the nearest visible enemy along this bearing (0 if none).
- `enemyType` — enum (grunt/swarmer/MLP/cohort-member).
- `projectileVisible` + `projectileDist` — same pattern for incoming projectiles.

Sensory rays share the DDA grid traversal with rendering rays but are a separate cast (fewer rays, richer output per ray). Non-ray inputs: `health` (normalized 0–1), `ammo` (normalized), `dashCooldown` (0–1), `enemyCount` (normalized), `lastKnownEnemyBearings[]` (from EpisodicSlot memory).

### 12.2 Action decoding (ticcmd_t as the shared input interface)

The `ticcmd_t` struct (§4.1.3: `forwardmove, sidemove, angleturn, buttons`) is the **single input interface for both human and NGE modes**:

- **Human mode:** pointer lock → `pendingYaw` accumulation in `mousemove` → applied as `angleturn` in the sim tick. `buttons` bits set by key/click events.
- **AI mode:** NGE output vector maps directly to ticcmd fields — no pointer lock, no `pendingYaw`. The network produces `(forwardmove, sidemove, angleturn, buttons)` directly.
- **`buttons` bit layout:** bit 0 = fire, bit 1 = dash. Activation: sigmoid output > 0.5 → bit set.
- **6–7 output indices:** `[0] forwardmove, [1] sidemove, [2] angleturn, [3] fire, [4] dash, [5] (reserved: weapon switch, unused in v1), [6] (reserved: use, unused in v1)`.

### 12.3 Replay buffer format (ticcmd-level, partial replay)

The replay buffer stores **`ticcmd_t[]`** (human input stream) + episode seed, NOT world-state snapshots. Re-evaluation re-simulates from `(seed, human ticcmd stream, variant network)` — deterministic.

- **Partial replay:** the human's ticcmds are replayed verbatim (frozen) while enemy AI is re-inferred per variant. This is the contract Phase 6's determinism check depends on.
- This is the DOOM demo-recording approach (seed + input stream), adapted for partial replay (one side frozen, other side varies).

### 12.4 Episode lifecycle

An **episode** = one life, 15–25s, bounded by the fixed timestep (§4.1) for determinism.

- **Episode start:** stamp seed, spawn player at known position, clear enemies, zero fitness accumulators, begin ticcmd recording.
- **Episode end:** death OR timeout → compute fitness composite (`CombatQualitySignal`), freeze replay buffer, emit fitness scalar.
- **Episode reset:** restore world to seeded initial state for the next variant. This is distinct from map RESET (§5.2, which generates a new grid). Episode reset reuses the same grid + seed.

### 12.5 Worker topology (three distinct roles)

The research file's "worker offload" framing implies one worker does everything. The plan's fitness evaluation requires three distinct roles:

1. **Render worker** (OffscreenCanvas, DDA + blit) — §3.2. Worker tier only.
2. **Authoritative world worker** (game state, sim, NGE inference for the display agent) — plan Phase 3. On Worker tier, this is the same worker as the render worker. On CPU/GPU tiers, this worker produces `NeatensteinRenderFrame` for the main thread to render.
3. **Stateless batch workers** (variant episode evaluation, no canvas, no persistent state) — plan Phase 3. A separate pool. Never touch a canvas. Run `(seed, ticcmd stream, variant network)` → fitness scalar.

### 12.6 Fitness telemetry hooks

Per-tick telemetry accumulators (read at episode end for `CombatQualitySignal`):

- `damageDealt` — sum of damage dealt by the agent this episode.
- `kills` — count of enemy kills.
- `survivalTicks` — ticks alive before death/timeout.
- `damageTaken` — sum of damage received.
- `aimMissRate` — (shots fired − shots hit) / shots fired.
  These are accumulated in the world worker's sim tick and read at episode end.

### 12.7 NGE inference cadence vs sim cadence

The display sim runs at 30/60 Hz. NGE inference for the _display_ agent runs every tick (one network, cheap). NGE _batch_ evaluation (2048 variants) runs on the batch worker pool, decoupled from the display sim — it evaluates full episodes (15–25s sim each) in parallel, not per-tick. The display agent's inference and the batch evaluation are independent workloads.

### 12.8 Enemy AI source by mode

The enemy's ticcmd source depends on mode:

- **ARMS RACE:** MLP network (fixed topology, weight-only co-evolution).
- **SWARM:** WeightSharedCohort network (shared weights, coordinate injection).
- **Baseline/test (static modes):** scripted AI (§4.3.4: pace → wake → beeline → random-on-block).
  The handoff from scripted AI to NGE-controlled AI is a mode switch — the ticcmd source changes, the ticcmd format does not.

---

## 13. Player Death Feedback (Round 2 gap — game director)

### 13.1 Freeze-frame visual treatment

- On player death: desaturate the screen to grayscale + preserve neon edges (neon stays, fill goes gray). Hold 400ms. This is the "time stops" signal.

### 13.2 Scrub render style

- Replay the last 10s at 4× speed with a neon time-scrub bar at the screen bottom (a horizontal neon line with a moving cursor). The lethal frame is marked with a vertical neon line on the scrub bar + brief slow-mo (200ms) on the lethal moment. The enemy that killed the player gets a neon highlight ring.

### 13.3 "THEY LEARNED FROM THAT" banner

- Slide-up + additive glow entrance, 1s hold, fade. Paired with the generation-up sound (§9.2 sound #6).

### 13.4 Before/after visual bridge (the "proof" moment)

- During the 1s banner hold, show a 300ms split-second flash: the enemy that killed you is highlighted with an "ascended" visual treatment (brighter outline + pulsing ring) when you respawn, so the viewer can track "that's the one that learned." If feasible, a brief predictive ghost (enemy now strafing instead of standing) — but the ascended-highlight is the minimum viable proof.

### 13.5 Cold-start spawn rule

- First 4 enemies spawn within 8 cells of the player/camera on episode 1; subsequent episodes use normal distribution. Guarantees action within the first 5–8 seconds (prevents the "empty corridor" streamer turn-off).

---

## 14. Generation Visual Lineage (Round 2 gap — game director)

### 14.1 Shape mutation by generation threshold

The §3.4.5 tiered silhouette is a _distance_ LOD. Generation signal needs a second channel (hue alone washes out under stream compression):

- Gen 1: bare diamond.
- Gen 10+: gains an inner cross-brace.
- Gen 25+: gains a pulsing ring.
- Gen 50+: gains angular spikes.

This pairs with the existing health shape-encoding (§3.4.5) and gives stream-compression-survival a second channel. A viewer can distinguish "these enemies are visibly older/more evolved than the ones from 30 seconds ago."

### 14.2 Map size locked to 24×24

- 32×32 is too sparse for 8 enemies in 15–25s episodes (too much wandering). Lock to 24×24 (576 cells).
- **Sightline constraint:** maximum straight-line sightline ≤ 12 cells. Place pillars/wall stubs in long corridors to break sightlines and force mid-range encounters where silhouettes read (diamond/triangle LOD, not the dot LOD).

---

## 15. References

- **DOOM engine (Wikipedia):** https://en.wikipedia.org/wiki/Doom_engine — overview of BSP, sectors, visplanes, sprites. CC-BY-SA 4.0 (paraphrased).
- **Fabien Sanglard, _Game Engine Black Book: DOOM_ / Doom Classic Renderer code review:** http://fabiensanglard.net/doomIphone/doomClassicRenderer.php — detailed engine breakdown including `solidsegs` span array, visplane merging, `R_MakeSpans`. Referenced for algorithm understanding; no text reproduced.
- **Doom Wiki — Doom rendering engine:** https://doomwiki.org/wiki/Doom_rendering_engine — segs, vissprites, visplanes, blockmap, COLORMAP, fake contrast. Referenced; no text reproduced.
- **Doom Wiki — Tic:** https://doomwiki.org/wiki/Tic — `realtic` vs `gametic`, `ticcmd_t` struct, no-interpolation vanilla behavior. Referenced; no text reproduced.
- **Doom Wiki — Blockmap:** https://doomwiki.org/wiki/Blockmap — blockmap + radius collision, blocklinks. Referenced; no text reproduced.
- **Doom Wiki — Sound:** https://doomwiki.org/wiki/Sound — DMX sound system, 8-bit mono 11025 Hz, stereo panning + distance attenuation. Referenced; no text reproduced.
- **Lode Vandevenne, Raycasting Tutorial (Part 1):** https://lodev.org/cgtutor/raycasting.html — the canonical DDA raycasting reference. Copyright 2004-2020 Lode Vandevenne. DDA algorithm paraphrased in original words above; no code or text reproduced.
- **Lode Vandevenne, Raycasting Tutorial (Part 2):** https://lodev.org/cgtutor/raycasting2.html — floor/ceiling casting, scanline-coherent floor interpolation, sprites. Referenced; no code reproduced.
- **carlini/js13k2019-yet-another-doom-clone:** https://github.com/carlini/js13k2019-yet-another-doom-clone — GPL v3. Referenced for game-feel ideas (camera bob, gun bob, enemy AI, procedural audio, exploding-cube death, 8-ray collision). No code reproduced. Rendering path is different (WebGL polygons vs. canvas 2D DDA).
- **carlini writeup:** https://nicholas.carlini.com/writing/2019/javascript-doom-clone-13k.html — author's design notes (8-ray collision, camera bob, JSFXR audio, enemy AI). Referenced; no text reproduced.
- **DOOM source code (id Software):** https://github.com/id-Software/DOOM — GPL v2-or-later. Referenced for algorithm understanding; no code reproduced.
- **MDN — Optimizing canvas:** https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas — `alpha: false`, pre-render to offscreen, batch calls, avoid `shadowBlur`, layered canvases, avoid sub-pixel coords. Referenced; no text reproduced.
- **MDN — OffscreenCanvas:** https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas — worker-side rendering via `transferControlToOffscreen()`. Referenced; no text reproduced.
- **MDN — CanvasRenderingContext2D.filter:** https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter — `drop-shadow()` and `blur()` as GPU-composited glow alternatives. Referenced; no text reproduced.
- **MDN — Pointer Lock API:** https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API — `requestPointerLock({ unadjustedMovement: true })`, engagement gesture, `pointerlockchange`/`pointerlockerror`, iOS Safari non-support. Referenced; no text reproduced.
- **MDN — MouseEvent.movementX:** https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/movementX — movement deltas, unit-inconsistency browser quirk. Referenced; no text reproduced.
- **MDN — Web Audio API:** https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API — `OscillatorNode`, `BiquadFilterNode`, `GainNode`, `DynamicsCompressorNode`, `StereoPannerNode`, `ConvolverNode`. Referenced; no text reproduced.
- **MDN — Web Audio best practices:** https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices — `AudioContext.resume()` on user gesture (autoplay policy). Referenced; no text reproduced.
- **W3C WCAG 2.1 §1.4.3 Contrast (Minimum):** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum — 4.5:1 body text, 3:1 large text, red-on-black protanopia advisory. Referenced; no text reproduced.
- **Jan Willem Nijman, "The Art of Screenshake" (GDC):** — hit-stop and screen-shake technique referenced for §4.3.2. Talk, no text reproduced.
- **Flappy Bird example (this repo):** `examples/flappy_bird/` — palette, frame snapshot, worker channel, and RAF loop patterns reused. Internal, no external license.

---
