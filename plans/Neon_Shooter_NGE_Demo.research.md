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
- **Floors/ceilings:** Solid color (or simple floor-casting texture projection in advanced versions). Neatenstein uses the Flappy ground grid adaptation (§3.3).
- **Sprites:** Billboarded, depth-sorted, drawn after walls.
- **Look up/down:** Not supported (vertical-wall limitation). Horizontal mouse look only (§4.2.6).
- **Strengths:** Trivial to implement, very fast, perfect for canvas 2D, no BSP precomputation.
- **Weaknesses:** No variable heights, no stairs, no rooms-over-rooms, grid-aligned walls only. None of these matter for the Neatenstein demo.

### Conceptual borrowings from DOOM (without its engine)

- Per-region light levels → we vary neon brightness per map region, encoded as a per-cell "light level" in the grid.
- Billboarded sprites clipped against walls → our enemy wireframes use the same depth-sort + clip approach.
- Top-down "map mode" → useful as a debug overlay and potentially as the network-view's spatial context.

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
  - `1..N` = wall variant (neon hue index into `NEATENSTEIN_PALETTE.wallHues`).
- Optional: a parallel `Uint8Array` of "light levels" per cell (0-15, DOOM-style) to vary neon brightness per region. This is a cheap way to get visual variety without variable floor heights.
- The grid is transferred to the worker once at episode start; the renderer caches it locally.

### 3.2 Neon wall rendering (replaces texture sampling)

Instead of sampling texels, each wall column is a **neon vertical line**:

1. Base color: `wallHues[wallVariant]` (from the neon palette, extending `FLAPPY_NEON_PALETTE`).
2. Distance fog: lerp the base color toward `#060b14` (background, see §3.3) by `1 - perpWallDist / maxViewDist`. Closer walls are brighter; far walls fade to the background.
3. **Borders (the "neon glow on borders"):** stroke a 1px brighter neon line (`wallEdgeInner`) at `drawStart` and `drawEnd` (top and bottom of the wall slice). This is the signature neon look — the wall body is dim, the edges glow.
4. **Neon grid wall pattern (the synthwave standard):** vertical mullion lines at every grid-cell boundary projected to screen X (compute screen X of each cell corner via camera transform, stroke a full-height neon line there) + horizontal scanlines at `spacing = clamp(8 * perpWallDist, 4, 64) px`. This is the "TRON grid wall" — the top/bottom borders alone only give edges, not the characteristic vertical mullions. This is a brightness modulation, NOT a sampled texture. Keeps the "neon lines, not pixels" aesthetic.
5. **Continuous horizon glow:** after all wall columns are drawn, stroke a single continuous `ctx.beginPath()` path along the wall-top pixels with glow + additive blend. The per-column approach produces a dotted/dashed horizon at distance.
6. **Side shading:** Y-side walls use Lode's `/2` (50% brightness) — divide RGB by 2. The originally-suggested 30% is too subtle for neon-on-black; 50% is the canonical Wolf3D/Lode value and reads clearly.
7. **COLORMAP-style LUT:** precompute a `Uint32Array` of neon colors indexed by `(distanceBucket << 1 | side)` and look up per column. Avoids per-column `lerp` math; mirrors DOOM's COLORMAP.

#### 3.2.1 Render paths (tier-gated)

The per-column `fillRect`/`stroke` approach is the slow path. Three tier-gated render paths:

- **CPU tier — ImageData framebuffer + single putImageData:** Write all wall pixels directly into a `Uint8ClampedArray` backing an `ImageData` (RGBA per pixel), flush once per frame with `putImageData`. Eliminates per-column `fillRect` overhead entirely. This is Lode's recommended pattern ("use a 2D array as screen buffer, copy to screen at once"). `drawStart`/`drawEnd` and all coordinates must be `Math.floor`-ed or `| 0` to avoid sub-pixel anti-aliasing cost (MDN: "Avoid floating-point coordinates"). Glow on CPU tier: pre-rendered glow sprites (one per hue × distance bucket) blitted via `drawImage` (GPU-accelerated, far cheaper than `shadowBlur` per stroke). MDN: "Pre-render similar primitives on an offscreen canvas."
- **Worker tier — OffscreenCanvas:** `canvas.transferControlToOffscreen()` → pass to worker. Entire DDA + blit runs off the main thread. MDN: "Rendering operations can also be run inside a worker context." Glow on Worker tier: `ctx.filter = "drop-shadow(0 0 4px <hue>)"` with feature detection (Safari disables `filter` — fall back to pre-rendered sprites).
- **GPU tier — stroke + shadowBlur (premium path):** Keep the stroke + `shadowBlur` path. `shadowBlur` is the most expensive option (MDN explicitly lists it as an anti-pattern) but gives the best glow quality. GPU tier can afford it.

#### 3.2.2 Universal optimizations

- `getContext("2d", { alpha: false })` — free perf for opaque neon-on-black (MDN).
- Batched polylines / color buckets: group columns by final packed color before issuing `fillStyle`/`strokeStyle` to minimize state changes. The `ImageData` path has no state at all.
- `fillRect(0,0,w,h)` with `alpha:false` for clearing (faster than `clearRect` on opaque contexts).

#### 3.2.3 Glow strategies (tier-gated)

`shadowBlur` is the documented anti-pattern. Four ranked alternatives, tier-gated:

1. **Pre-rendered glow sprites (CPU tier):** Pre-render each wall-hue glow as a soft radial-gradient `ImageData` sprite (once, at load). Composite via `drawImage` per column-edge. GPU-accelerated, far cheaper than `shadowBlur`.
2. **CSS `filter: blur()` / `drop-shadow()` on a separate glow canvas layer (Worker/GPU tier):** Draw only neon edge lines (no blur) onto a second `<canvas>` overlaid via CSS `position:absolute`. Apply `filter: blur(4px)` + `opacity` via CSS. Browser GPU-accelerates the blur. Feature-detect `ctx.filter` (Safari may disable).
3. **Radial gradients (per-edge, no blur):** Small `createRadialGradient` at each wall-edge endpoint. Cheaper than `shadowBlur`, gives localized glow "nodes." Doesn't glow along the whole line.
4. **`shadowBlur` (GPU tier only):** Keep as premium path. Most expensive, best quality.

**Layered canvas architecture (MDN-recommended):**

- Layer 1 (static, drawn once on resize): background `#060b14`, horizon line, floor grid, ceiling.
- Layer 2 (dynamic, per-frame): walls, enemies, projectiles, glow.
- Layer 3 (UI, per-frame or on-event): HUD, crosshair, mode dial, stats.
  This avoids re-drawing the static background every frame.

**Additive blending:** Use `globalCompositeOperation = "lighter"` for glow/border strokes and projectile tracers so overlapping neon saturates toward white (TRON/synthwave standard). Reset to `"source-over"` for wall body fill.

### 3.3 Floor and ceiling — Flappy Bird ground grid reuse

- Solid `#060b14` background (matches `FLAPPY_NEON_PALETTE.background`). Use `#060b14` consistently, NOT `#000` — keeps palette continuity with Flappy and gives neon edges a slightly warmer blend target.
- A faint neon **horizon line** at the vanishing point (reuse `FLAPPY_NEON_PALETTE.horizonLine` + `horizonGlow`).

**The floor reuses Flappy Bird's synthwave ground grid** (`examples/flappy_bird/browser-entry/playback/background/ground-grid/`) for visual coherence across the library. Both demos share the same neon aesthetic; the floor grid is the strongest style cue and should be familiar to anyone who has seen the Flappy demo.

#### 3.3.1 What Flappy's ground grid is

Flappy's ground grid is a **synthwave forced-perspective grid** — the classic TRON-style floor:

- **Horizontal depth bands:** 16 lines (`FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT = 16`) spaced by a power curve (`FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT = 2.35`) so they bunch up near the horizon and spread out near the viewer. This is the "receding into distance" effect.
- **Vertical perspective rays:** lines converging to a centered vanishing point at the horizon. They wrap/scroll with parallax to imply forward motion.
- **Depth-graded styling per line:** alpha (0.12 far → 0.58 near), blur (6px far → 0px near — inverse, far lines are _more_ blurred for atmospheric fog), thickness (1px far → 3px near). All driven by `resolvePlaybackGroundGridDepthCurve(depthRatio)`.
- **Colors:** `groundGridLine: '#0a8ea0'` (teal), `groundGridFog: 'rgba(10, 142, 160, 0.55)'`, `groundGridPulseFill: '#fff14a'` (yellow pulse accents).
- **Pulse system:** occasional yellow squares (`#fff14a`) that travel along grid lines every 6s, lifetime 5.9s — "occasional accent lights rather than a constant distraction."
- **Layered composition:** sky (starfield) → ground grid → horizon seam (glowing divider line). The horizon is both a crisp divider and a glow source.

#### 3.3.2 Neatenstein adaptation (camera-rotated grid)

The key difference: Flappy is a **side-scroller** (2D camera, grid scrolls horizontally). Neatenstein is a **first-person raycaster** (camera rotates, grid is the floor below the horizon). The grid geometry changes with camera rotation, not just scroll offset. But the _visual style_ — depth-curved horizontal bands, converging vertical rays, depth-graded alpha/blur/thickness, pulse accents, horizon seam — transfers directly.

| Flappy                                                 | Neatenstein adaptation                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Horizontal bands at fixed depth curve                  | Same 16 bands, same `DEPTH_CURVE_EXPONENT = 2.35`, same power-curve spacing                                                                                                                                                                                                                                                                                                                              |
| Vertical rays converge to fixed vanishing point        | Vertical rays converge to **camera yaw-rotated** vanishing point (rotates with mouse look)                                                                                                                                                                                                                                                                                                               |
| Rays scroll with `scrollBasePx` (horizontal parallax)  | Rays shift with **player position** — the grid cells under the player move. Cheap version: offset the vanishing point by player movement; true version: floor-cast grid lines that match wall DDA columns (Phase 8 stretch)                                                                                                                                                                              |
| `groundGridLine: '#0a8ea0'` (teal)                     | **Same color** — reuse `FLAPPY_NEON_PALETTE.groundGridLine` for coherence                                                                                                                                                                                                                                                                                                                                |
| `groundGridFog` atmospheric fog                        | Same fog, same depth-graded alpha                                                                                                                                                                                                                                                                                                                                                                        |
| Pulse squares every 6s (screen-space, scroll-anchored) | **Fake-perspective-anchored pulses** — reuse ambient color (`#fff14a`), depth-graded sprite styling. **Adapt:** interval 6000→3000ms, lifetime 5900→2700ms (90% ratio, §3.3.6); emission driver to sim tick (§3.3.7); vertical-pulse continuity to world-bearing match (§3.3.5); + event pulses (§3.3.8). Pulses render on Layer 2 (dynamic), depth-tested against z-buffer (§3.4.1). See §3.3.5–§3.3.9. |
| Horizon seam (glowing divider)                         | **Same horizon seam** — this is the line where walls meet floor. Reuse `horizonLine` + `horizonGlow`                                                                                                                                                                                                                                                                                                     |
| Static background layer (drawn once on resize)         | **Same** — the floor grid is a static-layer canvas (§3.2.3 layered architecture), redrawn only on resize or camera yaw change                                                                                                                                                                                                                                                                            |

**Reuse the geometry helpers directly:** `resolvePlaybackGroundGridDepthCurve`, `resolvePlaybackGroundGridLineAlpha`, `resolvePlaybackGroundGridLineBlur`, `resolvePlaybackGroundGridLineThickness` — all depth-ratio-driven, camera-agnostic. They work unchanged. Only the vertical-ray vanishing point needs camera yaw rotation.

**Reuse the palette:** `groundGridLine`, `groundGridFog`, `groundGridPulseFill`, `horizonLine`, `horizonGlow` — all from `FLAPPY_NEON_PALETTE`. This gives instant visual coherence with Flappy Bird.

**Reuse the pulse system:** ambient pulses reuse `groundGridPulseFill = '#fff14a'` (color) and the depth-graded sprite styling unchanged. The interval and lifetime are **adapted** (not reused): `NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 3000`, `NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS = 2700` (90% ratio, vs Flappy's 6000/5900ms which is too sparse for 15–25s episodes). See §3.3.6.

**The caveat (a plus, not a minus):** Flappy's grid is a _fake perspective_ (2D forced-perspective, not true 3D). In a raycaster, the floor is genuinely below the camera and the grid should ideally be floor-cast (true perspective from the raycaster's per-column geometry). But Flappy's fake-perspective grid is the perfect cheap substitute — it _looks_ like a 3D grid floor without the per-pixel floor-casting cost. The vertical rays won't perfectly align with wall columns, but at the neon aesthetic's level of abstraction (glowing lines on black), the slight misalignment reads as atmospheric, not wrong. If perfect alignment is wanted later (Phase 8 stretch), the vertical rays can be replaced with true floor-cast grid lines that match the wall DDA columns.

#### 3.3.3 Ceiling

- Ceiling mirrors the floor grid with a cooler hue (or the same grid inverted above the horizon). Alternatively, solid `#060b14` with just the horizon glow — the ceiling is less visible in a raycaster (walls fill most of the upper screen) and can be simpler than the floor.
- Recommend: ceiling = solid `#060b14` + horizon glow for v1; ceiling grid as a Phase 8 stretch goal if the arena feels too empty overhead.

#### 3.3.4 Scanline-coherent floor-casting (Phase 8 stretch)

- If true floor-casting is added later (replacing the fake-perspective grid), do it per-_scanline_ (not per-pixel): all pixels in a floor scanline share the same depth row, only X interpolation changes. Cite Lode Part 2 (`raycasting2.html`).
- If floor-casting is added, borrow DOOM's visplane merging: merge runs of identical `(floorHeight, hue, lightLevel)` before rasterizing to keep span count bounded.

#### 3.3.5 Pulse system adaptation (fake-perspective-anchored, sim-tick-driven)

Flappy's pulses are **screen-space**: they travel along grid lines drawn in screen coordinates, tied to `scrollBasePx`, with a fixed vanishing point. Neatenstein's camera yaws, so a screen-space pulse would swim against the world when the player turns.

**Anchor reconciliation:** The v1 floor is a fake-perspective grid (§3.3.2) whose vertical rays converge to a camera-yaw-rotated vanishing point. Pulses are anchored to this **same fake-perspective space** — they travel along the fake-perspective grid lines (depth bands and yaw-rotated rays), NOT world-space grid cells. This keeps pulses consistent with the v1 grid without promoting the whole grid to true floor-casting (which is Phase 8 scope). When Phase 8 adds true floor-casting, pulses promote to world-anchored at the same time.

**Pulse contract (fake-perspective-anchored):**

- Horizontal pulses (depth bands): camera-agnostic, reuse Flappy helpers unchanged (`resolvePlaybackGroundGridHorizontalPulsePath`, travel ratio, size/alpha). Lifetime is adapted to 2700ms (§3.3.6), not Flappy's 5900ms.
- Vertical pulses (perspective rays): continuity cache stores a **world bearing** (`worldBearingRad = player.yaw + rayScreenAngle`) at pulse birth, re-matched by nearest `Δθ` (wrapped to `[-π, π]`) each frame. If `|Δθ| > 0.1 rad`, the pulse fades out for its remaining lifetime — never re-anchors to a new ray. This preserves continuity under camera yaw without snapping.
- **Grazing-angle clamp:** pulse projected edge length clamped to ≥2px (so grazing-angle pulses don't alias to nothing). Thickness from the depth curve is unchanged.
- **Rendering layer:** pulses are dynamic (move every sim tick), so they render on **Layer 2** (dynamic, per-frame), NOT Layer 1 (static grid). The static grid (lines, horizon) stays on Layer 1; pulses are a dynamic sub-layer above it.
- **Wall occlusion:** pulses are depth-tested against the per-column z-buffer (§3.4.1). Each pulse's projected screen columns are clipped where `zBuffer[x] < pulseDist` (wall is closer). This prevents pulses from showing through walls. Cheap — reuses the existing z-buffer, one compare per pulse column.

#### 3.3.6 Pulse density for combat

Flappy's `PULSE_INTERVAL_MS = 6000` is calibrated for a calm side-scroller. Neatenstein has 15–25s episodes (§12.2); at 6s cadence the viewer sees only 2–4 pulses per episode — below the "floor is alive" threshold.

- **Ambient baseline:** `NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 3000` (lifetime 2700ms, 90% ratio). ~5–8 ambient pulses per episode. Reads as "the floor is alive" without competing with combat silhouettes. This is an **adaptation** of Flappy's 6000ms, not a reuse.
- **Event pulses (see §3.3.8):** additional pulses fired by game events, on top of the ambient baseline.
- **Hard ceiling:** max 8 concurrent pulses on screen (ambient + event). Above this, drop oldest event pulses first; never drop an in-flight ambient pulse mid-travel (it would pop). The low-health dim (§3.3.8 #3) is an alpha/interval modifier, NOT a concurrent pulse — it does not count against the ceiling.

#### 3.3.7 Pulse determinism contract

Pulses MUST reproduce identically under replay (Phase 2 determinism acceptance, Phase 6 replay buffer).

- **Driver:** sim tick count (fixed-timestep). NOT `frameIndex` (tier-dependent: 60Hz GPU vs 30Hz CPU produce different counts). NOT wall-clock (drifts across runs).
- **Emission:** `if (simTick % PULSE_AMBIENT_INTERVAL_TICKS === 0) emitPulse(seededRng.next())`. `PULSE_AMBIENT_INTERVAL_TICKS = round(3000 / tickMs)`.
- **Position:** `cellX = seededRng.nextInt(0, gridW)`, `cellY = seededRng.nextInt(0, gridH)`, `dir = seededRng.pick([+X, +Y])`. The RNG is the episode seed (same seed → same pulse sequence).
- **Lifetime:** advanced by sim ticks, not ms: `pulseTicksRemaining--` per tick; despawn at 0.
- **Event pulses** (§3.3.8) fire on the **same sim tick** as their triggering sim event (enemy death, generation-up), never deferred to a render tick. This closes the Phase 6 replay contract hole.
- **Render-side interpolation** (§4.1.2) may display the pulse between sim states via `lerp(pulsePrevPos, pulseCurrPos, alpha)` — cosmetic, does not affect sim determinism.

#### 3.3.8 Event pulses — secondary legibility signal

Event pulses turn the floor from decoration into a secondary legibility channel that communicates state without adding HUD text (respects Phase 7's ≤20-word transient cap). Three bindings, all sim-tick-driven, all deterministic:

1. **Generation-up ripple (flagship):** On generation counter increment, emit a radial pulse wave from the arena center (or player position in human modes) — a single expanding ring on the floor grid, lifetime 600ms, white-hot (lerp `#fff14a` → `#ffffff`), additive blend. This is the visual half of the generation-up audio-visual pair (§3.3.9).
2. **Enemy death pulse:** On enemy death, emit a single brief pulse at the enemy's world cell, tinted to the enemy's hue (`NEATENSTEIN_PALETTE.enemyHues[type]`), lifetime 400ms. Reinforces the death burst (§4.3.2) with a floor-level echo. Bounded by 8-enemy cap.
3. **Low-health dim:** When player health < 30%, dim ambient pulses (alpha × 0.5) and slow them (interval × 1.5). No new pulses; a global modulation. Reverts on heal. This is a modifier, not a concurrent pulse.

**Rejected (clutter risk):** player-damage pulse (competes with damage flash §4.3.2), per-projectile pulses (strobe during firefights), ambient combat-tint (fights enemy-hue system §3.4).

#### 3.3.9 Generation-up audio-visual pair

The generation-up sound (§9.2 #6, rising arpeggio 330→660→990 Hz, 200ms) and the generation-up ripple (§3.3.8 #1, white-hot expanding floor ring, 600ms) fire **on the same sim tick**. The audio is the signal, the floor ripple is the reinforcement — together they make "the enemies just learned" unmistakable. Audio punches in (200ms), ripple lingers (600ms). They share a start, not an end.

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

### 3.5 Projectiles

- Neon line segments (tracers) oriented along velocity: 2px core line + `shadowBlur` glow.
- Player projectiles: `#00bfff` (neon blue, per `FLAPPY_NEON_PALETTE`).
- Enemy projectiles: `#ff4a8d` (reuse `championBird` pink for contrast).
- Depth-sorted with enemies.

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
- **Wall variants:** 3-5 neon hues assigned to different regions to give visual variety (the "light level" idea from DOOM, expressed as hue variation).

### 5.2 Map transfer

- Grid is a `Uint8Array` transferred to the worker once at episode start (not per frame).
- On mode switch (within the same enemy family), the grid is NOT re-sent — only the enemy state changes. On RESET, a new seed generates a new grid.

---

## 6. Performance Budget (tier-aware)

| Tier           | Columns | Enemies | Glow                                                            | Target fps       |
| -------------- | ------- | ------- | --------------------------------------------------------------- | ---------------- |
| GPU 2048/2048  | 320     | 8       | full shadowBlur                                                 | 60               |
| Worker 256/256 | 240     | 8       | limited shadowBlur / `ctx.filter` drop-shadow                   | 60               |
| CPU 128/128    | 160     | 8       | no shadowBlur, pre-rendered glow sprites, no texture modulation | 60 (fallback 30) |

- Column count is the primary render cost lever (DDA per column). 320 columns at 60fps is trivial on any modern CPU; the bottleneck is `shadowBlur`, not DDA.
- Enemy count is capped at 8 across all tiers (legibility constraint, not performance).
- `shadowBlur` is the expensive operation; tier-gate it aggressively. CPU tier uses the ImageData framebuffer + pre-rendered glow sprites path (§3.2.1); Worker tier uses OffscreenCanvas + `ctx.filter`; GPU tier keeps `shadowBlur` (§3.2.1, §3.2.3).

---

## 7. What We Do NOT Take from the carlini Clone

To be explicit about the boundary (license safety + design clarity):

- **No WebGL.** The carlini clone uses WebGL with shadow maps, GLSL shaders, and 3D polygon rendering. We use canvas 2D with DDA raycasting. Completely different rendering path.
- **No turtle-graphics map format.** The carlini clone compresses maps as turtle commands. We use a plain `Uint8Array` grid.
- **No polygon sectors with variable floor/ceiling heights.** We use a flat-grid (Lineage B). Variable heights are a Lineage A feature we explicitly skip.
- **No shadow mapping.** The carlini clone computes shadow maps for lights. We use distance fog + side shading for depth cues.
- **No code copied.** All algorithms are reimplemented in original TypeScript from the paraphrased descriptions above.

What we _do_ take from the carlini clone conceptually (game-feel, not code):

- The _idea_ of light levels per region (we express this as neon hue/brightness variation per grid cell).
- The _idea_ of billboarded sprites clipped against walls (we use wireframe sprites, not textured billboards, but the depth-sort + clip approach is the same).
- The _idea_ of a muzzle flash light (we express this as a brief `shadowBlur` burst, not a dynamic light source).
- Game-feel techniques: camera bob, gun bob, hit-stop, screen shake, exploding-cube death, alert-propagation enemy AI (all attributed to carlini's design notes; no code reproduced).

---

## 8. Reuse from Flappy Bird / Existing Repo

| Flappy pattern                                              | Neatenstein reuse                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FLAPPY_NEON_PALETTE`                                       | Extended as `NEATENSTEIN_PALETTE` (wall hues, enemy hues, horizon)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Flappy ground grid** (`playback/background/ground-grid/`) | **Floor renderer.** **Reused unchanged:** `resolvePlaybackGroundGridDepthCurve`, `resolvePlaybackGroundGridLineAlpha/Blur/Thickness` (depth-ratio-driven, camera-agnostic); `FLAPPY_NEON_PALETTE.groundGridLine/Fog/horizonLine/horizonGlow`; ambient `groundGridPulseFill=#fff14a`; `resolvePlaybackGroundGridUnitHash` (deterministic selection); travel range 0.14–0.62. **Adapted:** vertical-ray vanishing point → camera-yaw-rotated; vertical-pulse continuity → world-bearing match (§3.3.5); pulse emission driver → sim tick (§3.3.7); `PULSE_INTERVAL_MS=6000` → `NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS=3000`; `PULSE_LIFETIME_MS=5900` → `NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS=2700` (90% ratio, §3.3.6); + event pulses (§3.3.8). See §3.3.5–§3.3.9. |
| `FLAPPY_GROUND_GRID_*` constants                            | Reuse: `HORIZONTAL_LINE_COUNT=16`, `DEPTH_CURVE_EXPONENT=2.35`, alpha/blur/thickness ranges, travel range 0.14/0.62. **Adapt:** `PULSE_INTERVAL_MS=6000` → `3000`, `PULSE_LIFETIME_MS=5900` → `2700` (§3.3.6); add `NEATENSTEIN_PULSE_EVENT_*` constants (§3.3.8).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `WorkerPlaybackFrameSnapshot` SoA + transfer list           | `NeatensteinRenderFrame` (player, enemies, projectiles as typed arrays)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `resolveWorkerPlaybackSnapshotTransferList`                 | `resolveNeatensteinSnapshotTransferList`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| WeakMap buffer pool                                         | Snapshot buffer reuse to avoid per-frame allocation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `requestId`-gated playback step                             | `request-render-step` / `render-step` protocol                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Fixed-timestep RAF loop                                     | Same loop structure, extended for FPS controls + interpolation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `autoEnableAcceleration` + `AccelerationStatus`             | Startup tier detection + chip label                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Racing `resolveAccelerationChipPresentation`                | Extended additively with `batchParallelCount`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Two-region layout + `ResizeObserver`                        | Canvas + sidebar, responsive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

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

## 11. Open Questions for Implementation

1. **Y-shearing (fake look up/down):** The plan specifies mouse look. True 3D look up/down is impossible in raycasting (walls are vertical). Y-shearing (moving the horizon line) fakes it but distorts. Recommend: horizontal mouse look only (rotate), no vertical look. The dash mechanic provides the "dodge" feel without vertical aim.
2. **Minimap:** DOOM had a map mode. A small top-down minimap in the corner (neon grid + player dot + enemy dots) would help spatial awareness and is cheap to draw. Recommend as a Phase 7 UI element, toggleable with `M`.
3. **Sprite rendering method:** Canvas 2D `drawImage` (if we use pre-rendered wireframe sprites) vs. direct line drawing. Direct line drawing is more flexible (wireframes scale without aliasing) but slower. Recommend direct line drawing for the neon aesthetic; the wireframe is only ~6 lines per enemy.

> Note: the original "floor-casting vs solid black" question is resolved in §3.3 — reuse Flappy Bird's synthwave ground grid (camera-adapted) for v1; true floor-casting is a Phase 8 stretch goal (§3.3.4).

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

1. **Render worker** (OffscreenCanvas, DDA + blit) — §3.2.1. Worker tier only.
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
