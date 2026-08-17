# Neatenstein Ultimate Quality Upgrade Plan

**Created:** 2025-01-20  
**Status:** APPROVED — all 16 specialist agents approved (8 original across 4 rounds + 8 floor-wall alignment across 3 rounds)
**Scope:** `examples/neatenstein/` — comprehensive multi-perspective analysis and upgrade plan

## Mandates

```yaml
pragmatic: true
broad_slices: true
single_model: false
bypass_ceremony: false
```

This plan declares pragmatic mode: broad slices are acceptable where cross-cutting concerns span multiple subsystems. The RED → IMPLEMENT → GREEN loop is enforced but gate ceremony is streamlined. Plan-verification green-light may be bypassed only for the documentation/quick-win phases.

## Non-Negotiable Invariants — Floor/Wall/Ceiling Grid Alignment

The floor and ceiling have a 3D neon grid with a traveling spark effect. The cell sizing on the floor/ceiling grid is the **main size unit**. ALL walls must always align to the lines on the floor and ceiling grid. This keeps ground and wall lines cohesive. **This currently works perfectly and must not be broken by any plan change.**

### Invariant 1: Shared Integer Grid

All walls occupy integer cells of the 120×120 grid (1 world unit = 1 map cell). Floor/ceiling grid lines are drawn at integer world coordinates. Both the wall DDA (`raycast.ts`, cell size = 1.0 world units) and the floor projection (`floor.projection.utils.ts`, integer world X/Y lines) share the same cell size. The alignment is mathematically exact: `screenX_wall = halfWidth + o·halfWidth = screenX_floor` (proven via the identity `planeScale·focalLength = halfWidth`).

### Invariant 2: Shared Projection Constants

Every rendering path — current Canvas 2D line projection, `Uint8ClampedArray` framebuffer, per-pixel floor caster, and WGSL/GLSL shader — MUST consume the same projection constants:
- `NEATENSTEIN_FLOOR_FOV_RADIANS` (vertical FOV)
- `NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD = 0.5` (camera at half-cell height)
- `NEATENSTEIN_FLOOR_HORIZON_RATIO = 0.5` (horizon at canvas center)
- `focalLength = canvasHeight/2 / tan(FOV/2)`
- `planeScale = (W/H) · tan(FOV/2)` (so `planeScale · focalLength = halfWidth`)
- `NEATENSTEIN_RENDER_DISTANCE_CAP` (shared depth cull for both walls and floor)

No rendering path may independently re-derive these values. The shader path must import them as uniform inputs from the same constants module.

### Invariant 3: Traveling Spark Preservation

The "traveling spark" is the ambient pulse system (`pulse.ts`): small shiny dots that travel along integer floor-grid lines, depth-tested against the per-column z-buffer. This is a **signature visual effect** and a **core part of the simulation's identity**. Every rendering change (framebuffer, shader, per-pixel casting) MUST preserve:
1. The visible 3D neon floor/ceiling grid at 1-world-unit spacing
2. The traveling spark overlay with world-space integer-line travel, z-buffer depth occlusion, and lifetime alpha
3. The spark's visual coupling to the grid (it slides along the same integer lines)

### Invariant 4: Procedural Floor Grid (Not Texture)

Any per-pixel floor casting (B3.7, C1.1) MUST render a **procedural world-space integer grid** computed via `fract(worldCoord)` or equivalent — NOT an arbitrary sampled texture. The grid line spacing MUST equal exactly 1 world unit = 1 map cell. A sampled texture's grid lines would only align to walls if the texture tiles at 1-unit pitch with the same integer origin; this is fragile and explicitly forbidden. The floor must remain a procedural neon grid.

### Invariant 5: Render Compositing Order

The render compositing order MUST be: **floor → ceiling → walls → sprites → pulses/sparks → bolts**. Walls are drawn on top of the floor grid so the wall base sits exactly where the integer floor line projects. Any refactor (B2 file extraction, B1 worker split) MUST preserve this draw order.

### Invariant 6: Depth Cap Synchronization (Step Count ≠ Perpendicular Distance)

The current code overloads `NEATENSTEIN_RENDER_DISTANCE_CAP = 30` for two different quantities: (a) **step count** (`raycast.ts:169`, `steps >= CAP`) and (b) **perpendicular distance** (`floor.projection.utils.ts:110`, `camSpaceY > CAP`; plus fog `walls.ts:91`). These MUST be split into two separate constants:

- `NEATENSTEIN_DDA_MAX_STEPS` — the DDA step budget (≈43 or angle-aware `ceil(CAP / min(|dirX|, |dirY|))`), used ONLY in `raycast.ts:169` to control how many grid-cell crossings the DDA walks.
- `NEATENSTEIN_RENDER_DISTANCE_CAP` — the perpendicular distance cap (30, unchanged), used by the floor cull (`floor.projection.utils.ts:110`), wall fog (`walls.ts:91`), and floor fog.

The angle-aware step-cap fix (A5 item 5 option a) raises the **step** count so walls are *detected* at all angles up to 30 perpendicular units — the perpendicular-distance cap stays 30 and the floor cull needs **NO change**. Only the conservative static-raise option (b) would raise the effective perpendicular cap and require the floor cull to rise in lockstep — but option (a) is recommended. **Do NOT raise `NEATENSTEIN_RENDER_DISTANCE_CAP` itself; the step count and the perpendicular cap must remain decoupled constants.**

### Invariant 7: Fog Coordination

If smooth fog (A5 item 2) is adopted, the same `smoothstep(FOG_START, CAP, d)` fog factor MUST be applied identically to: (a) wall color fog, (b) floor/ceiling grid color fog, and (c) floor/ceiling grid alpha. The existing alpha-band falloff MUST be folded into the single fog factor — do NOT multiply alpha-band falloff on top of color fog, or the grid vanishes before walls, making walls appear to float off the grid.

### Invariant 8: Regression Test (Horizontal + Vertical Alignment)

Before any rendering change, a regression test MUST be written asserting the floor-wall alignment:

- **Horizontal (X):** For several fixed camera poses and columns, cast the wall ray, take the wall-hit world point (`cam + perpWallDist · rayDir`), project it via `projectNeatensteinGridPoint`, and assert its `screenX` matches the **continuous** projected wall-face screen X (the ray-boundary crossing), not the quantized `xStart` — within the tier tolerance below. Additionally assert the constant-X and constant-Y integer grid lines passing through that hit point intersect the wall base at that same screen X.
- **Vertical (Y):** For the same camera poses, assert the wall-base screen Y (`horizonY + wallFocalLength / (2 · perpWallDist)`) equals the floor grid line screen Y at the same perpendicular distance, within the tolerance below.
- **Spark↔grid coupling:** Assert the traveling spark's projected position lies on an integer grid line (its `fixedCoord` is an integer) and that no pulse's `fixedCoord` becomes fractional after `updateNeatensteinPulses`.
- **Tolerance:** `< 0.5px` for per-pixel shader tiers; `≤ 1 · stripeWidth` for the per-column JS/fallback tier (column quantization is inherent). Cross-tier comparison (JS vs shader) may differ by up to `stripeWidth` and is acceptable.

This invariant is currently implicit via shared constants; it must be made explicit before implementing any rendering change (A2, A5, B3, C1).

## Analysis Methodology

Eight specialist agents analyzed the Neatenstein demo from independent perspectives:

| # | Agent | Perspective | Key Finding |
|---|-------|------------|-------------|
| 1 | maze-generation | Maze quality | "Maze" is random Bernoulli noise scatter, not a maze |
| 2 | enemy-neat-evolution | Enemy evolution | Enemies do NOT evolve — deterministic weight reseeding only |
| 3 | enemy-parallelism | AI independence | All enemies share one heuristic + one MLP, sequential on single worker |
| 4 | nge-hero-evolution | NGE hero pipeline | Embryo built then discarded; trains against static dummies |
| 5 | raycasting-impl | Raycasting correctness | CPU tier renders blank screen; GPU tier absent; fog is binary cliff |
| 6 | performance-analysis | Per-tick/frame allocations | ~20K floor objects/frame, 1.2MB ImageData/frame, BFS 2-4×/tick |
| 7 | algorithm-research | State-of-the-art algorithms | MAP-Elites, CMA-ES, FAMOU opponent pools, per-death evolution |
| 8 | code-quality-review | Code modernness | God-file display.worker.ts, leaky scripts↔browser-entry boundary |

---

## Phase A — Critical Foundation (Blocks Everything)

### Step A1: Maze Generation Overhaul

**Priority:** P0  
**Severity:** Critical (gameplay-breaking)  
**Source agents:** maze-generation, algorithm-research  
**Files:** `browser-entry/renderer/map.ts:201-215`

#### Problem

`buildNeatensteinMap(seed)` generates random Bernoulli noise at 12% density on a 120×120 grid. This is NOT a maze:
- No corridors, no connectivity guarantee, no path structure
- Below percolation threshold (0.5928 for square lattice) → floor fragmented into disconnected clusters
- Enemies can spawn in unreachable pockets
- Corridor width is 1 cell (scatter granularity)
- Central arena is a 9×9 clear circle — no pathways lead to it

#### Solution: Coarse-Grid Recursive Backtracker Rooted at Center

The user explicitly asked for "actual pathways, all leading to the center, 3 or 4 cells wide, even on the corners." A coarse-grid recursive backtracker rooted at the center satisfies this requirement natively: every cell has exactly one path back to the center root, producing a convergence topology. BSP produces a room-and-corridor dungeon, not a convergence maze — it was rejected because it does not guarantee "all leading to center" without additional plumbing.

**Algorithm: Coarse-Grid Recursive Backtracker with Structural Wide Corridors**

1. **Coarse grid** — Divide the 120×120 fine grid into a coarse grid of `120 / (W+1) = 30` cells, where `W = CORRIDOR_WIDTH` (3). Each coarse cell maps to a `(W+1)×(W+1) = 4×4` fine block. The extra +1 row/column per cell becomes the structural wall between corridors.
2. **Root at center** — Start the recursive backtracker at the coarse cell containing the map center (coarse cell 15,15). This guarantees every path leads to the center by construction.
3. **Recursive backtracker** — Standard depth-first carving on the coarse grid: visit a neighbor, carve the wall between them, recurse. This produces a perfect maze (spanning tree) where every coarse cell has exactly one path to the root.
4. **Fine expansion** — Each carved coarse corridor becomes a `W`-cell-wide fine corridor. The `+1` gap between coarse cells provides the structural wall. **Corners are uniformly wide by construction** — at every L-turn, the coarse cell becomes a `W×W` block, not a 1-cell pinch. No post-widening pass needed.
5. **Central arena** — The center coarse cell (15,15) is expanded to the 9×9 arena clear zone. This IS the hub — the arena is the root of the spanning tree.
6. **Flood-fill verification with repair** — After generation, flood-fill from the center. Any floor cell not reached indicates a bug (should not happen with a spanning tree, but verify). If unreachable clusters are found, **carve a connecting passage** from the isolated cluster to the nearest reachable floor cell — do not just assert.
7. **Add loops** — Remove ~3% of single-cell wall stubs between adjacent corridors (NOT 10% — with 3-wide corridors, each removed wall opens a 3-cell breach; 10% would collapse the corridor structure). Restrict removal to walls that are exactly 1 cell thick (between two corridor cells), preserving the overall structure.
8. **LCG draw order** — All random draws (neighbor selection, loop removal target selection, repair passage carving) must consume the Park-Miller LCG in a documented, stable sequence: (1) backtracker neighbor shuffle order, (2) loop-removal candidate selection, (3) repair passage selection. This preserves the seed-determinism guarantee.

Reuse existing Park-Miller LCG for determinism. The `CollisionMap` interface and `castRayDDAFromFlatMap` require no changes — they consume the same `Uint8Array` format.

**Constants to update:**
- `NEATENSTEIN_MAP_SIZE = 120` (keep)
- `INTERIOR_WALL_DENSITY = 0.12` (remove — replaced by backtracker parameters)
- `CENTRAL_ARENA_CLEARANCE_CELLS = 4` (keep — folded into step 5)
- Add: `MAZE_CORRIDOR_WIDTH = 3`, `MAZE_COARSE_GRID_DIVISOR = 4` (W+1), `MAZE_LOOP_REMOVAL_RATE = 0.03`

**Estimated effort:** ~250 lines replacing `scatterInteriorWalls()` + `carveCentralArena()` with `buildBacktrackerMaze(seed, side)`. The `buildNeatensteinMap` function signature stays the same.

### Step A2: Performance — Eliminate Per-Frame Allocation Bombs

**Priority:** P0  
**Severity:** Critical (sub-60fps risk on common hardware)  
**Source agent:** performance-analysis  
**Files:** Multiple — see below

#### Problem

Three critical allocation sources threaten frame budget:

1. **Floor/ceiling grid projection: ~20,000 object allocations/frame**  
   `floor.ts:194-273` + `floor.projection.utils.ts:91-148`  
   `projectNeatensteinGridPoint` returns fresh `ProjectedNeatensteinGridPoint` per sample. 2 calls (floor+ceiling) × 122 lines × 81 samples = ~19,764 objects/frame → ~1.2M objects/sec

2. **`getImageData` full-frame (≈1.2 MB) allocation every frame**  
   `display.worker.render.utils.ts:319`  
   Allocates full-frame `ImageData` (W×H×4) every frame. At 640×480 = ~1.2MB/frame → ~72MB/sec of garbage. Likely #1 cause of frame hitches.

3. **Full 120×120 BFS rebuilt 2–4× per tick**  
   `enemy-navigation.ts:106-187`  
   Fresh `Int32Array(14400)` (~57KB) + `DistanceMap` object every call. Called twice per tick (normal + zero-timestep pass), plus `extractSensors` builds a 2nd distance map. ~228KB typed-array allocation + 4 full-grid wall scans per tick.

#### Solution

**Immutability boundary:** Internal sim working state (`runSimStep` locals) becomes mutable for performance. The `GameState` returned to the host via `postMessage` remains a fresh immutable object. `tick.ts` retains its spread-based pattern. `fireGateState` returns a new state object (per B2 item 6), not mutated in place.

**Fix 1 — Floor projection:** Make `projectNeatensteinGridPoint` write into a **two-slot ping-pong scratch** (`prev`/`curr`) passed by reference — NOT a single shared scratch object, because `appendNeatensteinGridLine` (`floor.band.utils.ts:93-125`) keeps `previous = projected` across iterations and reads `previous.x/y/depthRatio` against the *next* `projected`; aliasing them to the same object produces zero-length segments and wrong band assignments (grid flicker). Pool `bands` arrays across frames (`createNeatensteinFloorSegmentBands` once, `.length = 0` each frame — safe for flat number arrays). Pool the `projection` context object. Pool the `floorCamera` literal (`display.worker.render.utils.ts:150-154`) as a module-level mutable object. Add a regression test asserting segment lengths are nonzero for a moving camera.

**Fix 2 — Sprite ImageData + wall framebuffer:** Allocate ONE `ImageData` at canvas size, keep on worker, reuse every frame. Write sprite pixels directly into persistent `Uint8ClampedArray` framebuffer, `putImageData` once at end. Only re-allocate on canvas resize. **Additionally, move wall rendering off per-column `fillStyle`/`fillRect`** — write fogged RGB triples directly into the same `Uint8ClampedArray` framebuffer (vertical run per column), then one `putImageData` flush for walls + sprites combined. This eliminates: (a) per-column CSS string allocations, (b) canvas fillStyle re-parse 480-640×/frame, (c) `createLinearGradient` per capped column. Note: this unifies with A5's smooth fog fix — continuous fog factors are trivial in a framebuffer (just interpolate RGB per pixel) but impossible with cached CSS strings. Sort sprites in place into a reused scratch array (not `[...sprites].sort(...)`).

**Framebuffer alignment safeguards (per Non-Negotiable Invariants §1, §5):**
- **Floor grid preservation:** The floor/ceiling neon grid is drawn via Canvas 2D path calls (`drawNeatensteinFloor/Ceiling`) BEFORE the wall framebuffer flush. The framebuffer MUST be seeded from the canvas (via `getImageData` of the already-drawn floor+ceiling) before wall+sprite pixels are written — `putImageData` replaces pixels (no compositing), so writing a fresh empty framebuffer would overwrite the floor/ceiling grid. Alternatively, draw the floor/ceiling grid into the framebuffer directly (procedural pixel writes — preferred as the allocation-free target, eliminating the ~1.2 MB `getImageData` allocation). Either approach preserves the visible grid. If `getImageData` seeding is used, ensure the canvas has an **opaque background fill** before the read (transparent canvas pixels yield alpha=0, which `putImageData` writes as black holes). Feathered fog-wall edges in the framebuffer path: blend wall-edge pixels toward floor color using the same fog factor (Invariant §7) — do NOT leave hard 1px edges where fog transitions.
- **Wall column coverage:** `writeNeonWallColumn` (`walls.ts:161`) writes a single 1px-wide column. The current worker path writes `stripePixelWidth`-wide rects. When `columnCount < canvasWidth`, the framebuffer wall writer MUST loop `x` from `xStart = Math.floor(column * stripeWidth)` to `xEnd = Math.floor((column+1) * stripeWidth)`, writing each pixel column — NOT a single 1px column. Otherwise 1px gaps appear between wall columns, letting floor grid lines show through walls.
- **Column→pixelX mapping:** The framebuffer write loop MUST use the identical `xStart = Math.floor(column * stripeWidth)` mapping that the current `fillRect` path uses. Do NOT write the raycast column index directly as the framebuffer X.

**Fix 3 — BFS cache:** Allocate one module-level reusable `Int32Array(14400)`, pass into every `buildEnemyDistanceMap` call. Precompute static `Uint8Array` wall-mask once per map seed. Cache distance map per (map-seed, player-cell) — only rebuild when player crosses cell boundary (keyed by `Math.floor(playerX)`, `Math.floor(playerY)`). This also benefits B1's enemy scaling (flow-field shared by all enemies targeting same player cell).

**Fix 4 — Zero-timestep pass (reconciled with B1):** The second controller pass at `sim.utils.ts:373-379` runs after de-rez pruning to refresh `separateEnemies` against the post-removal roster. It is NOT simply removable. Instead: (a) **reuse the distance map from pass 1** (player cell is identical between passes), eliminating the redundant BFS rebuild; (b) **conditionally skip the entire pass** only when `completedDeRezIndices.length === 0` and no bolt-spawn state changed. A2 and B1 are now aligned on this approach.

**Fix 5 — `gameTick` allocation chain (explicitly enumerated):** `gameTick` (`tick.ts:79-162`) is called from `runSimStep` and has its own per-tick allocations:
- `tick.ts:122-132, 143-146`: two `{...next}` GameState clones → keep immutable (per boundary above)
- `tick.ts:124, 145`: `updatedBolts.filter(bolt.active)` + `enemyBoltResult.bolts.filter(...)` → mutate bolts in place with `active` flag, compact in single pass
- `tick.bolt.utils.ts` / `tick.enemy-bolt.utils.ts`: `.filter().map(bolt => ({...bolt, position: {...}}))` → mutate bolt position in place, use pooled `Vector2` for `nextPosition`
- `tick.impact.utils.ts:20-55`: `ageImpacts`/`ageEnemyImpacts` `.map(clone).filter(...)` → mutate `lifetimeMs` in place, filter in single pass into reused array
- `tick.pickups.utils.ts:39-61`: `updateAmmoPickups` `.map(clone).filter(...)` → mutate `active` in place, compact in single pass
- `tick.lifecycle.utils.ts:53-64`: `[...(next.enemyImpacts ?? []), enemyImpact].slice(-MAX)` → use ring buffer instead of full copy + slice

**Fix 6 — De-rez pruning chain:** `sim.utils.ts:333-358` runs 3 chained `.map().filter().map()` every tick even when no enemy died, plus `completedDeRezIndices.includes(index)` is O(n²). Replace with single in-place loop + `Set` for completed indices. Only run the chain when `completedDeRezIndices.length > 0`.

**Fix 7 — Frame typed-array pooling (conditional on A5 tier decision):** If A5 retains the packed-frame path (CPU tier), pool the 6 typed arrays on the worker (module-level, reallocate only on column-count change) and double-buffer with transfer-back. If A5 removes the packed-frame path (shader raycaster), these arrays become dead code — remove them entirely. A5's tier strategy decision must precede this fix. **Double-buffer synchronization (per Invariant §5):** Floor, ceiling, walls, and sprites must all render into the **same buffer in the same render pass** before any transfer. Never let the floor-grid path draw to buffer A while walls draw to buffer B — the double-buffer swap must happen once per complete frame, atomically, after floor+walls+ceiling+sprites are all flushed.

**Fix 8 — Additional high-priority fixes:**
- `runSimStep` deep-clones game state every tick → switch to mutable internal state (per immutability boundary above)
- `updateEnemyController` rebuilds `Map` + arrays every tick → preallocate reusable (plain array indexed by `index` for 8 slots, `.length = 0` scratch arrays)
- Per-enemy `EnemyUpdateContext` + nested objects → pool 8 slot contexts, reset scalars and mutate `position`/`slotTarget` in place
- `computeMovement` clones direction tuples per enemy → use flat `Int32Array` of dx,dy pairs + `Int8Array(4)` index sort by score
- Per-ray hit object allocation (480-640/frame) → write into preallocated `Float32Array` column buffer (perpDist, side, mapX, mapY as 4 slots)
- `activateMlp` allocates `Float32Array` per layer per enemy → pool 2 ping-pong activation buffers per enemy slot, reset via `.fill(0)`

### Step A3: NGE Hero Evolution — Materialize the Embryo

**Priority:** P0  
**Severity:** Critical (core thesis not implemented)  
**Source agent:** nge-hero-evolution  
**Files:** `browser-entry/harness/main-runner.ts:208-224, 307-362`, `browser-entry/worker/eval.worker.ts`

#### Problem

The thesis "Train your own killer — then survive it" is NOT implemented:

1. **NGE embryo built then DISCARDED** — `main-runner.ts:316-320` constructs `new Network(22, 5, {seed})` (a random network) instead of using the embryo. The embryo's `nodeCount=3, edgeCount=6` is never grown through lifecycle stages.

2. **Fitness episodes run against enemies that NEVER MOVE OR FIRE** — `gameTick` doesn't call `updateEnemyController`. Frozen `enemySnapshot` passed in but never injected into episodes. Hero trains against static dummies.

3. **Fitness is reward-hackable** — `survivalTicks * 1` with max 312 and `damageTaken=0` (enemies don't fire) → survival is near-constant. Parsimony band [800,3000] vs actual complexity of 9 → constant penalty, zero ranking effect.

4. **popsize=4, no speciation, no novelty, no tournament** — Library default `speciation=false`. Hill-climbing with 4 samples.

5. **"Co-evolution" is cosmetic** — `enemyBehaviorMetrics` are `seedrandom()` draws, not real telemetry. No competitive fitness signal flows back.

6. **"Train your own killer" death feedback NOT wired in live runtime** — `replayBuffer` never pushed to outside tests.

#### Solution

1. **Materialize NGE embryo → Network (CRITICAL — specify the bridge)** — The NGE lifecycle modules in `src/neat/nge-main-agent/` produce typed state objects (embryo, juvenile, adult, reproducing), NOT `Network` instances. A materialization bridge must be built: a `materializeFromNgeState(state: NgeState): Network` function that walks the typed state's node and connection lists and constructs a `Network` instance via `new Network()` with the corresponding nodes/edges/synapses. This is the single most critical gap — without it, NGE produces state objects that can never be activated. The bridge lives in `src/neat/nge-main-agent/nge-to-network.ts` and must handle: (a) node gene → neuron with bias/activation, (b) connection gene → synapse with weight, (c) recurrent edges (NGE motifs: GatedRecurrentCell, EpisodicSlot). The main-agent `runEpisode` must call `materializeFromNgeState(grownState)` instead of `new Network(22, 5, {seed})`.

2. **Fix lifecycle terminology** — Use "reproducing" (not "assimilation") as the final NGE stage name, matching the existing `reproducing` lifecycle module in `src/neat/nge-main-agent/`. The embryo grows through embryo → juvenile → adult → reproducing, then the reproducing state is materialized into a Network for evaluation.

3. **Inject enemy snapshot weights into fitness episodes** — When running fitness episodes, load the frozen enemy `enemySnapshot` weights into active enemies that actually move and fire via `updateEnemyController`. The hero must train against live enemies. The `_enemySnapshot` parameter in `runEpisode` (currently underscore-prefixed and unused) must be injected: deserialize enemy MLP weights from the snapshot, pass them to `updateEnemyController`, and ensure `gameTick` calls `updateEnemyController` during fitness episodes.

4. **Headless vs live path reconciliation** — `main-runner.ts` (headless, uses random `new Network`) and `eval.worker.ts` (live, uses real NEAT popsize=4) are two divergent hero-eval paths. Unify: both paths must use the NGE materialization bridge (item 1). `main-runner.ts` becomes the headless fast-path for generation evaluation; `eval.worker.ts` becomes the live-path for interactive training. Both call `materializeFromNgeState`. The `createMainSnapshot` function must use the materialized Network, not a placeholder genome.

5. **Enable speciation + novelty** — Set `speciation: true`, enable novelty search (infrastructure exists in `src/neat/evaluate/novelty/`). Raise popsize to 50-100. Add tournament selection. Concrete option values: `{ speciation: { targetSpecies: 4-6, compatibilityThreshold: 3.0, excessCoeff: 1.0, disjointCoeff: 1.0, weightCoeff: 0.4 }, novelty: { noveltyThreshold: 0.3, archiveAdditionRate: 0.1, k: 5 }, popsize: 50 }`. **Popsize cost analysis:** Going from popsize=4 to popsize=50 is a 12.5× increase in evaluation cost per generation. Each evaluation runs a full fitness episode (one playthrough against enemies). Mitigation: (a) episodes are headless (no rendering), so they are fast (~10-50ms each); (b) `eval.worker.ts` can batch-evaluate in parallel using `ParallelInferencePool` (offline batch is its correct use case); (c) generation turnover happens between waves, not per-tick, so the latency is amortized. Start with popsize=25 as a middle ground, scale to 50 if performance allows.

6. **Fix fitness function** — Normalize survival ticks against max possible (divide by `maxEpisodeTicks`). **Terminate fitness episode on first hero death** — make `isEpisodeComplete` return `true` when `gameState.deaths > 0` (file: `browser-entry/host/game/episode.ts`). Then `survivalTicks` = actual ticks survived before first death, and `normalized_survival = survivalTicks / maxEpisodeTicks` becomes a meaningful [0, 1] survival duration metric. This makes every death terminal and meaningful, matching the thesis ("survive it" = don't die). Add novelty bonus. Fix parsimony band to match actual complexity range (embryo starts at 3 nodes / 6 edges; after growth, expect 10-40 nodes; set parsimony band to [10, 50]). Add engagement shaping: bonus for damage dealt, penalty for time spent in dead-end corridors (prevents corner-hiding reward hack). Formula: `fitness = normalized_survival + novelty_bonus + (player_damage_dealt - enemy_damage_dealt) * w_adversarial + engagement_bonus - corner_hiding_penalty`. **`w_adversarial` default:** 0.3 (adversarial component is a secondary signal, not dominant). **Schedule:** ramp from 0.1 (early generations, focus on survival) to 0.5 (late generations, focus on combat) linearly over the first 50 generations.

7. **Wire death feedback loop** — On hero death, push `DeathContext` to `replayBuffer`. Use replay buffer pressure to shift evaluation seeds (not cosmetic RNG — actually select different enemy variants from the opponent pool based on replay pressure). Feed real telemetry (not RNG) into behavior metrics.

8. **Unify hall-of-fame and opponent pool** — Maintain a bounded league of past champion enemies (AlphaStar-style). Evaluate against samples from this league (not just current snapshot) to prevent forgetting. This league is shared with B4's opponent pool — see B4.3 for the unified league structure. B4.10's adversarial formula is merged with this item (A3.4) — there is one adversarial fitness formula, defined here and referenced by B4.

9. **Resolve A3↔A4 bootstrap** — Generation 0 has no evolved enemies yet. Bootstrap order: (a) A4 runs first to produce an initial enemy population (even if unevolved — just the warm-start MLPs with per-variant fitness ledgers); (b) A3 evaluates the hero against this initial population; (c) both co-evolve from there. A4 does NOT depend on A3 — A4 can start with static warm-start enemies. A3 depends on A4 for live opponents. This breaks the circular dependency: A4 → A3 → (both co-evolve).

10. **Extend sensor vector** — Add health, ammo level (not binary gate), dash cooldown, enemy bearings, bolt-incoming sensors. Enable recurrent state (NGE motifs: GatedRecurrentCell, EpisodicSlot). Per-node evolvable time constants (from B4.7) add temporal memory.

11. **QD archive for main agent** — The hero also needs a Quality-Diversity archive (MAP-Elites grid keyed by behavioral descriptors like exploration coverage and combat style). Without it, co-evolutionary collapse is a risk: the hero overfits to the current enemy population and loses behavioral diversity. The hero's MAP-Elites archive is separate from the enemy archive (different descriptor dimensions). See B4.6 for CERL-style shared replay that complements this.

### Step A4: Enemy NEAT Evolution — Real Per-Death Evolution

**Priority:** P0  
**Severity:** Critical (core feature missing)  
**Source agents:** enemy-neat-evolution, algorithm-research  
**Files:** `browser-entry/harness/enemy-mlp.ts:88-95`, `scripts/enemy-controller.ts:264-279`, `browser-entry/harness/arms-race.ts:145-152`, `browser-entry/harness/death-feedback.ts:69-93`

#### Problem

Enemies do NOT evolve. No NEAT, no selection, no mutation. Just deterministic weight reseeding:

- `enemy-mlp.ts:88-95`: `update()` calls `warmStartTemplate(seed + generation * CHAMPION_SEED_PRIME)` every 5 generations — a pure function of (seed, generation), same result regardless of gameplay
- `enemy-swarm.ts`: Shared DNA string and 8-element weight vector generated once, never change
- `enemy-controller.ts`: ~90% hardcoded BFS/flank/if-else scripting. MLP only re-ranks BFS direction candidates. Fire is fully scripted. All respawns get `variantId=0` — no per-enemy identity across deaths
- `death-feedback.ts:69-93`: Only computes HUD display metrics, does NOT feed back into any learner. `EnemyBehaviorMetrics` are RNG draws, not measured from gameplay
- `arms-race.ts:145-152`: `enemyBehaviorMetrics` = `seedrandom()` draws — pure noise
- `replay-buffer.ts`: Records DeathContexts but never read by evolution code

#### Solution

1. **Per-variant fitness ledger** — Each enemy variant gets a `FitnessRecord { damageDealt, survivalTicks, kills, deaths }` accumulated across its lifetime. Stored on the variant object, not in a separate map.

2. **Lamarckian weight mutation on death** — When an enemy dies:
   a. Record `{ variantId, damageDealt, survivalTicks, kills }` into the variant's fitness ledger
   b. Sample parent by **seeded tournament/roulette selection** via a new `selectParentProportional(population, fitnessRecords, mutationSeed)` function (with uniform fallback when `Σfitness ≤ 0`) — NOT the existing `selectVariant` (which is deterministic elitist max-selection from `select.ts:39-56`, reserved for champion extraction at wave-sync). MAP-Elites admission (item 4) is the diversity mechanism; proportional parent sampling is the exploration mechanism.
   c. Copy parent weights, apply Gaussian perturbation (σ configurable)
   d. Optionally: run N backprop steps on transition sequence from this life (Lamarckian)
   e. Respawn with the mutated weights and a new `variantId`

3. **Replace RNG-filled behavior metrics with real telemetry** — `EnemyBehaviorMetrics` must be computed from actual gameplay events, not `seedrandom()` draws. Raw telemetry: `damageDealt`, `survivalTicks`, `meanDistanceToPlayer`. Normalized descriptors (used by MAP-Elites grid indexing, require [0,1] bounded): `aggression = clamp01(damageDealt / (damageDealt + survivalTicks * 0.1))`, `positioning = clamp01(meanDistanceToPlayer / maxMapDistance)`, `movementPattern = clamp01(dirChangeCount / max(1, survivalTicks))`. Both A4 and B4.1 use the same normalized formulas.

4. **MAP-Elites archive** — 2D grid (10×10) keyed by (aggression, positioning) behavioral descriptors. When a new variant dominates an existing cell, replace it. This maintains a diverse repertoire of enemy strategies rather than collapsing to one champion. MAP-Elites and per-death evolution (item 5) are complementary, not conflicting: per-death evolution is the local mutation operator that generates candidate variants; MAP-Elites is the global archive that curates which variants survive. Every per-death mutation produces a candidate that is evaluated for MAP-Elites admission.

5. **Per-death online evolution** — Instead of refreshing the entire population on a fixed cadence, each enemy death triggers a local evolution event (mutate the killed enemy's weights, respawn with mutations). Periodically sync best per-death variants back to the population.

6. **Hebbian within-lifetime plasticity (DEPENDS ON B1)** — When an enemy fires and hits/misses, update its weights with a small Hebbian delta using **Oja's rule** (not classic Hebbian) for stability: `Δw = η * x * (y - w * y²)`. Oja's rule prevents weight explosion by normalizing. **Topology note:** The enemy MLP is `6→6→4→4` with `tanh` activations. The basic Oja formula applies to a single linear output — restrict within-lifetime plasticity to the **final `4→4` output layer's readout weights only**, leaving hidden-layer adaptation to the evolutionary/Lamarckian path. This keeps the formula valid and implementation bounded. This requires B1's per-enemy mutable weight buffers (each enemy must own its own weight copy that persists across ticks within a lifetime). A4-core (items 1-5, 7) does NOT depend on B1; A4-advanced (item 6, Hebbian plasticity) depends on B1. Split: A4-core can ship first, A4-advanced follows B1.

7. **Transition-level experience replay** — Record (state, action, reward, next_state) transitions during each enemy's lifetime. On death, run N backprop steps on these transitions before respawning. Combines evolution (topology/initial weights) with gradient-based RL (lifetime learning). Reuses existing `trainMlpBackprop`.

8. **SWARM evolution** — `enemy-swarm.ts` currently uses shared DNA/weights generated once. Add an evolution step: on wave transition, mutate the swarm DNA with Gaussian perturbation and select the best-performing swarm configuration from the previous wave. This is a simpler evolution loop (one genome for the whole swarm) and complements the per-variant MLP evolution.

9. **Relax BFS gate + wire MLP fire output** — Currently the controller hard-gates movement through BFS distance-reducing candidates only, and fire is fully scripted. The MLP must be able to override: (a) movement — MLP output rank can select non-distance-reducing moves (flanking, retreating) when the MLP score exceeds a configurable threshold; (b) fire — MLP's 4th output (fire) must be wired to the fire decision, not the scripted heuristic. The scripted heuristic becomes the fallback when MLP confidence is low.

10. **Define enemy fitness scalar** — `enemyFitness = α * damageDealt + β * survivalTicks + γ * kills - δ * damageTaken`, with `α=1.0, β=0.1, γ=2.0, δ=0.5` as defaults. This scalar is used for parent selection in per-death evolution and MAP-Elites admission.

11. **Deterministic-replay contract** — Per-death mutation breaks strict determinism (each death introduces stochastic mutation). Preserve replayability via a **mutation seed strategy**: each enemy carries a `mutationSeed` derived from `(globalSeed, variantId, deathCount)`. Replaying with the same global seed produces the same mutation sequence. The `DeathContext` records `mutationSeed` so replay can reproduce exact mutation outcomes.

12. **Death→mutation→respawn plumbing** — The plumbing lives in `display.worker.sim.utils.ts:runSimStep`, which already handles death detection and respawn. Add: (a) on enemy death, call `evolveEnemyOnDeath(variantId, fitnessRecord, mutationSeed)` → returns new weights + new `variantId`; (b) respawn with the new weights and `variantId`. The existing `selectVariant` from `select.ts:39-56` (deterministic elitist max-selection, returns the fittest `Individual`) is reserved for champion extraction at wave-sync — NOT for per-death parent selection. The controller's `spawn.utils.ts:74` hardcodes `variantId=0` (separate issue from `selectVariant`). For per-death parent selection, use the new `selectParentProportional` (item 2b).

13. **variantId↔population-sample mapping** — `variantId` is the index into the enemy population array. At respawn, `selectParentProportional(population, fitnessRecords, mutationSeed)` returns a variantId via seeded tournament/roulette selection (uniform fallback when `Σfitness ≤ 0`). `selectVariant` is reserved for champion extraction at wave-sync (items 2b, 12). The variantId persists across the enemy's lifetime, accumulates fitness, and is used for parent selection on death.

### Step A5: Raycasting — Fix Critical Rendering Bugs

**Priority:** P0  
**Severity:** Critical (visual + correctness)  
**Source agent:** raycasting-impl  
**Files:** Multiple — see below

#### Problem

1. **CPU tier renders blank screen** — `browser-entry.ts:117` selects CPU tier but host frame consumer only reads scalar HUD fields. Packed frame geometry arrays (`wallDistances`, `zBuffer`, `wallSides`, `enemyScreenX`, `enemyScale`, `projectileScreenX`) are NEVER read in production. CPU fallback is a blank screen with HUD.

2. **GPU tier is completely absent** — `RENDER_TIER_GPU` defined but never selected. `types/webgpu-shim.d.ts` is explicit: "the neatenstein example never instantiates or calls WebGPU APIs." All GPU types are `any` stubs.

3. **"Linear" fog is actually a hard binary cliff** — `resolveWallFogFactor` returns `perpWallDist >= CAP ? 1 : 0`. JSDoc claims "linearly interpolates" but it's binary. Every surface is full color until exactly 30 cells, then fully background-colored. No gradual distance fog anywhere.

4. **Translucent sprite pixels overwrite instead of blend** — `putImageData` writes raw pixels (no alpha-composite). Semi-transparent sprite edges replace wall pixels, producing hard/incorrect edges.

5. **Sprites anchored on horizon, not floor** — Enemies float at eye level rather than standing on the floor.

6. **DDA out-of-bounds read** — `raycast.ts:149` reads `flatMap[mapY * side + mapX]` with no bounds check. `undefined !== 0` is true → false wall hit with garbage coordinates.

7. **Traveling bolts not z-buffer tested** — Bolts drawn on top of walls (no depth test against z-buffer).

#### Solution

1. **Implement CPU painter OR remove CPU/GPU tiers** — Declare explicitly: the JS DDA raycasting path is retained as the WebGL/WebGPU fallback for devices without shader support. The primary render path becomes a WebGL/WebGPU fragment-shader raycaster (see B3.6). The packed-frame geometry arrays (`wallDistances`, `zBuffer`, `wallSides`) are kept ONLY for the JS fallback path; `enemyScreenX`, `enemyScale`, `projectileScreenX` are consumed by both paths. If the shader path is selected (A5 tier decision), the JS DDA path is still compiled but not called at runtime unless shader initialization fails. For the JS fallback, add **WASM-SIMD acceleration** for the DDA inner loop (128-bit SIMD for sideDistX/Y stepping and map-cell lookup) to keep the fallback performant.

2. **Implement real smooth fog** — `resolveFogFactor(distance) = smoothstep(FOG_START, CAP, distance)` with `FOG_START ~ 0.6 * CAP`. Apply to walls, floor bands, sprite column loop. Update JSDoc to match. In the framebuffer path (A2 Fix 2), fog is per-pixel RGB interpolation — trivial and smooth. In the JS fallback path, fog is applied per-column to the packed RGB triple before write. **Fog coordination (per Invariant §7):** The same `smoothstep(FOG_START, CAP, d)` factor MUST be applied identically to wall color fog, floor/ceiling grid color fog, AND floor/ceiling grid alpha. The existing 4-band alpha falloff (`floor.shade.utils.ts:185-193`) MUST be folded into the single fog factor — do NOT multiply alpha-band falloff on top of color fog, or the grid vanishes before walls, making walls appear to float off the grid. Verify `FOG_START`/`CAP` are identical across all three consumers. Add a visual check that a wall at distance `0.6·CAP…CAP` still has a visible grid line meeting its base.

3. **Fix sprite compositing** — Manually src-over blend sprite pixels into `ImageData` framebuffer: `out = src * alpha + dst * (1 - alpha)`, write `255` to framebuffer alpha. One flush at end of all sprites.

4. **Fix sprite floor anchoring** — Compute floor-projected Y at sprite's perpendicular distance: `drawEnd = horizonY + (focalLength * cameraHeight / perpDist)`, `drawStart = drawEnd - scale`.

5. **Fix DDA bounds check (BOTH entry points)** — Guard before map read in BOTH DDA entry points: `castRayDDAFromFlatMap` (`raycast.ts:149`) AND `hasLineOfSight` (`raycast.ts:197-224`, which delegates to `castRayDDAFromFlatMap` — the bounds check at 149 covers both transitively, but `hasLineOfSight` should also guard its own map reads for defense-in-depth). The guard: `if (mapX < 0 || mapY < 0 || mapX >= side || mapY >= side) { return no-hit }`. Also fix the **step-cap inequivalence** — the DDA loop caps at 30 grid-cell crossings (`steps >= NEATENSTEIN_RENDER_DISTANCE_CAP` at `raycast.ts:169`), while fog caps at 30 units of perpendicular distance. At oblique angles (e.g., 45° where `dirX = dirY ≈ 0.707`), each DDA step covers `deltaDist ≈ 1.414` Euclidean units, but only ~21 units of perpendicular distance after 30 steps. Walls between 21–30 units at 45° pop out of existence. Fix: **split the overloaded constant** (per Invariant §6) into `NEATENSTEIN_DDA_MAX_STEPS` (for the DDA step budget) and `NEATENSTEIN_RENDER_DISTANCE_CAP` (30, unchanged, for floor cull + fog). Replace the static 30-step cap with an angle-aware cap using the new step-budget constant: `const maxSteps = Math.ceil(NEATENSTEIN_RENDER_DISTANCE_CAP / Math.min(Math.abs(dirX), Math.abs(dirY)))`, or conservatively raise the static step cap to `Math.ceil(NEATENSTEIN_RENDER_DISTANCE_CAP * Math.SQRT2)` ≈ 43 steps. **The floor/ceiling grid cull stays at `NEATENSTEIN_RENDER_DISTANCE_CAP = 30` — NO change needed for the angle-aware option (a).** Only the conservative static-raise option (b) would raise the effective perpendicular cap and require the floor cull to rise in lockstep. Option (a) is recommended.

6. **Thread zBuffer into bolt rendering** — Pass `zBuffer` to `renderPlayerBolt`/`renderEnemyBolt`, call `depthTestPulse` before drawing. This depends on B3.3 (sentinel unification) — bolts must compare against the same z-buffer values as sprites. Order: B3.3 sentinel unification must complete before this fix.

7. **Gun sprite decode cache** — Add `Map<EncodedGunSpriteFrame, VoxelSnapshot>` cache mirroring `decodedRobotSpriteCache`. Draw with single `putImageData` instead of per-pixel `fillRect`.

8. **Unify sentinel AND comparison operators** — Route both z-buffer paths through `fillNeatensteinZBuffer`, use `Infinity` consistently (not `30`). Additionally unify the comparison operator: `clipNeatensteinSpriteSpan` (`zbuffer.ts:261`) uses `spriteDistance < zBuffer[column]` (strict — wall wins ties) while `depthTestPulse` (`pulse.ts:232`) uses `pulse.distance <= zBuffer[column]` (inclusive — pulse wins ties). Standardize on strict `<` in both functions. **Behavior change:** pulses/impacts at exact wall distance become occluded (previously visible via `<=`). This is intentional — wall wins ties. Impact: spots and ammo pickups currently rendering at exactly the wall boundary will disappear.

9. **Clean up dead code** — Remove `gun-sprite.ts` stub, `voxel-gun.ts` tombstone. If shader path selected: remove dead `writeNeonWallColumn` and `NEATENSTEIN_WORKER_COLUMN_COUNT = 480` constant. If JS fallback retained: keep `writeNeonWallColumn` for fallback path.

10. **Flag `transferToImageBitmap` contract change** — C1.5 proposes `transferToImageBitmap` instead of `commit()`. This changes the host-side frame consumption contract: the host must call `createImageBitmap(transferredBitmap)` instead of reading `OffscreenCanvas` directly. The `browser-entry.ts` frame consumer must be updated in tandem. Flag this as a coordinated change between worker and host.

---

## Phase B — High-Priority Architectural Improvements

### Step B1: Enemy AI Parallelism — Independent Workers

**Priority:** P1  
**Severity:** High  
**Source agents:** enemy-parallelism, algorithm-research  
**Files:** `browser-entry/worker/display.worker.ts:576-587`, `scripts/enemy-controller.ts:264-279`, `browser-entry/worker/display.worker.sim.utils.ts:230-237`

#### Problem

Enemies are structurally unable to be "fully independent AIs":
- Single worker owns BOTH simulation AND rendering serially
- `enemy-controller.ts:264-279`: Sequential `for(i=0..n)` loop over ≤8 enemies. Runs TWICE per tick (redundant zero-timestep pass)
- One `enemyWeights` resolved per tick, passed to ALL enemies
- `NEATENSTEIN_ENEMY_MAX_CONCURRENT = 8` hardcoded
- `openInferenceChannel` + `exportTransferableInferencePayload` exist in repo but Neatenstein uses ZERO of it
- `SharedInferenceWorker` with SharedArrayBuffer also available
- `ParallelInferencePool` available for batching

FlappyBird comparison: each bird gets own `InferenceChannel` worker but live playback is serial (`await` inside loop). True parallelism only in offline evaluation.

#### Solution

Migration order:
1. **Drop ParallelInferencePool as real-time path** — `ParallelInferencePool` is designed for offline batch evaluation, not real-time per-agent inference. It has no per-agent recurrent state persistence — each inference call is stateless, which is correct for feed-forward MLPs but breaks for any recurrent topology (NGE GatedRecurrentCell, EpisodicSlot). For real-time enemy AI, use `SharedInferenceWorker` with `SharedArrayBuffer` (SAB) as the PRIMARY path. The SAB pool scales to dozens of agents from one worker with zero message-passing overhead.

2. **SAB pool is PRIMARY, InferenceChannel is small-N fallback** — For 16-32 enemies, one `SharedInferenceWorker` with SAB-backed weight slots is the primary inference path. Each enemy owns a weight slot in the SAB. The `InferenceChannel` (one-channel-per-enemy) is the fallback for small N (≤8) or when SharedArrayBuffer is unavailable (cross-origin isolation not configured). This is a tiered approach: SAB → InferenceChannel → inline activation.

3. **Determinism preservation under async inference** — Async inference introduces nondeterministic completion ordering. Preserve determinism via: (a) each enemy's inference result is tagged with `(simTick, enemyIndex)`; (b) the sim tick does not advance until ALL enemy inference results for that tick are collected (barrier); (c) results are applied in `enemyIndex` order, not completion order. This ensures the same seed produces the same simulation regardless of inference latency.

4. **Zero-timestep pass (reconciled with A2)** — The second controller pass is NOT flatly removable. A2 Fix 4 specifies: reuse the distance map from pass 1, conditionally skip when no de-rez occurred. B1 adopts this same approach — the pass stays but with reused distance map and conditional skip.

5. **Offload champion to InferenceChannel** — Use existing `openInferenceChannel` + `exportTransferableInferencePayload` for the main agent network activation, freeing the display worker thread. Configure `maxConcurrentRequests` to match the burst pattern (1 request per tick, so `maxConcurrentRequests: 2` is sufficient for double-buffering).

6. **Split sim/render workers** — Simulation worker owns game state + enemy AI. Render worker owns DDA + framebuffer + canvas. Communicate via `SharedArrayBuffer` (avoids transfer cost). Sim writes enemy state into shared memory; render reads it. **OffscreenCanvas ownership:** the render worker owns the `OffscreenCanvas` and its WebGL/WebGPU context. The sim worker never touches canvas. The `transferControlToOffscreen()` call happens in the render worker's init. **Map grid sharing contract (per Invariant §1):** The 120×120 `Uint8Array` map (the shared grid basis for both wall DDA and floor projection) MUST be made available to the render worker — either via SharedArrayBuffer or transfer. The plan must explicitly state that the map grid is shared, not just enemy positions. Without the map, the render worker cannot do wall DDA or floor projection, and alignment breaks.

7. **Give each enemy own genome** — Each enemy gets its own weight set (from A4's per-variant population). Activate all in parallel via SAB pool. With 16-32 enemies, SAB pool handles all in one inference batch. **Genome source contract with A4:** A4's `selectVariant` produces per-enemy weight vectors; B1's SAB pool loads them into SAB slots. The contract is: A4 produces `Float32Array[]` (one per enemy), B1 copies them into SAB weight slots at respawn time.

8. **Eval-worker boundary** — `eval.worker.ts` stays OFFLINE-ONLY. It is never called during live gameplay. It runs NEAT evaluation for the main agent during interactive training sessions. The display worker delegates to it via `postMessage` for batch evaluation, not real-time inference.

9. **Scale enemy count** — With SAB parallel inference, the 8-enemy cap can be raised. Target 16-32 enemies. Simulation cost scales with enemy count (BFS per enemy for sensor extraction), but A2 Fix 3 (shared flow-field) mitigates this — all enemies targeting the same player cell share one distance map.

### Step B2: Code Quality — Architecture and Debt

**Priority:** P1  
**Severity:** High  
**Source agent:** code-quality-review  
**Files:** Multiple — see below

#### Problem

1. **`display.worker.ts` is a 901-line god-file** — 34% is test infrastructure (21 `__testOnly*` exports). Mixes worker lifecycle, tier routing, packed-frame build, eval-worker delegation, champion deserialization.

2. **`scripts/ ↔ browser-entry/` boundary is leaky** — Bidirectional dependency: `scripts/` imports from `browser-entry/` in 16 files; `browser-entry/` imports from `scripts/` in 7 files. `scripts/enemy-controller.move.utils.ts:22` imports `activateMlp` from `browser-entry/harness/enemy-mlp` — runtime dependency from "script" to harness.

3. **130 re-export facade blocks, only 18 `@deprecated`** — Inconsistent policy. The pattern "extract to utils, re-export from facade" is applied universally but deprecated markers are only on harness files.

4. **Duplicated helpers** — 5 `clamp` implementations, 3 `isFiniteNumber` variants, 2 `isPositiveFiniteDimension` copies.

5. **`display.worker.test.ts` is 4543 lines** — Monolith test file.

#### Solution

1. **Introduce `shared/` (or `core/`) layer** — Move `scripts/enemy-controller*`, `enemy-navigation*`, `enemy-sprite*`, `voxel-enemy*`, `snapshot-renderer*` into `shared/` along with shared constants/types. Both `scripts/` and `browser-entry/` depend only downward into `shared/`. Eliminates bidirectional boundary. **Path consistency:** Use `browser-entry/shared/` (not top-level `shared/`) to avoid creating a third top-level directory. The existing `browser-entry/shared/` subdirectory is the target. `isPositiveIntegerDimension` will live in `browser-entry/shared/math-guards.utils.ts` (to be created) — it is the consolidation target, not a separate file.

2. **Extract `display.worker.ts` test hooks** — Move 21 `__testOnly*` exports to `display.worker.test-hooks.ts`. Extract `display.worker.message-handler.utils.ts` + `display.worker.eval-delegation.utils.ts`. **`__testOnlyInjectTestEnemies`** currently mutates the worker's enemy array in place — in the extracted test-hooks file, it must return a new array (not mutate), preserving immutability boundary. **SOLID pattern preservation:** The extracted files follow the existing orchestrator/executor pattern — `display.worker.message-handler.ts` (orchestrator, no suffix, complexity ≤ 10) calling `display.worker.message-handler.utils.ts` (executor, `.utils.ts` suffix, complexity ≤ 5), matching the existing `tick.ts` / `tick.*.utils.ts` convention. **Render compositing order (per Invariant §5):** The extraction MUST preserve the render compositing order: floor → ceiling → walls → sprites → pulses/sparks → bolts. A careless refactor could silently reorder draw calls, breaking the wall-base/floor-line junction. Add a comment block in the extracted render module declaring this order as a non-negotiable invariant.

3. **Adopt one re-export policy** — Recommend option (b): accept the facade re-export pattern as permanent architecture (it serves the folder-based module standard) and strip ALL 18 `@deprecated` markers. Rationale: all 18 markers are on re-export blocks (verified: all say "Import from `./types` instead. This re-export preserves the public API"), not API-change markers. Under option (b), they should be stripped along with the facade being accepted as permanent. Keeping them would perpetuate the original inconsistency.

4. **Consolidate duplicated helpers** — Create `browser-entry/shared/math-guards.utils.ts` exporting `clamp`, `clamp01`, `clampInt`, `clampByte`, `isFiniteNumber`, `isPositiveFinite`, `isPositiveFiniteDimension`. Remove 5+3+2 duplicates. **Note:** `isPositiveIntegerDimension` is already in this file — ensure all duplicates across the codebase import from here, not from local copies.

5. **Split `display.worker.test.ts`** — Into `display.worker.init.test.ts`, `display.worker.sim.test.ts`, `display.worker.render.test.ts`, `display.worker.eval-delegation.test.ts`, `display.worker.auto-ai.test.ts`.

6. **Quick wins:**
   - Fix copy-paste JSDoc in `display.worker.sim.utils.ts:48` — the summary for `createDisplayWorkerState` was copied from `resolveEnemyWeights` (line 86, same file). Replace `createDisplayWorkerState`'s summary with "Factory for the encapsulated display-worker mutable state object." Keep `resolveEnemyWeights`'s summary at line 86 as-is.
   - Replace `16` with `NEATENSTEIN_FIXED_TIMESTEP_MS` in `harness/constants.ts:41`
   - Make `fireGateState` return new state instead of mutating in place

### Step B3: Raycasting — Quality Improvements

**Priority:** P1  
**Severity:** Medium-High  
**Source agent:** raycasting-impl

#### Problem

Additional rendering quality issues beyond the critical ones in A5:

1. **No wall texture mapping** — Walls are flat-shaded single-color stripes
2. **Floor is line-projection approximation, not per-pixel casting** — 80-sample grid lines, no floor texture
3. **Only 4 floor alpha bands** → visible depth stepping
4. **Sprite single perpDist for whole span** — No per-column depth (edge occlusion artifacts)
5. **Per-sprite `putImageData` in `renderNeatensteinSprite`** — O(N·sprite) putImageData storm if real context used
6. **Inconsistent z-buffer empty sentinel** — `Infinity` vs `30` in different code paths
7. **NaN from `0 * Infinity` on grid-line-aligned axis rays**
8. **Bolts are 2D screen-space glow dots, not 3D projectiles**

#### Solution

1. **Add wall texture mapping** — Compute `wallX` texcoord, sample texture atlas per column. The framebuffer write loop already exists. **Aesthetic preservation (per Invariant §3):** Wall surfaces MUST remain flat-shaded neon-dominant (the `0,183,255`/`0,164,229` orientation-tinted walls are integral to the neon identity). If textures are introduced, restrict them to subtle edge/normal accents that preserve the saturated flat color as the primary read. Do NOT shift toward photoreal retro textures — the signature neon look is non-negotiable.
2. **Derive floor alpha from unified fog factor** — Per Invariant §7, per-segment alpha is derived from the single `smoothstep(FOG_START, CAP, d)` fog factor, superseding the band-count increase. Do NOT increase from 4 to 8-16 bands; the fog factor replaces the band system entirely.
3. **Unify z-buffer sentinel** — Route both paths through `fillNeatensteinZBuffer`, use `Infinity` consistently. Additionally unify comparison operators to strict `<` (see A5 item 8).
4. **Fix NaN edge case** — Initialize `sideDistX/Y` with `Number.POSITIVE_INFINITY` guard for near-zero direction.
5. **Remove per-sprite `putImageData`** — Make caller responsible for single flush.
6. **WebGL/WebGPU fragment-shader raycasting as primary path** — A WGSL/GLSL fragment shader can do DDA per-pixel, eliminating the JS column loop. The existing "GPU tier" slot is empty — this fills it. **Shader approach:** Use a compute shader (WGSL) for DDA ray marching (compute per-pixel wall distance + texcoord), then a fragment shader for texturing + fog. This is a compute+fragment pipeline, NOT a single shader. **WebGL fallback:** If WebGPU is unavailable, use a WebGL2 fragment shader with the DDA loop in GLSL. If WebGL2 is unavailable, fall back to the JS DDA path (A5 item 1). Tier selection: WebGPU compute → WebGL2 fragment → JS DDA (with WASM-SIMD if available).

**Shader alignment contract (per Invariants §1, §2, §3):**
- **Single canonical projection:** The wall DDA compute shader, the wall texturing fragment shader, the floor/ceiling per-pixel caster, and the pulse/spark projector MUST all consume a single shared camera uniform struct containing: `focalLength`, `planeScale`, `cameraDirection`, `cameraPlane`, `cameraHeight = NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD`, `horizon = 0.5·H`, and `NEATENSTEIN_RENDER_DISTANCE_CAP`. No shader may independently re-derive these values.
- **Identical ray-derivation function:** The `pixelCoord → rayDirection` function MUST be identical (shared WGSL/GLSL function or verbatim-equivalent inline) across the wall DDA, wall texturing, and floor/ceiling caster shaders, including the pixel-center convention (e.g., `uv = (pixelCoord + 0.5) / dimensions`; do NOT mix `uv*2-1` with `(uv-0.5)*2`). Sub-pixel drift between wall edge and floor grid line persists if the ray-derivation differs even with shared uniforms.
- **DDA replication:** The shader DDA MUST use the identical perpendicular distance formula (`perpWallDist = (mapX - posX + (1-stepX)/2)/dirX`), NOT Euclidean distance. Using normalized ray directions would introduce fish-eye and break wall heights. The DDA MUST step integer map cells of 1 world unit.
- **Precision:** WebGL2 paths MUST mandate `highp` float precision for all DDA/projection math. Cross-GPU bit-exactness is unattainable; instead define an alignment tolerance (wall edge ↔ nearest floor grid line ≤ 0.5px at all depths) and add a regression test asserting it.
- **JS/WASM-SIMD fallback acknowledgment:** The per-column JS DDA quantizes wall edges to column boundaries; the shader paths produce sub-pixel-continuous edges. This means the JS fallback cannot produce pixel-identical wall positions to the shader paths. Document this: switching tiers may cause ≤1px wall-edge popping. The floor grid regression test (Invariant §8) must pass within tolerance for all tiers.
- **Traveling spark shader replication (per Invariant §3):** The ambient pulse (traveling spark) MUST be replicated inside the shader/overlay pipeline. Today it is a Canvas 2D overlay projected via `projectNeatensteinFloorPoint`. Once the floor grid moves to GPU per-pixel casting, a Canvas 2D-projected pulse will drift off the shader-rendered grid lines, and its z-buffer test will reference a non-existent JS z-buffer. The shader pipeline MUST either: **(a) [PREFERRED]** render the pulse in the same fragment shader using the unified projection + shader-side depth buffer, or **(b) [FALLBACK ONLY]** read back the shader z-buffer/depth and re-project the pulse with the identical projection the shaders use — note (b) stalls the GPU pipeline and should only be used if (a) is infeasible. The pulse's per-instance world-line state (axis, position-along-line, direction, lifetime alpha) MUST be passed into the shader via a uniform array capped at `NEATENSTEIN_PULSE_MAX_CONCURRENT`. The spark's depth-test operator MUST use strict `<` (per A5 item 8) — wall wins ties.
- **Double-stroke neon glow replication:** The current floor grid uses a canvas-specific double-stroke technique (3px halo at `alpha × 0.35` + 1px bright core, `floor.shade.utils.ts:196-214`). The shader path MUST reimplement this glow effect in-shader (e.g., distance-field-based line rendering with glow falloff) or the neon aesthetic will regress. Treat the shader port as a re-art-directed port, not a drop-in. The CPU per-pixel caster (B3.7) MUST also replicate the halo glow (`NEATENSTEIN_FLOOR_GLOW_WIDTH_PX = 3` + 1px core) to avoid aesthetic regression in the JS fallback tier.
7. **Per-pixel floor casting** — Make this unconditional (not conditional on whether textured floor is desired). Per-pixel floor casting eliminates the 80-sample grid line approximation and the 4-band alpha stepping. Standard per-row floor casting: for each screen row below horizon, compute floor distance, step across columns, sample floor color + apply fog. **Procedural integer grid mandate (per Invariant §4):** The floor MUST render a **procedural world-space integer grid** computed via `fract(worldCoord)` or equivalent — NOT an arbitrary sampled texture. Grid line spacing MUST equal exactly 1 world unit = 1 map cell. The per-pixel caster MUST derive `rowDistance` from the exact same `NEATENSTEIN_FLOOR_*` constants (`FOV`, `cameraHeight`, `horizon`, `focalLength`) as the current line-projection path — do NOT re-derive `focalLength` independently (e.g., using `W/2/tan(hFOV/2)` instead of `H/2/tan(vFOV/2)`). The current line-projection (`floor.projection.utils.ts`) is the numerical source of truth and must be the reference implementation. Prefer casting at full row resolution and detecting grid-line crossings by world-coordinate modulus, so the grid line's screen position is a byproduct of the same per-row walk, not a second projection. **Formulation choice (do NOT mix):** Either (i) **ray-direction interpolation** with `rayDir(px) = dir + plane·offset(px)` (planeScale in rayDir, no focalLength on X) — matching the wall DDA's `castColumnRay` convention — or (ii) **inverse projection** via `projectNeatensteinGridPoint`'s unit right vector × `focalLength`. The two formulations must not be mixed (unit right vector × scaled plane, or scaled ray dir × focalLength double-counts `planeScale` and breaks the `planeScale·focalLength = halfWidth` identity). Pick one and assert per-pixel `screenX` matches `projectNeatensteinGridPoint` in the regression test (Invariant §8).

### Step B4: Algorithm Upgrades — State-of-the-Art Integration

**Priority:** P1  
**Severity:** High (competitive advantage)  
**Source agent:** algorithm-research

#### Problem

The codebase has sophisticated infrastructure (novelty search, NSGA-II, hybrid Baldwinian/Lamarckian, SharedArrayBuffer+Atomics, WebGPU detection) but key modern algorithms are absent:

1. **No MAP-Elites / Quality-Diversity archive** — Novelty search exists but no structured archive grid
2. **No CMA-ES** — Weight mutation is single-weight/multi-weight/perturb only; no adaptive mutation distribution
3. **No opponent pool** — Arms-race freezes one snapshot; no diverse champion pool
4. **No transition-level experience replay** — Death contexts stored but not used for gradient updates
5. **No per-node evolvable time constants** — Main agent lacks temporal memory
6. **No curriculum-based respawn difficulty** — All enemies equally difficult

#### Solution

1. **MAP-Elites archive for enemies** — 2D grid (10×10) keyed by (aggression, positioning). Replace dominated cells. ~200-300 lines (includes telemetry instrumentation, descriptor computation, archive admission logic, and archive persistence across waves — the initial ~200 line estimate was optimistic). Uses existing `EnemyBehaviorMetrics` type. Behavior descriptor formulas (same as A4 item 3, unified): `aggression = clamp01(damageDealt / (damageDealt + survivalTicks * 0.1))`, `positioning = clamp01(meanDistanceToPlayer / maxMapDistance)`. MAP-Elites and per-death evolution are complementary (see A4 item 4): per-death evolution generates candidates, MAP-Elites curates survivors.

2. **sep-CMA-ES for MLP weight optimization** — Full CMA-ES requires O(n³) eigendecomposition of the n×n covariance matrix (n=90 weights → 8100-element matrix → expensive in browser). Recommend **sep-CMA-ES** (diagonal covariance only, O(n) per generation) for browser feasibility. sep-CMA-ES is complementary to per-death evolution: per-death evolution is the fast local search (single-parent mutation per death), sep-CMA-ES is the slower global search (population-level distribution adaptation every N deaths). They operate on different timescales — no conflict.

3. **Unified league structure (hall-of-fame + opponent pool)** — Replace the separate hall-of-fame (A3.6) and opponent pool (B4.3) concepts with a single **league** (AlphaStar-style). The league contains: (a) current champions, (b) past champions (main exploiters), (c) diverse strategy samples from MAP-Elites. Both the hero and enemies evaluate against league samples. This eliminates the redundancy between A3.6 and B4.3. B4.10's adversarial formula is merged into A3.4 (one formula, defined there).

4. **Transition replay for Lamarckian updates** — Per-enemy transition buffer (~100 transitions/life). On death: run N backprop steps. Write updated weights back (Lamarckian). Existing `trainMlpBackprop` can be reused. `runEpisode` currently creates a fresh `Network` — it must instead use the variant's evolved topology and weights (data plumbing change: pass `variantId` → lookup weights → construct `Network` from evolved state).

5. **Prioritized death replay** — Weight transition replay by death "surprise" (unexpected direction, quick death, high health at death). Uses existing `death-feedback.ts` adaptation signal. The `replayPressure` semantics: `replayPressure = clamp(surpriseScore / maxSurprise, 0, 1)`, used to prioritize which transitions get backpropagated first when the transition budget is limited.

6. **CERL-style shared replay for main agent** — The main agent (hero) also needs a shared replay buffer (CERL-style). All hero variants contribute to and draw from a shared transition buffer. New hero variants warm-start from replay samples. This prevents co-evolutionary collapse (where the hero overfits to the current enemy population and forgets how to fight older strategies).

7. **Per-node evolvable time constants** — From neat-python v2.1. Add temporal memory to main agent via CTRNN formulation. Codebase mapping: (a) `Node` class in `src/architecture/network/node.ts` — add `timeConstant: number` property (default 1.0) AND `state: number` property (persistent internal state, initialized to 0, carries across ticks); (b) `mutation.ts` — add `mutateTimeConstant` that perturbs `timeConstant` by `±N(0, 0.1)`; (c) activation — use **exponential Euler integration**: `state += (input_sum - state) * (dt / timeConstant); output = activation(state)`. The `state` is a persistent per-node field that carries across ticks. `timeConstant = 1.0` means fast response (mostly input-driven), `timeConstant = 10.0` means slow response (accumulates over many ticks, providing temporal memory). Simply multiplying by `timeConstant` would NOT provide temporal memory — the Euler integration is essential; (d) NGE state — include `timeConstant` and `state` in node gene so they survive materialization (A3 item 1).

8. **Curriculum-based respawn difficulty** — Scale enemy capability based on player performance. Uses existing `arms-race.ts` wave logic but replaces RNG metrics with real telemetry.

9. **NSNE-style forgetting prevention** — Maintain curriculum of past enemy snapshots, periodically re-evaluate against them. This is the league structure (item 3) — the league's past champions serve as the forgetting-prevention curriculum.

10. **Novelty search + MAP-Elites integration** — Connect existing novelty search infrastructure (`src/neat/evaluate/novelty/`) to MAP-Elites. Novelty search provides the behavioral distance metric; MAP-Elites provides the archive grid. Integration: novelty score is one of the admission criteria for MAP-Elites cells (a variant can enter a cell either by dominating fitness OR by having high novelty in an empty/neigh cell).

11. **Simulation cost at scale** — With 16-32 enemies, per-enemy BFS for sensor extraction is expensive. Mitigation: A2 Fix 3's shared flow-field (all enemies targeting the same player cell share one distance map). Additionally, `Promise.all` serialization risk when enemies exceed worker count: use bounded concurrency (`Promise.all` with chunking — process in batches of `workerCount`) rather than unbounded `Promise.all`.

12. **Missing references** — Add: "Quality-Diversity for Neural Networks" (Mouret & Clune, 2015), "CMA-ES: sep-CMA-ES" (Ros & Hansen, 2008), "AlphaStar" (Vinyals et al., 2019), "Oja's Rule" (Oja, 1982), "TAME the BALROG" (2024, OpenReview — task-adaptive modular emergent framework, relevant to NGE lifecycle), "Dark Souls NEAT" (2025 — pixel-input NEAT combat, closest academic analog), "ES vs Deep RL" (Wong et al., 2024 — linear policy ES matches deep RL, supports MLP weight-only evolution), "Competitive Co-evolutionary Bandit Learning" (2025 — evolutionary bandit learning in matrix games, relevant to league opponent selection).

---

## Phase C — Polish and Refinement

### Step C1: Rendering Polish

**Priority:** P2  
**Severity:** Medium  
**Source agents:** raycasting-impl, performance-analysis

1. **Per-pixel floor casting (unconditional)** — Standard per-row floor casting: for each screen row below horizon, compute floor distance, step across columns, sample floor color + apply fog. This is no longer conditional — B3.7 makes it unconditional. **Procedural integer grid mandate (per Invariant §4):** The floor MUST render a procedural world-space integer grid at 1-unit spacing — NOT an arbitrary texture. The per-pixel caster MUST reuse the exact `NEATENSTEIN_FLOOR_*` constants from `floor.projection.utils.ts` as the single source of truth. See B3.7 for full requirements.
2. **Segment band splitting** — Split floor segments at band boundaries to avoid straddling seams.
3. **Floor color fog** — Apply smooth fog factor to `foggedRgb` per band (currently alpha-only falloff).
4. **Temporal coherence / half-resolution raycasting** — Render every other column, interpolate the rest. The `interpolate.ts` module already exists. **Half-res alignment safeguards (per Invariant §1):** Interpolate **wall color only**, NEVER the column's screen X — keep every column (even or odd) at its true integer pixel position. The floor/ceiling grid is **exempt from decimation** — it stays at full resolution (do not downsample the stroked grid). The wall layer's column-decimation + interpolation kernel applies **only to wall color interpolation**, while the z-buffer depth is cast at every column. Add a runtime quality toggle to disable half-res when alignment artifacts appear; gate behind a quality setting, not unconditional.
5. **OffscreenCanvas + `transferToImageBitmap`** — Replace `commit()` with `transferToImageBitmap` for 2024-preferred present path. **Coordinated change:** the host-side frame consumer (`browser-entry.ts`) must switch from reading `OffscreenCanvas` directly to calling `createImageBitmap(transferredBitmap)`. See A5 item 10.
6. **TAA / MSAA** — 2× MSAA resolve would clean wall-sprite seams cheaply.

### Step C2: Test Quality Improvements

**Priority:** P2  
**Severity:** Medium  
**Source agent:** code-quality-review

1. **Extract test harness helpers** — `installMockWorkerGlobal`, `sendInitMessage`, `sendSimStateMessage` duplicated across 3 worker test files → `worker-test-harness.utils.ts`.
2. **Replace `Record<string, any>` with `Record<string, unknown>`** — 41+ matches across 10 test files.
3. **Tighten `tick.test.ts:1317` mock** — `(state: any) => state` → proper typed mock.
4. **Add debug logging to catch blocks** — `display.worker.sim.utils.ts:204` and `enemy-controller.move.utils.ts:105` silently mask recurrent failures. Add `console.debug` once-per-tick guard.

### Step C3: Constants and Types Cleanup

**Priority:** P2  
**Severity:** Low-Medium  
**Source agent:** code-quality-review

1. **Fix magic number `16`** — `harness/constants.ts:41` uses literal `16` instead of `NEATENSTEIN_FIXED_TIMESTEP_MS`.
2. **Promote voxel anatomy literals** — `voxel-enemy.ts` inline literals → `voxel-enemy.constants.ts` as named constants.
3. **Remove tombstone files** — `voxel-gun.ts`, `gun-sprite.ts` (empty stubs). Move tombstone notes into replacement files' JSDoc.
4. **Clean up `@deprecated` re-export JSDoc** — Add `@example` migration code to deprecated re-exports.

### Step C4: Interpolation Safety

**Priority:** P2  
**Severity:** Low  
**Source agent:** raycasting-impl

1. **Add `lerpNeatensteinAngle`** — Shortest-arc interpolation sibling to `lerpNeatensteinState`.
2. **Clamp `shouldDissolvePixel` `t`** — `clamp(elapsedMs / durationMs, 0, 1)`, guard `durationMs > 0`.
3. **Don't throw on non-finite snapshot fields** — Log and clamp instead of `TypeError` in render hot path.

### Step C5: Pulse and LCG Hardening

**Priority:** P2  
**Severity:** Low  
**Source agent:** raycasting-impl  

**Note:** The pulse system IS the "traveling spark" effect (per Invariant §3) — small shiny dots that travel along integer floor-grid lines, depth-tested against the per-column z-buffer. This is a signature visual that MUST be preserved across all rendering changes.

1. **Normalize LCG seed** — `state = (((seed + simTick) % MOD) + MOD) % MOD` before first multiply.  
2. **Pool pulse updates** — Mutate pulse objects in place with `active` flag, compact in one pass. Avoid `.map().filter().slice()` triple allocation. Pooling preserves world positions (the spark stays on its integer grid line).  
3. **Use numeric key for team-color cache** — `(r << 16) | (g << 8) | b` instead of string key.

---

## Priority Matrix

| Step | Priority | Impact | Complexity | Dependencies |
|------|----------|--------|------------|--------------|
| A1: Backtracker Maze | P0 | High — transforms gameplay | Medium | None |
| A2: Perf Allocations | P0 | High — fixes sub-60fps risk | Medium | A5 (tier decision for frame arrays) |
| A3: NGE Hero Evolution | P0 | High — core thesis | High | A4 (needs live enemies — see bootstrap order A3 item 9) |
| A4: Enemy NEAT Evolution | P0 | High — core feature | Medium-High | None (A4-core); B1 (A4-advanced: Hebbian) |
| A5: Raycasting Bugs | P0 | High — visual + correctness | Medium | None; B3.3 before bolt z-buffer |
| B1: Enemy Parallelism | P1 | Medium — perf + independence | Medium | A4 (per-enemy genomes) |
| B2: Code Architecture | P1 | Medium — maintainability | Medium | None |
| B3: Raycasting Quality | P1 | Medium — visual polish | Medium | A5 |
| B4: Algorithm Upgrades | P1 | High — competitive advantage | High | A3, A4 |
| C1: Rendering Polish | P2 | Low-Medium — polish | Low-Medium | A5, B3 |
| C2: Test Quality | P2 | Low-Medium — maintainability | Low | B2 |
| C3: Constants Cleanup | P2 | Low — quick wins | Low | None |
| C4: Interpolation Safety | P2 | Low — edge cases | Low | None |
| C5: LCG Hardening | P2 | Low — robustness | Low | None |

---

## Buffer Reuse Summary

| Buffer | Current | Target |
|--------|---------|--------|
| Worker z-buffer | ✅ Reused | Keep |
| BFS queue buffer | ✅ Pooled | Keep |
| BFS distance Int32Array(14400) | ❌ Fresh every call | Module-level reusable, cached per player-cell |
| Frame typed arrays (6) | ❌ Fresh every frame | Pool on worker if JS path kept; remove if shader path (A5 tier decision) |
| Sprite ImageData | ❌ getImageData every frame | Allocate once, reuse as framebuffer |
| Wall framebuffer | ❌ Per-column fillStyle/fillRect | Write RGB to Uint8ClampedArray, single putImageData |
| Floor/ceiling bands + projection | ❌ Fresh every drawGrid | Pool across frames |
| floorCamera literal | ❌ Fresh per frame | Module-level mutable object |
| Sprite sort array | ❌ `[...sprites].sort(...)` clone per frame | Reused scratch array, sort in place |
| MLP activation buffers | ❌ Fresh per activation | Pool 2 per enemy slot |
| Sensor array (22) | ❌ Fresh per extraction | Module-level reusable |
| Raycast hit objects | ❌ Fresh per ray | Preallocated column buffer |
| Decoded robot sprite frames | ✅ Cached | Keep |
| Decoded gun sprite frame | ❌ Fresh every frame | Add decode cache |
| EnemyUpdateContext | ❌ Fresh per enemy per tick | Pool slot contexts |
| EnemyController Map + arrays | ❌ Fresh every tick | Preallocate reusable |
| Game state clones | ❌ Deep clone every tick | Mutable internal state, immutable postMessage output |
| gameTick bolt/impact/pickup arrays | ❌ `.filter().map(clone)` per tick | In-place mutation + single compact pass |
| De-rez pruning chain | ❌ 3 chained `.map().filter().map()` per tick | Single in-place loop + Set, conditional on de-rez |

---

## Research References

- **QDax** (JMLR 2024): https://qdax.readthedocs.io — reference QD library
- **ASCII-ME** (2025): Policy-gradient QD, 5× faster than prior PG-ME
- **FAMOU** (2026): Evaluator co-evolution, opponent pools, weakness pressure
- **COvolve** (2026): LLM co-evolution with MSNE forgetting prevention
- **Constitutional Arms Races** (2026): Coupled fitness (S_own - S_opp) requirement
- **Baldwin Effect in NEAT Chess** (2026): Hebbian plasticity variance crossover
- **neat-python v2.1** (2025): Per-node evolvable time constants, GPU-accelerated CTRNN
- **CERL** (2019): Shared replay buffer + neuroevolution
- **Dominated Novelty Search** (2025): arXiv:2502.00593
- **MEliTA** (2024): MAP-Elites with Transverse Assessment
- **WebGPU compute shaders**: https://webgpufundamentals.org
- **Quality-Diversity for Neural Networks** (Mouret & Clune, 2015): Foundations of QD archives
- **sep-CMA-ES** (Ros & Hansen, 2008): Diagonal covariance CMA-ES for high-dim optimization
- **AlphaStar** (Vinyals et al., 2019): League training, main/exploiter agents, forgetting prevention
- **Oja's Rule** (Oja, 1982): Stabilized Hebbian learning with implicit normalization
- **TAME the BALROG** (2024, OpenReview): Task-adaptive modular emergent framework, NGE lifecycle design
- **Dark Souls NEAT** (2025): Pixel-input NEAT combat, closest academic analog to Neatenstein
- **ES vs Deep RL** (Wong et al., 2024): Linear policy ES matches deep RL, supports MLP weight-only evolution
- **Competitive Co-evolutionary Bandit Learning** (2025): Evolutionary bandit learning in matrix games, league opponent selection

---

## Review Protocol

**Round 1:** All 8 agents reviewed — 0 approvals, 71 observations collected.

**Round 2:** 3 approvals (maze-generation, enemy-parallelism, performance-analysis), 5 agents returned observations (9 remaining issues). All round-2 observations addressed.

**Round 3:** 4 approvals (code-quality-review, nge-hero-evolution, raycasting-impl, algorithm-research), 1 agent returned 1 observation (enemy-neat-evolution). All round-3 observations addressed.

**Round 4:** 1 approval (enemy-neat-evolution). **All 8 agents approved.**

### Final Approval Status

| Agent | Round | Status |
|-------|-------|--------|
| maze-generation | 2 | ✅ APPROVED |
| enemy-parallelism | 2 | ✅ APPROVED |
| performance-analysis | 2 | ✅ APPROVED |
| code-quality-review | 3 | ✅ APPROVED |
| nge-hero-evolution | 3 | ✅ APPROVED |
| raycasting-impl | 3 | ✅ APPROVED |
| algorithm-research | 3 | ✅ APPROVED |
| enemy-neat-evolution | 4 | ✅ APPROVED |

**All 8 specialist agents have approved the plan.**

### Floor-Wall Alignment Specialist Review (Round 1)

8 new specialist agents verified the floor-wall grid alignment critical requirement from multiple angles:

| Agent | Perspective | Key Finding |
|-------|------------|-------------|
| floor-grid-alignment | Wall columns vs floor grid X | Alignment is mathematically exact (`planeScale·focalLength = halfWidth`); B3.6/B3.7 lack projection-reuse contract |
| traveling-spark-effect | Traveling spark preservation | B3.7 would replace visible neon grid with texture; spark never mentioned in plan |
| 3d-depth-perception | 3D depth illusion | Wall base meets floor grid line exactly; smooth fog decouples wall/floor fade; shader must reuse constants |
| artistic-visual-cohesion | Artistic/visual quality | Double-stroke neon glow is signature; per-pixel casting risks generic textured look; wall textures break neon identity |
| raycasting-math-alignment | DDA-floor projection math | Algebraic proof: `screenX_floor = column·stripeWidth = xStart` (exact); B3.6 highest risk (no verbatim replication mandate) |
| perf-vs-visual-artifacts | Perf changes visual artifacts | A2 Fix 1 aliasing risk (HIGH); A2 Fix 2 coverage gaps; smooth fog double-fade; half-res soft edges |
| shader-pipeline-alignment | Shader pipeline alignment | No shared projection specified; traveling spark not replicated in shader; WebGL2 precision unspecified |
| plan-impact-auditor | Cross-reference all plan steps | B3 is HIGH risk; 7 gaps found: no invariant, no spark mention, no texture spacing, no map sharing, no compositing order |

All 20 observations addressed by adding the **Non-Negotiable Invariants** section (8 invariants) and updating A2, A5, B1, B2, B3.6, B3.7, C1.1, C1.4, and C5 with explicit alignment safeguards.

### Floor-Wall Alignment Specialist Review (Round 2)

| Agent | Verdict | Key Observations |
|-------|---------|-----------------|
| floor-grid-alignment | ✅ APPROVED | Tolerance scoping, getImageData allocation, feathered edges (non-blocking) |
| traveling-spark-effect | ✅ APPROVED | Spark readback perf cost, spark↔grid coupling in test, shader depth operator (non-blocking) |
| 3d-depth-perception | ✅ APPROVED | Test X only not Y, tolerance inconsistency, B3.7 CPU caster halo (non-blocking) |
| artistic-visual-cohesion | ✅ APPROVED | Tolerance vs JS quantization, B3.2 vs §7 contradiction, wall texture aesthetic gap, C1.4 wording (non-blocking) |
| perf-vs-visual-artifacts | ✅ APPROVED | Step-cap vs distance-cap conflation, getImageData alpha semantics, C1.4 wording (non-blocking) |
| plan-impact-auditor | ✅ APPROVED | Test X only not Y, tolerance vs JS-fallback tension, C1.4 wording (non-blocking) |
| raycasting-math-alignment | ⚠️ CONDITIONAL | HIGH: step-count vs perpendicular-distance conflation in §6/A5#5; MEDIUM: B3.7 formulation ambiguity; LOW: §8 test wording |
| shader-pipeline-alignment | ⚠️ CONDITIONAL | HIGH: step-count vs distance-cap conflation in §6/A5#5; MEDIUM: §8 tolerance vs quantization; LOW: ray-derivation function, pulse-state plumbing |

6 of 8 APPROVED. 2 CONDITIONAL (both with the same HIGH blocker: step-count ≠ perpendicular-distance cap). Round 3 fixes address all observations from all 8 agents.

### Round 3 Fixes (addressing ALL Round 2 observations)

1. **§6 + A5 item 5 (HIGH):** Split overloaded `NEATENSTEIN_RENDER_DISTANCE_CAP = 30` into `NEATENSTEIN_DDA_MAX_STEPS` (step budget) and `NEATENSTEIN_RENDER_DISTANCE_CAP` (30, unchanged, floor cull + fog). Angle-aware option (a) raises step count only — floor cull needs NO change.
2. **§8 (4 agents):** Added vertical (Y) alignment, spark↔grid coupling, continuous wall-face screen X reference, tier-specific tolerance.
3. **C1.4 (4 agents):** Floor grid exempt from decimation; wall kernel applies only to wall color; z-buffer cast at every column.
4. **B3.2 vs §7:** B3.2 superseded — alpha from unified fog factor, not band increase.
5. **B3.7 formulation:** Two valid formulations specified, MUST NOT mix, assert in regression test.
6. **B3.1 aesthetic:** Walls remain flat-shaded neon-dominant; textures restricted to subtle accents.
7. **B3.7 CPU caster:** Must replicate halo glow.
8. **B3.6 spark:** Option (a) PREFERRED, (b) FALLBACK ONLY; ray-derivation function identical across shaders; pulse-state uniform array; strict `<` depth operator.
9. **A2 Fix 2:** Procedural pixel writes preferred (allocation-free); opaque background fill; feathered fog-wall edges.

### Floor-Wall Alignment Specialist Review (Round 3)

| Agent | Verdict | Notes |
|-------|---------|-------|
| raycasting-math-alignment | ✅ FULLY APPROVED | All 3 observations resolved; verified vertical identity `wall-base screenY ≡ floor point screenY` exactly; 2 trivial non-blocking doc notes |
| shader-pipeline-alignment | ✅ FULLY APPROVED | All 4 observations resolved; 1 non-blocking wording cleanup (§8 horizontal bullet — fixed) |

**All 8 floor-wall alignment specialists have FULLY APPROVED.**

Combined with the original 8 specialists (all approved across 4 rounds), **ALL 16 specialist agents have approved the plan.**