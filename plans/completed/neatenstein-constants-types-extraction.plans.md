# Plan: Neatenstein Constants & Types Extraction

**Created:** 2026-08-15  
**Scope:** Extract all magic strings/numbers into named constants, move constants to dedicated `.constants.ts` files, move type declarations to dedicated `.types.ts` files across `examples/neatenstein/`.  
**Goal:** DRY, SOLID-compliant, no duplicated constants, no inline magic values.  

---

## Mandates

- **Pragmatic mode**: broad slices, one dispatch per phase, follow-ups via `write_agent` to the same idle agent.
- **Parallel execution**: Phases 1–5 can be dispatched in parallel (5 agents, one per layer) since they touch independent file sets. Phase 0 must complete first (shared constants). Phase 6 runs after all others.
- **No RED/GREEN ceremony**: no failing tests first; just extract constants/types, update imports, validate.
- **Validation mandatory per phase**: (1) targeted Jest for affected area, (2) `npx tsc --noEmit` from `examples/neatenstein`, (3) `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`.
- **Model mandate**: ALL agents must use `glm-5.2:cloud`.
- **NEVER run git**: Git is UNINSTALLED. Use edit/create tools only.
- **Update folder-quality-metrics.mjs**: Add `.constants.ts` and `.types.ts` to the UTIL_FILE_SUFFIX skip logic so they're exempt from missing-sibling-test-file check.
- **Preserve public imports**: Main modules must re-export from new `.constants.ts`/.types.ts files so existing imports don't break.
- **No test file changes needed**: This is a pure refactoring (extracting constants/types). Existing tests should pass unchanged. Only update tests if imports break.
- **SOLID best practices**: File-specific constants → `file.constants.ts`; shared category constants → `category.constants.ts` to avoid circular deps. Types always co-located with their primary consumer or in shared category `.types.ts`.
- **Cross-cutting constants home**: All constants needed by multiple layers go into the EXISTING `browser-entry/constants.ts` (the central shared constants hub).

---

## Phase 0: Cross-Cutting Shared Constants

**Owner:** Phase-0 agent  
**Depends on:** Nothing (must complete first)  
**Files modified:** `examples/neatenstein/browser-entry/constants.ts`  

### Step 0.1: Add shared cross-layer constants to `browser-entry/constants.ts`

**Slice 0.1.1 — Animation & snapshot-kind constants**

Add to `examples/neatenstein/browser-entry/constants.ts`:

```typescript
// Animation states — used in scripts + renderer + worker
export const ANIM_STATE_IDLE = 'idle';
export const ANIM_STATE_MOVE = 'move';
export const ANIM_STATE_FIRE = 'fire';
export const ANIM_STATE_DEATH = 'death';
export const ANIM_STATE_DAMAGE = 'damage';
export const ANIMATION_STATES = ['idle', 'move', 'fire', 'death', 'damage'] as const;

// Snapshot kinds — used in harness + worker + entry
export const SNAPSHOT_KIND_MLP = 'mlp';
export const SNAPSHOT_KIND_SWARM = 'swarm';

// Tick input source — used in worker + host
export const TICK_INPUT_SOURCE_AUTO = 'auto';
export const TICK_INPUT_SOURCE_HUMAN = 'human';

// Canvas context
export const NEATENSTEIN_CANVAS_2D_CONTEXT = '2d';

// Time conversion
export const NEATENSTEIN_MS_PER_SECOND = 1000;

// Math constants
export const FULL_CIRCLE_RADIANS = Math.PI * 2;
export const HALF_ROTATION_RADIANS = Math.PI;

// RGBA
export const RGBA_CHANNELS = 4;
export const RGBA_OPAQUE_ALPHA = 255;

// Epsilon
export const NEATENSTEIN_EPSILON_1E9 = 1e-9;
export const NEATENSTEIN_INVISIBLE_SENTINEL = -1;

// Render tiers
export const RENDER_TIER_WORKER = 'worker';
export const RENDER_TIER_CPU = 'cpu';
export const RENDER_TIER_GPU = 'gpu';

// Worker message types — shared between worker + host/renderer-bridge
export const WORKER_MSG_INIT = 'init';
export const WORKER_MSG_RESIZE = 'resize';
export const WORKER_MSG_SIM_STATE = 'simState';
export const WORKER_MSG_INITIALIZED = 'initialized';
export const WORKER_MSG_FRAME = 'frame';

// Eval message types — shared between display.worker + eval.worker
export const EVAL_MSG_EVALUATE = 'evaluate';
export const EVAL_MSG_EVAL_COMPLETE = 'evalComplete';

// Worker bundle filenames
export const NEATENSTEIN_WORKER_BUNDLE = 'neatenstein.worker.js';
export const NEATENSTEIN_EVAL_WORKER_BUNDLE = 'neatenstein.eval-worker.js';

// Adaptation directions — shared: render-loop.utils + death-feedback.ts
export const ADAPTATION_STRONGER = 'stronger';
export const ADAPTATION_WEAKER = 'weaker';
export const ADAPTATION_SHIFTED = 'shifted';

// DNA prefix
export const SWARM_DNA_PREFIX = 'swarm:';

// Hex prefix and separators
export const HEX_PREFIX = '#';
export const CSV_SEPARATOR = ',';
export const CACHE_KEY_SEPARATOR = ':';
```

**Validation:** `npx tsc --noEmit` from `examples/neatenstein`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 1: scripts/ Layer

**Owner:** Phase-1 agent  
**Depends on:** Phase 0 complete (shared constants available)  
**Base path:** `examples/neatenstein/scripts/`  

### Step 1.1: enemy-animator + enemy-controller constants & types

**Slice 1.1.1 — Create `enemy-animator.constants.ts` and `enemy-animator.types.ts`**

Create `examples/neatenstein/scripts/enemy-animator.constants.ts`:
- Animation state string constants (import shared from `browser-entry/constants.ts`): `ANIM_STATE_IDLE`, `ANIM_STATE_MOVE`, `ANIM_STATE_FIRE`, `ANIM_STATE_DEATH`, `ANIM_STATE_DAMAGE`
- Frame size `128` → `ENEMY_FRAME_SIZE_PX = 128`
- Reference size `192` → `ENEMY_REFERENCE_SIZE_PX = 192`
- Animation timing constants if any inline

Create `examples/neatenstein/scripts/enemy-animator.types.ts`:
- `EnemyAnimationState` (type union of animation state strings)
- `EnemyAnimationFrame` (interface)

Update `enemy-animator.ts` and `enemy-animator.utils.ts` to import from new files; re-export from main module.

**Slice 1.1.2 — Create `enemy-controller.constants.ts` and `enemy-controller.types.ts`**

Create `examples/neatenstein/scripts/enemy-controller.constants.ts`:
- `PREVIOUS_STEP_DISTANCE_SENTINEL = -1` (used 5+ files)
- `CELL_CENTER_OFFSET = 0.5` (used 20+ times in move.utils)
- Direction labels: `'N'`, `'NW'`, `'W'`, `'SW'`, `'S'`, `'SE'`, `'E'`, `'NE'` → `DIR_N`, `DIR_NW`, etc.
- `DIRECTIONS` array (consolidate from enemy-navigation.ts + enemy-controller.move.utils.ts — single source of truth here)
- `NUM_DIRECTIONS = 8`

Create `examples/neatenstein/scripts/enemy-controller.types.ts`:
- `ControlledEnemy`
- `HitscanEvent`
- `EnemyControllerState`
- `EnemyUpdateContext`

Update `enemy-controller.ts` and all `enemy-controller.*.utils.ts` to import from new files; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/scripts/enemy-animator|enemy-controller`; `npx tsc --noEmit`

### Step 1.2: enemy-navigation + enemy-sprite constants & types

**Slice 1.2.1 — Create `enemy-navigation.constants.ts` and `enemy-navigation.types.ts`**

Create `examples/neatenstein/scripts/enemy-navigation.constants.ts`:
- `NUM_DIRECTIONS = 8` (import from controller constants if shared, or define here if controller not yet done)
- Sensor indices `0`–`21` → `SENSOR_INDEX_*` named constants
- `PREVIOUS_STEP_DISTANCE_SENTINEL = -1` (import from controller constants)
- `CELL_CENTER_OFFSET = 0.5` (import from controller constants)

Create `examples/neatenstein/scripts/enemy-navigation.types.ts`:
- `DistanceMap`
- `NavigationStep`

Update `enemy-navigation.ts` and `enemy-navigation.utils.ts`; re-export.

**Slice 1.2.2 — Create `enemy-sprite.constants.ts` and `enemy-sprite.types.ts`**

Create `examples/neatenstein/scripts/enemy-sprite.constants.ts`:
- `ENEMY_SPRITE_STATES` array (consolidate from generate-enemy-sprites.ts + enemy-sprite.utils.ts)
- `ENEMY_SPRITE_DIRECTIONS` array (consolidate from 3 files)
- `ENEMY_FRAME_SIZE_PX = 128` (import from animator constants)
- `ENEMY_REFERENCE_SIZE_PX = 192` (import from animator constants)
- Sprite atlas filename `'enemy-sprite-atlas.png'` → `ENEMY_SPRITE_ATLAS_FILENAME`
- Manifest filename `'enemy-sprite-manifest.json'` → `ENEMY_SPRITE_MANIFEST_FILENAME`

Create `examples/neatenstein/scripts/enemy-sprite.types.ts`:
- `NeatensteinEnemyCamera`
- `NeatensteinEnemyProjection`
- `NeatensteinDirectionalLight`
- `NeatensteinEnemyBillboard`
- `NeatensteinSpriteRenderContext`
- `NeatensteinSpriteAtlas`

Update `enemy-sprite.ts` and `enemy-sprite.utils.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/scripts/enemy-navigation|enemy-sprite`; `npx tsc --noEmit`

### Step 1.3: generate-enemy-sprites + snapshot-renderer constants & types

**Slice 1.3.1 — Create `generate-enemy-sprites.constants.ts` and `generate-enemy-sprites.types.ts`**

Create `examples/neatenstein/scripts/generate-enemy-sprites.constants.ts`:
- PNG chunk types: `PNG_CHUNK_IHDR = 'IHDR'`, `PNG_CHUNK_IDAT = 'IDAT'`, `PNG_CHUNK_IEND = 'IEND'`
- `RGBA_CHANNELS = 4` (import from browser-entry/constants.ts)
- `ENEMY_FRAME_SIZE_PX = 128` (import from animator constants)
- `ENEMY_REFERENCE_SIZE_PX = 192` (import from animator constants)
- Direction labels (import from controller constants)

Create `examples/neatenstein/scripts/generate-enemy-sprites.types.ts`:
- `DecodedPng`
- `SpriteSheetOptions`
- `SpriteSheetResult`
- `SpriteFrameDescriptor`
- `ReferenceSnapshotResult`
- `SnapshotComparison`

Update `generate-enemy-sprites.ts` and utils; re-export.

**Slice 1.3.2 — Create `snapshot-renderer.constants.ts` and `snapshot-renderer.types.ts`**

Create `examples/neatenstein/scripts/snapshot-renderer.constants.ts`:
- `ENEMY_FRAME_SIZE_PX = 128` (import)
- `NUM_DIRECTIONS = 8` (import)
- Camera angle `45` → `SNAPSHOT_CAMERA_ANGLE_DEG = 45`
- Scale `4` → `SNAPSHOT_SCALE = 4`
- Alpha values `0.35`, `0.55` → `SNAPSHOT_ALPHA_DARK = 0.35`, `SNAPSHOT_ALPHA_LIGHT = 0.55`
- Rotation `180` → `SNAPSHOT_ROTATION_DEG = 180`
- `255` → `RGBA_OPAQUE_ALPHA` (import)
- Outline alpha `0.08` → `SNAPSHOT_OUTLINE_ALPHA = 0.08`
- Outline width `6` → `SNAPSHOT_OUTLINE_WIDTH_PX = 6`

Create `examples/neatenstein/scripts/snapshot-renderer.types.ts`:
- `VoxelSnapshot`
- `SnapshotOptions`

Update `snapshot-renderer.ts` and utils; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/scripts/generate-enemy-sprites|snapshot-renderer`; `npx tsc --noEmit`

### Step 1.4: voxel-enemy constants & types + png.utils constants

**Slice 1.4.1 — Create `voxel-enemy.constants.ts` and `voxel-enemy.types.ts`**

Create `examples/neatenstein/scripts/voxel-enemy.constants.ts`:
- Part names: `PART_HEAD = 'head'`, `PART_TORSO = 'torso'`, `PART_ARMS = 'arms'`, `PART_LEGS = 'legs'`, `PART_CANNON = 'cannon'`, `PART_BACK_DISK = 'back disk'`
- Material slots: `MATERIAL_ACCENT = 'accent'`, `MATERIAL_NEON = 'neon'`, `MATERIAL_SUIT = 'suit'`, `MATERIAL_DARK = 'dark'`, `MATERIAL_DAMAGE = 'damage'`
- Voxel grid dims: `VOXEL_GRID_WIDTH = 64`, `VOXEL_GRID_HEIGHT = 192`, `VOXEL_GRID_DEPTH = 64`
- ~40 voxel body dimension numbers → named constants (e.g. `VOXEL_HEAD_HEIGHT = ...`, `VOXEL_TORSO_WIDTH = ...`, etc.)

Create `examples/neatenstein/scripts/voxel-enemy.types.ts`:
- `MaterialSlot`
- `VoxelGrid`
- `Voxel`
- `VoxelPalette`

Update `voxel-enemy.ts` and `voxel.utils.ts`; re-export.

**Slice 1.4.2 — Extract PNG/CRC constants into `png.utils.constants.ts` (or add to generate-enemy-sprites.constants.ts)**

Create `examples/neatenstein/scripts/png.utils.constants.ts`:
- `PNG_SIGNATURE` (bytes)
- CRC table constant `0xedb88320` → `PNG_CRC_POLYNOMIAL`
- `PNG_HASH_TABLE_SIZE = 256`
- `PNG_INITIAL_HASH = 8`
- `PNG_HASH_MASK = 0xffffffff`
- `PNG_BYTE_MASK = 0xff`
- Chunk lengths `13`, `8`, `6`, `9` → named constants

Also extract `png.utils` types if any.

Update `png.utils.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/scripts/voxel-enemy|png`; `npx tsc --noEmit`

### Step 1.5: Backprop & curriculum constants (in appropriate utils constants files)

**Slice 1.5.1 — Create `backprop.utils.constants.ts` and curriculum constants**

Create `examples/neatenstein/scripts/backprop.utils.constants.ts`:
- `KNUTH_HASH_MULTIPLIER = 2654435761`
- `LCG_MULTIPLIER = 1103515245`
- `LCG_INCREMENT = 12345`
- `LCG_MASK = 0x7fffffff`
- `BCE_EPSILON = 1e-7`
- `BCE_BATCH_SIZE = 3`
- `EARLY_STOP_LR = 0.001`

Create `examples/neatenstein/scripts/curriculum.utils.constants.ts`:
- ~30 numeric target values → named constants
- 13 soft-target constants
- `CURRICULUM_NUM_SOFT_TARGETS = 17` (if applicable)
- `CURRICULUM_PENALTY_WEIGHT = 2.0`
- `CURRICULUM_REWARD_WEIGHT = 1.0`

Update `backprop.utils.ts` and `curriculum.utils.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/scripts/backprop|curriculum`; `npx tsc --noEmit`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 2: browser-entry/renderer/ Layer

**Owner:** Phase-2 agent  
**Depends on:** Phase 0 complete (shared constants available)  
**Base path:** `examples/neatenstein/browser-entry/renderer/`  

### Step 2.1: Shared renderer constants & sprite/floor constants+types

**Slice 2.1.1 — Create `renderer.sprite.constants.ts` and `renderer.sprite.types.ts` (shared)**

Create `examples/neatenstein/browser-entry/renderer/renderer.sprite.constants.ts`:
- `RGBA_CHANNELS = 4` (import from browser-entry/constants.ts — single source)
- Pose names: `SPRITE_POSE_STAND = 'stand'`, `SPRITE_POSE_WALK1 = 'walk1'`, `SPRITE_POSE_WALK2 = 'walk2'`, `SPRITE_POSE_SHOOT = 'shoot'`
- Direction names: `SPRITE_DIR_FRONT = 'front'`, `SPRITE_DIR_FRONT_RIGHT = 'frontRight'`, ..., `SPRITE_DIR_FRONT_LEFT = 'frontLeft'` (8 directions)
- `SPRITE_INVISIBLE_SENTINEL = -1` (import from browser-entry/constants.ts)

Create `examples/neatenstein/browser-entry/renderer/renderer.sprite.types.ts`:
- `NeatensteinSpriteRenderContext`
- `NeatensteinCamera`
- `NeatensteinSprite`
- `NeatensteinDerezState`
- `NeatensteinSpriteSource`
- `NeatensteinSpriteProjection`
- `ResolvedSpriteFramebufferSize`

Update `sprites.ts`, `sprites.atlas.utils.ts`, `sprites.column.utils.ts`, `sprites.projection.utils.ts`, `sprites.guards.utils.ts`; re-export from `sprites.ts`.

**Slice 2.1.2 — Create `renderer.floor.constants.ts` and `renderer.floor.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.floor.constants.ts`:
- Floor rendering alpha/width/blur values (~15) → named constants
- `CELL_CENTER_OFFSET = 0.5` (import from browser-entry/constants.ts if shared, or define here)
- `'2d'` (import shared)

Create `examples/neatenstein/browser-entry/renderer/renderer.floor.types.ts`:
- `NeatensteinFloorCamera`
- `NeatensteinFloorRenderContext`
- `NeatensteinFloorSegmentBuffer`
- `SafeNeatensteinFloorCamera`
- `NeatensteinGridProjectionContext`
- `ProjectedNeatensteinGridPoint`

Update `floor.ts`, `floor.band.utils.ts`, `floor.projection.utils.ts`, `floor.shade.utils.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/renderer/sprites|neatenstein/renderer/floor`; `npx tsc --noEmit`

### Step 2.2: bolt + framebuffer + frame constants & types

**Slice 2.2.1 — Create `renderer.bolt.constants.ts` and `renderer.bolt.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.bolt.constants.ts`:
- `COMPOSITE_OP_LIGHTER = 'lighter'` (used 5×)
- `COLOR_WHITE_HEX = '#ffffff'`
- `COLOR_EMPTY_STRING = ''`
- ~20 bolt rendering values (radii, alphas, ratios) → named constants

Create `examples/neatenstein/browser-entry/renderer/renderer.bolt.types.ts`:
- `LateralProjectionContext`
- `FloorProjectionContext`
- `MuzzleScreenPosition`

Update `bolt-render.ts`, `bolt.utils.ts`; re-export.

**Slice 2.2.2 — Create `renderer.framebuffer.constants.ts`, `renderer.framebuffer.types.ts`, and `renderer.frame.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.framebuffer.constants.ts`:
- `RGBA_CHANNELS = 4` (import from browser-entry/constants.ts)
- Any framebuffer-specific numeric constants

Create `examples/neatenstein/browser-entry/renderer/renderer.framebuffer.types.ts`:
- `NeatensteinFramebufferSize`

Create `examples/neatenstein/browser-entry/renderer/renderer.frame.types.ts`:
- `NeatensteinRenderState`
- `NeatensteinRenderFrame`

Update `framebuffer.ts`, `frame.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/renderer/bolt|neatenstein/renderer/frame`; `npx tsc --noEmit`

### Step 2.3: map + pulse + rng constants & types

**Slice 2.3.1 — Create `renderer.rng.constants.ts`, `renderer.map.constants.ts`, and `renderer.map.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.rng.constants.ts` (shared LCG primes):
- `PARK_MILLER_MODULUS = 2_147_483_647`
- `PARK_MILLER_MULTIPLIER = 16_807`
- Derez hash primes: `DEREZ_HASH_PRIME_1 = 374761393`, `DEREZ_HASH_PRIME_2 = 668265263`, `DEREZ_HASH_PRIME_3 = 2246822519`, `DEREZ_HASH_MODULUS = 0x100000000`

Create `examples/neatenstein/browser-entry/renderer/renderer.map.constants.ts`:
- Map generation values: `MAP_WALL_DENSITY = 0.12`, `MAP_MIN_ROOMS = 4`, `MAP_MIN_OPEN = 0`, `MAP_MAX_OPEN = 1`
- Import RNG primes from `renderer.rng.constants.ts`

Create `examples/neatenstein/browser-entry/renderer/renderer.map.types.ts`:
- `CollisionMap`

Update `map.ts`, `pulse.ts` (import shared RNG primes instead of duplicating); re-export from `map.ts`.

**Slice 2.3.2 — Create `renderer.pulse.constants.ts` and `renderer.pulse.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.pulse.constants.ts`:
- Import RNG primes from `renderer.rng.constants.ts` (remove duplication from pulse.ts)
- Pulse-specific numeric constants

Create `examples/neatenstein/browser-entry/renderer/renderer.pulse.types.ts`:
- `NeatensteinPulseAxis`
- `NeatensteinDepthTestPulse`
- `NeatensteinPulse`

Update `pulse.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/renderer/map|neatenstein/renderer/pulse`; `npx tsc --noEmit`

### Step 2.4: raycast + gun + wall + zbuffer constants & types

**Slice 2.4.1 — Create `renderer.raycast.constants.ts`, `renderer.raycast.types.ts`, `renderer.gun.constants.ts`, and `renderer.gun.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.raycast.constants.ts`:
- `NEATENSTEIN_EPSILON_1E9 = 1e-9` (import from browser-entry/constants.ts)

Create `examples/neatenstein/browser-entry/renderer/renderer.raycast.types.ts`:
- `CastRayDDAHit`

Create `examples/neatenstein/browser-entry/renderer/renderer.gun.constants.ts`:
- `RGBA_OPAQUE_ALPHA = 255` (import from browser-entry/constants.ts)
- Gun-specific constants

Create `examples/neatenstein/browser-entry/renderer/renderer.gun.types.ts`:
- `EncodedGunSpriteFrame`

Update `raycast.ts`, `gun.ts`; re-export.

**Slice 2.4.2 — Create `renderer.wall.constants.ts`, `renderer.wall.types.ts`, `renderer.zbuffer.constants.ts`, `renderer.zbuffer.types.ts`**

Create `examples/neatenstein/browser-entry/renderer/renderer.wall.constants.ts`:
- `RGBA_OPAQUE_ALPHA = 255` (import from browser-entry/constants.ts)
- Wall-specific constants

Create `examples/neatenstein/browser-entry/renderer/renderer.wall.types.ts`:
- `NeatensteinWallRenderContext`
- `ParsedRgb`

Create `examples/neatenstein/browser-entry/renderer/renderer.zbuffer.constants.ts`:
- Z-buffer-specific constants

Create `examples/neatenstein/browser-entry/renderer/renderer.zbuffer.types.ts`:
- `NeatensteinSpriteClip`

Update `walls.ts`, `zbuffer.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/renderer/raycast|neatenstein/renderer/gun|neatenstein/renderer/wall|neatenstein/renderer/zbuffer`; `npx tsc --noEmit`

### Step 2.5: interpolate + sprite-decode constants & types

**Slice 2.5.1 — Create `renderer.interpolate.constants.ts`, `renderer.interpolate.types.ts`, gun-sprite-decode & robot-sprite-decode types**

Create `examples/neatenstein/browser-entry/renderer/renderer.interpolate.constants.ts`:
- `INTERP_PREVIOUS = 'previous'`, `INTERP_CURRENT = 'current'` (used 5× in interpolate.ts)

Create `examples/neatenstein/browser-entry/renderer/renderer.interpolate.types.ts`:
- `NeatensteinNumericState`

Add to `renderer.sprite.types.ts` (or create separate decode types files):
- `EncodedGunSpriteFrame` (if not in gun.types)
- `EncodedRobotSpriteFrame`

Update `interpolate.ts`, `gun-sprite-decode.ts`, `robot-sprite-decode.ts`; re-export.

**Slice 2.5.2 — Consolidate `RGBA_CHANNELS = 4` across all 6 definitions**

Remove local `RGBA_CHANNELS` definitions from:
- `gun-sprite-decode.ts`
- `robot-sprite-decode.ts`
- `sprites.column.utils.ts`
- `sprites.projection.utils.ts`
- `walls.ts`
- `framebuffer.ts`

Replace all with import from `browser-entry/constants.ts` (or `renderer.sprite.constants.ts` re-export).

**Validation:** `npx jest --testPathPattern=neatenstein/renderer/interpolate|neatenstein/renderer/sprite-decode|neatenstein/renderer/robot-sprite`; `npx tsc --noEmit`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 3: browser-entry/host/ Layer

**Owner:** Phase-3 agent  
**Depends on:** Phase 0 complete (shared constants available)  
**Base path:** `examples/neatenstein/browser-entry/host/`  

### Step 3.1: Shared host constants (DOM events, worker protocol) + host types

**Slice 3.1.1 — Create `host/dom-events.constants.ts` and `host/worker-protocol.constants.ts`**

Create `examples/neatenstein/browser-entry/host/dom-events.constants.ts`:
- `DOM_EVENT_KEYDOWN = 'keydown'`
- `DOM_EVENT_KEYUP = 'keyup'`
- `DOM_EVENT_MOUSEDOWN = 'mousedown'`
- `DOM_EVENT_MOUSEMOVE = 'mousemove'`
- `DOM_EVENT_CLICK = 'click'`
- `DOM_EVENT_TOUCHSTART = 'touchstart'`
- `DOM_EVENT_TOUCHMOVE = 'touchmove'`
- `DOM_EVENT_TOUCHEND = 'touchend'`
- `DOM_EVENT_TOUCHCANCEL = 'touchcancel'`
- `DOM_EVENT_BLUR = 'blur'`
- `DOM_EVENT_VISIBILITYCHANGE = 'visibilitychange'`
- `DOM_EVENT_CHANGE = 'change'`

Create `examples/neatenstein/browser-entry/host/worker-protocol.constants.ts`:
- Import `WORKER_MSG_*` from `browser-entry/constants.ts` and re-export (or reference directly)
- `RENDER_TIER_*` re-exports (or reference from browser-entry/constants.ts)

Update `controls.ts`, `input.ts`, `renderer-bridge.ts`, `hud.human-mode.utils.ts` to import from these files.

**Slice 3.1.2 — Create `host/types.ts` (new — host-level types not in game/types.ts)**

Create `examples/neatenstein/browser-entry/host/types.ts`:
- `EstimateCadenceOptions`
- `ContactPosition`
- `FireBoltResult`
- `FireEnemyBoltInput`
- `LookDelta`, `LookCallback`, `FireCallback`, `LightToggleCallback`, `TouchActiveCallback`, `BindingDetach`
- `CreateEpisodeOptions`, `Episode`
- `GameTickInputSnapshot`, `NormalizedGameTickInputSnapshot`, `UpdateEnemyBoltsResult`
- `BoltImpactResult`, `FireRecoilResult`, `SpawnWaveTickResult`
- `MugshotHeadCrop`, `MugshotLook`, `MugshotDirection`, `MugshotOverlay`
- `HumanMode`, `HumanModeSelector`, `NeonStatusBarState`
- `HiveDensityHudState`, `HiveDensityHud`, `DeathFeedbackSignal`, `DeathFeedbackIndicator`
- `HealthAmmoHudState`, `HealthAmmoHud`, `NeonStatusBarHud`, `WaveAnnouncementHud`
- `InputSnapshot`, `InputRouterDetach`, `InputRouter`
- `NeatensteinRendererBridgeOptions`, `NeatensteinRendererBridge`, `NeatensteinResizeResult`
- `AdvanceWaveOptions`, `AdvanceWaveResult`

Update all host modules to import from `host/types.ts`; re-export from main modules.

**Validation:** `npx jest --testPathPattern=neatenstein/host`; `npx tsc --noEmit`

### Step 3.2: Additions to existing game/constants.ts + game/types.ts + hud.constants.ts

**Slice 3.2.1 — Add missing constants to `game/constants.ts`**

Add to `examples/neatenstein/browser-entry/host/game/constants.ts`:
- `NEATENSTEIN_MS_PER_SECOND = 1000` (import from browser-entry/constants.ts — remove 3 local definitions in tick utils)
- `HUD_PERCENT_MULTIPLIER = 100`
- `HUD_OVERLAY_FONT_PX = 14`
- `HUD_Z_INDEX = 10`
- `HUD_DOM_FONT_PX = 16`
- `HUD_PADDING_PX = 4`
- `MAP_EDGE_OFFSET = 0.5`
- `MAP_HALF_SIZE = 2`
- `FALLBACK_SEED = 1`
- Mugshot crop values: `MUGSHOT_CROP_X = 19`, `MUGSHOT_CROP_Y = 30`, `MUGSHOT_CROP_W = 0`, `MUGSHOT_CROP_H = 15`, `MUGSHOT_OFFSET_X = 7`, `MUGSHOT_OFFSET_Y = 9`
- Mugshot colors: `MUGSHOT_COLOR_CYAN = [0, 240, 255, 255]`, `MUGSHOT_COLOR_GREY = [180, 190, 210, 255]`
- `NEAR_MISS_MULTIPLIER = 3`
- Shot outcome types: `SHOT_OUTCOME_WALL = 'wall'`, `SHOT_OUTCOME_ENEMY = 'enemy'`, `SHOT_OUTCOME_RANGE = 'range'`
- Spawn edge directions (import from scripts constants or define shared)

Update `game/*.ts` and tick utils to import; remove local `1000` definitions.

**Slice 3.2.2 — Add missing types to `game/types.ts` and missing constants to `hud.constants.ts`**

Add to `examples/neatenstein/browser-entry/host/game/types.ts`:
- Any game-specific types not already present (check against research list; most game types already in this file)

Add to `examples/neatenstein/browser-entry/host/hud.constants.ts`:
- CSS value constants: `CSS_POSITION_ABSOLUTE = 'absolute'`, `CSS_DISPLAY_FLEX = 'flex'`, `CSS_JUSTIFY_CENTER = 'center'`, `CSS_FONT_MONOSPACE = 'monospace'`, `CSS_WIDTH_100PCT = '100%'`, `CSS_HEIGHT_0PX = '0px'`, `CSS_HEIGHT_0PCT = '0%'`, `CSS_FONT_14PX = '14px'`, `CSS_PADDING_4PX_8PX = '4px 8px'`, `CSS_OVERLAY_BG = 'rgba(6, 11, 20, 0.85)'`
- `CSS_IMAGE_PIXELATED = 'pixelated'`
- `CSS_HEIGHT_AUTO = 'auto'`
- `CSS_FLEX_0_0_AUTO = '0 0 auto'`
- `ERROR_NOT_SUPPORTED = 'NotSupportedError'`
- `VISIBILITY_VISIBLE = 'visible'`

Update `hud.ts`, `hud.dom.utils.ts`, `hud.human-mode.utils.ts`, `hud.wave.utils.ts` to import; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/host`; `npx tsc --noEmit`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 4: browser-entry/worker/ Layer

**Owner:** Phase-4 agent  
**Depends on:** Phase 0 complete (shared constants available)  
**Base path:** `examples/neatenstein/browser-entry/worker/`  

### Step 4.1: display.worker constants & types + eval.worker consolidation

**Slice 4.1.1 — Create `display.worker.constants.ts` and `display.worker.types.ts`**

Create `examples/neatenstein/browser-entry/worker/display.worker.constants.ts`:
- `MAX_NODES = 64` (duplicated in display.worker + eval.worker — single source here)
- `MAX_CONNECTIONS = 256` (duplicated — single source here)
- `NEAT_POPSIZE = 4`
- `FOG_FEATHER_PX = 6`
- `PULSE_GLOW_ALPHA = 0.42`
- `GOLDEN_ANGLE_DEG = 137.508`
- HSL values: `COLOR_HSL_HUE = 360`, `COLOR_HSL_SAT = 0.65`, `COLOR_HSL_LIGHT = 0.5`, `COLOR_HSL_ALPHA = 60`
- `FALLBACK_FIRE_RANGE = 25`
- `FALLBACK_FIRE_ANGLE = Math.PI / 6`
- Import `WORKER_MSG_*`, `EVAL_MSG_*`, `SNAPSHOT_KIND_MLP`, `TICK_INPUT_SOURCE_*` from `browser-entry/constants.ts`
- Import `NEATENSTEIN_CANVAS_2D_CONTEXT` from `browser-entry/constants.ts`

Create `examples/neatenstein/browser-entry/worker/display.worker.types.ts`:
- `DisplayTier`
- `DisplayWorkerState` (27 fields — shared)
- `AutoAiState`
- `RaycastHit`
- `TickInputSource = 'auto' | 'human'` (or import from shared)
- `EvalRequestPayload` (consolidate duplicate — single source here)
- `EvalCompletePayload`

Update `display.worker.ts`, `sim.utils.ts`, `auto-ai.utils.ts`, `raycast.utils.ts`, `render.utils.ts`, `color.utils.ts`, `tick.utils.ts`; re-export from `display.worker.ts`.

**Slice 4.1.2 — Update `eval.worker.ts` to import shared constants & types**

Update `examples/neatenstein/browser-entry/worker/eval.worker.ts`:
- Import `MAX_NODES`, `MAX_CONNECTIONS` from `display.worker.constants.ts`
- Import `EvalRequestPayload` from `display.worker.types.ts` (remove duplicate definition)
- Import `EVAL_MSG_*` from `browser-entry/constants.ts`
- Import `NEAT_POPSIZE` from `display.worker.constants.ts`

**Validation:** `npx jest --testPathPattern=neatenstein/worker`; `npx tsc --noEmit`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 5: browser-entry/ top-level + harness/ Layer

**Owner:** Phase-5 agent  
**Depends on:** Phase 0 complete (shared constants available)  
**Base path:** `examples/neatenstein/browser-entry/` and `examples/neatenstein/browser-entry/harness/`  

### Step 5.1: audio constants & types + warmstart + curriculum constants

**Slice 5.1.1 — Create `audio.constants.ts` and `audio.types.ts`**

Create `examples/neatenstein/browser-entry/audio.constants.ts`:
- Oscillator types: `OSC_TYPE_SAWTOOTH = 'sawtooth'`, `OSC_TYPE_SQUARE = 'square'`, `OSC_TYPE_SINE = 'sine'`
- Filter types: `FILTER_TYPE_LOWPASS = 'lowpass'`, `FILTER_TYPE_HIGHPASS = 'highpass'`
- Audio cue params: all numeric values (880, 220, 0.25, 0.08, 1200, 330, 110, 0.35, 0.12, 800, etc.) → named constants
- `MIN_OSC_FREQ = 20`
- `GAIN_RAMP_FLOOR = 0.001`
- `VOICE_STOP_DELAY_MS = 0.01`

Create `examples/neatenstein/browser-entry/audio.types.ts`:
- `NeatensteinSoundName`
- `NeatensteinPlaySoundOptions`
- `NeatensteinAudioVoice`
- `NeatensteinAudioEngine`
- `NeatensteinCueParams`

Update `audio.ts`; re-export.

**Slice 5.1.2 — Create `warmstart.constants.ts` and `curriculum.constants.ts`**

Create `examples/neatenstein/browser-entry/warmstart.constants.ts` (MLP hyperparameters):
- `MLP_LEARNING_RATE = 0.7`
- `MLP_HIDDEN_NODES = 60`
- `MLP_MOMENTUM = 0.3`
- `MLP_DROPOUT_RATE = 0.08`
- `BCE_EPSILON = 1e-7` (import from scripts/backprop.utils.constants.ts or browser-entry/constants.ts)
- `BCE_BATCH_SIZE = 3`
- `EARLY_STOP_LR = 0.001`

Create `examples/neatenstein/browser-entry/harness/curriculum.constants.ts`:
- ~30 numeric target values → named constants
- 13 soft-target constants → named constants
- `CURRICULUM_NUM_TARGETS = 17` (if applicable)
- `CURRICULUM_PENALTY_WEIGHT = 2.0`
- `CURRICULUM_REWARD_WEIGHT = 1.0`

Update `warmstart.ts` (or wherever MLP hyperparameters live), `curriculum.utils.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/audio|neatenstein/warmstart|neatenstein/curriculum`; `npx tsc --noEmit`

### Step 5.2: harness enemy-mlp/enemy-swarm constants + entry-level constants & types

**Slice 5.2.1 — Create `harness/enemy-mlp.constants.ts` and harness type additions**

Create `examples/neatenstein/browser-entry/harness/enemy-mlp.constants.ts`:
- Mutation types: `MUTATION_WEIGHT = 'weight'`, `MUTATION_WEIGHTS = 'weights'`, `MUTATION_PERTURB = 'perturb'`
- MLP output labels: `MLP_OUTPUT_MOVE = 'move'`, `MLP_OUTPUT_STRAFE = 'strafe'`, `MLP_OUTPUT_TURN = 'turn'`, `MLP_OUTPUT_FIRE = 'fire'`
- `DASH_THRESHOLD = 0.5`
- Output indices: `MLP_OUTPUT_INDEX_MOVE = 0`, `MLP_OUTPUT_INDEX_STRAFE = 1`, `MLP_OUTPUT_INDEX_TURN = 2`, `MLP_OUTPUT_INDEX_FIRE = 3`, `MLP_OUTPUT_INDEX_DASH = 4`
- `CHAMPION_SEED_PRIME = 7919`
- `OUTPUT_PRECISION = 6`
- `HASH_SEED_PRIME = 100003`
- `LCG_SEED_MULTIPLIER = 2_654_435_761`
- `BOX_MULLER_FLOOR = 1e-10`
- Node/connection count ranges: `MIN_NODES = 20`, `MAX_NODES = 10`, `MIN_CONNECTIONS = 30`, `MAX_CONNECTIONS = 10` (verify actual values)
- Parsimony bounds: `PARSIMONY_MIN = 800`, `PARSIMONY_MAX = 3000`
- Hive density thresholds: `HIVE_DENSITY_LOW = 0.25`, `HIVE_DENSITY_MID = 0.5`, `HIVE_DENSITY_HIGH = 0.75`, `HIVE_DENSITY_FULL = 1.0`
- `REPLAY_PRESSURE = 0.1`
- Arms-race discriminator: `ARMS_RACE_REPLAY = 'replay'`, `ARMS_RACE_BASELINE = 'baseline'`

Add to `examples/neatenstein/browser-entry/harness/types.ts`:
- `CreateMlpEnemyPopulationOptions`, `MlpEnemyPopulation`
- `CreateSwarmEnemyPopulationOptions`, `SwarmVariant`, `SwarmEnemyPopulation`
- `RunArmsRaceGenerationOptions`, `ArmsRaceGenerationResult`
- `GenerationSnapshot`, `AdaptationSignal`
- `EvaluatedMainVariant`, `RunMainGenerationOptions`, `MainGenerationResult`
- `FireGateState`, `FireGateConfig`
- `CreateSeedPackOptions`
- **Consolidate `ReplayBuffer`** — remove duplicate from `replay-buffer.ts`, single definition in `types.ts`

Update `enemy-mlp.ts`, `enemy-swarm.ts`, `arms-race.ts`, `death-feedback.ts`, `main-runner.ts`, `neat-io-config.ts`, `seed-pack.ts`, `replay-buffer.ts`; re-export.

**Slice 5.2.2 — Add entry-level constants to `browser-entry/constants.ts` and entry types**

Add to `examples/neatenstein/browser-entry/constants.ts`:
- `REFERENCE_TIMESTEP_MS = 16` (remove duplicate from browser-entry.ts)
- `DEFAULT_MAX_HEALTH = 100`
- `DEFAULT_MAX_AMMO = 50`
- `DELTA_MULTIPLIER = 4`
- `SHA256_ALGORITHM = 'sha256'`
- `SHA256_ENCODING = 'hex'`

Add types to appropriate entry-level `.types.ts` or existing `harness/types.ts`:
- `NeatensteinStart`, `NeatensteinStop` (browser-entry.ts)
- `ShimHash` (node-crypto-shim)

Update `browser-entry.ts`, `node-crypto-shim.ts`; re-export.

**Validation:** `npx jest --testPathPattern=neatenstein/harness|neatenstein/browser-entry`; `npx tsc --noEmit`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`

---

## Phase 6: Verification & Cleanup Pass

**Owner:** Phase-6 agent (verification specialist)  
**Depends on:** Phases 0–5 ALL complete  

### Step 6.1: Update folder-quality-metrics.mjs + grep scan for remaining magic values

**Slice 6.1.1 — Update `scripts/folder-quality-metrics.mjs`**

Modify `examples/neatenstein/scripts/folder-quality-metrics.mjs` (or root `scripts/folder-quality-metrics.mjs`):
- Add `.constants.ts` and `.types.ts` to the `UTIL_FILE_SUFFIX` skip logic (same pattern as `.utils.ts` skip)
- These files are exempt from the missing-sibling-test-file check since they are pure constant/type definitions

**Slice 6.1.2 — Grep scan for remaining magic strings & numbers**

Run targeted grep scans across `examples/neatenstein/`:
- Search for bare string literals in non-`.constants.ts` files: `grep -rn "'[a-z].*'" --include="*.ts" --exclude="*.constants.ts" --exclude="*.types.ts" --exclude="*.test.ts" --exclude="*.spec.ts" examples/neatenstein/`
- Search for common magic numbers: `0.5`, `1000`, `255`, `128`, `192`, `4`, `8`, `-1` in non-constant/test files
- Verify NO duplicated constant definitions remain (e.g., `RGBA_CHANNELS`, `DIRECTIONS`, `ENEMY_SPRITE_STATES`, `ENEMY_SPRITE_DIRECTIONS`, `128`, `192`, `PNG_SIGNATURE`)
- Verify all `WORKER_MSG_*` usages import from shared constants
- Verify all `ANIM_STATE_*` usages import from shared constants
- Verify `1000` ms→s usages all import `NEATENSTEIN_MS_PER_SECOND`
- Verify `Math.PI * 2` usages import `FULL_CIRCLE_RADIANS`

Report any remaining inline magic values; create follow-up tickets if any are found.

**Validation:** `npx tsc --noEmit` from `examples/neatenstein`; `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein`; full `npx jest --testPathPattern=neatenstein` (all tests pass)

### Step 6.2: Bonus refactoring targets (document, optional execution)

**Slice 6.2.1 — Document duplicate helper consolidation**

These are BONUS targets — not required for this plan but noted for future work:

1. **Duplicate `clamp()` helpers** — 4 identical implementations in:
   - `bolt.utils.ts`
   - `floor.band.utils.ts`
   - `floor.projection.utils.ts`
   - `floor.shade.utils.ts`
   
   → Consolidate into `renderer.math.utils.ts` (or `browser-entry/math.utils.ts`) and import from all 4 locations.

2. **Duplicate `isPositiveIntegerDimension` predicate** — identical in:
   - `walls.ts`
   - `sprites.guards.utils.ts`
   
   → Consolidate into `renderer.guards.utils.ts` (or `sprites.guards.utils.ts` and import from `walls.ts`).

3. **Duplicate `ReplayBuffer` type** — in `types.ts` + `replay-buffer.ts`:
   → Single definition in `harness/types.ts`, import in `replay-buffer.ts`. (Handled in Phase 5 Slice 5.2.1.)

4. **Duplicate `EvalRequestPayload` type** — in `display.worker.ts` + `eval.worker.ts`:
   → Single definition in `display.worker.types.ts`, import in both. (Handled in Phase 4 Slice 4.1.1.)

**Slice 6.2.2 — Final full validation pass**

Run complete validation:
1. `npx tsc --noEmit` from `examples/neatenstein` — zero errors
2. `npx jest --testPathPattern=neatenstein` — all tests pass
3. `node scripts/folder-quality-metrics.mjs --folder=examples/neatenstein` — no missing-sibling-test warnings for `.constants.ts` or `.types.ts` files
4. Grep verification: no remaining duplicate constant definitions across files
5. Confirm all new files follow naming convention and re-export pattern

**Validation:** All of the above; this is the final gate.

---

## File Creation Summary

### New files to CREATE (total: ~40)

**Phase 0:** 0 new files (modify existing `browser-entry/constants.ts`)

**Phase 1 (scripts/):** 15 new files
- `enemy-animator.constants.ts`, `enemy-animator.types.ts`
- `enemy-controller.constants.ts`, `enemy-controller.types.ts`
- `enemy-navigation.constants.ts`, `enemy-navigation.types.ts`
- `enemy-sprite.constants.ts`, `enemy-sprite.types.ts`
- `generate-enemy-sprites.constants.ts`, `generate-enemy-sprites.types.ts`
- `snapshot-renderer.constants.ts`, `snapshot-renderer.types.ts`
- `voxel-enemy.constants.ts`, `voxel-enemy.types.ts`
- `png.utils.constants.ts`

**Phase 2 (renderer/):** ~20 new files
- `renderer.sprite.constants.ts`, `renderer.sprite.types.ts`
- `renderer.floor.constants.ts`, `renderer.floor.types.ts`
- `renderer.bolt.constants.ts`, `renderer.bolt.types.ts`
- `renderer.framebuffer.constants.ts`, `renderer.framebuffer.types.ts`
- `renderer.frame.types.ts`
- `renderer.rng.constants.ts`
- `renderer.map.constants.ts`, `renderer.map.types.ts`
- `renderer.pulse.constants.ts`, `renderer.pulse.types.ts`
- `renderer.raycast.constants.ts`, `renderer.raycast.types.ts`
- `renderer.gun.constants.ts`, `renderer.gun.types.ts`
- `renderer.wall.constants.ts`, `renderer.wall.types.ts`
- `renderer.zbuffer.constants.ts`, `renderer.zbuffer.types.ts`
- `renderer.interpolate.constants.ts`, `renderer.interpolate.types.ts`

**Phase 3 (host/):** 3 new files
- `host/dom-events.constants.ts`
- `host/worker-protocol.constants.ts`
- `host/types.ts`

**Phase 4 (worker/):** 2 new files
- `display.worker.constants.ts`
- `display.worker.types.ts`

**Phase 5 (entry/ + harness/):** 5 new files
- `audio.constants.ts`, `audio.types.ts`
- `warmstart.constants.ts`
- `curriculum.constants.ts` (in harness/)
- `harness/enemy-mlp.constants.ts`

### Existing files to MODIFY

- `browser-entry/constants.ts` — add cross-cutting shared constants (Phase 0 + Phase 5)
- `host/game/constants.ts` — add missing game constants (Phase 3)
- `host/game/types.ts` — add missing game types (Phase 3)
- `host/hud.constants.ts` — add CSS/event constants (Phase 3)
- `harness/constants.ts` — add harness-specific constants (Phase 5)
- `harness/types.ts` — add missing harness types, consolidate `ReplayBuffer` (Phase 5)
- `scripts/folder-quality-metrics.mjs` — add `.constants.ts`/`.types.ts` skip (Phase 6)
- All main modules and `.utils.ts` files across all layers — update imports, add re-exports

---

## Execution Order

```
Phase 0 (shared constants) ──────────────────────────────────── MUST COMPLETE FIRST
    │
    ├── Phase 1 (scripts/)         ─┐
    ├── Phase 2 (renderer/)        ─┤
    ├── Phase 3 (host/)            ─┼── ALL PARALLEL (independent file sets)
    ├── Phase 4 (worker/)          ─┤
    ├── Phase 5 (entry/ + harness/)─┘
    │
Phase 6 (verification + cleanup) ── MUST COMPLETE LAST
```

**Dispatch model:** Phase 0 as sync (blocking). Phases 1–5 as 5 parallel background agents. Phase 6 as sync after all 5 complete.

---

## DRY Consolidation Checklist

| Duplicate | Files | Resolution |
|---|---|---|
| `RGBA_CHANNELS = 4` | 6 files | Import from `browser-entry/constants.ts` |
| `DIRECTIONS` array | 2 files | Single source in `enemy-controller.constants.ts` |
| `ENEMY_SPRITE_STATES` | 2 files | Single source in `enemy-sprite.constants.ts` |
| `ENEMY_SPRITE_DIRECTIONS` | 3 files | Single source in `enemy-sprite.constants.ts` |
| `128` frame size | 2 files | `ENEMY_FRAME_SIZE_PX` in `enemy-animator.constants.ts` |
| `192` reference size | 2 files | `ENEMY_REFERENCE_SIZE_PX` in `enemy-animator.constants.ts` |
| `PNG_SIGNATURE` | source + test | Single source in `png.utils.constants.ts` |
| `1000` ms→s | 3 tick utils + scripts | `NEATENSTEIN_MS_PER_SECOND` in `browser-entry/constants.ts` |
| `Math.PI * 2` | multiple | `FULL_CIRCLE_RADIANS` in `browser-entry/constants.ts` |
| `255` opaque alpha | gun, walls | `RGBA_OPAQUE_ALPHA` in `browser-entry/constants.ts` |
| Park-Miller primes | map + pulse | `renderer.rng.constants.ts` |
| `clamp()` helper | 4 files | Consolidate into shared math utils (bonus) |
| `isPositiveIntegerDimension` | 2 files | Consolidate (bonus) |
| `ReplayBuffer` type | 2 files | Single in `harness/types.ts` |
| `EvalRequestPayload` type | 2 files | Single in `display.worker.types.ts` |
| `64` maxNodes | 2 workers | `MAX_NODES` in `display.worker.constants.ts` |
| `256` maxConns | 2 workers | `MAX_CONNECTIONS` in `display.worker.constants.ts` |
| Worker msg types | worker + host | `WORKER_MSG_*` in `browser-entry/constants.ts` |
| Anim states | scripts + renderer + worker | `ANIM_STATE_*` in `browser-entry/constants.ts` |
| Snapshot kinds | 6+ files | `SNAPSHOT_KIND_*` in `browser-entry/constants.ts` |
| `'2d'` canvas context | worker + entry + host | `NEATENSTEIN_CANVAS_2D_CONTEXT` in `browser-entry/constants.ts` |
| Adaptation directions | 2 files | `ADAPTATION_*` in `browser-entry/constants.ts` |

---

## Status Tracking

| Phase | Status | Agent | Notes |
|---|---|---|---|
| Phase 0: Shared constants | [DONE] | phase0-shared-constants | 40+ constants added, metrics script updated |
| Phase 1: scripts/ | [DONE] | phase1-scripts | 15 new files, 19 modified |
| Phase 2: renderer/ | [DONE] | phase2-renderer | 26 new files, 20 modified |
| Phase 3: host/ | [DONE] | phase3-host | 4 new files, 23 modified |
| Phase 4: worker/ | [DONE] | phase4-worker | 2 new files, 8 modified |
| Phase 5: entry/ + harness/ | [DONE] | phase5-harness | 7 new files, 15 modified |
| Phase 6: Verification | [DONE] | phase6-verification | 15 files fixed, SHA-256 bug fixed, test created |

**Final validation: tsc 0 errors, 1579/1581 tests pass (1 pre-existing eval.worker GPU type), metrics PASS, 546/546 JSDoc (1031/1031 across all files)**

**TOTALS: ~55 new files created, ~90+ files modified, 50+ .constants.ts files, 22+ .types.ts files**

---

## Notes

- **Re-export pattern**: Every main module (e.g., `sprites.ts`) must `export * from './renderer.sprite.constants.js'` and `export * from './renderer.sprite.types.js'` so existing import paths don't break.
- **Import cycles**: Category-level `.constants.ts` files (e.g., `renderer.rng.constants.ts`) must not import from main modules. They can import from `browser-entry/constants.ts` (the central hub) which has no upstream dependencies.
- **`.utils.ts` interaction**: Many `.utils.ts` files currently define constants inline. After extraction, they import from the new `.constants.ts` files. The `.utils.ts` files remain unchanged in structure — only their constant references change.
- **Test impact**: No test files should need changes. If a test imports a constant directly from a module that now re-exports from `.constants.ts`, the import still works via re-export. Only update tests if they import a constant that moved and the re-export is missing.
- **Validation commands**: All `npx tsc --noEmit` commands run from `examples/neatenstein/` directory. All Jest commands use `--testPathPattern=neatenstein` scope.