/**
 * Shared constants for the Neatenstein neon raycasting demo.
 *
 * These values are locked by the Phase 1 design consensus (see
 * `plans/Neon_Shooter_NGE_Demo.research.md`). They cover the versioned
 * renderer frame protocol, tier-aware column counts, pulse timing,
 * sound names, and the fixed map size.
 *
 * Gameplay balance constants (player health/ammo, dash timing, enemy cap,
 * episode bounds) live in {@link ./host/game/constants.ts} because the host-side
 * simulation owns those numeric contracts.
 */

/** Versioned format identifier carried in every render frame. */
export const NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION =
  'neatenstein-frame-v1' as const;

/** Column count for the GPU tier (premium shadow/glow path). */
export const NEATENSTEIN_GPU_COLUMN_COUNT = 640;

/** Column count for the Web Worker tier (OffscreenCanvas path). */
export const NEATENSTEIN_WORKER_COLUMN_COUNT = 480;

/** Column count for the CPU fallback tier (ImageData framebuffer). */
export const NEATENSTEIN_CPU_COLUMN_COUNT = 320;

/** Wall-clock milliseconds between ambient floor pulses. */
export const NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 500;

/** Milliseconds an ambient pulse remains visible. */
export const NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS = 4000;

/** Maximum number of concurrent floor pulses across all sources. */
export const NEATENSTEIN_PULSE_MAX_CONCURRENT = 40;

/** Minimum rendered alpha for an ambient floor pulse. */
export const NEATENSTEIN_PULSE_ALPHA_MIN = 0.05;

/** Maximum rendered alpha for an ambient floor pulse. */
export const NEATENSTEIN_PULSE_ALPHA_MAX = 0.9;

/** Glow blur radius in pixels for ambient floor pulses. */
export const NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS = 3;

/** World-space travel speed lower bound for ambient pulses (world units per tick). */
export const NEATENSTEIN_PULSE_WORLD_SPEED_MIN = 0.02;

/** World-space travel speed upper bound for ambient pulses (world units per tick). */
export const NEATENSTEIN_PULSE_WORLD_SPEED_MAX = 0.08;

/** Screen-space dot radius for ambient floor pulses, in pixels. */
export const NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX = 1.5;

/**
 * Probability threshold for choosing the X axis when spawning an ambient pulse.
 *
 * Values below this threshold travel along a line of constant world X; values
 * at or above it travel along a line of constant world Y.
 */
export const NEATENSTEIN_PULSE_AXIS_X_THRESHOLD = 0.5;

/**
 * Probability threshold for choosing the negative travel direction when
 * spawning an ambient pulse.
 *
 * Values below this threshold move in the positive axis direction; values at
 * or above it move in the negative direction.
 */
export const NEATENSTEIN_PULSE_DIRECTION_NEGATIVE_THRESHOLD = 0.5;

/** Milliseconds a wall-impact neon spot remains visible. */
export const NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS = 3000;

/** Screen-space radius of a wall-impact neon spot, in pixels. */
export const NEATENSTEIN_IMPACT_SPOT_RADIUS_PX = 4;

/**
 * CSS color applied to wall-impact neon spots.
 *
 * A bright white with a slight cool/blue tint so wall hits read as part of
 * the same neon family as the plasma bolt impacts.
 */
export const NEATENSTEIN_IMPACT_SPOT_COLOR = '#f0f8ff';

/**
 * Glow color applied behind wall-impact neon spots.
 *
 * A translucent cool white halo that matches the bolt glow family.
 */
export const NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR = 'rgba(240,248,255,0.5)';

/** Glow blur radius in pixels for wall-impact neon spots. */
export const NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX = 2;

/**
 * Milliseconds an enemy-impact neon spot remains visible.
 *
 * Shorter than the wall-impact lifetime ({@link NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS})
 * so hit markers on enemies feel snappier and do not clutter the scene.
 */
export const NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS = 1000;

/** Screen-space radius of an enemy-impact neon spot, in pixels. */
export const NEATENSTEIN_ENEMY_IMPACT_RADIUS_PX = 5;

/**
 * CSS color applied to enemy-impact neon spots.
 *
 * A red-orange that matches the enemy bolt palette so hits on enemies read as
 * part of the same energy family.
 */
export const NEATENSTEIN_ENEMY_IMPACT_COLOR = '#ff4400';

/**
 * Glow color applied behind enemy-impact neon spots.
 *
 * A translucent red-orange halo that intensifies the hit marker.
 */
export const NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR = 'rgba(255,68,0,0.5)';

/** Glow blur radius in pixels for enemy-impact neon spots. */
export const NEATENSTEIN_ENEMY_IMPACT_GLOW_BLUR_PX = 4;

/**
 * Maximum screen-space radius of the expanding burst effect on enemy impact,
 * in pixels.
 *
 * The burst expands from 0 to this radius over
 * {@link NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS} and fades out, providing a
 * brief explosion flash in addition to the persistent mark.
 */
export const NEATENSTEIN_ENEMY_IMPACT_BURST_RADIUS_PX = 20;

/** Duration of the expanding burst effect on enemy impact, in milliseconds. */
export const NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS = 200;

/** Ordered list of all procedural sound names used by the audio engine. */
export const NEATENSTEIN_AUDIO_SOUND_NAMES = [
  'fire',
  'enemy-hit',
  'player-damage',
  'dash',
  'kill',
  'generation-up',
] as const;

/** Fixed square map size in cells (120 x 120). */
export const NEATENSTEIN_MAP_SIZE = 120;

/**
 * Default deterministic seed used when no seed is supplied.
 *
 * Shared by episode creation, map generation, and the worker initialization
 * fallback so the same default always produces the same canonical world.
 */
export const NEATENSTEIN_DEFAULT_SEED = 1;

/**
 * Published worker bundle filename, resolved relative to the host script that
 * loads the browser entrypoint. This is a classic (non-module) worker bundle
 * because OffscreenCanvas transfer is not reliable with module workers in the
 * Chromium versions used by this demo's target runtime.
 */
export const NEATENSTEIN_WORKER_BUNDLE_FILENAME =
  'neatenstein.worker.js' as const;

/**
 * Published filename of the dedicated evaluation worker bundle.
 *
 * The eval worker offloads NEAT population evaluation from the display
 * worker's render loop, preventing multi-second render stalls during
 * generation evaluation. It is published alongside the display worker bundle
 * in `docs/assets/`.
 *
 * @see AC-P2S1b-001
 */
export const NEATENSTEIN_EVAL_WORKER_BUNDLE_FILENAME =
  'neatenstein.eval-worker.js' as const;

/**
 * Fallback canvas width in CSS pixels when neither the client dimensions nor
 * the computed style provide a usable value.
 */
export const NEATENSTEIN_FALLBACK_CANVAS_WIDTH = 640;

/**
 * Fallback canvas height in CSS pixels when neither the client dimensions nor
 * the computed style provide a usable value.
 */
export const NEATENSTEIN_FALLBACK_CANVAS_HEIGHT = 360;

/**
 * Fallback status text RGB used when the CPU tier draws the "OffscreenCanvas
 * not available" message on the visible canvas.
 */
export const NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB = {
  r: 159,
  g: 220,
  b: 255,
} as const;

/**
 * Host-to-worker message type tag for input snapshots.
 *
 * The display worker consumes messages of this type and applies the included
 * look deltas to its local render camera for the worker tier.
 */
export const NEATENSTEIN_INPUT_MESSAGE_TYPE = 'input' as const;

/**
 * CSS color applied to the gun body overlay.
 *
 * A warm near-white ("Neon White") so the weapon reads as painted plastic or
 * ceramic against the dark raycast scene.
 */
export const NEATENSTEIN_GUN_BODY_COLOR = '#FBFFFF';

/**
 * CSS color applied to gun accent lines and highlights.
 *
 * A bright teal used for energy strips, sight dots, and the matching dynamic
 * light overlay.
 */
export const NEATENSTEIN_GUN_ACCENT_COLOR = '#00f0ff';

/**
 * CSS color applied to the full-canvas dynamic light tint.
 *
 * A dim teal screen blend that brightens the scene while the light is on.
 */
export const NEATENSTEIN_DYNAMIC_LIGHT_COLOR = '#00f0ff';

// prettier-ignore
/**
 * RGB tint applied to surviving pixels during the Tron-style derez death
 * animation.
 *
 * As the de-rez progress `t` increases, surviving pixels are lerped toward
 * this cool gray so the crumbling silhouette shifts from the enemy's team
 * color to an icy neutral before the final pixels scatter.
 */
export const NEATENSTEIN_ENEMY_DEATH_COLOR: readonly [number, number, number] = [180, 190, 210];

/**
 * Keyboard `code` for toggling the dynamic light overlay.
 *
 * The router tracks this key in the input snapshot and the tick pipeline flips
 * {@link GameState.lightEnabled} when it is pressed.
 */
export const NEATENSTEIN_LIGHT_TOGGLE_KEY = 'KeyL' as const;

/**
 * Supported renderer tiers for the Neatenstein demo.
 *
 * - `worker` → computation and rasterization happen on a dedicated worker
 *   using an {@link OffscreenCanvas}.
 * - `cpu`    → computation happens on a worker; the host blits the packed
 *   frame to a main-thread canvas.
 * - `gpu`    → same split as `cpu`, reserved for future GPU-backed
 *   computation.
 */
export type NeatensteinTier = 'worker' | 'cpu' | 'gpu';

// ---------------------------------------------------------------------------
// HIVE DENSITY HUD overlay design tokens
// ---------------------------------------------------------------------------

/** Fixed width of the HIVE DENSITY meter track, in CSS pixels. */
export const NEATENSTEIN_HUD_METER_WIDTH_PX = 160;

/** Fixed height of the HIVE DENSITY meter track, in CSS pixels. */
export const NEATENSTEIN_HUD_METER_HEIGHT_PX = 8;

/** Static label text shown on the HIVE DENSITY meter before the percentage. */
export const NEATENSTEIN_HUD_LABEL_TEXT = 'HIVE DENSITY' as const;

/**
 * Density threshold separating the calm and low color bands.
 *
 * Values below this threshold use the calm color; values at or above use the
 * low color.
 */
export const NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW = 0.25;

/**
 * Density threshold separating the low and mid color bands.
 *
 * Values below this threshold use the low color; values at or above use the
 * mid color.
 */
export const NEATENSTEIN_HIVE_DENSITY_THRESHOLD_MID = 0.5;

/**
 * Density threshold separating the mid and high color bands.
 *
 * Values below this threshold use the mid color; values at or above use the
 * high color.
 */
export const NEATENSTEIN_HIVE_DENSITY_THRESHOLD_HIGH = 0.75;

/** CSS color for hive densities below the low threshold (calm cyan). */
export const NEATENSTEIN_HIVE_DENSITY_COLOR_CALM = 'rgb(0, 240, 255)' as const;

/** CSS color for hive densities in the [low, mid) band (low green). */
export const NEATENSTEIN_HIVE_DENSITY_COLOR_LOW = 'rgb(160, 240, 0)' as const;

/** CSS color for hive densities in the [mid, high) band (mid amber). */
export const NEATENSTEIN_HIVE_DENSITY_COLOR_MID = 'rgb(240, 160, 0)' as const;

/** CSS color for hive densities at or above the high threshold (high magenta). */
export const NEATENSTEIN_HIVE_DENSITY_COLOR_HIGH = 'rgb(255, 0, 85)' as const;

// ---------------------------------------------------------------------------
// Human-mode selector design tokens
// ---------------------------------------------------------------------------

/** Label text for the auto-play mode option in the human-mode selector. */
export const NEATENSTEIN_HUMAN_MODE_LABEL_AUTO = 'auto' as const;

/** Label text for the human-play mode option in the human-mode selector. */
export const NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN = 'human' as const;

// ---------------------------------------------------------------------------
// Health/ammo HUD overlay design tokens
// ---------------------------------------------------------------------------

/** Static label text shown before the ammo count in the ammo label. */
export const NEATENSTEIN_HEALTH_AMMO_LABEL_AMMO = 'AMMO' as const;

/** Static label text shown before the health percentage in the health label. */
export const NEATENSTEIN_HEALTH_AMMO_LABEL_HEALTH = 'HEALTH' as const;

/**
 * Health fraction threshold separating the cyan and amber color bands.
 *
 * At or above this fraction → cyan. Below → amber or magenta.
 */
export const NEATENSTEIN_HEALTH_THRESHOLD_CYAN = 0.7;

/**
 * Health fraction threshold separating the amber and magenta color bands.
 *
 * At or above this fraction (but below cyan) → amber. Below → magenta.
 */
export const NEATENSTEIN_HEALTH_THRESHOLD_AMBER = 0.3;

/** CSS color for health fractions at or above the cyan threshold. */
export const NEATENSTEIN_HEALTH_COLOR_CYAN = 'rgb(0, 240, 255)' as const;

/** CSS color for health fractions in the [amber, cyan) band. */
export const NEATENSTEIN_HEALTH_COLOR_AMBER = 'rgb(240, 160, 0)' as const;

/** CSS color for health fractions below the amber threshold. */
export const NEATENSTEIN_HEALTH_COLOR_MAGENTA = 'rgb(255, 0, 85)' as const;

/** CSS color for the ammo pickup white core. */
export const NEATENSTEIN_AMMO_PICKUP_COLOR = '#ffffff' as const;

/** CSS color for the ammo pickup cool-white halo glow. */
export const NEATENSTEIN_AMMO_PICKUP_GLOW_COLOR =
  'rgba(240, 248, 255, 0.5)' as const;

/** Shadow blur in CSS pixels for the ammo pickup halo. */
export const NEATENSTEIN_AMMO_PICKUP_GLOW_BLUR_PX = 8;

/** Base radius in CSS pixels for the ammo pickup square at unit distance. */
export const NEATENSTEIN_AMMO_PICKUP_RADIUS_PX = 6;

// ---------------------------------------------------------------------------
// Cross-cutting shared constants (Phase 0 extraction)
//
// These constants are shared across multiple layers (scripts, renderer, worker,
// harness, host). They are extracted here so that no layer hardcodes a magic
// string or number that another layer also needs.
// ---------------------------------------------------------------------------

// Animation states — used in scripts + renderer + worker

/** Idle animation state tag. */
export const ANIM_STATE_IDLE = 'idle';

/** Moving animation state tag. */
export const ANIM_STATE_MOVE = 'move';

/** Firing animation state tag. */
export const ANIM_STATE_FIRE = 'fire';

/** Death animation state tag. */
export const ANIM_STATE_DEATH = 'death';

/** Damage animation state tag. */
export const ANIM_STATE_DAMAGE = 'damage';

/** Ordered list of all animation state tags. */
export const ANIMATION_STATES = [
  'idle',
  'move',
  'fire',
  'death',
  'damage',
] as const;

// Snapshot kinds — used in harness + worker + entry

/** Snapshot kind tag for a single MLP network. */
export const SNAPSHOT_KIND_MLP = 'mlp' as const;

/** Snapshot kind tag for a swarm of networks. */
export const SNAPSHOT_KIND_SWARM = 'swarm' as const;

// Tick input source — used in worker + host

/** Auto-play tick input source tag. */
export const TICK_INPUT_SOURCE_AUTO = 'auto';

/** Human-play tick input source tag. */
export const TICK_INPUT_SOURCE_HUMAN = 'human';

// Canvas context

/** Canvas 2D context identifier used with `getContext`. */
export const NEATENSTEIN_CANVAS_2D_CONTEXT = '2d';

// Time conversion

/** Milliseconds per second, used for time-unit conversions. */
export const NEATENSTEIN_MS_PER_SECOND = 1000;

// Math constants

/** Full circle in radians (2π). */
export const FULL_CIRCLE_RADIANS = Math.PI * 2;

/** Half rotation in radians (π). */
export const HALF_ROTATION_RADIANS = Math.PI;

// RGBA

/** Number of channels in an RGBA pixel. */
export const RGBA_CHANNELS = 4;

/** Alpha value for a fully opaque RGBA pixel. */
export const RGBA_OPAQUE_ALPHA = 255;

// Epsilon

/** Small epsilon for floating-point comparisons. */
export const NEATENSTEIN_EPSILON_1E9 = 1e-9;

/** Sentinel value indicating an invisible or non-existent entity. */
export const NEATENSTEIN_INVISIBLE_SENTINEL = -1;

// Render tiers

/** Render tier tag for the OffscreenCanvas worker path. */
export const RENDER_TIER_WORKER = 'worker';

/** Render tier tag for the CPU fallback path. */
export const RENDER_TIER_CPU = 'cpu';

/** Render tier tag for the GPU path. */
export const RENDER_TIER_GPU = 'gpu';

// Worker message types — shared between worker + host/renderer-bridge

/** Worker message type for initialization requests. */
export const WORKER_MSG_INIT = 'init';

/** Worker message type for resize events. */
export const WORKER_MSG_RESIZE = 'resize';

/** Worker message type for simulation state updates. */
export const WORKER_MSG_SIM_STATE = 'simState';

/** Worker message type signaling initialization is complete. */
export const WORKER_MSG_INITIALIZED = 'initialized';

/** Worker message type for rendered frame data. */
export const WORKER_MSG_FRAME = 'frame';

// Eval message types — shared between display.worker + eval.worker

/** Eval worker message type for evaluation requests. */
export const EVAL_MSG_EVALUATE = 'evaluate';

/** Eval worker message type for completed evaluation results. */
export const EVAL_MSG_EVAL_COMPLETE = 'evalComplete';

// Worker bundle filenames

/** Published display worker bundle filename. */
export const NEATENSTEIN_WORKER_BUNDLE = 'neatenstein.worker.js';

/** Published eval worker bundle filename. */
export const NEATENSTEIN_EVAL_WORKER_BUNDLE = 'neatenstein.eval-worker.js';

// Adaptation directions — shared: render-loop.utils + death-feedback.ts

/** Adaptation direction tag for increasing difficulty. */
export const ADAPTATION_STRONGER = 'stronger';

/** Adaptation direction tag for decreasing difficulty. */
export const ADAPTATION_WEAKER = 'weaker';

/** Adaptation direction tag for lateral parameter shifts. */
export const ADAPTATION_SHIFTED = 'shifted';

// DNA prefix

/** Prefix tag identifying a swarm DNA entry. */
export const SWARM_DNA_PREFIX = 'swarm:';

// Hex prefix and separators

/** Hex color string prefix. */
export const HEX_PREFIX = '#';

/** Separator used in CSV-style serialized strings. */
export const CSV_SEPARATOR = ',';

/** Separator used in composite cache key strings. */
export const CACHE_KEY_SEPARATOR = ':';

// ---------------------------------------------------------------------------
// Phase 5 — Entry-level constants
// ---------------------------------------------------------------------------

/**
 * Fixed simulation timestep in milliseconds.
 *
 * This is a private mirror of the canonical value exported from
 * {@link ./host/game/constants.ts}. It is intentionally **not** re-exported
 * (see the `no longer exports the removed fixed-timestep` contract test) but
 * is used to derive {@link REFERENCE_TIMESTEP_MS} without creating a circular
 * module dependency between this file and `./host/game/constants.ts`.
 */
const NEATENSTEIN_FIXED_TIMESTEP_MS = 16;

/** Reference timestep for FPS-scaled simulation stepping in milliseconds. */
export const REFERENCE_TIMESTEP_MS = NEATENSTEIN_FIXED_TIMESTEP_MS;

/** Default maximum player health used as a fallback frame value. */
export const DEFAULT_MAX_HEALTH = 100;

/** Default maximum player ammo used as a fallback frame value. */
export const DEFAULT_MAX_AMMO = 50;

/** Multiplier for the rAF delta-time upper bound (4 × reference timestep). */
export const DELTA_MULTIPLIER = 4;

/** SHA-256 algorithm identifier used by the node-crypto shim. */
export const SHA256_ALGORITHM = 'sha256' as const;

/** Hex encoding identifier used by the node-crypto shim digest. */
export const SHA256_ENCODING = 'hex' as const;
