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
