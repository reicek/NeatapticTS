/**
 * Shared constants for the Neatenstein neon raycasting demo.
 *
 * These values are locked by the Phase 1 design consensus (see
 * `plans/Neon_Shooter_NGE_Demo.research.md`). They cover the versioned
 * renderer frame protocol, tier-aware column counts, pulse timing,
 * sound names, and the fixed map size.
 *
 * Gameplay balance constants (player health/ammo, dash timing, enemy cap,
 * episode bounds) live in {@link ./host/game/constants} because the host-side
 * simulation owns those numeric contracts.
 */

/** Versioned format identifier carried in every render frame. */
export const NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION =
  'neatenstein-frame-v1' as const;

/** Column count for the GPU tier (premium shadow/glow path). */
export const NEATENSTEIN_GPU_COLUMN_COUNT = 320;

/** Column count for the Web Worker tier (OffscreenCanvas path). */
export const NEATENSTEIN_WORKER_COLUMN_COUNT = 240;

/** Column count for the CPU fallback tier (ImageData framebuffer). */
export const NEATENSTEIN_CPU_COLUMN_COUNT = 160;

/** Wall-clock milliseconds between ambient floor pulses. */
export const NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS = 3000;

/** Milliseconds an ambient pulse remains visible. */
export const NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS = 2700;

/** Milliseconds the generation-up ripple remains visible. */
export const NEATENSTEIN_PULSE_EVENT_GENERATION_UP_LIFETIME_MS = 600;

/** Milliseconds an enemy-death pulse remains visible. */
export const NEATENSTEIN_PULSE_EVENT_ENEMY_DEATH_LIFETIME_MS = 400;

/** Maximum number of concurrent floor pulses across all sources. */
export const NEATENSTEIN_PULSE_MAX_CONCURRENT = 8;

/** World-bearing tolerance in radians for matching vertical pulses across frames. */
export const NEATENSTEIN_PULSE_WORLD_BEARING_TOLERANCE_RAD = 0.1;

/** Ordered list of all procedural sound names used by the audio engine. */
export const NEATENSTEIN_AUDIO_SOUND_NAMES = [
  'fire',
  'enemy-hit',
  'player-damage',
  'dash',
  'kill',
  'generation-up',
] as const;

/** Fixed square map size in cells (24 x 24). */
export const NEATENSTEIN_MAP_SIZE = 24;

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
