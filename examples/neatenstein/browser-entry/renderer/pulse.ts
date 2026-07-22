/**
 * Ambient and event pulse system for the Neatenstein neon raycasting demo.
 *
 * Pulses are fake-perspective overlays on the floor grid. They are spawned
 * deterministically from the simulation tick and seed, tracked with a world-space
 * bearing so they stay visually anchored while the camera rotates, and
 * depth-tested against the per-column z-buffer so walls occlude them.
 *
 * @module
 */

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_WORLD_BEARING_TOLERANCE_RAD,
} from '../constants';

/** Fixed simulation timestep used to convert ticks to wall-clock milliseconds. */
const NEATENSTEIN_SIM_TICK_MS = 16;

/** Ambient pulse interval rounded to whole simulation ticks. */
const NEATENSTEIN_PULSE_AMBIENT_INTERVAL_TICKS = Math.round(
  NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS / NEATENSTEIN_SIM_TICK_MS,
);

/** Deterministic ambient pulse screen span, in renderer columns. */
const NEATENSTEIN_PULSE_SCREEN_WIDTH_COLUMNS = 8;

/** Closest perpendicular distance an ambient pulse can spawn at. */
const NEATENSTEIN_PULSE_MIN_DISTANCE = 2;

/** Farthest perpendicular distance an ambient pulse can spawn at. */
const NEATENSTEIN_PULSE_MAX_DISTANCE = 16;

/** Ticks removed from a pulse when its bearing drifts beyond tolerance. */
const NEATENSTEIN_PULSE_WORLD_BEARING_FADE_PENALTY_TICKS = 1_000;

/** Modulus for the Park-Miller-style deterministic LCG. */
const PARK_MILLER_MODULUS = 2_147_483_647;

/** Multiplier for the Park-Miller-style deterministic LCG. */
const PARK_MILLER_MULTIPLIER = 16_807;

/** Full circle in radians. */
const TAU = Math.PI * 2;

/**
 * Minimal pulse shape needed for z-buffer depth testing.
 */
export interface NeatensteinDepthTestPulse {
  /** Inclusive first screen column covered by the pulse. */
  screenColumnStart: number;
  /** Inclusive last screen column covered by the pulse. */
  screenColumnEnd: number;
  /** Perpendicular distance from the camera plane to the pulse. */
  distance: number;
}

/**
 * A single rendered pulse.
 */
export interface NeatensteinPulse extends NeatensteinDepthTestPulse {
  /** World-space bearing the pulse was emitted on. */
  worldBearingRad: number;
  /** Seed that produced the pulse. */
  seed: number;
  /** Whether the pulse is still active. */
  active: boolean;
  /** Remaining lifetime in simulation ticks. */
  lifetimeTicks: number;
  /** Horizontal screen column of the pulse center. */
  screenX: number;
}

/**
 * Determine whether an ambient pulse should be emitted on this simulation tick.
 *
 * Ambient pulses are spaced by {@link NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS}
 * using the fixed 16 ms simulation tick.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @returns `true` when the tick lands on an ambient pulse boundary.
 */
function isAmbientTick(simTick: number): boolean {
  if (simTick < 0) return false;
  return simTick % NEATENSTEIN_PULSE_AMBIENT_INTERVAL_TICKS === 0;
}

/**
 * Advance a Park-Miller-style LCG state.
 *
 * @param state - Current LCG state.
 * @returns Next LCG state.
 */
function nextLcgState(state: number): number {
  return (state * PARK_MILLER_MULTIPLIER) % PARK_MILLER_MODULUS;
}

/**
 * Convert an LCG state to a floating point value in the range [0, 1).
 *
 * @param state - Current LCG state.
 * @returns Normalized value in [0, 1).
 */
function lcgToFloat(state: number): number {
  return state / PARK_MILLER_MODULUS;
}

/**
 * Wrap an angle to the range [-PI, PI).
 *
 * @param angle - Angle in radians.
 * @returns Wrapped angle.
 */
function normalizeAngle(angle: number): number {
  let wrapped = angle % TAU;
  if (wrapped < -Math.PI) wrapped += TAU;
  if (wrapped >= Math.PI) wrapped -= TAU;
  return wrapped;
}

/**
 * Emit an ambient floor pulse if the current simulation tick qualifies.
 *
 * The pulse position, distance, and screen column are derived from a
 * deterministic LCG seeded with `seed` and `simTick`, so the same seed and tick
 * always produce the same pulse.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @param seed - Deterministic seed for this pulse stream.
 * @param columnCount - Number of renderer columns for screen-space placement
 *   (defaults to {@link NEATENSTEIN_CPU_COLUMN_COUNT}).
 * @returns A new ambient pulse, or `null` when the tick does not qualify.
 *
 * @example
 * ```ts
 * const pulse = emitNeatensteinAmbientPulse(0, 12345);
 * if (pulse) {
 *   console.log(pulse.worldBearingRad, pulse.distance);
 * }
 * ```
 */
export function emitNeatensteinAmbientPulse(
  simTick: number,
  seed: number,
  columnCount: number = NEATENSTEIN_CPU_COLUMN_COUNT,
): NeatensteinPulse | null {
  if (!isAmbientTick(simTick)) return null;

  let state = ((seed + simTick) % PARK_MILLER_MODULUS) * PARK_MILLER_MULTIPLIER;
  state = state % PARK_MILLER_MODULUS;
  state = nextLcgState(state);

  const worldBearingRad = lcgToFloat(state) * TAU;
  state = nextLcgState(state);

  const distanceRange =
    NEATENSTEIN_PULSE_MAX_DISTANCE - NEATENSTEIN_PULSE_MIN_DISTANCE;
  const distance =
    NEATENSTEIN_PULSE_MIN_DISTANCE + lcgToFloat(state) * distanceRange;
  state = nextLcgState(state);

  const maxColumn = columnCount - 1;
  const screenX = lcgToFloat(state) * columnCount;
  const screenColumnStart = Math.max(
    0,
    Math.floor(screenX - NEATENSTEIN_PULSE_SCREEN_WIDTH_COLUMNS / 2),
  );
  const screenColumnEnd = Math.min(
    maxColumn,
    screenColumnStart + NEATENSTEIN_PULSE_SCREEN_WIDTH_COLUMNS - 1,
  );

  const lifetimeTicks = Math.ceil(
    NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS / NEATENSTEIN_SIM_TICK_MS,
  );

  return {
    worldBearingRad,
    seed,
    active: true,
    lifetimeTicks,
    distance,
    screenX,
    screenColumnStart,
    screenColumnEnd,
  };
}

/**
 * Update the active pulse list for a new frame.
 *
 * Decrements lifetime, fades pulses whose world bearing has drifted beyond
 * tolerance relative to the current ray bearing, and enforces the concurrent
 * pulse ceiling. The returned array is a shallow copy; pulse objects are
 * immutable copies.
 *
 * @param pulses - Current active pulses.
 * @param simTick - Current fixed-timestep simulation tick (unused today but
 *   reserved for future tick-driven spawn logic).
 * @param rayBearingRad - Optional current camera ray bearing. When provided,
 *   pulses whose bearing differs by more than the tolerance are faded faster.
 * @returns Updated pulse list with at most
 *   {@link NEATENSTEIN_PULSE_MAX_CONCURRENT} entries.
 *
 * @example
 * ```ts
 * const next = updateNeatensteinPulses(pulses, tick, cameraBearing);
 * ```
 */
export function updateNeatensteinPulses(
  pulses: readonly NeatensteinPulse[],
  simTick: number,
  rayBearingRad?: number,
): NeatensteinPulse[] {
  const next = pulses
    .slice(0, NEATENSTEIN_PULSE_MAX_CONCURRENT)
    .map((pulse) => {
      let lifetimeTicks = pulse.lifetimeTicks - 1;

      if (rayBearingRad !== undefined) {
        const delta = Math.abs(
          normalizeAngle(pulse.worldBearingRad - rayBearingRad),
        );
        if (delta > NEATENSTEIN_PULSE_WORLD_BEARING_TOLERANCE_RAD) {
          lifetimeTicks = Math.max(
            0,
            lifetimeTicks - NEATENSTEIN_PULSE_WORLD_BEARING_FADE_PENALTY_TICKS,
          );
        }
      }

      return { ...pulse, lifetimeTicks, active: lifetimeTicks > 0 };
    });

  return next;
}

/**
 * Depth-test a pulse against the per-column z-buffer.
 *
 * A pulse is visible only when it is closer than every wall that covers its
 * screen span. If any covered column stores a wall distance strictly less than
 * the pulse distance, the pulse is considered hidden behind that wall.
 *
 * @param pulse - Pulse with a screen span and distance.
 * @param zBuffer - Per-column depth buffer filled by the wall pass.
 * @returns `true` when the pulse is not occluded by a closer wall.
 *
 * @example
 * ```ts
 * const visible = depthTestPulse(pulse, frame.zBuffer);
 * ```
 */
export function depthTestPulse(
  pulse: NeatensteinDepthTestPulse,
  zBuffer: Readonly<Float32Array>,
): boolean {
  const start = Math.max(0, Math.floor(pulse.screenColumnStart));
  const end = Math.min(zBuffer.length - 1, Math.floor(pulse.screenColumnEnd));

  for (let column = start; column <= end; column++) {
    if (pulse.distance >= zBuffer[column]) {
      return false;
    }
  }

  return true;
}

/**
 * Determine whether a "generation up" visual pulse should fire on this tick.
 *
 * This is an event-slot marker: the game director will schedule the actual
 * generation transition, and the renderer checks this flag to pair the visual
 * pulse with the matching sound.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @returns `true` for every non-negative tick, indicating the slot is paired.
 */
function isGenerationUpTick(simTick: number): boolean {
  return simTick >= 0;
}

/**
 * Emit the visual pulse for a generation-up event.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @returns `true` when the visual pulse is scheduled for this tick.
 *
 * @example
 * ```ts
 * if (emitNeatensteinGenerationUpPulse(tick)) {
 *   // queue pulse render
 * }
 * ```
 */
export function emitNeatensteinGenerationUpPulse(simTick: number): boolean {
  return isGenerationUpTick(simTick);
}

/**
 * Schedule the generation-up sound for a generation-up event.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @returns `true` when the sound is scheduled for this tick.
 *
 * @example
 * ```ts
 * if (scheduleNeatensteinGenerationUpSound(tick)) {
 *   // queue WebAudio one-shot
 * }
 * ```
 */
export function scheduleNeatensteinGenerationUpSound(simTick: number): boolean {
  return isGenerationUpTick(simTick);
}
