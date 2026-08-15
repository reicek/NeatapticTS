/**
 * Ambient and event pulse system for the Neatenstein neon raycasting demo.
 *
 * Pulses are small shiny dots that travel along integer floor-grid lines. They are
 * spawned deterministically from the simulation tick and seed, updated with
 * world-space velocity along their chosen grid line, and depth-tested against the
 * per-column z-buffer so walls occlude them.
 *
 * @module
 */

import {
  NEATENSTEIN_PULSE_AXIS_X_THRESHOLD,
  NEATENSTEIN_PULSE_DIRECTION_NEGATIVE_THRESHOLD,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_WORLD_SPEED_MAX,
  NEATENSTEIN_PULSE_WORLD_SPEED_MIN,
} from '../constants';
import {
  NEATENSTEIN_PULSE_AMBIENT_INTERVAL_TICKS,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
  NEATENSTEIN_PULSE_GRID_SPAN,
  NEATENSTEIN_PULSE_LAYER_CEILING,
  NEATENSTEIN_PULSE_LAYER_FLOOR,
  NEATENSTEIN_PULSE_MAP_EDGE_MARGIN,
} from './renderer.pulse.constants';
import {
  PARK_MILLER_MODULUS,
  PARK_MILLER_MULTIPLIER,
} from './renderer.rng.constants';
import type {
  NeatensteinDepthTestPulse,
  NeatensteinPulse,
  NeatensteinPulseAxis,
} from './renderer.pulse.types';

// Re-export constants and types for external consumers.
export {
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
  NEATENSTEIN_PULSE_LAYER_CEILING,
  NEATENSTEIN_PULSE_LAYER_FLOOR,
} from './renderer.pulse.constants';
export type {
  NeatensteinDepthTestPulse,
  NeatensteinPulse,
  NeatensteinPulseAxis,
} from './renderer.pulse.types';

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
 * Emit an ambient floor pulse if the current simulation tick qualifies.
 *
 * The pulse is spawned on a random integer X or Y grid line, at a random
 * coordinate along that line, with a random travel direction and speed. All
 * values are derived from a deterministic LCG seeded with `seed` and
 * `simTick`, so the same seed and tick always produce the same pulse.
 *
 * @param simTick - Current fixed-timestep simulation tick.
 * @param seed - Deterministic seed for this pulse stream.
 * @param layer - Render layer for the pulse; defaults to the floor grid.
 * @returns A new ambient pulse, or `null` when the tick does not qualify.
 *
 * @example
 * ```ts
 * const pulse = emitNeatensteinAmbientPulse(0, 12345);
 * if (pulse) {
 *   console.log(pulse.axis, pulse.worldX, pulse.worldY);
 * }
 * ```
 */
export function emitNeatensteinAmbientPulse(
  simTick: number,
  seed: number,
  layer:
    | typeof NEATENSTEIN_PULSE_LAYER_FLOOR
    | typeof NEATENSTEIN_PULSE_LAYER_CEILING = NEATENSTEIN_PULSE_LAYER_FLOOR,
): NeatensteinPulse | null {
  if (!isAmbientTick(simTick)) return null;

  let state = ((seed + simTick) % PARK_MILLER_MODULUS) * PARK_MILLER_MULTIPLIER;
  state = state % PARK_MILLER_MODULUS;
  state = nextLcgState(state);

  const axis: NeatensteinPulseAxis =
    lcgToFloat(state) < NEATENSTEIN_PULSE_AXIS_X_THRESHOLD ? 'x' : 'y';
  state = nextLcgState(state);

  const fixedCoord =
    NEATENSTEIN_PULSE_MAP_EDGE_MARGIN +
    Math.floor(lcgToFloat(state) * NEATENSTEIN_PULSE_GRID_SPAN);
  state = nextLcgState(state);

  const travelCoord =
    NEATENSTEIN_PULSE_MAP_EDGE_MARGIN +
    lcgToFloat(state) * NEATENSTEIN_PULSE_GRID_SPAN;
  state = nextLcgState(state);

  const travelDirection =
    lcgToFloat(state) < NEATENSTEIN_PULSE_DIRECTION_NEGATIVE_THRESHOLD ? 1 : -1;
  state = nextLcgState(state);

  const speedRange =
    NEATENSTEIN_PULSE_WORLD_SPEED_MAX - NEATENSTEIN_PULSE_WORLD_SPEED_MIN;
  const travelSpeed =
    NEATENSTEIN_PULSE_WORLD_SPEED_MIN + lcgToFloat(state) * speedRange;

  const worldX = axis === 'x' ? fixedCoord : travelCoord;
  const worldY = axis === 'x' ? travelCoord : fixedCoord;

  return {
    worldX,
    worldY,
    seed,
    active: true,
    lifetimeTicks: NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
    axis,
    travelDirection,
    travelSpeed,
    screenColumn: 0,
    distance: 0,
    layer,
  };
}

/**
 * Update the active pulse list for a new frame.
 *
 * Decrements lifetime, advances each pulse along its grid line by its travel
 * speed, removes pulses that have expired or been marked inactive, and
 * enforces the concurrent pulse ceiling. The returned array is a shallow copy;
 * pulse objects are immutable copies.
 *
 * @param pulses - Current active pulses.
 * @param simTick - Current fixed-timestep simulation tick (unused today but
 *   reserved for future tick-driven spawn logic).
 * @returns Updated pulse list with at most
 *   {@link NEATENSTEIN_PULSE_MAX_CONCURRENT} entries.
 *
 * @example
 * ```ts
 * const next = updateNeatensteinPulses(pulses, tick);
 * ```
 */
export function updateNeatensteinPulses(
  pulses: readonly NeatensteinPulse[],
  simTick: number,
): NeatensteinPulse[] {
  void simTick;

  const next = pulses
    .map((pulse) => {
      const lifetimeTicks = pulse.lifetimeTicks - 1;
      const layerMultiplier =
        pulse.layer === NEATENSTEIN_PULSE_LAYER_CEILING ? -1 : 1;
      const delta = pulse.travelDirection * pulse.travelSpeed * layerMultiplier;
      const worldX = pulse.axis === 'y' ? pulse.worldX + delta : pulse.worldX;
      const worldY = pulse.axis === 'x' ? pulse.worldY + delta : pulse.worldY;

      return {
        ...pulse,
        worldX,
        worldY,
        lifetimeTicks,
        active: lifetimeTicks > 0,
      };
    })
    .filter((pulse) => pulse.active && pulse.lifetimeTicks > 0)
    .slice(0, NEATENSTEIN_PULSE_MAX_CONCURRENT);

  return next;
}

/**
 * Depth-test a pulse against the per-column z-buffer.
 *
 * A pulse is visible when the column it projects to stores a wall distance
 * greater than or equal to the pulse distance. This allows pulses that sit on
 * the wall surface (for example wall-impact neon spots) to render while still
 * hiding pulses behind closer walls.
 *
 * @param pulse - Pulse with a screen column and perpendicular distance.
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
  const column = Math.max(
    0,
    Math.min(zBuffer.length - 1, Math.floor(pulse.screenColumn)),
  );

  return pulse.distance <= zBuffer[column];
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
