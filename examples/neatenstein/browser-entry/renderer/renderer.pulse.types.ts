/**
 * Type definitions for the pulse system extracted from {@link module:./pulse}.
 *
 * @module
 */

import type {
  NEATENSTEIN_PULSE_LAYER_FLOOR,
  NEATENSTEIN_PULSE_LAYER_CEILING,
} from './renderer.pulse.constants';

/**
 * Axis a pulse travels along.
 *
 * - `x` means the pulse moves along a line of constant world X (varying Y).
 * - `y` means the pulse moves along a line of constant world Y (varying X).
 */
export type NeatensteinPulseAxis = 'x' | 'y';

/**
 * Minimal pulse shape needed for z-buffer depth testing.
 */
export interface NeatensteinDepthTestPulse {
  /** Screen column index the projected pulse occupies. */
  screenColumn: number;
  /** Perpendicular distance from the camera plane to the pulse. */
  distance: number;
}

/**
 * A single rendered pulse.
 */
export interface NeatensteinPulse extends NeatensteinDepthTestPulse {
  /** World X coordinate of the pulse. */
  worldX: number;
  /** World Y coordinate of the pulse. */
  worldY: number;
  /** Seed that produced the pulse. */
  seed: number;
  /** Whether the pulse is still active. */
  active: boolean;
  /** Remaining lifetime in simulation ticks. */
  lifetimeTicks: number;
  /** Grid axis this pulse travels along. */
  axis: NeatensteinPulseAxis;
  /** Direction of travel along the axis (+1 or -1). */
  travelDirection: 1 | -1;
  /** Speed of travel in world units per tick. */
  travelSpeed: number;
  /** Render layer: floor or ceiling mirror. */
  layer:
    | typeof NEATENSTEIN_PULSE_LAYER_FLOOR
    | typeof NEATENSTEIN_PULSE_LAYER_CEILING;
}