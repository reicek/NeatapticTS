/**
 * Pulse system constants extracted from {@link module:./pulse}.
 *
 * Centralises pulse timing, grid-margin, and layer-identifier constants so
 * they can be shared without duplication.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../host/game/constants';
import {
  NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS,
} from '../constants';

/** Ambient pulse interval rounded to whole simulation ticks. */
export const NEATENSTEIN_PULSE_AMBIENT_INTERVAL_TICKS = Math.round(
  NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS / NEATENSTEIN_FIXED_TIMESTEP_MS,
);

/** Ambient pulse lifetime rounded to whole simulation ticks. */
export const NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS = Math.ceil(
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS / NEATENSTEIN_FIXED_TIMESTEP_MS,
);

/** Number of cells reserved at each map edge so pulses stay on visible grid lines. */
export const NEATENSTEIN_PULSE_MAP_EDGE_MARGIN = 1;

/** Effective span of integer grid lines available for pulse travel. */
export const NEATENSTEIN_PULSE_GRID_SPAN =
  NEATENSTEIN_MAP_SIZE - NEATENSTEIN_PULSE_MAP_EDGE_MARGIN * 2;

/** Ambient pulse layer identifier for the floor grid. */
export const NEATENSTEIN_PULSE_LAYER_FLOOR = 'floor' as const;

/** Ambient pulse layer identifier for the ceiling grid. */
export const NEATENSTEIN_PULSE_LAYER_CEILING = 'ceiling' as const;
