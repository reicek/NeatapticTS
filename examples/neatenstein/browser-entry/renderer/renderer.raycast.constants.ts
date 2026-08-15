/**
 * Raycast constants extracted from {@link module:./raycast}.
 *
 * Centralises the direction-component epsilon used by the DDA traversal so
 * it can be shared and adjusted in one place.
 *
 * @module
 */

import { NEATENSTEIN_EPSILON_1E9 } from '../constants';

/**
 * Direction component below this magnitude is treated as zero.
 *
 * This avoids unstable reciprocal values for rays that are effectively
 * axis-aligned. Re-exports the shared epsilon from `browser-entry/constants.ts`
 * as the single source of truth.
 */
export const RAY_DIRECTION_EPSILON = NEATENSTEIN_EPSILON_1E9;
