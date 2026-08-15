/**
 * Gun recoil and fire-state executors for the Neatenstein tick pipeline.
 *
 * @module
 */

import {
  NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND,
  NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
  NEATENSTEIN_MS_PER_SECOND,
} from './constants';
import { resolveTickDurationMs } from './tick.time.utils';
import type { GunState } from './types';

/**
 * Decay gun recoil toward zero over time.
 *
 * The recoil offset is reduced by the configured decay rate each second and
 * clamped so it never becomes negative or exceeds the maximum offset.
 *
 * @param gun - Gun overlay state before decay.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns Updated gun state with decayed recoil offset.
 */
export function decayGunRecoil(gun: GunState, dtMs: number): GunState {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const decayPixels =
    NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND * (resolvedDtMs / NEATENSTEIN_MS_PER_SECOND);
  const nextOffset = Math.max(0, gun.recoilOffset - decayPixels);

  return {
    ...gun,
    recoilOffset: Math.min(nextOffset, NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX),
    firing: false,
  };
}
