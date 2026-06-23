import {
  NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
} from './neat.nge-adult.constants';
import type { AdultState } from './neat.nge-adult.types';

/**
 * Create the initial owner-local adult state for one fresh zone instance.
 *
 * Seeds empty plateau and gain-stability histories, marks the zone as not yet
 * in equilibrium, and disables growth cooling until the first adult cycle resolves it.
 *
 * @param zoneId - Stable adult-zone identifier that anchors the seeded state.
 * @returns Fresh adult state ready for the first `advanceAdultState` cycle.
 *
 * @example
 * ```ts
 * const adultState = createAdultState('zone:alpha');
 * console.log(adultState.equilibriumCandidate.zoneId); // 'zone:alpha'
 * ```
 */
export function createAdultState(zoneId: string): AdultState {
  return {
    plateauRecord: {
      windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
      rewardDeltas: [],
      isStagnant: false,
    },
    gainStabilityRecord: {
      windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
      gainHistory: [],
      isStable: false,
    },
    equilibriumCandidate: {
      zoneId,
      isGainStable: false,
      isPlateau: false,
    },
    growthCoolingActive: false,
    cycleCount: 0,
  };
}
