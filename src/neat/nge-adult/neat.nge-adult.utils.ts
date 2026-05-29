import {
  NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
} from './neat.nge-adult.constants';
import type { AdultState } from './neat.nge-adult.types';

/**
 * Create the initial owner-local adult state for one fresh zone instance.
 *
 * @param zoneId - Stable adult-zone identifier that anchors the seeded state.
 * @returns A compile-only placeholder while the Step 06 red contract is active.
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
