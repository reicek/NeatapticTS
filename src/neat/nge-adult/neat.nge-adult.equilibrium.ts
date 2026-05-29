import { NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE } from './neat.nge-adult.constants';
import type {
  EquilibriumCandidate,
  GainStabilityRecord,
  PlateauRecord,
} from './neat.nge-adult.types';

/**
 * Advances one rolling gain-stability record with a new adult neuromodulator gain sample.
 *
 * @param gainStabilityRecord - Existing gain evidence for the active adult zone.
 * @param gainMeasurement - Latest gain measurement observed for the current evaluation window.
 * @param gainStabilityTolerance - Allowed deviation around the rolling mean.
 * @returns Updated gain-stability evidence for the active adult zone.
 */
export function advanceGainStabilityRecord(
  gainStabilityRecord: GainStabilityRecord,
  gainMeasurement: number,
  gainStabilityTolerance = NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE,
): GainStabilityRecord {
  const nextGainHistory = [
    ...gainStabilityRecord.gainHistory,
    gainMeasurement,
  ].slice(-gainStabilityRecord.windowSize);
  const isStable =
    nextGainHistory.length === gainStabilityRecord.windowSize &&
    hasStableGainWindow(nextGainHistory, gainStabilityTolerance);

  return {
    windowSize: gainStabilityRecord.windowSize,
    gainHistory: nextGainHistory,
    isStable,
  };

  /**
   * Checks whether every gain sample remains inside the allowed tolerance of the rolling mean.
   *
   * @param gainHistory - Active rolling gain window.
   * @param tolerance - Maximum allowed deviation from the rolling mean.
   * @returns Whether the current gain window is stable enough for equilibrium checks.
   */
  function hasStableGainWindow(
    gainHistory: number[],
    tolerance: number,
  ): boolean {
    const rollingMean =
      gainHistory.reduce(
        (runningTotal, currentGain) => runningTotal + currentGain,
        0,
      ) / gainHistory.length;

    return gainHistory.every(
      (currentGain) => Math.abs(currentGain - rollingMean) <= tolerance,
    );
  }
}

/**
 * Detects whether one adult zone is ready to emit an equilibrium candidate.
 *
 * @param zoneId - Stable adult-zone identifier under evaluation.
 * @param plateauRecord - Plateau evidence for the current adult zone.
 * @param gainStabilityRecord - Gain-stability evidence for the current adult zone.
 * @returns An equilibrium candidate when both adult guards hold; otherwise null.
 */
export function detectEquilibriumCandidate(
  zoneId: string,
  plateauRecord: PlateauRecord,
  gainStabilityRecord: GainStabilityRecord,
): EquilibriumCandidate | null {
  if (!plateauRecord.isStagnant || !gainStabilityRecord.isStable) {
    return null;
  }

  return {
    zoneId,
    isGainStable: true,
    isPlateau: true,
  };
}
