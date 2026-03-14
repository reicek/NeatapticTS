import type {
  SpeciationOptions,
  SpeciationHarnessContext,
} from '../../neat.types';
import {
  DEFAULT_COMPATIBILITY_INTEGRAL_GAIN,
  DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN,
  DEFAULT_COMPAT_INTEGRAL,
  DEFAULT_TARGET_SPECIES,
} from '../shared/speciation.shared';
import type { CompatAdjust } from '../shared/speciation.shared';

/**
 * Compatibility-threshold tuning mechanics for speciation.
 *
 * This chapter isolates the PID-like controller that keeps the species count
 * near a target. It is useful when you want to study threshold adaptation
 * without also reading the assignment or history code.
 */

/**
 * Update the adaptive compatibility threshold and clamp to bounds.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param compatAdjust - Compatibility adjustment settings.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Nothing.
 */
export function adjustCompatibilityThreshold<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: CompatAdjust,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void {
  // Step 1: Ensure the integral term is initialized.
  if (typeof speciationContext._compatIntegral !== 'number')
    speciationContext._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
  // Step 2: Run PID update if the threshold is numeric.
  if (typeof options.compatibilityThreshold === 'number') {
    const updatedThreshold = computePidThreshold(
      speciationContext,
      options,
      compatAdjust,
      options.compatibilityThreshold,
      minCompatibilityThreshold,
      maxCompatibilityThreshold,
    );
    options.compatibilityThreshold = updatedThreshold;
  }
  // Step 3: Always clamp to configured min/max.
  clampCompatibilityThreshold(
    options,
    minCompatibilityThreshold,
    maxCompatibilityThreshold,
  );
}

/**
 * Compute a PID-based threshold update and clamp when needed.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param compatAdjust - Compatibility adjustment settings.
 * @param currentThreshold - Current compatibility threshold.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Updated threshold.
 */
function computePidThreshold<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: CompatAdjust,
  currentThreshold: number,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): number {
  // Step 1: Resolve target and observed species counts for the PID error.
  const targetSpeciesCount = options.targetSpecies ?? DEFAULT_TARGET_SPECIES;
  const observedSpeciesCount = speciationContext._species.length;
  // Step 2: Compute the signed error (positive means too few species).
  const speciesError = targetSpeciesCount - observedSpeciesCount;
  // Step 3: Resolve PID gains from configuration with defaults.
  const proportionalGain =
    compatAdjust.kp ?? DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN;
  const integralGain = compatAdjust.ki ?? DEFAULT_COMPATIBILITY_INTEGRAL_GAIN;
  // Step 4: Update the integral accumulator and compute the PID delta.
  const updatedIntegral = updateCompatibilityIntegral(
    speciationContext,
    speciesError,
  );
  const thresholdDelta = computePidDelta(
    speciesError,
    proportionalGain,
    integralGain,
    updatedIntegral,
  );
  // Step 5: Apply the delta to the current threshold.
  const rawThreshold = currentThreshold - thresholdDelta;
  // Step 6: Clamp to bounds and reset integral if clamped.
  return clampPidThreshold(
    speciationContext,
    rawThreshold,
    minCompatibilityThreshold,
    maxCompatibilityThreshold,
  );

  /**
   * @param context - Speciation harness context.
   * @param errorValue - Difference between target and observed species.
   * @returns Updated integral accumulator value.
   */
  function updateCompatibilityIntegral(
    context: SpeciationHarnessContext<TOptions>,
    errorValue: number,
  ): number {
    // Step 1: Read the current integral accumulator.
    const previousIntegral = context._compatIntegral ?? DEFAULT_COMPAT_INTEGRAL;
    // Step 2: Accumulate the error into the integral term.
    const nextIntegral = previousIntegral + errorValue;
    // Step 3: Persist the updated accumulator back to context.
    context._compatIntegral = nextIntegral;
    return nextIntegral;
  }

  /**
   * @param errorValue - Difference between target and observed species.
   * @param proportional - Proportional gain.
   * @param integral - Integral gain.
   * @param integralValue - Current integral accumulator value.
   * @returns Threshold delta to apply.
   */
  function computePidDelta(
    errorValue: number,
    proportional: number,
    integral: number,
    integralValue: number,
  ): number {
    // Step 1: Compute the proportional contribution.
    const proportionalContribution = proportional * errorValue;
    // Step 2: Compute the integral contribution.
    const integralContribution = integral * integralValue;
    // Step 3: Combine contributions into the delta.
    return proportionalContribution + integralContribution;
  }

  /**
   * @param context - Speciation harness context.
   * @param candidateThreshold - Threshold before clamping.
   * @param minThreshold - Lower clamp bound.
   * @param maxThreshold - Upper clamp bound.
   * @returns Clamped threshold.
   */
  function clampPidThreshold(
    context: SpeciationHarnessContext<TOptions>,
    candidateThreshold: number,
    minThreshold: number,
    maxThreshold: number,
  ): number {
    // Step 1: Clamp low and reset integral when below the minimum.
    if (candidateThreshold < minThreshold) {
      context._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
      return minThreshold;
    }
    // Step 2: Clamp high and reset integral when above the maximum.
    if (candidateThreshold > maxThreshold) {
      context._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
      return maxThreshold;
    }
    // Step 3: Return the unclamped threshold.
    return candidateThreshold;
  }
}

/**
 * Clamp the compatibility threshold to configured bounds.
 *
 * @param options - Speciation options.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Nothing.
 */
function clampCompatibilityThreshold(
  options: SpeciationOptions,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void {
  // Step 1: Clamp when the threshold is present.
  if (typeof options.compatibilityThreshold !== 'number') return;
  if (options.compatibilityThreshold < minCompatibilityThreshold)
    options.compatibilityThreshold = minCompatibilityThreshold;
  if (options.compatibilityThreshold > maxCompatibilityThreshold)
    options.compatibilityThreshold = maxCompatibilityThreshold;
}