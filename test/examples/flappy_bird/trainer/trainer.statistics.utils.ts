import { clampValue } from '../flappy.simulation.shared.utils';

/**
 * Computes the arithmetic mean of numeric samples.
 *
 * @param values - Numeric samples.
 * @returns Arithmetic mean.
 */
export function computeMean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return (
    values.reduce(function accumulate(sumValue, currentValue): number {
      return sumValue + currentValue;
    }, 0) / values.length
  );
}

/**
 * Computes population standard deviation from numeric samples.
 *
 * @param values - Numeric samples.
 * @param meanValue - Precomputed mean.
 * @returns Population standard deviation.
 */
export function computePopulationStandardDeviation(
  values: readonly number[],
  meanValue: number,
): number {
  if (values.length === 0) return 0;

  const variance =
    values.reduce(function accumulateVariance(sumValue, currentValue): number {
      const deltaFromMean = currentValue - meanValue;
      return sumValue + deltaFromMean * deltaFromMean;
    }, 0) / values.length;

  return Math.sqrt(Math.max(0, variance));
}

/**
 * Computes an interpolated percentile value.
 *
 * @param values - Numeric samples.
 * @param percentile - Percentile in [0, 1].
 * @returns Interpolated percentile value.
 */
export function computePercentile(
  values: readonly number[],
  percentile: number,
): number {
  if (values.length === 0) return Number.NaN;

  const sortedValues = values.toSorted(compareNumbersAscending);

  const clampedPercentile = clampValue(percentile, 0, 1);
  const percentileIndex = clampedPercentile * (sortedValues.length - 1);
  const lowerIndex = Math.floor(percentileIndex);
  const upperIndex = Math.ceil(percentileIndex);
  const interpolationWeight = percentileIndex - lowerIndex;

  const lowerValue = sortedValues[lowerIndex] ?? sortedValues[0] ?? Number.NaN;
  const upperValue = sortedValues[upperIndex] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * interpolationWeight;
}

/**
 * Numeric ascending comparator.
 *
 * @param leftValue - Left numeric value.
 * @param rightValue - Right numeric value.
 * @returns Ascending comparator delta.
 */
export function compareNumbersAscending(
  leftValue: number,
  rightValue: number,
): number {
  return leftValue - rightValue;
}
