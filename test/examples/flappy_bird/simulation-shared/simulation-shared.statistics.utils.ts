import { clampValue } from './simulation-shared.math.utils';

/**
 * Computes arithmetic mean for numeric samples.
 *
 * @param values - Numeric samples.
 * @returns Arithmetic mean.
 */
export function computeMean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return (
    values.reduce((accumulator, value) => accumulator + value, 0) /
    values.length
  );
}

/**
 * Computes population standard deviation.
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
    values.reduce((accumulator, value) => {
      const deltaFromMean = value - meanValue;
      return accumulator + deltaFromMean * deltaFromMean;
    }, 0) / values.length;

  return Math.sqrt(Math.max(0, variance));
}

/**
 * Computes percentile value via linear interpolation between nearest ranks.
 *
 * @param values - Numeric samples.
 * @param percentile - Percentile in [0, 1].
 * @returns Percentile value, or `Number.NaN` when `values` is empty.
 */
export function computePercentile(
  values: readonly number[],
  percentile: number,
): number {
  if (values.length === 0) return Number.NaN;

  const sortedValues = values.toSorted(compareNumbersAscending);
  const clampedPercentile = clampValue(percentile, 0, 1);
  const rawIndex = clampedPercentile * (sortedValues.length - 1);
  const lowerIndex = Math.floor(rawIndex);
  const upperIndex = Math.ceil(rawIndex);
  const interpolationWeight = rawIndex - lowerIndex;

  const lowerValue = sortedValues[lowerIndex] ?? sortedValues[0] ?? Number.NaN;
  const upperValue = sortedValues[upperIndex] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * interpolationWeight;
}

/**
 * Compares two numeric values in ascending order.
 *
 * @param leftValue - Left numeric value.
 * @param rightValue - Right numeric value.
 * @returns Comparator delta for `Array.prototype.toSorted`.
 */
function compareNumbersAscending(
  leftValue: number,
  rightValue: number,
): number {
  return leftValue - rightValue;
}
