import type { INetwork } from '../interfaces';

/**
 * Return the smallest power-of-two integer greater than or equal to `n`.
 *
 * @param n - Target minimum integer capacity.
 * @returns The smallest power of two greater than or equal to `n`.
 */
export function nextPowerOfTwo(n: number): number {
  const requested = Math.max(1, Math.floor(n));
  if (requested <= 1) return 1;

  if (requested <= 0xffffffff) {
    const value = (requested - 1) >>> 0;
    const leadingZeros = Math.clz32(value);
    const exponent = 32 - leadingZeros;
    const clampedExponent = Math.min(31, Math.max(0, exponent));
    const power = 1 << clampedExponent;
    return power >= requested ? power : power << 1;
  }

  let power = 1;
  while (power < requested) power *= 2;
  return power;
}

/**
 * Determine whether the provided value is a finite-number array.
 *
 * @param candidate - Value to inspect.
 * @returns True when the input is an array of finite numbers.
 */
export function isFiniteNumberArray(candidate: unknown): candidate is number[] {
  return (
    Array.isArray(candidate) &&
    candidate.every(
      (value: unknown) => typeof value === 'number' && Number.isFinite(value),
    )
  );
}

/**
 * Read the optional `_lastStepOutputs` history stored on a network.
 *
 * @param network - Network instance that may expose a reflected outputs history.
 * @returns Sanitized history buffer or `undefined` when absent or invalid.
 */
export function readOutputHistory(network: INetwork): number[][] | undefined {
  const historyCandidate = Reflect.get(network as object, '_lastStepOutputs');
  if (!Array.isArray(historyCandidate)) return undefined;
  return historyCandidate.every(isFiniteNumberArray)
    ? (historyCandidate as number[][])
    : undefined;
}

/**
 * Persist a bounded outputs history on the network via reflection.
 *
 * @param network - Target network to mutate.
 * @param history - Updated history buffer.
 */
export function writeOutputHistory(
  network: INetwork,
  history: number[][],
): void {
  Reflect.set(network as object, '_lastStepOutputs', history);
}

/**
 * Materialize the active prefix of pooled path buffers into a fresh array.
 *
 * @param length - Number of path entries to materialize.
 * @param pathX - Pooled X-coordinate buffer.
 * @param pathY - Pooled Y-coordinate buffer.
 * @returns A newly allocated array of path tuples.
 */
export function materializePath(
  length: number,
  pathX: Int32Array,
  pathY: Int32Array,
): [number, number][] {
  const entries = Math.max(0, Math.floor(length));
  if (entries === 0) return [];

  const output = new Array<[number, number]>(entries);
  for (let index = 0; index < entries; index++) {
    output[index] = [pathX[index], pathY[index]];
  }

  return output;
}

/**
 * Sum a contiguous group of entries from a vision vector into a reusable scratch buffer.
 *
 * @param vision - Flat perception vector.
 * @param start - Start index of the group to sum.
 * @param groupLength - Number of entries in the group.
 * @param scratch - Reusable scratch buffer populated with copied values.
 * @returns Numeric sum of the selected group.
 */
export function sumVisionGroup(
  vision: number[],
  start: number,
  groupLength: number,
  scratch: Float64Array,
): number {
  const sanitizedStart = Math.max(0, start | 0);
  const clampedEnd = Math.min(vision.length, sanitizedStart + groupLength);
  if (sanitizedStart >= clampedEnd) return 0;

  let sumAccumulator = 0;
  let writeIndex = 0;
  for (let readIndex = sanitizedStart; readIndex < clampedEnd; readIndex++) {
    const value = vision[readIndex] ?? 0;
    scratch[writeIndex++] = value;
    sumAccumulator += value;
  }

  return sumAccumulator;
}

/**
 * Compute normalized action entropy from direction counts.
 *
 * @param directionCounts - Number of moves taken in each direction.
 * @param logActions - Precomputed normalization factor for the action space.
 * @param scratch - Single-value scratch buffer reused by the caller.
 * @returns Normalized entropy in the range `[0, 1]`.
 */
export function computeActionEntropyFromCounts(
  directionCounts: number[],
  logActions: number,
  scratch: Int32Array,
): number {
  const totalCount =
    directionCounts.reduce((sum, value) => sum + (value | 0), 0) || 1;

  scratch[0] = 0;
  const logFn = Math.log;
  for (
    let directionIndex = 0, directionCount = directionCounts.length;
    directionIndex < directionCount;
    directionIndex++
  ) {
    const count = directionCounts[directionIndex] | 0;
    if (count === 0) continue;
    const probability = count / totalCount;
    scratch[0] -= probability * logFn(probability);
  }

  return scratch[0] / logActions;
}
