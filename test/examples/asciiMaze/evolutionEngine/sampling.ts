/**
 * Sampling and history helpers extracted from the ASCII maze evolution façade.
 *
 * Responsibilities:
 * 1. Provide allocation-light array sampling utilities that reuse the shared `EngineState` scratch pools.
 * 2. Expose history helpers that mirror the façade behaviour while keeping pooled buffers centralised.
 * 3. Centralise RNG parameter resolution for sampling paths to keep behaviour deterministic under shared state.
 */

import { EngineState } from './engineState';
import { drawFastRandom, resolveRngParameters } from './rngAndTiming';
import { MazeUtils } from '../mazeUtils';

/**
 * Sample `sampleCount` items (with replacement) from `source` into the pooled scratch buffer.
 *
 * Steps:
 * 1. Validate the input array and normalise `sampleCount` to an integer.
 * 2. Grow the shared `sampleResultBuffer` to the next power of two when capacity is insufficient.
 * 3. Fill the pooled buffer using the shared fast RNG and truncate the logical length to `sampleCount`.
 *
 * @template T Type of the array elements.
 * @param state Shared engine state providing pooled scratch buffers.
 * @param source Source array to sample from.
 * @param sampleCount Number of samples to draw (with replacement).
 * @returns Pooled ephemeral array containing the sampled elements.
 * @example
 * const picks = sampleArray(sharedState, population, 16);
 * const safeCopy = [...picks];
 */
export const sampleArray = <T>(
  state: EngineState,
  source: T[],
  sampleCount: number,
): T[] => {
  // Step 1: Fast validation and normalisation.
  if (!Array.isArray(source) || sampleCount <= 0) return [];
  const normalisedCount = Math.floor(sampleCount);
  const sourceLength = source.length | 0;
  if (normalisedCount <= 0 || sourceLength === 0) return [];

  const scratch = state.scratch;

  // Step 2: Ensure pooled output capacity using power-of-two growth.
  if (normalisedCount > scratch.sampleResultBuffer.length) {
    const nextSize = 1 << Math.ceil(Math.log2(normalisedCount));
    scratch.sampleResultBuffer = new Array(nextSize);
  }
  const pooledResults = scratch.sampleResultBuffer as T[];

  // Step 3: Populate the pooled buffer via the shared RNG (with replacement sampling).
  const rngParameters = resolveRngParameters();
  for (let sampleIndex = 0; sampleIndex < normalisedCount; sampleIndex++) {
    pooledResults[sampleIndex] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
  }

  pooledResults.length = normalisedCount;
  return pooledResults;
};

/**
 * Sample up to `sampleCount` items (with replacement) into the shared `samplePool` buffer.
 *
 * Steps:
 * 1. Validate inputs and ensure the pooled buffer exists.
 * 2. Grow the buffer by powers of two when more capacity is required.
 * 3. Fill the buffer with random selections and return the number of items written.
 *
 * @template T Type of the array elements.
 * @param state Shared engine state providing the pooled buffer.
 * @param source Source array to sample from.
 * @param sampleCount Number of elements requested (floored to an integer).
 * @returns Number of sampled elements written into the pooled buffer.
 * @example
 * const sampled = sampleIntoScratch(sharedState, population, 24);
 * for (let index = 0; index < sampled; index++) {
 *   processGenome(sharedState.scratch.samplePool[index]);
 * }
 */
export const sampleIntoScratch = <T>(
  state: EngineState,
  source: T[],
  sampleCount: number,
): number => {
  // Step 1: Validate inputs early.
  if (!Array.isArray(source) || sampleCount <= 0) return 0;

  const normalisedCount = Math.floor(sampleCount);
  const sourceLength = source.length | 0;
  if (sourceLength === 0 || normalisedCount <= 0) return 0;

  const scratch = state.scratch;
  let pooledBuffer = scratch.samplePool;
  if (!Array.isArray(pooledBuffer)) {
    pooledBuffer = [];
    scratch.samplePool = pooledBuffer;
  }

  // Step 2: Grow pooled buffer capacity using power-of-two increments.
  if (pooledBuffer.length < normalisedCount) {
    let newCapacity = pooledBuffer.length > 0 ? pooledBuffer.length : 1;
    while (newCapacity < normalisedCount) newCapacity <<= 1;
    const expandedBuffer: T[] = new Array(newCapacity);
    for (
      let existingIndex = 0;
      existingIndex < pooledBuffer.length;
      existingIndex++
    ) {
      expandedBuffer[existingIndex] = pooledBuffer[existingIndex]!;
    }
    scratch.samplePool = expandedBuffer;
    pooledBuffer = expandedBuffer;
  }

  // Step 3: Populate the pooled buffer with random selections.
  const writeCount = Math.min(normalisedCount, pooledBuffer.length);
  const rngParameters = resolveRngParameters();
  const blockLimit = writeCount & ~3;
  let writeIndex = 0;
  while (writeIndex < blockLimit) {
    pooledBuffer[writeIndex++] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
    pooledBuffer[writeIndex++] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
    pooledBuffer[writeIndex++] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
    pooledBuffer[writeIndex++] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
  }
  while (writeIndex < writeCount) {
    pooledBuffer[writeIndex++] =
      source[(drawFastRandom(state, rngParameters) * sourceLength) | 0];
  }

  return writeCount;
};

/**
 * Sample from a suffix of `source` starting at `segmentStart` into the pooled buffer.
 *
 * Steps:
 * 1. Clamp indices and ensure there is a non-empty segment.
 * 2. Ensure the pooled buffer can store the requested sample count.
 * 3. Fill the buffer via the shared RNG, returning the number of items written.
 *
 * @template T Type of the array elements.
 * @param state Shared engine state providing pooled sampling buffers.
 * @param source Source array to sample from.
 * @param segmentStart Inclusive start index of the segment.
 * @param sampleCount Number of items to sample (with replacement).
 * @returns Number of elements written into the pooled buffer.
 * @example
 * const written = sampleSegmentIntoScratch(sharedState, population, elitismCount, 12);
 * const genome = sharedState.scratch.samplePool[0];
 */
export const sampleSegmentIntoScratch = <T>(
  state: EngineState,
  source: T[],
  segmentStart: number,
  sampleCount: number,
): number => {
  if (!Array.isArray(source) || sampleCount <= 0) return 0;

  const totalLength = source.length | 0;
  const clampedStart = Math.max(0, segmentStart | 0);
  if (clampedStart >= totalLength) return 0;

  const segmentSize = totalLength - clampedStart;
  if (segmentSize <= 0) return 0;

  const normalisedCount = Math.max(0, Math.floor(sampleCount));
  if (normalisedCount === 0) return 0;

  const scratch = state.scratch;
  let pooledBuffer = Array.isArray(scratch.samplePool)
    ? (scratch.samplePool as T[])
    : (scratch.samplePool = [] as T[]);

  if (pooledBuffer.length < normalisedCount) {
    let newCapacity = pooledBuffer.length > 0 ? pooledBuffer.length : 1;
    while (newCapacity < normalisedCount) newCapacity <<= 1;
    const expandedBuffer: T[] = new Array(newCapacity);
    for (
      let existingIndex = 0;
      existingIndex < pooledBuffer.length;
      existingIndex++
    ) {
      expandedBuffer[existingIndex] = pooledBuffer[existingIndex]!;
    }
    scratch.samplePool = expandedBuffer;
    pooledBuffer = expandedBuffer;
  }

  const writeCount = Math.min(normalisedCount, pooledBuffer.length);
  const rngParameters = resolveRngParameters();
  const blockLimit = writeCount & ~3;
  let writeIndex = 0;
  while (writeIndex < blockLimit) {
    pooledBuffer[writeIndex++] =
      source[
        clampedStart +
          ((drawFastRandom(state, rngParameters) * segmentSize) | 0)
      ];
    pooledBuffer[writeIndex++] =
      source[
        clampedStart +
          ((drawFastRandom(state, rngParameters) * segmentSize) | 0)
      ];
    pooledBuffer[writeIndex++] =
      source[
        clampedStart +
          ((drawFastRandom(state, rngParameters) * segmentSize) | 0)
      ];
    pooledBuffer[writeIndex++] =
      source[
        clampedStart +
          ((drawFastRandom(state, rngParameters) * segmentSize) | 0)
      ];
  }
  while (writeIndex < writeCount) {
    pooledBuffer[writeIndex++] =
      source[
        clampedStart +
          ((drawFastRandom(state, rngParameters) * segmentSize) | 0)
      ];
  }

  return writeCount;
};

/**
 * Extract the last `count` items from `source` into the shared tail buffer.
 *
 * Steps:
 * 1. Validate the source array and clamp the requested tail length.
 * 2. Grow the pooled tail buffer to the next power of two when necessary.
 * 3. Copy the suffix into the pooled buffer and trim its logical length.
 *
 * @template T Type of the array elements.
 * @param state Shared engine state providing the tail history buffer.
 * @param source Source array reference.
 * @param count Number of trailing items requested.
 * @returns Pooled array containing the requested tail slice.
 * @example
 * const recent = getTail(sharedState, telemetryLog, 40);
 */
export const getTail = <T>(
  state: EngineState,
  source: T[] | undefined,
  count: number,
): T[] => {
  if (
    !Array.isArray(source) ||
    source.length === 0 ||
    !Number.isFinite(count) ||
    count <= 0
  )
    return [];

  const normalisedCount = Math.floor(count);
  const takeCount =
    normalisedCount >= source.length ? source.length : normalisedCount;
  if (takeCount === 0) return [];

  const scratch = state.scratch;
  if (takeCount > scratch.tailHistoryBuffer.length) {
    const nextSize = 1 << Math.ceil(Math.log2(takeCount));
    scratch.tailHistoryBuffer = new Array(nextSize);
  }

  const tailBuffer = scratch.tailHistoryBuffer as T[];
  const startIndex = source.length - takeCount;
  for (let elementIndex = 0; elementIndex < takeCount; elementIndex++) {
    tailBuffer[elementIndex] = source[startIndex + elementIndex]!;
  }

  tailBuffer.length = takeCount;
  return tailBuffer;
};

/**
 * Proxy to {@link MazeUtils.pushHistory} for consistency with the façade API.
 *
 * @template T Type of the history entries.
 * @param buffer Existing history buffer (may be undefined).
 * @param value Value to append to the buffer.
 * @param maxLength Maximum allowed buffer length.
 * @returns Updated history buffer with `value` appended and trimmed to `maxLength`.
 * @example
 * const history = pushHistory(existingHistory, snapshot, 20);
 */
export const pushHistory = <T>(
  buffer: T[] | undefined,
  value: T,
  maxLength: number,
): T[] => MazeUtils.pushHistory(buffer, value, maxLength);
