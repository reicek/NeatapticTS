/**
 * Scratch pool management helpers extracted from the maze evolution façade.
 *
 * These utilities centralise the logic that grows and shrinks pooled buffers attached to the
 * shared {@link EngineState}. They keep the façade lean by encapsulating heuristics for logits
 * ring sizing, telemetry scratch sizing, and connection flag pooling.
 */
import type { EngineState, TelemetryScratchRequest } from './engineState.types';
import { initialiseTelemetryScratch } from './engineState';

/**
 * Shape describing the parameters used when ensuring the logits ring capacity.
 */
export interface LogitsRingCapacityOptions {
  /** Shared engine state backing the façade. */
  state: EngineState;
  /** Estimated number of recent steps to retain in the logits ring. */
  desiredRecentSteps: number;
  /** Current non-shared logits ring capacity. */
  currentCapacity: number;
  /** Minimum capacity enforced before shrinkage. */
  minimumCapacity: number;
  /** Maximum capacity guard rail. */
  maximumCapacity: number;
  /** Number of action logits per row. */
  actionDimension: number;
  /** Whether shared-array-backed logits were previously active. */
  sharedModeEnabled: boolean;
}

/**
 * Result returned after resizing the logits ring.
 */
export interface LogitsRingCapacityResult {
  /** Effective capacity after the ensure call. */
  capacity: number;
  /** Indicates whether shared-array mode remains enabled. */
  sharedModeEnabled: boolean;
}

/**
 * Grow or shrink the logits ring to accomodate the requested recent-step budget.
 *
 * @param capacityRequest Parameters describing the requested resize.
 * @returns The resulting capacity and whether shared-array mode stayed enabled.
 */
export const ensureLogitsRingCapacity = (
  capacityRequest: LogitsRingCapacityOptions,
): LogitsRingCapacityResult => {
  const {
    state,
    desiredRecentSteps,
    currentCapacity,
    minimumCapacity,
    maximumCapacity,
    actionDimension,
    sharedModeEnabled,
  } = capacityRequest;

  // Step 1: Short-circuit when the caller supplies an invalid or negative window size.
  if (!Number.isFinite(desiredRecentSteps) || desiredRecentSteps < 0)
    return {
      capacity: currentCapacity,
      sharedModeEnabled,
    };

  // Helper: Normalise capacity adjustments to powers of two for quick modular arithmetic.
  const nextPowerOfTwo = (value: number): number => {
    if (value <= 1) return 1;
    return 1 << Math.ceil(Math.log2(value));
  };

  // Step 2: Clamp the minimum and maximum bounds before applying growth heuristics.
  let targetCapacity = currentCapacity;
  const boundedMinimum = Math.max(1, Math.trunc(minimumCapacity));
  const boundedMaximum = Math.max(boundedMinimum, Math.trunc(maximumCapacity));

  // Step 3: Grow the ring aggressively when utilisation approaches the upper bound.
  if (
    desiredRecentSteps > (currentCapacity * 3) / 4 &&
    currentCapacity < boundedMaximum
  ) {
    const desired = Math.min(desiredRecentSteps * 2, boundedMaximum);
    targetCapacity = Math.min(
      nextPowerOfTwo(Math.ceil(desired)),
      boundedMaximum,
    );
  } else if (
    desiredRecentSteps < currentCapacity / 4 &&
    currentCapacity > boundedMinimum
  ) {
    // Step 4: Shrink the ring gradually when the requested history collapses.
    let shrinkCandidate = currentCapacity;
    while (
      shrinkCandidate > boundedMinimum &&
      desiredRecentSteps * 2 <= shrinkCandidate / 2
    )
      shrinkCandidate >>= 1;
    targetCapacity = Math.max(shrinkCandidate, boundedMinimum);
  }

  // Step 5: Exit early when the existing capacity already satisfies the request.
  if (targetCapacity === currentCapacity) {
    return {
      capacity: currentCapacity,
      sharedModeEnabled,
    };
  }

  // Step 6: Reallocate process-local logits rows and restart the write cursor.
  state.scratch.logitsRing = allocateLogitsRing(
    targetCapacity,
    actionDimension,
  );
  state.scratch.logitsRingWriteCursor = 0;

  let sharedEnabledResult = sharedModeEnabled;
  // Step 7: Attempt to rebuild shared views when the feature was previously active.
  if (sharedModeEnabled) {
    sharedEnabledResult = initialiseSharedLogitsRing(state, {
      capacity: targetCapacity,
      actionDimension,
    });
  }

  // Step 8: Clear any dangling shared buffers when the rebuild fails.
  if (!sharedEnabledResult) {
    state.scratch.sharedLogits = undefined;
    state.scratch.sharedLogitsWriteIndex = undefined;
  }

  return {
    capacity: targetCapacity,
    sharedModeEnabled: sharedEnabledResult,
  };
};

/**
 * Parameters passed when attempting to initialise the shared logits ring buffers.
 */
export interface SharedLogitsConfig {
  /** Number of rows to allocate. */
  capacity: number;
  /** Number of actions represented in each row. */
  actionDimension: number;
}

/**
 * Attempt to allocate SharedArrayBuffer-backed logits ring storage.
 *
 * @param state Shared engine state containing the logits buffers.
 * @param config Shared ring configuration (capacity and action dimension).
 * @returns true when shared mode was activated successfully.
 */
export const initialiseSharedLogitsRing = (
  state: EngineState,
  config: SharedLogitsConfig,
): boolean => {
  const { capacity, actionDimension } = config;

  try {
    // Step 1: Abort when the environment cannot create SharedArrayBuffer-backed storage.
    if (typeof SharedArrayBuffer === 'undefined') return false;
    if (globalThis?.crossOriginIsolated !== true) return false;
    if (!Number.isInteger(capacity) || capacity <= 0) return false;

    // Step 2: Allocate a single buffer that stores both the write index and logits payload.
    const totalFloatCount = capacity * actionDimension;
    const indexBytes = Int32Array.BYTES_PER_ELEMENT;
    const floatBytes = Float32Array.BYTES_PER_ELEMENT;
    const sharedBuffer = new SharedArrayBuffer(
      indexBytes + totalFloatCount * floatBytes,
    );

    // Step 3: Project typed views over the shared buffer for atomics and the logits payload.
    const writeIndexView = new Int32Array(sharedBuffer, 0, 1);
    const logitsView = new Float32Array(
      sharedBuffer,
      indexBytes,
      totalFloatCount,
    );

    // Step 4: Reset the write cursor and zero the logits to avoid stale data leaks.
    Atomics.store(writeIndexView, 0, 0);
    logitsView.fill(0);

    // Step 5: Publish the shared views back to the engine state for consumer reuse.
    state.scratch.sharedLogitsWriteIndex = writeIndexView;
    state.scratch.sharedLogits = logitsView;

    return true;
  } catch {
    // Step 6: Swallow failures but ensure the state drops any partially-initialised views.
    state.scratch.sharedLogits = undefined;
    state.scratch.sharedLogitsWriteIndex = undefined;
    return false;
  }
};

/**
 * Build a non-shared logits ring sized to the requested capacity.
 *
 * @param capacity Number of rows to create.
 * @param actionDimension Number of logits stored per row.
 * @returns Array of typed rows representing the ring buffer.
 */
export const allocateLogitsRing = (
  capacity: number,
  actionDimension: number,
): Float32Array[] => {
  const safeCapacity = Math.max(0, Math.trunc(capacity));
  if (safeCapacity === 0) return [];

  const ringRows = new Array<Float32Array>(safeCapacity);
  for (let ringIndex = 0; ringIndex < safeCapacity; ringIndex++) {
    ringRows[ringIndex] = new Float32Array(actionDimension);
  }
  return ringRows;
};

/**
 * Parameters describing the scratch sizing requirements for ensureScratchCapacity.
 */
export interface ScratchCapacityRequest {
  /** Planned population size (used to size pooled arrays). */
  populationSize: number;
  /** Network input dimensionality. */
  inputSize: number;
  /** Network output dimensionality. */
  outputSize: number;
}

/**
 * Grow pooled scratch buffers to conservative sizes for the upcoming run.
 *
 * @param state Shared engine state exposing scratch buffers.
 * @param request Sizing request describing the evolution workload.
 */
export const ensureScratchCapacity = (
  state: EngineState,
  request: ScratchCapacityRequest,
): void => {
  const { populationSize, inputSize, outputSize } = request;

  // Helper: Cap sizes at powers of two to maintain amortised O(1) growth behaviour.
  const nextPowerOfTwo = (value: number): number => {
    if (value <= 1) return 1;
    return 2 ** Math.ceil(Math.log2(value));
  };

  try {
    // Step 1: Compute conservative pool sizes derived from the upcoming workload.
    const desiredSampleCapacity = Math.max(
      32,
      nextPowerOfTwo(Math.max(1, populationSize)),
    );
    const numericBase = Math.max(1, inputSize + outputSize);
    const desiredNumericCapacity = Math.max(64, nextPowerOfTwo(numericBase));

    // Step 2: Ensure the sample pool can hold the expanded population.
    const samplePool = state.scratch.samplePool;
    if (!Array.isArray(samplePool) || samplePool.length === 0) {
      state.scratch.samplePool = new Array(desiredSampleCapacity);
    } else if (samplePool.length < desiredSampleCapacity) {
      samplePool.length = desiredSampleCapacity;
    }

    // Step 3: Resize telemetry scratch buffers using the centralised initialiser.
    const telemetryRequest: TelemetryScratchRequest = {
      actionDimension: desiredNumericCapacity,
      biasCount: desiredNumericCapacity,
      stringBufferLength: desiredNumericCapacity,
      includeHigherMoments: true,
    };
    initialiseTelemetryScratch(telemetryRequest, state);

    // Step 4: Guarantee the bitmap, profiling, and bucket structures are ready for reuse.
    if (!(state.scratch.connectionFlags instanceof Uint8Array)) {
      state.scratch.connectionFlags = new Uint8Array(32);
    }

    if (!(state.scratch.profilingScratch instanceof Float64Array)) {
      state.scratch.profilingScratch = new Float64Array(4);
    }

    if (!Array.isArray(state.scratch.nodeBuckets)) {
      state.scratch.nodeBuckets = [[], [], []];
    }
  } catch {
    // Intentionally swallow failures: scratch sizing is best-effort.
  }
};

/**
 * Shrink oversized scratch buffers once the population size drops significantly.
 *
 * @param state Shared engine state exposing scratch buffers.
 * @param populationSize Current population size used to derive shrink heuristics.
 */
export const maybeShrinkScratch = (
  state: EngineState,
  populationSize: number,
): void => {
  try {
    // Step 1: Ignore shrink attempts when the population is empty or undefined.
    if (!populationSize) return;

    const shrinkFactor = 8;
    const minimumPoolSize = 8;
    // Helper: Mirror growth logic by shrinking to the nearest power of two.
    const nextPowerOfTwo = (value: number) =>
      1 << Math.ceil(Math.log2(Math.max(1, value)));

    // Step 2: Establish the scaled-down capacity target we want the pools to match.
    const desiredCapacity = nextPowerOfTwo(
      Math.max(minimumPoolSize, populationSize),
    );

    const { scratch } = state;

    // Step 3: Replace sorted index buffers when they dwarf the desired capacity.
    const sortedIndexBuffer = scratch.sortedIndexBuffer;
    if (
      Array.isArray(sortedIndexBuffer) &&
      sortedIndexBuffer.length > populationSize * shrinkFactor
    ) {
      scratch.sortedIndexBuffer = new Array(desiredCapacity);
    }

    // Step 4: Trim the sample pool while preserving existing array instances.
    const samplePool = scratch.samplePool;
    if (
      Array.isArray(samplePool) &&
      samplePool.length > populationSize * shrinkFactor
    ) {
      samplePool.length = desiredCapacity;
      scratch.samplePool = samplePool;
    }

    // Step 5: Rebuild typed telemetry buffers when they exceed the shrink heuristics.
    const exponentScratch = scratch.exps;
    if (
      exponentScratch instanceof Float64Array &&
      exponentScratch.length > populationSize * shrinkFactor
    ) {
      const smaller = new Float64Array(desiredCapacity);
      smaller.set(
        exponentScratch.subarray(
          0,
          Math.min(exponentScratch.length, desiredCapacity),
        ),
      );
      scratch.exps = smaller;
    }

    const biasScratch = scratch.biasTelemetryScratch;
    if (
      biasScratch instanceof Float64Array &&
      biasScratch.length > populationSize * shrinkFactor
    ) {
      const smaller = new Float64Array(desiredCapacity);
      smaller.set(
        biasScratch.subarray(0, Math.min(biasScratch.length, desiredCapacity)),
      );
      scratch.biasTelemetryScratch = smaller;
    }

    // Step 6: Downsize dense index buffers to the compact capacity.
    const nodeIndexBuffer = scratch.nodeIndexBuffer;
    if (
      nodeIndexBuffer instanceof Int32Array &&
      nodeIndexBuffer.length > populationSize * shrinkFactor
    ) {
      const smaller = new Int32Array(desiredCapacity);
      smaller.set(
        nodeIndexBuffer.subarray(
          0,
          Math.min(nodeIndexBuffer.length, desiredCapacity),
        ),
      );
      scratch.nodeIndexBuffer = smaller;
    }
  } catch {
    // Maintenance helper is best-effort; swallow any failures.
  }
};

/**
 * Ensure the recurrent/gated detection bitmap has sufficient capacity.
 *
 * @param state Shared engine state containing the pooled bitmap.
 * @param minimumCapacity Minimum number of entries required by the caller.
 * @returns Int8Array bitmap or null when allocation failed.
 */
export const ensureConnFlagsCapacity = (
  state: EngineState,
  minimumCapacity: number,
): Int8Array | null => {
  try {
    // Step 1: Normalise the caller-provided capacity to a safe, non-negative integer.
    const requiredCapacity = Math.max(
      0,
      Math.trunc(Number(minimumCapacity) || 0),
    );

    // Step 2: Return a cached empty bitmap when the caller only needs zero entries.
    if (requiredCapacity === 0) {
      const empty = new Int8Array(0);
      state.scratch.connectionFlagBitmap = empty;
      return empty;
    }

    const existingBitmap = state.scratch.connectionFlagBitmap;
    // Step 3: Reuse the existing bitmap when it already satisfies the capacity requirement.
    if (
      existingBitmap instanceof Int8Array &&
      existingBitmap.length >= requiredCapacity
    ) {
      return existingBitmap;
    }

    // Step 4: Grow the bitmap using power-of-two increments to reduce realloc churn.
    let newCapacity = 1;
    while (newCapacity < requiredCapacity) newCapacity <<= 1;

    // Step 5: Allocate the new bitmap and copy over any preserved metadata bits.
    const newBitmap = new Int8Array(newCapacity);
    if (existingBitmap instanceof Int8Array && existingBitmap.length > 0) {
      const preservedLength = Math.min(existingBitmap.length, newBitmap.length);
      newBitmap.set(existingBitmap.subarray(0, preservedLength), 0);
    }

    // Step 6: Publish the new bitmap back into the shared scratch state.
    state.scratch.connectionFlagBitmap = newBitmap;
    return newBitmap;
  } catch (allocationError: unknown) {
    // Step 7: Report failures softly by clearing the cached bitmap and returning null.
    state.scratch.connectionFlagBitmap = undefined;
    void allocationError;
    return null;
  }
};
