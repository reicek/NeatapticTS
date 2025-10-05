/**
 * populationDynamics.ts
 *
 * Population-level dynamics for NEAT evolution: generation state management,
 * population expansion/compaction, mutation application, sorting, species tracking,
 * and anti-collapse recovery.
 *
 * Responsibilities:
 * - Plateau detection and simplify-phase orchestration
 * - Population expansion with parent sampling and mutation
 * - Genome sorting by fitness (iterative quicksort with pooled scratch)
 * - Species history tracking and collapse detection
 * - Connection compaction and anti-collapse recovery
 *
 * All functions accept `EngineState` to access shared scratch buffers and RNG.
 * Follows ES2023 idioms: `toSorted`, `.at(-1)`, numeric separators, etc.
 */

import { methods } from '../../../../src/neataptic';
import type { EngineState } from './engineState';
import { drawFastRandom, resolveRngParameters } from './rngAndTiming';
import { applySimplifyPruningToPopulation } from './populationPruning';
import { pushHistory } from './sampling';
import {
  accumulateProfilingDuration,
  isProfilingDetailsEnabled,
  profilingStartTimestamp,
} from './rngAndTiming';
import type { NetworkNode, NetworkConnection } from '../interfaces';

/** Shared empty array to avoid repeated allocations for missing/invalid arrays. */
// Type assertion: Empty array for generic fallback when arrays are invalid
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const EMPTY_VEC: any[] = [];

/** Small numeric epsilon for variance checks (collapse detection). */
const NUMERIC_EPSILON_SMALL = 0.01;

/** Probability for applying 2 mutations instead of 1 (50% by default). */
const DEFAULT_HALF_PROB = 0.5;

/** Fraction of sorted population to use as parent pool (top 25%). */
const DEFAULT_PARENT_FRACTION = 0.25;

/** Quicksort partition size threshold for insertion sort fallback. */
const QS_SMALL_THRESHOLD = 24;

/** Half-range for output bias reset in anti-collapse recovery (±0.1). */
const BIAS_RESET_HALF_RANGE = 0.1;

/** Half-range for connection weight reset in anti-collapse recovery (±0.2). */
const CONN_WEIGHT_RESET_HALF_RANGE = 0.2;

// ========== Type Definitions for Runtime Network Structures ==========

/** Network genome with dynamic runtime properties */
interface RuntimeGenome {
  nodes?: NetworkNode[];
  connections?: NetworkConnection[];
  clone?: () => RuntimeGenome;
  mutate?: (method: unknown) => void;
  _id?: number;
  _parentId?: number;
  [key: string]: unknown;
}

/** Mutation operation from NEAT driver */
interface MutationOperation {
  length?: number;
  [key: string]: unknown;
}

/** Typed or array-based index buffer for sorting */
type IndexBuffer = Uint32Array | Int32Array | number[];

// ========== End Type Definitions ==========

/** Cached reference to mutation ops array (invalidated if driver replaces the reference). */
let cachedMutationOps: MutationOperation[] | null = null;

/**
 * Update plateau state based on current fitness vs baseline.
 *
 * Behaviour:
 * - Compares `fitness` against `lastBestFitnessForPlateau + threshold`.
 * - If improved, resets plateau counter to 0 and updates baseline.
 * - Otherwise, increments plateau counter (capped at a safe maximum).
 *
 * @param fitness - Current best fitness (must be finite).
 * @param lastBestFitnessForPlateau - Previous baseline fitness.
 * @param plateauCounter - Current consecutive plateau generations.
 * @param plateauImprovementThreshold - Minimum improvement to reset plateau.
 * @returns Updated `{ plateauCounter, lastBestFitnessForPlateau }`.
 *
 * @example
 * const state = updatePlateauState(1.23, 1.1, 3, 0.05);
 * // state => { plateauCounter: 0, lastBestFitnessForPlateau: 1.23 }
 */
export const updatePlateauState = (
  fitness: number,
  lastBestFitnessForPlateau: number,
  plateauCounter: number,
  plateauImprovementThreshold: number,
): { plateauCounter: number; lastBestFitnessForPlateau: number } => {
  // Step 1: Validate & normalise numeric inputs.
  if (!Number.isFinite(fitness)) {
    return { plateauCounter, lastBestFitnessForPlateau };
  }

  const baseline = Number.isFinite(lastBestFitnessForPlateau)
    ? lastBestFitnessForPlateau
    : -Infinity;

  const improvementThreshold =
    Number.isFinite(plateauImprovementThreshold) &&
    plateauImprovementThreshold > 0
      ? plateauImprovementThreshold
      : 0;

  let counter =
    Number.isFinite(plateauCounter) && plateauCounter >= 0
      ? Math.floor(plateauCounter)
      : 0;

  // Step 2: Compare fitness against baseline + threshold.
  if (fitness > baseline + improvementThreshold) {
    return { plateauCounter: 0, lastBestFitnessForPlateau: fitness };
  }

  // Step 3: No sufficient improvement — increment plateau counter (capped).
  const SAFE_CAP = 0x1fff_ffff; // ~536 million
  counter = Math.min(SAFE_CAP, counter + 1);

  return { plateauCounter: counter, lastBestFitnessForPlateau: baseline };
};

/**
 * Decide whether to start a simplify phase based on plateau duration.
 *
 * Behaviour:
 * - Returns requested simplify duration if plateau threshold is reached.
 * - Skips simplify in browser environments (presence of `window` global).
 *
 * @param plateauCounter - Observed consecutive plateau generations.
 * @param plateauGenerations - Threshold to trigger simplify.
 * @param simplifyDuration - Requested simplify phase length.
 * @returns Number of generations to run simplify (0 means "do not start").
 *
 * @example
 * const duration = maybeStartSimplify(plateauCount, 10, 5);
 * if (duration > 0) {
 *   // Begin simplify for `duration` generations
 * }
 */
export const maybeStartSimplify = (
  plateauCounter: number,
  plateauGenerations: number,
  simplifyDuration: number,
): number => {
  // Step 1: Defensive normalization.
  const observedPlateau = Number.isFinite(plateauCounter)
    ? Math.max(0, Math.floor(plateauCounter))
    : 0;
  const requiredPlateau = Number.isFinite(plateauGenerations)
    ? Math.max(0, Math.floor(plateauGenerations))
    : 0;
  const requestedDuration = Number.isFinite(simplifyDuration)
    ? Math.max(0, Math.floor(simplifyDuration))
    : 0;

  // Step 2: Threshold check.
  if (observedPlateau < requiredPlateau) return 0;

  // Step 3: Environment gate (skip in browsers).
  try {
    if (typeof window !== 'undefined') return 0;
  } catch {
    // Accessing `window` threw; continue (non-browser host).
  }

  // Step 4: All checks passed.
  return requestedDuration;
};

/**
 * Run a single simplify/pruning generation if conditions permit.
 *
 * Steps:
 * 1. Normalize inputs and perform fast exits for zero remaining or invalid population.
 * 2. Environment gate: skip pruning in browser-like hosts.
 * 3. Record profiling start time when enabled.
 * 4. Execute pruning across the population (best-effort per-genome).
 * 5. Record profiling delta and return remaining generations decremented.
 *
 * @param state - Shared engine state for RNG and profiling.
 * @param neat - NEAT instance with `population` array.
 * @param simplifyRemaining - Remaining simplify generations.
 * @param simplifyStrategy - Strategy identifier for pruning.
 * @param simplifyPruneFraction - Fraction in [0,1] controlling pruning aggressiveness.
 * @returns Remaining simplify generations after executing one cycle (0 means done).
 *
 * @example
 * const remaining = runSimplifyCycle(state, neat, 5, 'pruneWeak', 0.2);
 */
export const runSimplifyCycle = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  simplifyRemaining: number,
  simplifyStrategy: string,
  simplifyPruneFraction: number,
): number => {
  // Step 1: Defensive normalization & quick exits.
  const remainingGens = Number.isFinite(simplifyRemaining)
    ? Math.max(0, Math.floor(simplifyRemaining))
    : 0;
  if (remainingGens === 0) return 0;
  if (!neat || !Array.isArray(neat.population) || neat.population.length === 0)
    return 0;

  // Step 2: Environment gate (skip in browser).
  try {
    if (typeof window !== 'undefined') return remainingGens;
  } catch {
    // Continue (non-browser).
  }

  // Step 3: Profiling start.
  const profilingEnabled = isProfilingDetailsEnabled(state);
  const profileStartMs = profilingEnabled ? profilingStartTimestamp() : 0;

  // Step 4: Apply pruning.
  applySimplifyPruningToPopulation({
    neat,
    simplifyStrategy,
    simplifyPruneFraction,
  });

  // Step 5: Record profiling delta.
  if (profilingEnabled) {
    const elapsedMs = profilingStartTimestamp() - profileStartMs || 0;
    accumulateProfilingDuration(state, 'simplify', elapsedMs);
  }

  return Math.max(0, remainingGens - 1);
}

/**
 * Handle simplify entry and per-generation advance.
 *
 * Behaviour:
 * - Decides when to enter a simplification phase and runs one simplify cycle per generation.
 * - Delegates start decision to `maybeStartSimplify` and per-generation work to `runSimplifyCycle`.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver instance.
 * @param plateauCounter - Current plateau counter.
 * @param plateauGenerations - Window size for plateau decision.
 * @param simplifyDuration - Requested simplify duration.
 * @param simplifyMode - Current simplify active flag.
 * @param simplifyRemaining - Remaining simplify generations.
 * @param simplifyStrategy - Strategy identifier.
 * @param simplifyPruneFraction - Pruning fraction.
 * @returns Updated `{ simplifyMode, simplifyRemaining, plateauCounter }`.
 *
 * @example
 * const state = handleSimplifyState(engineState, neat, 3, 10, 5, false, 0, 'aggressive', 0.2);
 */
export const handleSimplifyState = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  plateauCounter: number,
  plateauGenerations: number,
  simplifyDuration: number,
  simplifyMode: boolean,
  simplifyRemaining: number,
  simplifyStrategy: string,
  simplifyPruneFraction: number,
): {
  simplifyMode: boolean;
  simplifyRemaining: number;
  plateauCounter: number;
} => {
  // Step 1: Defensive normalization.
  let counter =
    Number.isFinite(plateauCounter) && plateauCounter >= 0
      ? Math.floor(plateauCounter)
      : 0;
  const windowSize =
    Number.isFinite(plateauGenerations) && plateauGenerations > 0
      ? Math.floor(plateauGenerations)
      : 0;
  const requestedDuration =
    Number.isFinite(simplifyDuration) && simplifyDuration > 0
      ? Math.floor(simplifyDuration)
      : 0;
  let remaining =
    Number.isFinite(simplifyRemaining) && simplifyRemaining > 0
      ? Math.floor(simplifyRemaining)
      : 0;
  let active = Boolean(simplifyMode);

  // Step 2: When not active, check if we should start.
  if (!active) {
    try {
      const startBudget = maybeStartSimplify(
        counter,
        windowSize,
        requestedDuration,
      );
      if (Number.isFinite(startBudget) && startBudget > 0) {
        active = true;
        remaining = Math.floor(startBudget);
        counter = 0; // Reset plateau counter when starting.
      }
    } catch {
      // Best-effort: swallow and continue without starting.
    }
  }

  // Step 3: When active, run one cycle.
  if (active) {
    try {
      remaining = runSimplifyCycle(
        state,
        neat,
        remaining,
        simplifyStrategy,
        simplifyPruneFraction,
      );
      if (!Number.isFinite(remaining) || remaining <= 0) {
        active = false;
        remaining = 0;
      }
    } catch {
      // Best-effort: exit simplify mode on error.
      active = false;
      remaining = 0;
    }
  }

  return {
    simplifyMode: active,
    simplifyRemaining: remaining,
    plateauCounter: counter,
  };
}

/**
 * Expand the population by creating children from top-performing parents.
 *
 * Steps:
 * 1. Prepare working sets (population reference, sorted parent indices, parent pool size).
 * 2. Sample parents uniformly from the top parent pool.
 * 3. Create children via `createChildFromParent` (per-child failures ignored).
 * 4. Update `neat.options.popsize` and emit status line.
 *
 * @param state - Shared engine state for RNG.
 * @param neat - NEAT driver with `population` and `options`.
 * @param targetAdd - Desired number of new genomes.
 * @param safeWrite - Logging callback.
 * @param completedGenerations - Generation counter for telemetry.
 *
 * @example
 * expandPopulation(state, neat, 10, msg => process.stdout.write(msg), currentGen);
 */
export const expandPopulation = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  targetAdd: number,
  safeWrite: (msg: string) => void,
  completedGenerations: number,
) => {
  // Step 0: Defensive normalization.
  const additionsWanted = Number.isFinite(targetAdd)
    ? Math.max(0, Math.floor(targetAdd))
    : 0;
  if (additionsWanted <= 0) return;

  // Step 1: Prepare working sets.
  const { populationRef, sortedIdx, parentPoolSize } = prepareExpansion(
    state,
    neat,
  );

  // Fast exit when no parents.
  if (!Array.isArray(populationRef) || populationRef.length === 0) return;
  if (!Number.isFinite(parentPoolSize) || parentPoolSize === 0) return;

  // Step 2: Create children by sampling parents.
  const rngParameters = resolveRngParameters();
  const sampleUniformParent = () => drawFastRandom(state, rngParameters);
  for (let childIndex = 0; childIndex < additionsWanted; childIndex++) {
    const sampledParentIndex = Math.floor(
      sampleUniformParent() * parentPoolSize,
    );
    const parentGenome = populationRef[sortedIdx[sampledParentIndex]];

    try {
      createChildFromParent(state, neat, parentGenome);
    } catch {
      // Best-effort: ignore per-child failures.
    }
  }

  // Step 3: Finalize bookkeeping.
  if (!neat.options) neat.options = {};
  const currentPopSize = Array.isArray(neat.population)
    ? neat.population.length
    : 0;
  neat.options.popsize = currentPopSize;
  try {
    safeWrite?.(
      `[DYNAMIC_POP] Expanded population to ${currentPopSize} at gen ${completedGenerations}\n`,
    );
  } catch {
    // Swallow logging errors.
  }
}

/**
 * Prepare working sets for population expansion.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver with `population`.
 * @returns Object with `populationRef`, `sortedIdx`, `parentPoolSize`.
 *
 * @example
 * const { populationRef, sortedIdx, parentPoolSize } = prepareExpansion(state, neat);
 */
export const prepareExpansion = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
): {
  // Type assertion: Population array contains network genomes
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  populationRef: any[];
  sortedIdx: number[];
  parentPoolSize: number;
} => {
  // Type assertion: Population array contains network genomes
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const populationRef: any[] = Array.isArray(neat?.population)
    ? neat.population
    : [];

  // Fast path: empty population.
  if (populationRef.length === 0) {
    return { populationRef, sortedIdx: [], parentPoolSize: 0 };
  }

  // Sort indices by descending score.
  const sortedIdx = getSortedIndicesByScore(state, populationRef);

  // Compute parent pool size.
  const desiredParentCount = Math.ceil(
    sortedIdx.length * DEFAULT_PARENT_FRACTION,
  );
  const parentCount = Math.max(2, desiredParentCount);
  const parentPoolSize = Math.min(parentCount, sortedIdx.length);

  return { populationRef, sortedIdx, parentPoolSize };
};

/**
 * Determine how many mutation operations to attempt (1 or 2).
 *
 * @param state - Shared engine state for RNG.
 * @returns 1 or 2 based on random sample.
 */
export const determineMutateCount = (state: EngineState): number => {
  const rngParameters = resolveRngParameters();
  const uniformSample = drawFastRandom(state, rngParameters);
  return 1 + (uniformSample < DEFAULT_HALF_PROB ? 1 : 0);
};

/**
 * Apply up to `mutateCount` distinct mutation operations to `clone`.
 *
 * Behaviour:
 * 1. Uses cached mutation operation array from `getMutationOps`.
 * 2. Selects up to `mutateCount` unique operations via partial Fisher–Yates shuffle.
 * 3. For small `mutateCount` values, uses unrolled fast path.
 *
 * @param state - Shared engine state for RNG and scratch buffers.
 * @param clone - Genome-like object with `mutate(op)` method.
 * @param neat - NEAT driver for resolving mutation ops.
 * @param mutateCount - Desired number of distinct mutation ops.
 *
 * @example
 * applyMutationsToClone(state, someClone, neat, 2);
 */
export const applyMutationsToClone = (
  state: EngineState,
  // Type assertion: Genome/network with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  clone: any,
  // Type assertion: NEAT driver with dynamic mutation methods
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  mutateCount: number,
) => {
  // Step 1: Resolve mutation operations.
  const mutationOps = getMutationOps(neat);
  const operationCount = mutationOps.length | 0;
  if (operationCount === 0) return;

  // Step 2: Ensure pooled index buffer capacity.
  let indexBuffer = state.scratch.mutationOperatorIndices;
  if (indexBuffer.length < operationCount) {
    const nextSize = 1 << Math.ceil(Math.log2(operationCount));
    state.scratch.mutationOperatorIndices = indexBuffer = new Uint16Array(
      nextSize,
    );
  }

  // Step 3: Initialize identity permutation.
  for (let writeIndex = 0; writeIndex < operationCount; writeIndex++) {
    indexBuffer[writeIndex] = writeIndex;
  }

  // Step 4: Normalize apply count.
  const wanted = Math.max(0, Math.floor(mutateCount || 0));
  const toApply = Math.min(wanted, operationCount);
  if (toApply === 0) return;

  // Step 5: Fast paths for tiny counts.
  if (toApply === 1) {
    const opIndex = indexBuffer[0];
    const op = mutationOps[opIndex];
    if (typeof clone?.mutate === 'function') clone.mutate(op);
    return;
  }

  if (toApply === 2) {
    const firstIndex = indexBuffer[0];
    const secondIndex = indexBuffer[1];
    if (typeof clone?.mutate === 'function') {
      clone.mutate(mutationOps[firstIndex]);
      clone.mutate(mutationOps[secondIndex]);
    }
    return;
  }

  // Step 6: Partial Fisher–Yates for k distinct picks.
  const mutationRngParameters = resolveRngParameters();
  for (let selectionCursor = 0; selectionCursor < toApply; selectionCursor++) {
    const remaining = operationCount - selectionCursor;
    const pickOffset = Math.floor(
      drawFastRandom(state, mutationRngParameters) * remaining,
    );
    const pickIndex = selectionCursor + pickOffset;

    // Swap selected element into prefix position.
    const selectedOpIndex = indexBuffer[pickIndex];
    indexBuffer[pickIndex] = indexBuffer[selectionCursor];
    indexBuffer[selectionCursor] = selectedOpIndex;

    // Apply mutation immediately.
    const op = mutationOps[selectedOpIndex];
    if (typeof clone?.mutate === 'function') clone.mutate(op);
  }
};

/**
 * Register a cloned genome with the NEAT driver's bookkeeping.
 *
 * @param neat - NEAT driver instance.
 * @param clone - Cloned genome object.
 * @param parentId - Optional parent ID for lineage tracking.
 *
 * @example
 * registerClone(neat, genomeClone, parentId);
 */
export const registerClone = (
  // Type assertion: NEAT driver with dynamic population methods
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  // Type assertion: Genome/network with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  clone: any,
  // Type assertion: Parent ID can be number, string, or undefined
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parentId?: any,
) => {
  try {
    if (!neat || !clone) return;

    // Ensure population array exists.
    if (!Array.isArray(neat.population)) neat.population = [];

    // Add clone to population.
    neat.population.push(clone);

    // Optionally track parent ID.
    if (parentId != null) {
      const runtimeClone = clone as RuntimeGenome;
      runtimeClone._parentId = parentId;
    }
  } catch {
    // Best-effort: swallow errors.
  }
};

/**
 * Create a child genome from a parent via cloning and mutation.
 *
 * Behaviour:
 * 1. Clone the parent genome (with or without ID tracking).
 * 2. Determine mutation count (1 or 2).
 * 3. Apply mutations to the clone.
 * 4. Register the clone with the NEAT driver.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver instance.
 * @param parent - Parent genome object.
 *
 * @example
 * createChildFromParent(state, neat, someParentGenome);
 */
export const createChildFromParent = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamic mutation/population methods
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  // Type assertion: Parent genome with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parent: any,
  // Type assertion: Returns child genome with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any => {
  try {
    if (!parent) return;

    // Step 1: Clone parent.
    // Type assertion: Clone can be any genome/network type
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let clone: any = null;
    // Type assertion: Parent ID can be any type
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parentId = (parent as any)?._id;

    // Attempt clone with ID first (preferred path).
    if (typeof parent.clone === 'function' && parentId != null) {
      try {
        clone = parent.clone(parentId);
        if (clone) {
          registerClone(neat, clone, parentId);
        }
      } catch {
        // Fallback to no-ID clone.
        clone = null;
      }
    }

    // Fallback: clone without ID.
    if (!clone && typeof parent.clone === 'function') {
      clone = parent.clone();
      if (clone) {
        const runtimeParent = parent as RuntimeGenome;
        registerClone(neat, clone, runtimeParent?._id);
      }
    }

    if (!clone) return;

    // Step 2: Determine mutation count.
    const mutateCount = determineMutateCount(state);

    // Step 3: Apply mutations.
    applyMutationsToClone(state, clone, neat, mutateCount);
  } catch {
    // Best-effort: swallow errors.
  }
};

/**
 * Sort population indices by descending score using iterative quicksort.
 *
 * Implementation details:
 * 1. Uses pooled scratch buffers (Int32Array or number[]) to avoid allocations.
 * 2. Initializes identity permutation [0,1,2,...].
 * 3. Sorts by descending `population[idx].score` with median-of-three pivot.
 * 4. Falls back to insertion sort for small partitions.
 *
 * @param state - Shared engine state for scratch buffers.
 * @param population - Population array with `.score` property.
 * @returns Sorted indices (highest score first). Empty array when input empty.
 *
 * @example
 * const indices = getSortedIndicesByScore(state, population);
 */
export const getSortedIndicesByScore = (
  state: EngineState,
  // Type assertion: Population array contains network genomes with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  population: any[],
): number[] => {
  // Step 1: Validate inputs.
  const populationLength = population.length | 0;
  if (populationLength === 0) return [];

  // Step 2: Decide typed vs number[] scratch.
  let useTypedScratch = false;
  let typedScratchBuf = state.scratch.sortedIndexTypedArray;
  if (typedScratchBuf && typedScratchBuf.length >= populationLength) {
    useTypedScratch = true;
  } else if (!typedScratchBuf && populationLength > 512) {
    const allocSize = 1 << Math.ceil(Math.log2(populationLength));
    state.scratch.sortedIndexTypedArray = typedScratchBuf = new Int32Array(
      allocSize,
    );
    useTypedScratch = true;
  }

  // Ensure number[] scratch has capacity.
  if (state.scratch.sortedIndexBuffer.length < populationLength) {
    const nextSize = 1 << Math.ceil(Math.log2(populationLength));
    state.scratch.sortedIndexBuffer = new Array(nextSize);
  }

  // Local alias.
  const indexScratch: IndexBuffer = useTypedScratch
    ? typedScratchBuf!
    : state.scratch.sortedIndexBuffer;

  // Step 3: Initialize identity permutation.
  for (let initIdx = 0; initIdx < populationLength; initIdx++)
    indexScratch[initIdx] = initIdx;
  if (!useTypedScratch)
    state.scratch.sortedIndexBuffer.length = populationLength;

  // Step 4: Iterative quicksort.
  let qsStack = state.scratch.quicksortStack;
  if (qsStack.length < 2)
    qsStack = state.scratch.quicksortStack = new Int32Array(128);
  let stackPtr = 0;

  qsStack[stackPtr++] = 0;
  qsStack[stackPtr++] = populationLength - 1;

  while (stackPtr > 0) {
    const hi = qsStack[--stackPtr];
    const lo = qsStack[--stackPtr];
    if (lo >= hi) continue;

    // Small partition fallback.
    if (hi - lo <= QS_SMALL_THRESHOLD) {
      insertionSortIndices(indexScratch, lo, hi, population);
      continue;
    }

    // Median-of-three pivot.
    let leftPtr = lo;
    let rightPtr = hi;
    const pivotScore = medianOfThreePivot(indexScratch, lo, hi, population);

    // Partition (descending).
    while (leftPtr <= rightPtr) {
      while (true) {
        const li = indexScratch[leftPtr];
        if ((population[li]?.score ?? -Infinity) <= pivotScore) break;
        leftPtr++;
      }
      while (true) {
        const rj = indexScratch[rightPtr];
        if ((population[rj]?.score ?? -Infinity) >= pivotScore) break;
        rightPtr--;
      }
      if (leftPtr <= rightPtr) {
        const t = indexScratch[leftPtr];
        indexScratch[leftPtr] = indexScratch[rightPtr];
        indexScratch[rightPtr] = t;
        leftPtr++;
        rightPtr--;
      }
    }

    // Push larger partition first.
    const leftPartitionSize = rightPtr - lo;
    const rightPartitionSize = hi - leftPtr;

    if (leftPartitionSize > rightPartitionSize) {
      if (lo < rightPtr) {
        stackPtr = qsPushRange(state, stackPtr, lo, rightPtr);
        qsStack = state.scratch.quicksortStack;
      }
      if (leftPtr < hi) {
        stackPtr = qsPushRange(state, stackPtr, leftPtr, hi);
        qsStack = state.scratch.quicksortStack;
      }
    } else {
      if (leftPtr < hi) {
        stackPtr = qsPushRange(state, stackPtr, leftPtr, hi);
        qsStack = state.scratch.quicksortStack;
      }
      if (lo < rightPtr) {
        stackPtr = qsPushRange(state, stackPtr, lo, rightPtr);
        qsStack = state.scratch.quicksortStack;
      }
    }
  }

  // Step 5: Return number[] view.
  if (useTypedScratch) {
    if (state.scratch.sortedIndexBuffer.length < populationLength)
      state.scratch.sortedIndexBuffer = new Array(
        1 << Math.ceil(Math.log2(populationLength)),
      );
    const out = state.scratch.sortedIndexBuffer;
    const ta = typedScratchBuf!;
    for (let k = 0; k < populationLength; k++) out[k] = ta[k];
    out.length = populationLength;
    return out;
  }

  state.scratch.sortedIndexBuffer.length = populationLength;
  return state.scratch.sortedIndexBuffer;
}

/**
 * In-place insertion sort by descending score.
 *
 * @param indexBuf - Index buffer (number[] or Int32Array).
 * @param lo - Inclusive lower bound.
 * @param hi - Inclusive upper bound.
 * @param population - Population array with `.score`.
 *
 * @example
 * insertionSortIndices(idxBuf, 0, n - 1, population);
 */
const insertionSortIndices = (
  // Type assertion: Index buffer can be Int32Array or number[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  indexBuf: any,
  lo: number,
  hi: number,
  // Type assertion: Population array contains dynamic network genomes
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  population: any,
) => {
  if (!indexBuf || lo >= hi) return;

  const popRef = population ?? EMPTY_VEC;
  const scoreOf = (idx: number) =>
    (popRef[idx]?.score ?? Number.NEGATIVE_INFINITY) as number;

  for (let writePos = lo + 1; writePos <= hi; writePos++) {
    const keyIndex = indexBuf[writePos];
    const keyScore = scoreOf(keyIndex);

    let scanPos = writePos - 1;
    while (scanPos >= lo && scoreOf(indexBuf[scanPos]) < keyScore) {
      indexBuf[scanPos + 1] = indexBuf[scanPos];
      scanPos--;
    }

    indexBuf[scanPos + 1] = keyIndex;
  }
}

/**
 * Compute median-of-three pivot score for quicksort.
 *
 * @param indexBuf - Index buffer.
 * @param lo - Low index.
 * @param hi - High index.
 * @param population - Population array.
 * @returns Median score among three candidates.
 *
 * @example
 * const pivot = medianOfThreePivot(idxBuf, 0, n - 1, population);
 */
const medianOfThreePivot = (
  // Type assertion: Index buffer can be Int32Array or number[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  indexBuf: any,
  lo: number,
  hi: number,
  // Type assertion: Population array contains dynamic network genomes
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  population: any,
): number => {
  const mid = (lo + hi) >> 1;
  const leftIndex = indexBuf[lo];
  const middleIndex = indexBuf[mid];
  const rightIndex = indexBuf[hi];

  const popRef = population ?? EMPTY_VEC;

  let leftScore = popRef[leftIndex]?.score ?? Number.NEGATIVE_INFINITY;
  let middleScore = popRef[middleIndex]?.score ?? Number.NEGATIVE_INFINITY;
  let rightScore = popRef[rightIndex]?.score ?? Number.NEGATIVE_INFINITY;

  // Sort three scores to find median.
  if (leftScore > middleScore) {
    const tmp = leftScore;
    leftScore = middleScore;
    middleScore = tmp;
  }

  if (middleScore > rightScore) {
    const tmp = middleScore;
    middleScore = rightScore;
    rightScore = tmp;

    if (leftScore > middleScore) {
      const tmp2 = leftScore;
      leftScore = middleScore;
      middleScore = tmp2;
    }
  }

  return middleScore as number;
}

/**
 * Push a [lo, hi] range onto the pooled quicksort stack.
 *
 * @param state - Shared engine state.
 * @param stackPtr - Current stack pointer.
 * @param rangeLo - Inclusive lower bound.
 * @param rangeHi - Inclusive upper bound.
 * @returns Updated stack pointer.
 *
 * @example
 * ptr = qsPushRange(state, ptr, 0, population.length - 1);
 */
const qsPushRange = (
  state: EngineState,
  stackPtr: number,
  rangeLo: number,
  rangeHi: number,
): number => {
  let stackBuf = state.scratch.quicksortStack;

  const required = stackPtr + 2;
  if (required > stackBuf.length) {
    let newCapacity = Math.max(stackBuf.length << 1, 4);
    while (newCapacity < required) newCapacity <<= 1;

    const grown = new Int32Array(newCapacity);
    grown.set(stackBuf);
    state.scratch.quicksortStack = stackBuf = grown;
  }

  stackBuf[stackPtr++] = rangeLo | 0;
  stackBuf[stackPtr++] = rangeHi | 0;
  return stackPtr;
}

/**
 * Resolve and cache configured mutation operations from NEAT driver.
 *
 * Behaviour:
 * - Reads `neat?.options?.mutation` and caches stable array reference.
 * - Updates cache when driver replaces mutation reference.
 * - Returns read-only array (callers must not mutate).
 *
 * @param neat - NEAT driver object.
 * @returns Read-only array of mutation operation descriptors.
 *
 * @example
 * const ops = getMutationOps(neat);
 */
const getMutationOps = (
  // Type assertion: NEAT driver with dynamic mutation options
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  // Type assertion: Returns array of dynamic mutation operation objects
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any[] => {
  try {
    if (!neat) return EMPTY_VEC;

    const candidate = neat?.options?.mutation;

    if (candidate && cachedMutationOps !== candidate) {
      if (Array.isArray(candidate)) {
        cachedMutationOps = candidate as MutationOperation[];
      } else if (candidate && typeof candidate === 'object') {
        const maybeMutation = candidate as MutationOperation;
        const maybeLen = maybeMutation.length;
        if (maybeLen != null && Number.isFinite(maybeLen) && maybeLen >= 0) {
          cachedMutationOps = candidate as MutationOperation[];
        } else {
          cachedMutationOps = Object.values(candidate as Record<string, MutationOperation>);
        }
      } else {
        cachedMutationOps = EMPTY_VEC;
      }
    }

    return cachedMutationOps ?? EMPTY_VEC;
  } catch {
    return EMPTY_VEC;
  }
}

/**
 * Ensure all output nodes use identity activation.
 *
 * Behaviour:
 * - Iterates population genomes.
 * - Sets `node.squash = methods.Activation.identity` for output nodes.
 * - Best-effort: swallows errors.
 *
 * @param neat - NEAT driver with `population` array.
 *
 * @example
 * ensureOutputIdentity(neat);
 */
export const ensureOutputIdentity = (
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
) => {
  try {
    if (!neat) return;

    // Type assertion: Population array contains network genomes
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const populationRef: any[] = Array.isArray(neat.population)
      ? neat.population
      : EMPTY_VEC;

    for (
      let genomeIndex = 0;
      genomeIndex < populationRef.length;
      genomeIndex++
    ) {
      const genome = populationRef[genomeIndex] as RuntimeGenome;
      if (!genome) continue;

      const nodesRef: NetworkNode[] = Array.isArray(genome.nodes)
        ? genome.nodes
        : EMPTY_VEC;

      for (let nodeIndex = 0; nodeIndex < nodesRef.length; nodeIndex++) {
        const node = nodesRef[nodeIndex];
        if (node && node.type === 'output') {
          node.squash = methods.Activation.identity;
        }
      }
    }
  } catch {
    // Best-effort: swallow errors.
  }
}

/**
 * Update species history and detect species collapse.
 *
 * Behaviour:
 * - Counts unique species IDs in population.
 * - Pushes count into global history buffer.
 * - Inspects recent window for collapse (consecutive single-species).
 * - When collapsed, escalates mutation/novelty parameters.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver with `population`.
 * @param speciesHistory - Global species history array (mutated).
 * @returns `true` when collapse detected; `false` otherwise.
 *
 * @example
 * const collapsed = handleSpeciesHistory(state, neat, historyArray);
 */
export const handleSpeciesHistory = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  speciesHistory: number[],
): boolean => {
  try {
    // Type assertion: Population array contains network genomes
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const populationRef: any[] = Array.isArray(neat?.population)
      ? neat.population
      : EMPTY_VEC;

    // Ensure pooled scratch buffers.
    let speciesIdsBuf = state.scratch.speciesIds;
    let speciesCountsBuf = state.scratch.speciesCounts;
    if (populationRef.length > speciesIdsBuf.length) {
      const nextSize = 1 << Math.ceil(Math.log2(populationRef.length || 1));
      state.scratch.speciesIds = speciesIdsBuf = new Int32Array(nextSize);
      state.scratch.speciesCounts = speciesCountsBuf = new Int32Array(nextSize);
    }

    // Count unique species.
    let uniqueCount = 0;
    for (
      let genomeIndex = 0;
      genomeIndex < populationRef.length;
      genomeIndex++
    ) {
      const genome = populationRef[genomeIndex];
      if (!genome || genome.species == null) continue;

      const speciesId = genome.species | 0;
      let foundIndex = -1;

      for (let scan = 0; scan < uniqueCount; scan++) {
        if (speciesIdsBuf[scan] === speciesId) {
          foundIndex = scan;
          break;
        }
      }

      if (foundIndex === -1) {
        speciesIdsBuf[uniqueCount] = speciesId;
        speciesCountsBuf[uniqueCount] = 1;
        uniqueCount++;
      } else {
        speciesCountsBuf[foundIndex]++;
      }
    }

    // Push species count into history.
    const speciesCount = uniqueCount || 1;
    const updatedHistory = pushHistory<number>(
      speciesHistory,
      speciesCount,
      25,
    );

    // Update the caller's history reference.
    speciesHistory.length = 0;
    speciesHistory.push(...updatedHistory);

    // Check for collapse (last 3 entries all === 1).
    const recentWindow = updatedHistory.slice(-3);
    const isCollapsed =
      recentWindow.length === 3 && recentWindow.every((count) => count === 1);

    if (isCollapsed) {
      // Escalate mutation/novelty parameters.
      try {
        if (neat?.options) {
          const currentMutationRate = neat.options.mutationRate ?? 0;
          const currentMutationAmount = neat.options.mutationAmount ?? 0;
          neat.options.mutationRate = Math.min(1, currentMutationRate * 1.2);
          neat.options.mutationAmount = Math.min(
            1,
            currentMutationAmount * 1.1,
          );

          if (neat.options.config?.novelty) {
            const currentBlend = neat.options.config.novelty.blendFactor ?? 0;
            neat.options.config.novelty.blendFactor = Math.min(
              1,
              currentBlend * 1.15,
            );
          }
        }
      } catch {
        // Best-effort escalation.
      }
    }

    return isCollapsed;
  } catch {
    return false;
  }
}

/**
 * Attempt population expansion when conditions permit.
 *
 * Behaviour:
 * - Checks if dynamic expansion is enabled and interval/plateau conditions met.
 * - Computes target addition count based on current size and factor.
 * - Delegates to `expandPopulation` when there is room to grow.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver.
 * @param dynamicPopEnabled - Flag to enable dynamic expansion.
 * @param completedGenerations - Current generation index.
 * @param dynamicPopMax - Maximum allowed population size.
 * @param plateauGenerations - Plateau window size.
 * @param plateauCounter - Current plateau counter.
 * @param dynamicPopExpandInterval - Generations between expansion attempts.
 * @param dynamicPopExpandFactor - Fraction of current size to add.
 * @param dynamicPopPlateauSlack - Minimum plateau ratio to trigger.
 * @param safeWrite - Logging callback.
 *
 * @example
 * maybeExpandPopulation(state, neat, true, 100, 500, 10, 8, 5, 0.1, 0.75, console.log);
 */
export const maybeExpandPopulation = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  dynamicPopEnabled: boolean,
  completedGenerations: number,
  dynamicPopMax: number,
  plateauGenerations: number,
  plateauCounter: number,
  dynamicPopExpandInterval: number,
  dynamicPopExpandFactor: number,
  dynamicPopPlateauSlack: number,
  safeWrite: (msg: string) => void,
) => {
  try {
    if (!dynamicPopEnabled || completedGenerations <= 0) return;

    // Type assertion: Population array contains network genomes
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const populationRef: any[] = Array.isArray(neat?.population)
      ? neat.population
      : EMPTY_VEC;

    const maxAllowed = Number.isFinite(dynamicPopMax)
      ? Math.max(0, dynamicPopMax | 0)
      : 0;
    if (populationRef.length >= maxAllowed) return;

    const plateauWindow =
      Number.isFinite(plateauGenerations) && plateauGenerations > 0
        ? plateauGenerations | 0
        : 0;
    const plateauRatio =
      plateauWindow > 0 ? Math.min(1, (plateauCounter | 0) / plateauWindow) : 0;

    const expandInterval =
      Number.isFinite(dynamicPopExpandInterval) && dynamicPopExpandInterval > 0
        ? Math.max(1, dynamicPopExpandInterval | 0)
        : 0;

    if (expandInterval === 0) return;

    const isGenerationTrigger =
      (completedGenerations | 0) % expandInterval === 0;
    if (!isGenerationTrigger) return;

    const slackThreshold = Number.isFinite(dynamicPopPlateauSlack)
      ? dynamicPopPlateauSlack
      : 0;
    if (plateauRatio < slackThreshold) return;

    const currentSize = populationRef.length | 0;
    const factor = Number.isFinite(dynamicPopExpandFactor)
      ? Math.max(0, dynamicPopExpandFactor)
      : 0;
    const computedAdd = Math.floor(Math.max(1, currentSize * factor));
    const allowed = Math.max(0, maxAllowed - currentSize);
    const targetAdd = Math.min(computedAdd, allowed);

    if (targetAdd > 0) {
      expandPopulation(
        state,
        neat,
        targetAdd,
        safeWrite,
        completedGenerations | 0,
      );
    }
  } catch {
    // Best-effort: do not throw.
  }
}

/**
 * Prune saturated hidden-to-output connections for a single genome.
 *
 * Behaviour:
 * - Collects outgoing connections from hidden nodes to outputs.
 * - Computes mean and variance of absolute weights.
 * - If collapsed (low mean, near-zero variance), disables smallest half.
 *
 * @param state - Shared engine state for scratch buffers and profiling.
 * @param genome - Mutable genome with `nodes` array.
 * @param getNodeIndicesByType - Helper to collect node indices by type.
 * @param collectHiddenToOutputConns - Helper to collect outgoing connections.
 *
 * @example
 * pruneSaturatedHiddenOutputs(state, genome, getNodeIndicesByTypeFn, collectHiddenToOutputConnsFn);
 */
export const pruneSaturatedHiddenOutputs = (
  state: EngineState,
  genome: unknown,
  getNodeIndicesByType: (nodes: NetworkNode[], nodeType: string) => number,
  collectHiddenToOutputConns: (
    hiddenNode: NetworkNode,
    nodes: NetworkNode[],
    outputCount: number,
  ) => NetworkConnection[],
) => {
  try {
    const pruneProfilingEnabled = isProfilingDetailsEnabled(state);
    const startProfile = pruneProfilingEnabled
      ? profilingStartTimestamp()
      : 0;
    const runtimeGenome = genome as RuntimeGenome;
    const nodesRef = runtimeGenome?.nodes ?? EMPTY_VEC;

    const outputCount = getNodeIndicesByType(nodesRef, 'output');
    const hiddenCount = getNodeIndicesByType(nodesRef, 'hidden');

    let absWeightsTA = state.scratch.exps as Float64Array;
    const indexFlags = state.scratch.nodeIndexBuffer as Int32Array | number[];

    for (let hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++) {
      const hiddenNode =
        nodesRef[
          Number(state.scratch.nodeIndexBuffer[outputCount + hiddenIndex])
        ];
      if (!hiddenNode) continue;

      const outConns =
        collectHiddenToOutputConns(hiddenNode, nodesRef, outputCount) || [];
      const outConnsLen = outConns.length;
      if (outConnsLen < 2) continue;

      // Ensure capacity.
      const needed = outConnsLen;
      if (!absWeightsTA || absWeightsTA.length < needed) {
        let newCap = 1;
        while (newCap < needed) newCap <<= 1;
        absWeightsTA = new Float64Array(newCap);
        state.scratch.exps = absWeightsTA as Float64Array;
      }

      // Fill absolute weights.
      const fillLimit = Math.min(outConnsLen, absWeightsTA.length);
      for (let wi = 0; wi < fillLimit; wi++) {
        const conn = outConns[wi] as NetworkConnection;
        absWeightsTA[wi] = Math.abs(conn?.weight ?? 0) || 0;
      }

      // Welford variance.
      let mean = 0;
      let M2 = 0;
      for (let wi = 0; wi < fillLimit; wi++) {
        const value = absWeightsTA[wi];
        const n = wi + 1;
        const delta = value - mean;
        mean += delta / n;
        M2 += delta * (value - mean);
      }
      const variance = fillLimit ? M2 / fillLimit : 0;

      // Collapse check: disable smallest half.
      if (mean < 0.5 && variance < NUMERIC_EPSILON_SMALL) {
        const disableTarget = Math.max(1, Math.floor(outConnsLen / 2));

        for (let fi = 0; fi < outConnsLen; fi++) indexFlags[fi] = 0;

        for (let di = 0; di < disableTarget; di++) {
          let minPos = -1;
          let minAbs = Infinity;
          for (let j = 0; j < outConnsLen; j++) {
            if (indexFlags[j]) continue;
            const candidate = outConns[j] as NetworkConnection;
            if (!candidate || candidate.enabled === false) {
              indexFlags[j] = 1;
              continue;
            }
            const weightAbs = Math.abs(candidate.weight ?? 0) || 0;
            if (weightAbs < minAbs) {
              minAbs = weightAbs;
              minPos = j;
            }
          }
          if (minPos >= 0) {
            const connToDisable = outConns[minPos] as NetworkConnection;
            connToDisable.enabled = false;
            indexFlags[minPos] = 1;
          } else {
            break;
          }
        }

        for (let fi = 0; fi < outConnsLen; fi++) indexFlags[fi] = 0;
      }
    }

    if (pruneProfilingEnabled) {
      accumulateProfilingDuration(
        state,
        'prune',
        profilingStartTimestamp() - startProfile || 0,
      );
    }
  } catch {
    // Soft-fail: do not throw.
  }
};

/**
 * Reinitialize output biases and weights for anti-collapse recovery.
 *
 * Behaviour:
 * - Selects a fraction (up to 30%) of non-elite genomes.
 * - Samples from non-elite segment using pooled sample buffer.
 * - Reinitializes each sampled genome's outputs via `reinitializeGenomeOutputsAndWeights`.
 * - Emits diagnostic summary.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver with `population` and `options.elitism`.
 * @param completedGenerations - Current generation for logging.
 * @param safeWrite - Logging callback.
 * @param sampleSegmentIntoScratchFn - Sampling helper function.
 *
 * @example
 * antiCollapseRecovery(state, neat, genIndex, console.log, sampleSegmentIntoScratch);
 */
export const antiCollapseRecovery = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
  completedGenerations: number,
  safeWrite: (msg: string) => void,
  sampleSegmentIntoScratchFn: (
    state: EngineState,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    array: any[],
    startIdx: number,
    count: number,
  ) => number,
) => {
  try {
    if (!neat) return;

    const elitismCount = Number.isFinite(neat?.options?.elitism)
      ? Math.max(0, Math.floor(neat.options.elitism))
      : 0;

    const population = Array.isArray(neat.population)
      ? neat.population
      : EMPTY_VEC;

    const nonEliteStartIndex = elitismCount;
    const nonEliteCount = Math.max(0, population.length - nonEliteStartIndex);
    if (nonEliteCount === 0) return;

    const fractionToReinit = 0.3;
    const maxCandidates = Math.floor(nonEliteCount * fractionToReinit) || 1;

    let pooledSampleBuffer = state.scratch.samplePool;
    if (!Array.isArray(pooledSampleBuffer)) {
      pooledSampleBuffer = state.scratch.samplePool = [];
    }

    const sampledCount = sampleSegmentIntoScratchFn(
      state,
      population,
      nonEliteStartIndex,
      maxCandidates,
    );

    if (sampledCount <= 0) return;

    let totalConnectionResets = 0;
    let totalBiasResets = 0;

    for (let sampleIndex = 0; sampleIndex < sampledCount; sampleIndex++) {
      const genome = pooledSampleBuffer[sampleIndex] as RuntimeGenome;
      if (!genome) continue;

      try {
        const { connReset, biasReset } = reinitializeGenomeOutputsAndWeights(
          state,
          genome,
        ) || {
          connReset: 0,
          biasReset: 0,
        };
        totalConnectionResets += Number(connReset) || 0;
        totalBiasResets += Number(biasReset) || 0;
      } catch {
        // Swallow per-genome errors.
      }
    }

    try {
      safeWrite(
        `[ANTICOLLAPSE] gen=${completedGenerations} reinitGenomes=${sampledCount} connReset=${totalConnectionResets} biasReset=${totalBiasResets}\n`,
      );
    } catch {
      // Best-effort logging.
    }
  } catch {
    // Global swallow.
  }
};

/**
 * Reinitialize output node biases and outgoing weights for a single genome.
 *
 * Behaviour:
 * - Collects output nodes into pooled sample buffer.
 * - Randomizes each output's `bias` within ±BIAS_RESET_HALF_RANGE.
 * - Resets connection `weight` for connections targeting outputs.
 *
 * @param state - Shared engine state.
 * @param genome - Mutable genome with `nodes` and `connections`.
 * @returns Object with `{ connReset: number, biasReset: number }`.
 *
 * @example
 * const deltas = reinitializeGenomeOutputsAndWeights(state, genome);
 */
export const reinitializeGenomeOutputsAndWeights = (
  state: EngineState,
  genome: unknown,
): { connReset: number; biasReset: number } => {
  try {
    const runtimeGenome = genome as RuntimeGenome;
    const nodesList: NetworkNode[] = Array.isArray(runtimeGenome?.nodes) ? runtimeGenome.nodes : [];

    let sampleBuf = state.scratch.samplePool;
    if (!Array.isArray(sampleBuf)) sampleBuf = state.scratch.samplePool = [];

    const requiredCapacity = nodesList.length;
    if (sampleBuf.length < requiredCapacity) {
      let newCapacity = Math.max(1, sampleBuf.length);
      while (newCapacity < requiredCapacity) newCapacity <<= 1;
      sampleBuf.length = newCapacity;
    }

    // Collect outputs.
    let outputCount = 0;
    for (const node of nodesList) {
      if (node && node.type === 'output') {
        sampleBuf[outputCount++] = node;
      }
    }

    // Reinitialize biases.
    let biasReset = 0;
    const biasHalfRange = BIAS_RESET_HALF_RANGE;
    const randomParameters = resolveRngParameters();
    for (let idx = 0; idx < outputCount; idx++) {
      // Type assertion: pool holds Node objects at runtime
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const outNode = sampleBuf[idx] as any;
      if (!outNode) continue;
      outNode.bias =
        drawFastRandom(state, randomParameters) * (2 * biasHalfRange) -
        biasHalfRange;
      biasReset++;
    }

    // Reset weights targeting outputs.
    let connReset = 0;
    const connections: NetworkConnection[] = Array.isArray(runtimeGenome?.connections)
      ? runtimeGenome.connections
      : [];
    if (connections.length > 0 && outputCount > 0) {
      const outputsSet = new Set<NetworkNode>();
      for (let idx = 0; idx < outputCount; idx++) {
        const outNode = sampleBuf[idx] as NetworkNode;
        if (outNode) outputsSet.add(outNode);
      }

      const weightHalfRange = CONN_WEIGHT_RESET_HALF_RANGE;

      for (const conn of connections) {
        try {
          const connTarget = conn?.to;
          if (connTarget && outputsSet.has(connTarget)) {
            conn.weight =
              drawFastRandom(state, randomParameters) * (2 * weightHalfRange) -
              weightHalfRange;
            connReset++;
          }
        } catch {
          // Swallow per-connection errors.
        }
      }
    }

    return { connReset, biasReset };
  } catch {
    return { connReset: 0, biasReset: 0 };
  }
};

/**
 * Compact a single genome's connection list by removing disabled connections.
 *
 * Behaviour:
 * - Performs in-place stable compaction of `genome.connections`.
 * - Preserves relative order of enabled connections.
 * - Two-pointer write/read technique.
 *
 * @param genome - Mutable genome with `connections` array.
 * @returns Number of removed (disabled) connections.
 *
 * @example
 * const removed = compactGenomeConnections(genome);
 */
export const compactGenomeConnections = (
  // Type assertion: Genome with dynamic structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  genome: any,
): number => {
  try {
    // Type assertion: Connections array with dynamic structure
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const connectionsList: any[] = Array.isArray(genome?.connections)
      ? genome.connections
      : [];

    const totalConnections = connectionsList.length;
    if (totalConnections === 0) return 0;

    let writeIndex = 0;
    for (let readIndex = 0; readIndex < totalConnections; readIndex++) {
      const connection = connectionsList[readIndex];
      if (connection && connection.enabled !== false) {
        if (readIndex !== writeIndex) connectionsList[writeIndex] = connection;
        writeIndex++;
      }
    }

    const removedCount = totalConnections - writeIndex;
    if (removedCount > 0) connectionsList.length = writeIndex;
    return removedCount;
  } catch {
    return 0;
  }
};

/**
 * Compact entire population by removing disabled connections from each genome.
 *
 * Behaviour:
 * - Uses pooled sample buffer as scratch counts array.
 * - Compacts each genome via `compactGenomeConnections`.
 * - Returns total removed connections.
 *
 * @param state - Shared engine state.
 * @param neat - NEAT driver with `population`.
 * @returns Total removed disabled connections.
 *
 * @example
 * const totalRemoved = compactPopulation(state, neat);
 */
export const compactPopulation = (
  state: EngineState,
  // Type assertion: NEAT driver with dynamically typed population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  neat: any,
): number => {
  try {
    // Type assertion: Population array contains network genomes
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const populationList: any[] = Array.isArray(neat?.population)
      ? neat.population
      : [];

    const populationSize = populationList.length;
    if (populationSize === 0) return 0;

    let scratchCounts = state.scratch.samplePool;
    if (!Array.isArray(scratchCounts))
      scratchCounts = state.scratch.samplePool = [];

    if (scratchCounts.length < populationSize) {
      let newCapacity = Math.max(1, scratchCounts.length);
      while (newCapacity < populationSize) newCapacity <<= 1;
      scratchCounts.length = newCapacity;
    }

    let totalRemoved = 0;
    for (let idx = 0; idx < populationSize; idx++) {
      try {
        const genome = populationList[idx];
        const removedForGenome = compactGenomeConnections(genome) | 0;
        scratchCounts[idx] = removedForGenome;
        totalRemoved += removedForGenome;
      } catch {
        scratchCounts[idx] = 0;
      }
    }

    return totalRemoved;
  } catch {
    return 0;
  }
}
