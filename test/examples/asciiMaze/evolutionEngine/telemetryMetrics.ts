/**
 * Telemetry logging and metric computation helpers extracted from the ASCII maze evolution engine.
 *
 * The helpers in this module operate on the shared {@link EngineState} scratch buffers to avoid
 * per-call allocations while keeping the main façade lighter. All telemetry is best-effort: any
 * internal error is swallowed so that logging never impacts the evolution loop.
 */
import type { Neat, Network } from '../../../../src/neataptic';
import type { GenomeDetailed } from '../../../../src/neat/neat.types';
import type { EngineState } from './engineState.types';
import {
  ensureVisitedHashCapacity,
  initialiseTelemetryScratch,
} from './engineState';
import { getTail, sampleIntoScratch } from './sampling';

/**
 * Runtime type for NEAT instance with dynamic properties.
 * NEAT instances may have population, getTelemetry and other runtime methods.
 */
interface RuntimeNeat {
  population?: unknown[];
  getTelemetry?: () => unknown;
  [key: string]: unknown;
}

/**
 * Runtime type for network connection with dynamic properties.
 * Connections may have enabled, weight, and other runtime-added fields.
 */
interface RuntimeConnection {
  enabled?: boolean;
  weight?: number;
  [key: string]: unknown;
}

/**
 * Runtime type for NEAT genome/network with dynamic properties.
 * Genomes may have nodes, connections, score, species and runtime tracking fields.
 */
interface RuntimeGenome {
  nodes?: RuntimeNode[];
  connections?: RuntimeConnection[];
  score?: number;
  species?: number | null;
  _lastStepOutputs?: unknown[];
  [key: string]: unknown;
}

/**
 * Runtime type for network node with dynamic properties.
 * Nodes may have type, bias, and other runtime-added fields.
 */
interface RuntimeNode {
  type?: string;
  bias?: number;
  [key: string]: unknown;
}

/**
 * Telemetry tag emitted when logging action-entropy statistics.
 * @example
 * safeWrite(`${LOG_TAG_ACTION_ENTROPY} gen=1 entropyNorm=0.500 uniqueMoves=4 pathLen=32\n`);
 */
export const LOG_TAG_ACTION_ENTROPY = '[ACTION_ENTROPY]';
/**
 * Telemetry tag emitted when logging bias statistics for output nodes.
 * @example
 * safeWrite(`${LOG_TAG_OUTPUT_BIAS} gen=2 mean=0.001 std=0.010 biases=0.01,-0.02,0.03,-0.01\n`);
 */
export const LOG_TAG_OUTPUT_BIAS = '[OUTPUT_BIAS]';
/**
 * Telemetry tag emitted when logging logits statistics and collapse diagnostics.
 * @example
 * safeWrite(`${LOG_TAG_LOGITS} gen=3 means=0.001,0.002,-0.001,-0.002 stds=0.01,0.02,0.03,0.04 kurt=0,0,0,0 entMean=0.500 stability=0.750 steps=32\n`);
 */
export const LOG_TAG_LOGITS = '[LOGITS]';

/** Telemetry tag emitted when logging exploration metrics (kept internal to avoid cluttered exports). */
const LOG_TAG_EXPLORATION = '[EXPLORE]';
/** Telemetry tag emitted when logging population diversity metrics. */
const LOG_TAG_DIVERSITY = '[DIVERSITY]';

/** Cached inverse of ln(4) used to normalise entropy into the [0,1] range. */
const INV_LOG4 = 1 / Math.log(4);
/** Standard deviation threshold used when detecting collapsed logits. */
const LOG_STD_FLAT_THRESHOLD = 0.005;
/** Entropy threshold used when detecting collapsed logits. */
const ENTROPY_COLLAPSE_THRESHOLD = 0.35;
/** Decision stability threshold used when detecting collapsed logits. */
const STABILITY_COLLAPSE_THRESHOLD = 0.97;
/** Consecutive collapsed generations required before triggering anti-collapse recovery. */
const COLLAPSE_STREAK_TRIGGER = 6;
/** Default tail length sampled from logits history when none is provided. */
const DEFAULT_RECENT_WINDOW = 40;
/** Default cap for diversity sampling size. */
const DEFAULT_SAMPLE_SIZE = 40;
/** Default action dimension for the ASCII maze agent (N, E, S, W). */
const DEFAULT_ACTION_DIMENSION = 4;

/** Branchless lookup map translating (dx, dy) deltas into directional indices. */
const DIR_DELTA_TO_INDEX: Int8Array = (() => {
  const directionMap = new Int8Array(9);
  directionMap.fill(-1);
  directionMap[(0 + 1) * 3 + (-1 + 1)] = 0; // North
  directionMap[(1 + 1) * 3 + (0 + 1)] = 1; // East
  directionMap[(0 + 1) * 3 + (1 + 1)] = 2; // South
  directionMap[(-1 + 1) * 3 + (0 + 1)] = 3; // West
  return directionMap;
})();

/** Internal collapse streak counter maintained across telemetry invocations. */
let collapseStreak = 0;

/** Writer signature reused across telemetry helpers. */
type TelemetryWriter = (message: string) => void;

/**
 * Minimal structure representing a generation evolution result.
 * Expected to have a path property containing the movement history.
 */
interface GenerationResult {
  /** Path taken by the agent (array of [x, y] coordinate pairs). */
  path?: ReadonlyArray<[number, number]>;
  /** Additional properties may exist but are not strongly typed. */
  [key: string]: unknown;
}

/**
 * Parameters shared by telemetry helpers that require access to the shared state and writer.
 */
interface TelemetryBaseParams {
  /** Shared engine state providing pooled scratch buffers. */
  state: EngineState;
  /** Completed generation index used in formatted telemetry lines. */
  generationIndex: number;
  /** Logger invoked with a single newline-terminated string. */
  safeWrite: TelemetryWriter;
}

/**
 * Parameters required to emit action-entropy telemetry.
 */
export interface LogActionEntropyParams extends TelemetryBaseParams {
  /** Per-generation result expected to expose a `path` array (may be undefined early on). */
  generationResult: GenerationResult | undefined;
}

/**
 * Emit a best-effort telemetry line containing action-entropy statistics.
 *
 * @param params Shared state, generation metadata and logging callback.
 * @returns void
 */
export const logActionEntropy = ({
  state,
  generationResult,
  generationIndex,
  safeWrite,
}: LogActionEntropyParams): void => {
  if (typeof safeWrite !== 'function') return;

  const pathReference = generationResult?.path;

  try {
    // Step 1: Compute entropy statistics (pure helper that reuses pooled scratch buffers).
    const entropyStats = computeActionEntropy(state, pathReference);

    // Step 2: Format metrics using defensive coercions.
    const entropyNormStr = Number.isFinite(entropyStats.entropyNorm)
      ? entropyStats.entropyNorm.toFixed(3)
      : '0.000';
    const uniqueMovesStr = Number.isFinite(entropyStats.uniqueMoves)
      ? String(entropyStats.uniqueMoves)
      : '0';
    const pathLengthStr = Number.isFinite(entropyStats.pathLen)
      ? String(entropyStats.pathLen)
      : '0';

    // Step 3: Emit the single-line telemetry record.
    safeWrite(
      `${LOG_TAG_ACTION_ENTROPY} gen=${generationIndex} entropyNorm=${entropyNormStr} uniqueMoves=${uniqueMovesStr} pathLen=${pathLengthStr}\n`,
    );
  } catch {
    // Telemetry remains best-effort: swallow any unexpected failure.
  }
};

/**
 * Parameters required to emit output-bias telemetry.
 */
export interface LogOutputBiasParams extends TelemetryBaseParams {
  /** Fittest genome/network for the generation (may be undefined early on). */
  fittest: unknown;
}

/**
 * Emit bias statistics for the fittest network's output nodes.
 *
 * @param params Shared state, subject network and writer callback.
 * @returns void
 */
export const logOutputBiasStats = ({
  state,
  fittest,
  generationIndex,
  safeWrite,
}: LogOutputBiasParams): void => {
  if (typeof safeWrite !== 'function') return;

  const runtimeFittest = fittest as RuntimeGenome | undefined;
  const nodeList = runtimeFittest?.nodes ?? [];

  try {
    // Step 1: Populate pooled node-index scratch with output-node indices.
    const outputCount = collectNodeIndicesByType(state, nodeList, 'output');
    if (outputCount <= 0) return;

    // Step 2: Compute bias statistics using pooled typed arrays.
    const biasStats = computeOutputBiasStats(state, nodeList, outputCount);

    // Step 3: Format telemetry line with stable precision.
    const meanStr = Number.isFinite(biasStats.mean)
      ? biasStats.mean.toFixed(3)
      : '0.000';
    const stdStr = Number.isFinite(biasStats.std)
      ? biasStats.std.toFixed(3)
      : '0.000';
    const biasesStr = String(biasStats.biasesStr ?? '');

    safeWrite(
      `${LOG_TAG_OUTPUT_BIAS} gen=${generationIndex} mean=${meanStr} std=${stdStr} biases=${biasesStr}\n`,
    );
  } catch {
    // Bias telemetry is best-effort; swallow unexpected failures.
  }
};

/**
 * Parameters required to emit logits statistics, perform collapse detection and trigger recovery.
 */
export interface LogLogitsParams extends TelemetryBaseParams {
  /** NEAT instance passed to the anti-collapse recovery helper when triggered. */
  neat: unknown;
  /** Fittest genome/network which may expose a `_lastStepOutputs` logits history. */
  fittest: unknown;
  /** Optional override for action output dimensionality (defaults to 4). */
  actionDimension?: number;
  /** Optional override for the recent history window length (defaults to {@link DEFAULT_RECENT_WINDOW}). */
  recentWindow?: number;
  /** Optional override for reduced telemetry mode; defaults to `state.toggles.reducedTelemetry`. */
  reducedTelemetry?: boolean;
  /** Callback invoked when the collapse streak reaches the trigger threshold. */
  onCollapseRecovery: (
    neat: unknown,
    generationIndex: number,
    safeWrite: TelemetryWriter,
  ) => void;
}

/**
 * Emit logits-level telemetry, detect collapse streaks and trigger anti-collapse recovery when needed.
 *
 * @param params Shared state, subject genomes and telemetry configuration.
 * @returns void
 */
export const logLogitsAndCollapse = ({
  state,
  neat,
  fittest,
  generationIndex,
  safeWrite,
  actionDimension = DEFAULT_ACTION_DIMENSION,
  recentWindow = DEFAULT_RECENT_WINDOW,
  reducedTelemetry = state.toggles.reducedTelemetry,
  onCollapseRecovery,
}: LogLogitsParams): void => {
  if (typeof safeWrite !== 'function') return;

  try {
    // Step 1: Obtain the recent logits history from the fittest candidate.
    const runtimeFittest = fittest as RuntimeGenome | undefined;
    const logitsHistory: number[][] =
      (runtimeFittest?._lastStepOutputs as number[][]) ?? EMPTY_VECTOR;
    if (logitsHistory.length === 0) return;

    const recentTail = getTail<number[]>(state, logitsHistory, recentWindow);

    // Step 2: Compute aggregate statistics using pooled scratch buffers.
    const stats = computeLogitStats({
      state,
      recent: recentTail,
      actionDimension,
      reducedTelemetry,
    });

    // Step 3: Emit formatted telemetry line with defensive formatting.
    const entropyMeanStr = Number.isFinite(stats.entMean)
      ? stats.entMean.toFixed(3)
      : '0.000';
    const stabilityStr = Number.isFinite(stats.stability)
      ? stats.stability.toFixed(3)
      : '0.000';

    safeWrite(
      `${LOG_TAG_LOGITS} gen=${generationIndex} means=${stats.meansStr} stds=${stats.stdsStr} kurt=${stats.kurtStr} entMean=${entropyMeanStr} stability=${stabilityStr} steps=${recentTail.length}\n`,
    );

    // Step 4: Collapse detection using std, entropy and stability thresholds.
    const stdArray = stats.stds ?? EMPTY_VECTOR;
    let allStdBelowThreshold = true;
    for (let stdIndex = 0; stdIndex < stdArray.length; stdIndex++) {
      const stdValue = stdArray[stdIndex];
      if (!(stdValue < LOG_STD_FLAT_THRESHOLD)) {
        allStdBelowThreshold = false;
        break;
      }
    }

    const entropyCollapsed =
      stats.entMean < ENTROPY_COLLAPSE_THRESHOLD ||
      stats.stability > STABILITY_COLLAPSE_THRESHOLD;
    const isCollapsed = allStdBelowThreshold && entropyCollapsed;

    if (isCollapsed) collapseStreak += 1;
    else collapseStreak = 0;

    // Step 5: Trigger anti-collapse recovery when the streak threshold is reached.
    if (collapseStreak === COLLAPSE_STREAK_TRIGGER) {
      collapseStreak = 0; // reset before invoking recovery to avoid repeated triggers
      onCollapseRecovery?.(neat, generationIndex, safeWrite);
    }
  } catch {
    // Telemetry must remain best-effort; swallow any unexpected failure.
  }
};

/**
 * Parameters required to emit exploration telemetry.
 */
export interface LogExplorationParams extends TelemetryBaseParams {
  /** Per-generation result exposing `path`, `progress` and optional `saturationFraction`. */
  generationResult: GenerationResult | undefined;
}

/**
 * Emit exploration telemetry summarising unique coverage, path length and ratios.
 *
 * @param params Shared state, generation result and logger callback.
 * @returns void
 */
export const logExploration = ({
  state,
  generationResult,
  generationIndex,
  safeWrite,
}: LogExplorationParams): void => {
  if (typeof safeWrite !== 'function') return;

  const pathReference = generationResult?.path;
  const rawProgress = generationResult?.progress;
  const rawSaturationFraction = generationResult?.saturationFraction;

  try {
    // Step 1: Compute exploration metrics using pooled scratch helpers.
    const exploration = computeExplorationStats(state, pathReference);

    // Step 2: Format values with stable numeric precision.
    const uniqueStr = Number.isFinite(exploration.unique)
      ? String(exploration.unique)
      : '0';
    const pathLengthStr = Number.isFinite(exploration.pathLen)
      ? String(exploration.pathLen)
      : '0';
    const ratioStr = Number.isFinite(exploration.ratio)
      ? exploration.ratio.toFixed(3)
      : '0.000';
    const progressStr = Number.isFinite(rawProgress)
      ? (rawProgress as number).toFixed(1)
      : '0.0';
    const saturationStr = Number.isFinite(rawSaturationFraction)
      ? (rawSaturationFraction as number).toFixed(3)
      : '0.000';

    safeWrite(
      `${LOG_TAG_EXPLORATION} gen=${generationIndex} unique=${uniqueStr} pathLen=${pathLengthStr} ratio=${ratioStr} progress=${progressStr} satFrac=${saturationStr}\n`,
    );
  } catch {
    // Exploration telemetry is best-effort; swallow unexpected failures.
  }
};

/**
 * Parameters required to emit population diversity telemetry.
 */
export interface LogDiversityParams extends TelemetryBaseParams {
  /** NEAT instance exposing a `population` array. */
  neat: unknown;
  /** Optional override for the diversity sampling size (defaults to {@link DEFAULT_SAMPLE_SIZE}). */
  sampleSize?: number;
}

/**
 * Emit diversity telemetry including species count, Simpson index and weight standard deviation.
 *
 * @param params Shared state, NEAT population reference and logger callback.
 * @returns void
 */
export const logDiversity = ({
  state,
  neat,
  generationIndex,
  safeWrite,
  sampleSize = DEFAULT_SAMPLE_SIZE,
}: LogDiversityParams): void => {
  if (typeof safeWrite !== 'function') return;
  const runtimeNeat = neat as RuntimeNeat | undefined;
  if (!runtimeNeat || !Array.isArray(runtimeNeat.population)) return;

  try {
    const diversity = computeDiversityMetrics(state, runtimeNeat, sampleSize);

    const speciesCountStr = Number.isFinite(diversity.speciesUniqueCount)
      ? String(diversity.speciesUniqueCount)
      : '0';
    const simpsonStr = Number.isFinite(diversity.simpson)
      ? diversity.simpson.toFixed(3)
      : '0.000';
    const weightStdStr = Number.isFinite(diversity.weightStd)
      ? diversity.weightStd.toFixed(3)
      : '0.000';

    safeWrite(
      `${LOG_TAG_DIVERSITY} gen=${generationIndex} species=${speciesCountStr} simpson=${simpsonStr} weightStd=${weightStdStr}\n`,
    );
  } catch {
    // Diversity telemetry is best-effort; swallow unexpected failures.
  }
};

/**
 * Collect a short telemetry tail from a NEAT instance when available.
 *
 * @param state Shared engine state (provides pooled scratch buffers for `getTail`).
 * @param neat NEAT instance that may expose a `getTelemetry` function.
 * @param tailLength Desired tail length (floored to an integer >= 0). Defaults to 10.
 * @returns Tail array, raw telemetry value or `undefined` on missing API/errors.
 */
export const collectTelemetryTail = (
  state: EngineState,
  neat: unknown,
  tailLength = 10,
): unknown => {
  // Step 1: Guard against missing telemetry providers so callers can skip optional handling.
  const runtimeNeat = neat as RuntimeNeat | undefined;
  if (!runtimeNeat || typeof runtimeNeat.getTelemetry !== 'function')
    return undefined;

  // Step 2: Normalise the desired tail length to a bounded non-negative integer.
  const normalizedTailLength = Number.isFinite(tailLength)
    ? Math.max(0, Math.floor(tailLength))
    : 10;

  try {
    // Step 3: Probe the telemetry API and re-use pooled storage for array tails when possible.
    const telemetryRaw = runtimeNeat.getTelemetry?.();
    if (Array.isArray(telemetryRaw)) {
      return getTail(state, telemetryRaw, normalizedTailLength);
    }
    // Step 4: Fall back to forwarding opaque telemetry values untouched for downstream consumers.
    return telemetryRaw;
  } catch {
    return undefined;
  }
};

/** Empty shared vector reused when a fallback empty array is required. */
const EMPTY_VECTOR: unknown[] = [];

/** Structure describing the result of action-entropy computation. */
interface ActionEntropyStats {
  entropyNorm: number;
  uniqueMoves: number;
  pathLen: number;
}

/** Compute action-entropy metrics using pooled scratch buffers. */
const computeActionEntropy = (
  state: EngineState,
  path: ReadonlyArray<[number, number]> | undefined,
): ActionEntropyStats => {
  if (!Array.isArray(path) || path.length < 2) {
    return { entropyNorm: 0, uniqueMoves: 0, pathLen: path?.length ?? 0 };
  }

  let counts = state.scratch.moveCounts;
  if (!(counts instanceof Int32Array)) {
    counts = new Int32Array(DEFAULT_ACTION_DIMENSION);
    state.scratch.moveCounts = counts;
  }
  const activeLength = counts.length;
  counts.fill(0, 0, activeLength);

  let totalMoves = 0;
  for (let stepIndex = 1; stepIndex < path.length; stepIndex++) {
    const current = path[stepIndex];
    const previous = path[stepIndex - 1];
    if (!current || !previous) continue;

    const deltaX =
      (Number.isFinite(current[0]) ? current[0] : 0) -
      (Number.isFinite(previous[0]) ? previous[0] : 0);
    const deltaY =
      (Number.isFinite(current[1]) ? current[1] : 0) -
      (Number.isFinite(previous[1]) ? previous[1] : 0);

    if (deltaX < -1 || deltaX > 1 || deltaY < -1 || deltaY > 1) continue;

    const mapKey = (deltaX + 1) * 3 + (deltaY + 1);
    const mappedIndex = DIR_DELTA_TO_INDEX[mapKey];
    if (
      Number.isFinite(mappedIndex) &&
      mappedIndex >= 0 &&
      mappedIndex < activeLength
    ) {
      counts[mappedIndex] = (counts[mappedIndex] | 0) + 1;
      totalMoves += 1;
    }
  }

  if (totalMoves === 0) {
    return { entropyNorm: 0, uniqueMoves: 0, pathLen: path.length };
  }

  const invTotal = 1 / totalMoves;
  let entropy = 0;
  let uniqueMoves = 0;
  const dimLimit = Math.min(4, activeLength);
  for (let directionIndex = 0; directionIndex < dimLimit; directionIndex++) {
    const count = counts[directionIndex];
    if (count > 0) {
      const probability = count * invTotal;
      entropy += -probability * Math.log(probability);
      uniqueMoves += 1;
    }
  }

  return {
    entropyNorm: entropy * INV_LOG4,
    uniqueMoves,
    pathLen: path.length,
  };
};

/** Compute exploration statistics using either the tiny table or dynamic visited hash. */
const computeExplorationStats = (
  state: EngineState,
  path: ReadonlyArray<[number, number]> | undefined,
): { unique: number; pathLen: number; ratio: number } => {
  // Step 1: Handle empty paths up-front to avoid scratch allocations or divisions by zero.
  const pathLength = path?.length ?? 0;
  if (pathLength === 0) {
    return { unique: 0, pathLen: 0, ratio: 0 };
  }

  // Step 2: Choose the most efficient distinct-coordinate counter based on path length.
  let unique = 0;
  if (pathLength < 32) {
    unique = countDistinctCoordinatesTiny(state, path!, pathLength);
  } else {
    unique = countDistinctCoordinatesHashed(state, path!, pathLength);
  }

  // Step 3: Aggregate the summary metrics while keeping floating-point ratios guarded.
  return {
    unique,
    pathLen: pathLength,
    ratio: unique / pathLength,
  };
};

/** Count distinct coordinates for tiny paths using the pooled 64-slot table. */
const countDistinctCoordinatesTiny = (
  state: EngineState,
  path: ReadonlyArray<[number, number]>,
  pathLength: number,
): number => {
  // Step 1: Fast-path zero-length paths to avoid scratch-table churn.
  if (pathLength === 0) return 0;

  // Step 2: Ensure the tiny-table scratch buffer exists and is cleared before use.
  let tinyTable = state.scratch.smallExploreTable;
  if (!(tinyTable instanceof Int32Array) || tinyTable.length === 0) {
    tinyTable = new Int32Array(64);
    state.scratch.smallExploreTable = tinyTable;
  }
  const mask = tinyTable.length - 1;
  tinyTable.fill(0);

  // Step 3: Insert each packed coordinate using linear probing and count unique writes only once.
  let uniqueCount = 0;
  for (
    let coordinateIndex = 0;
    coordinateIndex < pathLength;
    coordinateIndex++
  ) {
    const coordinate = path[coordinateIndex];
    const packed = ((coordinate[0] & 0xffff) << 16) | (coordinate[1] & 0xffff);
    let hash = Math.imul(packed, HASH_KNUTH_32) >>> 0;
    const storedValue = (packed + 1) | 0;

    while (true) {
      const slot = hash & mask;
      const slotValue = tinyTable[slot];
      if (slotValue === 0) {
        tinyTable[slot] = storedValue;
        uniqueCount += 1;
        break;
      }
      if (slotValue === storedValue) break;
      hash = (hash + 1) | 0;
    }
  }

  // Step 4: Return the deduplicated coordinate count to the caller.
  return uniqueCount;
};

/** Count distinct coordinates for larger paths using the shared open-address hash table. */
const countDistinctCoordinatesHashed = (
  state: EngineState,
  path: ReadonlyArray<[number, number]>,
  pathLength: number,
): number => {
  // Step 1: Make sure the visited-hash scratch table is large enough for open addressing.
  const targetCapacity = pathLength << 1;
  const { table, slotMask } = ensureVisitedHashCapacity(targetCapacity, state);
  const mask = slotMask;
  let distinctCount = 0;

  // Step 2: Probe the shared hash table with packed coordinates and count first-time insertions.
  for (
    let coordinateIndex = 0;
    coordinateIndex < pathLength;
    coordinateIndex++
  ) {
    const coordinate = path[coordinateIndex];
    const packed = ((coordinate[0] & 0xffff) << 16) | (coordinate[1] & 0xffff);
    let hash = Math.imul(packed, HASH_KNUTH_32) >>> 0;
    const storedValue = (packed + 1) | 0;

    while (true) {
      const slot = hash & mask;
      const slotValue = table[slot];
      if (slotValue === 0) {
        table[slot] = storedValue;
        distinctCount += 1;
        break;
      }
      if (slotValue === storedValue) break;
      hash = (hash + 1) | 0;
    }
  }

  // Step 3: Expose the distinct coordinate count for telemetry consumers.
  return distinctCount;
};

/**
 * Compute diversity metrics (species count, Simpson index, weight std) for the population.
 */
const computeDiversityMetrics = (
  state: EngineState,
  neat: RuntimeNeat,
  sampleSize: number,
): { speciesUniqueCount: number; simpson: number; weightStd: number } => {
  // Step 1: Normalise the population reference to a safe array view.
  const population: unknown[] = Array.isArray(neat?.population)
    ? neat.population
    : EMPTY_VECTOR;
  const populationLength = population.length | 0;
  if (populationLength === 0) {
    return { speciesUniqueCount: 0, simpson: 0, weightStd: 0 };
  }

  // Step 2: Ensure cached scratch arrays can hold the current population size.
  let speciesIds = state.scratch.speciesIds;
  let speciesCounts = state.scratch.speciesCounts;
  if (populationLength > speciesIds.length) {
    const newSize = 1 << Math.ceil(Math.log2(populationLength));
    speciesIds = new Int32Array(newSize);
    speciesCounts = new Int32Array(newSize);
    state.scratch.speciesIds = speciesIds;
    state.scratch.speciesCounts = speciesCounts;
  } else {
    speciesCounts.fill(0);
  }

  // Step 3: Tally individuals per species while tracking unique identifiers.
  let speciesUniqueCount = 0;
  let individualCount = 0;
  for (let genomeIndex = 0; genomeIndex < populationLength; genomeIndex++) {
    const genome = population[genomeIndex] as RuntimeGenome | undefined;
    const speciesId =
      (genome && genome.species != null ? genome.species : -1) | 0;

    let foundIndex = -1;
    for (let lookupIndex = 0; lookupIndex < speciesUniqueCount; lookupIndex++) {
      if (speciesIds[lookupIndex] === speciesId) {
        foundIndex = lookupIndex;
        break;
      }
    }

    if (foundIndex === -1) {
      speciesIds[speciesUniqueCount] = speciesId;
      speciesCounts[speciesUniqueCount] = 1;
      speciesUniqueCount += 1;
    } else {
      speciesCounts[foundIndex] += 1;
    }

    individualCount += 1;
  }

  if (individualCount === 0) individualCount = 1;

  // Step 4: Compute the Simpson diversity index from the collected population counts.
  let simpsonAccumulator = 0;
  for (
    let speciesIndex = 0;
    speciesIndex < speciesUniqueCount;
    speciesIndex++
  ) {
    const proportion = speciesCounts[speciesIndex] / individualCount;
    simpsonAccumulator += proportion * proportion;
  }
  const simpson = 1 - simpsonAccumulator;

  // Step 5: Sample a subset of genomes and accumulate enabled connection weights for std-dev.
  const boundedSampleSize =
    sampleSize > 0 ? Math.min(populationLength, sampleSize | 0) : 0;
  const sampledLength = boundedSampleSize
    ? sampleIntoScratch(state, population, boundedSampleSize)
    : 0;

  let weightMean = 0;
  let weightM2 = 0;
  let enabledWeights = 0;
  const sampleBuffer = state.scratch.samplePool ?? EMPTY_VECTOR;
  for (let sampleIndex = 0; sampleIndex < sampledLength; sampleIndex++) {
    const genome = sampleBuffer[sampleIndex] as GenomeDetailed | undefined;
    const connections = Array.isArray(genome?.connections)
      ? (genome.connections as RuntimeConnection[])
      : EMPTY_VECTOR;
    for (
      let connectionIndex = 0;
      connectionIndex < connections.length;
      connectionIndex++
    ) {
      const connection = connections[connectionIndex] as
        | RuntimeConnection
        | undefined;
      if (connection && connection.enabled !== false) {
        const weight = Number.isFinite(connection.weight)
          ? connection.weight!
          : 0;
        enabledWeights += 1;
        const delta = weight - weightMean;
        weightMean += delta / enabledWeights;
        weightM2 += delta * (weight - weightMean);
      }
    }
  }

  const weightStd = enabledWeights ? Math.sqrt(weightM2 / enabledWeights) : 0;

  // Step 6: Return the composed diversity metrics with the computed aggregates.
  return {
    speciesUniqueCount,
    simpson,
    weightStd,
  };
};

/** Gather indices of nodes matching `nodeType` into the pooled scratch buffer. */
const collectNodeIndicesByType = (
  state: EngineState,
  nodes: RuntimeNode[] | undefined,
  nodeType: string,
): number => {
  // Step 1: Exit early when the node list is absent or already empty.
  if (!Array.isArray(nodes) || nodes.length === 0) return 0;

  // Step 2: Prepare or resize the pooled node-index buffer before writing into it.
  let indexBuffer = state.scratch.nodeIndexBuffer;
  if (!(indexBuffer instanceof Int32Array) || indexBuffer.length === 0) {
    indexBuffer = new Int32Array(Math.max(64, nodes.length));
    state.scratch.nodeIndexBuffer = indexBuffer;
  } else if (nodes.length > indexBuffer.length) {
    const nextCapacity = 1 << Math.ceil(Math.log2(nodes.length));
    indexBuffer = new Int32Array(nextCapacity);
    state.scratch.nodeIndexBuffer = indexBuffer;
  }

  // Step 3: Collect node indices matching the requested type into the scratch buffer.
  let writeCount = 0;
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const node = nodes[nodeIndex];
    if (node && node.type === nodeType) {
      indexBuffer[writeCount++] = nodeIndex;
    }
  }

  return writeCount;
};

/** Compute mean, standard deviation and CSV string for output-node biases. */
const computeOutputBiasStats = (
  state: EngineState,
  nodes: RuntimeNode[],
  outputCount: number,
): { mean: number; std: number; biasesStr: string } => {
  // Step 1: Initialise telemetry scratch sized to the output-node count.
  const telemetryScratch = initialiseTelemetryScratch(
    {
      biasCount: outputCount,
      stringBufferLength: outputCount,
    },
    state,
  );

  // Step 2: Copy the output-node biases into the reusable scratch buffer.
  const biasScratch = telemetryScratch.biasScratch;
  const nodeIndices = state.scratch.nodeIndexBuffer;
  for (let biasIndex = 0; biasIndex < outputCount; biasIndex++) {
    const nodeIndex = nodeIndices[biasIndex];
    biasScratch[biasIndex] = nodes[nodeIndex]?.bias ?? 0;
  }

  // Step 3: Perform numerically stable running mean and variance updates for the biases.
  let mean = 0;
  let m2 = 0;
  for (let valueIndex = 0; valueIndex < outputCount; valueIndex++) {
    const value = biasScratch[valueIndex];
    const sampleNumber = valueIndex + 1;
    const delta = value - mean;
    mean += delta / sampleNumber;
    m2 += delta * (value - mean);
  }
  const std = outputCount > 0 ? Math.sqrt(m2 / outputCount) : 0;

  // Step 4: Format the biases into the pooled string buffer for CSV emission.
  const stringBuffer = telemetryScratch.stringBuffer;
  for (let biasIndex = 0; biasIndex < outputCount; biasIndex++) {
    stringBuffer[biasIndex] = biasScratch[biasIndex].toFixed(2);
  }
  const previousLength = stringBuffer.length;
  stringBuffer.length = outputCount;
  const biasesStr = stringBuffer.join(',');
  stringBuffer.length = previousLength;

  return { mean, std, biasesStr };
};

/** Input parameters for computing logit statistics. */
interface LogitStatsParams {
  state: EngineState;
  recent: number[][];
  actionDimension: number;
  reducedTelemetry: boolean;
}

/** Structure describing aggregated logit statistics. */
interface LogitStatsResult {
  meansStr: string;
  stdsStr: string;
  kurtStr: string;
  entMean: number;
  stability: number;
  stds: Float64Array;
}

/** Compute logit statistics using pooled scratch buffers. */
const computeLogitStats = ({
  state,
  recent,
  actionDimension,
  reducedTelemetry,
}: LogitStatsParams): LogitStatsResult => {
  if (!Array.isArray(recent) || recent.length === 0) {
    return {
      meansStr: '',
      stdsStr: '',
      kurtStr: '',
      entMean: 0,
      stability: 0,
      stds: state.scratch.standardDeviations,
    };
  }

  resetLogitScratch(state, actionDimension, reducedTelemetry);

  let entropySum = 0;
  if (reducedTelemetry) {
    entropySum = accumulateLogitStatsReduced(state, recent, actionDimension);
    finalizeLogitStatsReduced(state, actionDimension, recent.length);
  } else if (actionDimension === 4) {
    entropySum = accumulateLogitStatsUnrolled4(state, recent, recent.length);
    finalizeLogitStatsFull(state, actionDimension, recent.length);
  } else {
    entropySum = accumulateLogitStatsGeneric(state, recent, actionDimension);
    finalizeLogitStatsFull(state, actionDimension, recent.length);
  }

  const stability = computeDecisionStability(recent, actionDimension);
  const entMean = entropySum / recent.length;

  const meansStr = joinNumberArray(
    state,
    state.scratch.means,
    actionDimension,
    3,
  );
  const stdsStr = joinNumberArray(
    state,
    state.scratch.standardDeviations,
    actionDimension,
    3,
  );
  const kurtStr = reducedTelemetry
    ? ''
    : joinNumberArray(
        state,
        state.scratch.kurtosis ?? new Float64Array(actionDimension),
        actionDimension,
        2,
      );

  return {
    meansStr,
    stdsStr,
    kurtStr,
    entMean,
    stability,
    stds: state.scratch.standardDeviations,
  };
};

/** Prepare and zero pooled scratch buffers for logit statistics. */
const resetLogitScratch = (
  state: EngineState,
  actionDimension: number,
  reducedTelemetry: boolean,
): void => {
  // Step 1: Validate the requested action dimension before touching scratch buffers.
  const dim = Number.isFinite(actionDimension)
    ? Math.max(0, Math.floor(actionDimension))
    : 0;
  if (dim === 0) return;

  // Step 2: Acquire telemetry scratch configured for the chosen telemetry mode.
  const telemetryScratch = initialiseTelemetryScratch(
    {
      actionDimension: dim,
      includeHigherMoments: !reducedTelemetry,
    },
    state,
  );

  // Step 3: Zero all relevant buffers so accumulators start from a predictable baseline.
  telemetryScratch.meanScratch.fill(0, 0, dim);
  telemetryScratch.secondMomentScratch.fill(0, 0, dim);
  telemetryScratch.standardDeviationScratch.fill(0, 0, dim);

  if (!reducedTelemetry) {
    telemetryScratch.thirdMomentScratch?.fill(0, 0, dim);
    telemetryScratch.fourthMomentScratch?.fill(0, 0, dim);
    telemetryScratch.kurtosisScratch?.fill(0, 0, dim);
  }
};

/** Accumulate logit stats in reduced-telemetry mode (mean + variance only). */
const accumulateLogitStatsReduced = (
  state: EngineState,
  recent: number[][],
  actionDimension: number,
): number => {
  // Step 1: Abort when no samples or dimensions are present to avoid wasted work.
  const sampleCount = Array.isArray(recent) ? recent.length : 0;
  if (sampleCount === 0 || actionDimension <= 0) return 0;

  // Step 2: Iterate over each sample, updating running means/variances and entropy.
  const means = state.scratch.means;
  const secondMoment = state.scratch.secondMomentRaw;
  const stds = state.scratch.standardDeviations;
  const exponentScratch = state.scratch.exps;

  let entropyAccumulator = 0;

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const vector = recent[sampleIndex] ?? EMPTY_VECTOR;
    const sampleNumber = sampleIndex + 1;

    for (let dimIndex = 0; dimIndex < actionDimension; dimIndex++) {
      const value = vector[dimIndex] ?? 0;
      const previousMean = means[dimIndex];
      const delta = value - previousMean;
      const deltaDivided = delta / sampleNumber;
      const updatedMean = previousMean + deltaDivided;
      means[dimIndex] = updatedMean;
      const correction = value - updatedMean;
      secondMoment[dimIndex] += delta * correction;
    }

    entropyAccumulator += softmaxEntropyFromVector(vector, exponentScratch);
  }

  // Step 3: Convert the accumulated second moment into standard deviation estimates.
  const invSampleCount = 1 / sampleCount;
  for (let dimIndex = 0; dimIndex < actionDimension; dimIndex++) {
    const variance = secondMoment[dimIndex] * invSampleCount;
    stds[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
  }

  return entropyAccumulator;
};

/** Accumulate full logit stats with unrolled ACTION_DIM === 4 path. */
const accumulateLogitStatsUnrolled4 = (
  state: EngineState,
  recent: number[][],
  sampleCount: number,
): number => {
  // Step 1: Guard against empty histories to keep pooled scratch stable.
  if (!Array.isArray(recent) || sampleCount === 0) return 0; // Ensure recent is an array and sampleCount is valid

  let meanNorth = 0,
    meanEast = 0,
    meanSouth = 0,
    meanWest = 0;
  let m2North = 0,
    m2East = 0,
    m2South = 0,
    m2West = 0;
  let m3North = 0,
    m3East = 0,
    m3South = 0,
    m3West = 0;
  let m4North = 0,
    m4East = 0,
    m4South = 0,
    m4West = 0;

  let entropyAccumulator = 0;
  const exponentScratch = state.scratch.exps;

  // Step 2: Update the per-direction running moments and entropy for each recent sample.
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const vector = recent[sampleIndex] ?? EMPTY_VECTOR;
    const north = vector[0] ?? 0;
    const east = vector[1] ?? 0;
    const south = vector[2] ?? 0;
    const west = vector[3] ?? 0;
    const sampleNumber = sampleIndex + 1;

    ({
      mean: meanNorth,
      m2: m2North,
      m3: m3North,
      m4: m4North,
    } = updateMoments(
      meanNorth,
      m2North,
      m3North,
      m4North,
      north,
      sampleNumber,
    ));
    ({
      mean: meanEast,
      m2: m2East,
      m3: m3East,
      m4: m4East,
    } = updateMoments(meanEast, m2East, m3East, m4East, east, sampleNumber));
    ({
      mean: meanSouth,
      m2: m2South,
      m3: m3South,
      m4: m4South,
    } = updateMoments(
      meanSouth,
      m2South,
      m3South,
      m4South,
      south,
      sampleNumber,
    ));
    ({
      mean: meanWest,
      m2: m2West,
      m3: m3West,
      m4: m4West,
    } = updateMoments(meanWest, m2West, m3West, m4West, west, sampleNumber));

    entropyAccumulator += softmaxEntropyFromVector(vector, exponentScratch);
  }

  // Step 3: Write accumulated means and standard deviations back into the shared scratch views.
  const means = state.scratch.means;
  means[0] = meanNorth;
  means[1] = meanEast;
  means[2] = meanSouth;
  means[3] = meanWest;

  const stds = state.scratch.standardDeviations;
  const invSampleCount = 1 / sampleCount;
  const varNorth = m2North * invSampleCount;
  const varEast = m2East * invSampleCount;
  const varSouth = m2South * invSampleCount;
  const varWest = m2West * invSampleCount;
  stds[0] = varNorth > 0 ? Math.sqrt(varNorth) : 0;
  stds[1] = varEast > 0 ? Math.sqrt(varEast) : 0;
  stds[2] = varSouth > 0 ? Math.sqrt(varSouth) : 0;
  stds[3] = varWest > 0 ? Math.sqrt(varWest) : 0;

  // Step 4: Derive excess kurtosis when higher telemetry detail is enabled.
  const kurtosis = state.scratch.kurtosis;
  if (kurtosis) {
    kurtosis[0] = computeExcessKurtosis(m2North, m4North, sampleCount);
    kurtosis[1] = computeExcessKurtosis(m2East, m4East, sampleCount);
    kurtosis[2] = computeExcessKurtosis(m2South, m4South, sampleCount);
    kurtosis[3] = computeExcessKurtosis(m2West, m4West, sampleCount);
  }

  return entropyAccumulator;
};

/** Accumulate full logit stats for arbitrary action dimensions. */
const accumulateLogitStatsGeneric = (
  state: EngineState,
  recent: number[][],
  actionDimension: number,
): number => {
  // Step 1: Quickly exit for empty histories or invalid dimensions.
  const sampleCount = Array.isArray(recent) ? recent.length : 0;
  if (sampleCount === 0 || actionDimension <= 0) return 0;

  // Step 2: Ensure higher-moment scratch buffers exist before accumulation begins.
  const means = state.scratch.means;
  const m2 = state.scratch.secondMomentRaw;
  let m3 = state.scratch.thirdMomentRaw;
  let m4 = state.scratch.fourthMomentRaw;

  if (!state.toggles.reducedTelemetry) {
    if (!m3 || m3.length < actionDimension) {
      m3 = new Float64Array(actionDimension);
      state.scratch.thirdMomentRaw = m3;
    }
    if (!m4 || m4.length < actionDimension) {
      m4 = new Float64Array(actionDimension);
      state.scratch.fourthMomentRaw = m4;
    }
  }

  // Step 3: Traverse each sample, updating moments and accumulating entropy in one pass.
  const exponentScratch =
    state.scratch.exps ?? new Float64Array(actionDimension);
  let entropyAccumulator = 0;

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const vector = recent[sampleIndex] ?? EMPTY_VECTOR;
    const sampleNumber = sampleIndex + 1;

    const dimensionLimit = Math.min(actionDimension, means.length);
    for (
      let dimensionIndex = 0;
      dimensionIndex < dimensionLimit;
      dimensionIndex++
    ) {
      const value = Number.isFinite(vector[dimensionIndex])
        ? vector[dimensionIndex]
        : 0;

      const delta = value - means[dimensionIndex];
      const deltaDivided = delta / sampleNumber;
      const deltaSquared = deltaDivided * deltaDivided;
      const term1 = delta * deltaDivided * (sampleNumber - 1);

      if (!state.toggles.reducedTelemetry && m3 && m4) {
        const previousM2 = m2[dimensionIndex];
        const previousM3 = m3[dimensionIndex];
        m4[dimensionIndex] +=
          term1 *
            deltaSquared *
            (sampleNumber * sampleNumber - 3 * sampleNumber + 3) +
          6 * deltaSquared * previousM2 -
          4 * deltaDivided * previousM3;
        m3[dimensionIndex] +=
          term1 * deltaDivided * (sampleNumber - 2) -
          3 * deltaDivided * previousM2;
      }

      m2[dimensionIndex] += term1;
      means[dimensionIndex] += deltaDivided;
    }

    entropyAccumulator += softmaxEntropyFromVector(vector, exponentScratch);
  }

  // Step 4: Surface the total entropy so callers can derive mean values later on.
  return entropyAccumulator;
};

/** Finalise full logit statistics (std + kurtosis) after accumulation. */
const finalizeLogitStatsFull = (
  state: EngineState,
  actionDimension: number,
  sampleCount: number,
): void => {
  // Step 1: Validate dimension and sample counts before touching scratch buffers.
  const dim = Number.isFinite(actionDimension)
    ? Math.max(0, Math.floor(actionDimension))
    : 0;
  const samples = Number.isFinite(sampleCount) ? Math.floor(sampleCount) : 0;
  if (dim === 0) return;

  // Step 2: Translate aggregated moments into standard deviations for each dimension.
  const m2 = state.scratch.secondMomentRaw;
  const m4 = state.scratch.fourthMomentRaw ?? new Float64Array(dim);
  const stds = state.scratch.standardDeviations;
  const kurtosis = state.scratch.kurtosis ?? new Float64Array(dim);
  state.scratch.kurtosis = kurtosis;

  const invSamples = samples > 0 ? 1 / samples : 0;
  for (let dimensionIndex = 0; dimensionIndex < dim; dimensionIndex++) {
    const variance = invSamples > 0 ? m2[dimensionIndex] * invSamples : 0;
    stds[dimensionIndex] = variance > 0 ? Math.sqrt(variance) : 0;

    if (!state.toggles.reducedTelemetry) {
      // Step 3: Compute the excess kurtosis only when higher moments are being tracked.
      kurtosis[dimensionIndex] = computeExcessKurtosis(
        m2[dimensionIndex],
        m4[dimensionIndex],
        samples,
      );
    }
  }
};

/** Finalise reduced logit statistics (std only). */
const finalizeLogitStatsReduced = (
  state: EngineState,
  actionDimension: number,
  sampleCount: number,
): void => {
  // Step 1: Ensure the inputs describe a meaningful accumulation before proceeding.
  const dim = Number.isFinite(actionDimension)
    ? Math.max(0, Math.floor(actionDimension))
    : 0;
  const samples = Number.isFinite(sampleCount) ? Math.floor(sampleCount) : 0;
  if (dim === 0) return;

  // Step 2: Convert the stored second moments into standard deviations for the reduced telemetry path.
  const m2 = state.scratch.secondMomentRaw;
  const stds = state.scratch.standardDeviations;
  const invSamples = samples > 0 ? 1 / samples : 0;
  for (let dimensionIndex = 0; dimensionIndex < dim; dimensionIndex++) {
    const variance = invSamples > 0 ? m2[dimensionIndex] * invSamples : 0;
    stds[dimensionIndex] = variance > 0 ? Math.sqrt(variance) : 0;
  }
};

/** Update running moments (mean, M2, M3, M4) for a single value. */
const updateMoments = (
  previousMean: number,
  previousM2: number,
  previousM3: number,
  previousM4: number,
  value: number,
  sampleNumber: number,
): { mean: number; m2: number; m3: number; m4: number } => {
  // Step 1: Compute the deltas between the new value and the running statistics.
  const delta = value - previousMean;
  const deltaDivided = delta / sampleNumber;
  const deltaSquared = deltaDivided * deltaDivided;
  const term1 = delta * deltaDivided * (sampleNumber - 1);

  // Step 2: Apply the numerically stable one-pass moment update formulas.
  const nextM4 =
    previousM4 +
    term1 *
      deltaSquared *
      (sampleNumber * sampleNumber - 3 * sampleNumber + 3) +
    6 * deltaSquared * previousM2 -
    4 * deltaDivided * previousM3;
  const nextM3 =
    previousM3 +
    term1 * deltaDivided * (sampleNumber - 2) -
    3 * deltaDivided * previousM2;
  const nextM2 = previousM2 + term1;
  const nextMean = previousMean + deltaDivided;

  // Step 3: Return the updated moments so callers can destructure the new state.
  return { mean: nextMean, m2: nextM2, m3: nextM3, m4: nextM4 };
};

/** Compute excess kurtosis guarded against degenerate denominators. */
const computeExcessKurtosis = (
  m2: number,
  m4: number,
  sampleCount: number,
): number => {
  // Step 1: Guard against degenerate inputs that would cause division by zero or noise.
  const denominator = m2 * m2;
  if (denominator <= 0 || sampleCount <= 0 || m2 <= 1e-18) return 0;
  // Step 2: Convert the central moments into excess kurtosis, subtracting the Gaussian baseline.
  return (sampleCount * m4) / denominator - 3;
};

/** Compute softmax entropy for a vector using pooled exponent scratch. */
const softmaxEntropyFromVector = (
  vector: number[] | undefined,
  exponentScratch: Float64Array,
): number => {
  // Step 1: Bail out for empty or scalar vectors where entropy is uninformative.
  if (!vector || vector.length === 0) return 0;
  const length = Math.min(vector.length, exponentScratch.length);
  if (length <= 1) return 0;

  // Step 2: Use the tight unrolled computation for the common 4-action case to minimise overhead.
  if (length === 4) {
    const v0 = vector[0] ?? 0;
    const v1 = vector[1] ?? 0;
    const v2 = vector[2] ?? 0;
    const v3 = vector[3] ?? 0;
    let maxVal = v0;
    if (v1 > maxVal) maxVal = v1;
    if (v2 > maxVal) maxVal = v2;
    if (v3 > maxVal) maxVal = v3;
    const e0 = Math.exp(v0 - maxVal);
    const e1 = Math.exp(v1 - maxVal);
    const e2 = Math.exp(v2 - maxVal);
    const e3 = Math.exp(v3 - maxVal);
    const sum = e0 + e1 + e2 + e3 || 1;
    const p0 = e0 / sum;
    const p1 = e1 / sum;
    const p2 = e2 / sum;
    const p3 = e3 / sum;
    let entropy = 0;
    if (p0 > 0) entropy += -p0 * Math.log(p0);
    if (p1 > 0) entropy += -p1 * Math.log(p1);
    if (p2 > 0) entropy += -p2 * Math.log(p2);
    if (p3 > 0) entropy += -p3 * Math.log(p3);
    return entropy * INV_LOG4;
  }

  // Step 3: Fallback to a generic path that stabilises the softmax via max shifting.
  let maxVal = -Infinity;
  for (let index = 0; index < length; index++) {
    const value = vector[index] ?? 0;
    if (value > maxVal) maxVal = value;
  }

  let sum = 0;
  for (let index = 0; index < length; index++) {
    const expValue = Math.exp((vector[index] ?? 0) - maxVal);
    exponentScratch[index] = expValue;
    sum += expValue;
  }
  if (!sum) sum = 1;

  let entropy = 0;
  for (let index = 0; index < length; index++) {
    const probability = exponentScratch[index] / sum;
    if (probability > 0) entropy += -probability * Math.log(probability);
  }
  const denominator = Math.log(length);
  return denominator > 0 ? entropy / denominator : 0;
};

/** Compute decision stability over a sequence of logits vectors. */
const computeDecisionStability = (
  recent: number[][],
  actionDimension: number,
): number => {
  // Step 1: Skip stability analysis when fewer than two steps are available.
  const sequenceLength = recent?.length ?? 0;
  if (sequenceLength < 2) return 0;

  // Step 2: Track consecutive argmax actions to determine the proportion of stable transitions.
  let stablePairCount = 0;
  let pairCount = 0;
  let previousArgmax = -1;
  const dim = Math.max(0, actionDimension);
  const unrolled = dim === 4;

  for (let rowIndex = 0; rowIndex < sequenceLength; rowIndex++) {
    const row = recent[rowIndex];
    if (!row || row.length === 0) continue;
    let argmax = 0;
    if (unrolled && row.length >= 4) {
      let bestValue = row[0] ?? 0;
      argmax = 0;
      const east = row[1] ?? 0;
      if (east > bestValue) {
        bestValue = east;
        argmax = 1;
      }
      const south = row[2] ?? 0;
      if (south > bestValue) {
        bestValue = south;
        argmax = 2;
      }
      const west = row[3] ?? 0;
      if (west > bestValue) argmax = 3;
    } else {
      let bestValue = row[0] ?? 0;
      argmax = 0;
      for (
        let dimIndex = 1;
        dimIndex < dim && dimIndex < row.length;
        dimIndex++
      ) {
        const candidate = row[dimIndex] ?? 0;
        if (candidate > bestValue) {
          bestValue = candidate;
          argmax = dimIndex;
        }
      }
    }

    if (previousArgmax !== -1) {
      pairCount += 1;
      if (previousArgmax === argmax) stablePairCount += 1;
    }
    previousArgmax = argmax;
  }

  return pairCount ? stablePairCount / pairCount : 0;
};

/** Join numeric arrays into a comma-separated string using the pooled string buffer. */
const joinNumberArray = (
  state: EngineState,
  arrayLike: ArrayLike<number>,
  length: number,
  digits = 3,
): string => {
  // Step 1: Validate inputs and clamp requested precision to a safe range.
  if (!arrayLike) return '';
  const availableLength = arrayLike.length >>> 0;
  if (!Number.isFinite(length) || length <= 0 || availableLength === 0)
    return '';

  const effectiveLength = length > availableLength ? availableLength : length;
  let fixedDigits = digits;
  if (!Number.isFinite(fixedDigits)) fixedDigits = 0;
  if (fixedDigits < 0) fixedDigits = 0;
  if (fixedDigits > 20) fixedDigits = 20;

  // Step 2: Ensure the pooled string buffer can hold the requested slice.
  let stringScratch = state.scratch.stringAssemblyBuffer;
  if (!Array.isArray(stringScratch)) {
    stringScratch = new Array(effectiveLength);
    state.scratch.stringAssemblyBuffer = stringScratch;
  }
  if (effectiveLength > stringScratch.length) {
    const nextSize = 1 << Math.ceil(Math.log2(effectiveLength));
    stringScratch = new Array(nextSize);
    state.scratch.stringAssemblyBuffer = stringScratch;
  }

  // Step 3: Populate the scratch buffer with formatted numbers and produce the joined string.
  for (let index = 0; index < effectiveLength; index++) {
    const rawValue = arrayLike[index] ?? 0;
    stringScratch[index] = Number.isFinite(rawValue)
      ? Number(rawValue).toFixed(fixedDigits)
      : 'NaN';
  }

  // Step 4: Join the formatted slice and restore the buffer length for future reuse.
  const previousLength = stringScratch.length;
  stringScratch.length = effectiveLength;
  const joined = stringScratch.join(',');
  stringScratch.length = previousLength;
  return joined;
};

/** Knuth-derived 32-bit hash constant reused by exploration helpers. */
const HASH_KNUTH_32 = 2654435761 >>> 0;

/**
 * Log comprehensive telemetry for a completed generation.
 *
 * This orchestrator function coordinates all telemetry logging for a generation,
 * including action entropy, output biases, logits statistics, exploration metrics,
 * and diversity metrics. It respects the minimal telemetry mode and performs
 * optional profiling when enabled.
 *
 * Design Rationale:
 *  - Orchestrates calls to individual telemetry functions (logActionEntropy, etc.)
 *  - Respects minimal telemetry toggle (early exit when disabled)
 *  - Best-effort error handling (swallow all exceptions)
 *  - Optional profiling accumulation when details are enabled
 *  - Accepts callback for anti-collapse recovery to avoid tight coupling
 *
 * Telemetry Steps (in order):
 *  1. Action entropy - normalized entropy and unique move statistics
 *  2. Output bias statistics - mean, std, and individual biases
 *  3. Logits statistics - means, stds, kurtosis, entropy, stability, collapse detection
 *  4. Exploration metrics - distinct coordinates visited and progress
 *  5. Diversity metrics - species richness, Simpson index, weight variance
 *
 * Parameters:
 * @param engineState - Shared engine state with scratch buffers and profiling
 * @param neat - NEAT instance with population
 * @param fittest - Best network/genome from this generation
 * @param genResult - Generation result with simulation data
 * @param generationIndex - Current generation index
 * @param writeLog - Safe logging function
 * @param actionDimension - Number of possible actions (for entropy normalization)
 * @param recentWindow - Window size for logits tail sampling
 * @param reducedTelemetry - Whether reduced telemetry mode is active
 * @param telemetryMinimal - Whether minimal telemetry mode is active (skip all if true)
 * @param onCollapseRecovery - Callback invoked when collapse is detected (for recovery)
 * @param isProfilingDetailsEnabledFn - Function to check profiling state
 * @param profilingStartTimestampFn - Function to get profiling start time
 * @param accumulateProfilingDurationFn - Function to accumulate profiling duration
 *
 * @example
 * // Log telemetry for generation 42
 * logGenerationTelemetry(
 *   state, neat, fittestNetwork, result, 42, console.log, 4, 40, false, false,
 *   () => antiCollapseRecovery(...),
 *   isProfilingDetailsEnabled, profilingStartTimestamp, accumulateProfilingDuration
 * );
 */
export const logGenerationTelemetry = (
  engineState: EngineState,
  neat: Neat,
  fittest: Network | undefined,
  genResult: GenerationResult | undefined,
  generationIndex: number,
  writeLog: (msg: string) => void,
  actionDimension: number,
  recentWindow: number,
  reducedTelemetry: boolean,
  telemetryMinimal: boolean,
  onCollapseRecovery: () => void,
  isProfilingDetailsEnabledFn: (state: EngineState) => boolean,
  profilingStartTimestampFn: (state: EngineState) => number,
  accumulateProfilingDurationFn: (
    state: EngineState,
    label: string,
    duration: number,
  ) => void,
): void => {
  // Step 0: Global guard for minimal telemetry mode.
  if (telemetryMinimal) return;

  // Start profiling window if enabled.
  const profilingEnabled = isProfilingDetailsEnabledFn(engineState);
  const profilingStart = profilingEnabled
    ? profilingStartTimestampFn(engineState)
    : 0;

  try {
    // Step 1: Action entropy telemetry
    logActionEntropy({
      state: engineState,
      generationResult: genResult,
      generationIndex,
      safeWrite: writeLog,
    });

    // Step 2: Output bias statistics (fittest may be undefined early on)
    logOutputBiasStats({
      state: engineState,
      fittest,
      generationIndex,
      safeWrite: writeLog,
    });

    // Step 3: Logits statistics and collapse detection/recovery
    logLogitsAndCollapse({
      state: engineState,
      neat,
      fittest,
      generationIndex,
      safeWrite: writeLog,
      actionDimension,
      recentWindow,
      reducedTelemetry,
      onCollapseRecovery,
    });

    // Step 4: Exploration telemetry (path uniqueness, progress)
    logExploration({
      state: engineState,
      generationResult: genResult,
      generationIndex,
      safeWrite: writeLog,
    });

    // Step 5: Diversity metrics (species richness, Simpson, weight std)
    logDiversity({
      state: engineState,
      neat,
      generationIndex,
      safeWrite: writeLog,
    });
  } catch {
    // Swallow any unexpected telemetry exception to avoid disrupting the evolution core loop.
  }

  // Step 6: Record profiling delta if profiling was enabled at entry.
  if (profilingEnabled) {
    accumulateProfilingDurationFn(
      engineState,
      'telemetry',
      profilingStartTimestampFn(engineState) - profilingStart || 0,
    );
  }
};
