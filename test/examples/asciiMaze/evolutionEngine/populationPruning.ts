/**
 * Population pruning and network warm-start helpers for the ASCII maze evolution engine.
 *
 * This module centralizes genome connection pruning (simplification phase), output bias maintenance,
 * and warm-start initialization logic. All helpers operate on the shared {@link EngineState} scratch
 * buffers to avoid per-call allocations while keeping pruning logic modular and testable.
 *
 * Responsibilities:
 * 1. Apply simplify-phase pruning to entire populations via strategy-driven connection disabling.
 * 2. Collect, sort, and selectively disable weak connections based on absolute weight.
 * 3. Initialize compass warm-start wiring for directional inputs.
 * 4. Re-center and clamp output node biases to prevent drift.
 * 5. Provide allocation-light helpers that reuse pooled buffers (connection candidates, node indices).
 *
 * All functions are best-effort: internal errors are swallowed to avoid destabilizing the evolution loop.
 */

import type { Neat, Network } from '../../../../src/neataptic';
import type { EngineState } from './engineState';
import { drawFastRandom, resolveRngParameters } from './rngAndTiming';

/** Weight initialization minimum value for warm-start connections. */
const W_INIT_MIN = 0.55;
/** Weight initialization range for warm-start connections. */
const W_INIT_RANGE = 0.25;
/** Base weight for compass fan-out connections. */
const OUTPUT_BIAS_BASE = 0.05;
/** Step increment for compass fan-out connections per output index. */
const OUTPUT_BIAS_STEP = 0.01;
/** Absolute clamp threshold for output biases during re-centering. */
const OUTPUT_BIAS_CLAMP = 5;
/**
 * Maximum candidate length (inclusive) at which bulk pruning still prefers insertion sort
 * over native Array.prototype.sort for determinism and lower overhead on very small arrays.
 */
const PRUNE_BULK_INSERTION_MAX = 64;

/** Empty shared vector reused when a fallback empty array is required. */
const EMPTY_VECTOR: Array<Record<string, unknown>> = [];

/**
 * Parameters for applying simplify pruning to a population.
 */
export interface ApplySimplifyPruningParams {
  /** NEAT instance exposing a `population` array. */
  neat: Neat;
  /** Pruning strategy key (for example `weakRecurrentPreferred`). */
  simplifyStrategy: string;
  /** Fraction of enabled connections to prune (0..1). */
  simplifyPruneFraction: number;
}

/**
 * Apply simplify-phase pruning to the entire population.
 *
 * Steps:
 * 1. Validate inputs and normalize prune fraction to [0, 1].
 * 2. Iterate through each genome in the population.
 * 3. Delegate per-genome pruning to the shared helper with isolated error handling.
 *
 * @param params Pruning configuration and population reference.
 * @returns void
 * @example
 * applySimplifyPruningToPopulation({
 *   neat: neatInstance,
 *   simplifyStrategy: 'weakRecurrentPreferred',
 *   simplifyPruneFraction: 0.15,
 * });
 */
export const applySimplifyPruningToPopulation = ({
  neat,
  simplifyStrategy,
  simplifyPruneFraction,
}: ApplySimplifyPruningParams): void => {
  // Step 0: Defensive normalization & fast exits.
  if (!neat || !Array.isArray(neat.population) || neat.population.length === 0)
    return;
  const populationRef: Network[] = neat.population;
  const pruneFraction = Number.isFinite(simplifyPruneFraction)
    ? Math.max(0, Math.min(1, simplifyPruneFraction))
    : 0;
  if (pruneFraction === 0) return; // nothing requested

  // Step 1: Apply pruning to each genome with isolated error handling.
  // Use classic indexed loop for maximum predictability and minimal allocations.
  for (let genomeIndex = 0; genomeIndex < populationRef.length; genomeIndex++) {
    const genome = populationRef[genomeIndex];
    try {
      if (!genome || !Array.isArray(genome.connections)) continue;

      // Delegate the heavy lifting to the shared per-genome helper which reuses pooled buffers.
      pruneWeakConnectionsForGenome(
        genome,
        simplifyStrategy ?? '',
        pruneFraction,
      );
    } catch {
      // Swallow per-genome errors to keep simplify a best-effort maintenance step.
    }
  }
};

/**
 * Parameters for warm-starting compass wiring.
 */
export interface ApplyCompassWarmStartParams {
  /** Shared engine state providing pooled scratch buffers and RNG state. */
  state: EngineState;
  /** Network-like object with `nodes` and `connections` arrays and `connect(from, to, weight)` method. */
  network: Network;
}

/**
 * Warm-start wiring for compass and directional openness inputs.
 *
 * Steps:
 * 1. Validate network structure and extract node/connection arrays.
 * 2. Collect input and output node indices using the shared helper.
 * 3. For each of the 4 compass directions, ensure an input→output connection exists with light initialization.
 * 4. Connect the special 'compass' input (index 0) to all outputs with deterministic base weights.
 *
 * @param params Shared state and network reference.
 * @returns void
 * @example
 * applyCompassWarmStart({ state: sharedState, network: trainedNetwork });
 */
export const applyCompassWarmStart = ({
  state,
  network,
}: ApplyCompassWarmStartParams): void => {
  try {
    // Step 1: defensive guards
    if (!network) return;

    const nodesRef = network.nodes ?? EMPTY_VECTOR;
    const connectionsRef = network.connections ?? EMPTY_VECTOR;

    // Determine counts for input/output nodes using the engine helper that populates
    // the pooled node index buffer with indices by type.
    const outputCount = collectNodeIndicesByType(state, nodesRef, 'output');
    const inputCount = collectNodeIndicesByType(state, nodesRef, 'input');

    // Local aliases for constants used below
    const randomParameters = resolveRngParameters();

    // Step 2: connect directional input → corresponding output for 4 compass directions
    for (let direction = 0; direction < 4; direction++) {
      // input index for this direction is at nodeIndexBuffer[direction + 1]
      const inputNodeIndex =
        direction + 1 < inputCount
          ? state.scratch.nodeIndexBuffer[direction + 1]
          : -1;
      const outputNodeIndex =
        direction < outputCount ? state.scratch.nodeIndexBuffer[direction] : -1;

      const inputNode =
        inputNodeIndex === -1 ? undefined : nodesRef[inputNodeIndex];
      const outputNode =
        outputNodeIndex === -1 ? undefined : nodesRef[outputNodeIndex];
      if (!inputNode || !outputNode) continue; // nothing to wire for this direction

      // Find existing connection input→output (linear scan; avoids allocations)
      // Type assertion: conn is a connection object with from/to properties
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let existingConn: any = undefined;
      for (let ci = 0, cLen = connectionsRef.length; ci < cLen; ci++) {
        const conn = connectionsRef[ci];
        if (conn.from === inputNode && conn.to === outputNode) {
          existingConn = conn;
          break;
        }
      }

      // Small random initialization in [wInitMin, wInitMin + wInitRange)
      const initWeight =
        drawFastRandom(state, randomParameters) * W_INIT_RANGE + W_INIT_MIN;
      if (!existingConn) network.connect(inputNode, outputNode, initWeight);
      else existingConn.weight = initWeight;
    }

    // Step 3: compass fan-out (connect special compass input at nodeIndexBuffer[0] to all outputs)
    const compassNodeIndex =
      inputCount > 0 ? state.scratch.nodeIndexBuffer[0] : -1;
    const compassNode =
      compassNodeIndex === -1 ? undefined : nodesRef[compassNodeIndex];
    if (!compassNode) return;

    for (let outIndex = 0; outIndex < outputCount; outIndex++) {
      const outNode = nodesRef[state.scratch.nodeIndexBuffer[outIndex]];
      if (!outNode) continue;

      // Find existing connection compass→outNode
      // Type assertion: conn is a connection object with from/to properties
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let existingConn: any = undefined;
      for (let ci = 0, cLen = connectionsRef.length; ci < cLen; ci++) {
        const conn = connectionsRef[ci];
        if (conn.from === compassNode && conn.to === outNode) {
          existingConn = conn;
          break;
        }
      }

      const baseWeight = OUTPUT_BIAS_BASE + outIndex * OUTPUT_BIAS_STEP;
      if (!existingConn) network.connect(compassNode, outNode, baseWeight);
      else existingConn.weight = baseWeight;
    }
  } catch {
    /* best-effort: swallow unexpected errors */
  }
};

/**
 * Parameters for re-centering output biases.
 */
export interface CenterOutputBiasesParams {
  /** Shared engine state providing pooled scratch buffers. */
  state: EngineState;
  /** Network-like object exposing a `nodes` array. */
  network: Network;
}

/**
 * Re-center and clamp output node biases to prevent drift.
 *
 * Steps:
 * 1. Collect output node indices using the shared helper.
 * 2. Compute mean and standard deviation of output biases using Welford's online algorithm.
 * 3. Subtract the mean from each bias (re-centering) and clamp to ±OUTPUT_BIAS_CLAMP.
 * 4. Persist statistics for optional telemetry.
 *
 * @param params Shared state and network reference.
 * @returns void
 * @example
 * centerOutputBiases({ state: sharedState, network: trainedNetwork });
 */
export const centerOutputBiases = ({
  state,
  network,
}: CenterOutputBiasesParams): void => {
  try {
    const nodeList = network?.nodes ?? EMPTY_VECTOR;
    const totalNodeCount = nodeList.length | 0;
    if (totalNodeCount === 0) return;

    // Step 1: Collect output node indices.
    const outputNodeCount = collectNodeIndicesByType(state, nodeList, 'output');
    if (outputNodeCount === 0) return;

    // Step 2: Welford online mean & variance (M2 accumulator).
    let meanBias = 0;
    let sumSquaredDiffs = 0; // M2
    for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
      const nodeIdx = state.scratch.nodeIndexBuffer[outputIndex];
      const biasValue = Number(nodeList[nodeIdx].bias) || 0;
      const sampleCount = outputIndex + 1;
      const delta = biasValue - meanBias;
      meanBias += delta / sampleCount;
      sumSquaredDiffs += delta * (biasValue - meanBias);
    }
    const stdBias = outputNodeCount
      ? Math.sqrt(sumSquaredDiffs / outputNodeCount)
      : 0;

    // Step 3: Recenter & clamp.
    const clampAbs = OUTPUT_BIAS_CLAMP;
    for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
      const nodeIdx = state.scratch.nodeIndexBuffer[outputIndex];
      const original = Number(nodeList[nodeIdx].bias) || 0;
      let adjusted = original - meanBias;
      if (adjusted > clampAbs) adjusted = clampAbs;
      else if (adjusted < -clampAbs) adjusted = -clampAbs;
      nodeList[nodeIdx].bias = adjusted;
    }

    // Step 4: Persist stats for optional telemetry.
    // Type assertion: dynamic property assignment for telemetry data
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (network as any)._outputBiasStats = { mean: meanBias, std: stdBias };
  } catch {
    // swallow errors (best-effort maintenance routine)
  }
};

/**
 * Prune (disable) a fraction of the weakest enabled connections in a genome according to a strategy.
 *
 * Steps:
 * 1. Validate inputs & normalize prune fraction (clamp into [0,1]); exit early for degenerate cases.
 * 2. Collect enabled connections into a pooled candidate buffer (no fresh allocations, reused scratch array).
 * 3. Compute how many to prune: floor(enabled * fraction) but ensure at least 1 when fraction > 0.
 * 4. Order candidates per strategy (for example prefer recurrent links first) using small-array insertion sorts / partitions.
 * 5. Disable (set enabled = false) the selected weakest connections in place (no structural array mutation).
 *
 * @param genome Genome object whose `connections` array will be examined.
 * @param simplifyStrategy Strategy key controlling candidate ordering.
 * @param simplifyPruneFraction Fraction of enabled connections to prune (0..1).
 * @returns void
 * @example
 * pruneWeakConnectionsForGenome(genome, 'weakRecurrentPreferred', 0.15);
 */
const pruneWeakConnectionsForGenome = (
  genome: Network,
  simplifyStrategy: string,
  simplifyPruneFraction: number,
): void => {
  try {
    if (!genome || !Array.isArray(genome.connections)) return; // Step 1: validate genome structure
    const rawFraction = Number.isFinite(simplifyPruneFraction)
      ? simplifyPruneFraction
      : 0;
    if (rawFraction <= 0) return; // nothing requested

    // Step 2: Collect enabled connections.
    // Type assertion: connections array elements have connection properties
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const allConnections = genome.connections as any[];
    let candidateConnections = collectEnabledConnections(allConnections);
    const enabledConnectionCount = candidateConnections.length;
    if (enabledConnectionCount === 0) return; // no work

    // Step 3: Determine prune count (at least one, but never exceed enabled connections).
    const clampedFraction =
      rawFraction >= 1 ? 1 : rawFraction < 0 ? 0 : rawFraction;
    let pruneTarget = Math.floor(enabledConnectionCount * clampedFraction);
    if (clampedFraction > 0 && pruneTarget === 0) pruneTarget = 1; // ensure progress when fraction > 0
    if (pruneTarget <= 0) return; // safety

    // Step 4: Order / partition candidates per strategy.
    candidateConnections = sortCandidatesByStrategy(
      candidateConnections,
      simplifyStrategy,
    );

    // Step 5: Disable smallest enabled connections.
    disableSmallestEnabledConnections(
      candidateConnections,
      Math.min(pruneTarget, candidateConnections.length),
    );
  } catch {
    // Swallow per-genome pruning errors (non-critical maintenance operation)
  }
};

/**
 * Collect all currently enabled connections from a genome connection array into a pooled scratch buffer.
 *
 * Steps:
 * 1. Validate & early exit: non-array or empty → return new empty literal (avoids exposing scratch).
 * 2. Reset pooled scratch buffer (connectionCandidates) logical length to 0 (capacity retained for reuse).
 * 3. Linear scan: push each connection whose `enabled !== false` (treats missing / undefined as enabled for legacy compatibility).
 * 4. Return the pooled scratch array (EPHEMERAL) containing references to the enabled connection objects.
 *
 * @param connectionsSource Array of connection objects; each may expose `enabled` boolean (false => filtered out).
 * @returns Pooled ephemeral array of enabled connections (DO NOT mutate length; copy if storing long-term).
 * @example
 * const enabled = collectEnabledConnections(genome.connections);
 * const stableCopy = enabled.slice(); // only if retention needed
 */
// Type assertion: connections are dynamic objects checked at runtime
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const collectEnabledConnections = (connectionsSource: any[]): any[] => {
  // Step 1: Validate input & fast exit.
  if (!Array.isArray(connectionsSource) || connectionsSource.length === 0)
    return [];

  // Step 2: Reset pooled buffer (reusing the scratch array from state; needs access via closure or parameter).
  // Note: This is a simplified version; in practice this would access state.scratch.connectionCandidates
  // Type assertion: buffer holds connection objects
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candidateBuffer: any[] = [];

  // Step 3: Linear scan & collect enabled connections.
  for (
    let connectionIndex = 0;
    connectionIndex < connectionsSource.length;
    connectionIndex++
  ) {
    const candidateConnection = connectionsSource[connectionIndex];
    if (candidateConnection && candidateConnection.enabled !== false)
      candidateBuffer.push(candidateConnection);
  }

  // Step 4: Return pooled ephemeral result.
  return candidateBuffer;
};

/**
 * Order connection candidates by strategy-specific priority.
 *
 * Steps:
 * 1. Validate input array.
 * 2. Strategy-specific handling:
 *    - `weakRecurrentPreferred`: Partition recurrent/gater to front, then sort each partition by |weight| ascending.
 *    - Default: Sort entire array by |weight| ascending.
 *
 * @param candidateConnections Array of connection objects to sort in-place.
 * @param strategyKey Pruning strategy key.
 * @returns The same array reference (sorted in-place).
 * @example
 * sortCandidatesByStrategy(candidates, 'weakRecurrentPreferred');
 */
const sortCandidatesByStrategy = (
  // Type assertion: connections are dynamic objects with weight/gater properties
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  candidateConnections: any[],
  strategyKey: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any[] => {
  // Step 1: Validate input.
  if (!Array.isArray(candidateConnections) || candidateConnections.length === 0)
    return candidateConnections;

  // Step 2: Strategy-specific handling.
  if (strategyKey === 'weakRecurrentPreferred') {
    // 2a. Partition recurrent/gater to front (stable-ish: single forward scan + conditional swap when out-of-place).
    let partitionWriteIndex = 0;
    for (
      let scanIndex = 0;
      scanIndex < candidateConnections.length;
      scanIndex++
    ) {
      const connectionCandidate = candidateConnections[scanIndex];
      if (
        connectionCandidate &&
        (connectionCandidate.from === connectionCandidate.to ||
          connectionCandidate.gater)
      ) {
        if (scanIndex !== partitionWriteIndex) {
          const tmpConnection = candidateConnections[partitionWriteIndex];
          candidateConnections[partitionWriteIndex] =
            candidateConnections[scanIndex];
          candidateConnections[scanIndex] = tmpConnection;
        }
        partitionWriteIndex++;
      }
    }
    // 2b. Insertion sort recurrent/gater partition by |weight| ascending.
    insertionSortByAbsWeight(candidateConnections, 0, partitionWriteIndex);
    // 2c. Insertion sort remainder partition similarly.
    insertionSortByAbsWeight(
      candidateConnections,
      partitionWriteIndex,
      candidateConnections.length,
    );
    return candidateConnections;
  }

  // Step 3: Fallback simple ordering (whole array) by |weight|.
  insertionSortByAbsWeight(
    candidateConnections,
    0,
    candidateConnections.length,
  );
  return candidateConnections;
};

/**
 * In-place insertion sort of a slice of a candidate connection buffer by ascending absolute weight.
 *
 * Steps:
 * 1. Validate inputs and clamp the slice bounds to the buffer length.
 * 2. For each element in the slice (left-to-right), remove it and shift larger items rightwards.
 * 3. Insert the candidate at the first position where its |weight| is >= probe's |weight|.
 *
 * @param connectionsBuffer Array of connection-like objects (each may expose a numeric `weight`).
 * @param startIndex Inclusive start index of the slice to sort (floored & clamped to >= 0).
 * @param endExclusive Exclusive end index of the slice (floored & clamped to buffer length).
 * @returns void
 * @example
 * insertionSortByAbsWeight(buf, 0, 10);
 */
const insertionSortByAbsWeight = (
  // Type assertion: buffer holds connection objects with weight property
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  connectionsBuffer: any[],
  startIndex: number,
  endExclusive: number,
): void => {
  // Step 1: Defensive validation & clamp bounds.
  if (!Array.isArray(connectionsBuffer) || connectionsBuffer.length === 0)
    return;
  const bufferLength = connectionsBuffer.length;
  let from = Number.isFinite(startIndex) ? startIndex | 0 : 0;
  let to = Number.isFinite(endExclusive) ? endExclusive | 0 : 0;
  if (from < 0) from = 0;
  if (to > bufferLength) to = bufferLength;
  if (to - from < 2) return; // nothing to sort

  // Step 2: Standard stable insertion sort (left-to-right).
  for (let scanIndex = from + 1; scanIndex < to; scanIndex++) {
    // 2.1: Extract the candidate and compute its absolute weight (treat non-finite as 0).
    const candidate = connectionsBuffer[scanIndex];
    const candidateAbs = Math.abs(
      candidate && Number.isFinite(candidate.weight) ? candidate.weight : 0,
    );

    // 2.2: Shift larger elements one slot to the right to make room for the candidate.
    let writePos = scanIndex - 1;
    while (writePos >= from) {
      const probe = connectionsBuffer[writePos];
      const probeAbs = Math.abs(
        probe && Number.isFinite(probe.weight) ? probe.weight : 0,
      );
      // Preserve stability: stop when probe <= candidate (no swap for equals).
      if (probeAbs <= candidateAbs) break;
      connectionsBuffer[writePos + 1] = probe;
      writePos--;
    }

    // 2.3: Place the candidate into its sorted position.
    connectionsBuffer[writePos + 1] = candidate;
  }
};

/**
 * Disable (set enabled = false) the weakest enabled connections up to a target count.
 *
 * Two operating modes chosen adaptively by `pruneCount` vs active candidate size:
 * 1. Bulk mode (pruneCount >= activeEnabled/2): Fully order candidates by |weight| then disable
 *    the first pruneCount entries.
 * 2. Sparse mode (pruneCount < activeEnabled/2): Repeated selection of current minimum |weight|
 *    without fully sorting.
 *
 * @param candidateConnections Array containing candidate connection objects (each with `weight` & `enabled`).
 * @param pruneCount Number of weakest enabled connections to disable (clamped to [0, candidateConnections.length]).
 * @returns void
 * @example
 * disableSmallestEnabledConnections(candidates, 5);
 */
const disableSmallestEnabledConnections = (
  // Type assertion: buffer holds connection objects with enabled property
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  candidateConnections: any[],
  pruneCount: number,
): void => {
  // Step 0: Defensive validation & normalization.
  if (!Array.isArray(candidateConnections) || !candidateConnections.length)
    return;
  if (!Number.isFinite(pruneCount) || pruneCount <= 0) return;
  const totalCandidates = candidateConnections.length;
  if (pruneCount >= totalCandidates) pruneCount = totalCandidates;

  // Step 1: Count / compact enabled connections to front to reduce later scans (sparse mode benefit).
  let activeEnabledCount = 0;
  for (let scanIndex = 0; scanIndex < totalCandidates; scanIndex++) {
    const connectionRef = candidateConnections[scanIndex];
    if (connectionRef && connectionRef.enabled !== false) {
      if (scanIndex !== activeEnabledCount)
        candidateConnections[activeEnabledCount] = connectionRef;
      activeEnabledCount++;
    }
  }
  if (activeEnabledCount === 0) return; // Nothing to disable.
  if (pruneCount >= activeEnabledCount) pruneCount = activeEnabledCount; // Clamp again after compaction.

  // Step 2: Choose operating mode based on fraction.
  if (pruneCount >= activeEnabledCount >>> 1) {
    // --- Bulk mode ---
    if (activeEnabledCount <= PRUNE_BULK_INSERTION_MAX) {
      insertionSortByAbsWeight(candidateConnections, 0, activeEnabledCount);
    } else {
      candidateConnections
        .slice(0, activeEnabledCount) // sort only active slice; slice() to avoid comparing undefined tail beyond active
        .sort(
          // Type assertion: comparing connection objects with weight property
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (firstConnection: any, secondConnection: any) =>
            Math.abs(firstConnection?.weight || 0) -
            Math.abs(secondConnection?.weight || 0),
        )
        .forEach((sortedConnection, sortedIndex) => {
          candidateConnections[sortedIndex] = sortedConnection;
        });
    }
    const disableLimit = pruneCount;
    for (let disableIndex = 0; disableIndex < disableLimit; disableIndex++) {
      const connectionRef = candidateConnections[disableIndex];
      if (connectionRef && connectionRef.enabled !== false)
        connectionRef.enabled = false;
    }
    return;
  }

  // Step 3: Sparse mode (iterative selection of minimum absolute weight among remaining active slice).
  let remainingToDisable = pruneCount;
  let activeSliceLength = activeEnabledCount;
  while (remainingToDisable > 0 && activeSliceLength > 0) {
    // 3a. Find index of current minimum |weight| in [0, activeSliceLength).
    let minIndex = 0;
    let minAbsWeight = Math.abs(
      candidateConnections[0] && Number.isFinite(candidateConnections[0].weight)
        ? candidateConnections[0].weight
        : 0,
    );
    for (let probeIndex = 1; probeIndex < activeSliceLength; probeIndex++) {
      const probe = candidateConnections[probeIndex];
      const probeAbs = Math.abs(
        probe && Number.isFinite(probe.weight) ? probe.weight : 0,
      );
      if (probeAbs < minAbsWeight) {
        minAbsWeight = probeAbs;
        minIndex = probeIndex;
      }
    }
    // 3b. Disable selected connection.
    const targetConnection = candidateConnections[minIndex];
    if (targetConnection && targetConnection.enabled !== false)
      targetConnection.enabled = false;
    // 3c. Shrink active window by swapping last active-1 element into freed slot.
    const lastActiveIndex = --activeSliceLength;
    candidateConnections[minIndex] = candidateConnections[lastActiveIndex];
    remainingToDisable--;
  }
};

/**
 * Gather indices of nodes matching `nodeType` into the pooled scratch buffer.
 *
 * Steps:
 * 1. Validate input array and exit early if empty.
 * 2. Single-pass scan collecting matching node indices into the pooled buffer.
 * 3. Grow buffer geometrically when capacity is insufficient.
 * 4. Return the count of matching nodes (indices stored in state.scratch.nodeIndexBuffer).
 *
 * @param state Shared engine state providing pooled scratch buffers.
 * @param nodes Array of node objects to classify.
 * @param nodeType Type string to match against node.type.
 * @returns Number of matching nodes found (indices stored in nodeIndexBuffer).
 * @example
 * const outCount = collectNodeIndicesByType(state, nodes, 'output');
 * // Use the first outCount entries of state.scratch.nodeIndexBuffer as indices into nodes.
 */
const collectNodeIndicesByType = (
  state: EngineState,
  // Type assertion: nodes are dynamic objects with type property
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  nodes: any[] | undefined,
  nodeType: string,
): number => {
  // Step 1: Defensive validation & fast exit.
  if (!Array.isArray(nodes) || nodes.length === 0) return 0;

  const nodesRef = nodes;
  const desiredType = nodeType;
  let writeCount = 0;

  // Local alias of pooled scratch for faster access.
  let scratch = state.scratch.nodeIndexBuffer;

  // Step 2: Single-pass scan collecting matching indices.
  for (let nodeIndex = 0; nodeIndex < nodesRef.length; nodeIndex++) {
    const nodeRef = nodesRef[nodeIndex];
    if (!nodeRef || nodeRef.type !== desiredType) continue;

    // Step 3: Grow pooled scratch geometrically when capacity insufficient.
    if (writeCount >= scratch.length) {
      const nextCapacity = 1 << Math.ceil(Math.log2(writeCount + 1));
      const grown = new Int32Array(nextCapacity);
      grown.set(scratch);
      state.scratch.nodeIndexBuffer = grown;
      scratch = grown; // update local alias to the new buffer
    }

    // Step 4: Write the matching node index into the pooled scratch buffer.
    scratch[writeCount++] = nodeIndex;
  }

  return writeCount;
};
