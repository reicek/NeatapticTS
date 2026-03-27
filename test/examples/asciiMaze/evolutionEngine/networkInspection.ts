/**
 * Network Inspection Module
 *
 * Purpose:
 * -------
 * Provides utilities for inspecting and analyzing neural network topology,
 * including node classification, activation function detection, and connection
 * analysis (recurrent/gated detection).
 *
 * This module encapsulates:
 *  - Node classification (input/hidden/output buckets)
 *  - Activation function name gathering
 *  - Recurrent and gated connection detection
 *  - Network structure printing for debugging
 *
 * ES2023 Policy:
 * -------------
 * - Uses nullish coalescing `??` and optional chaining `?.`
 * - Descriptive variable names (no short identifiers)
 * - Pooled scratch buffers to avoid allocations
 * - Best-effort error handling (swallow non-fatal errors)
 *
 * @module evolutionEngine/networkInspection
 */

import type { INetwork } from '../interfaces';
import type { EngineState } from './engineState';
import type { NetworkConnection, NetworkNode } from './evolutionEngine.types';
import { ensureConnFlagsCapacity } from './scratchPools';

interface ClassifiedNodeBuckets {
  nodeList: NetworkNode[];
  inputNodes: NetworkNode[];
  hiddenNodes: NetworkNode[];
  outputNodes: NetworkNode[];
}

type NodeBucketPool = [NetworkNode[], NetworkNode[], NetworkNode[]];

/**
 * Utility to explicitly mark swallowed errors for lint compliance.
 *
 * This function is used to explicitly void error objects in catch blocks
 * to satisfy linters that flag unused catch binding variables. It has no
 * runtime effect beyond the void operation.
 *
 * Design Rationale:
 *  - Explicit intent signaling for best-effort error handling
 *  - Lint-compliant alternative to catch binding suppression
 *  - Zero runtime overhead (optimized away by JIT)
 *
 * @param error - The error object to void (can be any type)
 *
 * @example
 * try {
 *   riskyOperation();
 * } catch (error) {
 *   swallowError(error); // Explicit void for lint compliance
 * }
 */
export const swallowError = (error: unknown): void => {
  void error;
};

/**
 * Normalize the incoming network into a safe node list reference.
 *
 * Small helper to keep the main method focused on orchestration.
 * Fast-guard: accept only arrays, fall back to empty array when missing.
 *
 * @param network - Network-like object with an optional `nodes` array.
 * @returns Safe array reference (network.nodes or empty array).
 */
const normalizeNodesArray = (network: INetwork): NetworkNode[] => {
  return Array.isArray(network?.nodes) ? network.nodes : [];
};

/**
 * Classify a node array into input / hidden / output buckets using pooled arrays.
 *
 * Description:
 * Performs a single in-place pass over `nodesArray` and places references into
 * three pooled buckets attached to the engine state to avoid per-call allocations.
 * The returned buckets are backed by pooled arrays and are reused by subsequent
 * callers; do not mutate them if you intend to reuse the engine pools.
 *
 * Implementation details:
 * - The pooled buckets live on the shared engine state and are lazily initialised.
 * - Buckets are cleared by setting `.length = 0` which preserves allocated capacity.
 * - The method is allocation-light and suitable for hot paths.
 *
 * @param engineState - Shared engine state containing pooled scratch buffers.
 * @param nodesArray - Array-like collection of node objects (each node may have a `type` property).
 * @returns An object with the following properties:
 *  - `{ nodeList }` - The original (normalized) node array reference used for classification.
 *  - `{ inputNodes }` - Pooled array containing all nodes whose `type` is `'input'`.
 *  - `{ hiddenNodes }` - Pooled array containing all non-input/non-output nodes.
 *  - `{ outputNodes }` - Pooled array containing all nodes whose `type` is `'output'`.
 *
 * @example
 * const { nodeList, inputNodes, hiddenNodes, outputNodes } = classifyNodesFromArray(state, net.nodes || []);
 * console.log(`inputs=${inputNodes.length} hidden=${hiddenNodes.length} outputs=${outputNodes.length}`);
 */
const classifyNodesFromArray = (
  engineState: EngineState,
  nodesArray: NetworkNode[],
): ClassifiedNodeBuckets => {
  // Step 1: Normalise the incoming node list to a safe, non-null array reference.
  const nodeList: NetworkNode[] = Array.isArray(nodesArray) ? nodesArray : [];

  // Step 2: Lazily create / reuse the pooled buckets structure on the state.
  const scratchBundle = engineState.scratch;
  let pooledBuckets = scratchBundle.nodeBuckets as NodeBucketPool | undefined;
  if (!Array.isArray(pooledBuckets?.[0])) {
    pooledBuckets = scratchBundle.nodeBuckets = [[], [], []] as NodeBucketPool;
  }

  // Descriptive bucket aliases for readability.
  const inputBucket = pooledBuckets[0];
  const hiddenBucket = pooledBuckets[1];
  const outputBucket = pooledBuckets[2];

  // Step 3: Clear buckets in-place (cheap; preserves allocated capacity where possible).
  inputBucket.length = 0;
  hiddenBucket.length = 0;
  outputBucket.length = 0;

  // Step 4: Single-pass classification. Keep the loop small and optimiser-friendly.
  for (let nodeIndex = 0; nodeIndex < nodeList.length; nodeIndex++) {
    const node = nodeList[nodeIndex];
    // Tolerate holes and malformed entries quickly.
    if (!node) continue;

    // Normalise the node type to a stable string and classify deterministically.
    const nodeType = String(node.type ?? 'hidden');
    if (nodeType === 'input') {
      inputBucket.push(node);
    } else if (nodeType === 'output') {
      outputBucket.push(node);
    } else {
      // Treat everything else as hidden (includes undefined/custom types).
      hiddenBucket.push(node);
    }
  }

  // Step 5: Return references (note: returned arrays are pooled and reused by subsequent callers).
  return {
    nodeList,
    inputNodes: inputBucket,
    hiddenNodes: hiddenBucket,
    outputNodes: outputBucket,
  };
};

/**
 * Classify nodes into input / hidden / output buckets.
 *
 * Behavior & contract:
 *  - Allocation-light: returns references into the original node array (no cloning).
 *  - Tolerates missing network or sparse node arrays (holes preserved by skipping).
 *  - Reuses a small pooled buckets structure across calls to reduce per-call allocations.
 *
 * @param engineState - Shared engine state containing pooled scratch buffers.
 * @param network - Network-like object with an optional `nodes` array.
 * @returns An object { nodeList, inputNodes, hiddenNodes, outputNodes } where each
 *          bucket is a (pooled) array referencing nodes from the original `nodes`.
 *
 * @example
 * const { nodeList, inputNodes, hiddenNodes, outputNodes } = classifyNodes(state, someNet);
 * console.log(`inputs=${inputNodes.length} hidden=${hiddenNodes.length} outputs=${outputNodes.length}`);
 */
const classifyNodes = (
  engineState: EngineState,
  network: INetwork,
): ClassifiedNodeBuckets => {
  // Orchestrator: normalize inputs then delegate to the fast, allocation-light classifier.
  const normalizedNodeList = normalizeNodesArray(network);
  return classifyNodesFromArray(engineState, normalizedNodeList);
};

/**
 * Populate and return a pooled array of activation (squash) function names for `network.nodes`.
 *
 * Behaviour & contract:
 * - Reuses a private pooled string array to avoid per-call allocations.
 * - Grows the pool capacity using a power-of-two strategy (next power-of-two) to reduce resize frequency.
 * - Fills the pooled array with readable names and trims `.length` to the exact node count before returning.
 *
 * Steps (high level):
 * 1) Normalise the incoming node list to a safe array reference.
 * 2) Lazily create the shared pooled names array when first used.
 * 3) Grow the pooled array to a power-of-two capacity when current capacity is insufficient.
 * 4) Populate the used prefix with function `name` when available or a stable string fallback.
 * 5) Trim the pooled array to `nodesCount` and return it (note: the returned array is reused; callers must not mutate it).
 *
 * @param engineState - Shared engine state containing pooled scratch buffers.
 * @param network - Network-like object with an optional `nodes` array.
 * @returns Pooled array of activation function names (length === number of nodes).
 *
 * @example
 * const names = gatherActivationNames(state, someNet);
 * console.log(names.join(','));
 */
const gatherActivationNames = (
  engineState: EngineState,
  network: INetwork,
): string[] => {
  // Step 1: Safe normalisation of the node list reference.
  const nodesArray = normalizeNodesArray(network);
  const nodesCount = nodesArray.length;

  // Step 2: Lazily ensure the shared pool exists.
  const scratchBundle = engineState.scratch;
  if (!Array.isArray(scratchBundle.activationNameBuffer)) {
    scratchBundle.activationNameBuffer = [];
  }

  const pooledNames: string[] = scratchBundle.activationNameBuffer;

  // Helper: compute next power-of-two for growth (keeps growth jumps friendly to the allocator).
  const nextPowerOfTwo = (value: number): number => {
    let power = 1;
    while (power < value) power <<= 1;
    return power;
  };

  // Step 3: Grow pooled capacity to the next power-of-two when necessary.
  if (pooledNames.length < nodesCount) {
    const targetCapacity = nextPowerOfTwo(Math.max(1, nodesCount));
    pooledNames.length = targetCapacity;
  }

  // Step 4: Populate the used prefix with readable names.
  for (
    let activationIndex = 0;
    activationIndex < nodesCount;
    activationIndex++
  ) {
    const nodeEntry = nodesArray[activationIndex];
    const squashCandidate = nodeEntry?.squash;

    // Prefer explicit function name when available; fall back to a stable string.
    if (typeof squashCandidate === 'function') {
      // Some anonymous functions may have an empty .name; normalise to 'anonymous' then.
      pooledNames[activationIndex] =
        squashCandidate.name && squashCandidate.name.length
          ? squashCandidate.name
          : 'anonymous';
    } else {
      pooledNames[activationIndex] = String(squashCandidate ?? 'unknown');
    }
  }

  // Step 5: Trim to exact length for consumer readability (non-allocating when shrinking a pre-sized array).
  pooledNames.length = nodesCount;
  return pooledNames;
};

/**
 * Fast, allocation-aware detector for recurrent or gated connections.
 *
 * Steps:
 * 1) Fast-guard invalid inputs (non-array / empty -> false).
 * 2) For small connection lists use a plain loop (lowest overhead).
 * 3) For large lists rent a pooled Int8Array via `ensureConnFlagsCapacity` and
 *    reuse it as a tiny scratch bitmap to reduce allocations and improve cache locality.
 * 4) Early-return on first detection (gated or recurrent), otherwise mark seen indices
 *    in the scratch buffer and return false when complete.
 * 5) Defensive fallback: if the pooled allocation fails, revert to the plain loop.
 *
 * @param engineState - Shared engine state containing pooled scratch buffers.
 * @param connectionsList - Array of connection-like objects with optional `from`, `to`, `gater`.
 * @returns true when any connection is recurrent (from === to) or gated (gater truthy), false otherwise.
 *
 * @example
 * const hasSpecial = detectRecurrentOrGated(state, network.connections);
 */
const detectRecurrentOrGated = (
  engineState: EngineState,
  connectionsList: NetworkConnection[],
): boolean => {
  // Step 1: Validate input quickly
  if (!Array.isArray(connectionsList) || connectionsList.length === 0)
    return false;

  const connectionCount = connectionsList.length;
  const SMALL_LIST_THRESHOLD = 128; // tuned threshold for typed-array trade-off

  // Step 2: Small-list fast path: direct inspection avoids typed-array overhead
  if (connectionCount < SMALL_LIST_THRESHOLD) {
    for (
      let connectionIndex = 0;
      connectionIndex < connectionCount;
      connectionIndex++
    ) {
      const connection = connectionsList[connectionIndex];
      if (!connection) continue; // tolerate sparse arrays
      if (connection.gater) return true; // gated connection detected
      if (connection.from === connection.to) return true; // recurrent self-connection
    }
    return false;
  }

  // Step 3: Large-list path: attempt to rent a pooled Int8Array for scratch flags
  try {
    const scratchFlags = ensureConnFlagsCapacity(engineState, connectionCount);

    // Step 5: Fallback to plain loop when pool allocation fails
    if (!scratchFlags) {
      for (
        let connectionIndex = 0;
        connectionIndex < connectionCount;
        connectionIndex++
      ) {
        const connection = connectionsList[connectionIndex];
        if (!connection) continue;
        if (connection.gater || connection.from === connection.to) return true;
      }
      return false;
    }

    // Initialize only the used prefix for deterministic behavior (cheap for Int8Array)
    scratchFlags.fill(0, 0, connectionCount);

    // Step 4: Iterate, early-return on detection, and mark seen indices in scratch buffer
    for (
      let connectionIndex = 0;
      connectionIndex < connectionCount;
      connectionIndex++
    ) {
      const connection = connectionsList[connectionIndex];
      if (!connection) continue;
      if (connection.gater) return true;
      if (connection.from === connection.to) return true;
      scratchFlags[connectionIndex] = 1; // mark index as visited in the pooled scratch
    }

    return false;
  } catch {
    // Robust degradation: on any runtime error, use the safe plain loop.
    for (
      let connectionIndex = 0;
      connectionIndex < connectionCount;
      connectionIndex++
    ) {
      const connection = connectionsList[connectionIndex];
      if (!connection) continue;
      if (connection.gater || connection.from === connection.to) return true;
    }
    return false;
  }
};

/**
 * Print a structured summary of network topology to the console.
 *
 * This is a developer-facing inspection utility that logs:
 * - Node counts by type (input, hidden, output)
 * - Activation function names used across the network
 * - Total connection count
 * - Whether the network contains recurrent or gated connections
 *
 * Design:
 * - Best-effort: swallows errors and logs partial data when inspection fails
 * - Allocation-light: reuses pooled scratch buffers for node classification and activation names
 * - Delegates to helper functions for modular, testable logic
 *
 * @param engineState - Shared engine state containing pooled scratch buffers.
 * @param network - Network-like object to inspect.
 *
 * @example
 * printNetworkStructure(state, bestNetwork);
 */
export const printNetworkStructure = (
  engineState: EngineState,
  network: INetwork,
): void => {
  // Orchestrator: gather lightweight facts and delegate formatting to helpers.
  try {
    console.log('Network Structure:');

    // Nodes classification
    const { nodeList, inputNodes, hiddenNodes, outputNodes } = classifyNodes(
      engineState,
      network,
    );
    console.log('Nodes:', nodeList.length);
    console.log('  Input nodes:', inputNodes.length);
    console.log('  Hidden nodes:', hiddenNodes.length);
    console.log('  Output nodes:', outputNodes.length);

    // Activation function names (reuses pooled activation name buffer)
    const activationNames = gatherActivationNames(engineState, network);
    console.log('Activation functions:', activationNames);

    // Connections summary and recurrent/gated detection
    const connectionsList = Array.isArray(network?.connections)
      ? network.connections
      : [];
    console.log('Connections:', connectionsList.length);
    const hasRecurrentOrGated = detectRecurrentOrGated(
      engineState,
      connectionsList,
    );
    console.log('Has recurrent/gated connections:', hasRecurrentOrGated);
  } catch (inspectError: unknown) {
    swallowError(inspectError);
    // Best-effort logging: swallow and surface a minimal message.
    // Avoid throwing from a debug helper.

    console.log(
      'printNetworkStructure: failed to inspect network (partial data)',
    );
  }
};
