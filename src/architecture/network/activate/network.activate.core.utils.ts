import type Network from '../../network/network';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type { ActivationArray } from '../../activationArrayPool/activationArrayPool';
import { config } from '../../../config';
import {
  INITIAL_OUTPUT_WRITE_INDEX,
  INPUT_NODE_TYPE,
  OUTPUT_NODE_TYPE,
  UNDEFINED_INPUT_LENGTH_TEXT,
} from './network.activate.utils.types';
import type {
  ActivateRuntimeNetworkProps,
  ActivationStats,
  NetworkLayer,
  NetworkLayerNodes,
  WeightNoiseApplyResult,
  WeightNoiseStats,
} from './network.activate.utils.types';
import type {
  ConnectionWeightNoiseProps,
  NetworkRuntimeProps,
} from '../network.types';
import {
  NetworkActivateCorruptedStructureError,
  NetworkActivateInputSizeMismatchError,
} from './network.activate.errors';
import {
  resolveActivationTraversalNodes,
  resolveInputValuesByNodeId,
  resolveOrderedOutputNodes,
} from './network.activate.schedule.utils';

/**
 * Produce a normally distributed random sample using the Box-Muller transform.
 *
 * @param rng Pseudo-random source in the interval [0, 1).
 * @returns Standard normal sample with mean 0 and variance 1.
 */
export function gaussianRand(rng: () => number = Math.random): number {
  let randomU = 0;
  let randomV = 0;

  while (randomU === 0) randomU = rng();
  while (randomV === 0) randomV = rng();

  return (
    Math.sqrt(-2.0 * Math.log(randomU)) * Math.cos(2.0 * Math.PI * randomV)
  );
}

/**
 * Execute the main activation routine and return plain numeric outputs.
 *
 * @param this Bound network instance.
 * @param input Input values with length matching network input count.
 * @param training Whether training-time stochastic behavior is enabled.
 * @returns Output activation values.
 */
export function activate(
  this: Network,
  input: number[],
  training = false,
): number[] {
  const runtimeNetwork = this as unknown as ActivateRuntimeNetworkProps;

  prepareTopologyForActivation(runtimeNetwork);
  validateInputVector(this, input);

  const fastSlabOutput = tryFastSlabActivation(runtimeNetwork, input, training);
  if (fastSlabOutput) return fastSlabOutput;

  const output = acquireOutputBuffer(this.output);
  validateNetworkNodes(this);

  resetSkippedLayers(this);
  const stats = createActivationStats(this.connections.length);

  const weightNoiseState = applyTrainingWeightNoise(
    this,
    runtimeNetwork,
    training,
    stats,
  );

  updateStochasticDepthFromSchedule(runtimeNetwork, training);

  executeActivationPath(this, runtimeNetwork, input, training, output, stats);

  finalizeNodePathWeightNoiseRestore(
    this,
    training,
    weightNoiseState.appliedWeightNoise,
  );

  finalizeTrainingStepAndStats(runtimeNetwork, stats, training);

  return releaseBufferAndCreateResult(output);
}

/**
 * Ensure compiled activation scheduling is refreshed before activation when topology changed.
 *
 * @param runtimeNetwork Runtime activation internals.
 * @returns Nothing.
 */
function prepareTopologyForActivation(
  runtimeNetwork: ActivateRuntimeNetworkProps,
): void {
  if (runtimeNetwork._topoDirty) {
    runtimeNetwork._computeTopoOrder();
  }
}

/**
 * Validate that the incoming input vector exists and matches expected input size.
 *
 * @param network Network being activated.
 * @param inputVector Input vector to validate.
 * @returns Nothing.
 */
function validateInputVector(network: Network, inputVector: number[]): void {
  if (!Array.isArray(inputVector) || inputVector.length !== network.input) {
    throw new NetworkActivateInputSizeMismatchError(
      `Input size mismatch: expected ${network.input}, got ${
        inputVector ? inputVector.length : UNDEFINED_INPUT_LENGTH_TEXT
      }`,
    );
  }
}

/**
 * Attempt fast slab activation and safely fall back to regular activation on failure.
 *
 * @param runtimeNetwork Runtime activation internals.
 * @param inputVector Input vector.
 * @param isTraining Training-time flag.
 * @returns Fast slab output when available, otherwise undefined.
 */
function tryFastSlabActivation(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
): number[] | undefined {
  if (!runtimeNetwork._canUseFastSlab(isTraining)) {
    return undefined;
  }

  try {
    return runtimeNetwork._fastSlabActivate(inputVector);
  } catch {
    return undefined;
  }
}

/**
 * Acquire a pooled activation output buffer for the current output width.
 *
 * @param outputSize Number of output slots.
 * @returns Mutable pooled output buffer.
 */
function acquireOutputBuffer(outputSize: number): ActivationArray {
  return activationArrayPool.acquire(outputSize) as ActivationArray;
}

/**
 * Assert that the network contains nodes before executing activation routines.
 *
 * @param network Network being activated.
 * @returns Nothing.
 */
function validateNetworkNodes(network: Network): void {
  if (!network.nodes || network.nodes.length === 0) {
    throw new NetworkActivateCorruptedStructureError(
      'Network structure is corrupted or empty. No nodes found.',
    );
  }
}

/**
 * Clear the runtime list of skipped layers before current activation pass.
 *
 * @param network Network runtime owner.
 * @returns Nothing.
 */
function resetSkippedLayers(network: Network): void {
  (network as unknown as NetworkRuntimeProps)._lastSkippedLayers = [];
}

/**
 * Create activation statistics container for the current pass.
 *
 * @param totalConnections Number of network connections.
 * @returns Initialized activation stats object.
 */
function createActivationStats(totalConnections: number): ActivationStats {
  return {
    droppedHiddenNodes: 0,
    totalHiddenNodes: 0,
    droppedConnections: 0,
    totalConnections,
    skippedLayers: [],
    weightNoise: createWeightNoiseStats(),
  };
}

/**
 * Create the weight-noise statistics record with zeroed aggregates.
 *
 * @returns Zero-initialized weight-noise stats.
 */
function createWeightNoiseStats(): WeightNoiseStats {
  return {
    count: 0,
    sumAbs: 0,
    maxAbs: 0,
    meanAbs: 0,
  };
}

/**
 * Apply per-connection training noise for the main activation flow.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param isTraining Training-time flag.
 * @param stats Activation stats accumulator.
 * @returns Applied-state information for downstream restore logic.
 */
function applyTrainingWeightNoise(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
  stats: ActivationStats,
): WeightNoiseApplyResult {
  if (!isTraining) {
    return { appliedWeightNoise: false };
  }

  const dynamicStandardDeviation = resolveDynamicWeightNoiseStd(runtimeNetwork);
  const hasNoiseConfig =
    dynamicStandardDeviation > 0 ||
    runtimeNetwork._weightNoisePerHidden.length > 0;

  if (!hasNoiseConfig) {
    return { appliedWeightNoise: false };
  }

  let appliedWeightNoise = false;

  for (const connection of network.connections) {
    if (hasOriginalWeightNoise(connection)) {
      continue;
    }

    persistOriginalWeightNoise(connection);

    const standardDeviation = resolveConnectionNoiseStd(
      network,
      runtimeNetwork,
      connection,
      dynamicStandardDeviation,
    );

    if (standardDeviation > 0) {
      const sampledNoise =
        standardDeviation * gaussianRand(runtimeNetwork._rand);
      connection.weight += sampledNoise;
      setLastSampledNoise(connection, sampledNoise);
      recordWeightNoiseSample(stats, sampledNoise);
      appliedWeightNoise = true;
      continue;
    }

    setLastSampledNoise(connection, 0);
  }

  return { appliedWeightNoise };
}

/**
 * Record one sampled weight-noise value in the activation statistics snapshot.
 *
 * @param stats Activation stats accumulator.
 * @param sampledNoise Sampled noise value before restoration.
 * @returns Nothing.
 */
function recordWeightNoiseSample(
  stats: ActivationStats,
  sampledNoise: number,
): void {
  const absoluteNoise = Math.abs(sampledNoise);

  stats.weightNoise.count++;
  stats.weightNoise.sumAbs += absoluteNoise;
  stats.weightNoise.maxAbs = Math.max(stats.weightNoise.maxAbs, absoluteNoise);
}

/**
 * Resolve the training-step adjusted global weight-noise standard deviation.
 *
 * @param runtimeNetwork Runtime activation internals.
 * @returns Effective weight-noise standard deviation for current training step.
 */
function resolveDynamicWeightNoiseStd(
  runtimeNetwork: ActivateRuntimeNetworkProps,
): number {
  if (!runtimeNetwork._weightNoiseSchedule) {
    return runtimeNetwork._weightNoiseStd;
  }

  return runtimeNetwork._weightNoiseSchedule(runtimeNetwork._trainingStep);
}

/**
 * Resolve connection-specific weight-noise standard deviation, including per-hidden overrides.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param connection Current connection.
 * @param fallbackStandardDeviation Global fallback deviation.
 * @returns Effective standard deviation for this connection.
 */
function resolveConnectionNoiseStd(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  connection: Network['connections'][number],
  fallbackStandardDeviation: number,
): number {
  if (!network.layers || runtimeNetwork._weightNoisePerHidden.length === 0) {
    return fallbackStandardDeviation;
  }

  const sourceLayerIndex = findSourceLayerIndex(network, connection);

  if (sourceLayerIndex <= 0 || sourceLayerIndex >= network.layers.length) {
    return fallbackStandardDeviation;
  }

  const hiddenLayerIndex = sourceLayerIndex - 1;

  if (hiddenLayerIndex >= runtimeNetwork._weightNoisePerHidden.length) {
    return fallbackStandardDeviation;
  }

  return runtimeNetwork._weightNoisePerHidden[hiddenLayerIndex];
}

/**
 * Find the layer index containing a connection source node.
 *
 * @param network Network being activated.
 * @param connection Connection to inspect.
 * @returns Layer index for source node, or -1 when not found.
 */
function findSourceLayerIndex(
  network: Network,
  connection: Network['connections'][number],
): number {
  if (!network.layers) {
    return -1;
  }

  for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
    if (network.layers[layerIndex].nodes.includes(connection.from)) {
      return layerIndex;
    }
  }

  return -1;
}

/**
 * Update stochastic depth probabilities using a training schedule when valid.
 *
 * @param runtimeNetwork Runtime activation internals.
 * @param isTraining Training-time flag.
 * @returns Nothing.
 */
function updateStochasticDepthFromSchedule(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
): void {
  if (!isTraining) {
    return;
  }

  if (!runtimeNetwork._stochasticDepthSchedule) {
    return;
  }

  if (runtimeNetwork._stochasticDepth.length === 0) {
    return;
  }

  const updatedProbabilities = runtimeNetwork._stochasticDepthSchedule(
    runtimeNetwork._trainingStep,
    runtimeNetwork._stochasticDepth.slice(),
  );

  if (!Array.isArray(updatedProbabilities)) {
    return;
  }

  if (updatedProbabilities.length !== runtimeNetwork._stochasticDepth.length) {
    return;
  }

  if (containsInvalidProbability(updatedProbabilities)) {
    return;
  }

  runtimeNetwork._stochasticDepth = updatedProbabilities.slice();
}

/**
 * Check whether a probability vector contains values outside the (0, 1] interval.
 *
 * @param probabilities Candidate probability vector.
 * @returns True when one or more probabilities are invalid.
 */
function containsInvalidProbability(probabilities: number[]): boolean {
  for (const probability of probabilities) {
    if (probability <= 0 || probability > 1) {
      return true;
    }
  }

  return false;
}

/**
 * Execute one of the three activation branches: stochastic layers, standard layers, or raw nodes.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param inputVector Input vector.
 * @param isTraining Training-time flag.
 * @param outputBuffer Mutable output buffer.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function executeActivationPath(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void {
  if (hasLayeredNetworkWithStochasticDepth(network, runtimeNetwork)) {
    activateLayeredNetworkWithStochasticDepth(
      network,
      runtimeNetwork,
      inputVector,
      isTraining,
      outputBuffer,
      stats,
    );
    return;
  }

  if (hasLayeredNetwork(network)) {
    activateLayeredNetworkWithDropout(
      network,
      runtimeNetwork,
      inputVector,
      isTraining,
      outputBuffer,
      stats,
    );
    return;
  }

  activateNodeNetworkFallback(
    network,
    runtimeNetwork,
    inputVector,
    isTraining,
    outputBuffer,
    stats,
  );
}

/**
 * Check whether the network has layers and stochastic-depth configuration for layer skipping path.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @returns True when stochastic-depth layer path should run.
 */
function hasLayeredNetworkWithStochasticDepth(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
): boolean {
  return (
    !!network.layers &&
    network.layers.length > 0 &&
    runtimeNetwork._stochasticDepth.length > 0
  );
}

/**
 * Check whether the network has at least one explicit layer.
 *
 * @param network Network being activated.
 * @returns True when layered activation path should run.
 */
function hasLayeredNetwork(network: Network): boolean {
  return !!network.layers && network.layers.length > 0;
}

/**
 * Run layered activation with stochastic-depth skipping and inverse-survival scaling.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param inputVector Input vector.
 * @param isTraining Training-time flag.
 * @param outputBuffer Mutable output buffer.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function activateLayeredNetworkWithStochasticDepth(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void {
  if (!network.layers) {
    return;
  }

  let lastLayerActivations: number[] | undefined;

  for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
    const currentLayer = network.layers[layerIndex];

    const skipDecision = decideLayerSkip(
      network,
      runtimeNetwork,
      currentLayer.nodes.length,
      layerIndex,
      isTraining,
      lastLayerActivations,
    );

    if (skipDecision.shouldSkipLayer) {
      recordSkippedLayer(network, stats, layerIndex);
      continue;
    }

    const rawActivations = activateLayer(
      currentLayer,
      layerIndex,
      inputVector,
      isTraining,
    );

    if (skipDecision.surviveProbability < 1) {
      lastLayerActivations = scaleActivations(
        rawActivations,
        1 / skipDecision.surviveProbability,
      );
      continue;
    }

    lastLayerActivations = rawActivations;
  }

  writeLayerActivationsToOutput(
    lastLayerActivations,
    outputBuffer,
    network.output,
  );
}

/**
 * Decide whether a hidden layer should be skipped in stochastic-depth mode.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param currentLayerNodeCount Number of nodes in current layer.
 * @param layerIndex Current layer index.
 * @param isTraining Training-time flag.
 * @param previousLayerActivations Last computed layer activations.
 * @returns Skip decision and survival probability for the layer.
 */
function decideLayerSkip(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  currentLayerNodeCount: number,
  layerIndex: number,
  isTraining: boolean,
  previousLayerActivations: number[] | undefined,
): { shouldSkipLayer: boolean; surviveProbability: number } {
  if (!network.layers) {
    return { shouldSkipLayer: false, surviveProbability: 1 };
  }

  if (!isTraining) {
    return { shouldSkipLayer: false, surviveProbability: 1 };
  }

  if (!isHiddenLayer(layerIndex, network.layers.length)) {
    return { shouldSkipLayer: false, surviveProbability: 1 };
  }

  const hiddenLayerIndex = layerIndex - 1;

  if (hiddenLayerIndex >= runtimeNetwork._stochasticDepth.length) {
    return { shouldSkipLayer: false, surviveProbability: 1 };
  }

  const surviveProbability = runtimeNetwork._stochasticDepth[hiddenLayerIndex];
  let shouldSkipLayer = runtimeNetwork._rand() >= surviveProbability;

  if (
    shouldSkipLayer &&
    !hasCompatibleSkipState(previousLayerActivations, currentLayerNodeCount)
  ) {
    shouldSkipLayer = false;
  }

  return { shouldSkipLayer, surviveProbability };
}

/**
 * Check whether a layer index refers to a hidden layer in a layered network.
 *
 * @param layerIndex Current layer index.
 * @param totalLayerCount Number of layers in the network.
 * @returns True when the layer is hidden.
 */
function isHiddenLayer(layerIndex: number, totalLayerCount: number): boolean {
  return layerIndex > 0 && layerIndex < totalLayerCount - 1;
}

/**
 * Validate whether previous activations can be reused as skip pass-through output.
 *
 * @param previousLayerActivations Last computed layer activations.
 * @param currentLayerNodeCount Current layer node count.
 * @returns True when pass-through activations are compatible.
 */
function hasCompatibleSkipState(
  previousLayerActivations: number[] | undefined,
  currentLayerNodeCount: number,
): boolean {
  if (!previousLayerActivations) {
    return false;
  }

  return previousLayerActivations.length === currentLayerNodeCount;
}

/**
 * Activate one layer, routing input only for the first layer.
 *
 * @param currentLayer Layer instance to activate.
 * @param layerIndex Layer index.
 * @param inputVector Network input vector.
 * @param isTraining Training-time flag.
 * @returns Layer activations.
 */
function activateLayer(
  currentLayer: NetworkLayer,
  layerIndex: number,
  inputVector: number[],
  isTraining: boolean,
): number[] {
  if (layerIndex === 0) {
    return currentLayer.activate(inputVector, isTraining);
  }

  return currentLayer.activate(undefined, isTraining);
}

/**
 * Record a skipped layer in runtime and stats trackers.
 *
 * @param network Network being activated.
 * @param stats Activation stats accumulator.
 * @param layerIndex Skipped layer index.
 * @returns Nothing.
 */
function recordSkippedLayer(
  network: Network,
  stats: ActivationStats,
  layerIndex: number,
): void {
  const runtimeProps = network as unknown as NetworkRuntimeProps;
  runtimeProps._lastSkippedLayers!.push(layerIndex);
  stats.skippedLayers.push(layerIndex);
}

/**
 * Create a new activation vector by multiplying each activation by a scale factor.
 *
 * @param activations Source activation vector.
 * @param scaleFactor Multiplicative scale factor.
 * @returns Scaled activation vector.
 */
function scaleActivations(
  activations: number[],
  scaleFactor: number,
): number[] {
  const scaledActivations: number[] = new Array(activations.length);

  for (
    let activationIndex = 0;
    activationIndex < activations.length;
    activationIndex++
  ) {
    scaledActivations[activationIndex] =
      activations[activationIndex] * scaleFactor;
  }

  return scaledActivations;
}

/**
 * Copy final layer activations into the pooled network output buffer.
 *
 * @param layerActivations Final layer activations.
 * @param outputBuffer Mutable output buffer.
 * @param outputSize Maximum output width.
 * @returns Nothing.
 */
function writeLayerActivationsToOutput(
  layerActivations: number[] | undefined,
  outputBuffer: ActivationArray,
  outputSize: number,
): void {
  if (!layerActivations) {
    return;
  }

  for (
    let outputIndex = 0;
    outputIndex < layerActivations.length && outputIndex < outputSize;
    outputIndex++
  ) {
    outputBuffer[outputIndex] = layerActivations[outputIndex];
  }
}

/**
 * Run layered activation with dropout masks and no stochastic-depth skips.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param inputVector Input vector.
 * @param isTraining Training-time flag.
 * @param outputBuffer Mutable output buffer.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function activateLayeredNetworkWithDropout(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void {
  if (!network.layers) {
    return;
  }

  let lastLayerActivations: number[] | undefined;

  for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
    const currentLayer = network.layers[layerIndex];

    const rawActivations = activateLayer(
      currentLayer,
      layerIndex,
      inputVector,
      false,
    );

    if (isHiddenLayer(layerIndex, network.layers.length)) {
      applyHiddenLayerDropout(
        currentLayer,
        rawActivations,
        runtimeNetwork,
        network.dropout,
        isTraining,
        stats,
      );
    }

    lastLayerActivations = rawActivations;
  }

  writeLayerActivationsToOutput(
    lastLayerActivations,
    outputBuffer,
    network.output,
  );
}

/**
 * Apply dropout masks to hidden layer nodes and enforce at least one active node.
 *
 * @param layer Hidden layer instance.
 * @param rawActivations Raw layer activations.
 * @param runtimeNetwork Runtime activation internals.
 * @param dropoutProbability Layer dropout probability.
 * @param isTraining Training-time flag.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function applyHiddenLayerDropout(
  layer: NetworkLayer,
  rawActivations: number[],
  runtimeNetwork: ActivateRuntimeNetworkProps,
  dropoutProbability: number,
  isTraining: boolean,
  stats: ActivationStats,
): void {
  if (!isTraining || dropoutProbability <= 0) {
    setAllMasksToOne(layer.nodes);
    return;
  }

  let droppedNodeCount = 0;

  for (let nodeIndex = 0; nodeIndex < layer.nodes.length; nodeIndex++) {
    const hiddenNode = layer.nodes[nodeIndex];

    hiddenNode.mask = runtimeNetwork._rand() < dropoutProbability ? 0 : 1;
    stats.totalHiddenNodes++;

    if (hiddenNode.mask !== 0) {
      continue;
    }

    stats.droppedHiddenNodes++;
    hiddenNode.activation = 0;
    droppedNodeCount++;
  }

  if (droppedNodeCount !== layer.nodes.length || layer.nodes.length === 0) {
    return;
  }

  const randomNodeIndex = Math.floor(
    runtimeNetwork._rand() * layer.nodes.length,
  );
  layer.nodes[randomNodeIndex].mask = 1;
  layer.nodes[randomNodeIndex].activation = rawActivations[randomNodeIndex];
}

/**
 * Set mask value to one for every node in a layer.
 *
 * @param nodes Layer nodes to normalize.
 * @returns Nothing.
 */
function setAllMasksToOne(nodes: NetworkLayerNodes): void {
  for (const hiddenNode of nodes) {
    hiddenNode.mask = 1;
  }
}

/**
 * Run schedule-aware node activation for networks without explicit layer definitions.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param inputVector Input vector.
 * @param isTraining Training-time flag.
 * @param outputBuffer Mutable output buffer.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function activateNodeNetworkFallback(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void {
  const hiddenNodes = collectHiddenNodes(network.nodes);
  const activationNodes = resolveActivationTraversalNodes(network);
  const inputValuesByNodeId = resolveInputValuesByNodeId(network, inputVector);
  const orderedOutputNodes = resolveOrderedOutputNodes(network);

  applyFallbackHiddenDropout(
    hiddenNodes,
    runtimeNetwork,
    network.dropout,
    isTraining,
    stats,
  );

  applyFallbackWeightNoise(network, runtimeNetwork, isTraining, stats);
  activateNodesAndCollectOutputs(
    activationNodes,
    inputValuesByNodeId,
    orderedOutputNodes,
    outputBuffer,
  );

  applyDropConnect(network, runtimeNetwork, isTraining, stats);
}

/**
 * Collect hidden nodes from a raw node list.
 *
 * @param nodes Network node collection.
 * @returns Hidden-only node list.
 */
function collectHiddenNodes(nodes: Network['nodes']): Network['nodes'] {
  const hiddenNodes: Network['nodes'] = [];

  for (const node of nodes) {
    if (node.type === 'hidden') {
      hiddenNodes.push(node);
    }
  }

  return hiddenNodes;
}

/**
 * Apply fallback dropout for hidden nodes in raw node traversal mode.
 *
 * @param hiddenNodes Hidden nodes.
 * @param runtimeNetwork Runtime activation internals.
 * @param dropoutProbability Dropout probability.
 * @param isTraining Training-time flag.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function applyFallbackHiddenDropout(
  hiddenNodes: Network['nodes'],
  runtimeNetwork: ActivateRuntimeNetworkProps,
  dropoutProbability: number,
  isTraining: boolean,
  stats: ActivationStats,
): void {
  if (!isTraining || dropoutProbability <= 0) {
    setAllMasksToOne(hiddenNodes);
    return;
  }

  let droppedHiddenCount = 0;

  for (const hiddenNode of hiddenNodes) {
    hiddenNode.mask = runtimeNetwork._rand() < dropoutProbability ? 0 : 1;
    stats.totalHiddenNodes++;

    if (hiddenNode.mask !== 0) {
      continue;
    }

    droppedHiddenCount++;
    stats.droppedHiddenNodes++;
  }

  if (droppedHiddenCount !== hiddenNodes.length || hiddenNodes.length === 0) {
    return;
  }

  const randomNodeIndex = Math.floor(
    runtimeNetwork._rand() * hiddenNodes.length,
  );
  hiddenNodes[randomNodeIndex].mask = 1;
}

/**
 * Apply raw fallback weight noise to all connections using global standard deviation.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param isTraining Training-time flag.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function applyFallbackWeightNoise(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
  stats: ActivationStats,
): void {
  if (!isTraining || runtimeNetwork._weightNoiseStd <= 0) {
    return;
  }

  if (!runtimeNetwork._wnOrig) {
    runtimeNetwork._wnOrig = new Array(network.connections.length);
  }

  for (
    let connectionIndex = 0;
    connectionIndex < network.connections.length;
    connectionIndex++
  ) {
    const connection = network.connections[connectionIndex];

    if (hasOriginalWeightNoise(connection)) {
      continue;
    }

    persistOriginalWeightNoise(connection);

    const sampledNoise =
      runtimeNetwork._weightNoiseStd * gaussianRand(runtimeNetwork._rand);
    connection.weight += sampledNoise;
    recordWeightNoiseSample(stats, sampledNoise);
  }
}

/**
 * Activate raw nodes in the resolved execution order and collect outputs by explicit role order.
 *
 * @param activationNodes Network nodes in activation order.
 * @param inputValuesByNodeId Stable lookup for explicit input-role injection.
 * @param orderedOutputNodes Output nodes in public vector order.
 * @param outputBuffer Mutable output buffer.
 * @returns Nothing.
 */
function activateNodesAndCollectOutputs(
  activationNodes: Network['nodes'],
  inputValuesByNodeId: Map<number, number>,
  orderedOutputNodes: Network['nodes'],
  outputBuffer: ActivationArray,
): void {
  for (const node of activationNodes) {

    if (node.type === INPUT_NODE_TYPE) {
      node.activate(inputValuesByNodeId.get(node.geneId));
      continue;
    }

    if (node.type === OUTPUT_NODE_TYPE) {
      node.activate();
      continue;
    }

    node.activate();
  }

  for (
    let outputIndex = INITIAL_OUTPUT_WRITE_INDEX;
    outputIndex < orderedOutputNodes.length;
    outputIndex++
  ) {
    outputBuffer[outputIndex] = orderedOutputNodes[outputIndex].activation;
  }
}

/**
 * Apply drop-connect masking and restore original weights where required.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param isTraining Training-time flag.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function applyDropConnect(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
  stats: ActivationStats,
): void {
  if (isTraining && runtimeNetwork._dropConnectProb > 0) {
    applyTrainingDropConnect(network, runtimeNetwork, stats);
    return;
  }

  restoreDropConnectWeights(network);
}

/**
 * Apply training-time drop-connect masks to each connection.
 *
 * @param network Network being activated.
 * @param runtimeNetwork Runtime activation internals.
 * @param stats Activation stats accumulator.
 * @returns Nothing.
 */
function applyTrainingDropConnect(
  network: Network,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  stats: ActivationStats,
): void {
  for (const connection of network.connections) {
    const dropConnectMask =
      runtimeNetwork._rand() < runtimeNetwork._dropConnectProb ? 0 : 1;

    if (dropConnectMask === 0) {
      stats.droppedConnections++;
    }

    setDropConnectMask(connection, dropConnectMask);

    if (dropConnectMask === 0) {
      stashOriginalDropConnectWeight(connection);
      connection.weight = 0;
      continue;
    }

    restoreOriginalDropConnectWeight(connection);
  }
}

/**
 * Restore drop-connect modified weights and normalize all masks back to one.
 *
 * @param network Network being activated.
 * @returns Nothing.
 */
function restoreDropConnectWeights(network: Network): void {
  for (const connection of network.connections) {
    restoreOriginalDropConnectWeight(connection);
    setDropConnectMask(connection, 1);
  }
}

/**
 * Restore temporary weight-noise values for fallback node path only.
 *
 * @param network Network being activated.
 * @param isTraining Training-time flag.
 * @param appliedWeightNoise Whether weight noise was applied during this pass.
 * @returns Nothing.
 */
function finalizeNodePathWeightNoiseRestore(
  network: Network,
  isTraining: boolean,
  appliedWeightNoise: boolean,
): void {
  if (!isTraining || !appliedWeightNoise) {
    return;
  }

  if (hasLayeredNetwork(network)) {
    return;
  }

  for (const connection of network.connections) {
    if (!hasOriginalWeightNoise(connection)) {
      continue;
    }

    restoreOriginalWeightNoise(connection);
  }
}

/**
 * Finalize training counters and attach activation statistics to runtime state.
 *
 * @param runtimeNetwork Runtime activation internals.
 * @param stats Activation stats.
 * @param isTraining Training-time flag.
 * @returns Nothing.
 */
function finalizeTrainingStepAndStats(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  stats: ActivationStats,
  isTraining: boolean,
): void {
  if (isTraining) {
    runtimeNetwork._trainingStep++;
  }

  if (stats.weightNoise.count > 0) {
    stats.weightNoise.meanAbs =
      stats.weightNoise.sumAbs / stats.weightNoise.count;
  }

  runtimeNetwork._lastStats = stats;
}

/**
 * Release pooled output buffer and return a detached plain array copy.
 *
 * @param outputBuffer Mutable pooled output buffer.
 * @returns Plain array of output values.
 */
function releaseBufferAndCreateResult(outputBuffer: ActivationArray): number[] {
  const result = Array.from(outputBuffer) as number[];
  activationArrayPool.release(outputBuffer);
  return result;
}

/**
 * Check whether a connection already has an original weight-noise snapshot.
 *
 * @param connection Connection to inspect.
 * @returns True when snapshot exists.
 */
function hasOriginalWeightNoise(
  connection: Network['connections'][number],
): boolean {
  return (
    (connection as unknown as ConnectionWeightNoiseProps)._origWeightNoise !=
    null
  );
}

/**
 * Store current connection weight before applying temporary weight-noise modifications.
 *
 * @param connection Connection to persist.
 * @returns Nothing.
 */
function persistOriginalWeightNoise(
  connection: Network['connections'][number],
): void {
  (connection as unknown as ConnectionWeightNoiseProps)._origWeightNoise =
    connection.weight;
}

/**
 * Restore and clear the original weight-noise snapshot for a connection.
 *
 * @param connection Connection to restore.
 * @returns Nothing.
 */
function restoreOriginalWeightNoise(
  connection: Network['connections'][number],
): void {
  connection.weight = (
    connection as unknown as ConnectionWeightNoiseProps
  )._origWeightNoise!;
  delete (connection as unknown as ConnectionWeightNoiseProps)._origWeightNoise;
}

/**
 * Persist last sampled weight-noise value for a connection.
 *
 * @param connection Connection to annotate.
 * @param sampledNoise Last sampled noise.
 * @returns Nothing.
 */
function setLastSampledNoise(
  connection: Network['connections'][number],
  sampledNoise: number,
): void {
  (connection as unknown as ConnectionWeightNoiseProps)._wnLast = sampledNoise;
}

/**
 * Set drop-connect mask value for a connection.
 *
 * @param connection Connection to annotate.
 * @param dropConnectMask Drop-connect mask value.
 * @returns Nothing.
 */
function setDropConnectMask(
  connection: Network['connections'][number],
  dropConnectMask: number,
): void {
  (connection as unknown as ConnectionWeightNoiseProps).dcMask =
    dropConnectMask;
}

/**
 * Store original connection weight before drop-connect zeroing.
 *
 * @param connection Connection to persist.
 * @returns Nothing.
 */
function stashOriginalDropConnectWeight(
  connection: Network['connections'][number],
): void {
  if (
    (connection as unknown as ConnectionWeightNoiseProps)._origWeight == null
  ) {
    (connection as unknown as ConnectionWeightNoiseProps)._origWeight =
      connection.weight;
  }
}

/**
 * Restore and clear original connection weight after drop-connect.
 *
 * @param connection Connection to restore.
 * @returns Nothing.
 */
function restoreOriginalDropConnectWeight(
  connection: Network['connections'][number],
): void {
  if (
    (connection as unknown as ConnectionWeightNoiseProps)._origWeight == null
  ) {
    return;
  }

  connection.weight = (
    connection as unknown as ConnectionWeightNoiseProps
  )._origWeight!;
  delete (connection as unknown as ConnectionWeightNoiseProps)._origWeight;
}

void config;
