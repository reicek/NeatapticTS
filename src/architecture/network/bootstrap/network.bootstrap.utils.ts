/**
 * Bootstrap helpers for the public `Network` class.
 *
 * This chapter owns the work that happens exactly once during construction:
 * resolve the caller's topology contract, initialize runtime flags, prewarm
 * pooled activation storage, seed deterministic randomness when requested, and
 * materialize the first input-to-output graph.
 *
 * Keeping that one-time setup here lets `network.ts` stay focused on the
 * long-lived runtime surface while this file teaches the difference between
 * constructor policy and the ongoing activation or training lifecycle.
 */

import { config } from '../../../config';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import Node from '../../node/node';
import { acquireNode } from '../../nodePool/nodePool';
import type {
  NetworkBootstrapInternals,
  NetworkConstructorOptions,
  NetworkTopologyIntent,
} from '../network.types';

interface NetworkBootstrapContext {
  input: number;
  output: number;
  options?: NetworkConstructorOptions;
  topologyIntent: NetworkTopologyIntent;
  enforceAcyclic: boolean;
}

/**
 * Resolves the public topology intent for one constructor call.
 *
 * Prefer this semantic contract over the legacy low-level acyclic flag when
 * both are available.
 *
 * @param options Optional constructor options.
 * @returns Resolved topology intent.
 */
export function resolveTopologyIntent(
  options?: NetworkConstructorOptions,
): NetworkTopologyIntent {
  if (options?.topologyIntent) {
    return options.topologyIntent;
  }

  return options?.enforceAcyclic ? 'feed-forward' : 'unconstrained';
}

/**
 * Validates that legacy acyclic flags do not contradict public topology intent.
 *
 * This protects callers from creating a constructor packet that says
 * "feed-forward" in one field and "cyclic is allowed" in another.
 *
 * @param options Optional constructor options.
 * @returns Nothing.
 */
export function validateTopologyIntentConfiguration(
  options?: NetworkConstructorOptions,
): void {
  if (
    options?.topologyIntent === 'feed-forward' &&
    options.enforceAcyclic === false
  ) {
    throw new Error(
      'Conflicting topology options: feed-forward intent cannot disable acyclic enforcement.',
    );
  }

  if (
    options?.topologyIntent === 'unconstrained' &&
    options.enforceAcyclic === true
  ) {
    throw new Error(
      'Conflicting topology options: unconstrained intent cannot enable acyclic enforcement.',
    );
  }
}

/**
 * Resolves whether acyclic enforcement should be enabled for one constructor call.
 *
 * The semantic topology intent remains the source of truth unless the caller
 * explicitly opted into the legacy boolean toggle.
 *
 * @param options Optional constructor options.
 * @param topologyIntent Resolved public topology intent.
 * @returns True when acyclic enforcement should be enabled.
 */
export function resolveAcyclicEnforcement(
  options: NetworkConstructorOptions | undefined,
  topologyIntent: NetworkTopologyIntent,
): boolean {
  if (typeof options?.enforceAcyclic === 'boolean') {
    return options.enforceAcyclic;
  }

  return topologyIntent === 'feed-forward';
}

/**
 * Performs one-time `Network` construction setup.
 *
 * This orchestration keeps the constructor readable by separating four concerns:
 * public topology-policy resolution, runtime flag initialization, pooled memory
 * warmup, and synthesis of the initial fully connected IO graph.
 *
 * @param network Network instance being constructed.
 * @param bootstrapContext Constructor inputs and resolved topology policy.
 * @returns Nothing.
 */
export function bootstrapNetwork(
  network: NetworkBootstrapInternals,
  bootstrapContext: NetworkBootstrapContext,
): void {
  const { input, output, options, topologyIntent, enforceAcyclic } =
    bootstrapContext;

  // Step 1: Initialize runtime fields before later helpers mutate graph state.
  initializeRuntimeState(
    network,
    input,
    output,
    options,
    topologyIntent,
    enforceAcyclic,
  );

  // Step 2: Prewarm activation pooling so the first forward pass avoids avoidable churn.
  prewarmActivationPool(output);

  // Step 3: Seed deterministic RNG state when the caller requested reproducibility.
  applySeedOption(network, options);

  // Step 4: Materialize the initial IO nodes and their dense starter graph.
  initializeIONodes(network);
  connectInitialInputToOutputGraph(network);

  // Step 5: Synthesize minimum hidden capacity through the public node-split flow.
  ensureMinimumHiddenNodes(network, options?.minHidden ?? 0);
}

/**
 * Initializes core runtime state for a new network instance.
 *
 * @param network Network instance being constructed.
 * @param input Number of input nodes.
 * @param output Number of output nodes.
 * @param options Optional constructor options.
 * @param topologyIntent Resolved public topology intent.
 * @param enforceAcyclic Resolved low-level acyclic policy.
 * @returns Nothing.
 */
function initializeRuntimeState(
  network: NetworkBootstrapInternals,
  input: number,
  output: number,
  options: NetworkConstructorOptions | undefined,
  topologyIntent: NetworkTopologyIntent,
  enforceAcyclic: boolean,
): void {
  network.input = input;
  network.output = output;
  network.nodes = [];
  network.connections = [];
  network.gates = [];
  network.selfconns = [];
  network.dropout = 0;
  network._topologyIntent = topologyIntent;
  network._enforceAcyclic = enforceAcyclic;

  if (options?.activationPrecision) {
    network._activationPrecision = options.activationPrecision;
  } else if (config.float32Mode) {
    network._activationPrecision = 'f32';
  }

  network._reuseActivationArrays = options?.reuseActivationArrays === true;
  network._returnTypedActivations = options?.returnTypedActivations === true;
}

/**
 * Prewarms pooled activation buffers for the initial output width.
 *
 * Pool configuration errors remain non-fatal because allocation policy is an
 * optimization layer, not a correctness requirement for constructing a graph.
 *
 * @param output Number of output nodes.
 * @returns Nothing.
 */
function prewarmActivationPool(output: number): void {
  try {
    if (typeof config.poolMaxPerBucket === 'number') {
      activationArrayPool.setMaxPerBucket(config.poolMaxPerBucket);
    }

    const prewarmCount =
      typeof config.poolPrewarmCount === 'number' ? config.poolPrewarmCount : 2;
    activationArrayPool.prewarm(output, prewarmCount);
  } catch {
    // Pool warmup is best-effort and should never block construction.
  }
}

/**
 * Applies the optional deterministic seed from constructor options.
 *
 * @param network Network instance being constructed.
 * @param options Optional constructor options.
 * @returns Nothing.
 */
function applySeedOption(
  network: NetworkBootstrapInternals,
  options?: NetworkConstructorOptions,
): void {
  if (options?.seed !== undefined) {
    network.setSeed(options.seed);
  }
}

/**
 * Creates the initial input and output nodes for a new network.
 *
 * When node pooling is enabled, each acquired node is reset before use so the
 * freshly constructed graph still starts from deterministic runtime state.
 *
 * @param network Network instance being constructed.
 * @returns Nothing.
 */
function initializeIONodes(network: NetworkBootstrapInternals): void {
  for (
    let nodeIndex = 0;
    nodeIndex < network.input + network.output;
    nodeIndex++
  ) {
    const nodeType = nodeIndex < network.input ? 'input' : 'output';
    const nextNode = config.enableNodePooling
      ? acquireNode({ type: nodeType, rng: network._rand })
      : new Node(nodeType, undefined, network._rand);
    network.nodes.push(nextNode);
  }
}

/**
 * Connects every input node to every output node to form the starter graph.
 *
 * The initial weight scaling mirrors the existing Network constructor behavior
 * so this split remains a pure structural refactor.
 *
 * @param network Network instance being constructed.
 * @returns Nothing.
 */
function connectInitialInputToOutputGraph(
  network: NetworkBootstrapInternals,
): void {
  for (let inputIndex = 0; inputIndex < network.input; inputIndex++) {
    for (
      let outputIndex = network.input;
      outputIndex < network.input + network.output;
      outputIndex++
    ) {
      const weight =
        network._rand() * network.input * Math.sqrt(2 / network.input);
      network.connect(
        network.nodes[inputIndex],
        network.nodes[outputIndex],
        weight,
      );
    }
  }
}

/**
 * Ensures the network reaches the caller's requested minimum hidden width.
 *
 * This relies on the public node-split mutation flow so the constructor and the
 * evolutionary runtime keep growing hidden structure the same way.
 *
 * @param network Network instance being constructed.
 * @param minimumHiddenNodes Requested minimum hidden-node count.
 * @returns Nothing.
 */
function ensureMinimumHiddenNodes(
  network: NetworkBootstrapInternals,
  minimumHiddenNodes: number,
): void {
  while (
    network.nodes.length <
    network.input + network.output + minimumHiddenNodes
  ) {
    network.addNodeBetween();
  }
}
