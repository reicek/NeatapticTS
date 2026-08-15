/**
 * Core neuron chapter for the architecture surface.
 *
 * This folder now owns the library's lowest-level unit of computation: the
 * mutable neuron object that stores bias, state, activation history, tracing
 * information, connection references, and optimizer scratch space.
 *
 * Read this chapter in three passes:
 *
 * 1. start with the `Node` class overview to understand the long-lived state a
 *    neuron carries between activation, propagation, mutation, and
 *    serialization,
 * 2. continue to the activation methods when you want the difference between
 *    traced and no-trace execution,
 * 3. finish with connectivity, gating, and optimizer helpers when you need to
 *    understand how one node participates in larger graph changes.
 *
 * Architecture building asks two different questions at this boundary:
 * what role does the neuron play at runtime, and what human-facing meaning
 * should later tooling remember about it? The runtime role lives in `type`
 * (`input`, `hidden`, `output`) and changes execution semantics. The optional
 * descriptor surface (`label`, `intent`, `metadata`) does not change
 * activation math; it keeps architecture intent visible for later diagnostics,
 * visualization, and construct-from-parts work.
 */
import Connection from '../connection';
import { config } from '../../config';
import * as methods from '../../methods/methods';
import {
  NodeInvalidConnectionTargetTypeError,
  NodeMutationMethodRequiredError,
  NodeUndefinedConnectionTargetError,
  NodeUnknownMutationMethodError,
  NodeUnsupportedMutationMethodError,
} from './node.errors';

/**
 * Internal interface for accessing dynamic optimizer properties on Node instances.
 * These properties are lazily allocated and not part of the main class definition.
 */
interface NodeOptimizerProps {
  opt_mB?: number;
  opt_mB2?: number;
  opt_vB?: number;
  opt_vhatB?: number;
  opt_uB?: number;
  _la_k?: number;
  _la_alpha?: number;
  _la_step?: number;
  _la_shadowBias?: number;
  batchNorm?: boolean;
}

interface OptimizerMomentCarrier {
  firstMoment?: number;
  gradientAccumulator?: number;
  infinityNorm?: number;
  maxSecondMoment?: number;
  previousDelta?: number;
  secondMoment?: number;
  secondMomentum?: number;
  tracksAuxiliaryVariance?: boolean;
}

interface OptimizerHyperparams {
  beta1: number;
  beta2: number;
  eps: number;
  lrScale: number;
  momentum: number;
  t: number;
}

interface BatchOptimizerOptions {
  type:
    | 'sgd'
    | 'rmsprop'
    | 'adagrad'
    | 'adam'
    | 'adamw'
    | 'amsgrad'
    | 'adamax'
    | 'nadam'
    | 'radam'
    | 'lion'
    | 'adabelief'
    | 'lookahead';
  momentum?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  lrScale?: number;
  t?: number;
  baseType?: string;
  la_k?: number;
  la_alpha?: number;
}

interface ResolvedBatchOptimizerPlan {
  effectiveType: string;
  optimizerParams: OptimizerHyperparams;
  type: string;
  weightDecay: number;
}

type OptimizerStepFn = (
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
) => number;

type NodeMutationMethodShape = {
  allowed?: ((x: number, derivate?: boolean) => number)[];
  max?: number;
  min?: number;
  name?: string;
};

type NodeMutationHandler = (
  node: Node,
  mutationMethod: NodeMutationMethodShape,
) => void;

const OPTIMIZER_STEP_TABLE: Record<string, OptimizerStepFn> = {
  adabelief: computeAdaBeliefStep,
  adagrad: computeAdaGradStep,
  adam: computeAdamStep,
  adamax: computeAdamaxStep,
  adamw: computeAdamStep,
  amsgrad: computeAmsgradStep,
  lion: computeLionStep,
  nadam: computeNadamStep,
  radam: computeRAdamStep,
  rmsprop: computeRmsPropStep,
  sgd: computeSgdStep,
};

const MUTATION_HANDLER_TABLE: Record<string, NodeMutationHandler> = {
  [methods.mutation.BATCH_NORM.name]: applyBatchNormMutation,
  [methods.mutation.MOD_ACTIVATION.name]: applyActivationMutation,
  [methods.mutation.MOD_BIAS.name]: applyBiasMutation,
  [methods.mutation.REINIT_WEIGHT.name]: applyReinitializeWeightMutation,
};

function applyActivationMutation(
  node: Node,
  mutationMethod: NodeMutationMethodShape,
): void {
  if (!mutationMethod.allowed || mutationMethod.allowed.length === 0) {
    console.warn(
      'MOD_ACTIVATION mutation called without allowed functions specified.',
    );
    return;
  }

  const allowedActivations = mutationMethod.allowed;
  const currentIndex = allowedActivations.indexOf(node.squash);
  let nextActivationIndex = currentIndex;

  if (allowedActivations.length > 1) {
    nextActivationIndex =
      (currentIndex +
        Math.floor(Math.random() * (allowedActivations.length - 1)) +
        1) %
      allowedActivations.length;
  }

  node.squash = allowedActivations[nextActivationIndex];
}

function applyBiasMutation(
  node: Node,
  mutationMethod: NodeMutationMethodShape,
): void {
  const minimumBiasDelta = mutationMethod.min ?? -1;
  const maximumBiasDelta = mutationMethod.max ?? 1;
  node.bias += resolveRandomizedMutationValue(
    minimumBiasDelta,
    maximumBiasDelta,
  );
}

function applyReinitializeWeightMutation(
  node: Node,
  mutationMethod: NodeMutationMethodShape,
): void {
  const minimumWeight = mutationMethod.min ?? -1;
  const maximumWeight = mutationMethod.max ?? 1;

  updateConnectionWeights(node.connections.in, minimumWeight, maximumWeight);
  updateConnectionWeights(node.connections.out, minimumWeight, maximumWeight);
  updateConnectionWeights(node.connections.self, minimumWeight, maximumWeight);
}

function applyBatchNormMutation(node: Node): void {
  (node as unknown as { batchNorm: boolean }).batchNorm = true;
}

function assertKnownMutationMethod(
  mutationMethod: NodeMutationMethodShape,
): void {
  const mutationName = resolveMutationMethodName(mutationMethod);

  if (!(mutationName && mutationName in methods.mutation)) {
    throw new NodeUnknownMutationMethodError(
      `Unknown mutation method: ${mutationMethod.name ?? 'undefined'}`,
    );
  }
}

function assertCanonicalMutationMethod(
  mutationMethod: NodeMutationMethodShape,
): void {
  const mutationName = resolveMutationMethodName(mutationMethod);
  const canonicalMutationMethod = mutationName
    ? Reflect.get(methods.mutation, mutationName)
    : undefined;

  if (canonicalMutationMethod !== mutationMethod) {
    throw new NodeUnsupportedMutationMethodError(
      `Unsupported mutation method: ${mutationMethod.name ?? 'undefined'}`,
    );
  }
}

function resolveMutationHandler(
  mutationMethod: NodeMutationMethodShape,
): NodeMutationHandler | undefined {
  const mutationName = resolveMutationMethodName(mutationMethod);

  return mutationName ? MUTATION_HANDLER_TABLE[mutationName] : undefined;
}

function resolveMutationMethodName(
  mutationMethod: NodeMutationMethodShape,
): string | undefined {
  return mutationMethod.name;
}

function updateConnectionWeights(
  connections: Connection[],
  minimumWeight: number,
  maximumWeight: number,
): void {
  for (const connection of connections) {
    connection.weight = resolveRandomizedMutationValue(
      minimumWeight,
      maximumWeight,
    );
  }
}

function resolveRandomizedMutationValue(
  minimumValue: number,
  maximumValue: number,
): number {
  return Math.random() * (maximumValue - minimumValue) + minimumValue;
}

function computeRmsPropStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  carrier.gradientAccumulator =
    (carrier.gradientAccumulator ?? 0) * 0.9 + 0.1 * (gradient * gradient);

  return (
    (gradient / (Math.sqrt(carrier.gradientAccumulator ?? 0) + params.eps)) *
    params.lrScale
  );
}

function computeAdaGradStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  carrier.gradientAccumulator =
    (carrier.gradientAccumulator ?? 0) + gradient * gradient;

  return (
    (gradient / (Math.sqrt(carrier.gradientAccumulator ?? 0) + params.eps)) *
    params.lrScale
  );
}

function computeAdamStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  updateAdamMoments(gradient, carrier, params);
  const { correctedFirstMoment, correctedSecondMoment } =
    resolveBiasCorrectedMoments(carrier, params, carrier.secondMoment ?? 0);

  return (
    (correctedFirstMoment / (Math.sqrt(correctedSecondMoment) + params.eps)) *
    params.lrScale
  );
}

function computeAmsgradStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  updateAdamMoments(gradient, carrier, params);
  carrier.maxSecondMoment = Math.max(
    carrier.maxSecondMoment ?? 0,
    carrier.secondMoment ?? 0,
  );

  const { correctedFirstMoment, correctedSecondMoment } =
    resolveBiasCorrectedMoments(carrier, params, carrier.maxSecondMoment ?? 0);

  return (
    (correctedFirstMoment / (Math.sqrt(correctedSecondMoment) + params.eps)) *
    params.lrScale
  );
}

function computeAdamaxStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  carrier.firstMoment =
    (carrier.firstMoment ?? 0) * params.beta1 + (1 - params.beta1) * gradient;
  carrier.infinityNorm = Math.max(
    (carrier.infinityNorm ?? 0) * params.beta2,
    Math.abs(gradient),
  );

  if (carrier.tracksAuxiliaryVariance) {
    carrier.secondMoment =
      (carrier.secondMoment ?? 0) * params.beta2 +
      (1 - params.beta2) * (gradient * gradient);
  }

  const correctedFirstMoment =
    (carrier.firstMoment ?? 0) / (1 - Math.pow(params.beta1, params.t));

  return (
    (correctedFirstMoment / ((carrier.infinityNorm ?? 0) || 1e-12)) *
    params.lrScale
  );
}

function computeNadamStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  updateAdamMoments(gradient, carrier, params);
  const { correctedFirstMoment, correctedSecondMoment } =
    resolveBiasCorrectedMoments(carrier, params, carrier.secondMoment ?? 0);
  const nesterovFirstMoment =
    correctedFirstMoment * params.beta1 +
    ((1 - params.beta1) * gradient) / (1 - Math.pow(params.beta1, params.t));

  return (
    (nesterovFirstMoment / (Math.sqrt(correctedSecondMoment) + params.eps)) *
    params.lrScale
  );
}

function computeRAdamStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  updateAdamMoments(gradient, carrier, params);
  const { correctedFirstMoment, correctedSecondMoment } =
    resolveBiasCorrectedMoments(carrier, params, carrier.secondMoment ?? 0);
  const rhoInfinity = 2 / (1 - params.beta2) - 1;
  const rhoCurrent =
    rhoInfinity -
    (2 * params.t * Math.pow(params.beta2, params.t)) /
      (1 - Math.pow(params.beta2, params.t));

  if (rhoCurrent <= 4) {
    return correctedFirstMoment * params.lrScale;
  }

  const rectificationTerm = Math.sqrt(
    ((rhoCurrent - 4) * (rhoCurrent - 2) * rhoInfinity) /
      ((rhoInfinity - 4) * (rhoInfinity - 2) * rhoCurrent),
  );

  return (
    ((rectificationTerm * correctedFirstMoment) /
      (Math.sqrt(correctedSecondMoment) + params.eps)) *
    params.lrScale
  );
}

function computeLionStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  carrier.firstMoment =
    (carrier.firstMoment ?? 0) * params.beta1 + (1 - params.beta1) * gradient;
  carrier.secondMomentum =
    (carrier.secondMomentum ?? 0) * params.beta2 +
    (1 - params.beta2) * gradient;

  if (carrier.tracksAuxiliaryVariance) {
    carrier.secondMoment =
      (carrier.secondMoment ?? 0) * params.beta2 +
      (1 - params.beta2) * (gradient * gradient);
  }

  return (
    -Math.sign((carrier.firstMoment ?? 0) + (carrier.secondMomentum ?? 0)) *
    params.lrScale
  );
}

function computeAdaBeliefStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  carrier.firstMoment =
    (carrier.firstMoment ?? 0) * params.beta1 + (1 - params.beta1) * gradient;

  const surpriseGradient = gradient - (carrier.firstMoment ?? 0);
  carrier.secondMoment =
    (carrier.secondMoment ?? 0) * params.beta2 +
    (1 - params.beta2) * (surpriseGradient * surpriseGradient);

  const { correctedFirstMoment, correctedSecondMoment } =
    resolveBiasCorrectedMoments(carrier, params, carrier.secondMoment ?? 0);

  return (
    (correctedFirstMoment /
      (Math.sqrt(correctedSecondMoment) + params.eps + 1e-12)) *
    params.lrScale
  );
}

function computeSgdStep(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): number {
  let currentDelta = gradient + params.momentum * (carrier.previousDelta ?? 0);

  if (!Number.isFinite(currentDelta)) {
    currentDelta = 0;
  }

  if (Math.abs(currentDelta) > 1e3) {
    currentDelta = Math.sign(currentDelta) * 1e3;
  }

  carrier.previousDelta = currentDelta;
  return currentDelta * params.lrScale;
}

function updateAdamMoments(
  gradient: number,
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
): void {
  carrier.firstMoment =
    (carrier.firstMoment ?? 0) * params.beta1 + (1 - params.beta1) * gradient;
  carrier.secondMoment =
    (carrier.secondMoment ?? 0) * params.beta2 +
    (1 - params.beta2) * (gradient * gradient);
}

function resolveBiasCorrectedMoments(
  carrier: OptimizerMomentCarrier,
  params: OptimizerHyperparams,
  effectiveVariance: number,
): { correctedFirstMoment: number; correctedSecondMoment: number } {
  return {
    correctedFirstMoment:
      (carrier.firstMoment ?? 0) / (1 - Math.pow(params.beta1, params.t)),
    correctedSecondMoment:
      effectiveVariance / (1 - Math.pow(params.beta2, params.t)),
  };
}

function createConnectionOptimizerCarrier(
  connection: Connection,
): OptimizerMomentCarrier {
  return {
    get firstMoment() {
      return connection.firstMoment;
    },
    set firstMoment(value: number | undefined) {
      connection.firstMoment = value;
    },
    get gradientAccumulator() {
      return connection.gradientAccumulator;
    },
    set gradientAccumulator(value: number | undefined) {
      connection.gradientAccumulator = value;
    },
    get infinityNorm() {
      return connection.infinityNorm;
    },
    set infinityNorm(value: number | undefined) {
      connection.infinityNorm = value;
    },
    get maxSecondMoment() {
      return connection.maxSecondMoment;
    },
    set maxSecondMoment(value: number | undefined) {
      connection.maxSecondMoment = value;
    },
    get previousDelta() {
      return connection.previousDeltaWeight;
    },
    set previousDelta(value: number) {
      connection.previousDeltaWeight = value;
    },
    get secondMoment() {
      return connection.secondMoment;
    },
    set secondMoment(value: number | undefined) {
      connection.secondMoment = value;
    },
    get secondMomentum() {
      return connection.secondMomentum;
    },
    set secondMomentum(value: number | undefined) {
      connection.secondMomentum = value;
    },
    tracksAuxiliaryVariance: false,
  };
}

function createBiasOptimizerCarrier(
  node: Node,
  optimizerState: NodeOptimizerProps,
): OptimizerMomentCarrier {
  return {
    get firstMoment() {
      return optimizerState.opt_mB;
    },
    set firstMoment(value: number | undefined) {
      optimizerState.opt_mB = value;
    },
    get gradientAccumulator() {
      return undefined;
    },
    set gradientAccumulator(_value: number | undefined) {},
    get infinityNorm() {
      return optimizerState.opt_uB;
    },
    set infinityNorm(value: number | undefined) {
      optimizerState.opt_uB = value;
    },
    get maxSecondMoment() {
      return optimizerState.opt_vhatB;
    },
    set maxSecondMoment(value: number | undefined) {
      optimizerState.opt_vhatB = value;
    },
    get previousDelta() {
      return node.previousDeltaBias;
    },
    set previousDelta(value: number) {
      node.previousDeltaBias = value;
    },
    get secondMoment() {
      return optimizerState.opt_vB;
    },
    set secondMoment(value: number | undefined) {
      optimizerState.opt_vB = value;
    },
    get secondMomentum() {
      return optimizerState.opt_mB2;
    },
    set secondMomentum(value: number | undefined) {
      optimizerState.opt_mB2 = value;
    },
    tracksAuxiliaryVariance: true,
  };
}

/**
 * Runtime-supported primitive roles for architecture-building surfaces.
 *
 * - `'input'`: receives the external stimulus vector; no bias computation.
 * - `'hidden'`: standard neuron with bias, activation function, and trace storage.
 * - `'output'`: emits the network result; error and gradient accumulation begin here during back-propagation.
 */
export type PrimitiveNodeType = 'input' | 'hidden' | 'output';

/**
 * Semantic role hints that later architecture tooling can attach to individual primitives.
 *
 * These labels do not change activation math — they let diagnostics, visualizers, and
 * builders identify what a node represents conceptually (e.g. `'gate'` for gating neurons,
 * `'memory'` for LSTM cell nodes, `'attention'` for attention-head units).
 */
export type PrimitiveIntent =
  | 'attention'
  | 'convolution'
  | 'gate'
  | 'hidden'
  | 'input'
  | 'memory'
  | 'normalization'
  | 'output'
  | 'recurrent'
  | 'state';

/** Scalar metadata value types retained on architecture primitives. Kept narrow so metadata bags are safe to serialize and diff across checkpoints. */
export type PrimitiveMetadataValue = boolean | null | number | string;

/** Lightweight key-value metadata bag attached to architecture primitives. Keys are free-form strings; values are restricted to serializable scalars. */
export type PrimitiveMetadata = Record<string, PrimitiveMetadataValue>;

/**
 * Optional human-readable descriptor attached to one architecture primitive.
 *
 * Descriptors are intentionally lightweight. They keep the public primitive
 * API teachable by separating runtime behavior from architectural meaning:
 * `type` still controls execution semantics, while `label`, `intent`, and
 * scalar metadata keep the boundary recognizable to later tooling.
 *
 * @example
 * ```ts
 * const readout = new Node('output');
 *
 * readout.describe({
 *   label: 'policyLogits',
 *   metadata: { stage: 'readout' },
 * });
 * ```
 */
export interface PrimitiveDescriptor {
  intent?: PrimitiveIntent | null;
  label?: string | null;
  metadata?: PrimitiveMetadata;
}

/**
 * Resolve a public primitive intent from a runtime node type string.
 *
 * Returns the intent that matches the type when the type is one of the
 * three standard roles (`'input'`, `'hidden'`, `'output'`), or `null`
 * for any other string. Used by descriptor helpers to stamp a default
 * intent without requiring the caller to repeat the role mapping.
 *
 * @param nodeType - Runtime node type string to resolve.
 * @returns The matching primitive intent or null for non-standard types.
 */
export function resolvePrimitiveIntent(
  nodeType: string,
): PrimitiveIntent | null {
  if (nodeType === 'input' || nodeType === 'hidden' || nodeType === 'output') {
    return nodeType;
  }

  return null;
}

const NEUTRAL_NODE_RESPONSE = 1;

/**
 * Node (Neuron)
 * =============
 * Fundamental computational unit: aggregates weighted inputs, applies an activation
 * function (squash) and emits an activation value. Supports:
 *  - Types: 'input' | 'hidden' | 'output' (affects bias initialization & error handling)
 *  - Recurrent self‑connections & gated connections (for dynamic / RNN behavior)
 *  - Dropout mask (`mask`), momentum terms, eligibility & extended traces (for
 *    a variety of learning rules beyond simple backprop).
 *  - Optional descriptors via `describe({ label, intent, metadata })` when a
 *    low-level node should keep a stable human-facing identity inside a larger
 *    architecture story.
 *
 * Educational note: Traces (`eligibility` and `xtrace`) illustrate how recurrent credit
 * assignment works in algorithms like RTRL / policy gradients. They are updated only when
 * using the traced activation path (`activate`) vs `noTraceActivate` (inference fast path).
 *
 * Most architecture code should only attach a descriptor when a node boundary
 * matters outside the current function. That keeps the primitive cheap for raw
 * graph math while still letting later passes recover names such as
 * `readoutNode`, `memoryCell`, or `temperatureGate`.
 *
 * @example
 * ```ts
 * const sensor = new Node('input');
 * const readout = new Node('output');
 *
 * readout.describe({
 *   label: 'readoutNode',
 *   metadata: { stage: 'policy' },
 * });
 * ```
 *
 * @see Instinct article (Section 1.1 Nodes) for conceptual background.
 */
export default class Node {
  /**
   * The bias value of the node. Added to the weighted sum of inputs before activation.
   * Input nodes typically have a bias of 0.
   */
  bias: number;
  /**
   * Response multiplier applied to the node state before the squash function.
   *
   * A neutral response of `1` preserves the historical runtime behavior. Values
   * above or below `1` steepen or flatten the node's effective transfer curve
   * without changing the chosen activation family.
   */
  response: number;
  /**
   * The activation function (squashing function) applied to the node's state.
   * Maps the internal state to the node's output (activation).
   * @param x The node's internal state (sum of weighted inputs + bias).
   * @param derivate If true, returns the derivative of the function instead of the function value.
   * @returns The activation value or its derivative.
   */
  squash: (x: number, derivate?: boolean) => number;
  /**
   * The type of the node: 'input', 'hidden', or 'output'.
   * Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).
   */
  type: string;
  /** Optional human-readable descriptor label for architecture tooling. */
  label: string | null;
  /** Optional semantic intent for architecture tooling and diagnostics. */
  intent: PrimitiveIntent | null;
  /** Optional scalar metadata retained on the primitive boundary. */
  metadata: PrimitiveMetadata;
  /**
   * The output value of the node after applying the activation function. This is the value transmitted to connected nodes.
   */
  activation: number;
  /**
   * The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.
   */
  state: number;
  /**
   * The node's state from the previous activation cycle. Used for recurrent self-connections.
   */
  old: number;
  /**
   * A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.
   */
  mask: number;
  /**
   * The change in bias applied in the previous training iteration. Used for calculating momentum.
   */
  previousDeltaBias: number;
  /**
   * Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.
   */
  totalDeltaBias: number;
  /**
   * Stores incoming, outgoing, gated, and self-connections for this node.
   */
  connections: {
    /** Incoming connections to this node. */
    in: Connection[];
    /** Outgoing connections from this node. */
    out: Connection[];
    /** Connections gated by this node's activation. */
    gated: Connection[];
    /** The recurrent self-connection. */
    self: Connection[];
  };
  /**
   * Stores error values calculated during backpropagation.
   */
  error: {
    /** The node's responsibility for the network error, calculated based on projected and gated errors. */
    responsibility: number;
    /** Error projected back from nodes this node connects to. */
    projected: number;
    /** Error projected back from connections gated by this node. */
    gated: number;
  };
  /**
   * The derivative of the activation function evaluated at the node's current state. Used in backpropagation.
   */
  derivative?: number;
  /**
   * Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.
   */
  index?: number;
  /**
   * Internal flag to detect cycles during activation.
   */
  private isActivating?: boolean;
  /** Stable per-node gene identifier for NEAT innovation reuse. */
  geneId: number;

  /**
   * Global index counter for assigning unique indices to nodes.
   */
  private static _globalNodeIndex = 0;
  private static _nextGeneId = 1;

  /**
   * Creates a new node.
   * @param type The type of the node ('input', 'hidden', or 'output'). Defaults to 'hidden'.
   * @param customActivation Optional custom activation function (should handle derivative if needed).
   */
  constructor(
    type: string = 'hidden',
    customActivation?: (x: number, derivate?: boolean) => number,
    rng: () => number = Math.random,
  ) {
    // Initialize bias: 0 for input nodes, small random value for others (deterministic if rng seeded)
    this.bias = type === 'input' ? 0 : rng() * 0.2 - 0.1;
    this.response = NEUTRAL_NODE_RESPONSE;
    // Set activation function. Default to logistic or identity if logistic is not available.
    this.squash = customActivation ?? methods.Activation.logistic ?? ((x) => x);
    this.type = type;
    this.label = null;
    this.intent = resolvePrimitiveIntent(type);
    this.metadata = {};

    // Initialize state and activation values.
    this.activation = 0;
    this.state = 0;
    this.old = 0;

    // Initialize mask for dropout (default is no dropout).
    this.mask = 1;

    // Initialize momentum tracking variables.
    this.previousDeltaBias = 0;

    // Initialize batch training accumulator.
    this.totalDeltaBias = 0;

    // Initialize connection storage.
    this.connections = {
      in: [],
      out: [],
      gated: [],
      // Self-connection initialized as an empty array.
      self: [],
    };

    // Initialize error tracking variables for backpropagation.
    this.error = {
      responsibility: 0,
      projected: 0,
      gated: 0,
    };

    // Assign a unique index for this live instance.
    this.index = Node._globalNodeIndex++;
    // Assign stable gene id (independent from per-network index)
    this.geneId = Node._nextGeneId++;
  }

  /**
   * Advances the global gene-id cursor past a restored maximum.
   *
   * Restore flows use this after hydrating persisted genomes so the next freshly
   * created node cannot collide with an older serialized `geneId`.
   *
   * @param maxObservedGeneId Highest restored node gene id currently in memory.
   * @returns Nothing.
   */
  static syncGeneIdCounter(maxObservedGeneId: number): void {
    if (!Number.isFinite(maxObservedGeneId)) {
      return;
    }

    Node._nextGeneId = Math.max(Node._nextGeneId, maxObservedGeneId + 1);
  }

  /**
   * Sets a custom activation function for this node at runtime.
   * @param fn The activation function (should handle derivative if needed).
   */
  setActivation(fn: (x: number, derivate?: boolean) => number) {
    this.squash = fn;
  }

  /**
   * Attaches optional descriptor metadata to the primitive boundary.
   *
   * This descriptor is advisory only. It does not change runtime activation,
   * mutation, or serialization behavior, but it gives later architecture
   * assembly, diagnostics, and visualization passes a stable place to read
   * human-facing labels and intent.
   *
   * Reach for this when the node is still the right abstraction but a later
   * reader should not have to infer its purpose from connection order alone.
   *
   * @param descriptor Optional label, intent, and scalar metadata to merge.
   * @returns Nothing.
   *
   * @example
   * ```ts
   * const readout = new Node('output');
   * readout.describe({
   *   label: 'readoutNode',
   *   metadata: { stage: 'readout' },
   * });
   * ```
   */
  describe(descriptor: PrimitiveDescriptor): void {
    if (descriptor.label !== undefined) {
      this.label = descriptor.label;
    }

    if (descriptor.intent !== undefined) {
      this.intent = descriptor.intent;
    }

    if (descriptor.metadata !== undefined) {
      this.metadata = {
        ...this.metadata,
        ...descriptor.metadata,
      };
    }
  }

  /**
   * Activates the node, calculating its output value based on inputs and state.
   * This method also calculates eligibility traces (`xtrace`) used for training recurrent connections.
   *
   * The activation process involves:
   * 1. Calculating the node's internal state (`this.state`) based on:
   *    - Incoming connections' weighted activations.
   *    - The recurrent self-connection's weighted state from the previous timestep (`this.old`).
   *    - The node's bias.
   * 2. Applying the activation function (`this.squash`) to the state to get the activation (`this.activation`).
   * 3. Applying the dropout mask (`this.mask`).
   * 4. Calculating the derivative of the activation function.
   * 5. Updating the gain of connections gated by this node.
   * 6. Calculating and updating eligibility traces for incoming connections.
   *
   * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
   * @returns The calculated activation value of the node.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
   */
  activate(input?: number): number {
    return this._activateCore(true, input);
  }

  /**
   * Activates the node without calculating eligibility traces (`xtrace`).
   * This is a performance optimization used during inference (when the network
   * is just making predictions, not learning) as trace calculations are only needed for training.
   *
   * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
   * @returns The calculated activation value of the node.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
   */
  noTraceActivate(input?: number): number {
    return this._activateCore(false, input);
  }

  /**
   * Internal shared implementation for activate/noTraceActivate.
   * @param withTrace Whether to update eligibility traces.
   * @param input Optional externally supplied activation (bypasses weighted sum if provided).
   */
  private _activateCore(withTrace: boolean, input?: number): number {
    if (this.mask === 0) {
      this.activation = 0;
      return 0;
    }

    if (typeof input !== 'undefined') {
      return this._activateFromInput(withTrace, input);
    }

    this.old = this.state;
    this.state = this._accumulateNodeState(
      this.bias,
      this.connections.self,
      this.connections.in,
      this.old,
    );
    this._normalizeActivationSettings();

    const effectiveState = this.state * this.response;
    this.activation = this.squash(effectiveState) * this.mask;
    this.derivative = this.squash(effectiveState, true) * this.response;
    this._updateGatedConnectionGains();
    this._updateEligibilityTraces(withTrace);

    return this.activation;
  }

  /**
   * Back-propagates the error signal through the node and calculates weight/bias updates.
   *
   * This method implements the backpropagation algorithm, including:
   * 1. Calculating the node's error responsibility based on errors from subsequent nodes (`projected` error)
   *    and errors from connections it gates (`gated` error).
   * 2. Calculating the gradient for each incoming connection's weight using eligibility traces (`xtrace`).
   * 3. Calculating the change (delta) for weights and bias, incorporating:
   *    - Learning rate.
   *    - L1/L2/custom regularization.
   *    - Momentum (using Nesterov Accelerated Gradient - NAG).
   * 4. Optionally applying the calculated updates immediately or accumulating them for batch training.
   *
   * @param rate The learning rate (controls the step size of updates).
   * @param momentum The momentum factor (helps accelerate learning and overcome local minima). Uses NAG.
   * @param update If true, apply the calculated weight/bias updates immediately. If false, accumulate them in `totalDelta*` properties for batch updates.
   * @param regularization The regularization setting. Can be:
   *   - number (L2 lambda)
   *   - { type: 'L1'|'L2', lambda: number }
   *   - (weight: number) => number (custom function)
   * @param target The target output value for this node. Only used if the node is of type 'output'.
   */
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    regularization:
      | number
      | { type: 'L1' | 'L2'; lambda: number }
      | ((weight: number) => number) = 0,
    target?: number,
  ): void {
    if (update && momentum > 0) {
      this._applyNagLookahead(momentum);
    }

    if (this.type === 'output') {
      this.error.responsibility = this.error.projected =
        target! - this.activation;
    } else {
      this.error.projected = this.derivative! * this._computeProjectedError();
      this.error.gated = this.derivative! * this._computeGatedError();
      this.error.responsibility = this.error.projected + this.error.gated;
    }

    if (this.type === 'constant') return;

    this._propagateConnections(
      this.connections.in,
      rate,
      momentum,
      update,
      regularization,
      false,
    );
    this._propagateConnections(
      this.connections.self,
      rate,
      momentum,
      update,
      regularization,
      true,
    );
    this._applyBiasDelta(rate, momentum, update);
  }

  private _activateFromInput(withTrace: boolean, input: number): number {
    if (this.type === 'input') {
      this.activation = input;
      return this.activation;
    }

    this.state = input;
    const effectiveState = this.state * this.response;
    this.activation = this.squash(effectiveState) * this.mask;
    this.derivative = this.squash(effectiveState, true) * this.response;
    this._updateGatedConnectionGains();
    this._updateEligibilityTraces(withTrace);

    return this.activation;
  }

  private _accumulateNodeState(
    bias: number,
    selfConnections: Connection[],
    incomingConnections: Connection[],
    oldState: number,
  ): number {
    let nextState = bias;

    for (const connection of selfConnections) {
      if (connection.dcMask === 0) continue;
      nextState += connection.gain * connection.weight * oldState;
    }

    for (const connection of incomingConnections) {
      if (connection.dcMask === 0 || connection.enabled === false) continue;
      nextState +=
        connection.from.activation * connection.weight * connection.gain;
    }

    return nextState;
  }

  private _normalizeActivationSettings(): void {
    if (typeof this.squash !== 'function') {
      if (config.warnings) {
        console.warn('Invalid activation function; using identity.');
      }
      this.squash = methods.Activation.identity;
    }

    if (typeof this.mask !== 'number') {
      this.mask = 1;
    }
  }

  private _updateGatedConnectionGains(): void {
    for (const connection of this.connections.gated) {
      connection.gain = this.activation;
    }
  }

  private _updateEligibilityTraces(withTrace: boolean): void {
    if (!withTrace) {
      return;
    }

    for (const connection of this.connections.in) {
      connection.eligibility = connection.from.activation;
    }
  }

  private _applyNagLookahead(momentum: number): void {
    for (const connection of this.connections.in) {
      connection.weight += momentum * connection.previousDeltaWeight;
      connection.eligibility += 1e-12;
    }

    this.bias += momentum * this.previousDeltaBias;
  }

  private _computeProjectedError(): number {
    let error = 0;

    for (const connection of this.connections.out) {
      error +=
        connection.to.error.responsibility *
        connection.weight *
        connection.gain;
    }

    return error;
  }

  private _computeGatedError(): number {
    let error = 0;

    for (const connection of this.connections.gated) {
      const node = connection.to;
      const influence =
        this._computeSelfGatedInfluence(node) +
        connection.weight * connection.from.activation;
      error += node.error.responsibility * influence;
    }

    return error;
  }

  private _computeSelfGatedInfluence(node: Node): number {
    return node.connections.self.reduce(
      (sum: number, selfConnection: Connection) =>
        sum + (selfConnection.gater === this ? node.old : 0),
      0,
    );
  }

  private _propagateConnections(
    connections: Connection[],
    rate: number,
    momentum: number,
    update: boolean,
    regularization:
      | number
      | { type: 'L1' | 'L2'; lambda: number }
      | ((weight: number) => number),
    isSelfConnection: boolean,
  ): void {
    for (const connection of connections) {
      if (connection.dcMask === 0) {
        connection.totalDeltaWeight += 0;
        continue;
      }

      const gradient = this._computeConnectionGradient(connection);
      const regularizationTerm = this._computeRegularizationTerm(
        connection.weight,
        regularization,
      );
      const deltaWeight = rate * (gradient * this.mask - regularizationTerm);

      this._applyConnectionDelta(
        connection,
        deltaWeight,
        momentum,
        update,
        isSelfConnection,
      );
    }
  }

  private _computeConnectionGradient(connection: Connection): number {
    let gradient = this.error.projected * connection.eligibility;

    for (
      let traceIndex = 0;
      traceIndex < connection.xtrace.nodes.length;
      traceIndex++
    ) {
      const node = connection.xtrace.nodes[traceIndex];
      const traceValue = connection.xtrace.values[traceIndex];
      gradient += node.error.responsibility * traceValue;
    }

    return gradient;
  }

  private _computeRegularizationTerm(
    weight: number,
    regularization:
      | number
      | { type: 'L1' | 'L2'; lambda: number }
      | ((weight: number) => number),
  ): number {
    if (typeof regularization === 'function') {
      return regularization(weight);
    }

    if (typeof regularization === 'object' && regularization !== null) {
      if (regularization.type === 'L1') {
        return regularization.lambda * Math.sign(weight);
      }
      if (regularization.type === 'L2') {
        return regularization.lambda * weight;
      }
      return 0;
    }

    return regularization * weight;
  }

  private _applyConnectionDelta(
    connection: Connection,
    deltaWeight: number,
    momentum: number,
    update: boolean,
    isSelfConnection: boolean,
  ): void {
    const nextDeltaWeight = this._sanitizeConnectionDeltaWeight(
      connection,
      deltaWeight,
      isSelfConnection,
    );
    connection.totalDeltaWeight = this._accumulateConnectionDeltaWeight(
      connection,
      nextDeltaWeight,
      isSelfConnection,
    );

    if (!update) {
      return;
    }

    const currentDeltaWeight = this._resolveCurrentConnectionDeltaWeight(
      connection,
      momentum,
      isSelfConnection,
    );
    this._applyValidatedConnectionWeight(
      connection,
      currentDeltaWeight,
      momentum,
      isSelfConnection,
    );
    connection.previousDeltaWeight = currentDeltaWeight;
    connection.totalDeltaWeight = 0;
  }

  private _sanitizeConnectionDeltaWeight(
    connection: Connection,
    deltaWeight: number,
    isSelfConnection: boolean,
  ): number {
    if (!Number.isFinite(deltaWeight)) {
      console.warn(
        this._resolveConnectionDeltaWarningMessage(
          isSelfConnection,
          'self deltaWeight is not finite, clamping to 0',
          'deltaWeight is not finite, clamping to 0',
        ),
        {
          node: this.index,
          connection,
          deltaWeight,
        },
      );
      return 0;
    }

    return this._clampAbsoluteValue(deltaWeight, 1e3);
  }

  private _accumulateConnectionDeltaWeight(
    connection: Connection,
    deltaWeight: number,
    isSelfConnection: boolean,
  ): number {
    const nextTotalDeltaWeight = connection.totalDeltaWeight + deltaWeight;

    if (!Number.isFinite(nextTotalDeltaWeight)) {
      console.warn(
        this._resolveConnectionDeltaWarningMessage(
          isSelfConnection,
          'self totalDeltaWeight became NaN/Infinity, resetting to 0',
          'totalDeltaWeight became NaN/Infinity, resetting to 0',
        ),
        { node: this.index, connection },
      );
      return 0;
    }

    return nextTotalDeltaWeight;
  }

  private _resolveCurrentConnectionDeltaWeight(
    connection: Connection,
    momentum: number,
    isSelfConnection: boolean,
  ): number {
    const currentDeltaWeight =
      connection.totalDeltaWeight + momentum * connection.previousDeltaWeight;

    if (!Number.isFinite(currentDeltaWeight)) {
      console.warn(
        this._resolveConnectionDeltaWarningMessage(
          isSelfConnection,
          'self currentDeltaWeight is not finite, clamping to 0',
          'currentDeltaWeight is not finite, clamping to 0',
        ),
        {
          node: this.index,
          connection,
          currentDeltaWeight,
        },
      );
      return 0;
    }

    return this._clampAbsoluteValue(currentDeltaWeight, 1e3);
  }

  private _applyValidatedConnectionWeight(
    connection: Connection,
    currentDeltaWeight: number,
    momentum: number,
    isSelfConnection: boolean,
  ): void {
    if (momentum > 0) {
      connection.weight -= momentum * connection.previousDeltaWeight;
    }

    connection.weight += currentDeltaWeight;
    if (!Number.isFinite(connection.weight)) {
      console.warn(
        this._resolveConnectionDeltaWarningMessage(
          isSelfConnection,
          'self weight update produced invalid value, resetting to 0',
          `Weight update produced invalid value: ${connection.weight}. Resetting to 0.`,
        ),
        { node: this.index, connection },
      );
      connection.weight = 0;
      return;
    }

    connection.weight = this._clampAbsoluteValue(connection.weight, 1e6);
  }

  private _resolveConnectionDeltaWarningMessage(
    isSelfConnection: boolean,
    selfConnectionMessage: string,
    connectionMessage: string,
  ): string {
    return isSelfConnection ? selfConnectionMessage : connectionMessage;
  }

  private _clampAbsoluteValue(
    value: number,
    maximumAbsoluteValue: number,
  ): number {
    if (Math.abs(value) > maximumAbsoluteValue) {
      return Math.sign(value) * maximumAbsoluteValue;
    }

    return value;
  }

  private _applyBiasDelta(
    rate: number,
    momentum: number,
    update: boolean,
  ): void {
    let deltaBias = rate * this.error.responsibility;
    if (!Number.isFinite(deltaBias)) {
      console.warn('deltaBias is not finite, clamping to 0', {
        node: this.index,
        deltaBias,
      });
      deltaBias = 0;
    } else if (Math.abs(deltaBias) > 1e3) {
      deltaBias = Math.sign(deltaBias) * 1e3;
    }

    this.totalDeltaBias += deltaBias;
    if (!Number.isFinite(this.totalDeltaBias)) {
      console.warn('totalDeltaBias became NaN/Infinity, resetting to 0', {
        node: this.index,
      });
      this.totalDeltaBias = 0;
    }

    if (!update) {
      return;
    }

    let currentDeltaBias =
      this.totalDeltaBias + momentum * this.previousDeltaBias;
    if (!Number.isFinite(currentDeltaBias)) {
      console.warn('currentDeltaBias is not finite, clamping to 0', {
        node: this.index,
        currentDeltaBias,
      });
      currentDeltaBias = 0;
    } else if (Math.abs(currentDeltaBias) > 1e3) {
      currentDeltaBias = Math.sign(currentDeltaBias) * 1e3;
    }

    if (momentum > 0) {
      this.bias -= momentum * this.previousDeltaBias;
    }

    this.bias += currentDeltaBias;
    if (!Number.isFinite(this.bias)) {
      console.warn('bias update produced invalid value, resetting to 0', {
        node: this.index,
      });
      this.bias = 0;
    } else if (Math.abs(this.bias) > 1e6) {
      this.bias = Math.sign(this.bias) * 1e6;
    }

    this.previousDeltaBias = currentDeltaBias;
    this.totalDeltaBias = 0;
  }

  /**
   * Converts the node's essential properties to a JSON object for serialization.
   * Does not include state, activation, error, or connection information, as these
   * are typically transient or reconstructed separately.
   * @returns A JSON representation of the node's configuration.
   */
  toJSON() {
    return {
      index: this.index,
      bias: this.bias,
      response: this.response,
      type: this.type,
      squash: this.squash ? this.squash.name : null,
      mask: this.mask,
    };
  }

  /**
   * Creates a Node instance from a JSON object.
   * @param json The JSON object containing node configuration.
   * @returns A new Node instance configured according to the JSON object.
   */
  static fromJSON(json: {
    bias: number;
    response?: number;
    type: string;
    squash: string;
    mask: number;
  }): Node {
    const node = new Node(json.type);
    node.bias = json.bias;
    node.response =
      typeof json.response === 'number' && Number.isFinite(json.response)
        ? json.response
        : NEUTRAL_NODE_RESPONSE;
    node.mask = json.mask;
    if (json.squash) {
      const squashFn =
        methods.Activation[json.squash as keyof typeof methods.Activation];
      if (typeof squashFn === 'function') {
        node.squash = squashFn as (x: number, derivate?: boolean) => number;
      } else {
        // Fallback to identity and log a warning
        console.warn(
          `fromJSON: Unknown or invalid squash function '${json.squash}' for node. Using identity.`,
        );
        node.squash = methods.Activation.identity;
      }
    }
    return node;
  }

  /**
   * Checks if this node is connected to another node.
   * @param target The target node to check the connection with.
   * @returns True if connected, otherwise false.
   */
  isConnectedTo(target: Node): boolean {
    return this.connections.out.some((conn) => conn.to === target);
  }

  /**
   * Applies a mutation method to the node. Used in neuro-evolution.
   *
   * This allows modifying the node's properties, such as its activation function or bias,
   * based on predefined mutation methods.
   *
   * @param method A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).
   * @throws {Error} If the mutation method is invalid, not provided, or not found in `methods.mutation`.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
   */
  mutate(method: unknown): void {
    // Validate the provided mutation method.
    if (!method) {
      throw new NodeMutationMethodRequiredError(
        'Mutation method cannot be null or undefined.',
      );
    }

    // Cast to a mutation method shape for internal usage
    const mutationMethod = method as NodeMutationMethodShape;

    assertKnownMutationMethod(mutationMethod);
    assertCanonicalMutationMethod(mutationMethod);

    const mutationHandler = resolveMutationHandler(mutationMethod);

    if (!mutationHandler) {
      throw new NodeUnsupportedMutationMethodError(
        `Unsupported mutation method: ${mutationMethod.name ?? 'undefined'}`,
      );
    }

    mutationHandler(this, mutationMethod);
  }

  /**
   * Creates a connection from this node to a target node or all nodes in a group.
   *
   * @param target The target Node or a group object containing a `nodes` array.
   * @param weight The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).
   * @returns An array containing the newly created Connection object(s).
   * @throws {Error} If the target is undefined.
   * @throws {Error} If trying to create a self-connection when one already exists (weight is not 0).
   */
  connect(target: Node | { nodes: Node[] }, weight?: number): Connection[] {
    const connections: Connection[] = [];
    if (!target) {
      throw new NodeUndefinedConnectionTargetError(
        'Cannot connect to an undefined target.',
      );
    }

    // Check if the target is a single Node.
    if ('bias' in target) {
      // Simple check if target looks like a Node instance.
      const targetNode = target as Node;
      if (targetNode === this) {
        // Handle self-connection. Only allow one self-connection.
        if (this.connections.self.length === 0) {
          const selfConnection = Connection.acquire(this, this, weight ?? 1);
          this.connections.self.push(selfConnection);
          connections.push(selfConnection);
        }
      } else {
        // Handle connection to a different node.
        const connection = Connection.acquire(this, targetNode, weight);
        // Add connection to the target's incoming list and this node's outgoing list.
        targetNode.connections.in.push(connection);
        this.connections.out.push(connection);

        connections.push(connection);
      }
    } else if ('nodes' in target && Array.isArray(target.nodes)) {
      // Handle connection to a group of nodes.
      for (const node of target.nodes) {
        // Create connection for each node in the group.
        const connection = Connection.acquire(this, node, weight);
        node.connections.in.push(connection);
        this.connections.out.push(connection);
        connections.push(connection);
      }
    } else {
      // Handle invalid target type.
      throw new NodeInvalidConnectionTargetTypeError(
        'Invalid target type for connection. Must be a Node or a group { nodes: Node[] }.',
      );
    }
    return connections;
  }

  /**
   * Removes the connection from this node to the target node.
   *
   * @param target The target node to disconnect from.
   * @param twosided If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.
   */
  disconnect(target: Node, twosided: boolean = false): void {
    // Handle self-connection disconnection.
    if (this === target) {
      // Remove all self-connections.
      this.connections.self = [];
      return;
    }

    // Filter out the connection to the target node from the outgoing list.
    this.connections.out = this.connections.out.filter((conn) => {
      if (conn.to === target) {
        // Remove the connection from the target's incoming list.
        target.connections.in = target.connections.in.filter(
          (inConn) => inConn !== conn, // Filter by reference.
        );
        // If the connection was gated, ungate it properly.
        if (conn.gater) {
          conn.gater.ungate(conn);
        }
        // Pooling deferred to higher-level network logic to ensure no stale references
        return false; // Remove from this.connections.out.
      }
      return true; // Keep other connections.
    });

    // If twosided is true, recursively call disconnect on the target node.
    if (twosided) {
      target.disconnect(this, false); // Pass false to avoid infinite recursion.
    }
  }

  /**
   * Makes this node gate the provided connection(s).
   * The connection's gain will be controlled by this node's activation value.
   *
   * @param connections A single Connection object or an array of Connection objects to be gated.
   */
  gate(connections: Connection | Connection[]): void {
    // Ensure connections is an array.
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      if (!connection || !connection.from || !connection.to) {
        console.warn('Attempted to gate an invalid or incomplete connection.');
        continue;
      }
      // Check if this node is already gating this connection.
      if (connection.gater === this) {
        console.warn('Node is already gating this connection.');
        continue;
      }
      // Check if the connection is already gated by another node.
      if (connection.gater !== null) {
        console.warn(
          'Connection is already gated by another node. Ungate first.',
        );
        // Optionally, automatically ungate from the previous gater:
        // connection.gater.ungate(connection);
        continue; // Skip gating if already gated by another.
      }

      // Add the connection to this node's list of gated connections.
      this.connections.gated.push(connection);
      // Set the gater property on the connection itself.
      connection.gater = this;
      // Gain will be updated during activation. Initialize?
      // connection.gain = this.activation; // Or 0? Or leave as is? Depends on desired initial state.
    }
  }

  /**
   * Removes this node's gating control over the specified connection(s).
   * Resets the connection's gain to 1 and removes it from the `connections.gated` list.
   *
   * @param connections A single Connection object or an array of Connection objects to ungate.
   */
  ungate(connections: Connection | Connection[]): void {
    // Ensure connections is an array.
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      if (!connection) continue; // Skip null/undefined entries

      // Find the connection in the gated list.
      const index = this.connections.gated.indexOf(connection);
      if (index !== -1) {
        // Remove from the gated list.
        this.connections.gated.splice(index, 1);
        // Reset the connection's gater property.
        connection.gater = null;
        // Reset the connection's gain to its default value (usually 1).
        connection.gain = 1;
      } else {
        // Optional: Warn if trying to ungate a connection not gated by this node.
        // console.warn("Attempted to ungate a connection not gated by this node, or already ungated.");
      }
    }
  }

  /**
   * Clears the node's dynamic state information.
   * Resets activation, state, previous state, error signals, and eligibility traces.
   * Useful for starting a new activation sequence (e.g., for a new input pattern).
   */
  clear(): void {
    // Reset eligibility traces for all incoming connections.
    for (const connection of this.connections.in) {
      connection.eligibility = 0;
      connection.xtrace = { nodes: [], values: [] };
    }
    // Also reset eligibility/xtrace for self-connections.
    for (const connection of this.connections.self) {
      connection.eligibility = 0;
      connection.xtrace = { nodes: [], values: [] };
    }
    // Reset gain for connections gated by this node to the neutral default (1).
    // Using 1 instead of 0 restores the same initial conditions as a fresh network
    // before any activation — a fresh connection's gain defaults to 1 via the accessor.
    for (const connection of this.connections.gated) {
      connection.gain = 1;
    }
    // Reset error values.
    this.error = { responsibility: 0, projected: 0, gated: 0 };
    // Reset state, activation, and old state.
    this.old = this.state = this.activation = 0;
    // Note: Does not reset bias, mask, or previousDeltaBias/totalDeltaBias as these
    // usually persist across activations or are handled by the training process.
  }

  /**
   * Checks if this node has a direct outgoing connection to the given node.
   * Considers both regular outgoing connections and the self-connection.
   *
   * @param node The potential target node.
   * @returns True if this node projects to the target node, false otherwise.
   */
  isProjectingTo(node: Node): boolean {
    // Check self-connection
    if (node === this && this.connections.self.length > 0) return true;
    // Compare by object identity to avoid stale index issues
    return this.connections.out.some((conn) => conn.to === node);
  }

  /**
   * Checks if the given node has a direct outgoing connection to this node.
   * Considers both regular incoming connections and the self-connection.
   *
   * @param node The potential source node.
   * @returns True if the given node projects to this node, false otherwise.
   */
  isProjectedBy(node: Node): boolean {
    // Check self-connection (only if weight is non-zero).
    if (node === this && this.connections.self.length > 0) return true;

    // Check regular incoming connections.
    return this.connections.in.some((conn) => conn.from === node);
  }

  /**
   * Applies accumulated batch updates to incoming and self connections and this node's bias.
   * Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
   * Resets accumulators after applying. Safe to call on every node type.
   * @param momentum Momentum factor (0 to disable)
   */
  applyBatchUpdates(momentum: number): void {
    return this.applyBatchUpdatesWithOptimizer({ type: 'sgd', momentum });
  }

  /**
   * Extended batch update supporting multiple optimizers.
   *
   * Applies accumulated (batch) gradients stored in `totalDeltaWeight` / `totalDeltaBias` to the
   * underlying weights and bias using the selected optimization algorithm. Supports both classic
   * SGD (with Nesterov-style momentum via preceding propagate logic) and a collection of adaptive
   * optimizers. After applying an update, gradient accumulators are reset to 0.
   *
   * Supported optimizers (type):
   *  - 'sgd'      : Standard gradient descent with optional momentum.
   *  - 'rmsprop'  : Exponential moving average of squared gradients (cache) to normalize step.
   *  - 'adagrad'  : Accumulate squared gradients; learning rate effectively decays per weight.
   *  - 'adam'     : Bias‑corrected first (m) & second (v) moment estimates.
   *  - 'adamw'    : Adam with decoupled weight decay (applied after adaptive step).
   *  - 'amsgrad'  : Adam variant maintaining a maximum of past v (vhat) to enforce non‑increasing step size.
   *  - 'adamax'   : Adam variant using the infinity norm (u) instead of second moment.
   *  - 'nadam'    : Adam + Nesterov momentum style update (lookahead on first moment).
   *  - 'radam'    : Rectified Adam – warms up variance by adaptively rectifying denominator when sample size small.
   *  - 'lion'     : Uses sign of combination of two momentum buffers (beta1 & beta2) for update direction only.
   *  - 'adabelief': Adam-like but second moment on (g - m) (gradient surprise) for variance reduction.
   *  - 'lookahead': Wrapper; performs k fast optimizer steps then interpolates (alpha) towards a slow (shadow) weight.
   *
   * Options:
   *  - momentum     : (SGD) momentum factor (Nesterov handled in propagate when update=true).
   *  - beta1/beta2  : Exponential decay rates for first/second moments (Adam family, Lion, AdaBelief, etc.).
   *  - eps          : Numerical stability epsilon added to denominator terms.
   *  - weightDecay  : Decoupled weight decay (AdamW) or additionally applied after main step when adamw selected.
   *  - lrScale      : Learning rate scalar already scheduled externally (passed as currentRate).
   *  - t            : Global step (1-indexed) for bias correction / rectification.
   *  - baseType     : Underlying optimizer for lookahead (not itself lookahead).
   *  - la_k         : Lookahead synchronization interval (number of fast steps).
   *  - la_alpha     : Interpolation factor towards slow (shadow) weights/bias at sync points.
   *
   * Internal per-connection temp fields (created lazily):
   *  - firstMoment / secondMoment / maxSecondMoment / infinityNorm : Moment / variance / max variance / infinity norm caches.
   *  - gradientAccumulator : Single accumulator (RMSProp / AdaGrad).
   *  - previousDeltaWeight : For classic SGD momentum.
   *  - lookaheadShadowWeight / _la_shadowBias : Lookahead shadow copies.
   *
   * Safety: We clip extreme weight / bias magnitudes and guard against NaN/Infinity.
   *
   * @param opts Optimizer configuration (see above).
   */
  applyBatchUpdatesWithOptimizer(opts: BatchOptimizerOptions): void {
    const optimizerPlan = this._resolveBatchOptimizerExecutionPlan(opts);
    const optimizerState = this as unknown as NodeOptimizerProps;

    this._initializeLookaheadState(optimizerPlan.type, optimizerState, opts);
    this._applyOptimizerToAllConnections(
      optimizerPlan.effectiveType,
      optimizerPlan.optimizerParams,
      optimizerPlan.weightDecay,
    );
    this._applyOptimizerToBiasIfEligible(
      optimizerPlan.effectiveType,
      optimizerPlan.optimizerParams,
      optimizerPlan.weightDecay,
    );
    this._applyLookaheadSyncIfNeeded(optimizerPlan.type);
  }

  private _resolveBatchOptimizerExecutionPlan(
    opts: BatchOptimizerOptions,
  ): ResolvedBatchOptimizerPlan {
    const type = opts.type ?? 'sgd';

    return {
      effectiveType: this._resolveEffectiveBatchOptimizerType(
        type,
        opts.baseType,
      ),
      optimizerParams: this._resolveBatchOptimizerHyperparams(opts),
      type,
      weightDecay: opts.weightDecay ?? 0,
    };
  }

  private _resolveEffectiveBatchOptimizerType(
    type: string,
    baseType: string | undefined,
  ): string {
    return type === 'lookahead' ? (baseType ?? 'sgd') : type;
  }

  private _resolveBatchOptimizerHyperparams(
    opts: BatchOptimizerOptions,
  ): OptimizerHyperparams {
    return {
      beta1: opts.beta1 ?? 0.9,
      beta2: opts.beta2 ?? 0.999,
      eps: opts.eps ?? 1e-8,
      lrScale: opts.lrScale ?? 1,
      momentum: opts.momentum ?? 0,
      t: Math.max(1, Math.floor(opts.t ?? 1)),
    };
  }

  private _applyOptimizerToAllConnections(
    effectiveType: string,
    optimizerParams: OptimizerHyperparams,
    weightDecay: number,
  ): void {
    this._applyOptimizerToConnections(
      this.connections.in,
      effectiveType,
      optimizerParams,
      weightDecay,
    );
    this._applyOptimizerToConnections(
      this.connections.self,
      effectiveType,
      optimizerParams,
      weightDecay,
    );
  }

  private _applyLookaheadSyncIfNeeded(type: string): void {
    if (type === 'lookahead') {
      this._applyLookaheadSync();
    }
  }

  private _initializeLookaheadState(
    type: string,
    optProps: NodeOptimizerProps,
    opts: {
      la_alpha?: number;
      la_k?: number;
    },
  ): void {
    if (type !== 'lookahead') {
      return;
    }

    optProps._la_k = optProps._la_k || opts.la_k || 5;
    optProps._la_alpha = optProps._la_alpha || opts.la_alpha || 0.5;
    optProps._la_step = (optProps._la_step || 0) + 1;
    if (!optProps._la_shadowBias) {
      optProps._la_shadowBias = this.bias;
    }
  }

  private _applyOptimizerToConnections(
    connections: Connection[],
    effectiveType: string,
    optimizerParams: OptimizerHyperparams,
    wd: number,
  ): void {
    for (const connection of connections) {
      this._applyOptimizerToConnection(
        connection,
        effectiveType,
        optimizerParams.momentum,
        optimizerParams.beta1,
        optimizerParams.beta2,
        optimizerParams.eps,
        wd,
        optimizerParams.lrScale,
        optimizerParams.t,
      );
    }
  }

  private _applyOptimizerToBiasIfEligible(
    effectiveType: string,
    optimizerParams: OptimizerHyperparams,
    wd: number,
  ): void {
    if (this.type === 'input' || this.type === 'constant') {
      this.previousDeltaBias = 0;
      this.totalDeltaBias = 0;
      return;
    }

    this._applyOptimizerToBias(
      effectiveType,
      optimizerParams.momentum,
      optimizerParams.beta1,
      optimizerParams.beta2,
      optimizerParams.eps,
      wd,
      optimizerParams.lrScale,
      optimizerParams.t,
    );
    this.totalDeltaBias = 0;
  }

  private _applyOptimizerToConnection(
    connection: Connection,
    effectiveType: string,
    momentum: number,
    beta1: number,
    beta2: number,
    eps: number,
    wd: number,
    lrScale: number,
    t: number,
  ): void {
    let gradient = connection.totalDeltaWeight || 0;
    if (!Number.isFinite(gradient)) {
      gradient = 0;
    }

    const optimizerStep = (
      OPTIMIZER_STEP_TABLE[effectiveType] ?? OPTIMIZER_STEP_TABLE.sgd
    )(gradient, createConnectionOptimizerCarrier(connection), {
      beta1,
      beta2,
      eps,
      lrScale,
      momentum,
      t,
    });

    this._safeUpdateWeight(connection, optimizerStep);

    if (effectiveType === 'adamw' && wd !== 0) {
      this._safeUpdateWeight(
        connection,
        -wd * (connection.weight || 0) * lrScale,
      );
    }

    connection.totalDeltaWeight = 0;
  }

  private _applyOptimizerToBias(
    effectiveType: string,
    momentum: number,
    beta1: number,
    beta2: number,
    eps: number,
    wd: number,
    lrScale: number,
    t: number,
  ): void {
    const optimizerState = this as unknown as NodeOptimizerProps;
    let biasGradient = this.totalDeltaBias || 0;
    if (!Number.isFinite(biasGradient)) {
      biasGradient = 0;
    }

    const optimizerStep = (
      OPTIMIZER_STEP_TABLE[effectiveType] ?? OPTIMIZER_STEP_TABLE.sgd
    )(biasGradient, createBiasOptimizerCarrier(this, optimizerState), {
      beta1,
      beta2,
      eps,
      lrScale,
      momentum,
      t,
    });

    this._safeUpdateBias(optimizerStep);

    if (effectiveType === 'adamw' && wd !== 0) {
      this._safeUpdateBias(-wd * (this.bias || 0) * lrScale);
    }
  }

  private _applyLookaheadSync(): void {
    const optimizerState = this as unknown as NodeOptimizerProps;
    const interval = optimizerState._la_k || 5;
    const alpha = optimizerState._la_alpha || 0.5;

    if ((optimizerState._la_step ?? 0) % interval !== 0) {
      return;
    }

    optimizerState._la_shadowBias =
      (1 - alpha) * (optimizerState._la_shadowBias ?? this.bias) +
      alpha * this.bias;
    this.bias = optimizerState._la_shadowBias;

    for (const connection of this.connections.in) {
      this._blendLookaheadConnection(connection, alpha);
    }
    for (const connection of this.connections.self) {
      this._blendLookaheadConnection(connection, alpha);
    }
  }

  private _blendLookaheadConnection(
    connection: Connection,
    alpha: number,
  ): void {
    if (!connection.lookaheadShadowWeight) {
      connection.lookaheadShadowWeight = connection.weight;
    }
    connection.lookaheadShadowWeight =
      (1 - alpha) * connection.lookaheadShadowWeight +
      alpha * connection.weight;
    connection.weight = connection.lookaheadShadowWeight;
  }

  /**
   * Internal helper to safely update a connection weight with clipping and NaN checks.
   */
  private _safeUpdateWeight(connection: Connection, delta: number) {
    let next = connection.weight + delta;
    if (!Number.isFinite(next)) next = 0;
    if (Math.abs(next) > 1e6) next = Math.sign(next) * 1e6;
    connection.weight = next;
  }

  /**
   * Internal helper to safely update the node bias with clipping and NaN checks.
   */
  private _safeUpdateBias(delta: number) {
    let next = this.bias + delta;
    if (!Number.isFinite(next)) next = 0;
    if (Math.abs(next) > 1e6) next = Math.sign(next) * 1e6;
    this.bias = next;
  }
}
