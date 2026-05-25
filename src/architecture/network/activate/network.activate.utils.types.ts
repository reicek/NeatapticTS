import type Network from '../../network/network';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type { ActivateNetworkInternals as NetworkInternals } from '../network.types';

/**
 * Node role label used by activation traversal to identify input neurons.
 * Input nodes do not aggregate incoming connections; they read directly from the input vector.
 */
export const INPUT_NODE_TYPE = 'input';

/**
 * Node role label used by activation traversal to identify output neurons.
 * Output nodes write their activation into the result array at the slot matching their ordered position.
 */
export const OUTPUT_NODE_TYPE = 'output';

/**
 * Training flag value used by no-trace fast-slab eligibility checks.
 * The slab fast path requires that no gradient traces are accumulated, so this literal
 * (`false`) is the only value that passes the eligibility predicate.
 */
export const NO_TRACE_FAST_SLAB_TRAINING_FLAG = false;

/**
 * Starting write index used when collecting output activations into the result buffer.
 * The output-collection loop increments from this value, writing one activation per slot.
 */
export const INITIAL_OUTPUT_WRITE_INDEX = 0;

/**
 * Increment applied after writing one output activation value into the result buffer.
 * Using an explicit constant (rather than `++`) keeps the protocol visible and testable.
 */
export const OUTPUT_WRITE_INDEX_INCREMENT = 1;

/**
 * Fallback text rendered when the actual input length is `undefined` inside an activation error message.
 * Prevents `'undefined'` from appearing as a raw JS coercion artifact in user-facing error strings.
 */
export const UNDEFINED_INPUT_LENGTH_TEXT = 'undefined';

/**
 * Hard limit on recursive activation depth used by the raw (non-slab) activation path.
 * Prevents unbounded recursion on networks with deep or cyclic structure when the
 * caller does not supply an explicit `maxActivationDepth` argument.
 */
export const DEFAULT_MAX_ACTIVATION_DEPTH = 1000;

/**
 * Error message thrown when `activateBatch` receives a non-array collection as its top-level argument.
 * Kept as a named constant so it can be matched in tests without coupling to a raw string literal.
 */
export const BATCH_INPUTS_COLLECTION_ERROR_MESSAGE =
  'inputs must be an array of input arrays';

/**
 * Type of the pooled activation output array acquired from the shared activation array pool.
 * Using the pool avoids per-call allocation in tight inference loops.
 */
export type ActivationOutputBuffer = ReturnType<
  typeof activationArrayPool.acquire
>;

/**
 * Shared context passed through the no-trace activation orchestration pipeline.
 * Collecting these fields into one object avoids repeating the same four arguments
 * across every helper in the activation chapter.
 */
export type NoTraceActivationContext = {
  network: Network;
  networkInternal: NetworkInternals;
  inputVector: number[];
  expectedInputSize: number;
};

/**
 * Shared context passed to node-traversal helpers during a no-trace activation pass.
 * Contains the subset of orchestration state needed to write one node's output into
 * the pooled result buffer.
 */
export type NoTraceNodeTraversalContext = {
  network: Network;
  inputVector: number[];
  pooledOutputBuffer: ActivationOutputBuffer;
};

/**
 * Shared activation state for a single node during no-trace traversal, carrying the accumulated input map and the current network node reference.
 */
export type SingleNodeNoTraceActivationContext = {
  inputValuesByNodeId: Map<number, number>;
  networkNode: NoTraceNodeTraversalContext['network']['nodes'][number];
};

/**
 * Shared orchestration state for the raw (non-slab) activation path, carrying the network internals and the caller-supplied input vector.
 */
export type RawActivationContext = {
  networkInternal: NetworkInternals;
  inputVector: number[];
  isTraining: boolean;
  maximumActivationDepth: number;
};

/**
 * Shared orchestration state for batch activation, carrying the internal network, the full batch input array, expected input size, and training flag.
 */
export type BatchActivationContext = {
  networkInternal: NetworkInternals;
  batchInputs: number[][];
  expectedInputSize: number;
  isTraining: boolean;
};

/**
 * Shared state used while validating and activating one row from a batch input collection,
 * carrying the input slice, its position index within the batch, and the expected input size for validation.
 */
export type BatchRowActivationContext = {
  networkInternal: NetworkInternals;
  inputVector: number[];
  batchIndex: number;
  expectedInputSize: number;
  isTraining: boolean;
};

/**
 * Runtime network view used by the object-graph activation pipeline.
 *
 * This intentionally describes the internal fields activation reads and writes
 * while orchestrating scheduling, RNG use, regularization, and slab fast-path hooks.
 */
export type ActivateRuntimeNetworkProps = {
  _enforceAcyclic?: boolean;
  _topoDirty: boolean;
  _computeTopoOrder: () => void;
  _canUseFastSlab: (training: boolean) => boolean;
  _fastSlabActivate: (input: number[]) => number[];
  _reuseSequenceBuffers?: boolean;
  _sequenceOutputRing?: number[][];
  _sequenceOutputRingIndex?: number;
  _rand: () => number;
  _trainingStep: number;
  _lastStats?: unknown;
  _weightNoiseStd: number;
  _weightNoisePerHidden: number[];
  _weightNoiseSchedule?: (step: number) => number;
  _wnOrig?: number[];
  _stochasticDepth: number[];
  _stochasticDepthSchedule?: (step: number, current: number[]) => number[];
  _dropConnectProb: number;
};

/**
 * Layer container type derived from the network's optional layers array for use by layered activation paths.
 */
export type NetworkLayer = NonNullable<Network['layers']>[number];

/**
 * Node collection type derived from one network layer, used by layered dropout and stochastic-depth traversal helpers.
 */
export type NetworkLayerNodes = NetworkLayer['nodes'];

/**
 * Weight-noise telemetry collected during a single activation pass, capturing perturbation count, absolute sum, maximum, and mean magnitude.
 */
export interface WeightNoiseStats {
  count: number;
  sumAbs: number;
  maxAbs: number;
  meanAbs: number;
}

/**
 * Activation telemetry collected during a single forward pass, tracking dropped nodes, skipped layers, dropped connections, and weight-noise statistics.
 */
export interface ActivationStats {
  droppedHiddenNodes: number;
  totalHiddenNodes: number;
  droppedConnections: number;
  totalConnections: number;
  skippedLayers: number[];
  weightNoise: WeightNoiseStats;
}

/**
 * Marker interface returned by the weight-noise application helper to signal whether noise was applied and whether a restore pass is needed.
 */
export interface WeightNoiseApplyResult {
  appliedWeightNoise: boolean;
}
