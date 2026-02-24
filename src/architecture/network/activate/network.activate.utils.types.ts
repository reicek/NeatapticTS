import type Network from '../../network';
import { activationArrayPool } from '../../activationArrayPool';
import type { ActivateNetworkInternals as NetworkInternals } from '../network.types';

/**
 * Node role label used by activation traversal for input neurons.
 */
export const INPUT_NODE_TYPE = 'input';

/**
 * Node role label used by activation traversal for output neurons.
 */
export const OUTPUT_NODE_TYPE = 'output';

/**
 * Training flag value used by no-trace fast slab eligibility checks.
 */
export const NO_TRACE_FAST_SLAB_TRAINING_FLAG = false;

/**
 * Initial write index used when collecting output activations.
 */
export const INITIAL_OUTPUT_WRITE_INDEX = 0;

/**
 * Increment applied after writing one output activation value.
 */
export const OUTPUT_WRITE_INDEX_INCREMENT = 1;

/**
 * Fallback text for undefined input lengths when formatting validation errors.
 */
export const UNDEFINED_INPUT_LENGTH_TEXT = 'undefined';

/**
 * Default hard limit for recursive activation depth in raw activation mode.
 */
export const DEFAULT_MAX_ACTIVATION_DEPTH = 1000;

/**
 * Error message used when batch activation receives a non-array container.
 */
export const BATCH_INPUTS_COLLECTION_ERROR_MESSAGE =
  'inputs must be an array of input arrays';

/**
 * Pooled activation output array type acquired from the shared activation array pool.
 */
export type ActivationOutputBuffer = ReturnType<
  typeof activationArrayPool.acquire
>;

/**
 * Shared state used by no-trace activation orchestration and helpers.
 */
export type NoTraceActivationContext = {
  network: Network;
  networkInternal: NetworkInternals;
  inputVector: number[];
  expectedInputSize: number;
};

/**
 * Shared state used for node traversal during no-trace activation.
 */
export type NoTraceNodeTraversalContext = {
  networkNodes: Network['nodes'];
  inputVector: number[];
  pooledOutputBuffer: ActivationOutputBuffer;
};

/**
 * Shared state used while activating one node during no-trace traversal.
 */
export type SingleNodeNoTraceActivationContext = {
  networkNode: NoTraceNodeTraversalContext['networkNodes'][number];
  nodeIndex: number;
  inputVector: NoTraceNodeTraversalContext['inputVector'];
  pooledOutputBuffer: ActivationOutputBuffer;
  outputWriteIndex: number;
};

/**
 * Shared state used by raw activation orchestration.
 */
export type RawActivationContext = {
  networkInternal: NetworkInternals;
  inputVector: number[];
  isTraining: boolean;
  maximumActivationDepth: number;
};

/**
 * Shared state used by batch activation orchestration.
 */
export type BatchActivationContext = {
  networkInternal: NetworkInternals;
  batchInputs: number[][];
  expectedInputSize: number;
  isTraining: boolean;
};

/**
 * Shared state used while validating and activating one row in a batch.
 */
export type BatchRowActivationContext = {
  networkInternal: NetworkInternals;
  inputVector: number[];
  batchIndex: number;
  expectedInputSize: number;
  isTraining: boolean;
};

/**
 * Runtime internals consumed by the core activation helper orchestration.
 */
export interface ActivateRuntimeNetworkProps {
  _enforceAcyclic: boolean;
  _topoDirty: boolean;
  _computeTopoOrder(): void;
  _canUseFastSlab(training: boolean): boolean;
  _fastSlabActivate(input: number[]): number[];
  _weightNoiseStd: number;
  _weightNoisePerHidden: number[];
  _weightNoiseSchedule?(step: number): number;
  _trainingStep: number;
  _rand(): number;
  _stochasticDepth: number[];
  _stochasticDepthSchedule?(step: number, current: number[]): number[];
  _wnOrig?: number[];
  _dropConnectProb: number;
  _lastStats: unknown;
}

/**
 * Aggregated statistics produced by one activation pass.
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
 * Summary metrics captured for temporary weight-noise effects.
 */
export interface WeightNoiseStats {
  count: number;
  sumAbs: number;
  maxAbs: number;
  meanAbs: number;
}

/**
 * Return contract for weight-noise application helper.
 */
export interface WeightNoiseApplyResult {
  appliedWeightNoise: boolean;
}

/**
 * Non-null layer item type extracted from the optional layered network definition.
 */
export type NetworkLayer = NonNullable<Network['layers']>[number];

/**
 * Node list type for one explicit layer.
 */
export type NetworkLayerNodes = NetworkLayer['nodes'];
