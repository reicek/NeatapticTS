import type {
  LayerActivationContext,
  LayerConnectionContext,
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerPropagationContext,
} from './layer.utils.types';
import {
  acquireActivationOutput,
  applyLayerMask,
  assertActivationInputSize,
  cloneActivationOutput,
  fillActivationOutput,
  releaseActivationOutput,
  resolveLayerMask,
} from './layer.activation.utils';
import {
  clearLayer as runLayerClear,
  connectLayer as runLayerConnect,
  disconnectLayer as runLayerDisconnect,
  gateLayer as runLayerGate,
  inputLayer as runLayerInput,
} from './layer.connection.utils';
import {
  assertTargetInputSize,
  propagateNodesInReverse,
} from './layer.propagation.utils';
import { buildDenseLayer } from './layer.factory.core.utils';
import {
  buildGruLayer,
  buildLstmLayer,
  buildMemoryLayer,
} from './layer.factory.recurrent.utils';
import {
  buildBatchNormLayer,
  buildLayerNormLayer,
} from './layer.factory.normalization.utils';
import {
  buildAttentionLayer,
  buildConv1dLayer,
} from './layer.factory.experimental.utils';

const DEFAULT_TRAINING_MODE = false;
const DEFAULT_TWO_SIDED_DISCONNECT = false;
const DEFAULT_CONV1D_STRIDE = 1;
const DEFAULT_CONV1D_PADDING = 0;
const DEFAULT_ATTENTION_HEADS = 1;

/**
 * Orchestrates layer activation behavior with a high-level flow.
 *
 * This is the recommended entry point for forward activation when you already
 * have a `LayerActivationContext`. It handles:
 * 1) Input validation
 * 2) Layer-level dropout masking (when `training` is true)
 * 3) Activation into a pooled buffer
 * 4) Cloning into a stable array for the caller
 *
 * Examples:
 *
 * ```ts
 * // Typical usage: activate without explicit per-node inputs.
 * const output = activateLayer({ nodes: layer.nodes, dropout: layer.dropout });
 *
 * // Explicit inputs: one number per node.
 * const output2 = activateLayer(
 *   { nodes: layer.nodes, dropout: layer.dropout },
 *   [0.1, 0.2, 0.3],
 *   true,
 * );
 * ```
 *
 * @param context - The layer state needed for activation.
 * @param values - Optional activation values to set per node.
 * @param training - Whether to apply dropout masking for training.
 * @returns A cloned array of activation values.
 */
export function activateLayer(
  context: LayerActivationContext,
  values?: number[],
  training: boolean = DEFAULT_TRAINING_MODE,
): number[] {
  const { nodes, dropout } = context;

  // Step 1: Validate inputs before touching node state.
  assertActivationInputSize(nodes.length, values);

  // Step 2: Apply a layer-wide mask for dropout if needed.
  const layerMask = resolveLayerMask(dropout, training);
  applyLayerMask(nodes, layerMask);

  // Step 3: Activate nodes and write outputs into the pooled buffer.
  const pooledOutput = acquireActivationOutput(nodes.length);
  fillActivationOutput(nodes, values, pooledOutput);

  // Step 4: Return a stable copy and release pooled memory.
  const clonedOutput = cloneActivationOutput(pooledOutput);
  releaseActivationOutput(pooledOutput);
  return clonedOutput;
}

/**
 * Orchestrates layer backpropagation behavior with a high-level flow.
 *
 * If `targets` is provided, it must be one value per node (output layer).
 * If omitted, propagation behaves like a hidden layer.
 *
 * Examples:
 *
 * ```ts
 * // Hidden layer propagation.
 * propagateLayer({ nodes: layer.nodes }, 0.3, 0.1);
 *
 * // Output layer propagation.
 * propagateLayer({ nodes: layer.nodes }, 0.3, 0.1, [1, 0, 0]);
 * ```
 *
 * @param context - The layer state needed for propagation.
 * @param rate - The learning rate for weight updates.
 * @param momentum - The momentum factor for smoothing updates.
 * @param targets - Optional target values for output layers.
 */
export function propagateLayer(
  context: LayerPropagationContext,
  rate: number,
  momentum: number,
  targets?: number[],
): void {
  const { nodes } = context;

  // Step 1: Validate target size before propagating.
  assertTargetInputSize(nodes.length, targets);

  // Step 2: Propagate errors in reverse order to match original behavior.
  propagateNodesInReverse(context, rate, momentum, targets);
}

/**
 * Orchestrates layer connection behavior with a high-level flow.
 *
 * This is a small wrapper around the focused helper in
 * `layer.connection.utils.ts`, kept here so the public layer API stays compact.
 *
 * Example:
 *
 * ```ts
 * connectLayer(layerConnectionContext, nextLayerLike);
 * ```
 *
 * @param context - The layer state needed for connections.
 * @param target - The layer, group, or node to connect to.
 * @param method - Optional connection method override.
 * @param weight - Optional fixed weight to apply.
 * @returns The created connection list.
 */
export function connectLayer(
  context: LayerConnectionContext,
  target: Parameters<typeof runLayerConnect>[1],
  method?: unknown,
  weight?: number,
): ReturnType<typeof runLayerConnect> {
  // Step 1: Delegate connection creation to the focused helper implementation.
  return runLayerConnect(context, target, method, weight);
}

/**
 * Orchestrates layer gating behavior with a high-level flow.
 *
 * Example:
 *
 * ```ts
 * gateLayer(layerConnectionContext, someConnections, method);
 * ```
 *
 * @param context - The layer state needed for gating.
 * @param connections - The connections to gate.
 * @param method - The gating method.
 */
export function gateLayer(
  context: LayerConnectionContext,
  connections: Parameters<typeof runLayerGate>[1],
  method: unknown,
): void {
  // Step 1: Delegate gating to the focused helper implementation.
  runLayerGate(context, connections, method);
}

/**
 * Orchestrates layer input wiring with a high-level flow.
 *
 * Example:
 *
 * ```ts
 * inputLayer(layerConnectionContext, previousLayerLike);
 * ```
 *
 * @param context - The layer state needed for input wiring.
 * @param from - The source layer or group.
 * @param method - Optional connection method override.
 * @param weight - Optional fixed weight to apply.
 * @returns The created connection list.
 */
export function inputLayer(
  context: LayerConnectionContext,
  from: Parameters<typeof runLayerInput>[1],
  method?: unknown,
  weight?: number,
): ReturnType<typeof runLayerInput> {
  // Step 1: Delegate input wiring to the focused helper implementation.
  return runLayerInput(context, from, method, weight);
}

/**
 * Orchestrates disconnection behavior with a high-level flow.
 *
 * Example:
 *
 * ```ts
 * disconnectLayer(layerConnectionContext, someGroup, true);
 * ```
 *
 * @param context - The layer state needed for disconnecting.
 * @param target - The group or node to disconnect.
 * @param twoSided - Whether to remove reciprocal connections as well.
 */
export function disconnectLayer(
  context: LayerConnectionContext,
  target: Parameters<typeof runLayerDisconnect>[1],
  twoSided: boolean = DEFAULT_TWO_SIDED_DISCONNECT,
): void {
  // Step 1: Delegate disconnection to the focused helper implementation.
  runLayerDisconnect(context, target, twoSided);
}

/**
 * Orchestrates clearing node activation state with a high-level flow.
 *
 * @param context - The layer state needed to reset nodes.
 * Example:
 *
 * ```ts
 * clearLayer(layerConnectionContext);
 * ```
 */
export function clearLayer(context: LayerConnectionContext): void {
  // Step 1: Delegate clearing to the focused helper implementation.
  runLayerClear(context);
}

/**
 * Orchestrates dense layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of nodes in the dense layer.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const dense = createDenseLayer(factoryContext, 8);
 * ```
 */
export function createDenseLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  // Step 1: Delegate dense layer creation to the focused helper implementation.
  return buildDenseLayer(context, size);
}

/**
 * Orchestrates LSTM layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of units in the LSTM layer.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const lstm = createLstmLayer(factoryContext, 8);
 * ```
 */
export function createLstmLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  // Step 1: Delegate LSTM layer creation to the focused helper implementation.
  return buildLstmLayer(context, size);
}

/**
 * Orchestrates GRU layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of units in the GRU layer.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const gru = createGruLayer(factoryContext, 8);
 * ```
 */
export function createGruLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  // Step 1: Delegate GRU layer creation to the focused helper implementation.
  return buildGruLayer(context, size);
}

/**
 * Orchestrates Memory layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of nodes in each memory block.
 * @param memory - Number of time steps to remember.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const memoryLayer = createMemoryLayer(factoryContext, 4, 3);
 * ```
 */
export function createMemoryLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  memory: number,
): TLayer {
  // Step 1: Delegate Memory layer creation to the focused helper implementation.
  return buildMemoryLayer(context, size, memory);
}

/**
 * Orchestrates batch normalization layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of nodes in the normalization layer.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const batchNorm = createBatchNormLayer(factoryContext, 16);
 * ```
 */
export function createBatchNormLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  // Step 1: Delegate batch norm layer creation to the focused helper implementation.
  return buildBatchNormLayer(context, size);
}

/**
 * Orchestrates layer normalization layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of nodes in the normalization layer.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const layerNorm = createLayerNormLayer(factoryContext, 16);
 * ```
 */
export function createLayerNormLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  // Step 1: Delegate layer norm creation to the focused helper implementation.
  return buildLayerNormLayer(context, size);
}

/**
 * Orchestrates 1D convolution layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of output nodes.
 * @param kernelSize - Size of the convolution kernel.
 * @param stride - Stride of the convolution.
 * @param padding - Padding size for the convolution.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const conv1d = createConv1dLayer(factoryContext, 8, 3);
 * ```
 */
export function createConv1dLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  kernelSize: number,
  stride: number = DEFAULT_CONV1D_STRIDE,
  padding: number = DEFAULT_CONV1D_PADDING,
): TLayer {
  // Step 1: Delegate conv1d creation to the focused helper implementation.
  return buildConv1dLayer(context, size, kernelSize, stride, padding);
}

/**
 * Orchestrates attention layer creation with a high-level flow.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of output nodes.
 * @param heads - Number of attention heads.
 * @returns The configured layer instance.
 * Example:
 *
 * ```ts
 * const attention = createAttentionLayer(factoryContext, 8, 4);
 * ```
 */
export function createAttentionLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  heads: number = DEFAULT_ATTENTION_HEADS,
): TLayer {
  // Step 1: Delegate attention creation to the focused helper implementation.
  return buildAttentionLayer(context, size, heads);
}
