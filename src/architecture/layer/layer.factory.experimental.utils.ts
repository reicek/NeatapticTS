import Group from '../group/group';
import Node from '../node';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
} from './layer.utils.types';

const DEFAULT_CONV1D_STRIDE = 1;
const DEFAULT_CONV1D_PADDING = 0;
const SLICE_START_INDEX = 0;
const DEFAULT_ATTENTION_HEADS = 1;
const AVERAGE_INITIAL_VALUE = 0;

/**
 * Builds a lightweight Conv1D-style stub layer.
 *
 * This is an experimental placeholder: it stores Conv1D metadata and returns
 * either activated node outputs (no input values) or a bounded slice from
 * provided values. It does not perform real convolution math.
 *
 * Educational note: this is designed as an integration seam. It lets you
 * prototype graphs that *mention* Conv1D without requiring a full convolution
 * implementation yet.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of output nodes (filters).
 * @param kernelSize Size of the convolution kernel.
 * @param stride Stride of the convolution.
 * @param padding Padding size for the convolution.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const conv = buildConv1dLayer(factoryContext, 8, 3, 1, 1);
 *
 * // When called with values, the stub returns a slice (not a true convolution).
 * const out = conv.activate([10, 11, 12, 13]);
 * ```
 */
export function buildConv1dLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  kernelSize: number,
  stride: number = DEFAULT_CONV1D_STRIDE,
  padding: number = DEFAULT_CONV1D_PADDING,
): TLayer {
  const layer = createStubLayer(context, size);
  (
    layer as unknown as {
      conv1d: { kernelSize: number; stride: number; padding: number };
    }
  ).conv1d = { kernelSize, stride, padding };

  layer.activate = createConv1dActivator(layer, size);

  return layer;
}

/**
 * Builds a lightweight attention-style stub layer.
 *
 * This placeholder stores head count metadata and uses a simple averaging
 * behavior for provided values. It is useful as an integration seam while
 * full attention internals are still under development.
 *
 * Educational note: because this is a stub, "heads" are metadata only. The
 * activation behavior is intentionally simple: it collapses provided values to
 * their average.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of output nodes.
 * @param heads Number of attention heads.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const attention = buildAttentionLayer(factoryContext, 8, 4);
 * const out = attention.activate([1, 2, 3, 4]);
 * // out is length 8, every entry is the average (2.5)
 * ```
 */
export function buildAttentionLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  heads: number = DEFAULT_ATTENTION_HEADS,
): TLayer {
  const layer = createStubLayer(context, size);
  (layer as unknown as { attention: { heads: number } }).attention = { heads };

  layer.activate = createAttentionActivator(layer, size);

  return layer;
}

/**
 * Creates shared node/output scaffolding for experimental layers.
 *
 * Centralizing this setup keeps the experimental builders focused on their
 * metadata and activation behavior.
 *
 * Implementation detail: the returned layer contains a `nodes` list (for basic
 * activation) and an `output` group (for API compatibility with the rest of the
 * architecture). In these stubs, the output group is not intended to be a fully
 * wired projection of `nodes`.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of output nodes to allocate.
 * @returns Initialized experimental layer.
 */
function createStubLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = context.createLayer();
  layer.nodes = Array.from({ length: size }, () => new Node());
  layer.output = new Group(size);
  return layer;
}

/**
 * Builds the activation function used by the Conv1D stub.
 *
 * When values are provided, this function returns a length-bounded slice.
 * When values are omitted, it delegates to node activation.
 *
 * This keeps the call signature compatible with real layers while remaining
 * intentionally cheap.
 *
 * @param layer Layer whose nodes can self-activate.
 * @param size Number of output values to return.
 * @returns Activation callback for Conv1D behavior.
 *
 * Example:
 *
 * ```ts
 * const activate = createConv1dActivator(layer, 3);
 * activate([9, 8, 7, 6]); // -> [9, 8, 7]
 * ```
 */
function createConv1dActivator<TLayer extends LayerFactoryLayer>(
  layer: TLayer,
  size: number,
): (values?: number[]) => number[] {
  return (values?: number[]): number[] => {
    if (!values) {
      return activateStubNodes(layer);
    }

    return values.slice(SLICE_START_INDEX, size);
  };
}

/**
 * Builds the activation function used by the attention stub.
 *
 * When values are provided, all outputs are filled with their average.
 * When values are omitted, it delegates to node activation.
 *
 * This behavior is *not* meant to represent real attention math; it simply
 * produces a stable, shape-correct output while attention internals evolve.
 *
 * @param layer Layer whose nodes can self-activate.
 * @param size Number of output values to return.
 * @returns Activation callback for attention behavior.
 *
 * Example:
 *
 * ```ts
 * const activate = createAttentionActivator(layer, 4);
 * activate([1, 3]); // -> [2, 2, 2, 2]
 * ```
 */
function createAttentionActivator<TLayer extends LayerFactoryLayer>(
  layer: TLayer,
  size: number,
): (values?: number[]) => number[] {
  return (values?: number[]): number[] => {
    if (!values) {
      return activateStubNodes(layer);
    }

    const averageValue =
      values.reduce((sum, value) => sum + value, AVERAGE_INITIAL_VALUE) /
      values.length;
    return Array(size).fill(averageValue);
  };
}

/**
 * Activates all nodes in a stub layer and returns their outputs.
 *
 * This helper keeps fallback activation behavior identical across experimental
 * layer variants.
 *
 * @param layer Layer containing nodes to activate.
 * @returns Activated node outputs.
 *
 * Example:
 *
 * ```ts
 * const outputs = activateStubNodes(layer);
 * ```
 */
function activateStubNodes<TLayer extends LayerFactoryLayer>(
  layer: TLayer,
): number[] {
  return layer.nodes.map((node) => node.activate());
}
