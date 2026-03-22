import type Connection from '../connection/connection';
import Group from '../group';
import * as methods from '../../methods/methods';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerLike,
} from './layer.utils.types';

const DEFAULT_GROUP_CONNECTION_METHOD = methods.groupConnection.ALL_TO_ALL;

/**
 * Builds a standard dense (fully connected) layer.
 *
 * The produced layer exposes its `Group` as `layer.output` and defines
 * `layer.input(...)` so callers can connect a source group or layer using
 * the selected connection strategy.
 *
 * This helper is intentionally "low ceremony": it does not decide *where* the
 * layer is used in a network. It only creates nodes, creates the output group,
 * and provides an `input(...)` function so external code can wire it.
 *
 * @param context - Factory helpers for constructing the layer instance.
 * @param size - Number of nodes to create in the dense layer.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * // Create a dense layer with 8 nodes.
 * const dense = buildDenseLayer(factoryContext, 8);
 *
 * // Wire: previous -> dense (the dense layer decides how to accept inputs).
 * dense.input(previousLayerLike);
 * ```
 */
export function buildDenseLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = context.createLayer();
  const block = new Group(size);

  layer.nodes.push(...block.nodes);
  layer.output = block;

  layer.input = (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ): Connection[] => {
    const sourceGroup = context.isLayer(from) ? from.output! : from;
    const resolvedMethod = method ?? DEFAULT_GROUP_CONNECTION_METHOD;
    return sourceGroup.connect(block, resolvedMethod, weight);
  };

  return layer;
}
