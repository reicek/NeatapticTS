import type Connection from '../connection/connection';
import type Group from '../group';
import type Node from '../node';

/**
 * Minimal state required to run layer activation helpers.
 *
 * `dropout` is layer-level dropout probability, while `nodes` holds the
 * activation units that will be read/written during forward activation.
 *
 * Example:
 *
 * ```ts
 * const activationContext: LayerActivationContext = {
 *   nodes: layer.nodes,
 *   dropout: 0.2,
 * };
 *
 * // activateLayer(activationContext, values, true)
 * ```
 */
export type LayerActivationContext = {
  nodes: Node[];
  dropout: number;
};

/**
 * Minimal state required to run backpropagation helpers.
 *
 * Helpers only need access to the node sequence to propagate in reverse order.
 *
 * Example:
 *
 * ```ts
 * const propagationContext: LayerPropagationContext = { nodes: layer.nodes };
 *
 * // propagateLayer(propagationContext, 0.3, 0.1, targets)
 * ```
 */
export type LayerPropagationContext = {
  nodes: Node[];
};

/**
 * Structural contract for "layer-like" objects used in utility wiring.
 *
 * This type avoids direct class coupling while preserving the behaviors needed
 * by connection helpers (`input`, `nodes`, and `output`).
 *
 * Example:
 *
 * ```ts
 * // A real layer class typically satisfies this shape.
 * const layerLike: LayerLike = {
 *   input: (from) => [],
 *   nodes: [],
 *   output: null,
 * };
 * ```
 */
export type LayerLike = {
  input: (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ) => Connection[];
  nodes: Node[];
  output: Group | null;
};

/**
 * Context bundle required by connection/disconnection orchestration.
 *
 * It packages raw connection arrays, a layer type guard, and the active layer
 * references so helpers remain pure and testable.
 *
 * Example:
 *
 * ```ts
 * const connectionContext: LayerConnectionContext = {
 *   connections: { in: [], out: [], self: [] },
 *   isLayer: (value): value is LayerLike =>
 *     !!value && typeof (value as LayerLike).input === 'function',
 *   layer: someLayerLike,
 *   nodes: someLayerLike.nodes,
 *   output: someLayerLike.output,
 * };
 * ```
 */
export type LayerConnectionContext = {
  connections: { in: Connection[]; out: Connection[]; self: Connection[] };
  isLayer: (value: unknown) => value is LayerLike;
  layer: LayerLike;
  nodes: Node[];
  output: Group | null;
};

/**
 * Public layer surface required by factory construction helpers.
 *
 * Factories only depend on activation/input wiring and output/node containers.
 *
 * Example:
 *
 * ```ts
 * // Factory builders create an object that has these members.
 * // (Concrete layer classes typically provide many more helpers.)
 * ```
 */
export type LayerFactoryLayer = {
  activate: (values?: number[], training?: boolean) => number[];
  input: (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ) => Connection[];
  nodes: Node[];
  output: Group | null;
};

/**
 * Generic factory context used to create layers without circular imports.
 *
 * `createLayer` builds the target instance, and `isLayer` enables structural
 * narrowing whenever helpers accept mixed layer/group values.
 *
 * Example:
 *
 * ```ts
 * const factoryContext: LayerFactoryContext<MyLayer> = {
 *   createLayer: () => new MyLayer(),
 *   isLayer: (value): value is LayerLike =>
 *     !!value && typeof (value as LayerLike).input === 'function',
 * };
 * ```
 */
export type LayerFactoryContext<TLayer extends LayerFactoryLayer> = {
  createLayer: () => TLayer;
  isLayer: (value: unknown) => value is LayerLike;
};
