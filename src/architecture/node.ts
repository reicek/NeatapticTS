/**
 * Stable public entry point for the `Node` primitive.
 *
 * A `Node` is the fundamental computational unit in NeatapticTS: one neuron
 * that aggregates weighted inputs, applies a non-linear activation function
 * (`squash`), and emits an activation value.
 *
 * The runtime role (`type`: `'input'` | `'hidden'` | `'output'`) controls
 * execution semantics. An optional descriptor surface (`label`, `intent`,
 * `metadata`) keeps architectural meaning visible for tooling without
 * affecting activation math.
 *
 * The concrete implementation, activation/propagation methods, and gating
 * helpers all live in `node/node.ts`. This root file is a stable facade.
 *
 * @example
 * ```ts
 * import Node from './node';
 *
 * const hidden = new Node('hidden');
 * hidden.describe({ label: 'encoder', intent: 'hidden' });
 * ```
 */
export { default, type PrimitiveNodeType } from './node/node';
export {
  type PrimitiveDescriptor,
  type PrimitiveIntent,
  type PrimitiveMetadata,
  type PrimitiveMetadataValue,
  resolvePrimitiveIntent,
} from './node/node';
