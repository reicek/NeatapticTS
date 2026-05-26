/**
 * Stable public entry point for the `Group` composite primitive.
 *
 * A `Group` is a named cluster of neurons that can be activated, propagated,
 * wired, and gated as one unit. It is the first place in the architecture
 * hierarchy where you interact with blocks of nodes rather than individual
 * neurons.
 *
 * Typical usage:
 *
 * - Dense connection building between regions (`group.connect(other, method)`).
 * - Collective activation and propagation for block-shaped motifs.
 * - Recurrent and gated substructures where node-level behavior is still
 *   needed but orchestration should stay above the single-neuron level.
 *
 * The full implementation lives in `group/group.ts`. This root file is a
 * stable facade that keeps the architecture import path short.
 *
 * @example
 * ```ts
 * import Group from './group';
 * import * as methods from '../methods/methods';
 *
 * const encoder = new Group(8, 'hidden');
 * const readout = new Group(2, 'output');
 * encoder.connect(readout, methods.groupConnection.ALL_TO_ALL);
 * ```
 */
export { default } from './group/group';
