/**
 * Stable public entry point for the `Connection` primitive.
 *
 * A `Connection` models one directed, weighted edge between two neurons.
 * It stores the raw scalar weight used during the forward pass, an optional
 * gain factor applied by a gate node, and the gradient accumulation fields
 * (`elegibility`, `xtrace`) needed for the backward pass and eligibility-trace
 * learning rules.
 *
 * Most callers interact with connections indirectly through `Network.connect()`
 * or `Group.connect()`. Import directly from this path only when you need to
 * inspect or mutate edge properties (e.g. reading `weight`, patching a gater
 * reference, or iterating `node.connections.out`).
 *
 * The concrete implementation and all helper logic live in `connection/connection.ts`.
 * This root file is a stable facade that keeps the import path short.
 *
 * @example
 * ```ts
 * import Connection from './connection';
 *
 * // Inspect the weight of a connection on a node:
 * for (const conn of node.connections.out) {
 *   console.log(conn.weight, conn.from.index, conn.to.index);
 * }
 * ```
 */
export { default } from './connection/connection';
