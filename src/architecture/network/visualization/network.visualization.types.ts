/**
 * Stable, versioned type contracts for network visualization export.
 *
 * `VisualizationGraphV1` is the canonical data shape produced by
 * {@link exportVisualizationGraph}. Any renderer — canvas-based, terminal
 * ASCII, Graphviz DOT, or external tooling — can consume this shape without
 * knowing anything about NeatapticTS internals.
 *
 * The schema is deliberately minimal but sufficient: nodes carry role and
 * activation metadata; edges carry weights, enable state, and an inferred
 * kind; `io` makes the I/O boundary explicit so renderers can highlight it
 * without re-detecting it.
 *
 * ```mermaid
 * classDiagram
 *   class VisualizationGraphV1 {
 *     +number version
 *     +VisualizationNodeV1[] nodes
 *     +VisualizationEdgeV1[] edges
 *     +VisualizationIOV1 io
 *     +VisualizationMetadataV1 metadata
 *   }
 *   class VisualizationNodeV1 {
 *     +number id
 *     +string label
 *     +string role
 *     +string activation
 *     +number bias
 *   }
 *   class VisualizationEdgeV1 {
 *     +number from
 *     +number to
 *     +number weight
 *     +boolean enabled
 *     +string kind
 *   }
 *   VisualizationGraphV1 --> VisualizationNodeV1
 *   VisualizationGraphV1 --> VisualizationEdgeV1
 * ```
 */

/**
 * A single node entry in a versioned visualization graph.
 *
 * `id` is the stable gene id (not a volatile runtime index), so it survives
 * serialization, crossover, and round-trips through evolution.
 */
export interface VisualizationNodeV1 {
  /** Stable gene id. Survives serialization and evolutionary alignment. */
  id: number;
  /** Optional human-readable label attached to this node. */
  label?: string;
  /** Semantic role of this node in the graph. */
  role?: 'input' | 'output' | 'hidden';
  /** Name of the activation (squash) function, e.g. `'LOGISTIC'`. */
  activation?: string;
  /**
   * Bias value added to the weighted input sum before activation.
   * Only present when the caller requested biases via {@link ExportVisualizationOptions}.
   */
  bias?: number;
  /**
   * Identifier for the primitive group this node was constructed from.
   * Populated when the node carries group metadata.
   */
  groupId?: string;
  /**
   * Zero-based layer index inferred from topology traversal.
   * Populated when layer metadata is available.
   */
  layerIndex?: number;
}

/**
 * A single directed edge in a versioned visualization graph.
 *
 * `from` and `to` are stable gene ids matching {@link VisualizationNodeV1.id}.
 */
export interface VisualizationEdgeV1 {
  /** Gene id of the source node. */
  from: number;
  /** Gene id of the target node. */
  to: number;
  /**
   * Scalar connection weight.
   * Only present when the caller requested weights via {@link ExportVisualizationOptions}.
   */
  weight: number;
  /**
   * Whether this connection is currently expressed (enabled).
   * Disabled connections are suppressed from the output by default; see
   * {@link ExportVisualizationOptions.includeDisabledEdges}.
   */
  enabled?: boolean;
  /**
   * Inferred connection kind.
   *
   * - `'self'` — source and target are the same node.
   * - `'forward'` — source appears before target in stable node order.
   * - `'recurrent'` — source appears after target (backward edge, creates a cycle).
   */
  kind?: 'forward' | 'recurrent' | 'self';
}

/**
 * Explicit I/O ordering for the visualization graph.
 *
 * The arrays use stable gene ids and are ordered consistently so renderers
 * can highlight the I/O boundary and map external input/output indices.
 */
export interface VisualizationIOV1 {
  /** Stable gene ids of input-role nodes in activation order. */
  inputNodeIds: number[];
  /** Stable gene ids of output-role nodes in activation order. */
  outputNodeIds: number[];
}

/**
 * Optional metadata block attached to a visualization graph.
 */
export interface VisualizationMetadataV1 {
  /** Optional human-readable name for this network. */
  name?: string;
  /**
   * Execution mode hint.
   *
   * - `'acyclic'` — the graph enforces no directed cycles (feed-forward only).
   * - `'recurrent'` — the graph permits directed cycles.
   */
  mode?: 'acyclic' | 'recurrent';
  /** ISO 8601 creation timestamp. */
  createdAtIso?: string;
}

/**
 * Versioned, deterministic graph schema for network visualization.
 *
 * Produced by {@link exportVisualizationGraph}. Version `1` is the initial
 * stable shape; future breaking changes will increment the version field.
 *
 * @example
 * ```ts
 * import { exportVisualizationGraph } from 'neataptic';
 *
 * const graph = exportVisualizationGraph(network, { includeBiases: true });
 * console.log(graph.nodes.length, graph.edges.length);
 * // → 4 6
 * ```
 */
export interface VisualizationGraphV1 {
  /** Schema version. Always `1` for this shape. */
  version: 1;
  /** Ordered node descriptors (sorted by stable gene id). */
  nodes: VisualizationNodeV1[];
  /** Ordered edge descriptors (sorted by from gene id, then to gene id). */
  edges: VisualizationEdgeV1[];
  /** Explicit I/O boundary with stable gene id ordering. */
  io: VisualizationIOV1;
  /** Optional metadata block. */
  metadata?: VisualizationMetadataV1;
}

/**
 * Options controlling which fields are included in the exported visualization graph.
 *
 * All options default to `true` (maximum detail) unless explicitly set to `false`.
 *
 * @example
 * ```ts
 * // Lightweight export: omit weights and biases for a large evolved network.
 * const graph = exportVisualizationGraph(network, {
 *   includeWeights: false,
 *   includeBiases: false,
 * });
 * ```
 */
export interface ExportVisualizationOptions {
  /**
   * Include the `bias` field on each node descriptor.
   * @defaultValue `true`
   */
  includeBiases?: boolean;
  /**
   * Include the `weight` field on each edge descriptor.
   * @defaultValue `true`
   */
  includeWeights?: boolean;
  /**
   * Include connections whose `enabled` flag is `false`.
   * By default disabled edges are omitted so renderers show the live graph.
   * @defaultValue `false`
   */
  includeDisabledEdges?: boolean;
}
