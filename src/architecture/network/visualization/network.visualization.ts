/**
 * Network visualization export — public API surface.
 *
 * This module provides two complementary export paths:
 *
 * - {@link exportVisualizationGraph} — produces a stable, versioned JSON
 *   object (`VisualizationGraphV1`) that any renderer can consume without
 *   knowing NeatapticTS internals.
 * - {@link toDot} — converts a `VisualizationGraphV1` into a Graphviz DOT
 *   string that can be pasted into any DOT renderer to produce a visual graph.
 *
 * ## Design intent
 *
 * The schema layer is intentionally renderer-agnostic. It produces data; it
 * does not draw anything. Rendering decisions (canvas, terminal, DOT, SVG) are
 * left to consumers. This makes the export path useful for:
 *
 * - In-browser canvas renderers (e.g. an interactive demo panel).
 * - Terminal ASCII renderers (e.g. a maze-exploration demo).
 * - External visualization tools via the DOT helper.
 * - Debugging architectures in issues or docs.
 *
 * ## Determinism guarantee
 *
 * For the same network state, `exportVisualizationGraph` always returns an
 * identical JSON structure. Nodes are sorted by stable gene id ascending;
 * edges are sorted by (from gene id, to gene id) ascending.
 *
 * ## Usage
 *
 * ```ts
 * import { exportVisualizationGraph, toDot } from 'neataptic';
 *
 * // Full export with biases and weights.
 * const graph = exportVisualizationGraph(network);
 *
 * // Lightweight export — omit weights and biases for a large population.
 * const compact = exportVisualizationGraph(network, {
 *   includeWeights: false,
 *   includeBiases: false,
 * });
 *
 * // Render via Graphviz.
 * const dot = toDot(graph);
 * console.log(dot);
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/DOT_(graph_description_language) Graphviz DOT language (Wikipedia)}
 * @see {@link https://en.wikipedia.org/wiki/Directed_graph Directed graph (Wikipedia)}
 */

import type Network from '../network';
import type {
  ExportVisualizationOptions,
  VisualizationEdgeV1,
  VisualizationGraphV1,
  VisualizationNodeV1,
} from './network.visualization.types';
import {
  buildNodePositionMap,
  collectSortedConnections,
  connectionToVisualizationDescriptor,
  nodeToVisualizationDescriptor,
  resolveNodeRole,
} from './network.visualization.utils';

/**
 * Exports a deterministic, versioned visualization graph from a live network.
 *
 * The returned `VisualizationGraphV1` object is a plain JSON-serializable value
 * with no circular references. It can be passed to a canvas renderer, terminal
 * renderer, or serialized to disk.
 *
 * **Ordering guarantees:**
 * - `nodes` are sorted by stable gene id ascending.
 * - `edges` are sorted by (from gene id, to gene id) ascending.
 * - `io.inputNodeIds` and `io.outputNodeIds` preserve the network's explicit
 *   I/O ordering (same as {@link Network.inputNodeIds} / {@link Network.outputNodeIds}).
 *
 * @param network - The network instance to export.
 * @param options - Optional flags controlling which fields are included.
 * @returns Versioned, deterministic visualization graph.
 *
 * @example
 * ```ts
 * const graph = exportVisualizationGraph(network, { includeBiases: true });
 * // graph.version === 1
 * // graph.nodes[0].role === 'input'
 * // graph.edges[0].kind === 'forward'
 * ```
 */
export function exportVisualizationGraph(
  network: Network,
  options?: ExportVisualizationOptions,
): VisualizationGraphV1 {
  // Step 1: Resolve option flags with defaults.
  const includeBiases = options?.includeBiases !== false;
  const includeWeights = options?.includeWeights !== false;
  const includeDisabledEdges = options?.includeDisabledEdges === true;

  // Step 2: Build stable role lookup sets from the network's explicit I/O ids.
  const inputIdSet = new Set<number>(network.inputNodeIds);
  const outputIdSet = new Set<number>(network.outputNodeIds);

  // Step 3: Sort nodes by gene id for deterministic ordering.
  const sortedNodes = network.nodes.toSorted(
    (firstNode, secondNode) => firstNode.geneId - secondNode.geneId,
  );

  // Step 4: Build position map for edge-kind inference (used in edge loop below).
  const nodePositionByGeneId = buildNodePositionMap(sortedNodes);

  // Step 5: Map nodes to schema descriptors.
  const nodes: VisualizationNodeV1[] = sortedNodes.map((node) => {
    const role = resolveNodeRole(node, inputIdSet, outputIdSet);
    return nodeToVisualizationDescriptor(node, role, includeBiases);
  });

  // Step 6: Merge regular connections and self-connections, then sort.
  const allConnections = [...network.connections, ...network.selfconns];
  const sortedConnections = collectSortedConnections(
    allConnections,
    includeDisabledEdges,
  );

  // Step 7: Map connections to schema descriptors.
  const edges: VisualizationEdgeV1[] = sortedConnections.map((connection) =>
    connectionToVisualizationDescriptor(
      connection,
      nodePositionByGeneId,
      includeWeights,
    ),
  );

  // Step 8: Assemble and return the versioned schema.
  return {
    version: 1,
    nodes,
    edges,
    io: {
      inputNodeIds: network.inputNodeIds,
      outputNodeIds: network.outputNodeIds,
    },
    metadata: {
      mode:
        network.getTopologyIntent() === 'feed-forward'
          ? 'acyclic'
          : 'recurrent',
      createdAtIso: new Date().toISOString(),
    },
  };
}

// --- DOT export helper ---

/** Graphviz node shape used for input nodes. */
const DOT_INPUT_SHAPE = 'invtriangle';
/** Graphviz node shape used for output nodes. */
const DOT_OUTPUT_SHAPE = 'doublecircle';
/** Graphviz node shape used for hidden nodes. */
const DOT_HIDDEN_SHAPE = 'circle';
/** Number of decimal places for weight labels. */
const DOT_WEIGHT_PRECISION = 3;

/**
 * Converts a {@link VisualizationGraphV1} to a Graphviz DOT string.
 *
 * The output can be pasted into any DOT renderer (e.g.
 * [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)) to produce
 * a visual graph diagram.
 *
 * Node shapes:
 * - **Inputs** — inverted triangle (`invtriangle`).
 * - **Outputs** — double circle (`doublecircle`).
 * - **Hidden** — circle (`circle`).
 *
 * Edge labels show weights when they are present in the graph.
 * Disabled edges are rendered as dashed lines.
 *
 * @param graph - Versioned visualization graph produced by {@link exportVisualizationGraph}.
 * @returns Graphviz DOT string.
 *
 * @example
 * ```ts
 * const graph = exportVisualizationGraph(network);
 * const dot = toDot(graph);
 * // Paste `dot` into https://dreampuf.github.io/GraphvizOnline/
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/DOT_(graph_description_language) DOT language (Wikipedia)}
 */
export function toDot(graph: VisualizationGraphV1): string {
  // Step 1: Build node attribute lines.
  const nodeLines = buildDotNodeLines(graph.nodes);

  // Step 2: Build edge attribute lines.
  const edgeLines = buildDotEdgeLines(graph.edges);

  // Step 3: Assemble the full DOT body.
  const bodyLines = [...nodeLines, ...edgeLines].join('\n  ');
  return `digraph NeatapticNetwork {\n  rankdir=LR;\n  ${bodyLines}\n}`;

  /**
   * Builds DOT attribute lines for all nodes.
   *
   * @param nodes - Node descriptors.
   * @returns Array of DOT attribute strings.
   */
  function buildDotNodeLines(nodes: VisualizationNodeV1[]): string[] {
    return nodes.map((node) => {
      const shape = resolveNodeShape(node.role);
      const displayLabel = node.label ?? String(node.id);
      return `${node.id} [label="${displayLabel}", shape=${shape}];`;
    });
  }

  /**
   * Builds DOT attribute lines for all edges.
   *
   * @param edges - Edge descriptors.
   * @returns Array of DOT attribute strings.
   */
  function buildDotEdgeLines(edges: VisualizationEdgeV1[]): string[] {
    return edges.map((edge) => {
      const weightLabel =
        edge.weight !== undefined
          ? ` [label="${edge.weight.toFixed(DOT_WEIGHT_PRECISION)}"${edge.enabled === false ? ', style=dashed' : ''}]`
          : '';
      return `${edge.from} -> ${edge.to}${weightLabel};`;
    });
  }

  /**
   * Resolves the Graphviz shape name for a node role.
   *
   * @param role - Node role or undefined.
   * @returns Graphviz shape string.
   */
  function resolveNodeShape(
    role: 'input' | 'output' | 'hidden' | undefined,
  ): string {
    if (role === 'input') {
      return DOT_INPUT_SHAPE;
    }
    if (role === 'output') {
      return DOT_OUTPUT_SHAPE;
    }
    return DOT_HIDDEN_SHAPE;
  }
}
