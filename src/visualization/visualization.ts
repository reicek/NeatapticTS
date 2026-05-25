/**
 * Browser-based network visualization renderer.
 *
 * This module provides the shared canvas renderer for demos and external users
 * who want to visualize neural networks in the browser. It accepts a
 * `VisualizationGraphV1` (from `exportVisualizationGraph`) and lays out/draws
 * the network with optional demo-specific overlay hooks.
 *
 * **Main entry point:**
 * - `renderNetworkView(canvas, graph, options)` — render a network on a canvas
 *
 * **Shared infrastructure:**
 * - `positionNetworkNodes()` — generic layer-based node positioning
 * - `centerPositionedNodesInDrawableArea()` — centering logic
 * - `resolveNetworkVisualizationTopologyPlan()` — topology inference
 *
 * ## Usage: canvas renderer (browser)
 *
 * Drop-in minimal usage — no overlay config required:
 *
 * ```ts
 * import { Network, exportVisualizationGraph, renderNetworkView } from 'neataptic';
 *
 * const network = Network.createMLP(2, [4], 1);
 * const graph = exportVisualizationGraph(network);
 * const canvas = document.getElementById('viz-canvas') as HTMLCanvasElement;
 * const frame = renderNetworkView(canvas, graph);
 * // frame.positionedNodes can be used for hover hit-testing
 * ```
 *
 * To inject demo-specific overlays (e.g. sensor-band labels), pass an
 * `overlayFactory` hook — the shared renderer stays unmodified:
 *
 * ```ts
 * const frame = renderNetworkView(canvas, graph, {
 *   nodeDimensions: { widthPx: 28, heightPx: 28 },
 *   colorScales: { weightPositive: '#00ff88', weightNegative: '#ff0088',
 *                  activationHot: '#ffcc00', activationCold: '#0088ff', bias: '#8800ff' },
 *   overlayFactory: {
 *     createDemoOverlayScenes: (positionedNodes, nodeDimensions) => {
 *       // return your custom overlay scene objects here
 *       return [];
 *     },
 *   },
 * });
 * ```
 *
 * ## Usage: schema export + Graphviz DOT (external tooling)
 *
 * Export a network as a portable JSON schema and convert it to a DOT diagram:
 *
 * ```ts
 * import { Network, exportVisualizationGraph, toDot } from 'neataptic';
 *
 * const network = Network.createMLP(3, [5], 2);
 * // Full export (weights + biases included by default).
 * const graph = exportVisualizationGraph(network);
 *
 * // Lightweight export — omit weights and biases for a large population.
 * const compact = exportVisualizationGraph(network, {
 *   includeWeights: false,
 *   includeBiases: false,
 * });
 *
 * // Convert to Graphviz DOT and paste into https://dreampuf.github.io/GraphvizOnline/
 * const dot = toDot(graph);
 * console.log(dot);
 * ```
 *
 * The two paths share one data contract (`VisualizationGraphV1`) so a single
 * `exportVisualizationGraph` call can feed both a canvas renderer and an
 * external tool in the same session.
 *
 * @see {@link https://en.wikipedia.org/wiki/DOT_(graph_description_language) Graphviz DOT language (Wikipedia)}
 */

export { renderNetworkView } from './network-view/network-view';
export type {
  PositionedNetworkNode,
  VisualNetworkConnection,
  NetworkNodeDimensions,
  EdgePadding,
  NetworkVisualizationColorScales,
  NetworkVisualizationResolvedFrame,
  OverlayFactoryHooks,
  RenderNetworkViewOptions,
} from './network-view/network-view.types';
export {
  positionNetworkNodes,
  centerPositionedNodesInDrawableArea,
} from './network-view/network-view.layout.utils';
export type { VisualNetworkNode } from './network-view/network-view.layout.utils';
export {
  /**
   * Re-export of the layer resolution helper used by browser visualization layout passes to extract ordered layer annotations from a topology plan.
   */
  resolveNetworkVisualizationLayers,
  resolveNetworkVisualizationTopologyPlan,
} from './network-view/network-view.topology.utils';
export type {
  NetworkLayerAnnotation,
  NetworkVisualizationTopologyPlan,
} from './network-view/network-view.topology.utils';
