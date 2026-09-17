/**
 * Public visualization facade for the shared network-visualization renderer.
 *
 * The dedicated visualization folder focuses on turning network structure and
 * parameter ranges into readable graphics. Layer resolution itself is shared
 * with the neighboring network-view topology boundary, so this facade exposes
 * that topology helper while keeping the visualization subsystem's public story
 * in one place.
 *
 * The goal of this boundary is not generic charting. It is specifically to make
 * evolved controllers inspectable: which nodes exist, how layers are grouped,
 * and how sign and magnitude are encoded visually.
 *
 * ```mermaid
 * flowchart TD
 *   Network["Network nodes + weights"] --> Range["observe min / max"]
 *   Range --> Tiers["build tiers"]
 *   Tiers --> Scales["NetworkVisualizationColorScales"]
 *   Scales --> Draw["drawWeightedConnectionsLayer / drawBiasNodesLayer"]
 *   Scales --> Legend["drawNetworkColorLegend"]
 *   Draw --> Panel["Readable network panel"]
 *   Legend --> Panel
 * ```
 *
 * @example
 * ```ts
 * const layers = resolveNetworkVisualizationLayers(
 *   network,
 *   network.inputNodeIds.length,
 *   network.outputNodeIds.length,
 * );
 * console.log(layers.length);
 * ```
 *
 * @example
 * ```ts
 * // Minimal consumer path: resolve color scales with defaults, then paint a legend.
 * const scales = resolveNetworkVisualizationColorScales(network);
 * drawNetworkColorLegend(canvasContext, '5 > 3 > 2', scales, {});
 * ```
 */
export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';
