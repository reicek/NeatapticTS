/**
 * Public visualization facade for browser-entry network rendering.
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
 * Minimal example:
 * ```ts
 * const layers = resolveNetworkVisualizationLayers(
 *   network,
 *   network.inputNodeIds.length,
 *   network.outputNodeIds.length,
 * );
 * console.log(layers.length);
 * ```
 */
export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';
