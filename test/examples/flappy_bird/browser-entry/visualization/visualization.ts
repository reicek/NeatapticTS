/**
 * Public visualization facade for browser-entry network rendering.
 *
 * The dedicated visualization folder focuses on turning network structure and
 * parameter ranges into readable graphics. Layer resolution itself is shared
 * with the neighboring network-view topology boundary, so this facade exposes
 * that topology helper while keeping the visualization subsystem's public story
 * in one place.
 */
export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';
