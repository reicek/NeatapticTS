/**
 * Shared topology formatting helpers used by network-view and visualization.
 *
 * The topology boundary resolves node layering, while this helper module adds a
 * few small presentation-oriented utilities that are reused by the visualization
 * panel.
 */
export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';

/**
 * Formats node activation labels with fixed precision.
 *
 * Two decimal places keep the compact node labels scannable while still giving
 * enough precision to distinguish meaningfully different activation values.
 *
 * @param activation - Node activation value.
 * @returns Label text.
 */
export function formatNodeActivationLabel(activation: number): string {
  const roundedActivation = Number.isFinite(activation) ? activation : 0;
  return roundedActivation.toFixed(2);
}
