/**
 * Shared topology formatting helpers used by network-view and visualization.
 *
 * The topology boundary resolves node layering, while this helper module adds a
 * few small presentation-oriented utilities that are reused by the visualization
 * panel.
 */
export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';

/**
 * Formats node bias labels with fixed sign and precision.
 *
 * Consistent sign and precision make dense node labels easier to scan quickly in
 * the rendered network panel.
 *
 * @param nodeBias - Node bias value.
 * @returns Label text.
 */
export function formatNodeBiasLabel(nodeBias: number): string {
  const roundedBias = Number.isFinite(nodeBias) ? nodeBias : 0;
  return `${roundedBias >= 0 ? '+' : ''}${roundedBias.toFixed(2)}`;
}
