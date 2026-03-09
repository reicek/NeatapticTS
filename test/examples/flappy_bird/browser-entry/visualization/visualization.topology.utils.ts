export { resolveNetworkVisualizationLayers } from '../network-view/network-view.topology.utils';

/**
 * Formats node bias labels with fixed sign and precision.
 *
 * @param nodeBias - Node bias value.
 * @returns Label text.
 */
export function formatNodeBiasLabel(nodeBias: number): string {
  const roundedBias = Number.isFinite(nodeBias) ? nodeBias : 0;
  return `${roundedBias >= 0 ? '+' : ''}${roundedBias.toFixed(2)}`;
}
