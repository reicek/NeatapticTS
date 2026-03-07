import Network from '../../../../../src/architecture/network';
import type { NetworkLegendLayout } from '../browser-entry.types';
import { resolveDefaultNetworkLegendLayoutInternal } from '../browser-entry.visualization.utils';

/**
 * Resolves default legend layout from internal tier definitions.
 *
 * @param context - Render context.
 * @param network - Active network instance.
 * @returns Legend layout.
 */
export function resolveDefaultNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
): NetworkLegendLayout {
  return resolveDefaultNetworkLegendLayoutInternal(context, network);
}
