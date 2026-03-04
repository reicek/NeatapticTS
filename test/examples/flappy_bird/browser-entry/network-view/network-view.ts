import Network from '../../../../../src/architecture/network';
import {
  drawNetworkVisualizationInternal,
  resolveNetworkArchitectureLabelInternal,
  resolveNetworkVisualizationHeightPxInternal,
} from '../browser-entry.network-view.utils';

/**
 * Draws a complete, layer-based visualization of the active network.
 */
export function drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): void {
  drawNetworkVisualizationInternal(context, network, inputSize, outputSize);
}

/**
 * Resolves responsive visualization canvas height from network shape.
 */
export function resolveNetworkVisualizationHeightPx(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
  viewportWidthPx?: number,
): number {
  return resolveNetworkVisualizationHeightPxInternal(
    network,
    inputSize,
    outputSize,
    viewportWidthPx,
  );
}

/**
 * Resolves compact architecture label text for headers and HUD rows.
 */
export function resolveNetworkArchitectureLabel(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): string {
  return resolveNetworkArchitectureLabelInternal(network, inputSize, outputSize);
}
