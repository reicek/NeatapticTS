/**
 * Browser-based network visualization for ASCII Maze demo.
 *
 * Displays the evolved network architecture alongside the maze simulation,
 * allowing users to see the network's decision-making in real-time.
 *
 * Layout:
 * - Centered title
 * - Below: Maze grid on left, network visualization on right
 */

import { exportVisualizationGraph } from '../../src/architecture/network/visualization/network.visualization';
import { renderNetworkView } from '../../src/visualization/visualization';
import type Network from '../../src/architecture/network';

/**
 * Options for the ASCII Maze visualizer.
 */
export interface AsciiMazeVisualizerOptions {
  containerSelector?: string;
  canvasWidth?: number;
  canvasHeight?: number;
  title?: string;
}

/**
 * Initializes the ASCII Maze browser visualization.
 *
 * Creates a split-pane layout with the maze on the left and network
 * visualization on the right, using the shared `renderNetworkView()` renderer.
 *
 * @param network - The evolved network to visualize.
 * @param options - Optional configuration.
 */
export function initializeAsciiMazeVisualizer(
  network: Network | undefined,
  options: AsciiMazeVisualizerOptions = {},
): void {
  const {
    containerSelector = '#ascii-maze-visualizer',
    canvasWidth = 400,
    canvasHeight = 400,
    title = 'ASCII Maze Network Visualization',
  } = options;

  const container = document.querySelector(containerSelector) as HTMLDivElement;
  if (!container) {
    console.warn(
      `Container "${containerSelector}" not found for ASCII Maze visualizer`,
    );
    return;
  }

  // Clear container.
  container.innerHTML = '';

  // Create title.
  const titleElement = document.createElement('h2');
  titleElement.textContent = title;
  titleElement.style.cssText = `
    text-align: center;
    color: #00ccff;
    font-family: monospace;
    margin-bottom: 24px;
    font-size: 18px;
    text-transform: uppercase;
    letter-spacing: 2px;
  `;
  container.appendChild(titleElement);

  // Create wrapper for split pane.
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `
    display: flex;
    gap: 24px;
    justify-content: center;
    align-items: flex-start;
  `;

  // Left pane: placeholder for maze (actual maze rendered separately).
  const mazePane = document.createElement('div');
  mazePane.id = 'ascii-maze-pane';
  mazePane.style.cssText = `
    flex: 0 0 auto;
    background: #1a1a1a;
    border: 1px solid #00ccff;
    padding: 16px;
    border-radius: 4px;
    overflow: auto;
    max-height: 600px;
  `;
  mazePane.textContent = 'Maze will be rendered here';
  wrapper.appendChild(mazePane);

  // Right pane: network visualization canvas.
  const canvasPane = document.createElement('div');
  canvasPane.style.cssText = `
    flex: 0 0 auto;
    background: #1a1a1a;
    border: 1px solid #00ccff;
    padding: 8px;
    border-radius: 4px;
  `;

  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvas.style.cssText = `
    display: block;
    background: #1a1a1a;
  `;
  canvasPane.appendChild(canvas);
  wrapper.appendChild(canvasPane);

  container.appendChild(wrapper);

  // Render the network visualization if a network is provided.
  if (network) {
    const graph = exportVisualizationGraph(network);
    renderNetworkView(canvas, graph, {
      nodeDimensions: { widthPx: 20, heightPx: 20 },
      panelPaddingPx: { topPx: 16, rightPx: 16, bottomPx: 16, leftPx: 16 },
    });
  }
}

/**
 * Updates the maze pane with new content.
 *
 * Call this after rendering the maze to update the left pane.
 *
 * @param mazeHtml - HTML string or element containing the maze visualization.
 */
export function updateMazePane(mazeHtml: string | HTMLElement): void {
  const mazePane = document.getElementById('ascii-maze-pane');
  if (!mazePane) return;

  if (typeof mazeHtml === 'string') {
    mazePane.innerHTML = mazeHtml;
  } else {
    mazePane.innerHTML = '';
    mazePane.appendChild(mazeHtml);
  }
}

/**
 * Updates the network visualization with a new network.
 *
 * Call this when the evolved network changes during simulation.
 *
 * @param network - The new network to visualize.
 * @param containerSelector - Container selector (must match initialization).
 */
export function updateNetworkVisualization(
  network: Network | undefined,
  containerSelector: string = '#ascii-maze-visualizer',
): void {
  if (!network) return;

  const container = document.querySelector(containerSelector) as HTMLDivElement;
  if (!container) return;

  const canvas = container.querySelector('canvas') as HTMLCanvasElement;
  if (!canvas) return;

  const graph = exportVisualizationGraph(network);
  renderNetworkView(canvas, graph, {
    nodeDimensions: { widthPx: 20, heightPx: 20 },
    panelPaddingPx: { topPx: 16, rightPx: 16, bottomPx: 16, leftPx: 16 },
  });
}
