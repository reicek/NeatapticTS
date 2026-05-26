/**
 * Generic layout helpers for browser network visualization.
 *
 * Once topology has been resolved into layers, these helpers place nodes
 * in canvas coordinates and center the final graph within available space.
 */

import type {
  PositionedNetworkNode,
  NetworkNodeDimensions,
} from './network-view.types';

/**
 * Minimal node representation used as input to the layout engine, carrying only the index, type role, and bias value needed for positioning.
 */
export interface VisualNetworkNode {
  index: number;
  type: 'input' | 'hidden' | 'output';
  bias: number;
}

/**
 * Positions network nodes into drawable canvas coordinates.
 *
 * The layout preserves layer ordering while adapting inter-node spacing to
 * available vertical space. Nodes in earlier layers are placed left; nodes
 * in later layers are placed right.
 *
 * @param networkLayers - Resolved network layers (each layer is a list of nodes).
 * @param leftPaddingPx - Left graph padding.
 * @param topPaddingPx - Top graph padding.
 * @param drawableWidthPx - Drawable graph width (canvas width minus horizontal padding).
 * @param drawableHeightPx - Drawable graph height (canvas height minus vertical padding).
 * @param nodeLayoutPaddingPx - Inner graph padding around nodes.
 * @param nodeDimensions - Node dimensions (width × height px).
 * @param inputLayerTargetGapPx - Optional target gap between input nodes (default: 0).
 * @returns Positioned nodes.
 */
export function positionNetworkNodes(
  networkLayers: VisualNetworkNode[][],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
  inputLayerTargetGapPx: number = 0,
): PositionedNetworkNode[] {
  const positionedNodes: PositionedNetworkNode[] = [];
  const lastLayerIndex = Math.max(0, networkLayers.length - 1);
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;
  const minimumNodeCenterYPx =
    topPaddingPx + nodeLayoutPaddingPx + halfNodeHeightPx;
  const maximumNodeCenterYPx =
    topPaddingPx + drawableHeightPx - nodeLayoutPaddingPx - halfNodeHeightPx;

  networkLayers.forEach((layerNodes, layerIndex) => {
    const horizontalProgress =
      lastLayerIndex === 0 ? 0.5 : layerIndex / lastLayerIndex;
    const layerXPx =
      leftPaddingPx +
      nodeLayoutPaddingPx +
      halfNodeWidthPx +
      horizontalProgress *
        Math.max(
          1,
          drawableWidthPx - nodeLayoutPaddingPx * 2 - nodeDimensions.widthPx,
        );

    const layerNodeCount = Math.max(1, layerNodes.length);
    const availableLayerStackHeightPx = Math.max(
      1,
      drawableHeightPx - nodeLayoutPaddingPx * 2,
    );
    const baselineInterNodeGapPx =
      layerNodeCount <= 1
        ? 0
        : Math.max(
            0,
            (availableLayerStackHeightPx -
              nodeDimensions.heightPx * layerNodeCount) /
              (layerNodeCount - 1),
          );
    const preferredLayerInterNodeGapPx =
      layerIndex === 0 ? inputLayerTargetGapPx : baselineInterNodeGapPx * 0.5;
    const maximumLayerInterNodeGapToFitPx =
      layerNodeCount <= 1
        ? 0
        : Math.max(
            0,
            (availableLayerStackHeightPx -
              nodeDimensions.heightPx * layerNodeCount) /
              (layerNodeCount - 1),
          );
    const resolvedLayerInterNodeGapPx = Math.min(
      preferredLayerInterNodeGapPx,
      maximumLayerInterNodeGapToFitPx,
    );
    const layerStackHeightPx =
      nodeDimensions.heightPx * layerNodeCount +
      resolvedLayerInterNodeGapPx * Math.max(0, layerNodeCount - 1);
    const centeredStackTopPx =
      topPaddingPx +
      nodeLayoutPaddingPx +
      Math.max(0, (availableLayerStackHeightPx - layerStackHeightPx) * 0.5);

    layerNodes.forEach((node, nodeIndexInLayer) => {
      const nodeCenterYPx =
        centeredStackTopPx +
        nodeIndexInLayer *
          (nodeDimensions.heightPx + resolvedLayerInterNodeGapPx) +
        halfNodeHeightPx;

      positionedNodes.push({
        index: node.index,
        type: node.type,
        centerXPx: layerXPx,
        centerYPx: Math.min(
          maximumNodeCenterYPx,
          Math.max(minimumNodeCenterYPx, nodeCenterYPx),
        ),
        widthPx: nodeDimensions.widthPx,
        heightPx: nodeDimensions.heightPx,
        bias: node.bias,
      });
    });
  });

  return positionedNodes;
}

/**
 * Centers positioned nodes horizontally within the drawable area.
 *
 * Shifts all node x-coordinates so the leftmost and rightmost nodes
 * are balanced around the center of available space.
 *
 * @param positionedNodes - Positioned nodes.
 * @param drawableWidthPx - Drawable width.
 * @returns Centered positioned nodes.
 */
export function centerPositionedNodesInDrawableArea(
  positionedNodes: PositionedNetworkNode[],
  drawableWidthPx: number,
  drawableLeftPx: number = 0,
): PositionedNetworkNode[] {
  if (positionedNodes.length === 0) {
    return positionedNodes;
  }

  const minXPx = Math.min(...positionedNodes.map((n) => n.centerXPx));
  const maxXPx = Math.max(...positionedNodes.map((n) => n.centerXPx));
  const currentCenterXPx = (minXPx + maxXPx) * 0.5;
  const targetCenterXPx = drawableLeftPx + drawableWidthPx * 0.5;
  const shiftXPx = targetCenterXPx - currentCenterXPx;

  return positionedNodes.map((node) => ({
    ...node,
    centerXPx: node.centerXPx + shiftXPx,
  }));
}
