import { clamp } from '../browser-entry.math.utils';
import { FLAPPY_NETWORK_INPUT_LAYER_TARGET_GAP_PX } from '../../constants/constants';
import type {
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
  VisualNetworkNodeLike,
} from '../browser-entry.types';

/**
 * Node-positioning helpers for the browser network view.
 *
 * Once topology has been resolved into layers, these helpers place nodes inside
 * the drawable panel and then center the final graph so it feels balanced inside
 * the available canvas space.
 */

/**
 * Positions network nodes into drawable canvas coordinates.
 *
 * The layout keeps layer ordering stable while adapting inter-node spacing to
 * the amount of available vertical space.
 *
 * @param networkLayers - Resolved network layers.
 * @param leftPaddingPx - Left graph padding.
 * @param topPaddingPx - Top graph padding.
 * @param drawableWidthPx - Drawable graph width.
 * @param drawableHeightPx - Drawable graph height.
 * @param nodeLayoutPaddingPx - Inner graph padding.
 * @param nodeDimensions - Node dimensions.
 * @returns Positioned nodes.
 */
export function positionNetworkNodes(
  networkLayers: VisualNetworkNodeLike[][],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
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
      layerIndex === 0
        ? FLAPPY_NETWORK_INPUT_LAYER_TARGET_GAP_PX
        : baselineInterNodeGapPx * 0.5;
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
    const minimumStackTopPx = topPaddingPx + nodeLayoutPaddingPx;
    const maximumStackTopPx =
      topPaddingPx +
      drawableHeightPx -
      nodeLayoutPaddingPx -
      layerStackHeightPx;
    const resolvedStackTopPx = clamp(
      centeredStackTopPx,
      minimumStackTopPx,
      Math.max(minimumStackTopPx, maximumStackTopPx),
    );

    layerNodes.forEach((node, nodeInLayerIndex) => {
      const unclampedLayerYPx =
        resolvedStackTopPx +
        halfNodeHeightPx +
        nodeInLayerIndex *
          (nodeDimensions.heightPx + resolvedLayerInterNodeGapPx);
      const layerYPx = clamp(
        unclampedLayerYPx,
        minimumNodeCenterYPx,
        maximumNodeCenterYPx,
      );
      positionedNodes.push({
        node,
        xPx: layerXPx,
        yPx: layerYPx,
      });
    });
  });

  return positionedNodes;
}

/**
 * Centers positioned nodes within the drawable graph area.
 *
 * Positioning establishes relative structure first; centering then shifts the
 * whole graph as a block so it sits comfortably within the padded draw region.
 *
 * @param positionedNodes - Positioned nodes before centering.
 * @param leftPaddingPx - Left graph padding.
 * @param topPaddingPx - Top graph padding.
 * @param drawableWidthPx - Drawable graph width.
 * @param drawableHeightPx - Drawable graph height.
 * @param nodeLayoutPaddingPx - Inner graph padding.
 * @param nodeDimensions - Node dimensions.
 * @returns Center-aligned positioned nodes.
 */
export function centerPositionedNodesInDrawableArea(
  positionedNodes: PositionedNetworkNode[],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
): PositionedNetworkNode[] {
  if (positionedNodes.length === 0) {
    return positionedNodes;
  }

  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;
  const currentLeftPx = Math.min(
    ...positionedNodes.map(
      (positionedNode) => positionedNode.xPx - halfNodeWidthPx,
    ),
  );
  const currentRightPx = Math.max(
    ...positionedNodes.map(
      (positionedNode) => positionedNode.xPx + halfNodeWidthPx,
    ),
  );
  const currentTopPx = Math.min(
    ...positionedNodes.map(
      (positionedNode) => positionedNode.yPx - halfNodeHeightPx,
    ),
  );
  const currentBottomPx = Math.max(
    ...positionedNodes.map(
      (positionedNode) => positionedNode.yPx + halfNodeHeightPx,
    ),
  );

  const minimumLeftPx = leftPaddingPx + nodeLayoutPaddingPx;
  const maximumRightPx = leftPaddingPx + drawableWidthPx - nodeLayoutPaddingPx;
  const minimumTopPx = topPaddingPx + nodeLayoutPaddingPx;
  const maximumBottomPx = topPaddingPx + drawableHeightPx - nodeLayoutPaddingPx;

  const currentCenterXPx = (currentLeftPx + currentRightPx) * 0.5;
  const targetCenterXPx = (minimumLeftPx + maximumRightPx) * 0.5;
  const currentCenterYPx = (currentTopPx + currentBottomPx) * 0.5;
  const targetCenterYPx = (minimumTopPx + maximumBottomPx) * 0.5;

  const desiredShiftXPx = targetCenterXPx - currentCenterXPx;
  const minimumShiftXPx = minimumLeftPx - currentLeftPx;
  const maximumShiftXPx = maximumRightPx - currentRightPx;
  const resolvedShiftXPx = clamp(
    desiredShiftXPx,
    minimumShiftXPx,
    maximumShiftXPx,
  );

  const desiredShiftYPx = targetCenterYPx - currentCenterYPx;
  const minimumShiftYPx = minimumTopPx - currentTopPx;
  const maximumShiftYPx = maximumBottomPx - currentBottomPx;
  const resolvedShiftYPx = clamp(
    desiredShiftYPx,
    minimumShiftYPx,
    maximumShiftYPx,
  );

  return positionedNodes.map((positionedNode) => ({
    ...positionedNode,
    xPx: positionedNode.xPx + resolvedShiftXPx,
    yPx: positionedNode.yPx + resolvedShiftYPx,
  }));
}
