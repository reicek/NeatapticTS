import {
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR,
} from '../../constants/constants';
import type {
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
} from '../browser-entry.types';
import { resolveInputGroupLabelBands } from './network-view.labels.utils';

/**
 * Overlay drawing helpers specific to the network-view panel.
 *
 * These helpers render semantic guides that sit on top of the raw graph, most
 * notably the colored input-group bands that explain how temporal observation
 * channels are organized.
 */

/**
 * Draws vertical neon bands that label semantic groups in the input layer.
 *
 * The bands make the input layer readable as domain features rather than just a
 * numbered stack of nodes.
 *
 * @param context - Canvas 2D rendering context.
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param nodeDimensions - Resolved node dimensions.
 * @returns Nothing.
 */
export function drawInputGroupLabelBands(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): void {
  const inputNodes = positionedNodes
    .filter(
      (positionedNode) =>
        positionedNode.node.type === 'input' ||
        positionedNode.node.type === 'constant',
    )
    .toSorted((leftNode, rightNode) => leftNode.yPx - rightNode.yPx);

  if (inputNodes.length === 0) {
    return;
  }

  const labelBands = resolveInputGroupLabelBands(inputNodes.length);
  if (labelBands.length === 0) {
    return;
  }

  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const leftmostInputCenterXPx = Math.min(
    ...inputNodes.map((inputNode) => inputNode.xPx),
  );
  const inputLeftEdgeXPx = leftmostInputCenterXPx - halfNodeWidthPx;
  const labelBandRightXPx =
    inputLeftEdgeXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX;
  const labelBandLeftXPx =
    labelBandRightXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX;

  labelBands.forEach((labelBand) => {
    const startNode = inputNodes[labelBand.startNodeIndex];
    const endNode = inputNodes[labelBand.endNodeIndex];
    if (!startNode || !endNode) {
      return;
    }

    const groupTopYPx = startNode.yPx - nodeDimensions.heightPx * 0.5;
    const groupBottomYPx = endNode.yPx + nodeDimensions.heightPx * 0.5;
    const desiredBandHeightPx =
      labelBand.orientation === 'vertical'
        ? Math.max(
            FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
            groupBottomYPx - groupTopYPx,
          )
        : Math.max(nodeDimensions.heightPx + 2, groupBottomYPx - groupTopYPx);
    const labelBandCenterYPx = (groupTopYPx + groupBottomYPx) * 0.5;
    const labelBandTopYPx = labelBandCenterYPx - desiredBandHeightPx * 0.5;

    drawRoundedRect(
      context,
      labelBandLeftXPx,
      labelBandTopYPx,
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
      desiredBandHeightPx,
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
      labelBand.backgroundColor,
    );

    context.save();
    context.translate(
      labelBandLeftXPx + FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX * 0.5,
      labelBandCenterYPx,
    );
    if (labelBand.orientation === 'vertical') {
      context.rotate(-Math.PI / 2);
    }
    context.fillStyle = FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR;
    context.font = `${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT} ${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(labelBand.label, 0, 0);
    context.restore();
  });
}

/**
 * Draws a filled rounded rectangle path.
 *
 * This is the small geometry primitive used by the input-group band renderer.
 */
export function drawRoundedRect(
  context: CanvasRenderingContext2D,
  leftXPx: number,
  topYPx: number,
  widthPx: number,
  heightPx: number,
  radiusPx: number,
  fillColor: string,
): void {
  const resolvedRadiusPx = Math.max(
    0,
    Math.min(radiusPx, widthPx * 0.5, heightPx * 0.5),
  );

  context.beginPath();
  context.moveTo(leftXPx + resolvedRadiusPx, topYPx);
  context.lineTo(leftXPx + widthPx - resolvedRadiusPx, topYPx);
  context.quadraticCurveTo(
    leftXPx + widthPx,
    topYPx,
    leftXPx + widthPx,
    topYPx + resolvedRadiusPx,
  );
  context.lineTo(leftXPx + widthPx, topYPx + heightPx - resolvedRadiusPx);
  context.quadraticCurveTo(
    leftXPx + widthPx,
    topYPx + heightPx,
    leftXPx + widthPx - resolvedRadiusPx,
    topYPx + heightPx,
  );
  context.lineTo(leftXPx + resolvedRadiusPx, topYPx + heightPx);
  context.quadraticCurveTo(
    leftXPx,
    topYPx + heightPx,
    leftXPx,
    topYPx + heightPx - resolvedRadiusPx,
  );
  context.lineTo(leftXPx, topYPx + resolvedRadiusPx);
  context.quadraticCurveTo(leftXPx, topYPx, leftXPx + resolvedRadiusPx, topYPx);
  context.closePath();
  context.fillStyle = fillColor;
  context.fill();
}
