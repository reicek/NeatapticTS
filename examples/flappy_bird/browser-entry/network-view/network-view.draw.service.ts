import {
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_CHARACTER_WIDTH_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FILL_COLOR,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FONT_SIZE_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_GAP_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_LINE_HEIGHT_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_MIN_WIDTH_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_RADIUS_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_STROKE_COLOR,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_STROKE_WIDTH_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_COLOR,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_PADDING_PX,
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_VERTICAL_PADDING_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_FILL_COLOR,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_FONT_SIZE_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_FONT_WEIGHT,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_RADIUS_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_STROKE_COLOR,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_STROKE_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_COLOR,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_LINE_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR,
  FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX,
  FLAPPY_NETWORK_INPUT_OVERLAY_FOCUS_STROKE_WIDTH_PX,
} from '../../constants/constants';
import type {
  NetworkInputDescriptionScene,
  NetworkInputGroupLabelBandScene,
  NetworkHiddenColumnLabelScene,
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
} from '../browser-entry.types';
import type { NetworkHiddenColumnAnnotation } from './network-view.topology.utils';
import {
  resolveInputDescriptionChipWidthPx,
  resolveInputDescriptionColumnWidthPx,
  resolveInputGroupLabelBands,
  resolveInputNodeDescriptionLabels,
} from './network-view.labels.utils';

type ResolvedInputDescriptionLayout = {
  descriptionHeightPx: number;
  descriptionWidthPx: number;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  nodeIndex: number;
};

type ResolvedInputDescriptionLayoutScene = ResolvedInputDescriptionLayout & {
  topPx: number;
};

type ResolvedInputGroupLayout = {
  label: string;
  labelLines: readonly string[];
  nodeIndices: number[];
  topPx: number;
  heightPx: number;
  descriptionLayouts: ResolvedInputDescriptionLayoutScene[];
};

/**
 * Overlay drawing helpers specific to the network-view panel.
 *
 * These helpers render semantic guides that sit on top of the raw graph, most
 * notably the colored input-group bands and per-input labels that explain how
 * the simplified observation shelf is organized.
 */

/**
 * Resolves vertical neon band scenes that label semantic groups in the input layer.
 *
 * Resolving the bands up front lets drawing and hover hit testing reuse the
 * same geometry instead of maintaining duplicate layout logic.
 *
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param nodeDimensions - Resolved node dimensions.
 * @returns Positioned label-band scenes.
 */
export function resolveInputGroupLabelBandScenes(
  positionedNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
  inputDescriptionScenes?: readonly NetworkInputDescriptionScene[],
): NetworkInputGroupLabelBandScene[] {
  const inputNodes = resolveSortedInputNodes(positionedNodes);

  if (inputNodes.length === 0) {
    return [];
  }

  const labelBands = resolveInputGroupLabelBands(inputNodes.length);
  if (labelBands.length === 0) {
    return [];
  }
  const resolvedInputDescriptionScenes =
    inputDescriptionScenes ??
    resolveInputDescriptionScenes(positionedNodes, nodeDimensions);

  const inputOverlayLayout = resolveInputOverlayLayout(
    inputNodes,
    nodeDimensions,
    resolveInputDescriptionColumnWidthPx(inputNodes.length),
  );

  return labelBands.flatMap((labelBand) => {
    const groupedInputDescriptionScenes = resolvedInputDescriptionScenes.slice(
      labelBand.startNodeIndex,
      labelBand.endNodeIndex + 1,
    );
    const firstGroupedInputDescriptionScene = groupedInputDescriptionScenes[0];
    const lastGroupedInputDescriptionScene =
      groupedInputDescriptionScenes.at(-1);
    if (
      !firstGroupedInputDescriptionScene ||
      !lastGroupedInputDescriptionScene
    ) {
      return [];
    }

    const groupTopYPx =
      firstGroupedInputDescriptionScene.topPx -
      FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX;
    const groupBottomYPx =
      lastGroupedInputDescriptionScene.topPx +
      lastGroupedInputDescriptionScene.heightPx +
      FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX;
    const desiredBandHeightPx = Math.max(
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
      groupBottomYPx - groupTopYPx,
    );
    const labelBandCenterYPx = (groupTopYPx + groupBottomYPx) * 0.5;
    const labelBandTopYPx = labelBandCenterYPx - desiredBandHeightPx * 0.5;

    return [
      {
        label: labelBand.label,
        labelLines: labelBand.labelLines,
        tooltipHeading: labelBand.tooltipHeading,
        tooltipBodyParagraphs: labelBand.tooltipBodyParagraphs,
        leftPx: inputOverlayLayout.labelBandLeftXPx,
        topPx: labelBandTopYPx,
        widthPx: FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
        heightPx: desiredBandHeightPx,
        backgroundColor: labelBand.backgroundColor,
        orientation: labelBand.orientation,
        nodeIndices: inputNodes
          .slice(labelBand.startNodeIndex, labelBand.endNodeIndex + 1)
          .map((inputNode) => inputNode.node.index),
      },
    ];
  });
}

/**
 * Resolves one horizontal description scene for each input node.
 *
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param nodeDimensions - Resolved node dimensions.
 * @returns Positioned input-description scenes.
 */
export function resolveInputDescriptionScenes(
  positionedNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): NetworkInputDescriptionScene[] {
  const inputNodes = resolveSortedInputNodes(positionedNodes);

  if (inputNodes.length === 0) {
    return [];
  }

  const inputDescriptions = resolveInputNodeDescriptionLabels(
    inputNodes.length,
  );
  if (inputDescriptions.length === 0) {
    return [];
  }
  const labelBands = resolveInputGroupLabelBands(inputNodes.length);
  if (labelBands.length === 0) {
    return [];
  }

  const descriptionColumnWidthPx = resolveInputDescriptionColumnWidthPx(
    inputNodes.length,
  );
  const inputOverlayLayout = resolveInputOverlayLayout(
    inputNodes,
    nodeDimensions,
    descriptionColumnWidthPx,
  );
  const inputOverlayLayouts = resolveInputOverlayLayouts(
    inputDescriptions,
    labelBands,
    inputNodes,
    nodeDimensions,
  );

  return inputOverlayLayouts.flatMap((inputOverlayLayoutGroup) =>
    inputOverlayLayoutGroup.descriptionLayouts.map((descriptionLayout) => ({
      labelLines: descriptionLayout.labelLines,
      leftPx:
        inputOverlayLayout.descriptionRightXPx -
        descriptionLayout.descriptionWidthPx,
      topPx: descriptionLayout.topPx,
      widthPx: descriptionLayout.descriptionWidthPx,
      heightPx: descriptionLayout.descriptionHeightPx,
      tooltipHeading: descriptionLayout.tooltipHeading,
      tooltipBodyParagraphs: descriptionLayout.tooltipBodyParagraphs,
      nodeIndex: descriptionLayout.nodeIndex,
    })),
  );
}

/**
 * Aligns input-node centers with the resolved description chip centers.
 *
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param inputDescriptionScenes - Positioned input-description scenes.
 * @returns Positioned nodes with input-node rows aligned to their description chips.
 */
export function alignInputNodesToDescriptionScenes(
  positionedNodes: PositionedNetworkNode[],
  inputDescriptionScenes: readonly NetworkInputDescriptionScene[],
): PositionedNetworkNode[] {
  if (inputDescriptionScenes.length === 0) {
    return positionedNodes;
  }

  const descriptionCenterYPxByNodeIndex = new Map(
    inputDescriptionScenes.map((inputDescriptionScene) => [
      inputDescriptionScene.nodeIndex,
      inputDescriptionScene.topPx + inputDescriptionScene.heightPx * 0.5,
    ]),
  );

  return positionedNodes.map((positionedNode) => {
    const alignedCenterYPx = descriptionCenterYPxByNodeIndex.get(
      positionedNode.node.index,
    );
    if (typeof alignedCenterYPx !== 'number') {
      return positionedNode;
    }

    return {
      ...positionedNode,
      yPx: alignedCenterYPx,
    };
  });
}

/**
 * Draws vertical neon bands that label semantic groups in the input layer.
 *
 * @param context - Canvas 2D rendering context.
 * @param inputGroupLabelBandScenes - Positioned label-band scenes.
 * @returns Nothing.
 */
export function drawInputGroupLabelBands(
  context: CanvasRenderingContext2D,
  inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[],
  hoveredNodeIndices?: readonly number[],
): void {
  context.save();
  context.fillStyle = FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR;
  context.font = `${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT} ${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  inputGroupLabelBandScenes.forEach((inputGroupLabelBandScene) => {
    drawRoundedRect(
      context,
      inputGroupLabelBandScene.leftPx,
      inputGroupLabelBandScene.topPx,
      inputGroupLabelBandScene.widthPx,
      inputGroupLabelBandScene.heightPx,
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
      inputGroupLabelBandScene.backgroundColor,
    );
    if (
      hoveredNodeIndices?.some((hoveredNodeIndex) =>
        inputGroupLabelBandScene.nodeIndices.includes(hoveredNodeIndex),
      )
    ) {
      strokeRoundedRect(
        context,
        inputGroupLabelBandScene.leftPx,
        inputGroupLabelBandScene.topPx,
        inputGroupLabelBandScene.widthPx,
        inputGroupLabelBandScene.heightPx,
        FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
        FLAPPY_NEON_PALETTE.statusText,
        FLAPPY_NETWORK_INPUT_OVERLAY_FOCUS_STROKE_WIDTH_PX,
      );
    }

    const labelLineBlockHeightPx =
      (inputGroupLabelBandScene.labelLines.length - 1) *
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_LINE_HEIGHT_PX;
    const labelCenterXPx =
      inputGroupLabelBandScene.leftPx + inputGroupLabelBandScene.widthPx * 0.5;
    const labelCenterYPx =
      inputGroupLabelBandScene.topPx + inputGroupLabelBandScene.heightPx * 0.5;

    inputGroupLabelBandScene.labelLines.forEach((labelLine, labelLineIndex) => {
      context.fillStyle = FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR;
      context.fillText(
        labelLine,
        labelCenterXPx,
        labelCenterYPx -
          labelLineBlockHeightPx * 0.5 +
          labelLineIndex * FLAPPY_NETWORK_INPUT_GROUP_LABEL_LINE_HEIGHT_PX,
      );
    });
  });

  context.restore();
}

/**
 * Draws the horizontal per-input description rows.
 *
 * @param context - Canvas 2D rendering context.
 * @param inputDescriptionScenes - Positioned input-description scenes.
 * @returns Nothing.
 */
export function drawInputNodeDescriptions(
  context: CanvasRenderingContext2D,
  inputDescriptionScenes: NetworkInputDescriptionScene[],
  hoveredNodeIndices?: readonly number[],
): void {
  context.save();
  context.font = `${FLAPPY_NETWORK_INPUT_DESCRIPTION_FONT_WEIGHT} ${FLAPPY_NETWORK_INPUT_DESCRIPTION_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  inputDescriptionScenes.forEach((inputDescriptionScene) => {
    drawRoundedRect(
      context,
      inputDescriptionScene.leftPx,
      inputDescriptionScene.topPx,
      inputDescriptionScene.widthPx,
      inputDescriptionScene.heightPx,
      FLAPPY_NETWORK_INPUT_DESCRIPTION_RADIUS_PX,
      FLAPPY_NETWORK_INPUT_DESCRIPTION_FILL_COLOR,
    );
    strokeRoundedRect(
      context,
      inputDescriptionScene.leftPx,
      inputDescriptionScene.topPx,
      inputDescriptionScene.widthPx,
      inputDescriptionScene.heightPx,
      FLAPPY_NETWORK_INPUT_DESCRIPTION_RADIUS_PX,
      hoveredNodeIndices?.includes(inputDescriptionScene.nodeIndex)
        ? FLAPPY_NEON_PALETTE.statusText
        : FLAPPY_NETWORK_INPUT_DESCRIPTION_STROKE_COLOR,
      hoveredNodeIndices?.includes(inputDescriptionScene.nodeIndex)
        ? FLAPPY_NETWORK_INPUT_OVERLAY_FOCUS_STROKE_WIDTH_PX
        : FLAPPY_NETWORK_INPUT_DESCRIPTION_STROKE_WIDTH_PX,
    );

    const descriptionLineBlockHeightPx =
      (inputDescriptionScene.labelLines.length - 1) *
      FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX;
    const descriptionCenterYPx =
      inputDescriptionScene.topPx + inputDescriptionScene.heightPx * 0.5;
    const descriptionCenterXPx =
      inputDescriptionScene.leftPx + inputDescriptionScene.widthPx * 0.5;
    context.fillStyle = FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_COLOR;

    inputDescriptionScene.labelLines.forEach((labelLine, labelLineIndex) => {
      context.fillText(
        labelLine,
        descriptionCenterXPx,
        descriptionCenterYPx -
          descriptionLineBlockHeightPx * 0.5 +
          labelLineIndex * FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX,
      );
    });
  });

  context.restore();
}

/**
 * Resolves hidden-column guide scenes for recurrent-aware layouts.
 *
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param nodeDimensions - Resolved node dimensions.
 * @param hiddenColumnAnnotations - Semantic hidden-column annotations.
 * @returns Positioned hidden-column label scenes.
 */
export function resolveHiddenColumnLabelScenes(
  positionedNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
  hiddenColumnAnnotations: readonly NetworkHiddenColumnAnnotation[],
): NetworkHiddenColumnLabelScene[] {
  if (hiddenColumnAnnotations.length === 0) {
    return [];
  }

  const positionedNodeByIndex = new Map(
    positionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;

  return hiddenColumnAnnotations.flatMap((hiddenColumnAnnotation) => {
    const resolvedColumnNodes = hiddenColumnAnnotation.nodeIndices
      .map((nodeIndex) => positionedNodeByIndex.get(nodeIndex))
      .filter(
        (positionedNode): positionedNode is PositionedNetworkNode =>
          positionedNode != null,
      );
    if (resolvedColumnNodes.length === 0) {
      return [];
    }

    const widestLineCharacterCount = Math.max(
      0,
      ...hiddenColumnAnnotation.labelLines.map((labelLine) => labelLine.length),
    );
    const minimumNodeLeftPx = Math.min(
      ...resolvedColumnNodes.map(
        (resolvedColumnNode) => resolvedColumnNode.xPx - halfNodeWidthPx,
      ),
    );
    const maximumNodeRightPx = Math.max(
      ...resolvedColumnNodes.map(
        (resolvedColumnNode) => resolvedColumnNode.xPx + halfNodeWidthPx,
      ),
    );
    const widthPx = Math.max(
      FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_MIN_WIDTH_PX,
      widestLineCharacterCount *
        FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_CHARACTER_WIDTH_PX +
        FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_PADDING_PX * 2,
      maximumNodeRightPx - minimumNodeLeftPx,
    );
    const heightPx = Math.max(
      nodeDimensions.heightPx +
        FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_VERTICAL_PADDING_PX * 2,
      hiddenColumnAnnotation.labelLines.length *
        FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_LINE_HEIGHT_PX +
        FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_VERTICAL_PADDING_PX * 2,
    );
    const minimumNodeTopPx = Math.min(
      ...resolvedColumnNodes.map(
        (resolvedColumnNode) => resolvedColumnNode.yPx - halfNodeHeightPx,
      ),
    );
    const columnCenterXPx = (minimumNodeLeftPx + maximumNodeRightPx) * 0.5;

    return [
      {
        labelLines: hiddenColumnAnnotation.labelLines,
        tooltipHeading: hiddenColumnAnnotation.tooltipHeading,
        tooltipBodyParagraphs: hiddenColumnAnnotation.tooltipBodyParagraphs,
        leftPx: columnCenterXPx - widthPx * 0.5,
        topPx:
          minimumNodeTopPx -
          FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_GAP_PX -
          heightPx,
        widthPx,
        heightPx,
        backgroundColor:
          hiddenColumnAnnotation.backgroundColor ||
          FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FILL_COLOR,
        nodeIndices: [...hiddenColumnAnnotation.nodeIndices],
      },
    ];
  });
}

/**
 * Draws hidden-column guide chips for recurrent-aware layouts.
 *
 * @param context - Canvas 2D rendering context.
 * @param hiddenColumnLabelScenes - Positioned hidden-column label scenes.
 * @param hoveredNodeIndices - Optional hovered-node indices used to focus the matching column.
 * @returns Nothing.
 */
export function drawHiddenColumnLabelScenes(
  context: CanvasRenderingContext2D,
  hiddenColumnLabelScenes: readonly NetworkHiddenColumnLabelScene[],
  hoveredNodeIndices?: readonly number[],
): void {
  if (hiddenColumnLabelScenes.length === 0) {
    return;
  }

  context.save();
  context.font = `${FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FONT_WEIGHT} ${FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  hiddenColumnLabelScenes.forEach((hiddenColumnLabelScene) => {
    drawRoundedRect(
      context,
      hiddenColumnLabelScene.leftPx,
      hiddenColumnLabelScene.topPx,
      hiddenColumnLabelScene.widthPx,
      hiddenColumnLabelScene.heightPx,
      FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_RADIUS_PX,
      hiddenColumnLabelScene.backgroundColor,
    );
    strokeRoundedRect(
      context,
      hiddenColumnLabelScene.leftPx,
      hiddenColumnLabelScene.topPx,
      hiddenColumnLabelScene.widthPx,
      hiddenColumnLabelScene.heightPx,
      FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_RADIUS_PX,
      hoveredNodeIndices?.some((hoveredNodeIndex) =>
        hiddenColumnLabelScene.nodeIndices.includes(hoveredNodeIndex),
      )
        ? FLAPPY_NEON_PALETTE.statusText
        : FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_STROKE_COLOR,
      hoveredNodeIndices?.some((hoveredNodeIndex) =>
        hiddenColumnLabelScene.nodeIndices.includes(hoveredNodeIndex),
      )
        ? FLAPPY_NETWORK_INPUT_OVERLAY_FOCUS_STROKE_WIDTH_PX
        : FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_STROKE_WIDTH_PX,
    );

    const labelLineBlockHeightPx =
      (hiddenColumnLabelScene.labelLines.length - 1) *
      FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_LINE_HEIGHT_PX;
    const labelCenterXPx =
      hiddenColumnLabelScene.leftPx + hiddenColumnLabelScene.widthPx * 0.5;
    const labelCenterYPx =
      hiddenColumnLabelScene.topPx + hiddenColumnLabelScene.heightPx * 0.5;

    hiddenColumnLabelScene.labelLines.forEach((labelLine, labelLineIndex) => {
      context.fillStyle = FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TEXT_COLOR;
      context.fillText(
        labelLine,
        labelCenterXPx,
        labelCenterYPx -
          labelLineBlockHeightPx * 0.5 +
          labelLineIndex * FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_LINE_HEIGHT_PX,
      );
    });
  });

  context.restore();
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
  traceRoundedRectPath(context, leftXPx, topYPx, widthPx, heightPx, radiusPx);
  context.fillStyle = fillColor;
  context.fill();
}

function resolveSortedInputNodes(
  positionedNodes: PositionedNetworkNode[],
): PositionedNetworkNode[] {
  return positionedNodes
    .filter(
      (positionedNode) =>
        positionedNode.node.type === 'input' ||
        positionedNode.node.type === 'constant',
    )
    .toSorted((leftNode, rightNode) => leftNode.yPx - rightNode.yPx);
}

function resolveInputOverlayLayout(
  inputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
  descriptionWidthPx: number,
): {
  descriptionColumnLeftXPx: number;
  descriptionRightXPx: number;
  labelBandLeftXPx: number;
} {
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const leftmostInputCenterXPx = Math.min(
    ...inputNodes.map((inputNode) => inputNode.xPx),
  );
  const inputLeftEdgeXPx = leftmostInputCenterXPx - halfNodeWidthPx;
  const descriptionRightXPx =
    inputLeftEdgeXPx - FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX;
  const descriptionColumnLeftXPx = descriptionRightXPx - descriptionWidthPx;
  const labelBandRightXPx =
    descriptionColumnLeftXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX;

  return {
    descriptionColumnLeftXPx,
    descriptionRightXPx,
    labelBandLeftXPx:
      labelBandRightXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  };
}

function resolveInputOverlayLayouts(
  inputDescriptions: readonly {
    labelLines: readonly string[];
    tooltipHeading: string;
    tooltipBodyParagraphs: readonly string[];
    nodeIndex: number;
  }[],
  labelBands: readonly {
    label: string;
    labelLines: readonly string[];
    startNodeIndex: number;
    endNodeIndex: number;
  }[],
  inputNodes: readonly PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): ResolvedInputGroupLayout[] {
  const descriptionLayouts = inputDescriptions.map((inputDescription) => ({
    labelLines: inputDescription.labelLines,
    tooltipHeading: inputDescription.tooltipHeading,
    tooltipBodyParagraphs: inputDescription.tooltipBodyParagraphs,
    nodeIndex: inputDescription.nodeIndex,
    descriptionWidthPx: resolveInputDescriptionChipWidthPx(
      inputDescription.labelLines,
    ),
    descriptionHeightPx: Math.max(
      FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_HEIGHT_PX,
      inputDescription.labelLines.length *
        FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX +
        FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX * 2,
      nodeDimensions.heightPx +
        FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX * 2,
    ),
  }));
  const currentInputStackCenterYPx = resolveInputStackCenterYPx(inputNodes);

  let runningTopYPx = 0;
  const relativeInputOverlayLayouts = labelBands.map(
    (labelBand, labelBandIndex) => {
      const groupedDescriptionLayouts = descriptionLayouts.slice(
        labelBand.startNodeIndex,
        labelBand.endNodeIndex + 1,
      );
      const groupedNodeIndices = inputNodes
        .slice(labelBand.startNodeIndex, labelBand.endNodeIndex + 1)
        .map((inputNode) => inputNode.node.index);
      const groupedDescriptionContentHeightPx =
        resolveGroupedDescriptionContentHeightPx(groupedDescriptionLayouts);
      const groupHeightPx = Math.max(
        FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
        groupedDescriptionContentHeightPx +
          FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX * 2,
      );
      const descriptionContentTopYPx =
        runningTopYPx +
        Math.max(
          FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
          (groupHeightPx - groupedDescriptionContentHeightPx) * 0.5,
        );
      let runningDescriptionTopYPx = descriptionContentTopYPx;
      const groupedDescriptionScenes = groupedDescriptionLayouts.map(
        (groupedDescriptionLayout, groupedDescriptionIndex) => {
          const nextDescriptionTopYPx = runningDescriptionTopYPx;
          runningDescriptionTopYPx +=
            groupedDescriptionLayout.descriptionHeightPx;
          if (groupedDescriptionIndex < groupedDescriptionLayouts.length - 1) {
            runningDescriptionTopYPx +=
              FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX;
          }

          return {
            ...groupedDescriptionLayout,
            topPx: nextDescriptionTopYPx,
          };
        },
      );

      const resolvedGroupLayout = {
        label: labelBand.label,
        labelLines: labelBand.labelLines,
        nodeIndices: groupedNodeIndices,
        topPx: runningTopYPx,
        heightPx: groupHeightPx,
        descriptionLayouts: groupedDescriptionScenes,
      };
      runningTopYPx += groupHeightPx;
      if (labelBandIndex < labelBands.length - 1) {
        runningTopYPx += FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX;
      }
      return resolvedGroupLayout;
    },
  );
  const totalOverlayHeightPx = relativeInputOverlayLayouts.at(-1)
    ? relativeInputOverlayLayouts.at(-1)!.topPx +
      relativeInputOverlayLayouts.at(-1)!.heightPx
    : 0;
  const overlayTopOffsetYPx =
    currentInputStackCenterYPx - totalOverlayHeightPx * 0.5;

  return relativeInputOverlayLayouts.map((relativeInputOverlayLayout) => ({
    ...relativeInputOverlayLayout,
    topPx: relativeInputOverlayLayout.topPx + overlayTopOffsetYPx,
    descriptionLayouts: relativeInputOverlayLayout.descriptionLayouts.map(
      (descriptionLayout) => ({
        ...descriptionLayout,
        topPx: descriptionLayout.topPx + overlayTopOffsetYPx,
      }),
    ),
  }));
}

function resolveGroupedDescriptionContentHeightPx(
  groupedDescriptionLayouts: readonly ResolvedInputDescriptionLayout[],
): number {
  if (groupedDescriptionLayouts.length === 0) {
    return 0;
  }

  return groupedDescriptionLayouts.reduce(
    (
      currentGroupedDescriptionHeightPx,
      groupedDescriptionLayout,
      groupedDescriptionIndex,
    ) =>
      currentGroupedDescriptionHeightPx +
      groupedDescriptionLayout.descriptionHeightPx +
      (groupedDescriptionIndex < groupedDescriptionLayouts.length - 1
        ? FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX
        : 0),
    0,
  );
}

function resolveInputStackCenterYPx(
  inputNodes: readonly PositionedNetworkNode[],
): number {
  const firstInputNode = inputNodes[0];
  const lastInputNode = inputNodes.at(-1);

  if (!firstInputNode || !lastInputNode) {
    return 0;
  }

  return (firstInputNode.yPx + lastInputNode.yPx) * 0.5;
}

function strokeRoundedRect(
  context: CanvasRenderingContext2D,
  leftXPx: number,
  topYPx: number,
  widthPx: number,
  heightPx: number,
  radiusPx: number,
  strokeColor: string,
  strokeWidthPx: number,
): void {
  traceRoundedRectPath(context, leftXPx, topYPx, widthPx, heightPx, radiusPx);
  context.strokeStyle = strokeColor;
  context.lineWidth = strokeWidthPx;
  context.stroke();
}

function traceRoundedRectPath(
  context: CanvasRenderingContext2D,
  leftXPx: number,
  topYPx: number,
  widthPx: number,
  heightPx: number,
  radiusPx: number,
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
}
