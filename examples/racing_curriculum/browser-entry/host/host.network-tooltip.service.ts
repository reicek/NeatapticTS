/**
 * Racing curriculum network tooltip resolver.
 *
 * Mirrors the Flappy Bird tooltip resolver so the right-sidebar network
 * visualizer can show educational hover annotations for input nodes, hidden
 * columns, and input group bands.
 */

/** Canvas-space point used when resolving network tooltip targets. */
export interface HostCanvasPointLike {
  xPx: number;
  yPx: number;
}

/** Positioned node used for hit testing. */
export interface PositionedNetworkNode {
  node: NetworkNodeLike;
  xPx: number;
  yPx: number;
}

/** Minimal node shape referenced by tooltip scenes. */
export interface NetworkNodeLike {
  index: number;
  type: 'input' | 'hidden' | 'output';
  bias?: number;
}

/** Scene describing one input-description overlay region. */
export interface NetworkInputDescriptionScene {
  nodeIndex: number;
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
}

/** Scene describing one hidden-column label overlay region. */
export interface NetworkHiddenColumnLabelScene {
  nodeIndices: readonly number[];
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
}

/** Scene describing one input group label band overlay region. */
export interface NetworkInputGroupLabelBandScene {
  nodeIndices: readonly number[];
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
}

/** Positioned scene produced by the network visualizer for hover hit testing. */
export interface NetworkVisualizationPositionedScene {
  positionedNodes: readonly PositionedNetworkNode[];
  nodeDimensions: { widthPx: number; heightPx: number };
  inputDescriptionScenes: readonly NetworkInputDescriptionScene[];
  inputGroupLabelBandScenes: readonly NetworkInputGroupLabelBandScene[];
  hiddenColumnLabelScenes?: readonly NetworkHiddenColumnLabelScene[];
}

/** Tooltip scene model resolved from the hovered network overlay target. */
export interface NetworkVisualizationTooltipScene {
  kind: 'group' | 'input' | 'column';
  heading: string;
  bodyParagraphs: readonly string[];
  anchorLeftPx: number;
  anchorTopPx: number;
  anchorWidthPx: number;
  anchorCenterXPx: number;
}

/**
 * Resolves the tooltip scene for the current hovered network overlay target.
 *
 * Hit-test priority, from narrowest to broadest:
 * 1. Hidden-column node hit.
 * 2. Input node hit.
 * 3. Hidden-column label region.
 * 4. Input description region.
 * 5. Input group label band.
 *
 * Input descriptions and input nodes intentionally share the same tooltip copy,
 * while semantic group bands resolve a broader group-level teaching tooltip.
 *
 * @param canvasPoint - Hover point in network-canvas coordinates.
 * @param positionedScene - Rendered positioned scene reused for hover hit testing.
 * @returns Tooltip scene model for the hovered overlay target, or undefined.
 *
 * @example
 * ```ts
 * const scene = renderNetworkView(networkCanvas, graph, options);
 * const tooltip = resolveRacingNetworkTooltipScene({ xPx: 120, yPx: 80 }, scene);
 * if (tooltip) {
 *   showTooltip(tooltip.heading, tooltip.bodyParagraphs, tooltip.anchorCenterXPx, tooltip.anchorTopPx);
 * }
 * ```
 */
export function resolveRacingNetworkTooltipScene(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  return (
    resolveHiddenColumnTooltipSceneFromNode(canvasPoint, positionedScene) ??
    resolveInputTooltipSceneFromInputNode(canvasPoint, positionedScene) ??
    resolveHiddenColumnTooltipSceneFromLabel(canvasPoint, positionedScene) ??
    resolveInputTooltipSceneFromDescription(canvasPoint, positionedScene) ??
    resolveInputTooltipSceneFromGroup(canvasPoint, positionedScene)
  );
}

function resolveHiddenColumnTooltipSceneFromNode(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  const hoveredNodeIndex = resolveHoveredNodeIndexFromCanvasPoint(
    canvasPoint,
    positionedScene,
  );
  if (typeof hoveredNodeIndex !== 'number') {
    return undefined;
  }

  const hoveredHiddenColumnLabelScene =
    positionedScene.hiddenColumnLabelScenes?.find((hiddenColumnLabelScene) =>
      hiddenColumnLabelScene.nodeIndices.includes(hoveredNodeIndex),
    );
  if (!hoveredHiddenColumnLabelScene) {
    return undefined;
  }

  return createHiddenColumnTooltipScene(hoveredHiddenColumnLabelScene);
}

function resolveInputTooltipSceneFromInputNode(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  const hoveredNodeIndex = resolveHoveredNodeIndexFromCanvasPoint(
    canvasPoint,
    positionedScene,
  );
  if (typeof hoveredNodeIndex !== 'number') {
    return undefined;
  }

  const hoveredInputDescriptionScene =
    positionedScene.inputDescriptionScenes.find(
      (inputDescriptionScene) =>
        inputDescriptionScene.nodeIndex === hoveredNodeIndex,
    );
  if (!hoveredInputDescriptionScene) {
    return undefined;
  }

  return createInputTooltipScene(
    hoveredInputDescriptionScene,
    positionedScene.positionedNodes,
    positionedScene.nodeDimensions.widthPx,
  );
}

function resolveHiddenColumnTooltipSceneFromLabel(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  const hoveredHiddenColumnLabelScene =
    positionedScene.hiddenColumnLabelScenes?.findLast(
      (hiddenColumnLabelScene) =>
        canvasPoint.xPx >= hiddenColumnLabelScene.leftPx &&
        canvasPoint.xPx <=
          hiddenColumnLabelScene.leftPx + hiddenColumnLabelScene.widthPx &&
        canvasPoint.yPx >= hiddenColumnLabelScene.topPx &&
        canvasPoint.yPx <=
          hiddenColumnLabelScene.topPx + hiddenColumnLabelScene.heightPx,
    );
  if (!hoveredHiddenColumnLabelScene) {
    return undefined;
  }

  return createHiddenColumnTooltipScene(hoveredHiddenColumnLabelScene);
}

function resolveInputTooltipSceneFromDescription(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  const hoveredInputDescriptionScene =
    positionedScene.inputDescriptionScenes.findLast(
      (inputDescriptionScene) =>
        canvasPoint.xPx >= inputDescriptionScene.leftPx &&
        canvasPoint.xPx <=
          inputDescriptionScene.leftPx + inputDescriptionScene.widthPx &&
        canvasPoint.yPx >= inputDescriptionScene.topPx &&
        canvasPoint.yPx <=
          inputDescriptionScene.topPx + inputDescriptionScene.heightPx,
    );
  if (!hoveredInputDescriptionScene) {
    return undefined;
  }

  return createInputTooltipScene(
    hoveredInputDescriptionScene,
    positionedScene.positionedNodes,
    positionedScene.nodeDimensions.widthPx,
  );
}

function resolveInputTooltipSceneFromGroup(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined {
  const hoveredInputGroupLabelBandScene =
    positionedScene.inputGroupLabelBandScenes.findLast(
      (inputGroupLabelBandScene) =>
        canvasPoint.xPx >= inputGroupLabelBandScene.leftPx &&
        canvasPoint.xPx <=
          inputGroupLabelBandScene.leftPx + inputGroupLabelBandScene.widthPx &&
        canvasPoint.yPx >= inputGroupLabelBandScene.topPx &&
        canvasPoint.yPx <=
          inputGroupLabelBandScene.topPx + inputGroupLabelBandScene.heightPx,
    );
  if (!hoveredInputGroupLabelBandScene) {
    return undefined;
  }

  const rightmostGroupedNodeEdgePx = resolveRightmostNodeEdgePx(
    hoveredInputGroupLabelBandScene.nodeIndices,
    positionedScene.positionedNodes,
    positionedScene.nodeDimensions.widthPx,
  );

  return {
    kind: 'group',
    heading: hoveredInputGroupLabelBandScene.tooltipHeading,
    bodyParagraphs: hoveredInputGroupLabelBandScene.tooltipBodyParagraphs,
    anchorLeftPx: hoveredInputGroupLabelBandScene.leftPx,
    anchorTopPx: hoveredInputGroupLabelBandScene.topPx,
    anchorWidthPx: Math.max(
      hoveredInputGroupLabelBandScene.widthPx,
      rightmostGroupedNodeEdgePx - hoveredInputGroupLabelBandScene.leftPx,
    ),
    anchorCenterXPx:
      hoveredInputGroupLabelBandScene.leftPx +
      hoveredInputGroupLabelBandScene.widthPx * 0.5,
  };
}

function createInputTooltipScene(
  hoveredInputDescriptionScene: NetworkInputDescriptionScene,
  positionedNodes: readonly PositionedNetworkNode[],
  nodeWidthPx: number,
): NetworkVisualizationTooltipScene {
  const hoveredInputNode = positionedNodes.find(
    (positionedNode) =>
      positionedNode.node.index === hoveredInputDescriptionScene.nodeIndex,
  );
  const hoveredInputNodeRightEdgePx = hoveredInputNode
    ? hoveredInputNode.xPx + nodeWidthPx * 0.5
    : hoveredInputDescriptionScene.leftPx +
      hoveredInputDescriptionScene.widthPx;
  const tooltipAnchorWidthPx = Math.max(
    hoveredInputDescriptionScene.widthPx,
    hoveredInputNodeRightEdgePx - hoveredInputDescriptionScene.leftPx,
  );

  return {
    kind: 'input',
    heading: hoveredInputDescriptionScene.tooltipHeading,
    bodyParagraphs: hoveredInputDescriptionScene.tooltipBodyParagraphs,
    anchorLeftPx: hoveredInputDescriptionScene.leftPx,
    anchorTopPx: hoveredInputDescriptionScene.topPx,
    anchorWidthPx: tooltipAnchorWidthPx,
    anchorCenterXPx:
      hoveredInputDescriptionScene.leftPx + tooltipAnchorWidthPx * 0.5,
  };
}

function createHiddenColumnTooltipScene(
  hiddenColumnLabelScene: NetworkHiddenColumnLabelScene,
): NetworkVisualizationTooltipScene {
  return {
    kind: 'column',
    heading: hiddenColumnLabelScene.tooltipHeading,
    bodyParagraphs: hiddenColumnLabelScene.tooltipBodyParagraphs,
    anchorLeftPx: hiddenColumnLabelScene.leftPx,
    anchorTopPx: hiddenColumnLabelScene.topPx,
    anchorWidthPx: hiddenColumnLabelScene.widthPx,
    anchorCenterXPx:
      hiddenColumnLabelScene.leftPx + hiddenColumnLabelScene.widthPx * 0.5,
  };
}

function resolveHoveredNodeIndexFromCanvasPoint(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): number | undefined {
  const halfNodeWidthPx = positionedScene.nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = positionedScene.nodeDimensions.heightPx * 0.5;

  return positionedScene.positionedNodes.findLast((positionedNode) => {
    const nodeLeftPx = positionedNode.xPx - halfNodeWidthPx;
    const nodeRightPx = positionedNode.xPx + halfNodeWidthPx;
    const nodeTopPx = positionedNode.yPx - halfNodeHeightPx;
    const nodeBottomPx = positionedNode.yPx + halfNodeHeightPx;

    return (
      canvasPoint.xPx >= nodeLeftPx &&
      canvasPoint.xPx <= nodeRightPx &&
      canvasPoint.yPx >= nodeTopPx &&
      canvasPoint.yPx <= nodeBottomPx
    );
  })?.node.index;
}

function resolveRightmostNodeEdgePx(
  nodeIndices: readonly number[],
  positionedNodes: readonly PositionedNetworkNode[],
  nodeWidthPx: number,
): number {
  const rightmostNodeEdgePx = nodeIndices.reduce(
    (currentRightmostNodeEdgePx, nodeIndex) => {
      const positionedNode = positionedNodes.find(
        (candidatePositionedNode) =>
          candidatePositionedNode.node.index === nodeIndex,
      );
      if (!positionedNode) {
        return currentRightmostNodeEdgePx;
      }

      return Math.max(
        currentRightmostNodeEdgePx,
        positionedNode.xPx + nodeWidthPx * 0.5,
      );
    },
    Number.NEGATIVE_INFINITY,
  );

  return Number.isFinite(rightmostNodeEdgePx) ? rightmostNodeEdgePx : 0;
}
