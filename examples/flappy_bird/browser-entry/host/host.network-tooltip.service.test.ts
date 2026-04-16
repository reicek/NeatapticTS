import { FLAPPY_NETWORK_INPUT_SIZE } from '../../constants/constants';
import type { NetworkVisualizationPositionedScene } from '../browser-entry.types';
import {
  alignInputNodesToDescriptionScenes,
  resolveInputDescriptionScenes,
  resolveInputGroupLabelBandScenes,
} from '../network-view/network-view.draw.service';
import { resolveHoveredNetworkVisualizationTooltipScene } from './host.network-tooltip.service';

const TEST_NODE_DIMENSIONS = {
  widthPx: 10,
  heightPx: 10,
} as const;

describe('resolveHoveredNetworkVisualizationTooltipScene', () => {
  it('reuses the input-description tooltip copy when the pointer is over an input node', () => {
    const positionedScene = createPositionedScene();
    const hoveredTooltipScene = resolveHoveredNetworkVisualizationTooltipScene(
      {
        xPx: positionedScene.positionedNodes[0]!.xPx,
        yPx: positionedScene.positionedNodes[0]!.yPx,
      },
      positionedScene,
    );

    expect({
      bodyParagraphCount: hoveredTooltipScene?.bodyParagraphs.length,
      heading: hoveredTooltipScene?.heading,
      kind: hoveredTooltipScene?.kind,
    }).toEqual({
      bodyParagraphCount: 2,
      heading: 'Bird Height',
      kind: 'input',
    });
  });

  it('resolves the wider group tooltip span when the pointer is over a first-column group band', () => {
    const positionedScene = createPositionedScene();
    const firstGroupScene = positionedScene.inputGroupLabelBandScenes[0]!;
    const hoveredTooltipScene = resolveHoveredNetworkVisualizationTooltipScene(
      {
        xPx: firstGroupScene.leftPx + 1,
        yPx: firstGroupScene.topPx + 1,
      },
      positionedScene,
    );

    expect({
      heading: hoveredTooltipScene?.heading,
      kind: hoveredTooltipScene?.kind,
      spansDescriptionAndNodeColumns:
        (hoveredTooltipScene?.anchorWidthPx ?? 0) > firstGroupScene.widthPx,
    }).toEqual({
      heading: 'Bird State',
      kind: 'group',
      spansDescriptionAndNodeColumns: true,
    });
  });

  it('returns no tooltip when the pointer is outside the input overlay surfaces', () => {
    expect(
      resolveHoveredNetworkVisualizationTooltipScene(
        {
          xPx: 400,
          yPx: 20,
        },
        createPositionedScene(),
      ),
    ).toBeUndefined();
  });
});

function createPositionedScene(): NetworkVisualizationPositionedScene {
  const initialPositionedNodes = Array.from(
    { length: FLAPPY_NETWORK_INPUT_SIZE },
    (_, nodeIndex) => ({
      node: {
        index: nodeIndex,
        type: 'input',
        bias: 0,
      },
      xPx: 120,
      yPx: 20 + nodeIndex * 10,
    }),
  );
  const initialInputDescriptionScenes = resolveInputDescriptionScenes(
    initialPositionedNodes,
    TEST_NODE_DIMENSIONS,
  );
  const alignedPositionedNodes = alignInputNodesToDescriptionScenes(
    initialPositionedNodes,
    initialInputDescriptionScenes,
  );
  const inputDescriptionScenes = resolveInputDescriptionScenes(
    alignedPositionedNodes,
    TEST_NODE_DIMENSIONS,
  );
  const inputGroupLabelBandScenes = resolveInputGroupLabelBandScenes(
    alignedPositionedNodes,
    TEST_NODE_DIMENSIONS,
    inputDescriptionScenes,
  );

  return {
    positionedNodes: alignedPositionedNodes,
    nodeDimensions: TEST_NODE_DIMENSIONS,
    inputDescriptionScenes,
    inputGroupLabelBandScenes,
  };
}