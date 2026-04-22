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

  it('reuses the recurrent-column tooltip copy when the pointer is over a hidden node inside that column', () => {
    const positionedScene = createHiddenColumnPositionedScene();
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
      heading: 'LSTM Input Gate',
      kind: 'column',
    });
  });

  it('resolves the recurrent-column tooltip directly from the guide chip hit box', () => {
    const positionedScene = createHiddenColumnPositionedScene();
    const hiddenColumnLabelScene = positionedScene.hiddenColumnLabelScenes?.[0];
    if (!hiddenColumnLabelScene) {
      throw new Error('Expected a hidden-column label scene for the recurrent tooltip test.');
    }
    const hoveredTooltipScene = resolveHoveredNetworkVisualizationTooltipScene(
      {
        xPx: hiddenColumnLabelScene.leftPx + 1,
        yPx: hiddenColumnLabelScene.topPx + 1,
      },
      positionedScene,
    );

    expect({
      anchorWidthPx: hoveredTooltipScene?.anchorWidthPx,
      heading: hoveredTooltipScene?.heading,
      kind: hoveredTooltipScene?.kind,
    }).toEqual({
      anchorWidthPx: hiddenColumnLabelScene.widthPx,
      heading: 'LSTM Input Gate',
      kind: 'column',
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
    hiddenColumnLabelScenes: [],
  };
}

function createHiddenColumnPositionedScene(): NetworkVisualizationPositionedScene {
  return {
    positionedNodes: [
      {
        node: {
          index: 41,
          type: 'hidden',
          bias: 0,
        },
        xPx: 160,
        yPx: 84,
      },
    ],
    nodeDimensions: TEST_NODE_DIMENSIONS,
    inputDescriptionScenes: [],
    inputGroupLabelBandScenes: [],
    hiddenColumnLabelScenes: [
      {
        labelLines: ['INPUT', 'GATE'],
        tooltipHeading: 'LSTM Input Gate',
        tooltipBodyParagraphs: [
          'The input gate decides how much new evidence is allowed to write into the cell state on this step.',
          'Open it wider and the block learns quickly from the present input; close it and the cell protects older memory.',
        ],
        leftPx: 140,
        topPx: 48,
        widthPx: 40,
        heightPx: 18,
        backgroundColor: '#7fe8ff',
        nodeIndices: [41],
      },
    ],
  };
}