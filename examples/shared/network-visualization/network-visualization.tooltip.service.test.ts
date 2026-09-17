import type { NetworkVisualizationPositionedScene } from './network-visualization.types';
import { resolveNetworkVisualizationTooltipScene } from './network-visualization.tooltip.service';

const TEST_NODE_DIMENSIONS = {
  widthPx: 20,
  heightPx: 20,
} as const;

describe('resolveNetworkVisualizationTooltipScene', () => {
  it('returns an input tooltip when the pointer is over an input node', () => {
    const positionedScene = createInputPositionedScene();
    const hoveredTooltipScene = resolveNetworkVisualizationTooltipScene(
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
      heading: 'Input Feature',
      kind: 'input',
    });
  });

  it('returns a group tooltip when the pointer is over an input group band', () => {
    const positionedScene = createGroupPositionedScene();
    const firstGroupScene = positionedScene.inputGroupLabelBandScenes[0]!;
    const hoveredTooltipScene = resolveNetworkVisualizationTooltipScene(
      {
        xPx: firstGroupScene.leftPx + 1,
        yPx: firstGroupScene.topPx + 1,
      },
      positionedScene,
    );

    expect({
      heading: hoveredTooltipScene?.heading,
      kind: hoveredTooltipScene?.kind,
    }).toEqual({
      heading: 'Sensor Group',
      kind: 'group',
    });
  });

  it('returns a column tooltip when the pointer is over a hidden node inside that column', () => {
    const positionedScene = createHiddenColumnPositionedScene();
    const hoveredTooltipScene = resolveNetworkVisualizationTooltipScene(
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
      heading: 'Hidden Column',
      kind: 'column',
    });
  });

  it('returns undefined when the pointer is outside every overlay surface', () => {
    const positionedScene = createInputPositionedScene();

    expect(
      resolveNetworkVisualizationTooltipScene(
        {
          xPx: 400,
          yPx: 20,
        },
        positionedScene,
      ),
    ).toBeUndefined();
  });
});

function createInputPositionedScene(): NetworkVisualizationPositionedScene {
  return {
    positionedNodes: [
      {
        node: { index: 0, type: 'input', bias: 0 },
        xPx: 30,
        yPx: 30,
      },
    ],
    nodeDimensions: TEST_NODE_DIMENSIONS,
    inputDescriptionScenes: [
      {
        labelLines: ['Input 0'],
        nodeIndex: 0,
        leftPx: 10,
        topPx: 10,
        widthPx: 20,
        heightPx: 20,
        tooltipHeading: 'Input Feature',
        tooltipBodyParagraphs: [
          'First paragraph about the input.',
          'Second paragraph about the input.',
        ],
      },
    ],
    inputGroupLabelBandScenes: [],
    hiddenColumnLabelScenes: [],
  };
}

function createGroupPositionedScene(): NetworkVisualizationPositionedScene {
  return {
    positionedNodes: [
      {
        node: { index: 0, type: 'input', bias: 0 },
        xPx: 80,
        yPx: 30,
      },
    ],
    nodeDimensions: TEST_NODE_DIMENSIONS,
    inputDescriptionScenes: [
      {
        labelLines: ['Sensor'],
        nodeIndex: 0,
        leftPx: 10,
        topPx: 10,
        widthPx: 30,
        heightPx: 20,
        tooltipHeading: 'Sensor Input',
        tooltipBodyParagraphs: ['A single sensor reading.'],
      },
    ],
    inputGroupLabelBandScenes: [
      {
        label: 'Sensors',
        labelLines: ['Sensors'],
        tooltipHeading: 'Sensor Group',
        tooltipBodyParagraphs: ['All sensor readings grouped together.'],
        leftPx: 5,
        topPx: 5,
        widthPx: 30,
        heightPx: 30,
        backgroundColor: '#aaffaa',
        orientation: 'vertical',
        nodeIndices: [0],
      },
    ],
    hiddenColumnLabelScenes: [],
  };
}

function createHiddenColumnPositionedScene(): NetworkVisualizationPositionedScene {
  return {
    positionedNodes: [
      {
        node: { index: 41, type: 'hidden', bias: 0 },
        xPx: 160,
        yPx: 60,
      },
    ],
    nodeDimensions: TEST_NODE_DIMENSIONS,
    inputDescriptionScenes: [],
    inputGroupLabelBandScenes: [],
    hiddenColumnLabelScenes: [
      {
        labelLines: ['HIDDEN'],
        tooltipHeading: 'Hidden Column',
        tooltipBodyParagraphs: [
          'First paragraph about the hidden column.',
          'Second paragraph about the hidden column.',
        ],
        leftPx: 140,
        topPx: 40,
        widthPx: 40,
        heightPx: 40,
        backgroundColor: '#7fe8ff',
        nodeIndices: [41],
      },
    ],
  };
}
