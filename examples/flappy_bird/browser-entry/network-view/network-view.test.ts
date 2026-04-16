import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import {
  FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX,
  FLAPPY_NETWORK_INPUT_SIZE,
} from '../../constants/constants';
import { resolveNetworkArchitectureLabel } from './network-view';
import {
  alignInputNodesToDescriptionScenes,
  resolveInputDescriptionScenes,
  resolveInputGroupLabelBandScenes,
} from './network-view.draw.service';
import {
  resolveInputDescriptionColumnWidthPx,
  resolveInputGroupLabelBands,
  resolveInputNodeDescriptionLabels,
} from './network-view.labels.utils';

describe('resolveNetworkArchitectureLabel', () => {
  it('prefers explicit runtime IO role sizes over caller-provided fallback hints', () => {
    const network = new Network(3, 2, { seed: 812 });
    const architectureLabel = resolveNetworkArchitectureLabel(network, 38, 99);

    expect(architectureLabel).toMatch(/^3 \| - \| 2\n\(\d+ nodes, \d+ connections\)$/);
  });

  it('adds a recurrent scheduling line when the runtime advertises recurrent execution', () => {
    const network = new Network(1, 1, {
      seed: 813,
      enforceAcyclic: false,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const hiddenNode = new Node('hidden');

    network.nodes = [inputNode, hiddenNode, outputNode];
    network.connections.slice().forEach((connection) => {
      network.disconnect(connection.from, connection.to);
    });
    network.connect(inputNode, hiddenNode);
    network.connect(hiddenNode, hiddenNode);
    network.connect(hiddenNode, outputNode);
    network.activate([1]);

    const architectureLabel = resolveNetworkArchitectureLabel(network, 1, 1);

    expect(architectureLabel).toContain('schedule: recurrent via compiled schedule');
  });

  it('adds a warning line when acyclic scheduling falls back because of a cycle', () => {
    const network = new Network(1, 1, {
      seed: 814,
      enforceAcyclic: true,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const hiddenNode = new Node('hidden');

    network.nodes.push(hiddenNode);
    inputNode.connect(hiddenNode);
    hiddenNode.connect(outputNode);
    outputNode.connect(hiddenNode);
    Network.rebuildConnections(network);
    network.activate([1]);

    const architectureLabel = resolveNetworkArchitectureLabel(network, 1, 1);

    expect(architectureLabel).toContain('warning: acyclic via cycle fallback');
  });
});

describe('resolveInputGroupLabelBands', () => {
  it('splits the simplified Flappy controller inputs into semantic vertical groups', () => {
    expect(
      resolveInputGroupLabelBands(FLAPPY_NETWORK_INPUT_SIZE).map(
        ({
          endNodeIndex,
          label,
          labelLines,
          orientation,
          startNodeIndex,
          tooltipBodyParagraphs,
          tooltipHeading,
        }) => ({
          endNodeIndex,
          label,
          labelLines,
          orientation,
          startNodeIndex,
          tooltipBodyParagraphCount: tooltipBodyParagraphs.length,
          tooltipHeading,
        }),
      ),
    ).toEqual([
      {
        endNodeIndex: 1,
        label: 'BIRD STATE',
        labelLines: ['BIRD', 'STATE'],
        orientation: 'vertical',
        startNodeIndex: 0,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Bird State',
      },
      {
        endNodeIndex: 5,
        label: 'NEXT GAP',
        labelLines: ['NEXT', 'GAP'],
        orientation: 'vertical',
        startNodeIndex: 2,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Next Gap',
      },
      {
        endNodeIndex: 7,
        label: 'LOOKAHEAD',
        labelLines: ['LOOK', 'AHEAD'],
        orientation: 'vertical',
        startNodeIndex: 6,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Lookahead',
      },
      {
        endNodeIndex: FLAPPY_NETWORK_INPUT_SIZE - 1,
        label: 'CONTROL',
        labelLines: ['CONT', 'ROL'],
        orientation: 'vertical',
        startNodeIndex: 8,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Control Pressure',
      },
    ]);
  });
});

describe('resolveInputNodeDescriptionLabels', () => {
  it('adds one compact chip label for each simplified Flappy input row', () => {
    expect(
      resolveInputNodeDescriptionLabels(FLAPPY_NETWORK_INPUT_SIZE).map(
        ({ labelLines, nodeIndex, tooltipBodyParagraphs, tooltipHeading }) => ({
          labelLines,
          nodeIndex,
          tooltipBodyParagraphCount: tooltipBodyParagraphs.length,
          tooltipHeading,
        }),
      ),
    ).toEqual([
      {
        labelLines: ['Bird height'],
        nodeIndex: 0,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Bird Height',
      },
      {
        labelLines: ['Vertical speed'],
        nodeIndex: 1,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Vertical Speed',
      },
      {
        labelLines: ['Next pipe distance'],
        nodeIndex: 2,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Next Pipe Distance',
      },
      {
        labelLines: ['Next gap offset'],
        nodeIndex: 3,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Next Gap Offset',
      },
      {
        labelLines: ['Next gap top'],
        nodeIndex: 4,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Next Gap Top',
      },
      {
        labelLines: ['Next gap bottom'],
        nodeIndex: 5,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Next Gap Bottom',
      },
      {
        labelLines: ['Lookahead distance'],
        nodeIndex: 6,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Lookahead Distance',
      },
      {
        labelLines: ['Lookahead offset'],
        nodeIndex: 7,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Lookahead Offset',
      },
      {
        labelLines: ['Time pressure'],
        nodeIndex: 8,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Time Pressure',
      },
      {
        labelLines: ['Gap clearance'],
        nodeIndex: 9,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Gap Clearance',
      },
      {
        labelLines: ['Needed climb'],
        nodeIndex: 10,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Needed Climb',
      },
      {
        labelLines: ['Gap transition'],
        nodeIndex: 11,
        tooltipBodyParagraphCount: 2,
        tooltipHeading: 'Gap Transition',
      },
    ]);
  });
});

describe('resolveInputDescriptionColumnWidthPx', () => {
  it('reserves only the width required by the widest simplified Flappy chip label', () => {
    expect(resolveInputDescriptionColumnWidthPx(FLAPPY_NETWORK_INPUT_SIZE)).toBe(118);
  });
});

describe('resolveInputDescriptionScenes', () => {
  it('keeps at least a 4px vertical gap between adjacent description chip outlines', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      {
        widthPx: 10,
        heightPx: 10,
      },
    );

    const minimumSceneGapPx = inputDescriptionScenes.reduce(
      (currentMinimumGapPx, inputDescriptionScene, inputDescriptionSceneIndex) => {
        if (inputDescriptionSceneIndex === 0) {
          return currentMinimumGapPx;
        }

        const previousInputDescriptionScene =
          inputDescriptionScenes[inputDescriptionSceneIndex - 1];
        if (!previousInputDescriptionScene) {
          return currentMinimumGapPx;
        }

        return Math.min(
          currentMinimumGapPx,
          inputDescriptionScene.topPx -
            (previousInputDescriptionScene.topPx +
              previousInputDescriptionScene.heightPx),
        );
      },
      Number.POSITIVE_INFINITY,
    );

    expect(minimumSceneGapPx).toBe(4);
  });

  it('realigns input-node centers to the description chip centers so the third column matches the second', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      {
        widthPx: 10,
        heightPx: 10,
      },
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      inputDescriptionScenes,
    );

    expect(
      alignedInputNodes.map((alignedInputNode) => alignedInputNode.yPx),
    ).toEqual(
      inputDescriptionScenes.map(
        (inputDescriptionScene) =>
          inputDescriptionScene.topPx + inputDescriptionScene.heightPx * 0.5,
      ),
    );
  });
});

describe('resolveInputGroupLabelBandScenes', () => {
  it('makes each group chip cover its description chips plus group padding', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      {
        widthPx: 10,
        heightPx: 10,
      },
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      inputDescriptionScenes,
    );
    const inputGroupLabelBandScenes = resolveInputGroupLabelBandScenes(
      alignedInputNodes,
      {
        widthPx: 10,
        heightPx: 10,
      },
      inputDescriptionScenes,
    );

    expect({
      bottomPx:
        inputGroupLabelBandScenes[0]!.topPx +
        inputGroupLabelBandScenes[0]!.heightPx,
      topPx: inputGroupLabelBandScenes[0]!.topPx,
    }).toEqual({
      bottomPx:
        inputDescriptionScenes[1]!.topPx +
        inputDescriptionScenes[1]!.heightPx +
        FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
      topPx:
        inputDescriptionScenes[0]!.topPx -
        FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
    });
  });

  it('keeps a 4px vertical gap between adjacent first-column group chips', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      {
        widthPx: 10,
        heightPx: 10,
      },
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: FLAPPY_NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
        node: {
          index: nodeIndex,
          type: 'input',
          bias: 0,
        },
        xPx: 120,
        yPx: 20 + nodeIndex * 10,
      })),
      inputDescriptionScenes,
    );
    const inputGroupLabelBandScenes = resolveInputGroupLabelBandScenes(
      alignedInputNodes,
      {
        widthPx: 10,
        heightPx: 10,
      },
      inputDescriptionScenes,
    );

    expect(
      inputGroupLabelBandScenes[1]!.topPx -
        (inputGroupLabelBandScenes[0]!.topPx +
          inputGroupLabelBandScenes[0]!.heightPx),
    ).toBe(FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX);
  });
});