import Architect from '../../../../src/architecture/architect/architect';
import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import {
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
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
import type { InputLabelGroupDefinition } from './network-view.types';
import { resolveNetworkVisualizationTopologyPlan } from './network-view.topology.utils';

const CUSTOM_INPUT_LABEL_GROUP_DEFINITIONS: readonly InputLabelGroupDefinition[] =
  [
    {
      label: 'SENSE',
      labelLines: ['SENSE'],
      tooltipHeading: 'Sense',
      tooltipBodyParagraphs: ['Custom sense group.'],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Bearing'],
          tooltipHeading: 'Bearing',
          tooltipBodyParagraphs: ['Bearing channel.'],
        },
      ],
      backgroundColor: '#2bd9ff',
      orientation: 'vertical',
    },
    {
      label: 'ACTION',
      labelLines: ['ACT', 'ION'],
      tooltipHeading: 'Action',
      tooltipBodyParagraphs: ['Custom action-prep group.'],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Open north'],
          tooltipHeading: 'Open North',
          tooltipBodyParagraphs: ['North channel.'],
        },
        {
          labelLines: ['Open south'],
          tooltipHeading: 'Open South',
          tooltipBodyParagraphs: ['South channel.'],
        },
      ],
      backgroundColor: '#7bff72',
      orientation: 'vertical',
    },
  ] as const;

const WIDE_INPUT_LABEL_GROUP_DEFINITIONS: readonly InputLabelGroupDefinition[] =
  [
    {
      label: 'WIDE',
      labelLines: ['WIDE'],
      tooltipHeading: 'Wide',
      tooltipBodyParagraphs: ['Wide custom overlay group.'],
      nodeDescriptionDefinitions: [
        {
          labelLines: ['Corridor clearance east'],
          tooltipHeading: 'Corridor Clearance East',
          tooltipBodyParagraphs: ['Wide custom overlay description.'],
        },
      ],
      backgroundColor: '#ffd166',
      orientation: 'vertical',
    },
  ] as const;

describe('resolveNetworkArchitectureLabel', () => {
  it('prefers explicit runtime IO role sizes over caller-provided fallback hints', () => {
    const network = new Network(3, 2, { seed: 812 });
    const architectureLabel = resolveNetworkArchitectureLabel(network, 38, 99);

    expect(architectureLabel).toMatch(
      /^3 \| - \| 2\n\(\d+ nodes, \d+ connections\)$/,
    );
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

    expect(architectureLabel).toContain(
      'schedule: recurrent via compiled schedule',
    );
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

  it('formats LSTM builders with explicit block sizes instead of an inferred hidden mass', () => {
    const network = Architect.lstm(8, 3, 2, { inputToOutput: false });
    const architectureLabel = resolveNetworkArchitectureLabel(network, 8, 2);

    expect(architectureLabel).toMatch(
      /^8 \| LSTM\[3\] \| 2\n\(\d+ nodes, \d+ connections\)$/,
    );
  });

  it('formats NARX builders with explicit delay-shelf sizes and dense hidden carry-through', () => {
    const network = Architect.narx(1, [3], 1, 2, 1);
    const architectureLabel = resolveNetworkArchitectureLabel(network, 1, 1);

    expect(architectureLabel).toMatch(
      /^1 \| NARX\[i2,o1,\+3\] \| 1\n\(\d+ nodes, \d+ connections\)$/,
    );
  });
});

describe('resolveNetworkVisualizationTopologyPlan', () => {
  it('splits one LSTM block into separate semantic hidden columns', () => {
    const network = Architect.lstm(8, 2, 2, { inputToOutput: false });
    const topologyPlan = resolveNetworkVisualizationTopologyPlan(network, 8, 2);

    expect({
      hiddenColumnLabels: topologyPlan.hiddenColumnAnnotations.map(
        (hiddenColumnAnnotation) => hiddenColumnAnnotation.label,
      ),
      layerSizes: topologyPlan.networkLayers.map(
        (networkLayer) => networkLayer.length,
      ),
    }).toStrictEqual({
      hiddenColumnLabels: [
        'INPUT GATE',
        'FORGET GATE',
        'MEMORY CELL',
        'OUTPUT GATE',
        'OUTPUT BLOCK',
      ],
      layerSizes: [8, 2, 2, 2, 2, 2, 2],
    });
  });

  it('keeps layered fallback columns when only part of the hidden graph is cyclic', () => {
    const network = new Network(1, 1, {
      seed: 815,
      enforceAcyclic: false,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const setupHiddenNode = new Node('hidden');
    const recurrentBridgeNode = new Node('hidden');
    const recurrentMemoryNode = new Node('hidden');

    network.nodes.push(
      setupHiddenNode,
      recurrentBridgeNode,
      recurrentMemoryNode,
    );
    network.connections.slice().forEach((connection) => {
      network.disconnect(connection.from, connection.to);
    });
    inputNode.connect(setupHiddenNode);
    setupHiddenNode.connect(recurrentBridgeNode);
    recurrentBridgeNode.connect(recurrentMemoryNode);
    recurrentMemoryNode.connect(recurrentBridgeNode);
    recurrentMemoryNode.connect(outputNode);
    Network.rebuildConnections(network);

    const topologyPlan = resolveNetworkVisualizationTopologyPlan(network, 1, 1);

    expect({
      hiddenColumnLabels: topologyPlan.hiddenColumnAnnotations.map(
        (hiddenColumnAnnotation) => hiddenColumnAnnotation.label,
      ),
      layerSizes: topologyPlan.networkLayers.map(
        (networkLayer) => networkLayer.length,
      ),
    }).toStrictEqual({
      hiddenColumnLabels: [],
      layerSizes: [1, 1, 2, 1],
    });
  });

  it('attaches educational LSTM tooltip copy to each recurrent role column', () => {
    const network = Architect.lstm(8, 2, 2, { inputToOutput: false });
    const topologyPlan = resolveNetworkVisualizationTopologyPlan(network, 8, 2);

    expect(
      topologyPlan.hiddenColumnAnnotations.map(
        ({ label, tooltipBodyParagraphs, tooltipHeading }) => ({
          label,
          tooltipHeading,
          firstSentence: tooltipBodyParagraphs[0],
          paragraphCount: tooltipBodyParagraphs.length,
        }),
      ),
    ).toStrictEqual([
      {
        label: 'INPUT GATE',
        tooltipHeading: 'LSTM Input Gate',
        firstSentence:
          'The input gate decides how much new evidence is allowed to write into the cell state on this step.',
        paragraphCount: 2,
      },
      {
        label: 'FORGET GATE',
        tooltipHeading: 'LSTM Forget Gate',
        firstSentence:
          'The forget gate decides how much of the previous cell state survives into the next step.',
        paragraphCount: 2,
      },
      {
        label: 'MEMORY CELL',
        tooltipHeading: 'LSTM Memory Cell',
        firstSentence:
          'The memory cell is the long-lived state lane that carries accumulated context across time.',
        paragraphCount: 2,
      },
      {
        label: 'OUTPUT GATE',
        tooltipHeading: 'LSTM Output Gate',
        firstSentence:
          'The output gate decides how much of the cell state is revealed to the rest of the network right now.',
        paragraphCount: 2,
      },
      {
        label: 'OUTPUT BLOCK',
        tooltipHeading: 'LSTM Output Block',
        firstSentence:
          'This column is the exposed state emitted after the memory cell has passed through the output gate.',
        paragraphCount: 2,
      },
    ]);
  });

  it('places NARX input and output delay shelves around dense hidden layers', () => {
    const network = Architect.narx(1, [3], 1, 2, 1);
    const topologyPlan = resolveNetworkVisualizationTopologyPlan(network, 1, 1);

    expect({
      hiddenColumnLabels: topologyPlan.hiddenColumnAnnotations.map(
        (hiddenColumnAnnotation) => hiddenColumnAnnotation.label,
      ),
      layerSizes: topologyPlan.networkLayers.map(
        (networkLayer) => networkLayer.length,
      ),
    }).toStrictEqual({
      hiddenColumnLabels: ['IN t-1', 'IN t-2', 'HIDDEN 1', 'OUT t-1'],
      layerSizes: [1, 1, 1, 3, 1, 1],
    });
  });

  it('attaches educational NARX tooltip copy to delay shelves and dense hidden columns', () => {
    const network = Architect.narx(1, [3], 1, 2, 1);
    const topologyPlan = resolveNetworkVisualizationTopologyPlan(network, 1, 1);

    expect(
      topologyPlan.hiddenColumnAnnotations.map(
        ({ label, tooltipBodyParagraphs, tooltipHeading }) => ({
          label,
          tooltipHeading,
          firstSentence: tooltipBodyParagraphs[0],
          paragraphCount: tooltipBodyParagraphs.length,
        }),
      ),
    ).toStrictEqual([
      {
        label: 'IN t-1',
        tooltipHeading: 'NARX Input Delay t-1',
        firstSentence:
          'This shelf stores one older external input, so the network reads recent history explicitly instead of burying it inside a gated cell.',
        paragraphCount: 2,
      },
      {
        label: 'IN t-2',
        tooltipHeading: 'NARX Input Delay t-2',
        firstSentence:
          'This shelf stores one older external input, so the network reads recent history explicitly instead of burying it inside a gated cell.',
        paragraphCount: 2,
      },
      {
        label: 'HIDDEN 1',
        tooltipHeading: 'Dense Hidden Layer 1',
        firstSentence:
          'This column is not a delay shelf; it is a standard nonlinear workspace that combines the remembered taps into prediction features.',
        paragraphCount: 2,
      },
      {
        label: 'OUT t-1',
        tooltipHeading: 'NARX Output Delay t-1',
        firstSentence:
          'This shelf stores one older model output, giving the network an explicit autoregressive trace of its own recent behavior.',
        paragraphCount: 2,
      },
    ]);
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
        endNodeIndex: 8,
        label: 'LOOK AHEAD',
        labelLines: ['LOOK', 'AHEAD'],
        orientation: 'vertical',
        startNodeIndex: 6,
        tooltipBodyParagraphCount: 4,
        tooltipHeading: 'Look Ahead',
      },
    ]);
  });

  it('supports custom grouped input definitions for shared demo overlays', () => {
    expect(
      resolveInputGroupLabelBands(3, CUSTOM_INPUT_LABEL_GROUP_DEFINITIONS).map(
        ({ endNodeIndex, label, startNodeIndex, tooltipHeading }) => ({
          endNodeIndex,
          label,
          startNodeIndex,
          tooltipHeading,
        }),
      ),
    ).toEqual([
      {
        endNodeIndex: 0,
        label: 'SENSE',
        startNodeIndex: 0,
        tooltipHeading: 'Sense',
      },
      {
        endNodeIndex: 2,
        label: 'ACTION',
        startNodeIndex: 1,
        tooltipHeading: 'Action',
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
        labelLines: ['Pipe entrance dist'],
        nodeIndex: 6,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Pipe Entrance Distance',
      },
      {
        labelLines: ['Gap clearance'],
        nodeIndex: 7,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Gap Clearance',
      },
      {
        labelLines: ['2nd gap offset'],
        nodeIndex: 8,
        tooltipBodyParagraphCount: 3,
        tooltipHeading: 'Second Gap Offset',
      },
    ]);
  });

  it('supports custom per-node descriptions for shared demo overlays', () => {
    expect(
      resolveInputNodeDescriptionLabels(
        3,
        CUSTOM_INPUT_LABEL_GROUP_DEFINITIONS,
      ).map(({ labelLines, nodeIndex, tooltipHeading }) => ({
        labelLines,
        nodeIndex,
        tooltipHeading,
      })),
    ).toEqual([
      {
        labelLines: ['Bearing'],
        nodeIndex: 0,
        tooltipHeading: 'Bearing',
      },
      {
        labelLines: ['Open north'],
        nodeIndex: 1,
        tooltipHeading: 'Open North',
      },
      {
        labelLines: ['Open south'],
        nodeIndex: 2,
        tooltipHeading: 'Open South',
      },
    ]);
  });
});

describe('resolveInputDescriptionColumnWidthPx', () => {
  it('reserves only the width required by the widest simplified Flappy chip label', () => {
    expect(
      resolveInputDescriptionColumnWidthPx(FLAPPY_NETWORK_INPUT_SIZE),
    ).toBe(118);
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
      (
        currentMinimumGapPx,
        inputDescriptionScene,
        inputDescriptionSceneIndex,
      ) => {
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

  it('keeps custom group bands separated from wider shared-demo description chips', () => {
    const positionedNodes = [
      {
        node: {
          index: 0,
          type: 'input',
          bias: 0,
        },
        xPx: 160,
        yPx: 60,
      },
    ];
    const nodeDimensions = {
      widthPx: 10,
      heightPx: 10,
    };
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      positionedNodes,
      nodeDimensions,
      WIDE_INPUT_LABEL_GROUP_DEFINITIONS,
    );
    const inputGroupLabelBandScenes = resolveInputGroupLabelBandScenes(
      positionedNodes,
      nodeDimensions,
      inputDescriptionScenes,
      WIDE_INPUT_LABEL_GROUP_DEFINITIONS,
    );

    expect(
      (inputGroupLabelBandScenes[0]?.leftPx ?? 0) +
        (inputGroupLabelBandScenes[0]?.widthPx ?? 0),
    ).toBe(
      (inputDescriptionScenes[0]?.leftPx ?? 0) -
        FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
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
