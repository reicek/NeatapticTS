import Architect from '../../../../src/architecture/architect/architect';
import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import {
  NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  NETWORK_INPUT_GROUP_PADDING_PX,
  NETWORK_INPUT_GROUP_VERTICAL_GAP_PX,
} from '../network-visualization.constants';
import {
  drawNetworkVisualization,
  resolveNetworkArchitectureLabel,
  resolveNetworkVisualizationFrame,
} from './network-view';
import {
  NETWORK_CENTER_BLUE_RAMP,
  NETWORK_FONT_FAMILY,
  NETWORK_HEADER_TEXT_COLOR,
  NETWORK_HIDDEN_NODE_STROKE_COLOR,
  NETWORK_HOVER_TRANSITION_DURATION_MS,
  NETWORK_LEGEND_BACKGROUND,
  NETWORK_LEGEND_BIAS_TITLE_COLOR,
  NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
  NETWORK_LEGEND_HEADER_COLOR,
  NETWORK_LEGEND_ROW_TEXT_COLOR,
  NETWORK_LEGEND_STROKE_COLOR,
  NETWORK_LIGHT_NEON_RAMP,
  NETWORK_NEON_PALETTE,
  NETWORK_NODE_LABEL_FILL_COLOR,
  NETWORK_OUTPUT_NODE_FILL_COLOR,
  NETWORK_OUTPUT_NODE_STROKE_COLOR,
  NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
  NETWORK_REGULAR_NEON_RAMP,
  NETWORK_UI_CANVAS_BACKGROUND,
} from './network-view.constants';
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
import type {
  InputLabelGroupDefinition,
  NetworkVisualizationSettings,
} from './network-view.types';
import { resolveNetworkVisualizationTopologyPlan } from './network-view.topology.utils';

/**
 * Local network-shape and settings fixtures.
 *
 * The shared visualization domain must not import host modules (zero
 * shared-to-host edges), so the moved test declares the former host fixture
 * values locally with value-identical content built from the in-domain
 * NETWORK_* defaults instead of an external settings module.
 */
const NETWORK_INPUT_SIZE = 9;
const NETWORK_OUTPUT_SIZE = 2;

/**
 * Branded input-label group fixture passed to the shared network visualizer
 * from these tests. Group colors come from the in-domain light neon ramp.
 */
const NETWORK_VIEW_TEST_INPUT_GROUPS: readonly InputLabelGroupDefinition[] = [
  {
    label: 'BIRD STATE',
    labelLines: ['BIRD', 'STATE'],
    tooltipHeading: 'Bird State',
    tooltipBodyParagraphs: [
      "This group tells the network where the bird is and how fast it is already moving. It is the controller's body-state check before any pipe geometry matters.",
      'Bird height says where the bird sits in the tunnel, while vertical speed says whether it is already rising or falling into trouble.',
      'Together these channels answer a control question that every later feature depends on: would a flap correct the current motion, or would it overreact?',
    ],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['Bird height'],
        tooltipHeading: 'Bird Height',
        tooltipBodyParagraphs: [
          "Bird height measures the bird's current vertical position inside the playfield, so the controller knows whether it is drifting toward the floor or ceiling.",
          'It does not say when to flap by itself, but it anchors every comparison with the next and following gaps.',
        ],
      },
      {
        labelLines: ['Vertical speed'],
        tooltipHeading: 'Vertical Speed',
        tooltipBodyParagraphs: [
          'Vertical speed measures how fast the bird is already rising or falling when this frame begins.',
          'That matters because a flap adds to current motion instead of resetting it, so the policy can avoid late oscillating corrections.',
        ],
      },
    ],
    backgroundColor: NETWORK_LIGHT_NEON_RAMP[0],
    orientation: 'vertical',
  },
  {
    label: 'NEXT GAP',
    labelLines: ['NEXT', 'GAP'],
    tooltipHeading: 'Next Gap',
    tooltipBodyParagraphs: [
      'This group describes the very next opening the bird must survive. It turns the obstacle into a compact navigation target instead of a wall of pixels.',
      'Distance tells the controller how much time is left, while offset, top, and bottom explain where the safe corridor sits around the bird.',
      'That separation matters because urgency and alignment are different problems: the policy needs to know both how soon to react and where to steer.',
    ],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['Next pipe distance'],
        tooltipHeading: 'Next Pipe Distance',
        tooltipBodyParagraphs: [
          'Next pipe distance acts like a countdown to the next real steering deadline.',
          'Large values allow calm setup, while small values tell the network that any remaining alignment error must be corrected quickly.',
        ],
      },
      {
        labelLines: ['Next gap offset'],
        tooltipHeading: 'Next Gap Offset',
        tooltipBodyParagraphs: [
          "Next gap offset measures the bird's vertical difference from the center of the next safe opening.",
          'It is one of the clearest steer-here signals because it says both how far off the bird is and in which direction.',
        ],
      },
      {
        labelLines: ['Next gap top'],
        tooltipHeading: 'Next Gap Top',
        tooltipBodyParagraphs: [
          'Next gap top marks the upper boundary of the next opening rather than just its center.',
          'When combined with bird height, it exposes how much ceiling-side safety margin is left.',
        ],
      },
      {
        labelLines: ['Next gap bottom'],
        tooltipHeading: 'Next Gap Bottom',
        tooltipBodyParagraphs: [
          'Next gap bottom marks the lower boundary of the next opening and completes the safe corridor geometry.',
          'When combined with bird height, it exposes how much floor-side safety margin remains.',
        ],
      },
    ],
    backgroundColor: NETWORK_LIGHT_NEON_RAMP[2],
    orientation: 'vertical',
  },
  {
    label: 'LOOK AHEAD',
    labelLines: ['LOOK', 'AHEAD'],
    tooltipHeading: 'Look Ahead',
    tooltipBodyParagraphs: [
      'This group gives the network three planning-oriented signals that complement the immediate next-gap geometry.',
      'The pipe entrance distance crosses zero the moment the bird enters the pipe body and goes negative while the bird is traversing it — a signal the other channels cannot provide.',
      'Gap clearance says how well-centred the bird currently is inside the opening, creating pressure to maintain position rather than drift.',
      'The second-gap offset introduces a lookahead horizon: if the next obstacle sits at a different height the network can start repositioning early instead of reacting late.',
    ],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['Pipe entrance dist'],
        tooltipHeading: 'Pipe Entrance Distance',
        tooltipBodyParagraphs: [
          'Pipe entrance distance measures the signed gap between the front of the bird and the left edge of the next pipe.',
          'The value is positive while the pipe is still ahead, crosses zero when the bird enters, and goes negative while the bird is inside the pipe body.',
          "That negative region gives the controller an unambiguous 'currently traversing' signal that the pipe-exit distance and gap-offset channels alone cannot supply.",
        ],
      },
      {
        labelLines: ['Gap clearance'],
        tooltipHeading: 'Gap Clearance',
        tooltipBodyParagraphs: [
          'Gap clearance measures how centred the bird currently is inside the next gap opening.',
          'A value near +1 means the bird is well inside the safe corridor; a value near 0 means it is on the edge; a negative value means it has already crossed the gap boundary.',
          'Unlike the gap-offset channel, clearance is symmetric around the corridor centre so the controller gets a direct safety margin reading rather than a directional correction signal.',
        ],
      },
      {
        labelLines: ['2nd gap offset'],
        tooltipHeading: 'Second Gap Offset',
        tooltipBodyParagraphs: [
          'Second gap offset measures the signed vertical difference between the bird and the centre of the second upcoming pipe opening.',
          'When the two gaps are at similar heights this channel is near zero and the controller can safely hold position.',
          'When the second gap sits noticeably higher or lower, this signal motivates early repositioning before the first pipe is even cleared — the key missing ingredient for smooth sequential navigation.',
        ],
      },
    ],
    backgroundColor: NETWORK_LIGHT_NEON_RAMP[4],
    orientation: 'vertical',
  },
] as const;

/** Settings object passed to {@link drawNetworkVisualization} in these tests. */
const NETWORK_VIEW_TEST_SETTINGS: NetworkVisualizationSettings = {
  inputLabelGroupDefinitions: NETWORK_VIEW_TEST_INPUT_GROUPS,
  canvasBackground: NETWORK_UI_CANVAS_BACKGROUND,
  palette: {
    currentRunText: NETWORK_NEON_PALETTE.currentRunText,
    statusText: NETWORK_NEON_PALETTE.statusText,
  },
  fontFamily: NETWORK_FONT_FAMILY,
  overlayHiddenBreakpointPx: NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
  hoverTransitionDurationMs: NETWORK_HOVER_TRANSITION_DURATION_MS,
  lightNeonRamp: NETWORK_LIGHT_NEON_RAMP,
  regularNeonRamp: NETWORK_REGULAR_NEON_RAMP,
  centerBlueRamp: NETWORK_CENTER_BLUE_RAMP,
  theme: {
    headerText: NETWORK_HEADER_TEXT_COLOR,
    nodeLabelFill: NETWORK_NODE_LABEL_FILL_COLOR,
    hiddenNodeStroke: NETWORK_HIDDEN_NODE_STROKE_COLOR,
    outputNodeStroke: NETWORK_OUTPUT_NODE_STROKE_COLOR,
    outputNodeFill: NETWORK_OUTPUT_NODE_FILL_COLOR,
    legendBackground: NETWORK_LEGEND_BACKGROUND,
    legendStroke: NETWORK_LEGEND_STROKE_COLOR,
    legendHeader: NETWORK_LEGEND_HEADER_COLOR,
    legendConnectionTitle: NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
    legendBiasTitle: NETWORK_LEGEND_BIAS_TITLE_COLOR,
    legendRowText: NETWORK_LEGEND_ROW_TEXT_COLOR,
  },
};

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
      resolveInputGroupLabelBands(
        NETWORK_INPUT_SIZE,
        NETWORK_VIEW_TEST_INPUT_GROUPS,
      ).map(
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
      resolveInputNodeDescriptionLabels(
        NETWORK_INPUT_SIZE,
        NETWORK_VIEW_TEST_INPUT_GROUPS,
      ).map(
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
      resolveInputDescriptionColumnWidthPx(
        NETWORK_INPUT_SIZE,
        NETWORK_VIEW_TEST_INPUT_GROUPS,
      ),
    ).toBe(118);
  });
});

describe('resolveInputDescriptionScenes', () => {
  it('keeps at least a 4px vertical gap between adjacent description chip outlines', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
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
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
        NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
    );
  });
});

describe('resolveInputGroupLabelBandScenes', () => {
  it('makes each group chip cover its description chips plus group padding', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
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
        NETWORK_INPUT_GROUP_PADDING_PX,
      topPx:
        inputDescriptionScenes[0]!.topPx -
        NETWORK_INPUT_GROUP_PADDING_PX,
    });
  });

  it('keeps a 4px vertical gap between adjacent first-column group chips', () => {
    const inputDescriptionScenes = resolveInputDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
    );
    const alignedInputNodes = alignInputNodesToDescriptionScenes(
      Array.from({ length: NETWORK_INPUT_SIZE }, (_, nodeIndex) => ({
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
      NETWORK_VIEW_TEST_INPUT_GROUPS,
    );

    expect(
      inputGroupLabelBandScenes[1]!.topPx -
        (inputGroupLabelBandScenes[0]!.topPx +
          inputGroupLabelBandScenes[0]!.heightPx),
    ).toBe(NETWORK_INPUT_GROUP_VERTICAL_GAP_PX);
  });
});

function createStubContext(
  canvasWidthPx: number,
  viewportWidthPx: number = canvasWidthPx,
) {
  const canvas = {
    width: canvasWidthPx,
    height: 600,
    ownerDocument: {
      defaultView: {
        innerWidth: viewportWidthPx,
      },
    },
  } as unknown as HTMLCanvasElement;

  const fillTextCalls: Array<{ text: string; x: number; y: number }> = [];
  const context = {
    canvas,
    save: jest.fn(),
    restore: jest.fn(),
    fillRect: jest.fn(),
    strokeRect: jest.fn(),
    clearRect: jest.fn(),
    beginPath: jest.fn(),
    closePath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    quadraticCurveTo: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn(),
    fillText: jest.fn((text: string, x: number, y: number) => {
      fillTextCalls.push({ text, x, y });
    }),
    measureText: jest.fn((text: string) => ({
      width: text.length * 6,
      actualBoundingBoxAscent: 8,
      actualBoundingBoxDescent: 2,
    })),
    setLineDash: jest.fn(),
    arc: jest.fn(),
    rect: jest.fn(),
    clip: jest.fn(),
    getImageData: jest.fn(() => ({ data: [] })),
    putImageData: jest.fn(),
    createLinearGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
    createPattern: jest.fn(() => null),
    drawImage: jest.fn(),
    translate: jest.fn(),
    rotate: jest.fn(),
    scale: jest.fn(),
    transform: jest.fn(),
    setTransform: jest.fn(),
    resetTransform: jest.fn(),
    createImageData: jest.fn(() => ({ data: [] })),
    getTransform: jest.fn(() => [1, 0, 0, 1, 0, 0]),
    isPointInPath: jest.fn(() => false),
    isPointInStroke: jest.fn(() => false),
    fillStyle: '',
    strokeStyle: '',
    font: '',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    lineWidth: 1,
    lineCap: 'butt',
    globalAlpha: 1,
    shadowBlur: 0,
    shadowColor: '',
  } as unknown as CanvasRenderingContext2D;

  return { context, fillTextCalls };
}

describe('drawNetworkVisualization overlay visibility', () => {
  it('produces input-group label band scenes when the host canvas is wide despite a narrow viewport', () => {
    const network = new Network(
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      { seed: 42 },
    );
    const { context } = createStubContext(900, 600);
    const positionedScene = drawNetworkVisualization(
      context,
      network,
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      undefined,
      undefined,
      NETWORK_VIEW_TEST_SETTINGS,
    );

    expect(positionedScene.inputGroupLabelBandScenes.length).toBeGreaterThan(0);
  });

  it('produces input-description scenes when the host canvas is wide despite a narrow viewport', () => {
    const network = new Network(
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      { seed: 42 },
    );
    const { context } = createStubContext(900, 600);
    const positionedScene = drawNetworkVisualization(
      context,
      network,
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      undefined,
      undefined,
      NETWORK_VIEW_TEST_SETTINGS,
    );

    expect(positionedScene.inputDescriptionScenes.length).toBeGreaterThan(0);
  });
});

describe('drawNetworkVisualization activation labels', () => {
  it('renders the input node activation value after a forward pass', () => {
    const network = Architect.perceptron(1, 1, 1);
    network.nodes.forEach((node) => {
      node.bias = 0;
    });
    network.connections.forEach((connection) => {
      connection.weight = 1;
    });
    network.activate([0.75]);
    const inputActivation = network.nodes[0].activation.toFixed(2);

    const { context, fillTextCalls } = createStubContext(900, 900);
    drawNetworkVisualization(context, network, 1, 1);

    expect(
      fillTextCalls.some((fillTextCall) =>
        fillTextCall.text.includes(inputActivation),
      ),
    ).toBe(true);
  });
});

describe('NetworkVisualizationSettings threading', () => {
  it('renders no input-group bands when default empty settings are threaded', () => {
    const network = new Network(
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      { seed: 42 },
    );
    const { context } = createStubContext(900, 900);
    const settings: NetworkVisualizationSettings = {
      inputLabelGroupDefinitions: [],
    };

    const scene = (drawNetworkVisualization as any)(
      context,
      network,
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      undefined,
      undefined,
      settings,
    );

    expect(scene.inputGroupLabelBandScenes.length).toBe(0);
  });

  it('threads custom input groups into the rendered positioned scene', () => {
    const network = new Network(3, 1, { seed: 42 });
    const { context } = createStubContext(900, 900);
    const settings: NetworkVisualizationSettings = {
      inputLabelGroupDefinitions: CUSTOM_INPUT_LABEL_GROUP_DEFINITIONS,
    };

    const scene = (drawNetworkVisualization as any)(
      context,
      network,
      3,
      1,
      undefined,
      undefined,
      settings,
    );

    expect(scene.inputGroupLabelBandScenes.length).toBe(2);
    expect(scene.inputGroupLabelBandScenes[0]?.backgroundColor).toBe(
      '#2bd9ff',
    );
  });

  it('paints the canvas background from custom settings', () => {
    const network = new Network(1, 1, { seed: 42 });
    const { context } = createStubContext(900, 900);
    const settings: NetworkVisualizationSettings = {
      canvasBackground: '#1a1a1a',
      inputLabelGroupDefinitions: [],
    };

    (drawNetworkVisualization as any)(
      context,
      network,
      1,
      1,
      undefined,
      undefined,
      settings,
    );

    expect(context.fillStyle).toBe('#1a1a1a');
  });

  it('threads a custom palette into the resolved color scales', () => {
    const network = new Network(1, 1, { seed: 42 });
    const { context } = createStubContext(900, 900);
    const settings: NetworkVisualizationSettings = {
      palette: { currentRunText: '#abcdef' } as any,
      inputLabelGroupDefinitions: [],
    };

    const frame = (resolveNetworkVisualizationFrame as any)(
      context,
      network,
      NETWORK_INPUT_SIZE,
      NETWORK_OUTPUT_SIZE,
      undefined,
      settings,
    );

    expect(frame.colorScales.connectionScale.aboveTierColor).toBe('#abcdef');
  });
});
