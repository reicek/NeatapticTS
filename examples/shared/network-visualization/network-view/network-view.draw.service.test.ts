import type { NetworkHiddenColumnAnnotation } from './network-view.topology.utils';
import type {
  InputLabelGroupDefinition,
  NetworkVisualizationSettings,
} from './network-view.types';
import {
  alignInputNodesToDescriptionScenes,
  drawHiddenColumnLabelScenes,
  drawInputGroupLabelBands,
  drawInputNodeDescriptions,
  drawRoundedRect,
  resolveHiddenColumnLabelScenes,
  resolveInputDescriptionScenes,
  resolveInputGroupLabelBandScenes,
} from './network-view.draw.service';

/**
 * Minimal stub for the Canvas 2D rendering context.
 *
 * The network-view drawing helpers only exercise a small subset of the canvas
 * API, so a typed mock with Jest spies is enough to verify draw paths without a
 * real browser surface.
 */
function createStubContext(canvasWidthPx = 1000) {
  const canvas = {
    width: canvasWidthPx,
    height: 600,
    ownerDocument: {
      defaultView: {
        innerWidth: canvasWidthPx,
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

const nodeDimensions = { widthPx: 20, heightPx: 20 };

const inputGroups: readonly InputLabelGroupDefinition[] = [
  {
    label: 'Group A',
    labelLines: ['Group', 'A'],
    tooltipHeading: 'Group A',
    tooltipBodyParagraphs: ['First semantic group.'],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['A1'],
        tooltipHeading: 'A1',
        tooltipBodyParagraphs: ['First channel.'],
      },
    ],
    backgroundColor: '#ff0000',
    orientation: 'vertical',
  },
  {
    label: 'Group B',
    labelLines: ['Group', 'B'],
    tooltipHeading: 'Group B',
    tooltipBodyParagraphs: ['Second semantic group.'],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['B1'],
        tooltipHeading: 'B1',
        tooltipBodyParagraphs: ['Second channel.'],
      },
    ],
    backgroundColor: '#00ff00',
    orientation: 'vertical',
  },
];

const singleInputGroup: readonly InputLabelGroupDefinition[] = [
  {
    label: 'Single group',
    labelLines: ['Single group'],
    tooltipHeading: 'Single group',
    tooltipBodyParagraphs: ['One semantic group.'],
    nodeDescriptionDefinitions: [
      {
        labelLines: ['Channel'],
        tooltipHeading: 'Channel',
        tooltipBodyParagraphs: ['Single channel.'],
      },
    ],
    backgroundColor: '#0000ff',
    orientation: 'vertical',
  },
];

function positionedInputNodes(count: number) {
  return Array.from({ length: count }, (_, nodeIndex) => ({
    node: { index: nodeIndex, type: 'input' as const, bias: 0 },
    xPx: 100,
    yPx: 20 + nodeIndex * 30,
  }));
}

function positionedHiddenNodes(count: number) {
  return Array.from({ length: count }, (_, nodeIndex) => ({
    node: { index: nodeIndex, type: 'hidden' as const, bias: 0 },
    xPx: 200 + nodeIndex * 40,
    yPx: 50 + nodeIndex * 30,
  }));
}

const testSettings: NetworkVisualizationSettings = {
  inputLabelGroupDefinitions: inputGroups,
  fontFamily: 'sans-serif',
};

describe('network-view.draw.service', () => {
  it('resolves one input-group label-band scene per group', () => {
    const inputNodes = positionedInputNodes(2);
    const scenes = resolveInputGroupLabelBandScenes(
      inputNodes,
      nodeDimensions,
      undefined,
      inputGroups,
    );

    expect(scenes.length).toBe(2);
  });

  it('returns empty input-group label-band scenes when there are no input nodes', () => {
    const scenes = resolveInputGroupLabelBandScenes(
      [],
      nodeDimensions,
      undefined,
      inputGroups,
    );

    expect(scenes.length).toBe(0);
  });

  it('resolves one input-description scene per input node', () => {
    const inputNodes = positionedInputNodes(2);
    const scenes = resolveInputDescriptionScenes(
      inputNodes,
      nodeDimensions,
      inputGroups,
    );

    expect(scenes.length).toBe(2);
  });

  it('returns empty input-description scenes when there are no input nodes', () => {
    const scenes = resolveInputDescriptionScenes(
      [],
      nodeDimensions,
      inputGroups,
    );

    expect(scenes.length).toBe(0);
  });

  it('aligns input-node centers to the matching description scene centers', () => {
    const inputNodes = positionedInputNodes(1);
    const descriptionScenes = resolveInputDescriptionScenes(
      inputNodes,
      nodeDimensions,
      singleInputGroup,
    );

    const alignedNodes = alignInputNodesToDescriptionScenes(
      inputNodes,
      descriptionScenes,
    );

    expect(descriptionScenes.length).toBe(1);
    expect(alignedNodes[0]!.yPx).toBe(
      descriptionScenes[0]!.topPx + descriptionScenes[0]!.heightPx * 0.5,
    );
  });

  it('leaves positioned nodes unchanged when description scenes are empty', () => {
    const inputNodes = positionedInputNodes(1);
    const alignedNodes = alignInputNodesToDescriptionScenes(inputNodes, []);

    expect(alignedNodes[0]!.yPx).toBe(inputNodes[0]!.yPx);
  });

  it('draws input-group label bands', () => {
    const { context } = createStubContext();
    const inputNodes = positionedInputNodes(2);
    const scenes = resolveInputGroupLabelBandScenes(
      inputNodes,
      nodeDimensions,
      undefined,
      inputGroups,
    );

    drawInputGroupLabelBands(context, scenes, [], testSettings);

    expect(context.save).toHaveBeenCalled();
    expect(context.restore).toHaveBeenCalled();
  });

  it('draws input-node descriptions', () => {
    const { context } = createStubContext();
    const inputNodes = positionedInputNodes(2);
    const scenes = resolveInputDescriptionScenes(
      inputNodes,
      nodeDimensions,
      inputGroups,
    );

    drawInputNodeDescriptions(context, scenes, [], testSettings);

    expect(context.save).toHaveBeenCalled();
    expect(context.restore).toHaveBeenCalled();
  });

  it('resolves hidden-column label scenes', () => {
    const hiddenNodes = positionedHiddenNodes(3);
    const annotation: NetworkHiddenColumnAnnotation = {
      label: 'Memory',
      labelLines: ['Memory'],
      tooltipHeading: 'Memory column',
      tooltipBodyParagraphs: ['Recurrent memory column.'],
      backgroundColor: '#0000ff',
      nodeIndices: [0, 1, 2],
    };

    const scenes = resolveHiddenColumnLabelScenes(hiddenNodes, nodeDimensions, [
      annotation,
    ]);

    expect(scenes.length).toBe(1);
  });

  it('returns empty hidden-column label scenes for empty annotations', () => {
    const hiddenNodes = positionedHiddenNodes(2);
    const scenes = resolveHiddenColumnLabelScenes(
      hiddenNodes,
      nodeDimensions,
      [],
    );

    expect(scenes.length).toBe(0);
  });

  it('draws hidden-column label scenes', () => {
    const { context } = createStubContext();
    const hiddenNodes = positionedHiddenNodes(2);
    const annotation: NetworkHiddenColumnAnnotation = {
      label: 'Memory',
      labelLines: ['Memory'],
      tooltipHeading: 'Memory column',
      tooltipBodyParagraphs: ['Recurrent memory column.'],
      backgroundColor: '#0000ff',
      nodeIndices: [0, 1],
    };
    const scenes = resolveHiddenColumnLabelScenes(hiddenNodes, nodeDimensions, [
      annotation,
    ]);

    drawHiddenColumnLabelScenes(context, scenes, [], testSettings);

    expect(context.save).toHaveBeenCalled();
    expect(context.restore).toHaveBeenCalled();
  });

  it('draws a rounded rectangle', () => {
    const { context } = createStubContext();

    drawRoundedRect(context, 10, 20, 50, 60, 5, '#ffffff');

    expect(context.beginPath).toHaveBeenCalled();
    expect(context.fill).toHaveBeenCalled();
  });
});
