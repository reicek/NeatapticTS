import type {
  PositionedNetworkNodeLike,
  VisualNetworkConnectionLike,
} from '../network-visualization.types';
import {
  drawBiasNodesLayer,
  drawNetworkColorLegend,
  drawNetworkVisualizationHeader,
  drawWeightedConnectionsLayer,
} from './visualization.draw.service';
import type {
  DynamicColorScale,
  NetworkVisualizationColorScales,
} from './visualization.types';

/**
 * Minimal CanvasRenderingContext2D stub for Node/Jest.
 *
 * The helper only relies on a small surface of the canvas API.  Spies let us
 * verify call paths without a real browser window.
 */
function createStubContext(canvasWidthPx = 1500) {
  const canvas = {
    width: canvasWidthPx,
    height: 800,
    ownerDocument: {
      defaultView: {
        innerWidth: canvasWidthPx,
      },
    },
  } as unknown as HTMLCanvasElement;

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
    fillText: jest.fn(),
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

  return context;
}

const nodeDimensions = { widthPx: 20, heightPx: 20 };

function positionedNodes(count: number): PositionedNetworkNodeLike[] {
  return Array.from({ length: count }, (_, nodeIndex) => ({
    node: { index: nodeIndex, type: 'hidden', bias: nodeIndex * 0.1 },
    xPx: 200 + nodeIndex * 40,
    yPx: 50 + nodeIndex * 30,
  }));
}

const dynamicScale: DynamicColorScale = {
  minimumValue: -1,
  maximumValue: 1,
  tiers: [
    { upperBound: 0, color: '#ff0000' },
    { upperBound: 1, color: '#00ff00' },
  ],
  aboveTierColor: '#ffffff',
};

const colorScales: NetworkVisualizationColorScales = {
  connectionScale: dynamicScale,
  biasScale: dynamicScale,
};

function buildPositionByNodeIndex(nodes: PositionedNetworkNodeLike[]) {
  return new Map(nodes.map((node) => [node.node.index, node]));
}

describe('visualization.draw.service', () => {
  it('draws weighted connections', () => {
    const context = createStubContext();
    const fromNodes = positionedNodes(2);
    const toNodes = positionedNodes(2);
    const positionByNodeIndex = buildPositionByNodeIndex([
      ...fromNodes,
      ...toNodes,
    ]);
    const runtimeConnections: VisualNetworkConnectionLike[] = [
      { from: { index: 0 }, to: { index: 1 }, weight: 0.5, enabled: true },
    ];

    drawWeightedConnectionsLayer(
      context,
      runtimeConnections,
      positionByNodeIndex,
      dynamicScale,
    );

    expect(context.stroke).toHaveBeenCalled();
  });

  it('draws bias nodes layer', () => {
    const context = createStubContext();
    const nodes = positionedNodes(2);

    drawBiasNodesLayer(context, nodes, nodeDimensions, dynamicScale);

    expect(context.fillRect).toHaveBeenCalled();
    expect(context.fillText).toHaveBeenCalled();
  });

  it('draws network visualization header', () => {
    const context = createStubContext();

    drawNetworkVisualizationHeader(context, 'Test network');

    expect(context.fillText).toHaveBeenCalled();
  });

  it('draws network color legend when the canvas is wide enough', () => {
    const context = createStubContext(1500);

    drawNetworkColorLegend(context, '2 inputs → 1 output', colorScales);

    expect(context.fillText).toHaveBeenCalled();
  });

  it('skips legend drawing on a narrow canvas', () => {
    const context = createStubContext(400);

    drawNetworkColorLegend(context, '2 inputs → 1 output', colorScales);

    expect(context.fillText).not.toHaveBeenCalled();
    expect(context.fillRect).not.toHaveBeenCalled();
  });
});
