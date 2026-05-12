import { exportVisualizationGraph } from '../../../../src/architecture/network';
import { buildExampleArchitectureProfileNetwork } from '../../../architectureProfiles';
import {
  drawMazeNetworkVisualization,
  resolveMazeArchitectureLabel,
  resolveMazeInputLabelGroupDefinitions,
  resolveMazeNetworkCanvasDimensions,
  resolveMazeVisualizationTopologyPlan,
} from './network-view';

describe('resolveMazeNetworkCanvasDimensions', () => {
  it('grows the network canvas to fill tall panel shelves', () => {
    expect(resolveMazeNetworkCanvasDimensions(700, 900)).toEqual({
      widthPx: 700,
      heightPx: 900,
    });
  });

  it('fits the network canvas to shorter panel shelves instead of forcing a taller aspect ratio', () => {
    expect(resolveMazeNetworkCanvasDimensions(700, 200)).toEqual({
      widthPx: 700,
      heightPx: 200,
    });
  });

  it('sizes the maze canvas from the padded host content box instead of the host shell', () => {
    const network = buildExampleArchitectureProfileNetwork('ascii-maze', 'mlp');
    const visualizationGraph = exportVisualizationGraph(network);
    const hostElement = document.createElement('div');
    const networkCanvasElement = document.createElement('canvas');

    hostElement.style.padding = '8px';
    hostElement.append(networkCanvasElement);
    document.body.append(hostElement);

    Object.defineProperty(hostElement, 'clientWidth', {
      configurable: true,
      value: 700,
    });
    Object.defineProperty(hostElement, 'clientHeight', {
      configurable: true,
      value: 500,
    });
    Object.defineProperty(networkCanvasElement, 'clientWidth', {
      configurable: true,
      value: 684,
    });
    Object.defineProperty(networkCanvasElement, 'clientHeight', {
      configurable: true,
      value: 484,
    });
    Object.defineProperty(networkCanvasElement, 'getContext', {
      configurable: true,
      value: jest.fn(() => null),
    });

    drawMazeNetworkVisualization(
      networkCanvasElement,
      network,
      visualizationGraph,
    );

    expect({
      heightPx: networkCanvasElement.height,
      styleHeight: networkCanvasElement.style.height,
      widthPx: networkCanvasElement.width,
    }).toEqual({
      heightPx: 484,
      styleHeight: '484px',
      widthPx: 684,
    });

    hostElement.remove();
  });
});

describe('resolveMazeInputLabelGroupDefinitions', () => {
  it('maps maze observation groups into the shared visualizer overlay contract', () => {
    expect(
      resolveMazeInputLabelGroupDefinitions().map(
        ({ backgroundColor, label, nodeDescriptionDefinitions }) => ({
          backgroundColor,
          label,
          nodeCount: nodeDescriptionDefinitions.length,
        }),
      ),
    ).toEqual([
      {
        backgroundColor: '#2bd9ff',
        label: 'HEADING',
        nodeCount: 1,
      },
      {
        backgroundColor: '#7bff72',
        label: 'OPENNESS',
        nodeCount: 4,
      },
      {
        backgroundColor: '#ffd166',
        label: 'PROGRESS',
        nodeCount: 1,
      },
    ]);
  });
});

describe('resolveMazeArchitectureLabel', () => {
  it('formats ASCII Maze NARX builders with explicit delay-shelf labels', () => {
    const network = buildExampleArchitectureProfileNetwork(
      'ascii-maze',
      'narx',
    );
    const visualizationGraph = exportVisualizationGraph(network);

    expect(resolveMazeArchitectureLabel(network, visualizationGraph)).toMatch(
      /^6 \| NARX\[i1,o1,\+6\] \| 4\n\(\d+ nodes, \d+ connections\)$/,
    );
  });
});

describe('resolveMazeVisualizationTopologyPlan', () => {
  it('builds dedicated recurrent columns for ASCII Maze LSTM profiles', () => {
    const network = buildExampleArchitectureProfileNetwork(
      'ascii-maze',
      'lstm',
    );
    const visualizationGraph = exportVisualizationGraph(network);
    const topologyPlan = resolveMazeVisualizationTopologyPlan(
      network,
      visualizationGraph,
    );

    expect({
      hiddenColumnLabels: topologyPlan.hiddenColumnAnnotations.map(
        (hiddenColumnAnnotation) => hiddenColumnAnnotation.label,
      ),
      layerSizes: topologyPlan.networkLayers.map(
        (networkLayer) => networkLayer.length,
      ),
    }).toEqual({
      hiddenColumnLabels: [
        'INPUT GATE',
        'FORGET GATE',
        'MEMORY CELL',
        'OUTPUT GATE',
        'OUTPUT BLOCK',
      ],
      layerSizes: [6, 6, 6, 6, 6, 6, 4],
    });
  });
});
