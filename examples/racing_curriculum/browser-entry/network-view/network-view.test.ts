/** @jest-environment jsdom */

import { Network } from '../../../../src/browser-entry.ts';
import * as sharedNetworkView from '../../../flappy_bird/browser-entry/network-view/network-view';
import {
  drawRacingNetworkVisualization,
  drawRacingNetworkVisualizationFromFrame,
  resolveRacingArchitectureLabel,
  resolveRacingInputLabelGroupDefinitions,
  resolveRacingNetworkCanvasDimensions,
  resolveRacingNetworkVisualizationFrame,
} from './network-view';
import { RACING_NETWORK_CONNECTION_LAYER_STYLE } from './network-view.constants';

describe('racing network-view adapter', () => {
  const buildStubContext = (): CanvasRenderingContext2D => {
    const canvas = document.createElement('canvas');
    canvas.width = 900;
    canvas.height = 240;
    const context = {
      canvas,
      clearRect: jest.fn(),
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      bezierCurveTo: jest.fn(),
      quadraticCurveTo: jest.fn(),
      ellipse: jest.fn(),
      stroke: jest.fn(),
      rect: jest.fn(),
      fill: jest.fn(),
      fillRect: jest.fn(),
      arc: jest.fn(),
      fillText: jest.fn(),
      strokeText: jest.fn(),
      measureText: jest.fn().mockReturnValue({ width: 0 }),
      drawImage: jest.fn(),
      save: jest.fn(),
      restore: jest.fn(),
      translate: jest.fn(),
      scale: jest.fn(),
      rotate: jest.fn(),
      closePath: jest.fn(),
      clip: jest.fn(),
      setLineDash: jest.fn(),
      getLineDash: jest.fn().mockReturnValue([]),
      setTransform: jest.fn(),
      strokeRect: jest.fn(),
      strokeStyle: '',
      lineWidth: 1,
      fillStyle: '',
      font: '',
      textAlign: 'center',
      textBaseline: 'middle',
    } as unknown as CanvasRenderingContext2D;
    return context;
  };

  const attachStubContext = (
    canvas: HTMLCanvasElement,
    context: CanvasRenderingContext2D,
  ): void => {
    jest.spyOn(canvas, 'getContext').mockImplementation(() => context);
  };

  it('returns an empty scene when the canvas has no 2D context', () => {
    const canvas = document.createElement('canvas');
    jest.spyOn(canvas, 'getContext').mockImplementation(() => null);

    const scene = drawRacingNetworkVisualization(canvas, new Network(70, 2));

    expect(scene.positionedNodes).toEqual([]);
    expect(scene.inputDescriptionScenes).toEqual([]);
  });

  it('resolves a frame for a 70/2 racing controller network', () => {
    const context = buildStubContext();
    const network = new Network(70, 2);

    const frame = resolveRacingNetworkVisualizationFrame(context, network);

    expect(frame.positionedScene.positionedNodes.length).toBeGreaterThan(0);
    expect(frame.positionedScene.inputDescriptionScenes.length).toBe(70);
  });

  it('draws a resolved frame and returns a positioned scene', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const context = buildStubContext();
    attachStubContext(canvas, context);

    const resolvedFrame = resolveRacingNetworkVisualizationFrame(
      context,
      new Network(70, 2),
    );
    const scene = drawRacingNetworkVisualizationFromFrame(
      context,
      resolvedFrame,
      [],
    );

    expect(scene.positionedNodes.length).toBeGreaterThan(0);
    expect(context.beginPath).toHaveBeenCalled();
  });

  it('forwards the bright racing connection-layer style to the shared visualizer', () => {
    const drawSpy = jest.spyOn(
      sharedNetworkView,
      'drawResolvedNetworkVisualization',
    );
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const context = buildStubContext();
    attachStubContext(canvas, context);

    const resolvedFrame = resolveRacingNetworkVisualizationFrame(
      context,
      new Network(70, 2),
    );
    drawRacingNetworkVisualizationFromFrame(context, resolvedFrame, []);

    expect(drawSpy.mock.calls[0]?.[3]).toEqual(
      RACING_NETWORK_CONNECTION_LAYER_STYLE,
    );

    drawSpy.mockRestore();
  });

  it('keeps the canvas aspect ratio near 0.75', () => {
    const { widthPx, heightPx } = resolveRacingNetworkCanvasDimensions(320, 0);

    expect(heightPx / widthPx).toBeCloseTo(0.75, 1);
  });

  it('exports seven input label groups for the Tier 1 racing layout', () => {
    const groups = resolveRacingInputLabelGroupDefinitions();

    expect(groups.length).toBe(7);
    expect(
      groups.reduce(
        (sum, group) => sum + group.nodeDescriptionDefinitions.length,
        0,
      ),
    ).toBe(70);
  });

  it('resolves a concise architecture label for the racing network', () => {
    const label = resolveRacingArchitectureLabel(new Network(70, 2));

    expect(label).toContain('70');
    expect(label).toContain('2');
    expect(label).toContain('nodes');
  });

  it('resolves input-description scenes using the canvas width, not the viewport width', () => {
    const context = buildStubContext();
    context.canvas.width = 900;
    const originalInnerWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', {
      value: 600,
      configurable: true,
    });

    const frame = resolveRacingNetworkVisualizationFrame(
      context,
      new Network(70, 2),
    );

    Object.defineProperty(window, 'innerWidth', {
      value: originalInnerWidth,
      configurable: true,
    });

    expect(frame.positionedScene.inputDescriptionScenes.length).toBe(70);
  });

  it('renders node activation values as fillText after activation', () => {
    const network = new Network(2, 1, {
      minHidden: 1,
      seed: 42,
      topologyIntent: 'unconstrained',
    });
    network.nodes.forEach((node) => {
      node.bias = 0;
    });
    network.connections.forEach((connection) => {
      connection.weight = 0;
    });
    network.selfconns.forEach((connection) => {
      connection.weight = 0;
    });
    network.gates.forEach((connection) => {
      connection.weight = 0;
    });
    network.activate([0.5, 0.5]);

    const context = buildStubContext();
    const frame = resolveRacingNetworkVisualizationFrame(context, network);
    drawRacingNetworkVisualizationFromFrame(context, frame, []);

    const activationValues = new Set(
      network.nodes.map((node) => node.activation),
    );
    const renderedTexts = (context.fillText as jest.Mock).mock.calls.map(
      (call) => call[0],
    );

    expect(
      renderedTexts.some((text) =>
        activationValues.has(Number.parseFloat(text)),
      ),
    ).toBe(true);
  });
});
