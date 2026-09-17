/** @jest-environment jsdom */

import { Network } from '../../../src/browser-entry.ts';
import {
  drawNetworkLODFromFrame,
  resolveNetworkLODFrame,
  shouldUseNetworkLOD,
  type NetworkVisualizationLODResolvedFrame,
} from './network-visualization.lod.service';

describe('shared network LOD service', () => {
  const buildStubContext = (): CanvasRenderingContext2D => {
    const canvas = document.createElement('canvas');
    canvas.width = 900;
    canvas.height = 600;

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

  const buildDenseNetwork = (): Network =>
    new Network(70, 2, {
      minHidden: 4000,
      topologyIntent: 'feed-forward',
      seed: 42,
    });

  describe('shouldUseNetworkLOD', () => {
    it('returns false when the network is undefined', () => {
      expect(shouldUseNetworkLOD(undefined, 2048)).toBe(false);
    });

    it('returns false when the hidden-node count is below the threshold', () => {
      const network = new Network(10, 2);

      expect(shouldUseNetworkLOD(network, 2048)).toBe(false);
    });

    it('returns true when the hidden-node count exceeds the threshold', () => {
      const network = buildDenseNetwork();

      expect(shouldUseNetworkLOD(network, 2048)).toBe(true);
    });
  });

  describe('resolveNetworkLODFrame', () => {
    it('returns a branded LOD frame tied to the source network', () => {
      const context = buildStubContext();
      const network = buildDenseNetwork();

      const frame = resolveNetworkLODFrame(context, network, 70, 2);

      expect((frame as NetworkVisualizationLODResolvedFrame).__networkLod).toBe(
        true,
      );
      expect(frame.sourceNetwork).toBe(network);
    });

    it('collapses hidden nodes above the threshold into a small cluster count', () => {
      const context = buildStubContext();
      const network = buildDenseNetwork();

      const frame = resolveNetworkLODFrame(context, network, 70, 2);

      expect(frame.positionedScene.positionedNodes.length).toBeLessThan(
        network.nodes.length * 0.5,
      );
    });

    it('keeps every input and output node visible in the abstract scene', () => {
      const context = buildStubContext();
      const network = buildDenseNetwork();

      const frame = resolveNetworkLODFrame(context, network, 70, 2);
      const ioNodeCount = frame.positionedScene.positionedNodes.filter(
        (positionedNode: { node: { type: string } }) =>
          positionedNode.node.type === 'input' ||
          positionedNode.node.type === 'output',
      ).length;

      expect(ioNodeCount).toBe(72);
    });
  });

  describe('drawNetworkLODFromFrame', () => {
    it('expands the hovered hidden node ego neighborhood into extra positioned nodes', () => {
      const context = buildStubContext();
      const network = buildDenseNetwork();
      const hoveredNodeIndex = network.nodes[100]?.index ?? -1;

      const frame = resolveNetworkLODFrame(context, network, 70, 2);
      const abstractScene = drawNetworkLODFromFrame(context, frame, []);
      const hoveredScene = drawNetworkLODFromFrame(context, frame, [
        hoveredNodeIndex,
      ]);

      expect({
        moreDetailed:
          hoveredScene.positionedNodes.length >
          abstractScene.positionedNodes.length,
        hoveredNodePresent: hoveredScene.positionedNodes.some(
          (positionedNode: { node: { index: number } }) =>
            positionedNode.node.index === hoveredNodeIndex,
        ),
      }).toEqual({
        moreDetailed: true,
        hoveredNodePresent: true,
      });
    });
  });
});
