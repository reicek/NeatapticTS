/**
 * Basic smoke tests for the shared network-view renderer.
 *
 * Note: Full browser rendering tests are integrated via Flappy Bird and ASCII Maze demos.
 * These tests verify type contracts and basic module structure.
 */

import {
  positionNetworkNodes,
  centerPositionedNodesInDrawableArea,
} from './network-view.layout.utils';
import { renderNetworkView } from './network-view';
import { resolveNetworkVisualizationTopologyPlan } from './network-view.topology.utils';
import type { VisualNetworkNode } from './network-view.layout.utils';
import type { VisualizationGraphV1 } from '../../architecture/network';

describe('network-view layout utilities', () => {
  describe('renderNetworkView()', () => {
    it('splits forward hidden chains into separate horizontal columns', () => {
      const canvas = {
        width: 760,
        height: 400,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const graph: VisualizationGraphV1 = {
        version: 1,
        nodes: [
          { id: 1, role: 'input', bias: 0 },
          { id: 2, role: 'input', bias: 0 },
          { id: 3, role: 'hidden', bias: 0 },
          { id: 4, role: 'hidden', bias: 0 },
          { id: 5, role: 'output', bias: 0 },
        ],
        edges: [
          { from: 1, to: 3, weight: 0.5, kind: 'forward' },
          { from: 2, to: 3, weight: 0.5, kind: 'forward' },
          { from: 3, to: 4, weight: 0.5, kind: 'forward' },
          { from: 4, to: 5, weight: 0.5, kind: 'forward' },
        ],
        io: {
          inputNodeIds: [1, 2],
          outputNodeIds: [5],
        },
        metadata: {
          mode: 'acyclic',
        },
      };

      const frame = renderNetworkView(canvas, graph);
      const hiddenColumns = new Set(
        frame.positionedNodes
          .filter((positionedNode) => positionedNode.type === 'hidden')
          .map((positionedNode) => Math.round(positionedNode.centerXPx)),
      );

      expect(hiddenColumns.size).toBe(2);
    });
  });

  describe('positionNetworkNodes()', () => {
    it('positions nodes in layers left to right', () => {
      const layers: VisualNetworkNode[][] = [
        [{ index: 0, type: 'input', bias: 0 }],
        [{ index: 1, type: 'hidden', bias: 0 }],
        [{ index: 2, type: 'output', bias: 0 }],
      ];

      const positioned = positionNetworkNodes(
        layers,
        32, // leftPadding
        32, // topPadding
        336, // drawableWidth (400 - 32 - 32)
        236, // drawableHeight (300 - 32 - 32)
        16, // nodeLayoutPadding
        { widthPx: 24, heightPx: 24 },
        8, // inputLayerTargetGap
      );

      expect(positioned.length).toBe(3);
      expect(positioned[0].type).toBe('input');
      expect(positioned[2].type).toBe('output');
      expect(positioned[0].centerXPx < positioned[2].centerXPx).toBe(true);
    });
  });

  describe('centerPositionedNodesInDrawableArea()', () => {
    it('centers nodes horizontally', () => {
      const nodes = [
        {
          index: 0,
          type: 'input' as const,
          centerXPx: 50,
          centerYPx: 100,
          widthPx: 24,
          heightPx: 24,
          bias: 0,
        },
        {
          index: 1,
          type: 'output' as const,
          centerXPx: 350,
          centerYPx: 100,
          widthPx: 24,
          heightPx: 24,
          bias: 0,
        },
      ];

      const centered = centerPositionedNodesInDrawableArea(nodes, 400);

      const avgX = (centered[0].centerXPx + centered[1].centerXPx) / 2;
      expect(avgX).toBeCloseTo(200, 1); // centered in 400-width area
    });

    it('respects drawable-area left padding when centering nodes', () => {
      const nodes = [
        {
          index: 0,
          type: 'input' as const,
          centerXPx: 80,
          centerYPx: 100,
          widthPx: 24,
          heightPx: 24,
          bias: 0,
        },
        {
          index: 1,
          type: 'output' as const,
          centerXPx: 220,
          centerYPx: 100,
          widthPx: 24,
          heightPx: 24,
          bias: 0,
        },
      ];

      const centered = centerPositionedNodesInDrawableArea(nodes, 320, 160);

      const avgX = (centered[0].centerXPx + centered[1].centerXPx) / 2;
      expect(avgX).toBeCloseTo(320, 1);
    });
  });

  describe('resolveNetworkVisualizationTopologyPlan()', () => {
    it('returns acyclic mode for undefined network', () => {
      const plan = resolveNetworkVisualizationTopologyPlan(undefined, 2, 1);

      expect(plan.networkLayers.length).toBe(2); // input + output
      expect(plan.topologyMode).toBe('acyclic');
    });
  });
});
