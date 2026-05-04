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
import { resolveNetworkVisualizationTopologyPlan } from './network-view.topology.utils';
import type { VisualNetworkNode } from './network-view.layout.utils';

describe('network-view layout utilities', () => {
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
