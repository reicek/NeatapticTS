/**
 * Basic smoke tests for the shared network-view renderer.
 *
 * Note: Full browser rendering tests are integrated via the browser-test harness.
 * These tests verify type contracts and basic module structure.
 */

import { describe, expect, it } from '@jest/globals';
import {
  positionNetworkNodes,
  centerPositionedNodesInDrawableArea,
} from './network-view.layout.utils';
import { renderNetworkView } from './network-view';
import {
  resolveNetworkVisualizationLayers,
  resolveNetworkVisualizationTopologyPlan,
} from './network-view.topology.utils';
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

    it('draws valid edges and bias indicators while skipping edges whose endpoints are missing', () => {
      const drawCalls: string[] = [];
      const context = createMockCanvasContext(drawCalls);
      const canvas = {
        width: 320,
        height: 240,
        getContext: () => context,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(canvas, {
        version: 1,
        nodes: [
          { id: 1, role: 'input', bias: 0.5 },
          { id: 2, role: 'output', bias: 0 },
        ],
        edges: [
          { from: 1, to: 2, weight: -0.4, kind: 'forward' },
          { from: 1, to: 2, weight: 0.6, kind: 'forward' },
          { from: 99, to: 2, weight: 0.9, kind: 'forward' },
        ],
        io: {
          inputNodeIds: [1],
          outputNodeIds: [2],
        },
      });

      expect({
        arcCalls: drawCalls.filter((callName) => callName === 'arc').length,
        lineToCalls: drawCalls.filter((callName) => callName === 'lineTo')
          .length,
        topologyMode: frame.topologyMode,
      }).toEqual({
        arcCalls: 3,
        lineToCalls: 2,
        topologyMode: 'recurrent',
      });
    });

    it('keeps hidden-only graphs in a single centered layer when no explicit io nodes exist', () => {
      const canvas = {
        width: 320,
        height: 240,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(canvas, {
        version: 1,
        nodes: [
          { id: 10, role: 'hidden', bias: 0.1 },
          { id: 20, role: 'hidden', bias: 0.2 },
        ],
        edges: [{ from: 20, to: 10, weight: 0.4, kind: 'recurrent' }],
        io: {
          inputNodeIds: [],
          outputNodeIds: [],
        },
      });

      expect(
        new Set(
          frame.positionedNodes.map(
            (positionedNode) => positionedNode.centerXPx,
          ),
        ).size,
      ).toBe(1);
    });

    it('defaults missing node biases to zero and promotes hidden depth without input seeding', () => {
      const canvas = {
        width: 360,
        height: 240,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(canvas, {
        version: 1,
        nodes: [
          { id: 1, role: 'input' },
          { id: 2, role: 'hidden' },
          { id: 3, role: 'hidden' },
          { id: 4, role: 'output' },
        ],
        edges: [
          { from: 2, to: 3, weight: 0.4, kind: 'forward' },
          { from: 3, to: 4, weight: 0.7, kind: 'forward' },
        ],
        io: {
          inputNodeIds: [1],
          outputNodeIds: [4],
        },
      });

      expect({
        biases: frame.positionedNodes.map(
          (positionedNode) => positionedNode.bias,
        ),
        hiddenColumns: new Set(
          frame.positionedNodes
            .filter((positionedNode) => positionedNode.type === 'hidden')
            .map((positionedNode) => positionedNode.centerXPx),
        ).size,
      }).toEqual({
        biases: [0, 0, 0, 0],
        hiddenColumns: 2,
      });
    });

    it('applies custom render options when a full options object is supplied', () => {
      const canvas = {
        width: 400,
        height: 300,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(
        canvas,
        {
          version: 1,
          nodes: [
            { id: 1, role: 'input', bias: 0 },
            { id: 2, role: 'output', bias: 0 },
          ],
          edges: [{ from: 1, to: 2, weight: 0.5, kind: 'forward' }],
          io: { inputNodeIds: [1], outputNodeIds: [2] },
        },
        {
          nodeDimensions: { widthPx: 32, heightPx: 32 },
          panelPaddingPx: {
            topPx: 10,
            rightPx: 10,
            bottomPx: 10,
            leftPx: 10,
          },
          colorScales: {
            weightPositive: '#fff',
            weightNegative: '#000',
            activationHot: '#f00',
            activationCold: '#0ff',
            bias: '#0f0',
          },
        },
      );

      const inputNode = frame.positionedNodes.find(
        (node) => node.type === 'input',
      );
      expect(inputNode?.widthPx).toBe(32);
      expect(inputNode?.heightPx).toBe(32);
    });

    it('applies partial render options and falls back to defaults for missing fields', () => {
      const canvas = {
        width: 400,
        height: 300,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(
        canvas,
        {
          version: 1,
          nodes: [
            { id: 1, role: 'input', bias: 0 },
            { id: 2, role: 'output', bias: 0 },
          ],
          edges: [{ from: 1, to: 2, weight: 0.5, kind: 'forward' }],
          io: { inputNodeIds: [1], outputNodeIds: [2] },
        },
        {
          panelPaddingPx: {
            topPx: 16,
            rightPx: 16,
            bottomPx: 16,
            leftPx: 16,
          },
        },
      );

      const inputNode = frame.positionedNodes.find(
        (node) => node.type === 'input',
      );
      expect(inputNode?.widthPx).toBe(24);
      expect(inputNode?.heightPx).toBe(24);
    });

    it('falls back to default panel padding when options omit panelPaddingPx', () => {
      const canvas = {
        width: 400,
        height: 300,
        getContext: () => null,
      } as unknown as HTMLCanvasElement;

      const frame = renderNetworkView(
        canvas,
        {
          version: 1,
          nodes: [
            { id: 1, role: 'input', bias: 0 },
            { id: 2, role: 'output', bias: 0 },
          ],
          edges: [{ from: 1, to: 2, weight: 0.5, kind: 'forward' }],
          io: { inputNodeIds: [1], outputNodeIds: [2] },
        },
        {
          nodeDimensions: { widthPx: 20, heightPx: 20 },
        },
      );

      // Default padding is 32px on all sides; with 400px width and 20px nodes,
      // the leftmost node x should be 32 (default left padding).
      const inputNode = frame.positionedNodes.find(
        (node) => node.type === 'input',
      );
      expect(inputNode?.centerXPx).toBeGreaterThanOrEqual(32);
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
    it('returns the original array when there are no positioned nodes to center', () => {
      const positionedNodes: Parameters<
        typeof centerPositionedNodesInDrawableArea
      >[0] = [];

      expect(centerPositionedNodesInDrawableArea(positionedNodes, 240)).toBe(
        positionedNodes,
      );
    });

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

  describe('positionNetworkNodes()', () => {
    it('clamps single-node layers inside the drawable vertical bounds', () => {
      const positionedNodes = positionNetworkNodes(
        [[{ index: 7, type: 'hidden', bias: 0.3 }]],
        16,
        12,
        64,
        20,
        8,
        { widthPx: 24, heightPx: 24 },
      );

      expect(positionedNodes).toEqual([
        {
          bias: 0.3,
          centerXPx: 48,
          centerYPx: 12,
          heightPx: 24,
          index: 7,
          type: 'hidden',
          widthPx: 24,
        },
      ]);
    });
  });

  describe('resolveNetworkVisualizationTopologyPlan()', () => {
    it('returns the layer array through the layers-only wrapper', () => {
      expect(resolveNetworkVisualizationLayers(undefined, 2, 1)).toEqual([
        [
          { bias: 0, index: 0, type: 'input' },
          { bias: 0, index: 1, type: 'input' },
        ],
        [{ bias: 0, index: 2, type: 'output' }],
      ]);
    });

    it('returns acyclic mode for undefined network', () => {
      const plan = resolveNetworkVisualizationTopologyPlan(undefined, 2, 1);

      expect(plan.networkLayers.length).toBe(2); // input + output
      expect(plan.topologyMode).toBe('acyclic');
    });

    it('builds ordered feed-forward layers from runtime network nodes', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [
          { bias: 0.4, geneId: 40, index: 4, type: 'output' },
          { bias: 0.2, geneId: 20, index: 2, type: 'hidden' },
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.3, geneId: 30, index: 3, type: 'hidden' },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [],
        networkLayers: [
          [{ bias: 0.1, index: 1, type: 'input' }],
          [
            { bias: 0.2, index: 2, type: 'hidden' },
            { bias: 0.3, index: 3, type: 'hidden' },
          ],
          [{ bias: 0.4, index: 4, type: 'output' }],
        ],
        topologyMode: 'acyclic',
      });
    });

    it('sorts input and output layers by runtime index when each layer has multiple nodes', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [
          { bias: 0.4, geneId: 40, index: 4, type: 'output' },
          { bias: 0.2, geneId: 20, index: 2, type: 'input' },
          { bias: 0.5, geneId: 50, index: 5, type: 'output' },
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 2, 2)).toEqual({
        layerAnnotations: [],
        networkLayers: [
          [
            { bias: 0.1, index: 1, type: 'input' },
            { bias: 0.2, index: 2, type: 'input' },
          ],
          [
            { bias: 0.4, index: 4, type: 'output' },
            { bias: 0.5, index: 5, type: 'output' },
          ],
        ],
        topologyMode: 'acyclic',
      });
    });

    it('omits the hidden layer when the runtime network has no hidden nodes', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.4, geneId: 40, index: 4, type: 'output' },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [],
        networkLayers: [
          [{ bias: 0.1, index: 1, type: 'input' }],
          [{ bias: 0.4, index: 4, type: 'output' }],
        ],
        topologyMode: 'acyclic',
      });
    });

    it('keeps only the hidden layer when a runtime network omits input and output nodes', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [{ bias: 0.2, geneId: 20, index: 2, type: 'hidden' }],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [],
        networkLayers: [[{ bias: 0.2, index: 2, type: 'hidden' }]],
        topologyMode: 'acyclic',
      });
    });

    it('defaults missing runtime node indices to zero when grouping layers', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [
          { bias: 0.1, geneId: 10, type: 'input' },
          { bias: 0.2, geneId: 20, type: 'hidden' },
          { bias: 0.4, geneId: 40, type: 'output' },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [],
        networkLayers: [
          [{ bias: 0.1, index: 0, type: 'input' }],
          [{ bias: 0.2, index: 0, type: 'hidden' }],
          [{ bias: 0.4, index: 0, type: 'output' }],
        ],
        topologyMode: 'acyclic',
      });
    });

    it('sorts same-type nodes by their fallback zero index when runtime indices are missing', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'feed-forward',
        nodes: [
          { bias: 0.3, geneId: 30, index: 3, type: 'hidden' },
          { bias: 0.2, geneId: 20, type: 'hidden' },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [],
        networkLayers: [
          [
            { bias: 0.2, index: 0, type: 'hidden' },
            { bias: 0.3, index: 3, type: 'hidden' },
          ],
        ],
        topologyMode: 'acyclic',
      });
    });

    it('derives recurrent layer annotations from temporal module descriptors', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'unconstrained',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.2, geneId: 20, index: 7, type: 'hidden' },
          { bias: 0.3, geneId: 30, index: 8, type: 'hidden' },
          { bias: 0.4, geneId: 40, index: 9, type: 'output' },
        ],
        recurrentModules: [
          {
            connectionInnovations: [1001],
            kind: 'gru',
            moduleId: 'gru-0',
            moduleLabel: 'Memory Bank',
            nodeGeneIdsByRole: {
              memoryCell: [20],
              output: [30, 999],
            },
          },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [
          {
            label: 'Memory Bank',
            labelLines: ['Memory Bank'],
            nodeIndices: [7, 8],
            tooltipBodyParagraphs: ['Roles: memoryCell, output.'],
            tooltipHeading: 'Memory Bank',
          },
        ],
        networkLayers: [
          [{ bias: 0.1, index: 1, type: 'input' }],
          [
            { bias: 0.2, index: 7, type: 'hidden' },
            { bias: 0.3, index: 8, type: 'hidden' },
          ],
          [{ bias: 0.4, index: 9, type: 'output' }],
        ],
        topologyMode: 'recurrent',
      });
    });

    it('falls back to a kind label when the recurrent module has no explicit label', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'unconstrained',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.2, geneId: 20, index: 7, type: 'hidden' },
          { bias: 0.4, geneId: 40, index: 9, type: 'output' },
        ],
        recurrentModules: [
          {
            connectionInnovations: [1001],
            kind: 'narx-memory',
            moduleId: 'narx-0',
            nodeGeneIdsByRole: {
              delayStep0: [20],
            },
          },
        ],
      });

      expect(
        resolveNetworkVisualizationTopologyPlan(network, 1, 1).layerAnnotations,
      ).toEqual([
        {
          label: 'NARX Memory',
          labelLines: ['NARX Memory'],
          nodeIndices: [7],
          tooltipBodyParagraphs: ['Roles: delayStep0.'],
          tooltipHeading: 'NARX Memory',
        },
      ]);
    });

    it('keeps annotations empty when the recurrent descriptor has no modules', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'unconstrained',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.2, geneId: 20, index: 7, type: 'hidden' },
          { bias: 0.4, geneId: 40, index: 9, type: 'output' },
        ],
        recurrentModules: [],
      });

      expect(
        resolveNetworkVisualizationTopologyPlan(network, 1, 1).layerAnnotations,
      ).toEqual([]);
    });

    it('uses unlabeled recurrent tooltip text when the descriptor exposes no roles', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'unconstrained',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.2, geneId: 20, type: 'hidden' },
          { bias: 0.4, index: 9, type: 'output' },
        ],
        recurrentModules: [
          {
            connectionInnovations: [1001],
            kind: 'lstm',
            moduleId: 'lstm-0',
            nodeGeneIdsByRole: {},
          },
        ],
      });

      expect(resolveNetworkVisualizationTopologyPlan(network, 1, 1)).toEqual({
        layerAnnotations: [
          {
            label: 'LSTM',
            labelLines: ['LSTM'],
            nodeIndices: [],
            tooltipBodyParagraphs: ['Roles: unlabeled module.'],
            tooltipHeading: 'LSTM',
          },
        ],
        networkLayers: [
          [{ bias: 0.1, index: 1, type: 'input' }],
          [{ bias: 0.2, index: 0, type: 'hidden' }],
          [{ bias: 0.4, index: 9, type: 'output' }],
        ],
        topologyMode: 'recurrent',
      });
    });

    it('falls back to empty annotations when temporal introspection throws', () => {
      const network = createTopologyTestNetwork({
        topologyIntent: 'unconstrained',
        nodes: [
          { bias: 0.1, geneId: 10, index: 1, type: 'input' },
          { bias: 0.2, geneId: 20, index: 7, type: 'hidden' },
          { bias: 0.4, geneId: 40, index: 9, type: 'output' },
        ],
        throwOnDescribeTemporalStructure: true,
      });

      expect(
        resolveNetworkVisualizationTopologyPlan(network, 1, 1).layerAnnotations,
      ).toEqual([]);
    });
  });
});

function createMockCanvasContext(
  drawCalls: string[],
): CanvasRenderingContext2D {
  return {
    arc: () => {
      drawCalls.push('arc');
    },
    beginPath: () => {
      drawCalls.push('beginPath');
    },
    fill: () => {
      drawCalls.push('fill');
    },
    fillRect: () => {
      drawCalls.push('fillRect');
    },
    lineTo: () => {
      drawCalls.push('lineTo');
    },
    moveTo: () => {
      drawCalls.push('moveTo');
    },
    stroke: () => {
      drawCalls.push('stroke');
    },
  } as unknown as CanvasRenderingContext2D;
}

function createTopologyTestNetwork(options: {
  nodes: Array<{
    bias: number;
    geneId?: number;
    index?: number;
    type: 'input' | 'hidden' | 'output';
  }>;
  topologyIntent: 'feed-forward' | 'unconstrained';
  recurrentModules?: Array<{
    connectionInnovations: number[];
    kind: 'lstm' | 'gru' | 'narx-memory';
    moduleId: string;
    moduleLabel?: string;
    nodeGeneIdsByRole: Record<string, number[]>;
  }>;
  throwOnDescribeTemporalStructure?: boolean;
}): Parameters<typeof resolveNetworkVisualizationTopologyPlan>[0] {
  return {
    describeTemporalStructure: () => {
      if (options.throwOnDescribeTemporalStructure) {
        throw new Error('temporal descriptor failed');
      }

      return {
        gatedBlocks: [],
        recurrentModules: options.recurrentModules ?? [],
      };
    },
    getTopologyIntent: () => options.topologyIntent,
    nodes: options.nodes,
  } as unknown as Parameters<typeof resolveNetworkVisualizationTopologyPlan>[0];
}
