import {
  processKahnQueue,
  resolveStableNodeTieBreakValue,
  seedProcessingQueue,
} from './network.topology.loop.utils';
import type {
  TopologyBuildContext,
  TopologyNode,
} from './network.topology.utils.types';

describe('network topology loop utility chapter', () => {
  describe('seedProcessingQueue', () => {
    describe('given a hidden node is missing from the in-degree table', () => {
      it('treats the node as zero in-degree and seeds it into the sorted processing queue', () => {
        // Arrange
        const hiddenNode = createTopologyNode({
          geneId: 5,
          index: 2,
          type: 'hidden',
        });
        const inputNode = createTopologyNode({
          geneId: 2,
          index: 0,
          type: 'input',
        });
        const buildContext = createBuildContext({
          networkNodes: [hiddenNode, inputNode],
          inDegreeEntries: [[inputNode, 3]],
        });

        // Act
        seedProcessingQueue(buildContext);

        // Assert
        expect(
          buildContext.processingQueue.map((nodeEntry) => nodeEntry.geneId),
        ).toEqual([2, 5]);
      });
    });

    describe('given a hidden node still has remaining in-degree', () => {
      it('keeps that node out of the initial processing queue', () => {
        // Arrange
        const blockedHiddenNode = createTopologyNode({
          geneId: 9,
          index: 1,
          type: 'hidden',
        });
        const inputNode = createTopologyNode({
          geneId: 3,
          index: 0,
          type: 'input',
        });
        const buildContext = createBuildContext({
          networkNodes: [blockedHiddenNode, inputNode],
          inDegreeEntries: [
            [blockedHiddenNode, 1],
            [inputNode, 4],
          ],
        });

        // Act
        seedProcessingQueue(buildContext);

        // Assert
        expect(
          buildContext.processingQueue.map((nodeEntry) => nodeEntry.geneId),
        ).toEqual([3]);
      });
    });
  });

  describe('processKahnQueue', () => {
    describe('given a node has both a self-loop and one downstream edge', () => {
      it('skips the self-loop while advancing the downstream node into the next activation wave', () => {
        // Arrange
        const sourceNode = createTopologyNode({
          geneId: 11,
          index: 0,
          type: 'input',
        });
        const targetNode = createTopologyNode({
          geneId: 15,
          index: 1,
          type: 'hidden',
        });
        sourceNode.connections.out = [
          createOutgoingConnection(sourceNode),
          createOutgoingConnection(targetNode),
        ];
        const buildContext = createBuildContext({
          networkNodes: [sourceNode, targetNode],
          inDegreeEntries: [
            [sourceNode, 0],
            [targetNode, 1],
          ],
          processingQueue: [sourceNode],
        });

        // Act
        processKahnQueue(buildContext);

        // Assert
        expect({
          topoOrder: buildContext.topoOrder.map(
            (nodeEntry) => nodeEntry.geneId,
          ),
          activationSteps: buildContext.activationSteps,
          remainingInDegree: buildContext.inDegreeByNode.get(targetNode),
          remainingQueueSize: buildContext.processingQueue.length,
        }).toEqual({
          topoOrder: [11, 15],
          activationSteps: [[11], [15]],
          remainingInDegree: 0,
          remainingQueueSize: 0,
        });
      });
    });

    describe('given an outgoing edge reduces in-degree without unlocking the downstream node', () => {
      it('updates the tracked in-degree without queueing the downstream node yet', () => {
        // Arrange
        const sourceNode = createTopologyNode({
          geneId: 21,
          index: 0,
          type: 'input',
        });
        const targetNode = createTopologyNode({
          geneId: 25,
          index: 1,
          type: 'hidden',
        });
        sourceNode.connections.out = [createOutgoingConnection(targetNode)];
        const buildContext = createBuildContext({
          networkNodes: [sourceNode, targetNode],
          inDegreeEntries: [
            [sourceNode, 0],
            [targetNode, 2],
          ],
          processingQueue: [sourceNode],
        });

        // Act
        processKahnQueue(buildContext);

        // Assert
        expect({
          topoOrder: buildContext.topoOrder.map(
            (nodeEntry) => nodeEntry.geneId,
          ),
          activationSteps: buildContext.activationSteps,
          remainingInDegree: buildContext.inDegreeByNode.get(targetNode),
          remainingQueueSize: buildContext.processingQueue.length,
        }).toEqual({
          topoOrder: [21],
          activationSteps: [[21]],
          remainingInDegree: 1,
          remainingQueueSize: 0,
        });
      });
    });
  });

  describe('resolveStableNodeTieBreakValue', () => {
    describe('given a node omits geneId but keeps a finite index', () => {
      it('uses the node index as the stable tie-break value', () => {
        // Arrange
        const node = createTopologyNode({
          index: 27,
          type: 'hidden',
        });

        // Act
        const tieBreakValue = resolveStableNodeTieBreakValue(node);

        // Assert
        expect(tieBreakValue).toBe(27);
      });
    });

    describe('given a node omits both finite geneId and finite index values', () => {
      it('falls back to the largest safe integer so valid ids always sort first', () => {
        // Arrange
        const node = createTopologyNode({
          geneId: Number.NaN,
          index: Number.POSITIVE_INFINITY,
          type: 'hidden',
        });

        // Act
        const tieBreakValue = resolveStableNodeTieBreakValue(node);

        // Assert
        expect(tieBreakValue).toBe(Number.MAX_SAFE_INTEGER);
      });
    });
  });
});

type MinimalOutgoingConnection = {
  to: TopologyNode;
};

type TopologyNodeOverrides = {
  geneId?: number;
  index?: number;
  type: string;
  outgoingConnections?: MinimalOutgoingConnection[];
};

type BuildContextOptions = {
  networkNodes: TopologyNode[];
  inDegreeEntries?: Array<[TopologyNode, number]>;
  processingQueue?: TopologyNode[];
};

function createTopologyNode(overrides: TopologyNodeOverrides): TopologyNode {
  return {
    geneId: overrides.geneId,
    index: overrides.index,
    type: overrides.type,
    connections: {
      out: overrides.outgoingConnections ?? [],
    },
  } as unknown as TopologyNode;
}

function createBuildContext(
  options: BuildContextOptions,
): TopologyBuildContext {
  return {
    network: {
      nodes: options.networkNodes,
    } as unknown as TopologyBuildContext['network'],
    internalTopologyProps: {
      _topoOrder: null,
    },
    inDegreeByNode: new Map(options.inDegreeEntries ?? []),
    processingQueue: options.processingQueue ?? [],
    activationSteps: [],
    topoOrder: [],
  } as TopologyBuildContext;
}

function createOutgoingConnection(
  targetNode: TopologyNode,
): TopologyNode['connections']['out'][number] {
  return {
    to: targetNode,
  } as unknown as TopologyNode['connections']['out'][number];
}
