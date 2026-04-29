import Group from '../group/group';
import * as methods from '../../methods/methods';
import Node from '../node/node';
import Layer from './layer';

type BatchNormLayer = Layer & { batchNorm?: boolean };
type LayerNormLayer = Layer & { layerNorm?: boolean };
type Conv1dLayer = Layer & {
  conv1d?: { kernelSize: number; padding: number; stride: number };
};
type AttentionLayer = Layer & { attention?: { heads: number } };

describe('layer facade chapter', () => {
  describe('describe()', () => {
    describe('when descriptor values are applied to a dense layer', () => {
      it('merges the label, intent, and metadata onto the layer boundary', () => {
        // Arrange
        const layer = Layer.dense(2);

        // Act
        layer.describe({
          intent: 'output',
          label: 'readoutHead',
          metadata: { stage: 'policy' },
        });

        // Assert
        expect({
          intent: layer.intent,
          label: layer.label,
          metadata: layer.metadata,
        }).toStrictEqual({
          intent: 'output',
          label: 'readoutHead',
          metadata: { family: 'dense', size: 2, stage: 'policy' },
        });
      });
    });

    describe('when descriptor fields are omitted', () => {
      it('leaves the existing boundary descriptor unchanged', () => {
        // Arrange
        const layer = Layer.dense(2, 'output');

        // Act
        layer.describe({});

        // Assert
        expect({
          intent: layer.intent,
          label: layer.label,
          metadata: layer.metadata,
        }).toStrictEqual({
          intent: 'output',
          label: null,
          metadata: { family: 'dense', size: 2 },
        });
      });
    });
  });

  describe('input()', () => {
    describe('when a group connects into a dense layer without an explicit method', () => {
      it('defaults to all-to-all wiring through the layer input surface', () => {
        // Arrange
        const layer = Layer.dense(2);
        const sourceGroup = new Group(2);

        // Act
        const createdConnections = layer.input(sourceGroup);

        // Assert
        expect(createdConnections).toHaveLength(4);
      });
    });
  });

  describe('disconnect()', () => {
    describe('when disconnecting a single node without two-sided cleanup', () => {
      it('removes only the outgoing layer bookkeeping entry', () => {
        // Arrange
        const layer = Layer.dense(2);
        const targetNode = new Node('hidden');
        const forwardConnection = layer.nodes[0].connect(targetNode)[0];
        const reverseConnection = targetNode.connect(layer.nodes[0])[0];
        layer.connections.out.push(forwardConnection);
        layer.connections.in.push(reverseConnection);

        // Act
        layer.disconnect(targetNode);

        // Assert
        expect({
          incomingCount: layer.connections.in.length,
          outgoingCount: layer.connections.out.length,
        }).toEqual({ incomingCount: 1, outgoingCount: 0 });
      });
    });

    describe('when disconnecting a group with two-sided cleanup enabled', () => {
      it('removes both outgoing and incoming layer bookkeeping entries', () => {
        // Arrange
        const layer = Layer.dense(2);
        const targetGroup = new Group(1);
        const firstConnection = layer.nodes[0].connect(targetGroup.nodes[0])[0];
        const secondConnection = layer.nodes[1].connect(
          targetGroup.nodes[0],
        )[0];
        const reverseConnection = targetGroup.nodes[0].connect(
          layer.nodes[0],
        )[0];
        layer.connections.out.push(firstConnection, secondConnection);
        layer.connections.in.push(reverseConnection);

        // Act
        layer.disconnect(targetGroup, true);

        // Assert
        expect({
          incomingCount: layer.connections.in.length,
          outgoingCount: layer.connections.out.length,
        }).toEqual({ incomingCount: 0, outgoingCount: 0 });
      });
    });
  });

  describe('clear()', () => {
    describe('when clearing a dense layer', () => {
      it('clears each node exactly once', () => {
        // Arrange
        const layer = Layer.dense(2);
        const clearSpies = layer.nodes.map((node) =>
          jest.spyOn(node, 'clear').mockImplementation(() => undefined),
        );

        // Act
        layer.clear();
        const everyNodeWasCleared = clearSpies.every(
          (clearSpy) => clearSpy.mock.calls.length === 1,
        );
        clearSpies.forEach((clearSpy) => clearSpy.mockRestore());

        // Assert
        expect(everyNodeWasCleared).toBe(true);
      });
    });
  });

  describe('set()', () => {
    describe('when setting squash and type on a dense layer', () => {
      it('updates every node and the layer intent together', () => {
        // Arrange
        const layer = Layer.dense(2);

        // Act
        layer.set({ squash: methods.Activation.relu, type: 'output' });

        // Assert
        expect({
          intent: layer.intent,
          nodeSquashes: layer.nodes.map((node) => node.squash),
          nodeTypes: layer.nodes.map((node) => node.type),
        }).toEqual({
          intent: 'output',
          nodeSquashes: [methods.Activation.relu, methods.Activation.relu],
          nodeTypes: ['output', 'output'],
        });
      });
    });

    describe('when setting bias on a memory layer', () => {
      it('delegates the value into each grouped memory block', () => {
        // Arrange
        const layer = Layer.memory(1, 2);

        // Act
        layer.set({ bias: 0.75 });

        // Assert
        expect(
          (layer.nodes as unknown as Group[]).map(
            (memoryBlock) => memoryBlock.nodes[0].bias,
          ),
        ).toEqual([0.75, 0.75]);
      });
    });

    describe('when the node list contains an unsupported entry', () => {
      it('ignores the unsupported entry and still updates valid nodes', () => {
        // Arrange
        const layer = Layer.dense(1);
        (layer.nodes as unknown as Array<Node | { sentinel: string }>).push({
          sentinel: 'unsupported',
        });

        // Act
        layer.set({ bias: 0.5 });

        // Assert
        expect({
          bias: layer.nodes[0].bias,
          nodeCount: layer.nodes.length,
        }).toEqual({
          bias: 0.5,
          nodeCount: 2,
        });
      });
    });
  });

  describe('static recurrent factories', () => {
    describe('when creating an LSTM layer', () => {
      it('stamps recurrent metadata and exposes one output unit per requested size', () => {
        // Arrange
        const unitCount = 2;

        // Act
        const layer = Layer.lstm(unitCount);

        // Assert
        expect({
          intent: layer.intent,
          metadata: layer.metadata,
          nodeCount: layer.nodes.length,
          outputCount: layer.output?.nodes.length,
        }).toEqual({
          intent: 'recurrent',
          metadata: { family: 'lstm', units: unitCount },
          nodeCount: 10,
          outputCount: 2,
        });
      });
    });

    describe('when creating a GRU layer', () => {
      it('stamps recurrent metadata and exposes one output unit per requested size', () => {
        // Arrange
        const unitCount = 2;

        // Act
        const layer = Layer.gru(unitCount);

        // Assert
        expect({
          intent: layer.intent,
          metadata: layer.metadata,
          nodeCount: layer.nodes.length,
          outputCount: layer.output?.nodes.length,
        }).toEqual({
          intent: 'recurrent',
          metadata: { family: 'gru', units: unitCount },
          nodeCount: 12,
          outputCount: 2,
        });
      });
    });
  });

  describe('static normalization factories', () => {
    describe('when creating a batch normalization layer', () => {
      it('marks the layer as batch-normalized', () => {
        // Act
        const layer = Layer.batchNorm(2) as BatchNormLayer;

        // Assert
        expect({
          batchNorm: layer.batchNorm,
          nodeCount: layer.nodes.length,
        }).toEqual({
          batchNorm: true,
          nodeCount: 2,
        });
      });
    });

    describe('when creating a layer normalization layer', () => {
      it('marks the layer as layer-normalized', () => {
        // Act
        const layer = Layer.layerNorm(2) as LayerNormLayer;

        // Assert
        expect({
          layerNorm: layer.layerNorm,
          nodeCount: layer.nodes.length,
        }).toEqual({
          layerNorm: true,
          nodeCount: 2,
        });
      });
    });
  });

  describe('static experimental factories', () => {
    describe('when creating a conv1d layer with default stride and padding', () => {
      it('applies the documented default convolution shape values', () => {
        // Act
        const layer = Layer.conv1d(2, 3) as Conv1dLayer;

        // Assert
        expect(layer.conv1d).toEqual({ kernelSize: 3, padding: 0, stride: 1 });
      });
    });

    describe('when creating a conv1d layer with explicit shape parameters', () => {
      it('stores the convolution metadata and returns the bounded slice activation', () => {
        // Arrange
        const layer = Layer.conv1d(2, 3, 2, 1) as Conv1dLayer;

        // Act
        const outputs = layer.activate([10, 20, 30]);

        // Assert
        expect({ conv1d: layer.conv1d, outputs }).toEqual({
          conv1d: { kernelSize: 3, padding: 1, stride: 2 },
          outputs: [10, 20],
        });
      });
    });

    describe('when creating an attention layer with the default head count', () => {
      it('uses one attention head in the stored metadata', () => {
        // Act
        const layer = Layer.attention(3) as AttentionLayer;

        // Assert
        expect(layer.attention).toEqual({ heads: 1 });
      });
    });

    describe('when creating an attention layer with explicit head count', () => {
      it('stores the head count and fills outputs with the input average', () => {
        // Arrange
        const layer = Layer.attention(3, 2) as AttentionLayer;

        // Act
        const outputs = layer.activate([1, 2, 3, 4]);

        // Assert
        expect({ attention: layer.attention, outputs }).toEqual({
          attention: { heads: 2 },
          outputs: [2.5, 2.5, 2.5],
        });
      });
    });
  });
});
