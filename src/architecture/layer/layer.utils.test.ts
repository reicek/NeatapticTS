import Connection from '../connection/connection';
import Layer from './layer';
import Node from '../node/node';
import {
  activateLayer,
  createAttentionLayer,
  createConv1dLayer,
  createDenseLayer,
  disconnectLayer,
  propagateLayer,
} from './layer.utils';
import type {
  LayerConnectionContext,
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerLike,
  LayerPropagationContext,
} from './layer.utils.types';

type TestLayer = LayerFactoryLayer & {
  attention?: { heads: number };
  conv1d?: { kernelSize: number; padding: number; stride: number };
};

describe('layer utils', () => {
  describe('activateLayer()', () => {
    describe('given dropout is disabled', () => {
      describe('when training is omitted', () => {
        it('uses the default non-training activation path', () => {
          // Arrange
          const layer = Layer.dense(2);

          // Act
          const outputValues = activateLayer(
            { nodes: layer.nodes, dropout: 0 },
            [0.25, 0.75],
          );

          // Assert
          expect(outputValues.length).toBe(2);
        });
      });
    });
  });

  describe('disconnectLayer()', () => {
    describe('given a layer has one forward and one reverse connection', () => {
      describe('when two-sided cleanup is omitted', () => {
        it('removes only the outgoing bookkeeping entry', () => {
          // Arrange
          const layer = Layer.dense(1);
          const targetNode = new Node('hidden');
          const forwardConnection = layer.nodes[0].connect(targetNode)[0];
          const reverseConnection = targetNode.connect(layer.nodes[0])[0];
          layer.connections.out.push(forwardConnection);
          layer.connections.in.push(reverseConnection);

          // Act
          disconnectLayer(
            layer as unknown as LayerConnectionContext,
            targetNode,
          );

          // Assert
          expect({
            incomingCount: layer.connections.in.length,
            outgoingCount: layer.connections.out.length,
          }).toEqual({ incomingCount: 1, outgoingCount: 0 });
        });
      });
    });
  });

  describe('propagateLayer()', () => {
    describe('given a propagation context with explicit targets', () => {
      describe('when the wrapper delegates to reverse propagation', () => {
        it('propagates each node in reverse order with aligned targets', () => {
          // Arrange
          const callLog: Array<{
            args: Array<boolean | number>;
            label: string;
          }> = [];
          const context = {
            nodes: [
              createPropagationNode('first', callLog),
              createPropagationNode('second', callLog),
              createPropagationNode('third', callLog),
            ],
          } as unknown as LayerPropagationContext;

          // Act
          propagateLayer(context, 0.3, 0.1, [0.2, 0.4, 0.6]);

          // Assert
          expect(callLog).toStrictEqual([
            { args: [0.3, 0.1, true, 0, 0.6], label: 'third' },
            { args: [0.3, 0.1, true, 0, 0.4], label: 'second' },
            { args: [0.3, 0.1, true, 0, 0.2], label: 'first' },
          ]);
        });
      });
    });
  });

  describe('createDenseLayer()', () => {
    describe('given the node role is omitted', () => {
      describe('when the dense layer is created', () => {
        it('defaults the block to hidden nodes', () => {
          // Arrange
          const createdLayer = createDenseLayer(createFactoryContext(), 2);

          // Assert
          expect(createdLayer.output?.nodes.map((node) => node.type)).toEqual([
            'hidden',
            'hidden',
          ]);
        });
      });
    });
  });

  describe('createConv1dLayer()', () => {
    describe('given the stride and padding are omitted', () => {
      describe('when the convolution layer is created', () => {
        it('uses the documented default convolution shape values', () => {
          // Arrange
          const createdLayer = createConv1dLayer(createFactoryContext(), 2, 3);

          // Assert
          expect((createdLayer as TestLayer).conv1d).toEqual({
            kernelSize: 3,
            padding: 0,
            stride: 1,
          });
        });
      });
    });
  });

  describe('createAttentionLayer()', () => {
    describe('given the head count is omitted', () => {
      describe('when the attention layer is created', () => {
        it('uses one attention head by default', () => {
          // Arrange
          const createdLayer = createAttentionLayer(createFactoryContext(), 3);

          // Assert
          expect((createdLayer as TestLayer).attention).toEqual({ heads: 1 });
        });
      });
    });
  });
});

function createFactoryContext(): LayerFactoryContext<TestLayer> {
  return {
    createLayer: () => ({
      activate: () => [],
      input: () => [] as Connection[],
      nodes: [],
      output: null,
    }),
    isLayer: (value: unknown): value is LayerLike =>
      typeof value === 'object' &&
      value !== null &&
      'input' in value &&
      typeof (value as LayerLike).input === 'function',
  };
}

function createPropagationNode(
  label: string,
  callLog: Array<{ args: Array<boolean | number>; label: string }>,
): Node {
  return {
    propagate: (...args: Array<boolean | number>) => {
      callLog.push({ args, label });
    },
  } as unknown as Node;
}
