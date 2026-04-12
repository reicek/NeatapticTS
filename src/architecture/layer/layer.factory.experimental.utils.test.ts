import type Connection from '../connection/connection';
import {
  buildAttentionLayer,
  buildConv1dLayer,
} from './layer.factory.experimental.utils';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerLike,
} from './layer.utils.types';

type TestLayer = LayerFactoryLayer & {
  attention?: { heads: number };
  conv1d?: { kernelSize: number; stride: number; padding: number };
};

describe('experimental layer factory chapter', () => {
  describe('buildConv1dLayer', () => {
    describe('given explicit convolution metadata and values', () => {
      it('stores the metadata and returns a bounded slice', () => {
        // Arrange
        const layer = buildConv1dLayer(createFactoryContext(), 2, 3, 2, 1);

        // Act
        const outputs = layer.activate([10, 20, 30]);

        // Assert
        expect({ conv1d: layer.conv1d, outputs }).toEqual({
          conv1d: { kernelSize: 3, stride: 2, padding: 1 },
          outputs: [10, 20],
        });
      });
    });

    describe('given no input values are provided', () => {
      it('falls back to activating the stub nodes', () => {
        // Arrange
        const layer = buildConv1dLayer(createFactoryContext(), 2, 3);
        const firstNodeSpy = jest
          .spyOn(layer.nodes[0], 'activate')
          .mockReturnValue(0.5);
        const secondNodeSpy = jest
          .spyOn(layer.nodes[1], 'activate')
          .mockReturnValue(1.5);

        // Act
        const outputs = layer.activate();
        firstNodeSpy.mockRestore();
        secondNodeSpy.mockRestore();

        // Assert
        expect(outputs).toEqual([0.5, 1.5]);
      });
    });
  });

  describe('buildAttentionLayer', () => {
    describe('given explicit attention heads and values', () => {
      it('stores the head count and fills the outputs with the average', () => {
        // Arrange
        const layer = buildAttentionLayer(createFactoryContext(), 3, 2);

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