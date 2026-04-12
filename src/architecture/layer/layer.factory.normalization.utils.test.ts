import type Connection from '../connection/connection';
import {
  buildBatchNormLayer,
  buildLayerNormLayer,
} from './layer.factory.normalization.utils';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerLike,
} from './layer.utils.types';

type TestLayer = LayerFactoryLayer & {
  batchNorm?: boolean;
  layerNorm?: boolean;
};

describe('normalization layer factory chapter', () => {
  describe('buildBatchNormLayer', () => {
    describe('given the base activator returns a centered spread of values', () => {
      it('marks the layer and normalizes the resulting activations', () => {
        // Arrange
        const baseActivate = jest.fn(
          (_values?: number[], _training: boolean = false) => [2, 4],
        );
        const layer = buildBatchNormLayer(
          createFactoryContext(baseActivate),
          2,
        );

        // Act
        const outputs = layer.activate([10, 20], true).map((value) =>
          +value.toFixed(6),
        );

        // Assert
        expect({
          batchNorm: layer.batchNorm,
          baseCall: baseActivate.mock.calls[0],
          outputs,
        }).toEqual({
          batchNorm: true,
          baseCall: [[10, 20], true],
          outputs: [-0.999995, 0.999995],
        });
      });
    });
  });

  describe('buildLayerNormLayer', () => {
    describe('given the base activator returns a zero-variance vector', () => {
      it('marks the layer and normalizes the output to zeros', () => {
        // Arrange
        const baseActivate = jest.fn(() => [3, 3]);
        const layer = buildLayerNormLayer(createFactoryContext(baseActivate), 2);

        // Act
        const outputs = layer.activate();

        // Assert
        expect({ layerNorm: layer.layerNorm, outputs }).toEqual({
          layerNorm: true,
          outputs: [0, 0],
        });
      });
    });
  });
});

function createFactoryContext(
  baseActivate: (values?: number[], training?: boolean) => number[],
): LayerFactoryContext<TestLayer> {
  return {
    createLayer: () => ({
      activate: baseActivate,
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