import Network from '../network';
import Connection from '../../connection';
import {
  mapActivationToOnnx,
  inferLayerOrdering,
} from './network.onnx.layer-analysis.utils';
import type { ActivationFunction } from './network.onnx.utils.types';
import { NetworkOnnxLayerOrderingUnresolvableError } from './network.onnx.errors';

describe('network onnx layer-analysis utils chapter', () => {
  describe('mapActivationToOnnx()', () => {
    describe('given a null squash reference', () => {
      it('returns Identity without emitting a warning', () => {
        // Arrange — null squash triggers the !context.squash early-return guard
        const nullSquash = null as unknown as ActivationFunction;

        // Act
        const result = mapActivationToOnnx(nullSquash);

        // Assert
        expect(result).toBe('Identity');
      });
    });
  });

  describe('inferLayerOrdering()', () => {
    describe('given a network whose hidden nodes have only cyclic incoming connections', () => {
      it('throws NetworkOnnxLayerOrderingUnresolvableError', () => {
        // Arrange — build a 1-2-1 MLP then rewire hidden nodes to form a cycle
        const network = Network.createMLP(1, [2], 1);
        const hiddenNodeA = network.nodes[1];
        const hiddenNodeB = network.nodes[2];

        // Replace each hidden node's incoming connections with a cross-reference
        // so neither can be resolved from the input layer.
        const internalA = hiddenNodeA as unknown as {
          connections: { in: Connection[] };
        };
        const internalB = hiddenNodeB as unknown as {
          connections: { in: Connection[] };
        };
        internalA.connections.in = [new Connection(hiddenNodeB, hiddenNodeA, 1)];
        internalB.connections.in = [new Connection(hiddenNodeA, hiddenNodeB, 1)];

        // Act + Assert
        expect(() => inferLayerOrdering(network)).toThrow(
          NetworkOnnxLayerOrderingUnresolvableError,
        );
      });
    });
  });
});
