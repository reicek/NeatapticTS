import Network from '../network';
import Connection from '../../connection';
import * as methods from '../../../methods/methods';
import {
  mapActivationToOnnx,
  inferLayerOrdering,
  resolveOnnxActivationNodeConfig,
} from './network.onnx.layer-analysis.utils';
import type { ActivationFunction } from './network.onnx.utils.types';
import { NetworkOnnxLayerOrderingUnresolvableError } from './network.onnx.errors';

function suppressConsoleWarn<T>(callback: () => T): T {
  const originalWarn = console.warn;
  console.warn = jest.fn();

  try {
    return callback();
  } finally {
    console.warn = originalWarn;
  }
}

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
    describe('given the runtime softplus activation', () => {
      it('maps to the ONNX Softplus operator', () => {
        // Act
        const result = mapActivationToOnnx(methods.Activation.softplus);

        // Assert
        expect(result).toBe('Softplus');
      });
    });

    describe('given the runtime softsign activation', () => {
      it('maps to the ONNX Softsign operator', () => {
        // Act
        const result = mapActivationToOnnx(methods.Activation.softsign);

        // Assert
        expect(result).toBe('Softsign');
      });
    });

    describe('given the runtime mish activation', () => {
      it('maps to the ONNX Mish operator at the default opset', () => {
        // Act
        const result = mapActivationToOnnx(methods.Activation.mish);

        // Assert
        expect(result).toBe('Mish');
      });
    });

    describe('given the runtime gelu activation at opset 20', () => {
      it('maps to the ONNX Gelu operator', () => {
        // Act
        const result = mapActivationToOnnx(methods.Activation.gelu, 20);

        // Assert
        expect(result).toBe('Gelu');
      });
    });

    describe('given the runtime selu activation', () => {
      it('returns the ONNX Selu payload with explicit alpha and gamma attributes', () => {
        // Act
        const result = resolveOnnxActivationNodeConfig(methods.Activation.selu);

        // Assert
        expect(result).toEqual({
          operation: 'Selu',
          attributes: [
            { name: 'alpha', type: 'FLOAT', f: 1.6732632423543772 },
            { name: 'gamma', type: 'FLOAT', f: 1.0507009873554805 },
          ],
        });
      });
    });

    describe('given the runtime mish activation below opset 18', () => {
      it('falls back to the ONNX Identity operator', () => {
        // Act
        const result = suppressConsoleWarn(() =>
          mapActivationToOnnx(methods.Activation.mish, 17),
        );

        // Assert
        expect(result).toBe('Identity');
      });
    });

    describe('given the runtime gelu activation below opset 20', () => {
      it('falls back to the ONNX Identity operator', () => {
        // Act
        const result = suppressConsoleWarn(() =>
          mapActivationToOnnx(methods.Activation.gelu, 18),
        );

        // Assert
        expect(result).toBe('Identity');
      });
    });

    describe('given an activation-like object without a name', () => {
      it('falls back to the ONNX Identity operator', () => {
        // Arrange
        const namelessActivation = {} as ActivationFunction;

        // Act
        const result = suppressConsoleWarn(() =>
          mapActivationToOnnx(namelessActivation),
        );

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
        internalA.connections.in = [
          new Connection(hiddenNodeB, hiddenNodeA, 1),
        ];
        internalB.connections.in = [
          new Connection(hiddenNodeA, hiddenNodeB, 1),
        ];

        // Act + Assert
        expect(() => inferLayerOrdering(network)).toThrow(
          NetworkOnnxLayerOrderingUnresolvableError,
        );
      });
    });
  });
});
