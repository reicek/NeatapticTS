import Network from '../../network';
import * as methods from '../../../../methods/methods';
import { exportToONNX, importFromONNX } from '../network.onnx';
import type { OnnxModel } from '../network.onnx';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network onnx import chapter', () => {
  describe('importFromONNX()', () => {
    describe('given a 1-1 input-output network was exported', () => {
      let sourceNetwork: Network;
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        sourceNetwork = new Network(1, 1);
        sourceNetwork.nodes[1].squash = methods.Activation.tanh;
        const onnxModel = exportToONNX(sourceNetwork);

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when reading the rebuilt node collection', () => {
        it('recreates the original node count', () => {
          // Assert
          expect(importedNetwork.nodes.length).toBe(2);
        });
      });

      describe('when reading the rebuilt output activation', () => {
        it('preserves the output squash function', () => {
          // Assert
          expect(importedNetwork.nodes[1].squash).toBe(methods.Activation.tanh);
        });
      });

      describe('when reading the rebuilt output bias', () => {
        it('preserves the output bias value', () => {
          // Assert
          expect(importedNetwork.nodes[1].bias).toBeCloseTo(
            sourceNetwork.nodes[1].bias,
            12,
          );
        });
      });
    });

    describe('given a 2-2-1 network was exported', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        sourceNetwork.nodes[2].bias = 0.5;
        sourceNetwork.nodes[3].bias = -0.5;
        sourceNetwork.nodes[4].bias = 1.0;
        sourceNetwork.nodes[2].connections.in[0].weight = 0.1;
        sourceNetwork.nodes[2].connections.in[1].weight = 0.2;
        sourceNetwork.nodes[3].connections.in[0].weight = 0.3;
        sourceNetwork.nodes[3].connections.in[1].weight = 0.4;
        sourceNetwork.nodes[4].connections.in[0].weight = 0.5;
        sourceNetwork.nodes[4].connections.in[1].weight = 0.6;
        sourceNetwork.nodes[2].squash = methods.Activation.relu;
        sourceNetwork.nodes[3].squash = methods.Activation.relu;
        sourceNetwork.nodes[4].squash = methods.Activation.sigmoid;
        const onnxModel = exportToONNX(sourceNetwork);

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when reading the rebuilt graph size', () => {
        it('recreates all input, hidden, and output nodes', () => {
          // Assert
          expect(importedNetwork.nodes.length).toBe(5);
        });
      });

      describe('when reading hidden-node biases', () => {
        it('preserves the first hidden-node bias', () => {
          // Assert
          expect(importedNetwork.nodes[2].bias).toBeCloseTo(0.5, 12);
        });

        it('preserves the second hidden-node bias', () => {
          // Assert
          expect(importedNetwork.nodes[3].bias).toBeCloseTo(-0.5, 12);
        });
      });

      describe('when reading the rebuilt output bias', () => {
        it('preserves the output-node bias', () => {
          // Assert
          expect(importedNetwork.nodes[4].bias).toBeCloseTo(1.0, 12);
        });
      });

      describe('when reading imported connection weights', () => {
        it('preserves the first hidden node input-0 weight', () => {
          // Assert
          expect(importedNetwork.nodes[2].connections.in[0].weight).toBeCloseTo(
            0.1,
            12,
          );
        });

        it('preserves the first hidden node input-1 weight', () => {
          // Assert
          expect(importedNetwork.nodes[2].connections.in[1].weight).toBeCloseTo(
            0.2,
            12,
          );
        });

        it('preserves the second hidden node input-0 weight', () => {
          // Assert
          expect(importedNetwork.nodes[3].connections.in[0].weight).toBeCloseTo(
            0.3,
            12,
          );
        });

        it('preserves the second hidden node input-1 weight', () => {
          // Assert
          expect(importedNetwork.nodes[3].connections.in[1].weight).toBeCloseTo(
            0.4,
            12,
          );
        });

        it('preserves the output node hidden-1 weight', () => {
          // Assert
          expect(importedNetwork.nodes[4].connections.in[0].weight).toBeCloseTo(
            0.5,
            12,
          );
        });

        it('preserves the output node hidden-2 weight', () => {
          // Assert
          expect(importedNetwork.nodes[4].connections.in[1].weight).toBeCloseTo(
            0.6,
            12,
          );
        });
      });

      describe('when reading imported activation functions', () => {
        it('preserves the first hidden-node activation', () => {
          // Assert
          expect(importedNetwork.nodes[2].squash).toBe(methods.Activation.relu);
        });

        it('preserves the second hidden-node activation', () => {
          // Assert
          expect(importedNetwork.nodes[3].squash).toBe(methods.Activation.relu);
        });

        it('preserves the output-node activation', () => {
          // Assert
          expect(importedNetwork.nodes[4].squash).toBe(
            methods.Activation.sigmoid,
          );
        });
      });
    });

    describe('given the ONNX payload is null', () => {
      describe('when importFromONNX() is called', () => {
        it('throws', () => {
          // Arrange
          const importCallback = () =>
            importFromONNX(null as unknown as OnnxModel);

          // Assert
          expect(importCallback).toThrow();
        });
      });
    });

    describe('given the ONNX payload is undefined', () => {
      describe('when importFromONNX() is called', () => {
        it('throws', () => {
          // Arrange
          const importCallback = () =>
            importFromONNX(undefined as unknown as OnnxModel);

          // Assert
          expect(importCallback).toThrow();
        });
      });
    });
  });
});
