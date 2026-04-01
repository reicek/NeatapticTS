import Network from '../../network';
import Node from '../../../node';
import * as methods from '../../../../methods/methods';
import { exportToONNX } from '../network.onnx';
import type { OnnxModel } from '../network.onnx';
import {
  NetworkOnnxMixedActivationsUnsupportedError,
  NetworkOnnxPartialConnectivityUnsupportedError,
} from '../network.onnx.errors';

type OnnxValueDim = { dim_value?: number; dim_param?: string };
type OnnxValueInfo = {
  name: string;
  type: {
    tensor_type: {
      shape: {
        dim: OnnxValueDim[];
      };
    };
  };
};
type OnnxAttributeView = { name: string; f?: number; i?: number };
type OnnxNodeView = {
  op_type: string;
  input: string[];
  output: string[];
  attributes?: OnnxAttributeView[];
};
type OnnxInitializerView = { name: string };

const activationOperations = new Set(['Tanh', 'Sigmoid', 'Relu', 'Identity']);

function toInputs(model: OnnxModel): OnnxValueInfo[] {
  return model.graph.inputs as OnnxValueInfo[];
}

function toOutputs(model: OnnxModel): OnnxValueInfo[] {
  return model.graph.outputs as OnnxValueInfo[];
}

function toInitializers(model: OnnxModel): OnnxInitializerView[] {
  return model.graph.initializer as OnnxInitializerView[];
}

function toNodes(model: OnnxModel): OnnxNodeView[] {
  return model.graph.node as OnnxNodeView[];
}

function getHiddenNodes(network: Network): Node[] {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden');
}

function suppressConsoleWarn(callback: () => void): void {
  const originalWarn = console.warn;
  console.warn = jest.fn();

  try {
    callback();
  } finally {
    console.warn = originalWarn;
  }
}

function createDisconnectedExportNetwork(): Network {
  const network = new Network(1, 1, {});
  network.connections = [];
  network.nodes.forEach((nodeEntry: Node) => {
    nodeEntry.connections.out = [];
    nodeEntry.connections.in = [];
  });
  Network.rebuildConnections(network);

  return network;
}

function createPartiallyDisconnectedNetwork(): Network {
  const network = new Network(2, 1, {});
  network.connections = [];

  const firstHiddenNode = new Node('hidden');
  const secondHiddenNode = new Node('hidden');
  network.nodes.splice(2, 0, firstHiddenNode, secondHiddenNode);

  const firstInputNode = network.nodes[0];
  const secondInputNode = network.nodes[1];
  const outputNode = network.nodes[4];

  firstInputNode.connect(firstHiddenNode, 0.5);
  firstInputNode.connect(secondHiddenNode, 0.5);
  secondInputNode.connect(firstHiddenNode, 0.5);
  secondInputNode.connect(secondHiddenNode, 0.5);
  firstHiddenNode.connect(outputNode, 1.0);
  secondHiddenNode.connect(outputNode, 1.0);

  firstHiddenNode.connections.in = firstHiddenNode.connections.in.filter(
    (connectionEntry) => connectionEntry.from !== firstInputNode,
  );
  firstInputNode.connections.out = firstInputNode.connections.out.filter(
    (connectionEntry) => connectionEntry.to !== firstHiddenNode,
  );
  Network.rebuildConnections(network);

  return network;
}

function getFirstActivationNode(model: OnnxModel): OnnxNodeView | undefined {
  return toNodes(model).find((nodeEntry) =>
    activationOperations.has(nodeEntry.op_type),
  );
}

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network onnx export chapter', () => {
  describe('exportToONNX()', () => {
    describe('given a minimal 1-1 network', () => {
      describe('when exportToONNX() is called', () => {
        it('uses the input width as the first input dimension', () => {
          // Arrange
          const network = new Network(1, 1, {});

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(
            toInputs(onnxModel)[0].type.tensor_type.shape.dim[0]?.dim_value,
          ).toBe(1);
        });
      });
    });

    describe('given feed-forward networks with hidden layers', () => {
      describe('when exporting a 2-2-1 network', () => {
        it('emits at least one initializer tensor', () => {
          // Arrange
          const network = Network.createMLP(2, [2], 1);

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(toInitializers(onnxModel).length > 0).toBe(true);
        });
      });

      describe('when exporting a 3-3-2-1 network', () => {
        it('emits weight and bias initializers', () => {
          // Arrange
          const network = Network.createMLP(3, [3, 2], 1);

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(toInitializers(onnxModel).length > 0).toBe(true);
        });
      });

      describe('when exporting a 2-2-2 network', () => {
        let onnxModel: OnnxModel;

        beforeEach(() => {
          // Arrange
          const network = Network.createMLP(2, [2], 2);

          // Act
          onnxModel = exportToONNX(network);
        });

        it('keeps a single output tensor entry', () => {
          // Assert
          expect(toOutputs(onnxModel).length).toBe(1);
        });

        it('stores the output feature width in the output tensor shape', () => {
          // Assert
          expect(
            toOutputs(onnxModel)[0].type.tensor_type.shape.dim[0]?.dim_value,
          ).toBe(2);
        });
      });
    });

    describe('given an output activation', () => {
      describe('when the activation is tanh', () => {
        it('maps the layer activation to ONNX Tanh', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = methods.Activation.tanh;

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(getFirstActivationNode(onnxModel)?.op_type).toBe('Tanh');
        });
      });

      describe('when the activation is logistic', () => {
        it('maps the layer activation to ONNX Sigmoid', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = methods.Activation.logistic;

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(getFirstActivationNode(onnxModel)?.op_type).toBe('Sigmoid');
        });
      });

      describe('when the activation is relu', () => {
        it('maps the layer activation to ONNX Relu', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = methods.Activation.relu;

          // Act
          const onnxModel = exportToONNX(network);

          // Assert
          expect(getFirstActivationNode(onnxModel)?.op_type).toBe('Relu');
        });
      });

      describe('when the activation is not recognized', () => {
        it('falls back to the ONNX Identity operator', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = (value: number) => value;

          // Act / Assert
          suppressConsoleWarn(() => {
            const onnxModel = exportToONNX(network);
            expect(getFirstActivationNode(onnxModel)?.op_type).toBe('Identity');
          });
        });

        it('emits a warning about the unsupported activation', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = (value: number) => value;
          const originalWarn = console.warn;
          const warnSpy = jest.fn();
          console.warn = warnSpy;

          try {
            // Act
            exportToONNX(network);

            // Assert
            expect(warnSpy).toHaveBeenCalledTimes(1);
          } finally {
            console.warn = originalWarn;
          }
        });
      });
    });

    describe('given unsupported dense export structures', () => {
      describe('when the network has no connections', () => {
        it('throws the simple-MLP export message', () => {
          // Arrange
          const network = createDisconnectedExportNetwork();

          // Act
          const exportCallback = () => exportToONNX(network);

          // Assert
          expect(exportCallback).toThrow(
            'ONNX export currently only supports simple MLPs',
          );
        });
      });

      describe('when a dense layer is partially disconnected', () => {
        it('throws the partial-connectivity export error type', () => {
          // Arrange
          const network = createPartiallyDisconnectedNetwork();

          // Act
          const exportCallback = () => exportToONNX(network);

          // Assert
          expect(exportCallback).toThrow(
            NetworkOnnxPartialConnectivityUnsupportedError,
          );
        });
      });
    });

    describe('given node indices are missing', () => {
      describe('when exportToONNX() is called', () => {
        it('assigns numeric indices to every node', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes.forEach((nodeEntry: Node) => {
            nodeEntry.index = undefined;
          });

          // Act
          exportToONNX(network);

          // Assert
          expect(
            network.nodes.every(
              (nodeEntry: Node) => typeof nodeEntry.index === 'number',
            ),
          ).toBe(true);
        });
      });
    });

    describe('given the warning-suppression helper', () => {
      describe('when console.warn is called inside the wrapper', () => {
        it('still executes the wrapped callback', () => {
          // Arrange
          let callbackRan = false;

          // Act
          suppressConsoleWarn(() => {
            console.warn('test');
            callbackRan = true;
          });

          // Assert
          expect(callbackRan).toBe(true);
        });
      });
    });

    describe('given the default dense ordering', () => {
      let emittedNodes: OnnxNodeView[];
      let gemmNodes: OnnxNodeView[];

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(3, [3, 2], 1);

        // Act
        const onnxModel = exportToONNX(network);
        emittedNodes = toNodes(onnxModel);
        gemmNodes = emittedNodes.filter(
          (nodeEntry) => nodeEntry.op_type === 'Gemm',
        );
      });

      describe('when reading the Gemm nodes', () => {
        it('emits at least one Gemm node', () => {
          // Assert
          expect(gemmNodes.length > 0).toBe(true);
        });

        it('sets alpha to 1 on every Gemm node', () => {
          // Assert
          expect(
            gemmNodes.every(
              (nodeEntry) =>
                nodeEntry.attributes?.find(
                  (attributeEntry) => attributeEntry.name === 'alpha',
                )?.f === 1,
            ),
          ).toBe(true);
        });

        it('sets beta to 1 on every Gemm node', () => {
          // Assert
          expect(
            gemmNodes.every(
              (nodeEntry) =>
                nodeEntry.attributes?.find(
                  (attributeEntry) => attributeEntry.name === 'beta',
                )?.f === 1,
            ),
          ).toBe(true);
        });

        it('sets transB to 1 on every Gemm node', () => {
          // Assert
          expect(
            gemmNodes.every(
              (nodeEntry) =>
                nodeEntry.attributes?.find(
                  (attributeEntry) => attributeEntry.name === 'transB',
                )?.i === 1,
            ),
          ).toBe(true);
        });
      });

      describe('when checking activation placement', () => {
        it('places the activation node after each Gemm node', () => {
          // Assert
          expect(
            gemmNodes.every((gemmNode) => {
              const gemmIndex = emittedNodes.indexOf(gemmNode);
              const activationNode = emittedNodes.find(
                (candidateNode) =>
                  candidateNode !== gemmNode &&
                  candidateNode.input[0] === gemmNode.output[0] &&
                  candidateNode.op_type !== 'Gemm',
              );

              return (
                activationNode != null &&
                emittedNodes.indexOf(activationNode) > gemmIndex
              );
            }),
          ).toBe(true);
        });
      });
    });

    describe('given legacy node ordering is enabled', () => {
      describe('when exportToONNX() is called', () => {
        it('places activation nodes before their Gemm nodes', () => {
          // Arrange
          const network = Network.createMLP(2, [2], 1);

          // Act
          const emittedNodes = toNodes(
            exportToONNX(network, { legacyNodeOrdering: true }),
          );
          const gemmNodes = emittedNodes.filter(
            (nodeEntry) => nodeEntry.op_type === 'Gemm',
          );

          // Assert
          expect(
            gemmNodes.every((gemmNode) => {
              const gemmIndex = emittedNodes.indexOf(gemmNode);
              const activationNode = emittedNodes.find(
                (candidateNode) =>
                  candidateNode.input[0] === gemmNode.output[0] &&
                  candidateNode.op_type !== 'Gemm',
              );

              return (
                activationNode != null &&
                emittedNodes.indexOf(activationNode) < gemmIndex
              );
            }),
          ).toBe(true);
        });
      });
    });

    describe('given metadata inclusion is enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);

        // Act
        onnxModel = exportToONNX(network, { includeMetadata: true, opset: 18 });
      });

      describe('when reading the top-level metadata', () => {
        it('includes an ir_version field', () => {
          // Assert
          expect(typeof onnxModel.ir_version).not.toBe('undefined');
        });

        it('includes an opset_import array', () => {
          // Assert
          expect(Array.isArray(onnxModel.opset_import)).toBe(true);
        });

        it('includes a producer_name string', () => {
          // Assert
          expect(typeof onnxModel.producer_name).toBe('string');
        });
      });
    });

    describe('given batchDimension is enabled', () => {
      let inputDimensions: OnnxValueDim[];
      let outputDimensions: OnnxValueDim[];

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(4, [3], 2);

        // Act
        const onnxModel = exportToONNX(network, { batchDimension: true });
        inputDimensions = toInputs(onnxModel)[0].type.tensor_type.shape.dim;
        outputDimensions = toOutputs(onnxModel)[0].type.tensor_type.shape.dim;
      });

      describe('when reading tensor shapes', () => {
        it('adds batch and feature dimensions to the input tensor', () => {
          // Assert
          expect(inputDimensions.length).toBe(2);
        });

        it('adds batch and feature dimensions to the output tensor', () => {
          // Assert
          expect(outputDimensions.length).toBe(2);
        });

        it('uses the symbolic batch dimension N on the input tensor', () => {
          // Assert
          expect(inputDimensions[0].dim_param).toBe('N');
        });

        it('uses the symbolic batch dimension N on the output tensor', () => {
          // Assert
          expect(outputDimensions[0].dim_param).toBe('N');
        });

        it('keeps the input feature width in the second input dimension', () => {
          // Assert
          expect(inputDimensions[1].dim_value).toBe(4);
        });

        it('keeps the output feature width in the second output dimension', () => {
          // Assert
          expect(outputDimensions[1].dim_value).toBe(2);
        });
      });
    });

    describe('given relaxed export validation options', () => {
      describe('when allowPartialConnectivity is disabled', () => {
        it('rejects partially connected dense layers', () => {
          // Arrange
          const network = createPartiallyDisconnectedNetwork();

          // Act
          const exportCallback = () => exportToONNX(network);

          // Assert
          expect(exportCallback).toThrow(
            NetworkOnnxPartialConnectivityUnsupportedError,
          );
        });
      });

      describe('when allowPartialConnectivity is enabled', () => {
        it('exports partially connected dense layers', () => {
          // Arrange
          const network = createPartiallyDisconnectedNetwork();

          // Act
          const exportCallback = () =>
            exportToONNX(network, { allowPartialConnectivity: true });

          // Assert
          expect(exportCallback).not.toThrow();
        });
      });

      describe('when a dense layer mixes activations without relaxed mode', () => {
        it('throws the mixed-activations export error type', () => {
          // Arrange
          const network = Network.createMLP(1, [3], 1);
          network.nodes[2].squash = methods.Activation.relu;
          network.nodes[3].squash = methods.Activation.tanh;
          network.nodes[4].squash = methods.Activation.sigmoid;

          // Act
          const exportCallback = () => exportToONNX(network);

          // Assert
          expect(exportCallback).toThrow(
            NetworkOnnxMixedActivationsUnsupportedError,
          );
        });
      });

      describe('when allowMixedActivations is enabled', () => {
        it('exports a dense layer with mixed activations', () => {
          // Arrange
          const network = Network.createMLP(1, [3], 1);
          network.nodes[2].squash = methods.Activation.relu;
          network.nodes[3].squash = methods.Activation.tanh;
          network.nodes[4].squash = methods.Activation.sigmoid;

          // Act
          const exportCallback = () =>
            exportToONNX(network, { allowMixedActivations: true });

          // Assert
          expect(exportCallback).not.toThrow();
        });
      });
    });

    describe('given recurrent single-step export is enabled', () => {
      describe('when every hidden node has self-recurrence', () => {
        let onnxModel: OnnxModel;

        beforeEach(() => {
          // Arrange
          const network = Network.createMLP(2, [3], 1);
          getHiddenNodes(network).forEach((hiddenNode) => {
            if (hiddenNode.connections.self.length === 0) {
              hiddenNode.connect(hiddenNode, 0.42);
              return;
            }

            hiddenNode.connections.self[0].weight = 0.42;
          });

          // Act
          onnxModel = exportToONNX(network, {
            allowRecurrent: true,
            recurrentSingleStep: true,
          });
        });

        it('adds a previous hidden state input', () => {
          // Assert
          expect(
            toInputs(onnxModel).some(
              (valueInfo) => valueInfo.name === 'hidden_prev',
            ),
          ).toBe(true);
        });

        it('emits the first recurrent weight matrix initializer', () => {
          // Assert
          expect(
            toInitializers(onnxModel).some(
              (initializerView) => initializerView.name === 'R0',
            ),
          ).toBe(true);
        });
      });

      describe('when only the second hidden layer has self-recurrence', () => {
        let onnxModel: OnnxModel;

        beforeEach(() => {
          // Arrange
          const network = Network.createMLP(2, [2, 2], 1);
          const secondHiddenLayerNodes = getHiddenNodes(network).slice(2, 4);

          secondHiddenLayerNodes.forEach((hiddenNode, hiddenNodeIndex) => {
            const recurrentWeight = 0.1 + hiddenNodeIndex;
            if (hiddenNode.connections.self.length === 0) {
              hiddenNode.connect(hiddenNode, recurrentWeight);
              return;
            }

            hiddenNode.connections.self[0].weight = recurrentWeight;
          });

          // Act
          onnxModel = exportToONNX(network, {
            allowRecurrent: true,
            recurrentSingleStep: true,
          });
        });

        it('does not add a previous-state input for the first hidden layer', () => {
          // Assert
          expect(
            toInputs(onnxModel).some(
              (valueInfo) => valueInfo.name === 'hidden_prev',
            ),
          ).toBe(false);
        });

        it('adds a previous-state input for the second hidden layer', () => {
          // Assert
          expect(
            toInputs(onnxModel).some(
              (valueInfo) => valueInfo.name === 'hidden_prev_l2',
            ),
          ).toBe(true);
        });

        it('does not emit the first recurrent weight matrix initializer', () => {
          // Assert
          expect(
            toInitializers(onnxModel).some(
              (initializerView) => initializerView.name === 'R0',
            ),
          ).toBe(false);
        });

        it('emits the second recurrent weight matrix initializer', () => {
          // Assert
          expect(
            toInitializers(onnxModel).some(
              (initializerView) => initializerView.name === 'R1',
            ),
          ).toBe(true);
        });
      });

      describe('when a recurrent layer mixes activations', () => {
        it('throws the recurrent mixed-activations error type', () => {
          // Arrange
          const network = Network.createMLP(1, [2], 1);
          const hiddenNodes = getHiddenNodes(network);
          hiddenNodes.forEach((hiddenNode) => {
            hiddenNode.connect(hiddenNode, 0.2);
          });
          hiddenNodes[0].squash = methods.Activation.relu;
          hiddenNodes[1].squash = methods.Activation.tanh;

          // Act
          const exportCallback = () =>
            exportToONNX(network, {
              allowRecurrent: true,
              recurrentSingleStep: true,
            });

          // Assert
          expect(exportCallback).toThrow(
            NetworkOnnxMixedActivationsUnsupportedError,
          );
        });
      });
    });
  });
});
