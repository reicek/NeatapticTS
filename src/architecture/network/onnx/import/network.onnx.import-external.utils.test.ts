import onnxProto from 'onnx-proto';
import Network from '../../network';
import * as methods from '../../../../methods/methods';
import { exportToONNXBinary } from '../network.onnx';
import {
  normalizeExternalBinaryOnnxModel,
  normalizeExternalDenseChain,
} from './network.onnx.import-external.utils';
import { OnnxExternalImportError } from './network.onnx.import-external.types';

function resolveRequiredGraph(
  decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>,
): NonNullable<InstanceType<typeof onnxProto.onnx.ModelProto>['graph']> {
  const decodedGraph = decodedModel.graph;
  if (!decodedGraph) {
    throw new Error('Expected decoded binary model graph.');
  }

  return decodedGraph;
}

function resolveNamedNode(
  decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>,
  nodeName: string,
) {
  const decodedGraph = resolveRequiredGraph(decodedModel);
  const namedNode = decodedGraph.node?.find(
    (graphNode) => graphNode.name === nodeName,
  );
  if (!namedNode) {
    throw new Error(`Expected graph node '${nodeName}'.`);
  }

  return namedNode;
}

function createBaselineBinaryModel(): Uint8Array {
  const sourceNetwork = Network.createMLP(2, [2], 1);
  sourceNetwork.nodes[2].bias = 0.5;
  sourceNetwork.nodes[3].bias = -0.5;
  sourceNetwork.nodes[4].bias = 0.25;
  sourceNetwork.nodes[2].connections.in[0].weight = 0.1;
  sourceNetwork.nodes[2].connections.in[1].weight = 0.2;
  sourceNetwork.nodes[3].connections.in[0].weight = 0.3;
  sourceNetwork.nodes[3].connections.in[1].weight = 0.4;
  sourceNetwork.nodes[4].connections.in[0].weight = 0.5;
  sourceNetwork.nodes[4].connections.in[1].weight = 0.6;
  sourceNetwork.nodes[2].squash = methods.Activation.relu;
  sourceNetwork.nodes[3].squash = methods.Activation.relu;
  sourceNetwork.nodes[4].squash = methods.Activation.sigmoid;
  return exportToONNXBinary(sourceNetwork, { opset: 18 });
}

function mutateBinaryModel(
  binaryModel: Uint8Array,
  mutateModel: (
    decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>,
  ) => void,
): Uint8Array {
  const decodedModel = onnxProto.onnx.ModelProto.decode(binaryModel);
  mutateModel(decodedModel);
  return onnxProto.onnx.ModelProto.encode(decodedModel).finish();
}

function withPatchedPlainDecodedModel<T>(
  binaryModel: Uint8Array,
  mutatePlainModel: (
    plainModel: ReturnType<typeof onnxProto.onnx.ModelProto.toObject>,
  ) => void,
  callback: () => T,
): T {
  const plainModel = onnxProto.onnx.ModelProto.toObject(
    onnxProto.onnx.ModelProto.decode(binaryModel),
  );
  mutatePlainModel(plainModel);

  const toObjectSpy = jest
    .spyOn(onnxProto.onnx.ModelProto, 'toObject')
    .mockReturnValue(plainModel);

  try {
    return callback();
  } finally {
    toObjectSpy.mockRestore();
  }
}

function captureExternalImportFailure(callback: () => unknown): {
  category: string;
  message: string;
} {
  try {
    callback();
    return { category: 'no-error', message: '' };
  } catch (error) {
    if (error instanceof OnnxExternalImportError) {
      return {
        category: error.category,
        message: error.message,
      };
    }

    return {
      category: 'unexpected-error',
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

function encodeFloat32RawData(floatValues: number[]): Uint8Array {
  const rawBuffer = new ArrayBuffer(floatValues.length * 4);
  const rawView = new DataView(rawBuffer);

  floatValues.forEach((floatValue, valueIndex) => {
    rawView.setFloat32(valueIndex * 4, floatValue, true);
  });

  return new Uint8Array(rawBuffer);
}

describe('external ONNX import normalizer', () => {
  describe('normalizeExternalDenseChain()', () => {
    it('accepts float32 raw-data initializers for the first external lane', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          const floatValues = [...(weightTensor.floatData ?? [])];
          weightTensor.floatData = [];
          weightTensor.rawData = encodeFloat32RawData(floatValues);
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers[0]!.weightValues[0]).toBeCloseTo(0.1, 8);
    });

    it('rejects malformed protobuf bytes with an invalid-binary category', () => {
      // Arrange
      const truncatedBinaryModel = createBaselineBinaryModel().subarray(0, 8);

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(truncatedBinaryModel),
      );

      // Assert
      expect(failure.category).toBe('invalid-binary');
    });

    it('rethrows verifier failures as invalid-binary errors', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const verifySpy = jest
        .spyOn(onnxProto.onnx.ModelProto, 'verify')
        .mockReturnValue('forced verification failure');

      try {
        // Act
        const failure = captureExternalImportFailure(() =>
          normalizeExternalDenseChain(baselineBinaryModel),
        );

        // Assert
        expect(failure.category).toBe('invalid-binary');
      } finally {
        verifySpy.mockRestore();
      }
    });

    it('rejects a decoded model that omits the graph payload', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.graph = null as never;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('invalid-model');
    });

    it('rejects models that omit the standard-domain opset import', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport = [];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-domain');
    });

    it('rejects declared opsets below the first supported external floor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 17;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-opset');
    });

    it('rejects unnamed initializer tensors', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.name = '';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('invalid-model');
    });

    it('rejects multiple public inputs', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const modelInput = decodedGraph.input?.[0];
          if (!decodedGraph.input || !modelInput) {
            throw new Error('Expected baseline graph input.');
          }

          decodedGraph.input.push(structuredClone(modelInput));
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects multiple public outputs', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const modelOutput = decodedGraph.output?.[0];
          if (!decodedGraph.output || !modelOutput) {
            throw new Error('Expected baseline graph output.');
          }

          decodedGraph.output.push(structuredClone(modelOutput));
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('accepts supporting value-info tensors that stay rank 2 and float32', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          decodedGraph.valueInfo = [
            {
              name: 'Gemm_1',
              type: {
                tensorType: {
                  elemType: 1,
                  shape: {
                    dim: [{ dimParam: 'N' }, { dimValue: 2 }],
                  },
                },
              },
            },
          ];
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers.length).toBe(2);
    });

    it('rejects duplicate initializers', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!decodedGraph.initializer || !weightTensor) {
            throw new Error('Expected baseline graph initializer.');
          }

          decodedGraph.initializer.push(structuredClone(weightTensor));
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('duplicate-initializer');
    });

    it('rejects non-float32 public input tensors', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const modelInput = decodedGraph.input?.[0]?.type?.tensorType;
          if (!modelInput) {
            throw new Error('Expected baseline input tensor type.');
          }

          modelInput.elemType = 10;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-tensor-type');
    });

    it('rejects non-positive batch dimensions on public tensors', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const batchDimension =
            decodedGraph.input?.[0]?.type?.tensorType?.shape?.dim?.[0];
          if (!batchDimension) {
            throw new Error('Expected baseline batch dimension.');
          }

          batchDimension.dimParam = null;
          batchDimension.dimValue = 0;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('rank-mismatch');
    });

    it('rejects non-positive public feature widths', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const featureDimension =
            decodedGraph.input?.[0]?.type?.tensorType?.shape?.dim?.[1];
          if (!featureDimension) {
            throw new Error('Expected baseline feature dimension.');
          }

          featureDimension.dimValue = 0;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('rank-mismatch');
    });

    it('rejects public tensors without a concrete feature width', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const featureDimension =
            decodedGraph.input?.[0]?.type?.tensorType?.shape?.dim?.[1];
          if (!featureDimension) {
            throw new Error('Expected baseline feature dimension.');
          }

          featureDimension.dimParam = null;
          featureDimension.dimValue = null;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('rank-mismatch');
    });

    it('rejects custom-domain nodes', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const firstNode = decodedGraph.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline graph node.');
          }

          firstNode.domain = 'custom.domain';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-domain');
    });

    it('rejects multiple producers for the same tensor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const firstNodeOutput = decodedGraph.node?.[0]?.output?.[0];
          const secondNode = decodedGraph.node?.[1];
          if (!firstNodeOutput || !secondNode?.output) {
            throw new Error('Expected baseline node outputs.');
          }

          secondNode.output[0] = firstNodeOutput;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects starting tensors with multiple consumers', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const duplicateNode = structuredClone(
            resolveNamedNode(decodedModel, 'gemm_l1'),
          );
          duplicateNode.name = 'gemm_l1_branch';
          duplicateNode.output = ['Branch_1'];
          decodedGraph.node?.push(duplicateNode);
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects nodes that do not emit exactly one output tensor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const firstNode = resolveNamedNode(decodedModel, 'gemm_l1');
          firstNode.output = [];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects graphs that never reach a Gemm layer', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          decodedGraph.node = [];
          decodedGraph.output![0]!.name = decodedGraph.input![0]!.name;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects disconnected nodes outside the walked dense chain', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          decodedGraph.node?.push({
            name: 'orphan_identity',
            opType: 'Identity',
            input: ['orphan_input'],
            output: ['orphan_output'],
          });
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects Gemm cycles that revisit a previously walked layer', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          if (!outputActivationNode.output || !decodedGraph.output?.[0]) {
            throw new Error('Expected baseline output activation output.');
          }

          outputActivationNode.output[0] = 'Activation_1';
          decodedGraph.output[0].name = 'Loop_Output';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects non-Gemm nodes at the start of the dense chain', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const firstNode = resolveNamedNode(decodedModel, 'gemm_l1');
          firstNode.opType = 'Identity';
          firstNode.input = ['input'];
          firstNode.output = ['Gemm_1'];
          firstNode.attribute = [];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-node');
    });

    it('rejects unsupported activation operators', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const outputActivationNode = decodedGraph.node?.find(
            (graphNode) => graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.opType = 'LeakyRelu';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-activation');
    });

    it('rejects non-Gelu activations with unexpected attributes', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.attribute = [{ name: 'alpha', f: 1 }];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects dense chains that end before the declared public output', () => {
      // Arrange
      const singleLayerNetwork = Network.createMLP(2, [], 1);
      singleLayerNetwork.nodes[2].squash = methods.Activation.identity;
      const binaryModel = mutateBinaryModel(
        exportToONNXBinary(singleLayerNetwork, { opset: 18 }),
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          decodedGraph.output![0]!.name = 'Missing_Output';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects branching or merging consumers between dense layers', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstGemmOutput = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null; output?: string[] | null }) =>
              graphNode.name === 'gemm_l1',
          )?.output?.[0];
          if (
            !plainModel.graph?.node ||
            !plainModel.graph.output?.[0] ||
            !firstGemmOutput
          ) {
            throw new Error('Expected baseline plain model.');
          }

          plainModel.graph.node = plainModel.graph.node.filter(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'gemm_l1' || graphNode.name === 'act_l1',
          );
          plainModel.graph.node.push({
            name: 'branch_identity',
            opType: 'Identity',
            input: [firstGemmOutput],
            output: ['Branch_1'],
          });
          plainModel.graph.output[0].name = 'Missing_Output';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects cycles that revisit an already consumed activation node', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const secondGemmNode = resolveNamedNode(decodedModel, 'gemm_l2');
          if (!secondGemmNode.output || !decodedGraph.output?.[0]) {
            throw new Error('Expected baseline second Gemm output.');
          }

          secondGemmNode.output[0] = 'Layer_1';
          decodedGraph.output[0].name = 'Loop_Output';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects missing initializer-owned weights', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const firstNode = decodedGraph.node?.[0];
          if (!firstNode?.input) {
            throw new Error('Expected baseline Gemm node inputs.');
          }

          firstNode.input[1] = 'MissingWeight';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('missing-initializer');
    });

    it('rejects Gemm nodes with too many inputs', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const firstNode = resolveNamedNode(decodedModel, 'gemm_l1');
          firstNode.input = [...(firstNode.input ?? []), 'ExtraInput'];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects Gemm nodes whose first input is not the active tensor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const firstNode = resolveNamedNode(decodedModel, 'gemm_l1');
          const inputNames = firstNode.input ?? [];
          firstNode.input = [
            inputNames[1]!,
            inputNames[0]!,
            ...inputNames.slice(2),
          ];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects weight tensors whose feature width no longer matches the active chain tensor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor?.dims) {
            throw new Error('Expected baseline weight tensor dimensions.');
          }

          weightTensor.dims[1] = 3;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('rejects Gemm weight tensors that are not rank 2', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.dims = [2, 2, 1];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('rejects non-float32 initializer tensor types', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.dataType = 10;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-tensor-type');
    });

    it('rejects non-positive initializer dimensions', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor?.dims) {
            throw new Error('Expected baseline weight tensor dimensions.');
          }

          weightTensor.dims[0] = 0;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('rejects float32 raw-data payloads that are not 4-byte aligned', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = Uint8Array.from([1, 2, 3]);
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-tensor-type');
    });

    it('rejects non-canonical Gemm attributes', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const transBAttribute = decodedGraph.node?.[0]?.attribute?.find(
            (attributeEntry) => attributeEntry.name === 'transB',
          );
          if (!transBAttribute) {
            throw new Error('Expected baseline Gemm transB attribute.');
          }

          transBAttribute.i = 0;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects Gemm attributes outside the first supported subset', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const firstNode = resolveNamedNode(decodedModel, 'gemm_l1');
          firstNode.attribute = [{ name: 'gamma', i: 1 }];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects initializer payloads whose decoded value count misses the declared shape', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = undefined;
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('accepts float32 raw-data payloads encoded as numeric arrays', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = Array.from(encodeFloat32RawData([1, 2, 3, 4]));
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = rawBytes as never;
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers[0]!.weightValues[0]).toBeCloseTo(1, 5);
    });

    it('accepts float32 raw-data payloads encoded as base64 strings', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = encodeFloat32RawData([1, 2, 3, 4]);
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = Buffer.from(rawBytes).toString(
            'base64',
          ) as never;
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers[0]!.weightValues[1]).toBeCloseTo(2, 5);
    });

    it('accepts float32 raw-data base64 strings through the atob fallback when Buffer is unavailable', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = encodeFloat32RawData([1, 2, 3, 4]);
      const originalBuffer = (
        globalThis as typeof globalThis & { Buffer?: typeof Buffer }
      ).Buffer;
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const weightTensor = decodedGraph.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = Buffer.from(rawBytes).toString(
            'base64',
          ) as never;
        },
      );

      try {
        (globalThis as typeof globalThis & { Buffer?: typeof Buffer }).Buffer =
          undefined as unknown as typeof Buffer;

        // Act
        const denseChain = normalizeExternalDenseChain(binaryModel);

        // Assert
        expect(denseChain.layers[0]!.weightValues[2]).toBeCloseTo(3, 5);
      } finally {
        (globalThis as typeof globalThis & { Buffer?: typeof Buffer }).Buffer =
          originalBuffer;
      }
    });

    it('accepts single-row bias tensors that still resolve to the output width', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const biasTensor = decodedGraph.initializer?.[1];
          if (!biasTensor?.floatData) {
            throw new Error('Expected baseline bias tensor.');
          }

          biasTensor.dims = [1, biasTensor.floatData.length];
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers[0]!.biasValues.length).toBe(2);
    });

    it('rejects ambiguous bias broadcast shapes', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const biasTensor = decodedGraph.initializer?.[1];
          if (!biasTensor) {
            throw new Error('Expected baseline bias tensor.');
          }

          biasTensor.dims = [2, 2];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('rejects rank-mismatched public inputs', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const modelInput = decodedGraph.input?.[0];
          const inputDimensions = modelInput?.type?.tensorType?.shape?.dim;
          if (!inputDimensions) {
            throw new Error('Expected baseline input dimensions.');
          }

          modelInput!.type!.tensorType!.shape!.dim = [inputDimensions[0]!];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('rank-mismatch');
    });

    it('accepts Gelu with the current tanh approximation contract', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 20;
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.opType = 'Gelu';
          outputActivationNode.attribute = [
            { name: 'approximate', s: new TextEncoder().encode('tanh') },
          ];
        },
      );

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers.at(-1)!.activation).toBe('Gelu');
    });

    it('rejects Gelu attributes outside the supported name contract', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 20;
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.opType = 'Gelu';
          outputActivationNode.attribute = [
            { name: 'mode', s: new TextEncoder().encode('tanh') },
          ];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects Gelu attributes that omit the approximation value', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 20;
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.opType = 'Gelu';
          outputActivationNode.attribute = [{ name: 'approximate' }];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('accepts Gelu approximation values that decode from plain string attributes', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const decodedModel = onnxProto.onnx.ModelProto.toObject(
        onnxProto.onnx.ModelProto.decode(baselineBinaryModel),
      ) as InstanceType<typeof onnxProto.onnx.ModelProto> & {
        graph: {
          node: Array<{
            name?: string | null;
            opType?: string | null;
            attribute?: unknown[] | null;
          }>;
        };
      };
      decodedModel.opsetImport![0]!.version = 20;
      const outputActivationNode = decodedModel.graph.node.find(
        (graphNode) => graphNode.name === 'act_l2',
      );
      if (!outputActivationNode) {
        throw new Error('Expected baseline output activation node.');
      }

      outputActivationNode.opType = 'Gelu';
      outputActivationNode.attribute = [
        { name: 'approximate', s: 'tanh' as never },
      ];
      const toObjectSpy = jest
        .spyOn(onnxProto.onnx.ModelProto, 'toObject')
        .mockReturnValue(decodedModel as never);

      try {
        // Act
        const denseChain = normalizeExternalDenseChain(baselineBinaryModel);

        // Assert
        expect(denseChain.layers.at(-1)!.activation).toBe('Gelu');
      } finally {
        toObjectSpy.mockRestore();
      }
    });

    it('rejects Gelu approximations outside the current contract', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 20;
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.opType = 'Gelu';
          outputActivationNode.attribute = [
            { name: 'approximate', s: new TextEncoder().encode('none') },
          ];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects Gelu when the declared opset is below the current first-lane floor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const decodedGraph = resolveRequiredGraph(decodedModel);
          const outputActivationNode = decodedGraph.node?.find(
            (graphNode) => graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.opType = 'Gelu';
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-opset');
    });

    it('builds a canonical model without activation nodes for identity-only layers', () => {
      // Arrange
      const sourceNetwork = Network.createMLP(2, [2], 1);
      sourceNetwork.nodes[2].squash = methods.Activation.identity;
      sourceNetwork.nodes[3].squash = methods.Activation.identity;
      sourceNetwork.nodes[4].squash = methods.Activation.identity;
      const binaryModel = exportToONNXBinary(sourceNetwork, { opset: 18 });

      // Act
      const normalizedModel = normalizeExternalBinaryOnnxModel(binaryModel);

      // Assert
      expect(
        normalizedModel.graph.node.map((graphNode) => graphNode.name),
      ).toEqual(['gemm_l1', 'gemm_l2']);
    });

    it('keeps single-layer identity outputs on the Gemm tensor without inserting an activation node', () => {
      // Arrange
      const sourceNetwork = Network.createMLP(2, [], 1);
      sourceNetwork.nodes[2].squash = methods.Activation.identity;
      const binaryModel = exportToONNXBinary(sourceNetwork, { opset: 18 });

      // Act
      const denseChain = normalizeExternalDenseChain(binaryModel);

      // Assert
      expect(denseChain.layers[0]!.activation).toBe('Identity');
    });

    it('takes the direct Gemm-to-output path on the plain decoded model surface', () => {
      // Arrange
      const sourceNetwork = Network.createMLP(2, [], 1);
      sourceNetwork.nodes[2].squash = methods.Activation.identity;
      const binaryModel = exportToONNXBinary(sourceNetwork, { opset: 18 });

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        binaryModel,
        (plainModel) => {
          const directOutputTensorName =
            plainModel.graph?.node?.[0]?.output?.[0];
          if (
            !plainModel.graph?.output?.[0] ||
            !plainModel.graph?.node ||
            !directOutputTensorName
          ) {
            throw new Error('Expected single-layer graph output tensor.');
          }

          plainModel.graph.node = [plainModel.graph.node[0]!];
          plainModel.graph.output[0].name = directOutputTensorName;
        },
        () => normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(denseChain.layers[0]!.activation).toBe('Identity');
    });

    it('rejects dead-end Gemm outputs on the plain decoded model surface', () => {
      // Arrange
      const sourceNetwork = Network.createMLP(2, [], 1);
      sourceNetwork.nodes[2].squash = methods.Activation.identity;
      const binaryModel = exportToONNXBinary(sourceNetwork, { opset: 18 });

      // Act
      const failure = withPatchedPlainDecodedModel(
        binaryModel,
        (plainModel) => {
          if (!plainModel.graph?.output?.[0]) {
            throw new Error('Expected single-layer graph output.');
          }

          plainModel.graph.output[0].name = 'Missing_Output';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(binaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('reports the explicit dead-end message when no consumer continues past a Gemm output', () => {
      // Arrange
      const binaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        binaryModel,
        (plainModel) => {
          if (!plainModel.graph?.node?.[0] || !plainModel.graph.output?.[0]) {
            throw new Error('Expected baseline plain model.');
          }

          plainModel.graph.node = [plainModel.graph.node[0]!];
          plainModel.graph.output[0].name = 'Missing_Output';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(binaryModel),
          ),
      );

      // Assert
      expect(
        failure.message.includes('ended before reaching the public output'),
      ).toBe(true);
    });

    it('accepts numeric-array raw data on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = Array.from(encodeFloat32RawData([1, 2, 3, 4]));

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const weightTensor = plainModel.graph?.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = rawBytes as never;
        },
        () => normalizeExternalDenseChain(baselineBinaryModel),
      );

      // Assert
      expect(denseChain.layers[0]!.weightValues[0]).toBeCloseTo(1, 5);
    });

    it('accepts base64-string raw data on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = encodeFloat32RawData([1, 2, 3, 4]);

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const weightTensor = plainModel.graph?.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.floatData = [];
          weightTensor.rawData = Buffer.from(rawBytes).toString(
            'base64',
          ) as never;
        },
        () => normalizeExternalDenseChain(baselineBinaryModel),
      );

      // Assert
      expect(denseChain.layers[0]!.weightValues[1]).toBeCloseTo(2, 5);
    });

    it('accepts atob-fallback raw data on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const rawBytes = encodeFloat32RawData([1, 2, 3, 4]);
      const base64RawBytes = Buffer.from(rawBytes).toString('base64');
      const originalBuffer = (
        globalThis as typeof globalThis & { Buffer?: typeof Buffer }
      ).Buffer;

      try {
        (globalThis as typeof globalThis & { Buffer?: typeof Buffer }).Buffer =
          undefined as unknown as typeof Buffer;

        // Act
        const denseChain = withPatchedPlainDecodedModel(
          baselineBinaryModel,
          (plainModel) => {
            const weightTensor = plainModel.graph?.initializer?.[0];
            if (!weightTensor) {
              throw new Error('Expected baseline weight initializer.');
            }

            weightTensor.floatData = [];
            weightTensor.rawData = base64RawBytes as never;
          },
          () => normalizeExternalDenseChain(baselineBinaryModel),
        );

        // Assert
        expect(denseChain.layers[0]!.weightValues[2]).toBeCloseTo(3, 5);
      } finally {
        (globalThis as typeof globalThis & { Buffer?: typeof Buffer }).Buffer =
          originalBuffer;
      }
    });

    it('rejects revisiting an already consumed activation on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const secondGemmNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'gemm_l2',
          );
          if (!secondGemmNode?.output || !plainModel.graph?.output?.[0]) {
            throw new Error('Expected baseline second Gemm output.');
          }

          secondGemmNode.output[0] = 'Layer_1';
          plainModel.graph.output[0].name = 'Loop_Output';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('reports the explicit acyclic message when a Gemm layer loops back to a visited Gemm', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph?.node || !plainModel.graph.output?.[0]) {
            throw new Error('Expected baseline plain model.');
          }

          plainModel.graph.node = plainModel.graph.node.filter(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'gemm_l1' ||
              graphNode.name === 'act_l1' ||
              graphNode.name === 'gemm_l2' ||
              graphNode.name === 'act_l2',
          );
          const outputActivationNode = plainModel.graph.node.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode?.output) {
            throw new Error('Expected output activation node.');
          }

          outputActivationNode.output[0] = 'input';
          plainModel.graph.output[0].name = 'Loop_Output';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('reports the explicit unary-activation message when an activation keeps an extra input', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph?.node || !plainModel.graph.output?.[0]) {
            throw new Error('Expected baseline plain model.');
          }

          plainModel.graph.node = plainModel.graph.node.filter(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'gemm_l1' || graphNode.name === 'act_l1',
          );
          const outputActivationNode = plainModel.graph.node.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l1',
          );
          const activationInputName = outputActivationNode?.input?.[0];
          if (!outputActivationNode || !activationInputName) {
            throw new Error('Expected output activation node.');
          }

          plainModel.graph.output[0].name =
            outputActivationNode.output?.[0] ?? 'Missing_Output';
          outputActivationNode.input = [activationInputName, 'Extra_Input'];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects malformed unary activations on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.input = ['Layer_2', 'Extra_Input'];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rethrows non-Error decode failures through the string fallback path', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const decodeSpy = jest
        .spyOn(onnxProto.onnx.ModelProto, 'decode')
        .mockImplementation(() => {
          throw 'forced string decode failure';
        });

      try {
        // Act
        const failure = captureExternalImportFailure(() =>
          normalizeExternalDenseChain(baselineBinaryModel),
        );

        // Assert
        expect(failure.message).toBe('forced string decode failure');
      } finally {
        decodeSpy.mockRestore();
      }
    });

    it('rejects plain models that omit initializer and input arrays', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph) {
            throw new Error('Expected baseline graph.');
          }

          plainModel.graph.initializer = undefined;
          plainModel.graph.input = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects plain models that omit output arrays', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph) {
            throw new Error('Expected baseline graph.');
          }

          plainModel.graph.output = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('falls back missing public tensor names to empty strings on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph?.input?.[0] || !plainModel.graph.output?.[0]) {
            throw new Error('Expected baseline public tensors.');
          }

          plainModel.graph.input[0].name = null;
          plainModel.graph.output[0].name = null;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects missing opset versions through the numeric fallback', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.opsetImport?.[0]) {
            throw new Error('Expected baseline opset import.');
          }

          plainModel.opsetImport[0].version = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-opset');
    });

    it('rejects null-named initializers on the plain decoded model surface', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const weightTensor = plainModel.graph?.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.name = null;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('invalid-model');
    });

    it('accepts supporting value-info entries with null names', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.graph) {
            throw new Error('Expected baseline graph.');
          }

          plainModel.graph.valueInfo = [
            {
              name: null,
              type: {
                tensorType: {
                  elemType: 1,
                  shape: {
                    dim: [{ dimParam: 'N' }, { dimValue: 2 }],
                  },
                },
              },
            },
          ];
        },
        () => normalizeExternalDenseChain(baselineBinaryModel),
      );

      // Assert
      expect(denseChain.layers.length).toBe(2);
    });

    it('rejects tensors whose shape omits the dimension list', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const modelInput =
            plainModel.graph?.input?.[0]?.type?.tensorType?.shape;
          if (!modelInput) {
            throw new Error('Expected baseline input shape.');
          }

          modelInput.dim = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('rank-mismatch');
    });

    it('rejects disconnected nodes whose input list is omitted', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          plainModel.graph?.node?.push({
            name: 'orphan_without_inputs',
            opType: 'Identity',
            input: undefined,
            output: ['orphan_output'],
          });
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects custom-domain nodes even when both name and opType labels are missing', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline first node.');
          }

          firstNode.name = null;
          firstNode.opType = null as never;
          firstNode.domain = 'custom.domain';
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-domain');
    });

    it('rejects missing node operation labels with the unknown-node fallback text', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline first node.');
          }

          firstNode.opType = null as never;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.message.includes('unknown')).toBe(true);
    });

    it('fills missing Gemm bias inputs with zeros', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode?.input) {
            throw new Error('Expected baseline first node inputs.');
          }

          firstNode.input = firstNode.input.slice(0, 2);
        },
        () => normalizeExternalDenseChain(baselineBinaryModel),
      );

      // Assert
      expect(denseChain.layers[0]!.biasValues).toEqual([0, 0]);
    });

    it('accepts Gemm nodes whose attribute list is omitted', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const denseChain = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline first node.');
          }

          firstNode.attribute = undefined;
        },
        () => normalizeExternalDenseChain(baselineBinaryModel),
      );

      // Assert
      expect(denseChain.layers.length).toBe(2);
    });

    it('rejects Gemm attributes whose numeric value is omitted', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline first node.');
          }

          firstNode.attribute = [{ name: 'transB' }];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects null Gemm attribute names through the empty-string fallback', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const firstNode = plainModel.graph?.node?.[0];
          if (!firstNode) {
            throw new Error('Expected baseline first node.');
          }

          firstNode.attribute = [{ name: null as never, i: 1 }];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects missing initializer dimensions through the empty-dims fallback', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const weightTensor = plainModel.graph?.initializer?.[0];
          if (!weightTensor) {
            throw new Error('Expected baseline weight initializer.');
          }

          weightTensor.dims = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('shape-mismatch');
    });

    it('rejects unsupported activations whose opType label is missing', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.opType = null as never;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.message.includes('unknown')).toBe(true);
    });

    it('rejects activation nodes whose input array is omitted', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.input = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects activation nodes whose output array is omitted', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.output = undefined;
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects activation nodes whose single input references the wrong tensor', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.input = ['Wrong_Tensor'];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });

    it('rejects Gelu attributes whose name is null', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();

      // Act
      const failure = withPatchedPlainDecodedModel(
        baselineBinaryModel,
        (plainModel) => {
          if (!plainModel.opsetImport?.[0]) {
            throw new Error('Expected baseline opset import.');
          }

          plainModel.opsetImport[0].version = 20;
          const outputActivationNode = plainModel.graph?.node?.find(
            (graphNode: { name?: string | null }) =>
              graphNode.name === 'act_l2',
          );
          if (!outputActivationNode) {
            throw new Error('Expected baseline output activation node.');
          }

          outputActivationNode.opType = 'Gelu';
          outputActivationNode.attribute = [
            { name: null as never, s: new TextEncoder().encode('tanh') },
          ];
        },
        () =>
          captureExternalImportFailure(() =>
            normalizeExternalDenseChain(baselineBinaryModel),
          ),
      );

      // Assert
      expect(failure.category).toBe('unsupported-attribute');
    });

    it('rejects unary activations that do not preserve one-input one-output flow', () => {
      // Arrange
      const baselineBinaryModel = createBaselineBinaryModel();
      const binaryModel = mutateBinaryModel(
        baselineBinaryModel,
        (decodedModel) => {
          const outputActivationNode = resolveNamedNode(decodedModel, 'act_l2');
          outputActivationNode.input = ['Layer_2', 'Extra_Input'];
        },
      );

      // Act
      const failure = captureExternalImportFailure(() =>
        normalizeExternalDenseChain(binaryModel),
      );

      // Assert
      expect(failure.category).toBe('unsupported-topology');
    });
  });
});
