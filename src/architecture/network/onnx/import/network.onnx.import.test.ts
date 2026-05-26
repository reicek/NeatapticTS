import Network from '../../network';
import * as methods from '../../../../methods/methods';
import { exportToONNX, importFromONNX } from '../network.onnx';
import type {
  AttentionMapping,
  ConcatMapping,
  OnnxModel,
} from '../network.onnx';

type AdvancedGraphAwareNetwork = Network & {
  _onnxAdvancedGraph?: {
    crossLayerConnections: {
      sourceNodeIndex: number;
      sourceLayerIndex: number;
      targetNodeIndex: number;
      targetLayerIndex: number;
      branchTensorName: string;
    }[];
    concatMerges?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      concatNodeName: string;
      concatOutputName: string;
      inputOrder: 'previous_then_source';
    }[];
    residualAdds?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      branchTensorName: string;
      mergeNodeName: string;
      mergeOutputName: string;
    }[];
    sharedInitializerAliases?: {
      aliasTensorName: string;
      canonicalTensorName: string;
      initializerKind: string;
    }[];
    attentionBlocks?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      sequenceLength: number;
      modelWidth: number;
      heads: number;
      shadowOutputName: string;
    }[];
  };
};

function createFeedForwardSkipConnectionNetwork(): Network {
  const network = Network.createMLP(2, [2], 1);
  const inputNode = network.nodes[0];
  const outputNode = network.nodes.at(-1)!;

  inputNode.connect(outputNode, 0.75);
  Network.rebuildConnections(network);
  return network;
}

function createResidualAddRoundTripNetwork(): Network {
  const network = Network.createMLP(2, [2], 2);
  const firstInputNode = network.nodes[0];
  const secondInputNode = network.nodes[1];
  const firstOutputNode = network.nodes.at(-2)!;
  const secondOutputNode = network.nodes.at(-1)!;

  firstInputNode.connect(firstOutputNode, 0.75);
  secondInputNode.connect(secondOutputNode, -0.5);
  Network.rebuildConnections(network);
  return network;
}

function createConcatMergeRoundTripNetwork(): Network {
  const network = Network.createMLP(2, [2], 1);
  const firstInputNode = network.nodes[0];
  const secondInputNode = network.nodes[1];
  const outputNode = network.nodes.at(-1)!;

  firstInputNode.connect(outputNode, 0.75);
  secondInputNode.connect(outputNode, -0.5);
  Network.rebuildConnections(network);
  return network;
}

function createSharedInitializerAliasNetwork(): Network {
  const network = Network.createMLP(2, [2, 2], 1);
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );

  hiddenNodes[0].bias = 0.5;
  hiddenNodes[1].bias = -0.25;
  hiddenNodes[2].bias = 0.5;
  hiddenNodes[3].bias = -0.25;

  hiddenNodes[0].connections.in[0].weight = 0.1;
  hiddenNodes[0].connections.in[1].weight = 0.2;
  hiddenNodes[1].connections.in[0].weight = 0.3;
  hiddenNodes[1].connections.in[1].weight = 0.4;
  hiddenNodes[2].connections.in[0].weight = 0.1;
  hiddenNodes[2].connections.in[1].weight = 0.2;
  hiddenNodes[3].connections.in[0].weight = 0.3;
  hiddenNodes[3].connections.in[1].weight = 0.4;

  return network;
}

function createAttentionProjectionWeights(width: number): number[] {
  return Array.from({ length: width * width }, (_unused, weightIndex) =>
    weightIndex % (width + 1) === 0 ? 1 : 0,
  );
}

function createAttentionRoundTripMapping(): AttentionMapping {
  const projectionWeights = createAttentionProjectionWeights(4);

  return {
    layerIndex: 1,
    sequenceLength: 2,
    modelWidth: 4,
    heads: 2,
    queryWeights: projectionWeights,
    keyWeights: projectionWeights,
    valueWeights: projectionWeights,
    queryBias: [0, 0, 0, 0],
    keyBias: [0, 0, 0, 0],
    valueBias: [0, 0, 0, 0],
  };
}

function createConcatRoundTripMapping(): ConcatMapping {
  return {
    sourceLayerIndex: 0,
    targetLayerIndex: 2,
  };
}

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

    describe('given a Conv-mapped 9-4-1 network was exported', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = Network.createMLP(9, [4], 1);

        sourceNetwork.nodes
          .filter((nodeEntry) => nodeEntry.type === 'hidden')
          .forEach((hiddenNode) => {
            hiddenNode.squash = methods.Activation.relu;
          });

        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: [
            {
              layerIndex: 1,
              inHeight: 3,
              inWidth: 3,
              inChannels: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 2,
              outWidth: 2,
              outChannels: 1,
            },
          ],
        });

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when reading imported hidden activations', () => {
        it('restores the Conv-hidden activation instead of defaulting to identity', () => {
          // Assert
          expect(
            importedNetwork.nodes
              .filter((nodeEntry) => nodeEntry.type === 'hidden')
              .every(
                (hiddenNode) => hiddenNode.squash === methods.Activation.relu,
              ),
          ).toBe(true);
        });
      });
    });

    describe('given an exported network includes feed-forward skip metadata', () => {
      describe('when the model is imported back into the runtime', () => {
        it('attaches the advanced-graph audit payload without changing the fallback scaffold', () => {
          // Arrange
          const sourceNetwork = createFeedForwardSkipConnectionNetwork();
          const onnxModel = exportToONNX(sourceNetwork, {
            includeMetadata: true,
          });

          // Act
          const importedNetwork = importFromONNX(
            onnxModel,
          ) as AdvancedGraphAwareNetwork;

          // Assert
          expect(importedNetwork._onnxAdvancedGraph).toEqual({
            crossLayerConnections: [
              {
                sourceNodeIndex: 0,
                sourceLayerIndex: 0,
                targetNodeIndex: 4,
                targetLayerIndex: 2,
                branchTensorName: 'Branch_l0_to_l2_from_n0_to_n4',
              },
            ],
            residualAdds: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                branchTensorName: 'ResidualBranch_l0_to_l2',
                mergeNodeName: 'residual_add_l2',
                mergeOutputName: 'ResidualAdd_2',
              },
            ],
          });
        });

        it('restores one-hop residual inference and attaches residual merge audit metadata', () => {
          // Arrange
          const sourceNetwork = createResidualAddRoundTripNetwork();
          const sampleInput = [0.25, 0.75];
          const onnxModel = exportToONNX(sourceNetwork, {
            includeMetadata: true,
          });
          const expectedOutput = sourceNetwork.activate(sampleInput);

          // Act
          const importedNetwork = importFromONNX(
            onnxModel,
          ) as AdvancedGraphAwareNetwork;
          const importedOutput = importedNetwork.activate(sampleInput);

          // Assert
          expect({
            inferenceMatches:
              importedOutput.length === expectedOutput.length &&
              importedOutput.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - expectedOutput[outputIndex]!) <=
                  Number.EPSILON,
              ),
            residualAdds: importedNetwork._onnxAdvancedGraph?.residualAdds,
          }).toEqual({
            inferenceMatches: true,
            residualAdds: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                branchTensorName: 'ResidualBranch_l0_to_l2',
                mergeNodeName: 'residual_add_l2',
                mergeOutputName: 'ResidualAdd_2',
              },
            ],
          });
        });
      });
    });

    describe('given an exported network includes shared dense initializer aliases', () => {
      describe('when the model is imported back into the runtime', () => {
        it('preserves inference and attaches the shared-initializer audit payload', () => {
          // Arrange
          const sourceNetwork = createSharedInitializerAliasNetwork();
          const sampleInput = [0.25, 0.75];
          const onnxModel = exportToONNX(sourceNetwork, {
            includeMetadata: true,
          });
          const expectedOutput = sourceNetwork.activate(sampleInput);

          // Act
          const importedNetwork = importFromONNX(
            onnxModel,
          ) as AdvancedGraphAwareNetwork;
          const importedOutput = importedNetwork.activate(sampleInput);

          // Assert
          expect({
            inferenceMatches:
              importedOutput.length === expectedOutput.length &&
              importedOutput.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - expectedOutput[outputIndex]!) <=
                  Number.EPSILON,
              ),
            sharedInitializerAliases:
              importedNetwork._onnxAdvancedGraph?.sharedInitializerAliases,
          }).toEqual({
            inferenceMatches: true,
            sharedInitializerAliases: [
              {
                aliasTensorName: 'W1',
                canonicalTensorName: 'W0',
                initializerKind: 'dense_weight',
              },
              {
                aliasTensorName: 'B1',
                canonicalTensorName: 'B0',
                initializerKind: 'dense_bias',
              },
            ],
          });
        });
      });
    });

    describe('given an exported network includes the supported concat merge metadata', () => {
      describe('when the model is imported back into the runtime', () => {
        it('preserves concat-backed inference and attaches the concat merge audit payload', () => {
          // Arrange
          const sourceNetwork = createConcatMergeRoundTripNetwork();
          const sampleInput = [0.25, 0.75];
          const onnxModel = exportToONNX(sourceNetwork, {
            includeMetadata: true,
            concatMappings: [createConcatRoundTripMapping()],
          });
          const expectedOutput = sourceNetwork.activate(sampleInput);

          // Act
          const importedNetwork = importFromONNX(
            onnxModel,
          ) as AdvancedGraphAwareNetwork;
          const importedOutput = importedNetwork.activate(sampleInput);

          // Assert
          expect({
            inferenceMatches:
              importedOutput.length === expectedOutput.length &&
              importedOutput.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - expectedOutput[outputIndex]!) <=
                  Number.EPSILON,
              ),
            concatMerges: importedNetwork._onnxAdvancedGraph?.concatMerges,
          }).toEqual({
            inferenceMatches: true,
            concatMerges: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                concatNodeName: 'concat_merge_l0_to_l2',
                concatOutputName: 'ConcatMerge_0_to_2',
                inputOrder: 'previous_then_source',
              },
            ],
          });
        });
      });
    });

    describe('given an exported network includes the supported attention shadow metadata', () => {
      describe('when the model is imported back into the runtime', () => {
        it('preserves dense fallback inference and attaches the attention-block audit payload', () => {
          // Arrange
          const sourceNetwork = Network.createMLP(8, [8], 2);
          const sampleInput = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75];
          const onnxModel = exportToONNX(sourceNetwork, {
            includeMetadata: true,
            attentionMappings: [createAttentionRoundTripMapping()],
          });
          const expectedOutput = sourceNetwork.activate(sampleInput);

          // Act
          const importedNetwork = importFromONNX(
            onnxModel,
          ) as AdvancedGraphAwareNetwork;
          const importedOutput = importedNetwork.activate(sampleInput);

          // Assert
          expect({
            inferenceMatches:
              importedOutput.length === expectedOutput.length &&
              importedOutput.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - expectedOutput[outputIndex]!) <=
                  Number.EPSILON,
              ),
            attentionBlocks:
              importedNetwork._onnxAdvancedGraph?.attentionBlocks,
          }).toEqual({
            inferenceMatches: true,
            attentionBlocks: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 1,
                sequenceLength: 2,
                modelWidth: 4,
                heads: 2,
                shadowOutputName: 'AttentionShadow_1',
              },
            ],
          });
        });
      });
    });
  });
});
