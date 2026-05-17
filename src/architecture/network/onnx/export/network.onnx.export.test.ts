import Network from '../../network';
import Node from '../../../node';
import * as methods from '../../../../methods/methods';
import { exportToONNX, importFromONNX } from '../network.onnx';
import type { OnnxModel } from '../network.onnx';
import type { OnnxExportOptions } from './network.onnx.export.types';
import { runOnnxExportFlow } from './network.onnx.export-flow.utils';
import {
  appendAdvancedGraphMetadata,
  appendConcatMergeMetadata,
  appendResidualAddMetadata,
  resolveOneHopResidualSourceLayerIndex,
} from './network.onnx.export-advanced-graph.utils';
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
  name: string;
  attributes?: OnnxAttributeView[];
};
type OnnxInitializerView = {
  name: string;
  data_type?: number;
  float_data?: number[];
  int32_data?: number[];
};

type MixedTargetPreservedUnaryActivationCase = {
  activationFunction: (x: number, derivate?: boolean) => number;
  activationLabel: string;
  activationNodeType: string;
  opset?: number;
  outputRange: {
    min: number;
    max: number;
  };
  sampleInput: number[];
  tolerance: number;
};

const activationOperations = new Set([
  'Tanh',
  'Sigmoid',
  'Relu',
  'Identity',
  'Softplus',
  'Softsign',
  'Selu',
  'Mish',
  'Gelu',
]);

const additionalMixedTargetPreservedUnaryActivationCases: MixedTargetPreservedUnaryActivationCase[] =
  [
    {
      activationFunction: methods.Activation.relu,
      activationLabel: 'Relu',
      activationNodeType: 'Relu',
      outputRange: { min: 0, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.sigmoid,
      activationLabel: 'Sigmoid',
      activationNodeType: 'Sigmoid',
      outputRange: { min: 0, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.tanh,
      activationLabel: 'Tanh',
      activationNodeType: 'Tanh',
      outputRange: { min: -1, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.softsign,
      activationLabel: 'Softsign',
      activationNodeType: 'Softsign',
      outputRange: { min: -1, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.selu,
      activationLabel: 'Selu',
      activationNodeType: 'Selu',
      outputRange: { min: -1, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.gelu,
      activationLabel: 'Gelu',
      activationNodeType: 'Gelu',
      opset: 20,
      outputRange: { min: -0.5, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
    {
      activationFunction: methods.Activation.mish,
      activationLabel: 'Mish',
      activationNodeType: 'Mish',
      opset: 18,
      outputRange: { min: -0.5, max: 1 },
      sampleInput: [0.25, -0.75],
      tolerance: 2e-2,
    },
  ];

function toInputs(model: OnnxModel): OnnxValueInfo[] {
  return model.graph.inputs as OnnxValueInfo[];
}

function toOutputs(model: OnnxModel): OnnxValueInfo[] {
  return model.graph.outputs as OnnxValueInfo[];
}

function toInitializers(model: OnnxModel): OnnxInitializerView[] {
  return model.graph.initializer as OnnxInitializerView[];
}

function getInitializer(
  model: OnnxModel,
  initializerName: string,
): OnnxInitializerView | undefined {
  return toInitializers(model).find(
    (initializerEntry) => initializerEntry.name === initializerName,
  );
}

function toNodes(model: OnnxModel): OnnxNodeView[] {
  return model.graph.node as OnnxNodeView[];
}

function createMinimalOnnxModel(
  metadataProps?: OnnxModel['metadata_props'],
): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
    metadata_props: metadataProps,
  } as unknown as OnnxModel;
}

function getMetadataValue(model: OnnxModel, key: string): string | undefined {
  return model.metadata_props?.find((entry) => entry.key === key)?.value;
}

function getParsedMetadataArray(
  model: OnnxModel,
  key: string,
): string[] | undefined {
  const metadataValue = getMetadataValue(model, key);
  return metadataValue ? (JSON.parse(metadataValue) as string[]) : undefined;
}

function getHiddenNodes(network: Network): Node[] {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden');
}

function getRequiredInitializer(
  model: OnnxModel,
  initializerName: string,
): OnnxInitializerView {
  const initializerEntry = getInitializer(model, initializerName);

  if (!initializerEntry) {
    throw new Error(`Missing initializer ${initializerName}.`);
  }

  return initializerEntry;
}

function resolveQuantizedIntegerBounds(dataType: number | undefined): {
  minimumValue: number;
  maximumValue: number;
} {
  if (dataType === 3) {
    return {
      minimumValue: -128,
      maximumValue: 127,
    };
  }

  return {
    minimumValue: 0,
    maximumValue: 255,
  };
}

function clampQuantizedInteger(
  value: number,
  minimumValue: number,
  maximumValue: number,
): number {
  return Math.min(maximumValue, Math.max(minimumValue, value));
}

function roundToNearestEven(value: number): number {
  const lowerInteger = Math.floor(value);
  const fractionalOffset = value - lowerInteger;

  if (fractionalOffset < 0.5) {
    return lowerInteger;
  }

  if (fractionalOffset > 0.5) {
    return lowerInteger + 1;
  }

  return lowerInteger % 2 === 0 ? lowerInteger : lowerInteger + 1;
}

function applySupportedDenseActivation(
  model: OnnxModel,
  layerIndex: number,
  activationInputValues: number[],
): number[] {
  const activationNode = toNodes(model).find(
    (nodeEntry) => nodeEntry.name === `act_l${layerIndex}`,
  );

  if (!activationNode || activationNode.op_type === 'Identity') {
    return activationInputValues;
  }

  const supportedUnaryActivation = {
    Gelu: methods.Activation.gelu,
    Mish: methods.Activation.mish,
    Relu: methods.Activation.relu,
    Selu: methods.Activation.selu,
    Sigmoid: methods.Activation.sigmoid,
    Softplus: methods.Activation.softplus,
    Softsign: methods.Activation.softsign,
    Tanh: methods.Activation.tanh,
  }[activationNode.op_type];

  if (supportedUnaryActivation) {
    return activationInputValues.map((activationInputValue) =>
      supportedUnaryActivation(activationInputValue),
    );
  }

  throw new Error(`Unsupported qlinear test activation ${activationNode.op_type}.`);
}

function applySupportedOneOutputActivation(
  model: OnnxModel,
  layerIndex: number,
  activationInputValue: number,
): number {
  return applySupportedDenseActivation(
    model,
    layerIndex,
    [activationInputValue],
  )[0]!;
}

function evaluateOneOutputQLinearDenseLayerOutput(
  model: OnnxModel,
  layerIndex: number,
  inputValues: number[],
): number {
  const inputScale = getRequiredInitializer(
    model,
    `QuantDenseInputScale_l${layerIndex}`,
  ).float_data?.[0]!;
  const inputZeroPointInitializer = getRequiredInitializer(
    model,
    `QuantDenseInputZeroPoint_l${layerIndex}`,
  );
  const inputZeroPoint = inputZeroPointInitializer.int32_data?.[0]!;
  const weightScale = getRequiredInitializer(
    model,
    `QuantDenseWeightScale_l${layerIndex}`,
  ).float_data?.[0]!;
  const weightZeroPoint = getRequiredInitializer(
    model,
    `QuantDenseWeightZeroPoint_l${layerIndex}`,
  ).int32_data?.[0]!;
  const outputScale = getRequiredInitializer(
    model,
    `QuantDenseOutputScale_l${layerIndex}`,
  ).float_data?.[0]!;
  const outputZeroPointInitializer = getRequiredInitializer(
    model,
    `QuantDenseOutputZeroPoint_l${layerIndex}`,
  );
  const outputZeroPoint = outputZeroPointInitializer.int32_data?.[0]!;
  const biasValues = getRequiredInitializer(model, `B${layerIndex - 1}`).float_data!;
  const quantizedWeightValues = getRequiredInitializer(
    model,
    `QuantDenseWeight_l${layerIndex}`,
  ).int32_data!;

  if (biasValues.length !== 1) {
    throw new Error('Test helper expects a one-output dense layer.');
  }

  const inputCount = quantizedWeightValues.length / biasValues.length;
  const inputBounds = resolveQuantizedIntegerBounds(
    inputZeroPointInitializer.data_type,
  );
  const outputBounds = resolveQuantizedIntegerBounds(
    outputZeroPointInitializer.data_type,
  );

  const quantizedInputValues = inputValues.map((inputValue) =>
    clampQuantizedInteger(
      roundToNearestEven(inputValue / inputScale + inputZeroPoint),
      inputBounds.minimumValue,
      inputBounds.maximumValue,
    ),
  );
  const quantizedAccumulator = Array.from(
    { length: inputCount },
    (_unused, inputIndex) =>
      (quantizedInputValues[inputIndex]! - inputZeroPoint) *
      (quantizedWeightValues[inputIndex]! - weightZeroPoint),
  ).reduce((runningTotal, accumulatorValue) => runningTotal + accumulatorValue, 0);
  const quantizedOutputValue = clampQuantizedInteger(
    roundToNearestEven(
      (inputScale * weightScale * quantizedAccumulator) / outputScale +
        outputZeroPoint,
    ),
    outputBounds.minimumValue,
    outputBounds.maximumValue,
  );
  const dequantizedOutputValue =
    (quantizedOutputValue - outputZeroPoint) * outputScale;

  return applySupportedOneOutputActivation(
    model,
    layerIndex,
    dequantizedOutputValue + biasValues[0]!,
  );
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

function createMixedTargetPreservedUnaryFixture(
  activationFunction: (x: number, derivate?: boolean) => number,
  sampleInput: number[],
): {
  baselineOutput: number;
  hiddenLayerOutputs: number[];
  network: Network;
} {
  const network = Network.createMLP(2, [2], 1);
  const hiddenNodes = getHiddenNodes(network);
  network.connections[0].weight = 0.5;
  network.connections[1].weight = -0.25;
  network.connections[2].weight = 0.75;
  network.connections[3].weight = 0.5;
  network.connections[4].weight = 1.25;
  network.connections[5].weight = -0.5;
  hiddenNodes[0]!.bias = 0.125;
  hiddenNodes[1]!.bias = -0.25;
  hiddenNodes[0]!.squash = methods.Activation.relu;
  hiddenNodes[1]!.squash = methods.Activation.relu;
  (network.nodes.at(-1) as Node).bias = 0.0625;
  (network.nodes.at(-1) as Node).squash = activationFunction;
  const baselineOutput = network.activate(sampleInput)[0] as number;
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const inputValuesByNode = new Map(
    inputNodes.map((inputNode, inputIndex) => [
      inputNode,
      sampleInput[inputIndex]!,
    ]),
  );
  const hiddenLayerOutputs = hiddenNodes.map((hiddenNode) =>
    hiddenNode.squash(
      hiddenNode.connections.in.reduce(
        (weightedSum, inboundConnection) =>
          weightedSum +
          (inputValuesByNode.get(inboundConnection.from) ?? 0) *
            inboundConnection.weight,
        hiddenNode.bias,
      ),
    ),
  );

  return {
    baselineOutput,
    hiddenLayerOutputs,
    network,
  };
}

function createMixedTargetQuantizationOptions(
  outputRange: { min: number; max: number },
  opset?: number,
): OnnxExportOptions {
  return {
    includeMetadata: true,
    ...(opset !== undefined ? { opset } : {}),
    quantization: {
      mode: 'static-8bit',
      targets: ['dense'],
      calibration: {
        source: 'external',
        layerTargets: [
          {
            target: 'dense',
            layerIndex: 1,
            inputRange: { min: -1, max: 1 },
            outputRange: { min: 0, max: 1 },
          },
          {
            target: 'dense',
            layerIndex: 2,
            inputRange: { min: 0, max: 0.5 },
            outputRange,
          },
        ],
      },
      representation: 'qlinear',
    },
  };
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

function createFeedForwardSkipConnectionNetwork(): Network {
  const network = Network.createMLP(2, [2], 1);
  const inputNode = network.nodes[0];
  const outputNode = network.nodes.at(-1)!;

  inputNode.connect(outputNode, 0.75);
  Network.rebuildConnections(network);
  return network;
}

function createResidualAddExportNetwork(): Network {
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

function createConcatMergeExportNetwork(): Network {
  const network = Network.createMLP(2, [2], 1);
  const firstInputNode = network.nodes[0];
  const secondInputNode = network.nodes[1];
  const targetLayerNodes = [network.nodes.at(-1)!];

  firstInputNode.connect(targetLayerNodes[0], 0.75);
  secondInputNode.connect(targetLayerNodes[0], 0.5);
  Network.rebuildConnections(network);
  return network;
}

function createMultiHopSkipConnectionNetwork(): Network {
  const network = Network.createMLP(2, [2, 2], 2);
  const firstInputNode = network.nodes[0];
  const firstOutputNode = network.nodes.at(-2)!;

  firstInputNode.connect(firstOutputNode, 0.75);
  Network.rebuildConnections(network);
  return network;
}

type AttentionMappingProbe = {
  layerIndex: number;
  sequenceLength: number;
  modelWidth: number;
  heads: number;
  queryWeights: number[];
  keyWeights: number[];
  valueWeights: number[];
  queryBias: number[];
  keyBias: number[];
  valueBias: number[];
};

function createAttentionShadowExportNetwork(): Network {
  return Network.createMLP(8, [8], 2);
}

function createIdentityProjectionWeights(width: number): number[] {
  return Array.from({ length: width * width }, (_unused, weightIndex) =>
    weightIndex % (width + 1) === 0 ? 1 : 0,
  );
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
      describe('when runOnnxExportFlow() is called without explicit options', () => {
        it('uses the default export options and keeps the input width as the first input dimension', () => {
          // Arrange
          const network = new Network(1, 1, {});

          // Act
          const onnxModel = runOnnxExportFlow(network);

          // Assert
          expect(
            toInputs(onnxModel)[0].type.tensor_type.shape.dim[0]?.dim_value,
          ).toBe(1);
        });
      });

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
    describe('given a hidden layer uses the softplus activation', () => {
      it('emits a Softplus activation node', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        getHiddenNodes(network)[0].squash = methods.Activation.softplus;

        // Act
        const emittedNodes = toNodes(exportToONNX(network));

        // Assert
        expect(
          emittedNodes.some((nodeEntry) => nodeEntry.op_type === 'Softplus'),
        ).toBe(true);
      });
    });

    describe('given a hidden layer uses the gelu activation at opset 20', () => {
      it('emits a Gelu node with the tanh approximation attribute', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        getHiddenNodes(network)[0].squash = methods.Activation.gelu;

        // Act
        const emittedNodes = toNodes(exportToONNX(network, { opset: 20 }));

        // Assert
        expect(
          emittedNodes.some(
            (nodeEntry) =>
              nodeEntry.op_type === 'Gelu' &&
              nodeEntry.attributes?.some(
                (attributeEntry) =>
                  attributeEntry.name === 'approximate' &&
                  (attributeEntry as OnnxAttributeView & { s?: string }).s === 'tanh',
              ) === true,
          ),
        ).toBe(true);
      });
    });

    describe('given a hidden layer uses the identity activation', () => {
      it('prunes Identity activation nodes from the exported graph', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        getHiddenNodes(network)[0].squash = methods.Activation.identity;

        // Act
        const emittedNodes = toNodes(exportToONNX(network));

        // Assert
        expect(
          emittedNodes.some((nodeEntry) => nodeEntry.op_type === 'Identity'),
        ).toBe(false);
      });
    });

    describe('given Phase 7 quantization metadata and the first dense qlinear slice are active', () => {
      it('records the static quantization calibration contract and emits dense parameter initializers', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 0.75;
        getHiddenNodes(network)[0].bias = 0.125;
        (network.nodes.at(-1) as Node).bias = -0.25;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              packetId: 'calibration-packet',
              sampleCount: 32,
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: -0.5, max: 0.75 },
                  outputRange: { min: -0.25, max: 0.5 },
                },
              ],
              weightRangePolicy: 'min-max',
              zeroInclusion: 'required',
              activationSymmetry: 'asymmetric',
              weightSymmetry: 'symmetric',
              roundingMode: 'nearest-even',
            },
            activationGranularity: 'per-tensor',
            weightGranularity: 'per-tensor',
            representation: 'qlinear',
          },
        });

        // Assert
        expect({
          requestedQuantizationMode: getMetadataValue(
            onnxModel,
            'requested_quantization_mode',
          ),
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          calibrationSource: getMetadataValue(
            onnxModel,
            'quantization_calibration_source',
          ),
          activationGranularity: getMetadataValue(
            onnxModel,
            'quantization_activation_granularity',
          ),
          weightGranularity: getMetadataValue(
            onnxModel,
            'quantization_weight_granularity',
          ),
          calibrationPacketId: getMetadataValue(
            onnxModel,
            'quantization_calibration_packet_id',
          ),
          calibrationSampleCount: getMetadataValue(
            onnxModel,
            'quantization_calibration_sample_count',
          ),
          calibrationTargetCount: getMetadataValue(
            onnxModel,
            'quantization_calibration_target_count',
          ),
          weightRangePolicy: getMetadataValue(
            onnxModel,
            'quantization_weight_range_policy',
          ),
          zeroInclusionPolicy: getMetadataValue(
            onnxModel,
            'quantization_zero_inclusion_policy',
          ),
          activationSymmetry: getMetadataValue(
            onnxModel,
            'quantization_activation_symmetry',
          ),
          weightSymmetry: getMetadataValue(
            onnxModel,
            'quantization_weight_symmetry',
          ),
          roundingMode: getMetadataValue(
            onnxModel,
            'quantization_rounding_mode',
          ),
          denseInputScale: getInitializer(
            onnxModel,
            'QuantDenseInputScale_l1',
          )?.float_data?.[0],
          denseInputZeroPointType: getInitializer(
            onnxModel,
            'QuantDenseInputZeroPoint_l1',
          )?.data_type,
          denseInputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseInputZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightScale: getInitializer(
            onnxModel,
            'QuantDenseWeightScale_l1',
          )?.float_data?.[0],
          denseWeightZeroPointType: getInitializer(
            onnxModel,
            'QuantDenseWeightZeroPoint_l1',
          )?.data_type,
          denseWeightZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseWeightZeroPoint_l1',
          )?.int32_data?.[0],
          denseOutputScale: getInitializer(
            onnxModel,
            'QuantDenseOutputScale_l2',
          )?.float_data?.[0],
          denseOutputZeroPointType: getInitializer(
            onnxModel,
            'QuantDenseOutputZeroPoint_l2',
          )?.data_type,
          denseOutputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseOutputZeroPoint_l2',
          )?.int32_data?.[0],
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          requestedQuantizationMode: 'static-8bit',
          effectiveQuantizationMode: 'static-8bit',
          calibrationSource: 'external',
          activationGranularity: 'per-tensor',
          weightGranularity: 'per-tensor',
          calibrationPacketId: 'calibration-packet',
          calibrationSampleCount: '32',
          calibrationTargetCount: '2',
          weightRangePolicy: 'min-max',
          zeroInclusionPolicy: 'required',
          activationSymmetry: 'asymmetric',
          weightSymmetry: 'symmetric',
          roundingMode: 'nearest-even',
          denseInputScale: 0.00784313725490196,
          denseInputZeroPointType: 2,
          denseInputZeroPoint: 128,
          denseWeightScale: 0.003937007874015748,
          denseWeightZeroPointType: 3,
          denseWeightZeroPoint: 0,
          denseOutputScale: 0.0029411764705882353,
          denseOutputZeroPointType: 2,
          denseOutputZeroPoint: 85,
          fallbackReasons: [],
        });
      });

      it('uses nearest-even asymmetric rounding and the zero-range symmetric fallback for dense parameters', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections.forEach((connectionEntry) => {
          connectionEntry.weight = 0;
        });
        getHiddenNodes(network)[0].bias = 0;
        (network.nodes.at(-1) as Node).bias = 0;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1.1, max: 1 },
                  outputRange: { min: -0.61, max: 0.5 },
                },
              ],
              activationSymmetry: 'asymmetric',
              weightSymmetry: 'symmetric',
            },
          },
        });

        // Assert
        expect({
          denseInputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseInputZeroPoint_l1',
          )?.int32_data?.[0],
          denseOutputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseOutputZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightScale: getInitializer(
            onnxModel,
            'QuantDenseWeightScale_l1',
          )?.float_data?.[0],
          denseWeightZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseWeightZeroPoint_l1',
          )?.int32_data?.[0],
        }).toEqual({
          denseInputZeroPoint: 134,
          denseOutputZeroPoint: 140,
          denseWeightScale: 1,
          denseWeightZeroPoint: 0,
        });
      });

      it('uses even-tie rounding and symmetric uint8 zero points when the packet requests them explicitly', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections.forEach((connectionEntry) => {
          connectionEntry.weight = 0;
        });

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -126.5, max: 128.5 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
              activationSymmetry: 'asymmetric',
              weightSymmetry: 'symmetric',
            },
            weightEncoding: 'uint8',
          },
        });

        // Assert
        expect({
          denseInputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseInputZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseWeightZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightScale: getInitializer(
            onnxModel,
            'QuantDenseWeightScale_l1',
          )?.float_data?.[0],
        }).toEqual({
          denseInputZeroPoint: 126,
          denseWeightZeroPoint: 128,
          denseWeightScale: 1,
        });
      });

      it('supports asymmetric int8 activations alongside nonzero symmetric uint8 weights', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 0.75;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -0.2, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
              activationSymmetry: 'asymmetric',
              weightSymmetry: 'symmetric',
            },
            activationEncoding: 'int8',
            weightEncoding: 'uint8',
          },
        });

        // Assert
        expect({
          denseInputZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseInputZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightZeroPoint: getInitializer(
            onnxModel,
            'QuantDenseWeightZeroPoint_l1',
          )?.int32_data?.[0],
          denseWeightScale: getInitializer(
            onnxModel,
            'QuantDenseWeightScale_l1',
          )?.float_data?.[0],
        }).toEqual({
          denseInputZeroPoint: -86,
          denseWeightZeroPoint: 128,
          denseWeightScale: 0.003937007874015748,
        });
      });

      it('lowers one biased dense target into a qlinear affine path with an explicit bias bridge and preserved activation', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 0.75;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          qlinearMatMulCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'QLinearMatMul',
          ).length,
          quantizeLinearCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'QuantizeLinear',
          ).length,
          dequantizeLinearCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'DequantizeLinear',
          ).length,
          biasAddCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'Add' && nodeEntry.name === 'bias_add_l1',
          ).length,
          reluCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'Relu' && nodeEntry.name === 'act_l1',
          ).length,
          gemmCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'Gemm',
          ).length,
          quantizedWeightDataType: getInitializer(
            onnxModel,
            'QuantDenseWeight_l1',
          )?.data_type,
          hiddenBiasInitializer: getInitializer(onnxModel, 'B0')?.float_data,
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          effectiveQuantizationMode: 'static-8bit',
          qlinearMatMulCount: 1,
          quantizeLinearCount: 1,
          dequantizeLinearCount: 1,
          biasAddCount: 1,
          reluCount: 1,
          gemmCount: 1,
          quantizedWeightDataType: 3,
          hiddenBiasInitializer: [0.125],
          fallbackReasons: [],
        });
      });

      it('keeps the biased qlinear dense layer output within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          1e-2,
        );
      });

      it('keeps one-output static-8bit dense exports byte-stable across repeated runs', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const exportOptions: Parameters<typeof exportToONNX>[1] = {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        };
        const firstExport = exportToONNX(network, exportOptions);

        // Act
        const secondExport = exportToONNX(network, exportOptions);

        // Assert
        expect(JSON.stringify(secondExport)).toBe(JSON.stringify(firstExport));
      });

      it('keeps multi-output dense quantization requests on float32 with an explicit fallback reason', () => {
        // Arrange
        const network = Network.createMLP(2, [], 2);
        const outputNodes = network.nodes.filter(
          (nodeEntry) => nodeEntry.type === 'output',
        );
        outputNodes.forEach((outputNode) => {
          outputNode.squash = methods.Activation.sigmoid;
        });

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          hasDenseInputScale:
            getInitializer(onnxModel, 'QuantDenseInputScale_l1') !== undefined,
          hasQLinearMatMul: toNodes(onnxModel).some(
            (nodeEntry) => nodeEntry.op_type === 'QLinearMatMul',
          ),
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          effectiveQuantizationMode: 'none',
          hasDenseInputScale: false,
          hasQLinearMatMul: false,
          fallbackReasons: [
            'static_8bit_not_implemented',
            'multi_output_dense_boundary_requires_float32',
          ],
        });
      });

      it('lowers the supported one-output dense target while keeping a wider dense target on float32 in the same request', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: 0, max: 0.5 },
                  outputRange: { min: 0, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
          qlinearNodes: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.op_type === 'QLinearMatMul')
            .map((nodeEntry) => nodeEntry.name),
          floatDenseNodes: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.op_type === 'Gemm')
            .map((nodeEntry) => nodeEntry.name),
          hasFirstDenseInputScale:
            getInitializer(onnxModel, 'QuantDenseInputScale_l1') !== undefined,
          hasSecondDenseInputScale:
            getInitializer(onnxModel, 'QuantDenseInputScale_l2') !== undefined,
        }).toEqual({
          effectiveQuantizationMode: 'static-8bit',
          fallbackReasons: ['multi_output_dense_boundary_requires_float32'],
          qlinearNodes: ['qlinear_matmul_l2'],
          floatDenseNodes: ['gemm_l1'],
          hasFirstDenseInputScale: false,
          hasSecondDenseInputScale: true,
        });
      });

      it('keeps a mixed-target request within a small tolerance when the supported one-output dense layer lowers and the wider dense layer stays float32', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const sampleInput = [0.25, -0.75];
        const hiddenNodes = getHiddenNodes(network);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 0.75;
        network.connections[3].weight = 0.5;
        network.connections[4].weight = 1.25;
        network.connections[5].weight = -0.5;
        hiddenNodes[0]!.bias = 0.125;
        hiddenNodes[1]!.bias = -0.25;
        hiddenNodes[0]!.squash = methods.Activation.relu;
        hiddenNodes[1]!.squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = 0.0625;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;
        const inputNodes = network.nodes.filter(
          (nodeEntry) => nodeEntry.type === 'input',
        );
        const inputValuesByNode = new Map(
          inputNodes.map((inputNode, inputIndex) => [
            inputNode,
            sampleInput[inputIndex]!,
          ]),
        );
        const hiddenLayerOutputs = hiddenNodes.map((hiddenNode) =>
          hiddenNode.squash(
            hiddenNode.connections.in.reduce(
              (weightedSum, inboundConnection) =>
                weightedSum +
                (inputValuesByNode.get(inboundConnection.from) ?? 0) *
                  inboundConnection.weight,
              hiddenNode.bias,
            ),
          ),
        );

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: 0, max: 0.5 },
                  outputRange: { min: 0, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          2,
          hiddenLayerOutputs,
        );

        // Assert
        expect(Math.abs(quantizedOutput - baselineOutput)).toBeLessThanOrEqual(
          1e-2,
        );
      });

      it('preserves Softplus on the lowered one-output dense target while a wider dense target stays float32 in the same request', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const hiddenNodes = getHiddenNodes(network);
        hiddenNodes[0]!.squash = methods.Activation.relu;
        hiddenNodes[1]!.squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).squash = methods.Activation.softplus;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: 0, max: 0.5 },
                  outputRange: { min: 0, max: 1.5 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
          qlinearNodes: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.op_type === 'QLinearMatMul')
            .map((nodeEntry) => nodeEntry.name),
          floatDenseNodes: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.op_type === 'Gemm')
            .map((nodeEntry) => nodeEntry.name),
          loweredActivationNodes: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.name === 'act_l2')
            .map((nodeEntry) => nodeEntry.op_type),
          hasFirstDenseInputScale:
            getInitializer(onnxModel, 'QuantDenseInputScale_l1') !== undefined,
          hasSecondDenseInputScale:
            getInitializer(onnxModel, 'QuantDenseInputScale_l2') !== undefined,
        }).toEqual({
          effectiveQuantizationMode: 'static-8bit',
          fallbackReasons: ['multi_output_dense_boundary_requires_float32'],
          qlinearNodes: ['qlinear_matmul_l2'],
          floatDenseNodes: ['gemm_l1'],
          loweredActivationNodes: ['Softplus'],
          hasFirstDenseInputScale: false,
          hasSecondDenseInputScale: true,
        });
      });

      it('keeps a Softplus mixed-target request within a small tolerance when the lowered one-output dense target preserves its unary activation', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const sampleInput = [0.25, -0.75];
        const hiddenNodes = getHiddenNodes(network);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 0.75;
        network.connections[3].weight = 0.5;
        network.connections[4].weight = 1.25;
        network.connections[5].weight = -0.5;
        hiddenNodes[0]!.bias = 0.125;
        hiddenNodes[1]!.bias = -0.25;
        hiddenNodes[0]!.squash = methods.Activation.relu;
        hiddenNodes[1]!.squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = 0.0625;
        (network.nodes.at(-1) as Node).squash = methods.Activation.softplus;
        const baselineOutput = network.activate(sampleInput)[0] as number;
        const inputNodes = network.nodes.filter(
          (nodeEntry) => nodeEntry.type === 'input',
        );
        const inputValuesByNode = new Map(
          inputNodes.map((inputNode, inputIndex) => [
            inputNode,
            sampleInput[inputIndex]!,
          ]),
        );
        const hiddenLayerOutputs = hiddenNodes.map((hiddenNode) =>
          hiddenNode.squash(
            hiddenNode.connections.in.reduce(
              (weightedSum, inboundConnection) =>
                weightedSum +
                (inputValuesByNode.get(inboundConnection.from) ?? 0) *
                  inboundConnection.weight,
              hiddenNode.bias,
            ),
          ),
        );

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: 0, max: 0.5 },
                  outputRange: { min: 0, max: 1.5 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          2,
          hiddenLayerOutputs,
        );

        // Assert
        expect(Math.abs(quantizedOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      describe.each(additionalMixedTargetPreservedUnaryActivationCases)(
        'given a $activationLabel mixed-target preserved-unary request',
        ({
          activationFunction,
          activationLabel,
          activationNodeType,
          opset,
          outputRange,
          sampleInput,
          tolerance,
        }) => {
          it('preserves the lowered one-output activation while the wider dense target stays float32', () => {
            // Arrange
            const { network } = createMixedTargetPreservedUnaryFixture(
              activationFunction,
              sampleInput,
            );

            // Act
            const onnxModel = exportToONNX(
              network,
              createMixedTargetQuantizationOptions(outputRange, opset),
            );

            // Assert
            expect({
              activationLabel,
              effectiveQuantizationMode: getMetadataValue(
                onnxModel,
                'effective_quantization_mode',
              ),
              fallbackReasons: getParsedMetadataArray(
                onnxModel,
                'quantization_fallback_reasons',
              ),
              qlinearNodes: toNodes(onnxModel)
                .filter((nodeEntry) => nodeEntry.op_type === 'QLinearMatMul')
                .map((nodeEntry) => nodeEntry.name),
              floatDenseNodes: toNodes(onnxModel)
                .filter((nodeEntry) => nodeEntry.op_type === 'Gemm')
                .map((nodeEntry) => nodeEntry.name),
              loweredActivationNodes: toNodes(onnxModel)
                .filter((nodeEntry) => nodeEntry.name === 'act_l2')
                .map((nodeEntry) => nodeEntry.op_type),
              hasFirstDenseInputScale:
                getInitializer(onnxModel, 'QuantDenseInputScale_l1') !== undefined,
              hasSecondDenseInputScale:
                getInitializer(onnxModel, 'QuantDenseInputScale_l2') !== undefined,
            }).toEqual({
              activationLabel,
              effectiveQuantizationMode: 'static-8bit',
              fallbackReasons: ['multi_output_dense_boundary_requires_float32'],
              qlinearNodes: ['qlinear_matmul_l2'],
              floatDenseNodes: ['gemm_l1'],
              loweredActivationNodes: [activationNodeType],
              hasFirstDenseInputScale: false,
              hasSecondDenseInputScale: true,
            });
          });

          it('keeps the lowered one-output activation within tolerance while the wider dense target stays float32', () => {
            // Arrange
            const { baselineOutput, hiddenLayerOutputs, network } =
              createMixedTargetPreservedUnaryFixture(
                activationFunction,
                sampleInput,
              );

            // Act
            const onnxModel = exportToONNX(
              network,
              createMixedTargetQuantizationOptions(outputRange, opset),
            );
            const quantizedOutput = evaluateOneOutputQLinearDenseLayerOutput(
              onnxModel,
              2,
              hiddenLayerOutputs,
            );

            // Assert
            expect(Math.abs(quantizedOutput - baselineOutput)).toBeLessThanOrEqual(
              tolerance,
            );
          });
        },
      );

      it('keeps consecutive targeted qlinear dense layers within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1.5;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.relu;
        (network.nodes.at(-1) as Node).bias = -0.0625;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 0.75 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: 0, max: 0.75 },
                  outputRange: { min: -0.25, max: 0.75 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const firstQuantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );
        const finalQuantizedOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          2,
          [firstQuantizedLayerOutput],
        );

        // Assert
        expect(Math.abs(finalQuantizedOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Tanh-to-Sigmoid qlinear dense chain within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1.5;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.tanh;
        (network.nodes.at(-1) as Node).bias = -0.0625;
        (network.nodes.at(-1) as Node).squash = methods.Activation.sigmoid;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -1, max: 1 },
                },
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const firstQuantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );
        const finalQuantizedOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          2,
          [firstQuantizedLayerOutput],
        );

        // Assert
        expect(Math.abs(finalQuantizedOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Softplus qlinear dense layer within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.softplus;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: 0, max: 1.5 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Softsign qlinear dense layer within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.softsign;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -1, max: 1 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Gelu qlinear dense layer within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [-0.5, 0.25];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.gelu;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          opset: 20,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.5 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Selu qlinear dense layer within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [-0.5, 0.25];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.selu;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -1, max: 1 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps a Mish qlinear dense layer within a small tolerance of the baseline dense path', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [-0.5, 0.25];
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.mish;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          opset: 18,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.5 },
                },
              ],
            },
            representation: 'qlinear',
          },
        });
        const quantizedLayerOutput = evaluateOneOutputQLinearDenseLayerOutput(
          onnxModel,
          1,
          sampleInput,
        );

        // Assert
        expect(Math.abs(quantizedLayerOutput - baselineOutput)).toBeLessThanOrEqual(
          2e-2,
        );
      });

      it('keeps below-opset Mish qlinear dense requests on the static-8bit path without quantization fallback metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        network.connections[0].weight = 0.5;
        network.connections[1].weight = -0.25;
        network.connections[2].weight = 1;
        getHiddenNodes(network)[0].bias = 0.125;
        getHiddenNodes(network)[0].squash = methods.Activation.mish;
        (network.nodes.at(-1) as Node).bias = 0;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;

        // Act / Assert
        suppressConsoleWarn(() => {
          const onnxModel = exportToONNX(network, {
            includeMetadata: true,
            opset: 17,
            quantization: {
              mode: 'static-8bit',
              targets: ['dense'],
              calibration: {
                source: 'external',
                layerTargets: [
                  {
                    target: 'dense',
                    layerIndex: 1,
                    inputRange: { min: -1, max: 1 },
                    outputRange: { min: -0.5, max: 0.5 },
                  },
                ],
              },
              representation: 'qlinear',
            },
          });

          expect({
            effectiveQuantizationMode: getMetadataValue(
              onnxModel,
              'effective_quantization_mode',
            ),
            qlinearMatMulCount: toNodes(onnxModel).filter(
              (nodeEntry) => nodeEntry.op_type === 'QLinearMatMul',
            ).length,
            hiddenActivationNodes: toNodes(onnxModel)
              .filter((nodeEntry) => nodeEntry.name === 'act_l1')
              .map((nodeEntry) => nodeEntry.op_type),
            fallbackReasons: getParsedMetadataArray(
              onnxModel,
              'quantization_fallback_reasons',
            ),
          }).toEqual({
            effectiveQuantizationMode: 'static-8bit',
            qlinearMatMulCount: 1,
            hiddenActivationNodes: [],
            fallbackReasons: [],
          });
        });
      });

      it('emits the existing activation fallback warning for below-opset Mish qlinear dense requests', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        getHiddenNodes(network)[0].squash = methods.Activation.mish;
        (network.nodes.at(-1) as Node).squash = methods.Activation.identity;
        const originalWarn = console.warn;
        const warnSpy = jest.fn();
        console.warn = warnSpy;

        try {
          // Act
          exportToONNX(network, {
            opset: 17,
            quantization: {
              mode: 'static-8bit',
              targets: ['dense'],
              calibration: {
                source: 'external',
                layerTargets: [
                  {
                    target: 'dense',
                    layerIndex: 1,
                    inputRange: { min: -1, max: 1 },
                    outputRange: { min: -0.5, max: 0.5 },
                  },
                ],
              },
              representation: 'qlinear',
            },
          });

          // Assert
          expect(warnSpy).toHaveBeenCalledTimes(1);
        } finally {
          console.warn = originalWarn;
        }
      });

      it('emits DynamicQuantizeLinear guidance for the supported dense baseline', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'DynamicQuantizeLinear',
          },
        });

        // Assert
        expect({
          requestedQuantizationMode: getMetadataValue(
            onnxModel,
            'requested_quantization_mode',
          ),
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          dynamicQuantizeNodeCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'DynamicQuantizeLinear',
          ).length,
          dynamicDequantizeNodeCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.name === 'dynamic_dequantize_input_l1',
          ).length,
          loweredDenseInputName: toNodes(onnxModel).find(
            (nodeEntry) => nodeEntry.name === 'gemm_l1',
          )?.input[0],
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          requestedQuantizationMode: 'dynamic-uint8',
          effectiveQuantizationMode: 'dynamic-uint8',
          dynamicQuantizeNodeCount: 2,
          dynamicDequantizeNodeCount: 1,
          loweredDenseInputName: 'DynamicDenseInputFloat_l1',
          fallbackReasons: [],
        });
      });

      it('keeps DynamicQuantizeLinear dense guidance exports byte-stable across repeated runs', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const exportOptions: Parameters<typeof exportToONNX>[1] = {
          includeMetadata: true,
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'DynamicQuantizeLinear',
          },
        };
        const firstExport = exportToONNX(network, exportOptions);

        // Act
        const secondExport = exportToONNX(network, exportOptions);

        // Assert
        expect(JSON.stringify(secondExport)).toBe(JSON.stringify(firstExport));
      });

      it('lands metadata-only dynamic guidance for the supported dense baseline', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'metadata-only',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          dynamicQuantizeNodeCount: toNodes(onnxModel).filter(
            (nodeEntry) => nodeEntry.op_type === 'DynamicQuantizeLinear',
          ).length,
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          effectiveQuantizationMode: 'dynamic-uint8',
          dynamicQuantizeNodeCount: 0,
          fallbackReasons: [],
        });
      });

      it('records a recurrent boundary fallback reason for quantization requests', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const recurrentHiddenNode = getHiddenNodes(network)[0];
        recurrentHiddenNode.connect(recurrentHiddenNode, 0.25);
        Network.rebuildConnections(network);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowRecurrent: true,
          recurrentSingleStep: true,
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'metadata-only',
          },
        });

        // Assert
        expect(getParsedMetadataArray(onnxModel, 'quantization_fallback_reasons')).toEqual([
          'recurrent_boundary_requires_float32',
        ]);
      });

      it('records an advanced-graph fallback reason for dynamic concat requests', () => {
        // Arrange
        const network = createConcatMergeExportNetwork();

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          concatMappings: [
            {
              sourceLayerIndex: 0,
              targetLayerIndex: 2,
              inputOrder: 'previous_then_source',
            },
          ],
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'metadata-only',
          },
        });

        // Assert
        expect({
          effectiveQuantizationMode: getMetadataValue(
            onnxModel,
            'effective_quantization_mode',
          ),
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          effectiveQuantizationMode: 'none',
          fallbackReasons: ['advanced_graph_boundary_requires_float32'],
        });
      });

      it('does not emit static quantization parameter tensors across recurrent boundaries', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const recurrentHiddenNode = getHiddenNodes(network)[0];
        recurrentHiddenNode.connect(recurrentHiddenNode, 0.25);
        Network.rebuildConnections(network);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowRecurrent: true,
          recurrentSingleStep: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
          },
        });

        // Assert
        expect({
          hasDenseInputScale: getInitializer(
            onnxModel,
            'QuantDenseInputScale_l1',
          ) !== undefined,
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          hasDenseInputScale: false,
          fallbackReasons: [
            'static_8bit_not_implemented',
            'recurrent_boundary_requires_float32',
          ],
        });
      });

      it('records an advanced-graph fallback reason for concat quantization requests', () => {
        // Arrange
        const network = createConcatMergeExportNetwork();

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          concatMappings: [
            {
              sourceLayerIndex: 0,
              targetLayerIndex: 2,
              inputOrder: 'previous_then_source',
            },
          ],
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 2,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
          },
        });

        // Assert
        expect(getParsedMetadataArray(onnxModel, 'quantization_fallback_reasons')).toEqual([
          'static_8bit_not_implemented',
          'advanced_graph_boundary_requires_float32',
        ]);
      });

      it('records relaxed mixed-activation and partial-connectivity fallback reasons for quantization requests', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowMixedActivations: true,
          allowPartialConnectivity: true,
          quantization: {
            mode: 'dynamic-uint8',
            target: 'dense',
            representation: 'metadata-only',
          },
        });

        // Assert
        expect(getParsedMetadataArray(onnxModel, 'quantization_fallback_reasons')).toEqual([
          'mixed_activation_boundary_requires_float32',
          'partial_connectivity_boundary_requires_float32',
        ]);
      });

      it('does not emit static quantization parameter tensors when relaxed mixed-activation export is requested', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowMixedActivations: true,
          quantization: {
            mode: 'static-8bit',
            targets: ['dense'],
            calibration: {
              source: 'external',
              layerTargets: [
                {
                  target: 'dense',
                  layerIndex: 1,
                  inputRange: { min: -1, max: 1 },
                  outputRange: { min: -0.5, max: 0.75 },
                },
              ],
            },
          },
        });

        // Assert
        expect({
          hasDenseInputScale: getInitializer(
            onnxModel,
            'QuantDenseInputScale_l1',
          ) !== undefined,
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'quantization_fallback_reasons',
          ),
        }).toEqual({
          hasDenseInputScale: false,
          fallbackReasons: [
            'static_8bit_not_implemented',
            'mixed_activation_boundary_requires_float32',
          ],
        });
      });
    });

    describe('given Phase 7 storage-fp16 export is requested', () => {
      it('rewrites dense initializers to float16 payloads and inserts float32 cast bridges', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          precision: {
            mode: 'storage-fp16',
          },
        });

        // Assert
        expect({
          weightDataType: getInitializer(onnxModel, 'W0')?.data_type,
          weightFloatDataLength: getInitializer(onnxModel, 'W0')?.float_data?.length,
          weightPackedLength: getInitializer(onnxModel, 'W0')?.int32_data?.length,
          biasDataType: getInitializer(onnxModel, 'B0')?.data_type,
          biasFloatDataLength: getInitializer(onnxModel, 'B0')?.float_data?.length,
          biasPackedLength: getInitializer(onnxModel, 'B0')?.int32_data?.length,
          castInputs: toNodes(onnxModel)
            .filter((nodeEntry) => nodeEntry.op_type === 'Cast')
            .map((nodeEntry) => nodeEntry.input[0]),
          firstGemmInputs: toNodes(onnxModel).find(
            (nodeEntry) => nodeEntry.op_type === 'Gemm',
          )?.input.slice(1),
          effectivePrecisionMode: getMetadataValue(
            onnxModel,
            'effective_precision_mode',
          ),
        }).toEqual({
          weightDataType: 10,
          weightFloatDataLength: 0,
          weightPackedLength: 2,
          biasDataType: 10,
          biasFloatDataLength: 0,
          biasPackedLength: 1,
          castInputs: ['W0', 'B0', 'W1', 'B1'],
          firstGemmInputs: ['W0_fp32', 'B0_fp32'],
          effectivePrecisionMode: 'storage-fp16',
        });
      });

      it('keeps storage-fp16 exports byte-stable across repeated runs', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const exportOptions: Parameters<typeof exportToONNX>[1] = {
          includeMetadata: true,
          precision: {
            mode: 'storage-fp16',
          },
        };
        const firstExport = exportToONNX(network, exportOptions);

        // Act
        const secondExport = exportToONNX(network, exportOptions);

        // Assert
        expect(JSON.stringify(secondExport)).toBe(JSON.stringify(firstExport));
      });

      it('keeps recurrent storage-fp16 requests on float32 with an explicit fallback reason', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const recurrentHiddenNode = getHiddenNodes(network)[0];
        recurrentHiddenNode.connect(recurrentHiddenNode, 0.25);
        Network.rebuildConnections(network);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowRecurrent: true,
          recurrentSingleStep: true,
          precision: {
            mode: 'storage-fp16',
          },
        });

        // Assert
        expect({
          weightDataType: getInitializer(onnxModel, 'W0')?.data_type,
          effectivePrecisionMode: getMetadataValue(
            onnxModel,
            'effective_precision_mode',
          ),
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'precision_fallback_reasons',
          ),
        }).toEqual({
          weightDataType: 1,
          effectivePrecisionMode: 'float32',
          fallbackReasons: ['recurrent_boundary_requires_float32'],
        });
      });

      it('keeps exports byte-stable when the precision packet is absent', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const baselineExport = exportToONNX(network);

        // Act
        const repeatedExport = exportToONNX(network, {});

        // Assert
        expect(JSON.stringify(repeatedExport)).toBe(JSON.stringify(baselineExport));
      });

      it('keeps same-family dense roundtrip outputs within a small tolerance', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const sampleInput = [0.25, -0.75];
        const baselineOutput = network.activate(sampleInput)[0] as number;

        // Act
        const restoredNetwork = importFromONNX(
          exportToONNX(network, {
            precision: {
              mode: 'storage-fp16',
            },
          }),
        );
        const restoredOutput = restoredNetwork.activate(sampleInput)[0] as number;

        // Assert
        expect(Math.abs(restoredOutput - baselineOutput) < 1e-3).toBe(true);
      });

      it('records relaxed mixed-activation and partial-connectivity precision fallback reasons', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          allowMixedActivations: true,
          allowPartialConnectivity: true,
          precision: {
            mode: 'storage-fp16',
          },
        });

        // Assert
        expect(getParsedMetadataArray(onnxModel, 'precision_fallback_reasons')).toEqual([
          'mixed_activation_boundary_requires_float32',
          'partial_connectivity_boundary_requires_float32',
        ]);
      });

      it('keeps concat storage-fp16 requests on float32 with an advanced-graph fallback reason', () => {
        // Arrange
        const network = createConcatMergeExportNetwork();

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          concatMappings: [
            {
              sourceLayerIndex: 0,
              targetLayerIndex: 2,
            },
          ],
          precision: {
            mode: 'storage-fp16',
          },
        } as unknown as Parameters<typeof exportToONNX>[1]);

        // Assert
        expect({
          fallbackReasons: getParsedMetadataArray(
            onnxModel,
            'precision_fallback_reasons',
          ),
          effectivePrecisionMode: getMetadataValue(
            onnxModel,
            'effective_precision_mode',
          ),
        }).toEqual({
          fallbackReasons: ['advanced_graph_boundary_requires_float32'],
          effectivePrecisionMode: 'float32',
        });
      });

      it('records float32 as both requested and effective precision when the packet omits storage-fp16', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = exportToONNX(network, {
          includeMetadata: true,
          precision: {},
        });

        // Assert
        expect({
          requestedPrecisionMode: getMetadataValue(
            onnxModel,
            'requested_precision_mode',
          ),
          effectivePrecisionMode: getMetadataValue(
            onnxModel,
            'effective_precision_mode',
          ),
          precisionFallbackReasons: getMetadataValue(
            onnxModel,
            'precision_fallback_reasons',
          ),
        }).toEqual({
          requestedPrecisionMode: 'float32',
          effectivePrecisionMode: 'float32',
          precisionFallbackReasons: undefined,
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

      describe('when the activation is mish below opset 18', () => {
        it('falls back to an implicit identity activation', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = methods.Activation.mish;

          // Act / Assert
          suppressConsoleWarn(() => {
            const onnxModel = exportToONNX(network, { opset: 17 });
            expect(getFirstActivationNode(onnxModel)).toBeUndefined();
          });
        });

        it('emits a warning about the opset-gated fallback', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = methods.Activation.mish;
          const originalWarn = console.warn;
          const warnSpy = jest.fn();
          console.warn = warnSpy;

          try {
            // Act
            exportToONNX(network, { opset: 17 });

            // Assert
            expect(warnSpy).toHaveBeenCalledTimes(1);
          } finally {
            console.warn = originalWarn;
          }
        });
      });

      describe('when the activation is not recognized', () => {
        it('falls back to an implicit identity activation', () => {
          // Arrange
          const network = new Network(1, 1, {});
          network.nodes[1].squash = (value: number) => value;

          // Act / Assert
          suppressConsoleWarn(() => {
            const onnxModel = exportToONNX(network);
            expect(getFirstActivationNode(onnxModel)).toBeUndefined();
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

      describe('when the network includes a feed-forward skip edge', () => {
        it('records deterministic advanced-graph cross-layer metadata', () => {
          // Arrange
          const network = createFeedForwardSkipConnectionNetwork();

          // Act
          const advancedGraphModel = exportToONNX(network, {
            includeMetadata: true,
          });

          // Assert
          expect(
            getMetadataValue(
              advancedGraphModel,
              'advanced_graph_cross_layer_connections',
            ),
          ).toBe(
            JSON.stringify([
              {
                sourceNodeIndex: 0,
                sourceLayerIndex: 0,
                targetNodeIndex: 4,
                targetLayerIndex: 2,
                branchTensorName: 'Branch_l0_to_l2_from_n0_to_n4',
              },
            ]),
          );
        });

        it('emits a one-hop residual Add node and merge metadata when the skipped branch targets a width-compatible layer', () => {
          // Arrange
          const network = createResidualAddExportNetwork();

          // Act
          const advancedGraphModel = exportToONNX(network, {
            includeMetadata: true,
          });

          // Assert
          expect({
            addNodeNames: toNodes(advancedGraphModel)
              .filter((nodeEntry) => nodeEntry.op_type === 'Add')
              .map((nodeEntry) => nodeEntry.name),
            residualMetadata: getMetadataValue(
              advancedGraphModel,
              'advanced_graph_residual_adds',
            ),
          }).toEqual({
            addNodeNames: ['residual_add_l2'],
            residualMetadata: JSON.stringify([
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                branchTensorName: 'ResidualBranch_l0_to_l2',
                mergeNodeName: 'residual_add_l2',
                mergeOutputName: 'ResidualAdd_2',
              },
            ]),
          });
        });

        it('emits an explicit concat merge node and metadata when a source-owned concat mapping is provided', () => {
          // Arrange
          const network = createConcatMergeExportNetwork();

          // Act
          const advancedGraphModel = exportToONNX(network, {
            includeMetadata: true,
            concatMappings: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
              },
            ],
          } as unknown as Parameters<typeof exportToONNX>[1]);

          // Assert
          expect({
            concatNodeNames: toNodes(advancedGraphModel)
              .filter((nodeEntry) => nodeEntry.op_type === 'Concat')
              .map((nodeEntry) => nodeEntry.name),
            concatMetadata: getMetadataValue(
              advancedGraphModel,
              'advanced_graph_concat_merges',
            ),
          }).toEqual({
            concatNodeNames: ['concat_merge_l0_to_l2'],
            concatMetadata: JSON.stringify([
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                concatNodeName: 'concat_merge_l0_to_l2',
                concatOutputName: 'ConcatMerge_0_to_2',
                inputOrder: 'previous_then_source',
              },
            ]),
          });
        });

        it('emits a fixed-width self-attention shadow block and metadata when an explicit attention mapping is provided', () => {
          // Arrange
          const network = createAttentionShadowExportNetwork();
          const identityProjectionWeights = createIdentityProjectionWeights(4);
          const attentionMapping: AttentionMappingProbe = {
            layerIndex: 1,
            sequenceLength: 2,
            modelWidth: 4,
            heads: 2,
            queryWeights: identityProjectionWeights,
            keyWeights: identityProjectionWeights,
            valueWeights: identityProjectionWeights,
            queryBias: [0, 0, 0, 0],
            keyBias: [0, 0, 0, 0],
            valueBias: [0, 0, 0, 0],
          };

          // Act
          const advancedGraphModel = exportToONNX(network, {
            includeMetadata: true,
            attentionMappings: [attentionMapping],
          } as unknown as Parameters<typeof exportToONNX>[1]);

          // Assert
          expect({
            softmaxNodeNames: toNodes(advancedGraphModel)
              .filter((nodeEntry) => nodeEntry.op_type === 'Softmax')
              .map((nodeEntry) => nodeEntry.name),
            attentionMetadata: getMetadataValue(
              advancedGraphModel,
              'advanced_graph_attention_blocks',
            ),
          }).toEqual({
            softmaxNodeNames: ['attention_l1_softmax'],
            attentionMetadata: JSON.stringify([
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 1,
                sequenceLength: 2,
                modelWidth: 4,
                heads: 2,
                shadowOutputName: 'AttentionShadow_1',
              },
            ]),
          });
        });

        it('keeps multi-hop skip edges on the audit-only fallback path', () => {
          // Arrange
          const network = createMultiHopSkipConnectionNetwork();

          // Act
          const advancedGraphModel = exportToONNX(network, {
            includeMetadata: true,
          });

          // Assert
          expect(
            toNodes(advancedGraphModel).some(
              (nodeEntry) => nodeEntry.op_type === 'Add',
            ),
          ).toBe(false);
        });
      });

      describe('when advanced-graph metadata is attached directly', () => {
        it('initializes metadata_props and falls back to sentinel node indices when export indices are absent', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenNode = new Node('hidden');
          const outputNode = new Node('output');
          const network = {
            nodes: [inputNode, hiddenNode, outputNode],
          } as Network;
          const model = createMinimalOnnxModel(undefined);

          inputNode.connect(hiddenNode, 0.2);
          hiddenNode.connect(outputNode, 0.3);
          inputNode.connect(outputNode, 0.75);
          network.nodes.forEach((nodeEntry) => {
            nodeEntry.index = undefined;
          });

          // Act
          appendAdvancedGraphMetadata(
            model,
            network,
            [[inputNode], [hiddenNode], [outputNode]],
            true,
          );

          // Assert
          expect(model.metadata_props).toEqual([
            {
              key: 'advanced_graph_cross_layer_connections',
              value: JSON.stringify([
                {
                  sourceNodeIndex: -1,
                  sourceLayerIndex: 0,
                  targetNodeIndex: -1,
                  targetLayerIndex: 2,
                  branchTensorName: 'Branch_l0_to_l2_from_n-1_to_n-1',
                },
              ]),
            },
          ]);
        });

        it('returns null for one-hop residual detection when multiple non-adjacent source layers feed the same target layer', () => {
          // Arrange
          const inputNode = new Node('input');
          const firstHiddenNode = new Node('hidden');
          const secondHiddenNode = new Node('hidden');
          const outputNode = new Node('output');

          inputNode.connect(firstHiddenNode, 0.2);
          firstHiddenNode.connect(secondHiddenNode, 0.3);
          secondHiddenNode.connect(outputNode, 0.4);
          inputNode.connect(outputNode, 0.5);
          firstHiddenNode.connect(outputNode, 0.6);

          // Assert
          expect(
            resolveOneHopResidualSourceLayerIndex(
              [outputNode],
              [[inputNode], [firstHiddenNode], [secondHiddenNode], [outputNode]],
              3,
            ),
          ).toBeNull();
        });

        it('appends residual-add metadata to an existing metadata entry', () => {
          // Arrange
          const firstResidualAdd = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            branchTensorName: 'ResidualBranch_l0_to_l2',
            mergeNodeName: 'residual_add_l2',
            mergeOutputName: 'ResidualAdd_2',
          };
          const secondResidualAdd = {
            sourceLayerIndex: 1,
            targetLayerIndex: 3,
            branchTensorName: 'ResidualBranch_l1_to_l3',
            mergeNodeName: 'residual_add_l3',
            mergeOutputName: 'ResidualAdd_3',
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_residual_adds',
              value: JSON.stringify([firstResidualAdd]),
            },
          ]);

          // Act
          appendResidualAddMetadata(model, secondResidualAdd, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_residual_adds')).toBe(
            JSON.stringify([firstResidualAdd, secondResidualAdd]),
          );
        });

        it('replaces malformed residual-add metadata payloads with the new record', () => {
          // Arrange
          const residualAdd = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            branchTensorName: 'ResidualBranch_l0_to_l2',
            mergeNodeName: 'residual_add_l2',
            mergeOutputName: 'ResidualAdd_2',
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_residual_adds',
              value: 'not-json',
            },
          ]);

          // Act
          appendResidualAddMetadata(model, residualAdd, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_residual_adds')).toBe(
            JSON.stringify([residualAdd]),
          );
        });

        it('replaces valid non-array residual-add metadata payloads with the new record array', () => {
          // Arrange
          const residualAdd = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            branchTensorName: 'ResidualBranch_l0_to_l2',
            mergeNodeName: 'residual_add_l2',
            mergeOutputName: 'ResidualAdd_2',
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_residual_adds',
              value: JSON.stringify({ unexpected: true }),
            },
          ]);

          // Act
          appendResidualAddMetadata(model, residualAdd, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_residual_adds')).toBe(
            JSON.stringify([residualAdd]),
          );
        });

        it('leaves concat merge metadata unchanged when metadata emission is disabled', () => {
          // Arrange
          const concatMerge = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source' as const,
          };
          const model = createMinimalOnnxModel([{ key: 'existing', value: '1' }]);

          // Act
          appendConcatMergeMetadata(model, concatMerge, false);

          // Assert
          expect(model.metadata_props).toEqual([{ key: 'existing', value: '1' }]);
        });

        it('appends concat-merge metadata to an existing metadata entry', () => {
          // Arrange
          const firstConcatMerge = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source' as const,
          };
          const secondConcatMerge = {
            sourceLayerIndex: 1,
            targetLayerIndex: 3,
            concatNodeName: 'concat_merge_l1_to_l3',
            concatOutputName: 'ConcatMerge_1_to_3',
            inputOrder: 'previous_then_source' as const,
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_concat_merges',
              value: JSON.stringify([firstConcatMerge]),
            },
          ]);

          // Act
          appendConcatMergeMetadata(model, secondConcatMerge, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_concat_merges')).toBe(
            JSON.stringify([firstConcatMerge, secondConcatMerge]),
          );
        });

        it('replaces malformed concat-merge metadata payloads with the new record', () => {
          // Arrange
          const concatMerge = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source' as const,
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_concat_merges',
              value: 'not-json',
            },
          ]);

          // Act
          appendConcatMergeMetadata(model, concatMerge, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_concat_merges')).toBe(
            JSON.stringify([concatMerge]),
          );
        });

        it('replaces valid non-array concat-merge metadata payloads with the new record array', () => {
          // Arrange
          const concatMerge = {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source' as const,
          };
          const model = createMinimalOnnxModel([
            {
              key: 'advanced_graph_concat_merges',
              value: JSON.stringify({ unexpected: true }),
            },
          ]);

          // Act
          appendConcatMergeMetadata(model, concatMerge, true);

          // Assert
          expect(getMetadataValue(model, 'advanced_graph_concat_merges')).toBe(
            JSON.stringify([concatMerge]),
          );
        });

        it('ignores connections whose target node is outside the resolved layered ordering', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenNode = new Node('hidden');
          const outputNode = new Node('output');
          const detachedTargetNode = new Node('output');
          const network = {
            nodes: [inputNode, hiddenNode, outputNode],
          } as Network;
          const model = createMinimalOnnxModel([{ key: 'existing', value: '1' }]);

          inputNode.connect(hiddenNode, 0.2);
          hiddenNode.connect(outputNode, 0.3);
          inputNode.connect(detachedTargetNode, 0.75);

          // Act
          appendAdvancedGraphMetadata(
            model,
            network,
            [[inputNode], [hiddenNode], [outputNode]],
            true,
          );

          // Assert
          expect(model.metadata_props).toEqual([{ key: 'existing', value: '1' }]);
        });

        it('ignores connections whose source node is outside the resolved layered ordering', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenNode = new Node('hidden');
          const outputNode = new Node('output');
          const detachedSourceNode = new Node('input');
          const network = {
            nodes: [inputNode, hiddenNode, outputNode, detachedSourceNode],
          } as Network;
          const model = createMinimalOnnxModel([{ key: 'existing', value: '1' }]);

          inputNode.connect(hiddenNode, 0.2);
          hiddenNode.connect(outputNode, 0.3);
          detachedSourceNode.connect(outputNode, 0.75);

          // Act
          appendAdvancedGraphMetadata(
            model,
            network,
            [[inputNode], [hiddenNode], [outputNode]],
            true,
          );

          // Assert
          expect(model.metadata_props).toEqual([{ key: 'existing', value: '1' }]);
        });

        it('sorts multiple cross-layer records by target index when the source index matches', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenNode = new Node('hidden');
          const firstOutputNode = new Node('output');
          const secondOutputNode = new Node('output');
          const network = {
            nodes: [inputNode, hiddenNode, firstOutputNode, secondOutputNode],
          } as Network;
          const model = createMinimalOnnxModel(undefined);

          inputNode.index = 0;
          hiddenNode.index = 1;
          firstOutputNode.index = 2;
          secondOutputNode.index = 3;

          inputNode.connect(hiddenNode, 0.2);
          hiddenNode.connect(firstOutputNode, 0.3);
          hiddenNode.connect(secondOutputNode, 0.4);
          inputNode.connect(secondOutputNode, 0.75);
          inputNode.connect(firstOutputNode, 0.65);

          // Act
          appendAdvancedGraphMetadata(
            model,
            network,
            [[inputNode], [hiddenNode], [firstOutputNode, secondOutputNode]],
            true,
          );

          // Assert
          expect(model.metadata_props).toEqual([
            {
              key: 'advanced_graph_cross_layer_connections',
              value: JSON.stringify([
                {
                  sourceNodeIndex: 0,
                  sourceLayerIndex: 0,
                  targetNodeIndex: 2,
                  targetLayerIndex: 2,
                  branchTensorName: 'Branch_l0_to_l2_from_n0_to_n2',
                },
                {
                  sourceNodeIndex: 0,
                  sourceLayerIndex: 0,
                  targetNodeIndex: 3,
                  targetLayerIndex: 2,
                  branchTensorName: 'Branch_l0_to_l2_from_n0_to_n3',
                },
              ]),
            },
          ]);
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
