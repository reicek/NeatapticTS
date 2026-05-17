import onnxProto from 'onnx-proto';
import {
  ONNX_STANDARD_DOMAIN,
  ONNX_STANDARD_DOMAIN_ALIAS,
} from '../export/network.onnx.export-setup.utils';
import type { OnnxActivationOperation } from '../network.onnx.utils.types';
import type { OnnxModel, OnnxNode, OnnxTensor } from '../schema/network.onnx.schema.types';
import { ONNX_FLOAT_DATA_TYPE } from '../schema/network.onnx.schema.tensor-data.utils';
import type {
  DecodedExternalOnnxAttribute,
  DecodedExternalOnnxModel,
  DecodedExternalOnnxNode,
  DecodedExternalOnnxTensor,
  DecodedExternalOnnxValueInfo,
  OnnxDecodedBytes,
  OnnxDecodedLongLike,
  OnnxExternalDenseChain,
  OnnxExternalDenseLayer,
} from './network.onnx.import-external.types';
import { OnnxExternalImportError } from './network.onnx.import-external.types';

const FIRST_EXTERNAL_IMPORT_ONNX_OPSET = 18;
const GELU_EXTERNAL_IMPORT_MINIMUM_OPSET = 20;
const FLOAT32_TENSOR_ELEMENT_TYPE = ONNX_FLOAT_DATA_TYPE;
const GEMM_NODE_TYPE = 'Gemm';
const GELU_NODE_TYPE = 'Gelu';
const IDENTITY_ACTIVATION: OnnxActivationOperation = 'Identity';
const STANDARD_DOMAIN_NAMES = new Set([
  ONNX_STANDARD_DOMAIN,
  ONNX_STANDARD_DOMAIN_ALIAS,
]);
const ALLOWED_ACTIVATION_OPERATIONS = new Set<OnnxActivationOperation>([
  'Identity',
  'Relu',
  'Sigmoid',
  'Tanh',
  'Softplus',
  'Softsign',
  'Selu',
  'Mish',
  'Gelu',
]);
const ALLOWED_GEMM_ATTRIBUTE_NAMES = new Set(['alpha', 'beta', 'transA', 'transB']);
const CANONICAL_GEMM_ATTRIBUTE_VALUES = new Map<string, number>([
  ['alpha', 1],
  ['beta', 1],
  ['transA', 0],
  ['transB', 1],
]);
const GEMM_A_INPUT_INDEX = 0;
const GEMM_WEIGHT_INPUT_INDEX = 1;
const GEMM_BIAS_INPUT_INDEX = 2;
const GEMM_WEIGHT_DIMENSIONS = 2;
const BIAS_VECTOR_DIMENSIONS = 1;
const BIAS_MATRIX_DIMENSIONS = 2;
const RANK_TWO_TENSOR_DIMENSIONS = 2;
const FLOAT32_BYTES_PER_VALUE = 4;
const VALUE_INFO_FEATURE_DIMENSION_INDEX = 1;
const VALUE_INFO_BATCH_DIMENSION_INDEX = 0;
const FIRST_CANONICAL_LAYER_INDEX = 1;

/**
 * Normalize the first supported external binary ONNX subset into an importer-owned model.
 *
 * @param binaryModel Binary `ModelProto` payload.
 * @returns Canonical JSON-first model that the existing import flow can reconstruct.
 */
export function normalizeExternalBinaryOnnxModel(
  binaryModel: Uint8Array,
): OnnxModel {
  // Step 1: Decode and normalize one accepted external dense chain.
  const denseChain = normalizeExternalDenseChain(binaryModel);

  // Step 2: Fold that chain into the importer-owned JSON-first model shape.
  return buildImporterOwnedOnnxModel(denseChain);
}

/**
 * Normalize the first supported external binary ONNX subset into a canonical dense chain.
 *
 * @param binaryModel Binary `ModelProto` payload.
 * @returns Canonical dense-chain payload for importer reconstruction.
 */
export function normalizeExternalDenseChain(
  binaryModel: Uint8Array,
): OnnxExternalDenseChain {
  // Step 1: Decode the protobuf payload and verify the declared structure.
  const decodedModel = decodeVerifiedExternalModel(binaryModel);
  const decodedGraph = resolveDecodedGraph(decodedModel);
  const declaredOpset = resolveDeclaredOpset(decodedModel);

  // Step 2: Resolve public graph boundaries and initializer ownership.
  const initializerMap = buildInitializerMap(decodedGraph.initializer ?? []);
  const publicInput = resolvePublicInput(decodedGraph.input ?? [], initializerMap);
  const publicOutput = resolvePublicOutput(decodedGraph.output ?? []);
  const inputWidth = readRankTwoFeatureWidth(publicInput, 'model input');
  const outputWidth = readRankTwoFeatureWidth(publicOutput, 'model output');
  validateSupportingValueInfo(decodedGraph.valueInfo ?? []);

  // Step 3: Walk one unambiguous dense chain from the public input to the public output.
  const graphNodes = decodedGraph.node ?? [];
  const producerMap = buildProducerMap(graphNodes);
  const consumerMap = buildConsumerMap(graphNodes, initializerMap);
  const denseLayers = collectDenseLayers({
    graphNodes,
    initializerMap,
    producerMap,
    consumerMap,
    declaredOpset,
    publicInputName: publicInput.name ?? '',
    publicOutputName: publicOutput.name ?? '',
    inputWidth,
  });

  // Step 4: Fold the walked chain into the canonical importer payload.
  return {
    opsetVersion: declaredOpset,
    inputWidth,
    outputWidth,
    layers: denseLayers,
  };
}

function decodeVerifiedExternalModel(
  binaryModel: Uint8Array,
): DecodedExternalOnnxModel {
  try {
    const decodedModel = onnxProto.onnx.ModelProto.decode(binaryModel);
    const plainDecodedModel =
      onnxProto.onnx.ModelProto.toObject(decodedModel) as DecodedExternalOnnxModel;
    const verificationError = onnxProto.onnx.ModelProto.verify(plainDecodedModel);

    if (verificationError) {
      throw new OnnxExternalImportError(
        'invalid-binary',
        verificationError,
      );
    }

    return plainDecodedModel;
  } catch (error) {
    if (error instanceof OnnxExternalImportError) {
      throw error;
    }

    throw new OnnxExternalImportError(
      'invalid-binary',
      error instanceof Error ? error.message : String(error),
    );
  }
}

function resolveDecodedGraph(decodedModel: DecodedExternalOnnxModel) {
  const decodedGraph = decodedModel.graph;
  if (!decodedGraph) {
    throw new OnnxExternalImportError(
      'invalid-model',
      'External ONNX binary must declare exactly one graph.',
    );
  }

  return decodedGraph;
}

function resolveDeclaredOpset(decodedModel: DecodedExternalOnnxModel): number {
  const standardDomainImports = (decodedModel.opsetImport ?? []).filter(
    (operatorSetImport) =>
      STANDARD_DOMAIN_NAMES.has(operatorSetImport.domain ?? ONNX_STANDARD_DOMAIN_ALIAS),
  );

  if (standardDomainImports.length !== 1) {
    throw new OnnxExternalImportError(
      'unsupported-domain',
      'External ONNX import requires exactly one standard-domain opset import.',
    );
  }

  const declaredOpset = readLongLikeNumber(standardDomainImports[0]!.version ?? 0);
  if (declaredOpset < FIRST_EXTERNAL_IMPORT_ONNX_OPSET) {
    throw new OnnxExternalImportError(
      'unsupported-opset',
      `External ONNX import requires opset ${FIRST_EXTERNAL_IMPORT_ONNX_OPSET} or newer for the first lane.`,
    );
  }

  return declaredOpset;
}

function buildInitializerMap(
  initializerTensors: DecodedExternalOnnxTensor[],
): Map<string, DecodedExternalOnnxTensor> {
  return initializerTensors.reduce<Map<string, DecodedExternalOnnxTensor>>(
    (initializerMap, initializerTensor) => {
      const initializerName = initializerTensor.name ?? '';
      if (!initializerName) {
        throw new OnnxExternalImportError(
          'invalid-model',
          'External ONNX initializers must be named.',
        );
      }

      if (initializerMap.has(initializerName)) {
        throw new OnnxExternalImportError(
          'duplicate-initializer',
          `External ONNX initializer '${initializerName}' is duplicated.`,
        );
      }

      initializerMap.set(initializerName, initializerTensor);
      return initializerMap;
    },
    new Map<string, DecodedExternalOnnxTensor>(),
  );
}

function resolvePublicInput(
  graphInputs: DecodedExternalOnnxValueInfo[],
  initializerMap: Map<string, DecodedExternalOnnxTensor>,
): DecodedExternalOnnxValueInfo {
  const publicInputs = graphInputs.filter(
    (valueInfo) => !initializerMap.has(valueInfo.name ?? ''),
  );

  if (publicInputs.length !== 1) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX import requires exactly one public model input.',
    );
  }

  return publicInputs[0]!;
}

function resolvePublicOutput(
  graphOutputs: DecodedExternalOnnxValueInfo[],
): DecodedExternalOnnxValueInfo {
  if (graphOutputs.length !== 1) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX import requires exactly one public model output.',
    );
  }

  return graphOutputs[0]!;
}

function validateSupportingValueInfo(
  supportingValueInfo: DecodedExternalOnnxValueInfo[],
): void {
  supportingValueInfo.forEach((valueInfo) => {
    readRankTwoFeatureWidth(valueInfo, `value-info tensor '${valueInfo.name ?? ''}'`);
  });
}

function readRankTwoFeatureWidth(
  valueInfo: DecodedExternalOnnxValueInfo,
  valueInfoLabel: string,
): number {
  const tensorType = valueInfo.type?.tensorType;
  if (!tensorType || tensorType.elemType !== FLOAT32_TENSOR_ELEMENT_TYPE) {
    throw new OnnxExternalImportError(
      'unsupported-tensor-type',
      `${valueInfoLabel} must be a float32 tensor.`,
    );
  }

  const dimensions = tensorType.shape?.dim ?? [];
  if (dimensions.length !== RANK_TWO_TENSOR_DIMENSIONS) {
    throw new OnnxExternalImportError(
      'rank-mismatch',
      `${valueInfoLabel} must stay rank 2 in the first external lane.`,
    );
  }

  const batchDimension = dimensions[VALUE_INFO_BATCH_DIMENSION_INDEX];
  const batchDimensionValue = batchDimension?.dimValue;
  if (
    batchDimensionValue !== undefined &&
    batchDimensionValue !== null &&
    readLongLikeNumber(batchDimensionValue) < 1
  ) {
    throw new OnnxExternalImportError(
      'rank-mismatch',
      `${valueInfoLabel} must not declare a non-positive batch dimension.`,
    );
  }

  const featureDimension = dimensions[VALUE_INFO_FEATURE_DIMENSION_INDEX];
  if (featureDimension?.dimValue === undefined || featureDimension.dimValue === null) {
    throw new OnnxExternalImportError(
      'rank-mismatch',
      `${valueInfoLabel} must declare a concrete positive feature width.`,
    );
  }

  const featureWidth = readLongLikeNumber(featureDimension.dimValue);
  if (featureWidth < 1) {
    throw new OnnxExternalImportError(
      'rank-mismatch',
      `${valueInfoLabel} must declare a concrete positive feature width.`,
    );
  }

  return featureWidth;
}

function buildProducerMap(
  graphNodes: DecodedExternalOnnxNode[],
): Map<string, DecodedExternalOnnxNode> {
  return graphNodes.reduce<Map<string, DecodedExternalOnnxNode>>(
    (producerMap, graphNode) => {
      const outputNames = graphNode.output ?? [];
      if (outputNames.length !== 1 || !outputNames[0]) {
        throw new OnnxExternalImportError(
          'unsupported-topology',
          'External ONNX import requires every node to produce exactly one output tensor.',
        );
      }

      const outputName = outputNames[0]!;
      if (producerMap.has(outputName)) {
        throw new OnnxExternalImportError(
          'unsupported-topology',
          `External ONNX tensor '${outputName}' has multiple producers.`,
        );
      }

      producerMap.set(outputName, graphNode);
      return producerMap;
    },
    new Map<string, DecodedExternalOnnxNode>(),
  );
}

function buildConsumerMap(
  graphNodes: DecodedExternalOnnxNode[],
  initializerMap: Map<string, DecodedExternalOnnxTensor>,
): Map<string, DecodedExternalOnnxNode[]> {
  return graphNodes.reduce<Map<string, DecodedExternalOnnxNode[]>>(
    (consumerMap, graphNode) => {
      (graphNode.input ?? [])
        .filter((inputName) => !initializerMap.has(inputName))
        .forEach((inputName) => {
          const existingConsumers = consumerMap.get(inputName) ?? [];
          existingConsumers.push(graphNode);
          consumerMap.set(inputName, existingConsumers);
        });

      return consumerMap;
    },
    new Map<string, DecodedExternalOnnxNode[]>(),
  );
}

function collectDenseLayers(context: {
  graphNodes: DecodedExternalOnnxNode[];
  initializerMap: Map<string, DecodedExternalOnnxTensor>;
  producerMap: Map<string, DecodedExternalOnnxNode>;
  consumerMap: Map<string, DecodedExternalOnnxNode[]>;
  declaredOpset: number;
  publicInputName: string;
  publicOutputName: string;
  inputWidth: number;
}): OnnxExternalDenseLayer[] {
  const visitedNodes = new Set<DecodedExternalOnnxNode>();
  const denseLayers: OnnxExternalDenseLayer[] = [];
  let currentTensorName = context.publicInputName;
  let currentTensorWidth = context.inputWidth;

  while (currentTensorName !== context.publicOutputName) {
    const finalOutputAliasNode = tryResolveFinalOutputAlias({
      consumerMap: context.consumerMap,
      visitedNodes,
      currentTensorName,
      publicOutputName: context.publicOutputName,
      declaredOpset: context.declaredOpset,
    });
    if (finalOutputAliasNode) {
      visitedNodes.add(finalOutputAliasNode);
      currentTensorName = context.publicOutputName;
      continue;
    }

    const gemmNode = resolveSingleConsumer(
      context.consumerMap,
      currentTensorName,
      'External ONNX dense chain requires exactly one consumer per tensor.',
    );
    validateStandardDomainNode(gemmNode);
    validateExpectedNodeType(gemmNode, GEMM_NODE_TYPE, currentTensorName);
    if (visitedNodes.has(gemmNode)) {
      throw new OnnxExternalImportError(
        'unsupported-topology',
        'External ONNX dense chain must remain acyclic.',
      );
    }

    visitedNodes.add(gemmNode);
    const denseLayer = normalizeDenseLayer({
      gemmNode,
      initializerMap: context.initializerMap,
      currentTensorName,
      currentTensorWidth,
    });

    const gemmOutputName = gemmNode.output![0]!;
    const nextConsumers = context.consumerMap.get(gemmOutputName) ?? [];
    const activationResolution = resolveLayerActivation({
      nextConsumers,
      visitedNodes,
      declaredOpset: context.declaredOpset,
      publicOutputName: context.publicOutputName,
      gemmOutputName,
    });
    if (activationResolution.activationNode) {
      visitedNodes.add(activationResolution.activationNode);
    }

    denseLayers.push({
      ...denseLayer,
      activation: activationResolution.activation,
    });
    currentTensorName = activationResolution.nextTensorName;
    currentTensorWidth = denseLayer.outputWidth;
  }

  if (denseLayers.length === 0) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX import requires at least one Gemm layer.',
    );
  }

  if (visitedNodes.size !== context.graphNodes.length) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX import only accepts one unambiguous dense chain with no disconnected nodes.',
    );
  }

  return denseLayers;
}

function tryResolveFinalOutputAlias(context: {
  consumerMap: Map<string, DecodedExternalOnnxNode[]>;
  visitedNodes: Set<DecodedExternalOnnxNode>;
  currentTensorName: string;
  publicOutputName: string;
  declaredOpset: number;
}): DecodedExternalOnnxNode | null {
  const nextConsumers = context.consumerMap.get(context.currentTensorName) ?? [];
  if (nextConsumers.length !== 1) {
    return null;
  }

  const aliasNode = nextConsumers[0]!;
  if (aliasNode.opType !== IDENTITY_ACTIVATION) {
    return null;
  }

  validateStandardDomainNode(aliasNode);
  validateActivationNode(
    aliasNode,
    context.declaredOpset,
    context.currentTensorName,
  );
  if (aliasNode.output![0]! !== context.publicOutputName) {
    return null;
  }

  return aliasNode;
}

function resolveSingleConsumer(
  consumerMap: Map<string, DecodedExternalOnnxNode[]>,
  tensorName: string,
  errorMessage: string,
): DecodedExternalOnnxNode {
  const nextConsumers = consumerMap.get(tensorName) ?? [];
  if (nextConsumers.length !== 1) {
    throw new OnnxExternalImportError('unsupported-topology', errorMessage);
  }

  return nextConsumers[0]!;
}

function validateStandardDomainNode(graphNode: DecodedExternalOnnxNode): void {
  const nodeDomain = graphNode.domain ?? ONNX_STANDARD_DOMAIN_ALIAS;
  if (!STANDARD_DOMAIN_NAMES.has(nodeDomain)) {
    throw new OnnxExternalImportError(
      'unsupported-domain',
      `External ONNX node '${graphNode.name ?? graphNode.opType ?? ''}' must stay in the standard domain.`,
    );
  }
}

function validateExpectedNodeType(
  graphNode: DecodedExternalOnnxNode,
  expectedNodeType: string,
  inputTensorName: string,
): void {
  if (graphNode.opType !== expectedNodeType) {
    throw new OnnxExternalImportError(
      'unsupported-node',
      `External ONNX tensor '${inputTensorName}' must flow into '${expectedNodeType}', not '${graphNode.opType ?? 'unknown'}'.`,
    );
  }
}

function normalizeDenseLayer(context: {
  gemmNode: DecodedExternalOnnxNode;
  initializerMap: Map<string, DecodedExternalOnnxTensor>;
  currentTensorName: string;
  currentTensorWidth: number;
}): OnnxExternalDenseLayer {
  validateCanonicalGemmNode(context.gemmNode, context.currentTensorName);

  const weightTensorName = context.gemmNode.input![GEMM_WEIGHT_INPUT_INDEX]!;
  const weightTensor = resolveInitializerTensor(
    context.initializerMap,
    weightTensorName,
  );
  const weightShape = readPositiveTensorDimensions(weightTensor);
  if (weightShape.length !== GEMM_WEIGHT_DIMENSIONS) {
    throw new OnnxExternalImportError(
      'shape-mismatch',
      `External ONNX weight tensor '${weightTensorName}' must stay rank 2 for Gemm.`,
    );
  }

  const outputWidth = weightShape[0]!;
  const inputWidth = weightShape[1]!;
  if (inputWidth !== context.currentTensorWidth) {
    throw new OnnxExternalImportError(
      'shape-mismatch',
      `External ONNX weight tensor '${weightTensorName}' does not match the active feature width.`,
    );
  }

  const weightValues = readFloat32TensorValues(weightTensor, outputWidth * inputWidth);
  const biasTensorName = context.gemmNode.input?.[GEMM_BIAS_INPUT_INDEX];
  const biasValues = biasTensorName
    ? readBiasValues(context.initializerMap, biasTensorName, outputWidth)
    : Array(outputWidth).fill(0);

  return {
    inputWidth,
    outputWidth,
    weightValues,
    biasValues,
    activation: IDENTITY_ACTIVATION,
  };
}

function validateCanonicalGemmNode(
  gemmNode: DecodedExternalOnnxNode,
  expectedInputTensorName: string,
): void {
  const inputNames = gemmNode.input!;
  if (inputNames.length < GEMM_WEIGHT_INPUT_INDEX + 1 || inputNames.length > GEMM_BIAS_INPUT_INDEX + 1) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External Gemm nodes must declare input activation, weight initializer, and optional bias initializer only.',
    );
  }

  if (inputNames[GEMM_A_INPUT_INDEX] !== expectedInputTensorName) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External Gemm nodes must consume the active tensor as their first input.',
    );
  }

  const attributeEntries = gemmNode.attribute ?? [];
  attributeEntries.forEach((attributeEntry) => {
    const attributeName = attributeEntry.name ?? '';
    if (!ALLOWED_GEMM_ATTRIBUTE_NAMES.has(attributeName)) {
      throw new OnnxExternalImportError(
        'unsupported-attribute',
        `External Gemm attribute '${attributeName}' is outside the first supported lane.`,
      );
    }

    const expectedValue = CANONICAL_GEMM_ATTRIBUTE_VALUES.get(attributeName)!;
    const actualValue = attributeEntry.f ?? readLongLikeNumber(attributeEntry.i ?? 0);
    if (actualValue !== expectedValue) {
      throw new OnnxExternalImportError(
        'unsupported-attribute',
        `External Gemm attribute '${attributeName}' must stay at its canonical first-lane value.`,
      );
    }
  });
}

function resolveInitializerTensor(
  initializerMap: Map<string, DecodedExternalOnnxTensor>,
  initializerName: string,
): DecodedExternalOnnxTensor {
  const initializerTensor = initializerMap.get(initializerName);
  if (!initializerTensor) {
    throw new OnnxExternalImportError(
      'missing-initializer',
      `External ONNX initializer '${initializerName}' is required but missing.`,
    );
  }

  if (initializerTensor.dataType !== FLOAT32_TENSOR_ELEMENT_TYPE) {
    throw new OnnxExternalImportError(
      'unsupported-tensor-type',
      `External ONNX initializer '${initializerName}' must stay float32 in the first lane.`,
    );
  }

  return initializerTensor;
}

function readPositiveTensorDimensions(
  initializerTensor: DecodedExternalOnnxTensor,
): number[] {
  const tensorDimensions = (initializerTensor.dims ?? []).map(readLongLikeNumber);
  if (tensorDimensions.some((dimensionValue) => dimensionValue < 1)) {
    throw new OnnxExternalImportError(
      'shape-mismatch',
      `External ONNX initializer '${initializerTensor.name!}' must declare only positive dimensions.`,
    );
  }

  return tensorDimensions;
}

function readFloat32TensorValues(
  initializerTensor: DecodedExternalOnnxTensor,
  expectedValueCount: number,
): number[] {
  const floatValues = initializerTensor.floatData?.length
    ? [...initializerTensor.floatData]
    : decodeRawFloat32Data(initializerTensor.rawData);

  if (floatValues.length !== expectedValueCount) {
    throw new OnnxExternalImportError(
      'shape-mismatch',
      `External ONNX initializer '${initializerTensor.name!}' does not match its declared tensor shape.`,
    );
  }

  return floatValues;
}

function decodeRawFloat32Data(rawData: OnnxDecodedBytes | null | undefined): number[] {
  if (!rawData) {
    return [];
  }

  const rawBytes = asByteArray(rawData);
  if (rawBytes.byteLength % FLOAT32_BYTES_PER_VALUE !== 0) {
    throw new OnnxExternalImportError(
      'unsupported-tensor-type',
      'External ONNX raw float32 payload must be aligned to 4-byte values.',
    );
  }

  const rawDataView = new DataView(
    rawBytes.buffer,
    rawBytes.byteOffset,
    rawBytes.byteLength,
  );

  return Array.from(
    { length: rawBytes.byteLength / FLOAT32_BYTES_PER_VALUE },
    (_unused, valueIndex) =>
      rawDataView.getFloat32(valueIndex * FLOAT32_BYTES_PER_VALUE, true),
  );
}

function asByteArray(rawData: OnnxDecodedBytes): Uint8Array {
  if (rawData instanceof Uint8Array) {
    return rawData;
  }

  if (Array.isArray(rawData)) {
    return Uint8Array.from(rawData);
  }

  if (typeof Buffer !== 'undefined') {
    return Uint8Array.from(Buffer.from(rawData, 'base64'));
  }

  const binaryText = globalThis.atob(rawData);
  return Uint8Array.from(binaryText, (character) => character.charCodeAt(0));
}

function readBiasValues(
  initializerMap: Map<string, DecodedExternalOnnxTensor>,
  biasTensorName: string,
  expectedOutputWidth: number,
): number[] {
  const biasTensor = resolveInitializerTensor(initializerMap, biasTensorName);
  const biasShape = readPositiveTensorDimensions(biasTensor);
  const isVectorBias =
    biasShape.length === BIAS_VECTOR_DIMENSIONS && biasShape[0] === expectedOutputWidth;
  const isSingleRowBias =
    biasShape.length === BIAS_MATRIX_DIMENSIONS &&
    biasShape[0] === 1 &&
    biasShape[1] === expectedOutputWidth;

  if (!isVectorBias && !isSingleRowBias) {
    throw new OnnxExternalImportError(
      'shape-mismatch',
      `External ONNX bias tensor '${biasTensorName}' must resolve to the layer output width without ambiguous broadcast.`,
    );
  }

  return readFloat32TensorValues(biasTensor, expectedOutputWidth);
}

function resolveLayerActivation(context: {
  nextConsumers: DecodedExternalOnnxNode[];
  visitedNodes: Set<DecodedExternalOnnxNode>;
  declaredOpset: number;
  publicOutputName: string;
  gemmOutputName: string;
}): {
  activation: OnnxActivationOperation;
  activationNode?: DecodedExternalOnnxNode;
  nextTensorName: string;
} {
  if (context.gemmOutputName === context.publicOutputName) {
    return {
      activation: IDENTITY_ACTIVATION,
      nextTensorName: context.publicOutputName,
    };
  }

  if (context.nextConsumers.length === 0) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX dense chain ended before reaching the public output.',
    );
  }

  if (context.nextConsumers.length > 1) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX dense chain must not branch or merge between layers.',
    );
  }

  const nextNode = context.nextConsumers[0]!;
  if (nextNode.opType === GEMM_NODE_TYPE) {
    return {
      activation: IDENTITY_ACTIVATION,
      nextTensorName: context.gemmOutputName,
    };
  }

  validateStandardDomainNode(nextNode);
  const activationOperation = nextNode.opType as OnnxActivationOperation;
  if (!ALLOWED_ACTIVATION_OPERATIONS.has(activationOperation)) {
    throw new OnnxExternalImportError(
      'unsupported-activation',
      `External ONNX activation '${nextNode.opType ?? 'unknown'}' is outside the first supported lane.`,
    );
  }

  validateActivationNode(nextNode, context.declaredOpset, context.gemmOutputName);
  return {
    activation: activationOperation,
    activationNode: nextNode,
    nextTensorName: nextNode.output![0]!,
  };
}

function validateActivationNode(
  activationNode: DecodedExternalOnnxNode,
  declaredOpset: number,
  expectedInputTensorName: string,
): void {
  const inputNames = activationNode.input!;
  if (inputNames.length !== 1) {
    throw new OnnxExternalImportError(
      'unsupported-topology',
      'External ONNX unary activations must consume one tensor and emit one tensor in the dense chain.',
    );
  }

  if (activationNode.opType === GELU_NODE_TYPE && declaredOpset < GELU_EXTERNAL_IMPORT_MINIMUM_OPSET) {
    throw new OnnxExternalImportError(
      'unsupported-opset',
      `External ONNX Gelu import requires opset ${GELU_EXTERNAL_IMPORT_MINIMUM_OPSET} or newer.`,
    );
  }

  const attributeEntries = activationNode.attribute ?? [];
  if (activationNode.opType !== GELU_NODE_TYPE && attributeEntries.length > 0) {
    throw new OnnxExternalImportError(
      'unsupported-attribute',
      `External ONNX activation '${activationNode.opType}' must not declare custom attributes in the first lane.`,
    );
  }

  if (activationNode.opType === GELU_NODE_TYPE) {
    validateGeluAttributes(attributeEntries);
  }
}

function validateGeluAttributes(
  attributeEntries: DecodedExternalOnnxAttribute[],
): void {
  attributeEntries.forEach((attributeEntry) => {
    if ((attributeEntry.name ?? '') !== 'approximate') {
      throw new OnnxExternalImportError(
        'unsupported-attribute',
        'External ONNX Gelu attributes must stay within the current approximation contract.',
      );
    }

    const attributeValue = readAttributeString(attributeEntry);
    if (attributeValue !== 'tanh') {
      throw new OnnxExternalImportError(
        'unsupported-attribute',
        'External ONNX Gelu currently supports only approximate=tanh in the first lane.',
      );
    }
  });
}

function readAttributeString(attributeEntry: DecodedExternalOnnxAttribute): string {
  const rawAttributeValue = attributeEntry.s;
  if (!rawAttributeValue) {
    return '';
  }

  if (typeof rawAttributeValue === 'string') {
    return rawAttributeValue;
  }

  return new TextDecoder().decode(asByteArray(rawAttributeValue));
}

function buildImporterOwnedOnnxModel(
  denseChain: OnnxExternalDenseChain,
): OnnxModel {
  const initializerTensors: OnnxTensor[] = [];
  const graphNodes: OnnxNode[] = [];
  let currentInputName = 'input';
  let currentOutputName = currentInputName;

  denseChain.layers.forEach((denseLayer, layerIndex) => {
    const canonicalLayerIndex = layerIndex + FIRST_CANONICAL_LAYER_INDEX;
    const weightTensorName = `W${layerIndex}`;
    const biasTensorName = `B${layerIndex}`;
    const gemmOutputName = `Layer_${canonicalLayerIndex}`;

    initializerTensors.push({
      name: weightTensorName,
      data_type: FLOAT32_TENSOR_ELEMENT_TYPE,
      dims: [denseLayer.outputWidth, denseLayer.inputWidth],
      float_data: denseLayer.weightValues,
    });
    initializerTensors.push({
      name: biasTensorName,
      data_type: FLOAT32_TENSOR_ELEMENT_TYPE,
      dims: [denseLayer.outputWidth],
      float_data: denseLayer.biasValues,
    });
    graphNodes.push({
      name: `gemm_l${canonicalLayerIndex}`,
      op_type: GEMM_NODE_TYPE,
      input: [currentInputName, weightTensorName, biasTensorName],
      output: [gemmOutputName],
    });

    if (denseLayer.activation === IDENTITY_ACTIVATION) {
      currentInputName = gemmOutputName;
      currentOutputName = gemmOutputName;
      return;
    }

    const activationOutputName = `Activation_${canonicalLayerIndex}`;
    graphNodes.push({
      name: `act_l${canonicalLayerIndex}`,
      op_type: denseLayer.activation,
      input: [gemmOutputName],
      output: [activationOutputName],
    });
    currentInputName = activationOutputName;
    currentOutputName = activationOutputName;
  });

  return {
    opset_import: [
      {
        version: denseChain.opsetVersion,
        domain: ONNX_STANDARD_DOMAIN_ALIAS,
      },
    ],
    graph: {
      inputs: [createCanonicalValueInfo('input', denseChain.inputWidth)],
      outputs: [createCanonicalValueInfo(currentOutputName, denseChain.outputWidth)],
      initializer: initializerTensors,
      node: graphNodes,
    },
  };
}

function createCanonicalValueInfo(name: string, featureWidth: number) {
  return {
    name,
    type: {
      tensor_type: {
        elem_type: FLOAT32_TENSOR_ELEMENT_TYPE,
        shape: {
          dim: [{ dim_param: 'N' }, { dim_value: featureWidth }],
        },
      },
    },
  };
}

function readLongLikeNumber(value: OnnxDecodedLongLike): number {
  return Number.parseInt(value.toString(), 10);
}