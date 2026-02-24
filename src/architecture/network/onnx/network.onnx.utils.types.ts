/**
 * Types for NeatapticTS’s ONNX-like JSON export/import.
 *
 * The exporter produces an `OnnxModel` (a JSON-serializable object) and the importer
 * reconstructs a `Network` from that object.
 *
 * Practical notes:
 * - These types intentionally resemble ONNX’s `ModelProto`/`GraphProto` concepts, but they
 *   are *not* a full ONNX protobuf implementation.
 * - `opset` and `ir_version` are recorded as metadata for inspection/compat bookkeeping.
 *   They are not a promise of universal ONNX-runtime compatibility.
 *
 * Stability & compatibility expectations:
 * - This repo’s importer is only guaranteed to accept models produced by this repo’s
 *   exporter.
 * - The schema is JSON-first and may evolve; prefer re-exporting/importing through the
 *   library rather than hand-editing blobs.
 */

import Connection from '../../connection';
import type Layer from '../../layer';
import type Network from '../../network';
import type NeatapticNode from '../../node';

/**
 * Runtime perceptron factory signature used by ONNX import orchestration.
 *
 * This factory is injected so the ONNX import path can rebuild an MLP without taking a
 * hard dependency on a specific constructor shape.
 */
export type OnnxRuntimePerceptronFactory = (...sizes: number[]) => Network;

/**
 * Runtime layer-constructor signature used for recurrent layer reconstruction.
 *
 * ONNX import can optionally reconstruct higher-level recurrent layers (like LSTM/GRU)
 * from exported metadata. This factory provides the concrete layer implementation.
 */
export type OnnxRuntimeLayerFactory = (size: number) => Layer;

/**
 * Runtime layer module shape consumed by ONNX import orchestration.
 *
 * This is the minimal set of recurrent factories needed by the importer.
 */
export type OnnxRuntimeLayerModule = {
  lstm: OnnxRuntimeLayerFactory;
  gru: OnnxRuntimeLayerFactory;
};

/**
 * Runtime factories consumed during ONNX import network reconstruction.
 *
 * These factories let the importer reconstruct runtime objects (network + layers)
 * while keeping the ONNX parser itself mostly pure.
 */
export type OnnxRuntimeFactories = {
  perceptronFactory: OnnxRuntimePerceptronFactory;
  layerModule: OnnxRuntimeLayerModule;
};

/** Validation context for perceptron size-list checks during ONNX import. */
export type OnnxPerceptronSizeValidationContext = {
  sizes: number[];
  minimumSizeCount: number;
  errorMessage: string;
};

/** Build context for mapping ONNX layer sizes into a Neataptic MLP factory call. */
export type OnnxPerceptronBuildContext = {
  sizes: number[];
  inputIndex: number;
  hiddenSliceStartIndex: number;
  hiddenSliceEndOffset: number;
  outputFallbackCount: number;
};

/**
 * Runtime interface for accessing node internal properties.
 *
 * This is intentionally "internal": it exposes mutable fields that the ONNX exporter/importer
 * needs (connections, bias, squash). Regular library users should generally interact with
 * the public `Node` API instead.
 */
export interface NodeInternals {
  connections: {
    in: Connection[];
    out: Connection[];
    self: Connection[];
  };
  bias: number;
  squash: ((x: number, derivate?: boolean) => number) & { name?: string };
}

/** Runtime node internals augmented with optional export index metadata. */
export type NodeInternalsWithExportIndex = NodeInternals & {
  index?: number;
};

/**
 * Runtime activation function signature used by ONNX activation import/export paths.
 *
 * Neataptic-style activations support a dual-purpose call pattern:
 * - `derivate === false | undefined`: return activation output $f(x)$
 * - `derivate === true`: return derivative $f'(x)$
 *
 * This matches historical Neataptic semantics and keeps ONNX import/export compatible.
 *
 * Example:
 *
 * ```ts
 * const y = activation(x);
 * const dy = activation(x, true);
 * ```
 */
export type ActivationFunction = NodeInternals['squash'];

/** Node partitions used by ONNX layered-ordering inference traversal. */
export type LayerOrderingNodeGroups = {
  inputNodes: NeatapticNode[];
  hiddenNodes: NeatapticNode[];
  outputNodes: NeatapticNode[];
};

/** Mutable traversal state while resolving hidden-layer ordering. */
export type LayerOrderingResolutionContext = {
  remainingHiddenNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  orderedLayers: NeatapticNode[][];
};

/** Layer-wise validation context for activation and connectivity checks. */
export type LayerValidationTraversalContext = {
  layerIndex: number;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  options: OnnxExportOptions;
};

/** Activation-homogeneity decision context for one current layer. */
export type LayerActivationValidationContext = {
  layerIndex: number;
  activationNames: (string | undefined)[];
  allowMixedActivations: boolean;
};

/** Connectivity decision context for one source-target node pair. */
export type LayerConnectivityValidationContext = {
  layerIndex: number;
  sourceNode: NeatapticNode;
  targetNode: NeatapticNode;
  allowPartialConnectivity: boolean;
};

/** Supported ONNX activation operators recognized during activation import. */
export type OnnxActivationOperation =
  | 'Tanh'
  | 'Sigmoid'
  | 'Logistic'
  | 'Relu'
  | 'Identity';

/** Layer-indexed activation operator lookup extracted from ONNX graph nodes. */
export type OnnxActivationLayerOperations = Record<
  number,
  OnnxActivationOperation[]
>;

/** Parsed ONNX activation-node naming payload. */
export type OnnxActivationParseResult = {
  layerIndex: number;
  neuronIndex?: number;
};

/** Shared activation-assignment context for hidden and output traversal. */
export type OnnxActivationAssignmentContext = {
  hiddenLayerSizes: number[];
  hiddenNodes: NodeInternals[];
  outputNodes: NodeInternals[];
  operationsByLayer: OnnxActivationLayerOperations;
};

/** Hidden-layer traversal context for assigning imported activation functions. */
export type HiddenLayerActivationTraversalContext = {
  hiddenLayerIndex: number;
  hiddenLayerSize: number;
  hiddenOffset: number;
  hiddenNodes: NodeInternals[];
  operationsByLayer: OnnxActivationLayerOperations;
};

/** Output-layer activation assignment context. */
export type OutputLayerActivationContext = {
  outputLayerIndex: number;
  outputNodes: NodeInternals[];
  operationsByLayer: OnnxActivationLayerOperations;
};

/** Activation operation resolution context for one neuron or layer default. */
export type OnnxActivationOperationResolutionContext = {
  operations: OnnxActivationOperation[];
  neuronIndex: number;
};

/** Context for assigning a stable export index to one node. */
export type ExportNodeIndexAssignmentContext = {
  node: NeatapticNode;
  exportIndex: number;
};

/** Heuristic LSTM pattern stub for metadata output. */
export type LstmPatternStub = {
  layerIndex: number;
  unitSize: number;
};

/** Traversal context for one hidden layer during LSTM stub collection. */
export type LstmLayerTraversalContext = {
  layerIndex: number;
  hiddenLayerNodes: NeatapticNode[];
};

/** Candidate context for validating one LSTM-like hidden layer pattern. */
export type LstmCandidateContext = {
  layerIndex: number;
  totalNodes: number;
  unitSize: number;
  memorySliceNodes: NeatapticNode[];
};

/** Traversal context for one hidden layer during Conv inference. */
export type ConvInferenceTraversalContext = {
  layerIndex: number;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  declaredMappings: Conv2DMapping[] | undefined;
};

/** Width and shape evaluation context used by Conv inference helpers. */
export type ConvInferenceEvaluationContext = {
  layerIndex: number;
  currentWidth: number;
  squareWidth: number;
  isSquareInputWidth: boolean;
};

/** Kernel candidate context for one Conv inference evaluation pass. */
export type ConvInferenceKernelEvaluationContext = {
  evaluationContext: ConvInferenceEvaluationContext;
  kernelSize: number;
};

/** Collected inferred Conv metadata payload. */
export type ConvInferenceResult = {
  inferredLayers: number[];
  inferredSpecs: (Conv2DMapping & { note?: string })[];
};

/**
 * Options controlling ONNX-like export.
 *
 * These options trade off strictness, portability, and fidelity:
 *
 * - **Strict (default-ish)** export tries to keep the graph easy to interpret:
 *   layered topology, homogeneous activations per layer, and fully-connected layers.
 *
 * - **Relaxed** export (`allowPartialConnectivity` / `allowMixedActivations`) can represent
 *   more networks, but it may generate graphs that are primarily meant for NeatapticTS’s
 *   importer (and may be less friendly to external ONNX tooling).
 *
 * - **Recurrent export** (`allowRecurrent`) is intentionally conservative and currently
 *   focuses on a constrained single-step representation and optional fused heuristics.
 *
 * Key fields (high-level):
 * - `includeMetadata`: includes `metadata_props` with architecture hints.
 * - `opset`: numeric opset version stored in the exported model metadata (default is
 *   resolved by the exporter; commonly 18 in this codebase).
 * - `legacyNodeOrdering`: keeps older node ordering for backward compatibility.
 * - `conv2dMappings` / `pool2dMappings`: encode conv/pool semantics for fully-connected
 *   layers via explicit mapping declarations.
 */
export interface OnnxExportOptions {
  opset?: number;
  includeMetadata?: boolean;
  batchDimension?: boolean;
  legacyNodeOrdering?: boolean;
  producerName?: string;
  producerVersion?: string;
  docString?: string;
  allowPartialConnectivity?: boolean;
  allowMixedActivations?: boolean;
  allowRecurrent?: boolean;
  recurrentSingleStep?: boolean;
  conv2dMappings?: Conv2DMapping[];
  pool2dMappings?: Pool2DMapping[];
  validateConvSharing?: boolean;
  flattenAfterPooling?: boolean;
}

/** Resolved options used by ONNX model build orchestration. */
export type OnnxBuildResolvedOptions = {
  includeMetadata: boolean;
  opset: number;
  batchDimension: boolean;
  legacyNodeOrdering: boolean;
  producerName: string;
  producerVersion?: string;
  docString?: string;
};

/** Context for constructing input/output ONNX graph dimensions. */
export type OnnxGraphDimensionBuildContext = {
  inputWidth: number;
  outputWidth: number;
  batchDimension: boolean;
};

/** Output dimensions used by ONNX graph input/output value info payloads. */
export type OnnxGraphDimensions = {
  inputDims: OnnxDimension[];
  outputDims: OnnxDimension[];
};

/** Context for constructing a base ONNX model shell. */
export type OnnxBaseModelBuildContext = {
  inputDims: OnnxDimension[];
  outputDims: OnnxDimension[];
};

/** Context for applying optional ONNX model metadata. */
export type OnnxModelMetadataContext = {
  model: OnnxModel;
  includeMetadata: boolean;
  opset: number;
  producerName: string;
  producerVersion?: string;
  docString?: string;
};

/** Context for collecting recurrent layer indices during model build. */
export type OnnxRecurrentCollectionContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  options: OnnxExportOptions;
  batchDimension: boolean;
};

/** Traversal context for one hidden layer during recurrent-input collection. */
export type OnnxRecurrentLayerTraversalContext = {
  layerIndex: number;
  hiddenLayerNodes: NeatapticNode[];
  batchDimension: boolean;
};

/** Context for constructing one recurrent previous-state graph input payload. */
export type OnnxRecurrentInputValueInfoContext = {
  previousStateInputName: string;
  hiddenLayerWidth: number;
  batchDimension: boolean;
};

/** Execution context for processing one hidden recurrent layer. */
export type OnnxRecurrentLayerProcessingContext = {
  model: OnnxModel;
  traversalContext: OnnxRecurrentLayerTraversalContext;
  recurrentLayerIndices: number[];
};

/** Result of emitting non-input export layers. */
export type OnnxLayerEmissionResult = {
  previousOutputName: string;
  hiddenSizesMetadata: number[];
};

/** Context for emitting non-input layers during model build. */
export type OnnxLayerEmissionContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  options: OnnxExportOptions;
  recurrentLayerIndices: number[];
  batchDimension: boolean;
  legacyNodeOrdering: boolean;
};

/** Layer build context used while emitting one ONNX graph layer segment. */
export type LayerBuildContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  options: OnnxExportOptions;
  layerIndex: number;
  previousOutputName: string;
  recurrentLayerIndices: number[];
  batchDimension: boolean;
  legacyNodeOrdering: boolean;
};

/** Layer traversal context with adjacent layers and output classification. */
export type LayerTraversalContext = LayerBuildContext & {
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  isOutputLayer: boolean;
};

/** Activation analysis context for one layer. */
export type LayerActivationContext = {
  hasMixedActivations: boolean;
};

/** Context used to decide recurrent emission branch usage. */
export type LayerRecurrentDecisionContext = {
  recurrentLayerIndices: number[];
  layerIndex: number;
  isOutputLayer: boolean;
};

/** Context for post-processing and export metadata finalization. */
export type OnnxPostProcessingContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  options: OnnxExportOptions;
  includeMetadata: boolean;
  recurrentLayerIndices: number[];
  layerEmissionResult: OnnxLayerEmissionResult;
};

/** Context for heuristic recurrent operator emission traversal. */
export type RecurrentHeuristicEmissionContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  previousOutputName: string;
};

/** Context for one hidden layer during heuristic recurrent emission. */
export type HiddenLayerHeuristicContext = {
  model: OnnxModel;
  layers: NeatapticNode[][];
  layerIndex: number;
  previousOutputName: string;
  currentLayerNodes: NeatapticNode[];
  currentSize: number;
};

/** Context for heuristic LSTM emission when a layer matches expected shape. */
export type LstmEmissionContext = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  unitSize: number;
};

/** Context for heuristic GRU emission when a layer matches expected shape. */
export type GruEmissionContext = {
  model: OnnxModel;
  layerIndex: number;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  unitSize: number;
};

/** Context for collecting one recurrent gate row (one neuron). */
export type RecurrentGateRowCollectionContext = {
  previousLayerNodes: NeatapticNode[];
  targetNodeInternal: NodeInternals;
  rowIndex: number;
  unitSize: number;
  useDiagonalSelfWeights: boolean;
};

/** One recurrent gate row payload before flatten fold. */
export type RecurrentGateRow = {
  inputWeights: number[];
  recurrentWeights: number[];
  bias: number;
};

/** Flattened recurrent gate parameter vectors for one fused operator. */
export type RecurrentGateParameterCollectionResult = {
  inputWeights: number[];
  recurrentWeights: number[];
  biases: number[];
};

/** Context for collecting one gate parameter block. */
export type RecurrentGateBlockCollectionContext = {
  gateNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  unitSize: number;
  useDiagonalSelfWeights: boolean;
};

/** Context for ONNX fused recurrent initializer names. */
export type FusedRecurrentInitializerNames = {
  weightName: string;
  recurrentWeightName: string;
  biasName: string;
};

/** Context for ONNX fused recurrent node payload names. */
export type FusedRecurrentGraphNames = {
  nodeName: string;
  outputName: string;
};

/** Shared execution context for emitting one fused recurrent layer payload. */
export type FusedRecurrentEmissionExecutionContext = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  gateNodeGroups: NeatapticNode[][];
  unitSize: number;
  diagonalGateIndex: number;
  operatorType: 'LSTM' | 'GRU';
  metadataKey: string;
  nodePrefix: string;
  outputSuffix: string;
};

/** Parameters for single-step recurrent layer emission. */
export type RecurrentLayerEmissionParams = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
};

/** Derived execution context for single-step recurrent layer emission. */
export type RecurrentLayerEmissionContext = RecurrentLayerEmissionParams & {
  layerSlot: number;
  previousLayerWidth: number;
  currentLayerWidth: number;
};

/** Initializer tensor names for one single-step recurrent layer. */
export type RecurrentInitializerNames = {
  weightTensorName: string;
  biasTensorName: string;
  recurrentTensorName: string;
};

/** Collected initializer vectors for one single-step recurrent layer. */
export type RecurrentInitializerValues = {
  weightMatrixValues: number[];
  biasVector: number[];
  recurrentWeights: number[];
};

/** Context for pushing recurrent initializers into ONNX graph state. */
export type RecurrentInitializerEmissionContext = {
  model: OnnxModel;
  previousLayerWidth: number;
  currentLayerWidth: number;
  names: RecurrentInitializerNames;
  values: RecurrentInitializerValues;
};

/** Context for emitting one Gemm node for recurrent single-step export. */
export type RecurrentGemmEmissionContext = {
  model: OnnxModel;
  inputNames: string[];
  outputName: string;
  nodeName: string;
};

/** Derived graph names for one recurrent single-step layer payload. */
export type RecurrentGraphNames = {
  inputGemmOutputName: string;
  recurrentGemmOutputName: string;
  recurrentSumOutputName: string;
  layerOutputName: string;
  inputGemmNodeName: string;
  recurrentGemmNodeName: string;
  recurrentAddNodeName: string;
  activationNodeName: string;
};

/** Context for selecting and emitting recurrent activation node payload. */
export type RecurrentActivationEmissionContext = {
  model: OnnxModel;
  currentLayerNodes: NeatapticNode[];
  recurrentSumOutputName: string;
  layerOutputName: string;
  activationNodeName: string;
};

/** Result of Conv sharing validation across declared mappings. */
export type ConvSharingValidationResult = {
  verifiedLayers: number[];
  mismatchedLayers: number[];
};

/** Context for validating Conv sharing across all declared mappings. */
export type ConvSharingValidationContext = {
  layers: NeatapticNode[][];
  mappings: Conv2DMapping[];
};

/** Context for one resolved Conv mapping layer pair. */
export type ConvLayerPairContext = {
  convSpec: Conv2DMapping;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
};

/** Coordinate for one Conv output neuron position. */
export type ConvOutputCoordinate = {
  outChannelIndex: number;
  outRowIndex: number;
  outColumnIndex: number;
};

/** Context for representative Conv kernel collection per output channel. */
export type ConvRepresentativeKernelContext = {
  convSpec: Conv2DMapping;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  outChannelIndex: number;
};

/** Context for kernel-coordinate consistency checks at one output position. */
export type ConvKernelConsistencyContext = {
  convSpec: Conv2DMapping;
  previousLayerNodes: NeatapticNode[];
  neuronInternal: NodeInternals;
  outputCoordinate: ConvOutputCoordinate;
  kernelCoordinate: OnnxConvKernelCoordinate;
  representativeKernelWeights: number[];
  kernelPointer: number;
  tolerance: number;
};

/** Context for comparing two scalar weights with numeric tolerance. */
export type WeightToleranceComparisonContext = {
  leftWeight: number;
  rightWeight: number;
  tolerance: number;
};

/** Parameters accepted by Conv layer emission. */
export type OnnxConvEmissionParams = {
  model: OnnxModel;
  options: OnnxExportOptions;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
};

/** Context used after resolving Conv mapping for one layer. */
export type OnnxConvEmissionContext = OnnxConvEmissionParams & {
  convSpec: Conv2DMapping;
};

/** Flattened Conv parameters for ONNX initializers. */
export type OnnxConvParameters = {
  weights: number[];
  biases: number[];
};

/** Tensor names generated for Conv parameters. */
export type OnnxConvTensorNames = {
  convWeightName: string;
  convBiasName: string;
};

/** Coordinate for one Conv kernel weight lookup. */
export type OnnxConvKernelCoordinate = {
  inChannelIndex: number;
  kernelRowIndex: number;
  kernelColumnIndex: number;
};

/**
 * Mapping declaration for treating a fully-connected layer as a 2D convolution during export.
 *
 * This does **not** magically turn an MLP into a convolutional network at runtime.
 * It annotates a particular export-layer index with a conv interpretation so that:
 * - The exported graph uses conv-shaped tensors/operators, and
 * - Import can re-attach pooling/flatten metadata appropriately.
 *
 * Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
 * does not correspond to the prior layer width (and similarly for outputs), export or import
 * may reject the model.
 */
export interface Conv2DMapping {
  layerIndex: number;
  inHeight: number;
  inWidth: number;
  inChannels: number;
  kernelHeight: number;
  kernelWidth: number;
  strideHeight: number;
  strideWidth: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  outHeight: number;
  outWidth: number;
  outChannels: number;
  activation?: string;
}

/**
 * Mapping describing a pooling operation inserted after a given export-layer index.
 *
 * This is represented as metadata and optional graph nodes during export.
 * Import uses it to attach pooling-related runtime metadata back onto the reconstructed
 * network (when supported).
 */
export interface Pool2DMapping {
  afterLayerIndex: number;
  type: 'MaxPool' | 'AveragePool';
  kernelHeight: number;
  kernelWidth: number;
  strideHeight: number;
  strideWidth: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  activation?: string;
}

/** ONNX tensor type shape dimension. */
export type OnnxDimension = {
  dim_value?: number;
  dim_param?: string;
};

/** ONNX tensor type shape. */
export type OnnxShape = {
  dim: OnnxDimension[];
};

/** ONNX tensor type. */
export type OnnxTensorType = {
  elem_type: number;
  shape: OnnxShape;
};

/** ONNX value info (input/output description). */
export type OnnxValueInfo = {
  name: string;
  type: {
    tensor_type: OnnxTensorType;
  };
};

/** ONNX node attribute. */
export type OnnxAttribute = {
  name: string;
  type?: string;
  f?: number;
  i?: number;
  s?: string;
  t?: OnnxTensor;
  g?: OnnxGraph;
  floats?: number[];
  ints?: number[];
  strings?: string[];
};

/**
 * ONNX-like model container (JSON-serializable).
 *
 * This is the main “wire format” object in this folder. Persist it as JSON text:
 *
 * ```ts
 * const jsonText = JSON.stringify(model);
 * const restoredModel = JSON.parse(jsonText) as OnnxModel;
 * ```
 *
 * Notes:
 * - `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
 *   conv/pool mappings, etc.). This is where most round-trip hints live.
 * - Initializers currently store floating-point weights in `float_data`.
 *
 * Security/trust boundary:
 * - Treat this as untrusted input if it comes from outside your process.
 */
export type OnnxModel = {
  ir_version?: number;
  opset_import?: { version: number; domain: string }[];
  producer_name?: string;
  producer_version?: string;
  doc_string?: string;
  metadata_props?: { key: string; value: string }[];
  graph: OnnxGraph;
};

export type OnnxGraph = {
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializer: OnnxTensor[];
  node: OnnxNode[];
};

export type OnnxTensor = {
  name: string;
  data_type: number;
  dims: number[];
  float_data: number[];
};

export type OnnxNode = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes?: OnnxAttribute[];
};

/** Activation function signature used by ONNX layer emission helpers. */
export type ActivationSquashFunction = ((
  x: number,
  derivate?: boolean,
) => number) & {
  name?: string;
};

/** Shared parameters for constructing a Gemm node payload. */
export type SharedGemmNodeBuildParams = {
  previousOutputName: string;
  weightTensorName: string;
  biasTensorName: string;
  gemmOutputName: string;
  nodeName: string;
};

/** Shared parameters for constructing an activation node payload. */
export type SharedActivationNodeBuildParams = {
  activationType: string;
  gemmOutputName: string;
  activationOutputName: string;
  nodeName: string;
};

/** Shared parameters for optional pooling/flatten output emission. */
export type OptionalLayerOutputParams = {
  model: OnnxModel;
  options: OnnxExportOptions;
  layerIndex: number;
  sourceOutputName: string;
};

/** Context for building dense layer initializers from two adjacent layers. */
export type DenseWeightBuildContext = {
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
};

/** One collected dense row before fold to flattened initializers. */
export type DenseWeightRow = {
  bias: number;
  weights: number[];
};

/** Dense layer initializer fold output. */
export type DenseWeightBuildResult = {
  weightMatrixValues: number[];
  biasVector: number[];
};

/** Context for collecting one dense row. */
export type DenseWeightRowCollectionContext = {
  previousLayerNodes: NeatapticNode[];
  targetNodeInternal: NodeInternals;
};

/** Context for building a diagonal recurrent matrix from self-connections. */
export type DiagonalRecurrentBuildContext = {
  currentLayerNodes: NeatapticNode[];
};

/** Context for collecting one recurrent matrix row. */
export type RecurrentRowCollectionContext = {
  currentLayerNodes: NeatapticNode[];
  rowIndex: number;
};

/** Parameters for optional pooling + flatten emission after a layer output. */
export type OptionalPoolingAndFlattenParams = OptionalLayerOutputParams & {
  poolSpec?: Pool2DMapping;
};

/** Pooling emission context resolved for one layer output. */
export type PoolingEmissionContext = {
  model: OnnxModel;
  options: OnnxExportOptions;
  layerIndex: number;
  sourceOutputName: string;
  poolSpec: Pool2DMapping;
};

/** Flatten emission context after optional pooling. */
export type FlattenAfterPoolingContext = {
  model: OnnxModel;
  flattenAfterPooling: boolean | undefined;
  layerIndex: number;
  sourceOutputName: string;
};

/** Pooling tensor attributes for ONNX node payloads. */
export type PoolingAttributes = {
  kernelShape: number[];
  strides: number[];
  pads: number[];
};

/** Append-an-index metadata context for JSON-array metadata keys. */
export type IndexedMetadataAppendContext = {
  model: OnnxModel;
  key: string;
  layerIndex: number;
};

/** Append-a-spec metadata context for JSON-array metadata keys. */
export type SpecMetadataAppendContext = {
  model: OnnxModel;
  key: string;
  spec: Conv2DMapping | Pool2DMapping;
};

/** Canonical metadata key-value pair used in ONNX model metadata_props. */
export type OnnxMetadataProperty = {
  key: string;
  value: string;
};

/** Parameters for dense layer emission. */
export type DenseLayerParams = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  legacyNodeOrdering: boolean;
  options: OnnxExportOptions;
};

/** Dense layer context enriched with resolved activation function. */
export type DenseLayerContext = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  legacyNodeOrdering: boolean;
  options: OnnxExportOptions;
  activationSquash: ActivationSquashFunction;
};

/** Dense initializer tensor names. */
export type DenseTensorNames = {
  weightTensorName: string;
  biasTensorName: string;
};

/** Dense initializer value arrays. */
export type DenseInitializerValues = {
  weightMatrixValues: number[];
  biasVector: number[];
};

/** Dense graph tensor names. */
export type DenseGraphNames = {
  gemmOutputName: string;
  activationOutputName: string;
};

/** Dense activation emission context. */
export type DenseActivationContext = {
  layerIndex: number;
  previousOutputName: string;
  legacyNodeOrdering: boolean;
  tensorNames: DenseTensorNames;
  graphNames: DenseGraphNames;
  squash: ActivationSquashFunction;
};

/** Strongly typed Gemm node payload used by dense export helpers. */
export type DenseGemmNodePayload = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes: { name: string; type: string; f?: number; i?: number }[];
};

/** Strongly typed activation node payload used by dense export helpers. */
export type DenseActivationNodePayload = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
};

/** Dense node payload union used by ordered append helpers. */
export type DenseOrderedNodePayload =
  | DenseGemmNodePayload
  | DenseActivationNodePayload;

/** Parameters for per-neuron layer emission. */
export type PerNeuronLayerParams = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  options: OnnxExportOptions;
  batchDimension: boolean;
};

/** Per-neuron layer context alias. */
export type PerNeuronLayerContext = PerNeuronLayerParams;

/** Per-neuron subgraph emission context. */
export type PerNeuronSubgraphContext = {
  model: OnnxModel;
  layerIndex: number;
  neuronIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  targetNode: NeatapticNode;
};

/** Per-neuron normalized node context. */
export type PerNeuronNodeContext = {
  model: OnnxModel;
  layerIndex: number;
  neuronIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  targetNodeInternal: NodeInternals;
};

/** Per-neuron initializer tensor names. */
export type PerNeuronTensorNames = {
  weightTensorName: string;
  biasTensorName: string;
};

/** Per-neuron graph tensor names. */
export type PerNeuronGraphNames = {
  gemmOutputName: string;
  activationOutputName: string;
};

/** Per-neuron concat node payload. */
export type PerNeuronConcatNodePayload = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes: { name: string; type: string; i?: number }[];
};

/** Runtime factory map used to construct dynamic recurrent layer modules. */
export type OnnxLayerFactory = Record<string, (...args: unknown[]) => unknown>;

/** Runtime layer module shape widened for fused-recurrent reconstruction wiring. */
export type OnnxRuntimeLayerFactoryMap = OnnxRuntimeLayerModule &
  OnnxLayerFactory;

/** Supported fused recurrent operator families recognized during ONNX import. */
export type OnnxFusedRecurrentKind = 'LSTM' | 'GRU';

/** Runtime interface of a reconstructed fused recurrent layer instance. */
export interface OnnxFusedLayerRuntime {
  nodes: NeatapticNode[];
  input: (groupLike: unknown) => void;
  output: { nodes: NeatapticNode[] } | null;
}

/** Fused recurrent family specification used during import reconstruction. */
export type OnnxFusedRecurrentSpec = {
  kind: OnnxFusedRecurrentKind;
  gateCount: number;
  gateOrder: string[];
  recurrentGateName: string;
  metadataKey: string;
};

/** Hidden-layer neighborhood slices around a reconstructed fused layer. */
export type OnnxFusedLayerNeighborhood = {
  hiddenNodes: NeatapticNode[];
  oldLayerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  nextLayerNodes: NeatapticNode[];
  start: number;
  end: number;
};

/** Fused recurrent tensor payload read from ONNX initializers. */
export type OnnxFusedTensorPayload = {
  inputWeights: number[];
  recurrentWeights: number[];
  biases: number[];
  rows: number;
  previousLayerWidth: number;
};

/** Execution context for one fused recurrent layer reconstruction. */
export type OnnxFusedLayerReconstructionContext = {
  spec: OnnxFusedRecurrentSpec;
  exportLayerIndex: number;
  hiddenLayerIndex: number;
};

/** Gate-weight application context for one reconstructed fused layer. */
export type OnnxFusedGateApplicationContext = {
  fusedLayer: OnnxFusedLayerRuntime;
  spec: OnnxFusedRecurrentSpec;
  unitSize: number;
  previousLayerWidth: number;
  biases: number[];
  inputWeights: number[];
  recurrentWeights: number[];
  previousLayerNodes: NeatapticNode[];
};

/** Context for assigning one gate-neuron row from flattened ONNX tensors. */
export type OnnxFusedGateRowAssignmentContext = {
  gateNeuronInternal: NodeInternals;
  gateName: string;
  recurrentGateName: string;
  rowOffset: number;
  rowIndex: number;
  unitSize: number;
  previousLayerWidth: number;
  biases: number[];
  inputWeights: number[];
  recurrentWeights: number[];
  previousLayerNodes: NeatapticNode[];
};

/** Context for assigning dense incoming weights for one gate-neuron row. */
export type OnnxIncomingWeightAssignmentContext = {
  gateNeuronInternal: NodeInternals;
  rowOffset: number;
  previousLayerWidth: number;
  inputWeights: number[];
  previousLayerNodes: NeatapticNode[];
};

/** Bucketed ONNX dense/per-neuron tensors for one exported layer index. */
export type OnnxImportLayerWeightBucket = {
  aggregated?: OnnxTensor;
  perNeuron: OnnxTensor[];
};

/** Context for deriving hidden layer sizes from initializer tensors and metadata. */
export type OnnxImportHiddenSizeDerivationContext = {
  initializers: OnnxTensor[];
  metadataProps: OnnxMetadataProperty[];
};

/** Shared weight-assignment context built once per ONNX import. */
export type OnnxImportWeightAssignmentContext = {
  onnx: OnnxModel;
  hiddenLayerSizes: number[];
  metadataProps: OnnxMetadataProperty[];
  initializerMap: Record<string, OnnxTensor>;
  sortedLayerIndices: number[];
  inputNodes: NeatapticNode[];
  hiddenNodes: NeatapticNode[];
  outputNodes: NeatapticNode[];
};

/** Build params for creating shared ONNX import weight-assignment context. */
export type OnnxImportWeightAssignmentBuildParams = {
  network: Network;
  onnx: OnnxModel;
  hiddenLayerSizes: number[];
  metadataProps?: OnnxMetadataProperty[];
};

/** Node slices for one sequential imported layer assignment pass. */
export type OnnxImportLayerNodePair = {
  sequentialIndex: number;
  layerIndex: number;
  currentLayerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Build params for one sequential layer node-pair slice operation. */
export type OnnxImportLayerNodePairBuildParams = {
  layerIndex: number;
  sequentialIndex: number;
};

/** Weight tensor names for one imported layer index. */
export type OnnxImportLayerTensorNames = {
  weightTensorName: string;
  biasTensorName: string;
};

/** Context for assigning aggregated dense tensors for one layer. */
export type OnnxImportAggregatedLayerAssignmentContext = {
  initializerMap: Record<string, OnnxTensor>;
  nodePair: OnnxImportLayerNodePair;
};

/** Context for assigning per-neuron tensors for one layer. */
export type OnnxImportPerNeuronLayerAssignmentContext = {
  initializerMap: Record<string, OnnxTensor>;
  nodePair: OnnxImportLayerNodePair;
};

/** Context for assigning one aggregated dense target neuron row. */
export type OnnxImportAggregatedNeuronAssignmentContext = {
  previousLayerNodes: NeatapticNode[];
  targetNode: NeatapticNode;
  targetNodeIndex: number;
  aggregatedWeights: OnnxTensor;
  biasTensor: OnnxTensor;
};

/** Context for assigning one per-neuron imported target node. */
export type OnnxImportPerNeuronAssignmentContext = {
  previousLayerNodes: NeatapticNode[];
  targetNode: NeatapticNode;
  weightTensor: OnnxTensor;
  biasTensor: OnnxTensor;
};

/** Parsed Conv metadata payload used for optional reconstruction pass. */
export type OnnxImportConvMetadata = {
  convLayers: number[];
  convSpecs: Conv2DMapping[];
};

/** Context for reconstructing one Conv layer's imported connectivity. */
export type OnnxImportConvLayerContext = {
  onnx: OnnxModel;
  hiddenLayerSizes: number[];
  hiddenNodes: NeatapticNode[];
  inputNodes: NeatapticNode[];
  layerExportIndex: number;
  convSpec: Conv2DMapping;
};

/** Build params for creating one Conv reconstruction layer context. */
export type OnnxImportConvLayerContextBuildParams = {
  assignmentContext: OnnxImportWeightAssignmentContext;
  convMetadata: OnnxImportConvMetadata;
  layerExportIndex: number;
};

/** Resolved Conv initializer tensors and dimensions for one layer. */
export type OnnxImportConvTensorContext = {
  convWeightTensor: OnnxTensor;
  convBiasTensor: OnnxTensor;
  outChannels: number;
  inChannels: number;
  kernelHeight: number;
  kernelWidth: number;
};

/** Layer node slices used while applying Conv reconstruction assignments. */
export type OnnxImportConvNodeSlices = {
  layerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Coordinate for one Conv output neuron traversal position. */
export type OnnxImportConvOutputCoordinate = {
  outChannelIndex: number;
  outRowIndex: number;
  outColumnIndex: number;
};

/** Context for applying Conv weights/bias at one output coordinate. */
export type OnnxImportConvCoordinateAssignmentContext = {
  coordinate: OnnxImportConvOutputCoordinate;
  convSpec: Conv2DMapping;
  tensorContext: OnnxImportConvTensorContext;
  kernelCoordinates: OnnxConvKernelCoordinate[];
  layerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Inbound connection lookup map keyed by source node for one target neuron. */
export type OnnxImportInboundConnectionMap = Map<NeatapticNode, Connection>;

/** Context for assigning one concrete Conv kernel connection weight. */
export type OnnxImportConvKernelAssignmentContext = {
  tensorContext: OnnxImportConvTensorContext;
  convSpec: Conv2DMapping;
  coordinate: OnnxImportConvOutputCoordinate;
  inChannelIndex: number;
  kernelRowIndex: number;
  kernelColumnIndex: number;
  inboundConnectionMap: OnnxImportInboundConnectionMap;
  previousLayerNodes: NeatapticNode[];
};

/** Parsed architecture dimensions extracted from ONNX import graph payloads. */
export type OnnxImportArchitectureResult = {
  inputCount: number;
  outputCount: number;
  hiddenLayerSizes: number[];
};

/** Shared architecture extraction context with resolved graph dimensions. */
export type OnnxImportArchitectureContext = {
  inputShapeDimensions: OnnxDimension[];
  outputShapeDimensions: OnnxDimension[];
  initializers: OnnxTensor[];
  metadata: OnnxMetadataProperty[];
};

/** Loose ONNX shape-dimension record used by legacy import payload access. */
export type OnnxImportDimensionRecord = Record<string, number>;

/** Context for recurrent self-connection restoration from ONNX metadata and tensors. */
export type OnnxImportRecurrentRestorationContext = {
  hiddenLayerSizes: number[];
  metadata: OnnxMetadataProperty[];
};

/** Hidden-layer span payload with one-based layer numbering and global offset. */
export type OnnxImportHiddenLayerSpan = {
  layerNumber: number;
  hiddenLayerSize: number;
  hiddenStart: number;
};

/** Execution context for assigning one hidden-layer recurrent diagonal tensor. */
export type OnnxImportLayerConnectionContext = {
  onnx: OnnxModel;
  hiddenNodes: NeatapticNode[];
  span: OnnxImportHiddenLayerSpan;
};

/** Context for upserting one hidden node self-connection from recurrent weight. */
export type OnnxImportSelfConnectionUpsertContext = {
  node: NeatapticNode;
  recurrentWeight: number;
};

/** Parsed pooling metadata payload attached to imported network instances. */
export type OnnxImportPoolingMetadata = {
  layers: number[];
  specs: Pool2DMapping[];
};

/** Network instance augmented with optional imported ONNX pooling metadata. */
export type NetworkWithOnnxImportPooling = Network & {
  _onnxPooling?: OnnxImportPoolingMetadata;
};
