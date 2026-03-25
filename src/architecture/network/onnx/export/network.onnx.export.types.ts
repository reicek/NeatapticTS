/**
 * Export-owned execution and payload types for NeatapticTS ONNX serialization.
 *
 * This chapter holds the types that belong to the exporter implementation:
 * export options, build/setup contexts, recurrent and Conv heuristics, and the
 * dense/per-neuron layer-emission payloads used by the export helpers.
 *
 * Shared runtime bridge types such as `NodeInternals` remain root-owned in
 * `network.onnx.utils.types.ts`, and the persisted wire-format schema stays in
 * `schema/network.onnx.schema.types.ts`.
 *
 * ```mermaid
 * flowchart LR
 *   Options[OnnxExportOptions] --> Setup[Setup and build contexts]
 *   Setup --> Heuristics[Conv and recurrent heuristics]
 *   Heuristics --> Layers[Layer emission payloads]
 *   Layers --> Model[Schema model and tensors]
 * ```
 */

import type NeatapticNode from '../../../node';
import type {
  Conv2DMapping,
  OnnxDimension,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxValueInfo,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type {
  NodeInternals,
  NodeInternalsWithExportIndex,
  OnnxConvKernelCoordinate,
} from '../network.onnx.utils.types';

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