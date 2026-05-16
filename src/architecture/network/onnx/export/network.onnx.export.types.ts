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
  OnnxAttribute,
  OnnxModel,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type {
  NodeInternals,
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
 * - `concatMappings`: opt one skipped source layer into the narrow same-family
 *   `Concat -> Gemm` merge subset with deterministic `previous_then_source`
 *   input order.
 * - `attentionMappings`: opt one target layer into the fixed-width same-family
 *   self-attention shadow subset.
 * - `precision`: opt into reduced-precision export. The current landed lane is
 *   `storage-fp16`, which packs eligible same-family dense and Conv weight or
 *   bias initializers into float16 storage and inserts deterministic
 *   `Cast -> float32` bridges so operator inputs stay type-consistent.
 * - `quantization`: declare an explicit quantization request packet. The
 *   current exporter can validate static calibration contracts, emit
 *   deterministic scale or zero-point parameter initializers for the supported
 *   same-family dense and explicit Conv subset, and lower explicitly targeted
 *   same-family dense layers into a
 *   `QuantizeLinear -> QLinearMatMul -> DequantizeLinear` path with an
 *   explicit float-domain bias bridge plus the exporter-owned unary
 *   activation node when present. Spatial and dynamic quantized lowering
 *   remains later Phase 7 work.
 * - `autoPromoteInferredConv`: upgrades heuristic Conv-like layers into real `Conv`
 *   emission only when the exporter can prove the dense weights already behave like a
 *   shared-kernel spatial layout, including the current conservative multi-channel and
 *   unpooled stacked-chain subsets, deeper single-channel post-pool chains whose
 *   pooled tensor shapes can be derived sequentially, and deeper pooled
 *   multi-channel chains when the pooled tensor shapes can be derived sequentially
 *   and the pooled source stays compact per channel. The only proven
 *   flatten-after-pool promotion path is the narrow final hidden-stage
 *   reshape-bridge subset. Earlier flattened pooled consumers and repeated
 *   flatten-bridge chains stay on the honest fallback path.
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
  /**
   * Promote heuristic Conv metadata into real Conv emission only when the inferred
   * layer passes the shared-kernel safety gate.
   *
    * The default remains metadata-only inference so unsupported or ambiguous spatial
    * layouts stay on the dense fallback path honestly. Promotion can reuse derived
    * post-pool shapes when the exporter can keep the tensor spatial and the pooled
    * source stays compact per channel, including deeper single-channel and deeper
    * pooled multi-channel chains. The currently proven flatten-after-pool surface is
    * narrower: a final hidden-stage Conv -> Pool -> Flatten bridge can still promote
    * when the exporter restores the derived pooled `[C,H,W]` shape with an explicit
    * reshape before the later Conv. Earlier flattened pooled consumers and repeated
    * flatten-bridge chains stay on the honest fallback path because later Conv
    * inference stops before inferred metadata or reshape bridges survive.
   */
  autoPromoteInferredConv?: boolean;
  validateConvSharing?: boolean;
  flattenAfterPooling?: boolean;
  concatMappings?: ConcatMapping[];
  attentionMappings?: AttentionMapping[];
  precision?: OnnxPrecisionOptions;
  quantization?: OnnxQuantizationOptions;
}

/** Opt-in reduced-precision export controls for the Phase 7 storage lane. */
export type OnnxPrecisionOptions = {
  mode?: 'float32' | 'storage-fp16';
  metadata?: boolean;
};

/** External calibration packet declaration for static quantization requests. */
export type OnnxQuantizationCalibrationRange = {
  min: number;
  max: number;
};

/** One explicitly calibrated layer target used to build deterministic parameter tensors. */
export type OnnxQuantizationCalibrationLayerTarget = {
  target: 'dense' | 'conv';
  layerIndex: number;
  inputRange: OnnxQuantizationCalibrationRange;
  outputRange: OnnxQuantizationCalibrationRange;
};

/** Supported weight-range reduction policy for the first calibration contract. */
export type OnnxQuantizationCalibrationWeightRangePolicy = 'min-max';

/** Zero-inclusion policy for exported calibration parameters. */
export type OnnxQuantizationCalibrationZeroInclusionPolicy = 'required';

/** Symmetry policy for activation and weight quantization parameters. */
export type OnnxQuantizationCalibrationSymmetry = 'symmetric' | 'asymmetric';

/** Rounding policy for deterministic zero-point resolution. */
export type OnnxQuantizationCalibrationRoundingMode = 'nearest-even';

/** External calibration packet declaration for static quantization requests. */
export type OnnxQuantizationCalibrationOptions = {
  source: 'external';
  packetId?: string;
  sampleCount?: number;
  layerTargets: OnnxQuantizationCalibrationLayerTarget[];
  weightRangePolicy?: OnnxQuantizationCalibrationWeightRangePolicy;
  zeroInclusion?: OnnxQuantizationCalibrationZeroInclusionPolicy;
  activationSymmetry?: OnnxQuantizationCalibrationSymmetry;
  weightSymmetry?: OnnxQuantizationCalibrationSymmetry;
  roundingMode?: OnnxQuantizationCalibrationRoundingMode;
};

/** Resolved calibration packet with exporter-owned defaults applied. */
export type OnnxResolvedQuantizationCalibrationOptions = {
  source: 'external';
  packetId?: string;
  sampleCount?: number;
  layerTargets: OnnxQuantizationCalibrationLayerTarget[];
  weightRangePolicy: OnnxQuantizationCalibrationWeightRangePolicy;
  zeroInclusion: OnnxQuantizationCalibrationZeroInclusionPolicy;
  activationSymmetry: OnnxQuantizationCalibrationSymmetry;
  weightSymmetry: OnnxQuantizationCalibrationSymmetry;
  roundingMode: OnnxQuantizationCalibrationRoundingMode;
};

/** Static 8-bit quantization request packet for the narrow first Phase 7 lane. */
export type OnnxStaticQuantizationOptions = {
  mode: 'static-8bit';
  targets: Array<'dense' | 'conv'>;
  calibration: OnnxQuantizationCalibrationOptions;
  activationEncoding?: 'uint8' | 'int8';
  weightEncoding?: 'uint8' | 'int8';
  activationGranularity?: 'per-tensor';
  weightGranularity?: 'per-tensor' | 'per-output-channel';
  representation?: 'qlinear' | 'qdq';
};

/** Dynamic uint8 quantization request packet for supported dense guidance only. */
export type OnnxDynamicQuantizationOptions = {
  mode: 'dynamic-uint8';
  target?: 'dense';
  representation?: 'DynamicQuantizeLinear' | 'metadata-only';
};

/** Supported quantization request packets for the narrow first Phase 7 lane. */
export type OnnxQuantizationOptions =
  | OnnxStaticQuantizationOptions
  | OnnxDynamicQuantizationOptions;

/** Resolved reduced-precision packet used by build orchestration. */
export type OnnxResolvedPrecisionOptions = {
  requested: boolean;
  mode: 'float32' | 'storage-fp16';
  metadata: boolean;
};

/** Resolved quantization packet used by build orchestration. */
export type OnnxResolvedQuantizationOptions =
  | {
      requested: false;
      mode: null;
      fallbackReasons: string[];
    }
  | {
      requested: true;
      mode: 'static-8bit';
      targets: Array<'dense' | 'conv'>;
      calibration: OnnxResolvedQuantizationCalibrationOptions;
      activationEncoding: 'uint8' | 'int8';
      weightEncoding: 'uint8' | 'int8';
      activationGranularity: 'per-tensor';
      weightGranularity: 'per-tensor' | 'per-output-channel';
      representation: 'qlinear' | 'qdq';
      fallbackReasons: string[];
    }
  | {
      requested: true;
      mode: 'dynamic-uint8';
      target: 'dense';
      representation: 'DynamicQuantizeLinear' | 'metadata-only';
      fallbackReasons: string[];
    };

/**
 * Explicit export-only concat mapping for the narrow Phase 5 merge subset.
 *
 * This contract keeps concat source-owned instead of inferred: callers name one
 * skipped source layer and one target layer, and export preserves the merge as a
 * deterministic `Concat -> Gemm` path with the default adjacent-layer slice kept
 * first in the merged input order.
 */
export interface ConcatMapping {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  inputOrder?: 'previous_then_source';
}

/**
 * Explicit export-only attention mapping for the Phase 5E shadow subset.
 *
 * This contract keeps attention source-owned rather than heuristic: callers
 * opt one target layer into a fixed-width self-attention shadow block while the
 * stable dense path remains the canonical runtime behavior.
 */
export interface AttentionMapping {
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
  scaleScores?: boolean;
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
  availableConvSpecsByLayerIndex: Map<number, Conv2DMapping>;
  poolMappingsByAfterLayerIndex: Map<number, Pool2DMapping>;
  flattenAfterPooling: boolean | undefined;
  totalHiddenLayerCount: number;
};

/** Width and shape evaluation context used by Conv inference helpers. */
export type ConvInferenceEvaluationContext = {
  layerIndex: number;
  currentWidth: number;
  inputChannels: number;
  inputHeight: number;
  inputWidth: number;
  allowsExactFitKernel: boolean;
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
  precision: OnnxResolvedPrecisionOptions;
  quantization: OnnxResolvedQuantizationOptions;
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
  layerOutputNamesByLayerIndex: Map<number, string>;
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
  layerOutputNamesByLayerIndex: Map<number, string>;
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
  opset: number;
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
  opset: number;
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
  hasLaterHiddenLayers: boolean;
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
  activationAttributes?: OnnxAttribute[];
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

/** Parameters for one-hop residual-add dense layer emission. */
export type ResidualAddLayerParams = {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  residualSourceOutputName: string;
  previousLayerNodes: NeatapticNode[];
  residualSourceLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  branchTensorName: string;
  mergeNodeName: string;
  mergeOutputName: string;
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
  opset: number;
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
  attributes?: OnnxAttribute[];
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
  opset: number;
};

/** Per-neuron normalized node context. */
export type PerNeuronNodeContext = {
  model: OnnxModel;
  layerIndex: number;
  neuronIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  targetNodeInternal: NodeInternals;
  opset: number;
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
