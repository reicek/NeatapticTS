/**
 * Types for NeatapticTS’s ONNX-like JSON export/import.
 *
 * This file is now the root compatibility barrel for shared ONNX type surfaces.
 * Most exporter-owned, importer-owned, and schema-owned type families have
 * already moved into their chapter-local files. What remains here is the
 * narrow bridge layer that still needs to be visible from the root ONNX API.
 *
 * How to read this type surface:
 * - Start with `schema/` if you want the persisted wire-format document model.
 * - Continue into `export/` for graph-emission contexts and layer payloads.
 * - Continue into `import/` for reconstruction-only contexts.
 * - Use this root file when you specifically need shared runtime bridge types,
 *   compatibility re-exports, or the small set of contracts that still span
 *   multiple ONNX chapters.
 *
 * What still belongs here:
 * - Root re-exports that preserve public ergonomics while the underlying
 *   ownership lives in `schema/`, `export/`, or `import/`.
 * - Shared bridge types such as `NodeInternals`, activation-assignment
 *   contracts, and the runtime layer-factory widening used across chapters.
 * - Transitional compatibility groupings that are still safer to keep at the
 *   root until a later cleanup proves they can move without widening seams.
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
import type NeatapticNode from '../../node';
import type { OnnxExportOptions } from './export/network.onnx.export.types';
export type {
  Conv2DMapping,
  OnnxAttribute,
  OnnxDimension,
  OnnxGraph,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxNode,
  OnnxShape,
  OnnxTensor,
  OnnxTensorType,
  OnnxValueInfo,
  Pool2DMapping,
} from './schema/network.onnx.schema.types';
export type {
  AttentionMapping,
  ConcatMapping,
  ActivationSquashFunction,
  ConvKernelConsistencyContext,
  ConvLayerPairContext,
  ConvOutputCoordinate,
  ConvRepresentativeKernelContext,
  ConvSharingValidationContext,
  ConvSharingValidationResult,
  DenseActivationContext,
  DenseActivationNodePayload,
  DenseGemmNodePayload,
  DenseGraphNames,
  DenseInitializerValues,
  DenseLayerContext,
  DenseLayerParams,
  DenseOrderedNodePayload,
  DenseTensorNames,
  DenseWeightBuildContext,
  DenseWeightBuildResult,
  DenseWeightRow,
  DenseWeightRowCollectionContext,
  DiagonalRecurrentBuildContext,
  FlattenAfterPoolingContext,
  FusedRecurrentEmissionExecutionContext,
  FusedRecurrentGraphNames,
  FusedRecurrentInitializerNames,
  GruEmissionContext,
  HiddenLayerHeuristicContext,
  IndexedMetadataAppendContext,
  LayerActivationContext,
  LayerBuildContext,
  LayerRecurrentDecisionContext,
  LayerTraversalContext,
  LstmEmissionContext,
  OnnxBaseModelBuildContext,
  OnnxBuildResolvedOptions,
  OnnxConvEmissionContext,
  OnnxConvEmissionParams,
  OnnxConvParameters,
  OnnxConvTensorNames,
  OnnxExportOptions,
  OnnxGraphDimensionBuildContext,
  OnnxGraphDimensions,
  OnnxLayerEmissionContext,
  OnnxLayerEmissionResult,
  OnnxModelMetadataContext,
  OnnxPostProcessingContext,
  OnnxRecurrentCollectionContext,
  OnnxRecurrentInputValueInfoContext,
  OnnxRecurrentLayerProcessingContext,
  OnnxRecurrentLayerTraversalContext,
  OptionalLayerOutputParams,
  OptionalPoolingAndFlattenParams,
  PerNeuronConcatNodePayload,
  PerNeuronGraphNames,
  PerNeuronLayerContext,
  PerNeuronLayerParams,
  PerNeuronNodeContext,
  PerNeuronSubgraphContext,
  PerNeuronTensorNames,
  PoolingAttributes,
  PoolingEmissionContext,
  RecurrentActivationEmissionContext,
  RecurrentGateBlockCollectionContext,
  RecurrentGateParameterCollectionResult,
  RecurrentGateRow,
  RecurrentGateRowCollectionContext,
  RecurrentGemmEmissionContext,
  RecurrentGraphNames,
  RecurrentHeuristicEmissionContext,
  RecurrentInitializerEmissionContext,
  RecurrentInitializerNames,
  RecurrentInitializerValues,
  RecurrentLayerEmissionContext,
  RecurrentLayerEmissionParams,
  RecurrentRowCollectionContext,
  SharedActivationNodeBuildParams,
  SharedGemmNodeBuildParams,
  SpecMetadataAppendContext,
  WeightToleranceComparisonContext,
} from './export/network.onnx.export.types';
export type {
  OnnxImportAttentionBlock,
  OnnxImportAdvancedGraphCrossLayerConnection,
  OnnxImportConcatMerge,
  OnnxImportAdvancedGraphMetadata,
  OnnxImportResidualAdd,
  OnnxImportSharedInitializerAlias,
  OnnxImportFlattenConsistencyAudit,
  NetworkWithOnnxImportPooling,
  NetworkWithOnnxImportAdvancedGraph,
  OnnxImportArchitectureContext,
  OnnxImportArchitectureResult,
  OnnxImportDimensionRecord,
  OnnxImportHiddenLayerSpan,
  OnnxImportLayerConnectionContext,
  OnnxImportPoolingMetadata,
  OnnxImportPoolingVirtualShape,
  OnnxImportRecurrentRestorationContext,
  OnnxImportSelfConnectionUpsertContext,
} from './import/network.onnx.import-orchestrators.types';
export type {
  OnnxImportAggregatedLayerAssignmentContext,
  OnnxImportAggregatedNeuronAssignmentContext,
  OnnxImportConvCoordinateAssignmentContext,
  OnnxImportConvKernelAssignmentContext,
  OnnxImportConvLayerContext,
  OnnxImportConvLayerContextBuildParams,
  OnnxImportConvMetadata,
  OnnxImportConvNodeSlices,
  OnnxImportConvOutputCoordinate,
  OnnxImportConvTensorContext,
  OnnxImportHiddenSizeDerivationContext,
  OnnxImportInboundConnectionMap,
  OnnxImportLayerNodePair,
  OnnxImportLayerNodePairBuildParams,
  OnnxImportLayerTensorNames,
  OnnxImportLayerWeightBucket,
  OnnxImportPerNeuronAssignmentContext,
  OnnxImportPerNeuronLayerAssignmentContext,
  OnnxImportWeightAssignmentBuildParams,
  OnnxImportWeightAssignmentContext,
} from './import/network.onnx.import-weights.types';
export type {
  OnnxFusedGateApplicationContext,
  OnnxFusedGateRowAssignmentContext,
  OnnxFusedLayerNeighborhood,
  OnnxFusedLayerReconstructionContext,
  OnnxFusedLayerRuntime,
  OnnxFusedRecurrentKind,
  OnnxFusedRecurrentSpec,
  OnnxFusedTensorPayload,
  OnnxIncomingWeightAssignmentContext,
} from './import/network.onnx.import-fused-recurrent.types';
export type {
  OnnxPerceptronBuildContext,
  OnnxPerceptronSizeValidationContext,
  OnnxRuntimeFactories,
  OnnxRuntimeLayerFactory,
  OnnxRuntimeLayerModule,
  OnnxRuntimePerceptronFactory,
} from './import/network.onnx.runtime-load.types';

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
  | 'Identity'
  | 'Softplus'
  | 'Softsign'
  | 'Selu'
  | 'Mish'
  | 'Gelu';

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

/** Coordinate for one Conv kernel weight lookup. */
export type OnnxConvKernelCoordinate = {
  inChannelIndex: number;
  kernelRowIndex: number;
  kernelColumnIndex: number;
};

/** Runtime factory map used to construct dynamic recurrent layer modules. */
export type OnnxLayerFactory = Record<string, (...args: unknown[]) => unknown>;

/** Runtime layer module shape widened for fused-recurrent reconstruction wiring. */
export type OnnxRuntimeLayerFactoryMap =
  import('./import/network.onnx.runtime-load.types').OnnxRuntimeLayerModule &
    OnnxLayerFactory;
