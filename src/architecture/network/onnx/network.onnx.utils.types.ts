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
 * - The importer-facing schema is JSON-first and may evolve; prefer
 *   re-exporting/importing through the library rather than hand-editing blobs,
 *   and use `exportToONNXBinary()` when you need the primary runtime-validated
 *   artifact for the approved subset.
 */

import Connection from '../../connection';
import type NeatapticNode from '../../node';
import type { OnnxExportOptions } from './export/network.onnx.export.types';
import type {
  OnnxShape as OnnxShapeContract,
  OnnxTensorType as OnnxTensorTypeContract,
  OnnxValueInfo as OnnxValueInfoContract,
} from './schema/network.onnx.schema.types';
import type { OnnxImportConvLayerContext as OnnxImportConvLayerContextContract } from './import/network.onnx.import-weights.types';

// Schema re-exports expose persisted wire-format contracts.
export type {
  Conv2DMapping,
  OnnxAttribute,
  OnnxDimension,
  OnnxGraph,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxNode,
  OnnxTensor,
  Pool2DMapping,
} from './schema/network.onnx.schema.types';

// Export re-exports expose exporter-owned orchestration payloads.
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

// Import re-exports expose reconstruction-only payload and restoration contracts.
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
 * Canonical shape descriptor for ONNX tensors used by export, import, and schema validation paths.
 * Each entry preserves axis intent so runtime bridges can validate rank-sensitive operators without guessing dimension semantics.
 */
export type OnnxShape = OnnxShapeContract;

/**
 * Canonical tensor element type shelf used by schema, import coercion, and export metadata emission.
 * Keep this alias at the ONNX root so callers can depend on one stable type name while chapter ownership remains in schema contracts.
 */
export type OnnxTensorType = OnnxTensorTypeContract;

/**
 * Canonical tensor value-info descriptor used to name and type graph inputs, outputs, and intermediate values.
 * This alias keeps metadata surfaces consistent across ONNX schema parsing, importer reconstruction, and exporter graph emission.
 */
export type OnnxValueInfo = OnnxValueInfoContract;

/**
 * Context payload used when rebuilding one imported convolution layer from ONNX graph metadata and tensor shelves.
 * The contract captures grouped node slices, tensor mappings, and assignment state so reconstruction stays deterministic across import passes.
 */
export type OnnxImportConvLayerContext = OnnxImportConvLayerContextContract;

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

/** Runtime node internals augmented with optional export index metadata, used for deterministic ONNX graph ordering. */
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

/** Node partitions used by ONNX layered-ordering inference traversal, grouping input, hidden, and output nodes. */
export type LayerOrderingNodeGroups = {
  inputNodes: NeatapticNode[];
  hiddenNodes: NeatapticNode[];
  outputNodes: NeatapticNode[];
};

/** Mutable traversal state while resolving hidden-layer ordering, carrying remaining nodes and accumulated layer groups. */
export type LayerOrderingResolutionContext = {
  remainingHiddenNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  orderedLayers: NeatapticNode[][];
};

/** Layer-wise validation context for activation and connectivity checks, supplying layer index and adjacent node lists. */
export type LayerValidationTraversalContext = {
  layerIndex: number;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  options: OnnxExportOptions;
};

/** Activation-homogeneity decision context for one current layer, capturing activation names and mixed-activation policy. */
export type LayerActivationValidationContext = {
  layerIndex: number;
  activationNames: (string | undefined)[];
  allowMixedActivations: boolean;
};

/** Connectivity decision context for one source-target node pair, including layer index and partial-connectivity policy. */
export type LayerConnectivityValidationContext = {
  layerIndex: number;
  sourceNode: NeatapticNode;
  targetNode: NeatapticNode;
  allowPartialConnectivity: boolean;
};

/** Supported ONNX activation operator strings recognized and mapped during network activation import traversal. */
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

/** Layer-indexed activation operator lookup extracted from ONNX graph nodes for import assignment passes. */
export type OnnxActivationLayerOperations = Record<
  number,
  OnnxActivationOperation[]
>;

/** Parsed ONNX activation-node naming payload, carrying the extracted layer index and optional neuron index. */
export type OnnxActivationParseResult = {
  layerIndex: number;
  neuronIndex?: number;
};

/** Shared activation-assignment context for hidden and output traversal, grouping node lists and per-layer operations. */
export type OnnxActivationAssignmentContext = {
  hiddenLayerSizes: number[];
  hiddenNodes: NodeInternals[];
  outputNodes: NodeInternals[];
  operationsByLayer: OnnxActivationLayerOperations;
};

/** Hidden-layer traversal context for assigning imported activation functions, carrying layer index, size, and node lists. */
export type HiddenLayerActivationTraversalContext = {
  hiddenLayerIndex: number;
  hiddenLayerSize: number;
  hiddenOffset: number;
  hiddenNodes: NodeInternals[];
  operationsByLayer: OnnxActivationLayerOperations;
};

/** Output-layer activation assignment context, carrying output layer index, node list, and activation operations map. */
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

/** Coordinate for one Conv kernel weight lookup, encoding input channel index, kernel row, and column position. */
export type OnnxConvKernelCoordinate = {
  inChannelIndex: number;
  kernelRowIndex: number;
  kernelColumnIndex: number;
};

/** Runtime factory map used to construct dynamic recurrent layer modules. */
export type OnnxLayerFactory = Record<string, (...args: unknown[]) => unknown>;

/** Runtime layer module shape widened for fused-recurrent reconstruction wiring and dynamic layer factory dispatch. */
export type OnnxRuntimeLayerFactoryMap =
  import('./import/network.onnx.runtime-load.types').OnnxRuntimeLayerModule &
    OnnxLayerFactory;
