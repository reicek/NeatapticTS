/**
 * Import-owned type surface for ONNX architecture reconstruction orchestration.
 *
 * These payloads stay close to the orchestration helpers that parse terminal
 * dimensions, restore recurrent self-connections, and attach pooling metadata.
 * Keeping them here makes the import chapter explain its own execution state
 * without forcing the root ONNX compatibility barrel to remain the ownership
 * home for importer-only details.
 */

import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  OnnxDimension,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';

/** Parsed architecture dimensions extracted from ONNX import graph payloads, with input, output, and hidden sizes. */
export type OnnxImportArchitectureResult = {
  inputCount: number;
  outputCount: number;
  hiddenLayerSizes: number[];
};

/** Shared architecture extraction context with resolved graph dimensions, initializers, and metadata properties. */
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
  onnx: OnnxModel;
};

/** Hidden-layer span payload with one-based layer numbering and global offset. */
export type OnnxImportHiddenLayerSpan = {
  layerNumber: number;
  hiddenLayerSize: number;
  hiddenStart: number;
};

/** Execution context for assigning one hidden-layer recurrent diagonal tensor, carrying model, nodes, and span. */
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

/** Virtual spatial shape derived from Conv and Pool metadata during import. */
export type OnnxImportPoolingVirtualShape = {
  afterLayerIndex: number;
  inputHeight: number;
  inputWidth: number;
  inputChannels: number;
  outputHeight: number;
  outputWidth: number;
  outputChannels: number;
  flattenedSize?: number;
};

/** Metadata-only audit record comparing a flattened pooled width to the next dense width. */
export type OnnxImportFlattenConsistencyAudit = {
  afterLayerIndex: number;
  consumerLayerIndex: number;
  consumerWidth: number;
  flattenedSize: number;
  matches: boolean;
};

/** Parsed pooling metadata payload attached to imported network instances, listing pool specs and virtual shapes. */
export type OnnxImportPoolingMetadata = {
  layers: number[];
  specs: Pool2DMapping[];
  flattenLayers: number[];
  virtualShapes: OnnxImportPoolingVirtualShape[];
  flattenConsistency?: OnnxImportFlattenConsistencyAudit[];
};

/** Audit-only cross-layer feed-forward edge carried through Phase 5 import fallback. */
export type OnnxImportAdvancedGraphCrossLayerConnection = {
  sourceNodeIndex: number;
  sourceLayerIndex: number;
  targetNodeIndex: number;
  targetLayerIndex: number;
  branchTensorName: string;
};

/** Audit-only shared initializer alias carried through Phase 5 import fallback. */
export type OnnxImportSharedInitializerAlias = {
  aliasTensorName: string;
  canonicalTensorName: string;
  initializerKind: string;
};

/** Explicit one-hop residual-add merge carried through Phase 5 import hardening. */
export type OnnxImportResidualAdd = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  branchTensorName: string;
  mergeNodeName: string;
  mergeOutputName: string;
};

/** Explicit concat merge carried through Phase 5 import hardening, identifying layer indices and merge tensor names. */
export type OnnxImportConcatMerge = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  concatNodeName: string;
  concatOutputName: string;
  inputOrder: 'previous_then_source';
};

/** Explicit fixed-width self-attention block carried through Phase 5 import fallback. */
export type OnnxImportAttentionBlock = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  sequenceLength: number;
  modelWidth: number;
  heads: number;
  shadowOutputName: string;
};

/** Parsed advanced-graph metadata attached to imported network instances, grouping merges, residual adds, and blocks. */
export type OnnxImportAdvancedGraphMetadata = {
  crossLayerConnections?: OnnxImportAdvancedGraphCrossLayerConnection[];
  concatMerges?: OnnxImportConcatMerge[];
  residualAdds?: OnnxImportResidualAdd[];
  sharedInitializerAliases?: OnnxImportSharedInitializerAlias[];
  attentionBlocks?: OnnxImportAttentionBlock[];
};

/** Network instance augmented with optional imported ONNX pooling metadata via the _onnxPooling field. */
export type NetworkWithOnnxImportPooling = Network & {
  _onnxPooling?: OnnxImportPoolingMetadata;
};

/** Network instance augmented with optional imported advanced-graph metadata via the _onnxAdvancedGraph field. */
export type NetworkWithOnnxImportAdvancedGraph = Network & {
  _onnxAdvancedGraph?: OnnxImportAdvancedGraphMetadata;
};
