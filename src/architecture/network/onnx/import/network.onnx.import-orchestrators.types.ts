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