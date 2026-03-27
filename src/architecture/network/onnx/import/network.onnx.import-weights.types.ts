/**
 * Import-owned types for ONNX weight restoration and Conv reconstruction.
 *
 * This chapter explains the state carried through the importer's heaviest
 * restoration pass: hidden-size derivation, dense and per-neuron tensor
 * assignment, and the optional Conv metadata replay that maps flattened ONNX
 * initializers back onto runtime connections.
 *
 * Keeping these types near the weight importer makes the generated import README
 * read like a reconstruction guide instead of scattering the execution model
 * across the root compatibility barrel.
 *
 * Example:
 * ```ts
 * const assignmentContext: OnnxImportWeightAssignmentContext = {
 *   onnx,
 *   hiddenLayerSizes,
 *   metadataProps,
 *   initializerMap,
 *   sortedLayerIndices,
 *   inputNodes,
 *   hiddenNodes,
 *   outputNodes,
 * };
 * ```
 */

import Connection from '../../../connection';
import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  Conv2DMapping,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type { OnnxConvKernelCoordinate } from '../network.onnx.utils.types';

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

/** Context for applying Conv weights and bias at one output coordinate. */
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
