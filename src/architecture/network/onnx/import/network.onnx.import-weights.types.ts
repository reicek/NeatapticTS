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
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type { OnnxConvKernelCoordinate } from '../network.onnx.utils.types';

/** Bucketed ONNX dense/per-neuron tensors for one exported layer index, holding the aggregated and per-neuron lists. */
export type OnnxImportLayerWeightBucket = {
  aggregated?: OnnxTensor;
  perNeuron: OnnxTensor[];
};

/** Context for deriving hidden layer sizes from initializer tensors and metadata. */
export type OnnxImportHiddenSizeDerivationContext = {
  initializers: OnnxTensor[];
  metadataProps: OnnxMetadataProperty[];
};

/** Shared weight-assignment context built once per ONNX import, carrying model, layers, metadata, and initializer map. */
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

/** Build params for creating shared ONNX import weight-assignment context, supplying network, model, and hidden sizes. */
export type OnnxImportWeightAssignmentBuildParams = {
  network: Network;
  onnx: OnnxModel;
  hiddenLayerSizes: number[];
  metadataProps?: OnnxMetadataProperty[];
};

/** Node slices for one sequential imported layer assignment pass, carrying current and previous layer node lists. */
export type OnnxImportLayerNodePair = {
  sequentialIndex: number;
  layerIndex: number;
  currentLayerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Build params for one sequential layer node-pair slice operation, specifying layer index and sequential position. */
export type OnnxImportLayerNodePairBuildParams = {
  layerIndex: number;
  sequentialIndex: number;
};

/** Weight tensor names for one imported layer index, identifying weight and bias initializer name strings. */
export type OnnxImportLayerTensorNames = {
  weightTensorName: string;
  biasTensorName: string;
};

/** Context for assigning aggregated dense tensors for one layer, supplying the initializer map and layer node pair. */
export type OnnxImportAggregatedLayerAssignmentContext = {
  initializerMap: Record<string, OnnxTensor>;
  nodePair: OnnxImportLayerNodePair;
};

/** Context for assigning per-neuron tensors for one layer, supplying the initializer map and sequential layer node pair. */
export type OnnxImportPerNeuronLayerAssignmentContext = {
  initializerMap: Record<string, OnnxTensor>;
  nodePair: OnnxImportLayerNodePair;
};

/** Context for assigning one aggregated dense target neuron row, carrying previous nodes, target, and tensor refs. */
export type OnnxImportAggregatedNeuronAssignmentContext = {
  previousLayerNodes: NeatapticNode[];
  targetNode: NeatapticNode;
  targetNodeIndex: number;
  aggregatedWeights: OnnxTensor;
  biasTensor: OnnxTensor;
};

/** Context for assigning one per-neuron imported target node, carrying previous nodes and weight and bias tensors. */
export type OnnxImportPerNeuronAssignmentContext = {
  previousLayerNodes: NeatapticNode[];
  targetNode: NeatapticNode;
  weightTensor: OnnxTensor;
  biasTensor: OnnxTensor;
};

/** Parsed Conv metadata payload used for optional reconstruction pass, listing Conv layer indices and mapping specs. */
export type OnnxImportConvMetadata = {
  convLayers: number[];
  convSpecs: Conv2DMapping[];
};

/** Context object for reconstructing one Conv layer's imported connectivity weights. */
export type OnnxImportConvLayerContext = {
  onnx: OnnxModel;
  hiddenLayerSizes: number[];
  hiddenNodes: NeatapticNode[];
  inputNodes: NeatapticNode[];
  layerExportIndex: number;
  convSpec: Conv2DMapping;
  convSpecs: Conv2DMapping[];
  poolingSpecs: Pool2DMapping[];
};

/** Build params for creating one Conv reconstruction layer context, supplying assignment context and Conv metadata. */
export type OnnxImportConvLayerContextBuildParams = {
  assignmentContext: OnnxImportWeightAssignmentContext;
  convMetadata: OnnxImportConvMetadata;
  layerExportIndex: number;
};

/** Resolved Conv initializer tensors and dimensions for one layer, including channels, kernel height, and width. */
export type OnnxImportConvTensorContext = {
  convWeightTensor: OnnxTensor;
  convBiasTensor: OnnxTensor;
  outChannels: number;
  inChannels: number;
  kernelHeight: number;
  kernelWidth: number;
};

/** Layer node slices used while applying Conv reconstruction assignments, carrying target and previous layer nodes. */
export type OnnxImportConvNodeSlices = {
  layerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Source layout used when replaying Conv weights onto dense source nodes. */
export type OnnxImportConvSourceLayout = {
  channelStride: number;
  sourceHeight: number;
  sourceWidth: number;
};

/** Coordinate for one Conv output neuron traversal position, encoding output channel, row, and column indices. */
export type OnnxImportConvOutputCoordinate = {
  outChannelIndex: number;
  outRowIndex: number;
  outColumnIndex: number;
};

/** Context for applying Conv weights and bias at one output coordinate. */
export type OnnxImportConvCoordinateAssignmentContext = {
  coordinate: OnnxImportConvOutputCoordinate;
  convSpec: Conv2DMapping;
  sourceLayout: OnnxImportConvSourceLayout;
  tensorContext: OnnxImportConvTensorContext;
  kernelCoordinates: OnnxConvKernelCoordinate[];
  layerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
};

/** Inbound connection lookup map keyed by source node for one target neuron. */
export type OnnxImportInboundConnectionMap = Map<NeatapticNode, Connection>;

/** Context for assigning one concrete Conv kernel connection weight, carrying tensor context, coordinate, and channels. */
export type OnnxImportConvKernelAssignmentContext = {
  tensorContext: OnnxImportConvTensorContext;
  convSpec: Conv2DMapping;
  sourceLayout: OnnxImportConvSourceLayout;
  coordinate: OnnxImportConvOutputCoordinate;
  inChannelIndex: number;
  kernelRowIndex: number;
  kernelColumnIndex: number;
  inboundConnectionMap: OnnxImportInboundConnectionMap;
  previousLayerNodes: NeatapticNode[];
};
