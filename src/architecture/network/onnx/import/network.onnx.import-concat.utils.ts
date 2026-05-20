import Connection from '../../../connection';
import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type {
  NetworkWithOnnxImportAdvancedGraph,
  OnnxImportConcatMerge,
} from './network.onnx.import-orchestrators.types';
import type { NodeInternals } from '../network.onnx.utils.types';
import { readOnnxTensorFloatData } from '../schema/network.onnx.schema.tensor-data.utils';

const METADATA_KEY_ADVANCED_GRAPH_CONCAT_MERGES =
  'advanced_graph_concat_merges';
const NODE_TYPE_INPUT = 'input';
const NODE_TYPE_HIDDEN = 'hidden';
const NODE_TYPE_OUTPUT = 'output';
const HIDDEN_LAYER_NUMBER_OFFSET = 1;
const ZERO_VALUE = 0;

/**
 * Restore supported concat-merge skip connections from ONNX metadata.
 *
 * The import side stays narrow and deterministic: it only rebuilds skipped
 * source-layer fan-in for the explicit concat subset emitted by the exporter,
 * using the widened dense weight tensor tail while the ordinary adjacent-layer
 * slice remains assigned by the baseline dense import path.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes from architecture extraction.
 * @param metadata ONNX metadata payload.
 * @returns Nothing.
 */
export function restoreConcatMergeConnections(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void {
  const concatMerges = parseConcatMergeMetadata(metadata, onnx);
  if (!concatMerges) {
    return;
  }

  const importedLayers = buildImportedLayers(network, hiddenLayerSizes);
  concatMerges.forEach((concatMerge) => {
    restoreSingleConcatMergeConnections(onnx, importedLayers, concatMerge);
  });
}

/**
 * Attach validated concat-merge audit metadata onto an imported network.
 *
 * @param network Target network.
 * @param metadata ONNX metadata payload.
 * @param onnx Source ONNX model used for same-family validation.
 * @returns Nothing.
 */
export function attachOnnxConcatMergeMetadata(
  network: Network,
  metadata: OnnxMetadataProperty[],
  onnx?: OnnxModel,
): void {
  const concatMerges = onnx ? parseConcatMergeMetadata(metadata, onnx) : null;
  if (!concatMerges) {
    return;
  }

  const networkWithOnnxAdvancedGraph =
    network as NetworkWithOnnxImportAdvancedGraph;
  networkWithOnnxAdvancedGraph._onnxAdvancedGraph = {
    ...(networkWithOnnxAdvancedGraph._onnxAdvancedGraph ?? {}),
    concatMerges,
  };
}

/** Parse valid explicit concat-merge metadata. */
function parseConcatMergeMetadata(
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel,
): OnnxImportConcatMerge[] | null {
  const metadataProperty = findMetadataProperty(
    metadata,
    METADATA_KEY_ADVANCED_GRAPH_CONCAT_MERGES,
  );
  if (!metadataProperty) {
    return null;
  }

  try {
    const parsedConcatMerges = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedConcatMerges)) {
      return null;
    }

    const concatMerges = parsedConcatMerges.filter(isConcatMerge);
    if (concatMerges.length !== parsedConcatMerges.length) {
      return null;
    }

    return concatMerges.every((concatMerge) =>
      matchesExportedConcatMergeSubset(onnx, concatMerge),
    )
      ? concatMerges
      : null;
  } catch {
    return null;
  }
}

/** Find one ONNX metadata property by key. */
function findMetadataProperty(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): OnnxMetadataProperty | undefined {
  return metadata.find((property) => property.key === metadataKey);
}

/** Validate one parsed concat-merge record. */
function isConcatMerge(value: unknown): value is OnnxImportConcatMerge {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<OnnxImportConcatMerge>;
  return (
    typeof candidate.sourceLayerIndex === 'number' &&
    typeof candidate.targetLayerIndex === 'number' &&
    typeof candidate.concatNodeName === 'string' &&
    typeof candidate.concatOutputName === 'string' &&
    candidate.inputOrder === 'previous_then_source'
  );
}

/** Validate the same-family explicit concat subset emitted by this exporter. */
function matchesExportedConcatMergeSubset(
  onnx: OnnxModel,
  concatMerge: OnnxImportConcatMerge,
): boolean {
  const concatNode = findNodeByName(onnx, concatMerge.concatNodeName);
  const denseWeightTensor = findInitializer(
    onnx,
    `W${concatMerge.targetLayerIndex - HIDDEN_LAYER_NUMBER_OFFSET}`,
  );

  return Boolean(
    concatMerge.sourceLayerIndex < concatMerge.targetLayerIndex - 1 &&
    concatNode?.op_type === 'Concat' &&
    concatNode.output[0] === concatMerge.concatOutputName &&
    concatNode.input.length === 2 &&
    denseWeightTensor &&
    denseWeightTensor.dims.length === 2,
  );
}

/** Resolve one initializer by tensor name. */
function findInitializer(
  onnx: OnnxModel,
  tensorName: string,
): OnnxTensor | undefined {
  return onnx.graph.initializer.find(
    (initializerEntry) => initializerEntry.name === tensorName,
  );
}

/** Resolve one node by its deterministic export name. */
function findNodeByName(onnx: OnnxModel, nodeName: string) {
  return onnx.graph.node.find((nodeEntry) => nodeEntry.name === nodeName);
}

/** Build the imported layer ordering from runtime nodes and hidden-layer widths. */
function buildImportedLayers(
  network: Network,
  hiddenLayerSizes: number[],
): NeatapticNode[][] {
  const inputNodes = collectNodesByType(network.nodes, NODE_TYPE_INPUT);
  const hiddenNodes = collectNodesByType(network.nodes, NODE_TYPE_HIDDEN);
  const outputNodes = collectNodesByType(network.nodes, NODE_TYPE_OUTPUT);
  const hiddenLayers = hiddenLayerSizes.reduce(
    (state, hiddenLayerSize) => {
      const nextOffset = state.hiddenOffset + hiddenLayerSize;
      return {
        hiddenLayers: [
          ...state.hiddenLayers,
          hiddenNodes.slice(state.hiddenOffset, nextOffset),
        ],
        hiddenOffset: nextOffset,
      };
    },
    {
      hiddenLayers: [] as NeatapticNode[][],
      hiddenOffset: 0,
    },
  ).hiddenLayers;

  return [inputNodes, ...hiddenLayers, outputNodes];
}

/** Collect nodes matching one runtime node-type discriminator. */
function collectNodesByType(
  nodes: NeatapticNode[],
  nodeType: 'input' | 'hidden' | 'output',
): NeatapticNode[] {
  return nodes.filter((nodeItem) => nodeItem.type === nodeType);
}

/** Restore one concat-merge branch using the widened dense weight tensor tail. */
function restoreSingleConcatMergeConnections(
  onnx: OnnxModel,
  importedLayers: NeatapticNode[][],
  concatMerge: OnnxImportConcatMerge,
): void {
  const sourceLayerNodes = importedLayers[concatMerge.sourceLayerIndex];
  const previousLayerNodes = importedLayers[concatMerge.targetLayerIndex - 1];
  const targetLayerNodes = importedLayers[concatMerge.targetLayerIndex];
  const denseWeightTensor = onnx.graph.initializer.find(
    (tensor) =>
      tensor.name ===
      `W${concatMerge.targetLayerIndex - HIDDEN_LAYER_NUMBER_OFFSET}`,
  );
  if (
    !sourceLayerNodes ||
    !previousLayerNodes ||
    !targetLayerNodes ||
    !denseWeightTensor ||
    concatMerge.inputOrder !== 'previous_then_source'
  ) {
    return;
  }

  const expectedMergedWidth =
    previousLayerNodes.length + sourceLayerNodes.length;
  if (denseWeightTensor.dims.at(-1) !== expectedMergedWidth) {
    return;
  }

  const denseWeightValues = readOnnxTensorFloatData(denseWeightTensor);

  targetLayerNodes.forEach((targetNode, targetNodeIndex) => {
    sourceLayerNodes.forEach((sourceNode, sourceNodeIndex) => {
      const sourceWeight = resolveConcatMergeWeight(
        denseWeightValues,
        expectedMergedWidth,
        previousLayerNodes.length,
        targetNodeIndex,
        sourceNodeIndex,
      );
      upsertFeedForwardConnection(sourceNode, targetNode, sourceWeight);
    });
  });
}

/** Resolve one concat branch weight from the widened row-major target-by-merged-source matrix. */
function resolveConcatMergeWeight(
  mergedWeights: number[],
  mergedSourceWidth: number,
  previousLayerWidth: number,
  targetLayerIndex: number,
  sourceLayerIndex: number,
): number {
  return (
    mergedWeights[
      targetLayerIndex * mergedSourceWidth +
        previousLayerWidth +
        sourceLayerIndex
    ] ?? ZERO_VALUE
  );
}

/** Upsert one feed-forward connection between two runtime nodes. */
function upsertFeedForwardConnection(
  sourceNode: NeatapticNode,
  targetNode: NeatapticNode,
  weight: number,
): void {
  const targetNodeInternal = targetNode as NodeInternals;
  const existingConnection = targetNodeInternal.connections.in.find(
    (connection) => connection.from === sourceNode,
  );
  if (existingConnection) {
    existingConnection.weight = weight;
    return;
  }

  const connection = Connection.acquire(sourceNode, targetNode, weight);
  sourceNode.connections.out.push(connection);
  targetNode.connections.in.push(connection);
}
