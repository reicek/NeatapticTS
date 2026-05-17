import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
} from '../schema/network.onnx.schema.types';
import type {
  NodeInternals,
  NodeInternalsWithExportIndex,
} from '../network.onnx.utils.types';

const ADVANCED_GRAPH_CROSS_LAYER_CONNECTIONS_KEY =
  'advanced_graph_cross_layer_connections';
const ADVANCED_GRAPH_RESIDUAL_ADDS_KEY = 'advanced_graph_residual_adds';
const ADVANCED_GRAPH_CONCAT_MERGES_KEY = 'advanced_graph_concat_merges';

type AdvancedGraphCrossLayerConnection = {
  sourceNodeIndex: number;
  sourceLayerIndex: number;
  targetNodeIndex: number;
  targetLayerIndex: number;
  branchTensorName: string;
};

type AdvancedGraphResidualAdd = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  branchTensorName: string;
  mergeNodeName: string;
  mergeOutputName: string;
};

type AdvancedGraphConcatMerge = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  concatNodeName: string;
  concatOutputName: string;
  inputOrder: 'previous_then_source';
};

/**
 * Append deterministic advanced-graph metadata for cross-layer feed-forward edges.
 *
 * Phase 5 starts by making skip-style ancestry visible instead of silently
 * dropping it. The current exporter still serializes adjacent-layer dense paths
 * only, so this metadata is an audit seam: it records the exact non-adjacent
 * feed-forward edges that later residual, concat, and attention passes can
 * promote into explicit ONNX graph structure.
 *
 * @param model Target ONNX model.
 * @param network Source network.
 * @param layers Resolved layered ordering.
 * @param includeMetadata Whether metadata emission is enabled.
 * @returns Nothing.
 */
export function appendAdvancedGraphMetadata(
  model: OnnxModel,
  network: Network,
  layers: NeatapticNode[][],
  includeMetadata: boolean,
): void {
  if (!includeMetadata) {
    return;
  }

  const crossLayerConnections = collectCrossLayerConnections(network, layers);
  if (!crossLayerConnections.length) {
    return;
  }

  appendMetadataProperty(
    model,
    buildMetadataProperty(
      ADVANCED_GRAPH_CROSS_LAYER_CONNECTIONS_KEY,
      crossLayerConnections,
    ),
  );
}

/**
 * Resolve the single one-hop residual source layer for a target layer.
 *
 * The first explicit residual-add subset stays narrow and deterministic:
 * exactly one non-adjacent source layer may feed the target layer, and that
 * source must skip exactly one intermediate layer.
 *
 * @param currentLayerNodes Target-layer nodes.
 * @param layers Resolved layered ordering.
 * @param targetLayerIndex Target-layer index.
 * @returns One-hop residual source layer index, or null when the layer stays on fallback.
 */
export function resolveOneHopResidualSourceLayerIndex(
  currentLayerNodes: NeatapticNode[],
  layers: NeatapticNode[][],
  targetLayerIndex: number,
): number | null {
  const layerIndexByNode = buildLayerIndexByNode(layers);
  const nonAdjacentSourceLayerIndices = Array.from(
    new Set(
      currentLayerNodes.flatMap((targetNode) => {
        const targetNodeInternal = targetNode as NodeInternals;
        return targetNodeInternal.connections.in.flatMap((connection) => {
          const sourceLayerIndex = layerIndexByNode.get(connection.from);
          if (
            sourceLayerIndex === undefined ||
            sourceLayerIndex >= targetLayerIndex - 1
          ) {
            return [];
          }
          return [sourceLayerIndex];
        });
      }),
    ),
  ).toSorted((leftLayerIndex, rightLayerIndex) => leftLayerIndex - rightLayerIndex);

  if (nonAdjacentSourceLayerIndices.length !== 1) {
    return null;
  }

  const sourceLayerIndex = nonAdjacentSourceLayerIndices[0];
  if (sourceLayerIndex !== targetLayerIndex - 2) {
    return null;
  }

  return sourceLayerIndex;
}

/**
 * Build the reserved residual-branch tensor name for one layer pair.
 *
 * @param sourceLayerIndex Residual source layer index.
 * @param targetLayerIndex Residual target layer index.
 * @returns Deterministic residual branch tensor name.
 */
export function buildResidualBranchTensorName(
  sourceLayerIndex: number,
  targetLayerIndex: number,
): string {
  return `ResidualBranch_l${sourceLayerIndex}_to_l${targetLayerIndex}`;
}

/**
 * Build the deterministic residual merge node name for one target layer.
 *
 * @param targetLayerIndex Target layer index.
 * @returns Residual merge node name.
 */
export function buildResidualMergeNodeName(targetLayerIndex: number): string {
  return `residual_add_l${targetLayerIndex}`;
}

/**
 * Build the deterministic residual merge output tensor name for one target layer.
 *
 * @param targetLayerIndex Target layer index.
 * @returns Residual merge output tensor name.
 */
export function buildResidualMergeOutputName(targetLayerIndex: number): string {
  return `ResidualAdd_${targetLayerIndex}`;
}

/**
 * Build the deterministic concat merge node name for one layer pair.
 *
 * @param sourceLayerIndex Skipped source layer index.
 * @param targetLayerIndex Concat target layer index.
 * @returns Concat merge node name.
 */
export function buildConcatMergeNodeName(
  sourceLayerIndex: number,
  targetLayerIndex: number,
): string {
  return `concat_merge_l${sourceLayerIndex}_to_l${targetLayerIndex}`;
}

/**
 * Build the deterministic concat merge output tensor name for one layer pair.
 *
 * @param sourceLayerIndex Skipped source layer index.
 * @param targetLayerIndex Concat target layer index.
 * @returns Concat merge output tensor name.
 */
export function buildConcatMergeOutputName(
  sourceLayerIndex: number,
  targetLayerIndex: number,
): string {
  return `ConcatMerge_${sourceLayerIndex}_to_${targetLayerIndex}`;
}

/**
 * Append residual-add metadata for an emitted one-hop merge.
 *
 * @param model Target ONNX model.
 * @param residualAdd Emitted residual-add metadata record.
 * @param includeMetadata Whether metadata emission is enabled.
 * @returns Nothing.
 */
export function appendResidualAddMetadata(
  model: OnnxModel,
  residualAdd: AdvancedGraphResidualAdd,
  includeMetadata: boolean,
): void {
  if (!includeMetadata) {
    return;
  }

  const existingProperty = model.metadata_props?.find(
    (metadataProperty) =>
      metadataProperty.key === ADVANCED_GRAPH_RESIDUAL_ADDS_KEY,
  );
  if (!existingProperty) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(ADVANCED_GRAPH_RESIDUAL_ADDS_KEY, [residualAdd]),
    );
    return;
  }

  try {
    const parsedResidualAdds = JSON.parse(existingProperty.value);
    const nextResidualAdds = Array.isArray(parsedResidualAdds)
      ? [...parsedResidualAdds, residualAdd]
      : [residualAdd];
    existingProperty.value = JSON.stringify(nextResidualAdds);
  } catch {
    existingProperty.value = JSON.stringify([residualAdd]);
  }
}

/**
 * Append concat-merge metadata for an emitted explicit concat branch.
 *
 * @param model Target ONNX model.
 * @param concatMerge Emitted concat metadata record.
 * @param includeMetadata Whether metadata emission is enabled.
 * @returns Nothing.
 */
export function appendConcatMergeMetadata(
  model: OnnxModel,
  concatMerge: AdvancedGraphConcatMerge,
  includeMetadata: boolean,
): void {
  if (!includeMetadata) {
    return;
  }

  const existingProperty = model.metadata_props?.find(
    (metadataProperty) =>
      metadataProperty.key === ADVANCED_GRAPH_CONCAT_MERGES_KEY,
  );
  if (!existingProperty) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(ADVANCED_GRAPH_CONCAT_MERGES_KEY, [concatMerge]),
    );
    return;
  }

  try {
    const parsedConcatMerges = JSON.parse(existingProperty.value);
    const nextConcatMerges = Array.isArray(parsedConcatMerges)
      ? [...parsedConcatMerges, concatMerge]
      : [concatMerge];
    existingProperty.value = JSON.stringify(nextConcatMerges);
  } catch {
    existingProperty.value = JSON.stringify([concatMerge]);
  }
}

/**
 * Collect deterministic cross-layer feed-forward edges.
 *
 * @param network Source network.
 * @param layers Resolved layered ordering.
 * @returns Sorted cross-layer connection descriptors.
 */
function collectCrossLayerConnections(
  network: Network,
  layers: NeatapticNode[][],
): AdvancedGraphCrossLayerConnection[] {
  const layerIndexByNode = buildLayerIndexByNode(layers);

  return network.nodes
    .flatMap((sourceNode) =>
      collectSourceNodeCrossLayerConnections(sourceNode, layerIndexByNode),
    )
    .toSorted(compareCrossLayerConnections);
}

/**
 * Build a stable node->layer index lookup for the resolved layered ordering.
 *
 * @param layers Resolved layered ordering.
 * @returns Node-to-layer lookup.
 */
function buildLayerIndexByNode(
  layers: NeatapticNode[][],
): Map<NeatapticNode, number> {
  return layers.reduce((layerIndexByNode, layerNodes, layerIndex) => {
    layerNodes.forEach((node) => {
      layerIndexByNode.set(node, layerIndex);
    });
    return layerIndexByNode;
  }, new Map<NeatapticNode, number>());
}

/**
 * Collect cross-layer feed-forward edges from one source node.
 *
 * @param sourceNode Source node.
 * @param layerIndexByNode Node-to-layer lookup.
 * @returns Cross-layer descriptors for this source node.
 */
function collectSourceNodeCrossLayerConnections(
  sourceNode: NeatapticNode,
  layerIndexByNode: Map<NeatapticNode, number>,
): AdvancedGraphCrossLayerConnection[] {
  const sourceNodeInternal = sourceNode as NodeInternalsWithExportIndex;
  const sourceLayerIndex = layerIndexByNode.get(sourceNode);

  if (sourceLayerIndex === undefined) {
    return [];
  }

  return sourceNodeInternal.connections.out
    .map((connection) =>
      createCrossLayerConnectionDescriptor(
        sourceNodeInternal,
        sourceLayerIndex,
        connection.to,
        layerIndexByNode,
      ),
    )
    .filter(isAdvancedGraphCrossLayerConnection);
}

/**
 * Create one cross-layer descriptor when the target is non-adjacent.
 *
 * @param sourceNodeInternal Source-node internals.
 * @param sourceLayerIndex Source-layer index.
 * @param targetNode Target node.
 * @param layerIndexByNode Node-to-layer lookup.
 * @returns Descriptor when the edge skips one or more layers.
 */
function createCrossLayerConnectionDescriptor(
  sourceNodeInternal: NodeInternalsWithExportIndex,
  sourceLayerIndex: number,
  targetNode: NeatapticNode,
  layerIndexByNode: Map<NeatapticNode, number>,
): AdvancedGraphCrossLayerConnection | undefined {
  const targetNodeInternal = targetNode as NodeInternalsWithExportIndex;
  const targetLayerIndex = layerIndexByNode.get(targetNode);

  if (targetLayerIndex === undefined) {
    return undefined;
  }

  if (targetLayerIndex <= sourceLayerIndex + 1) {
    return undefined;
  }

  const sourceNodeIndex = resolveNodeIndex(sourceNodeInternal);
  const targetNodeIndex = resolveNodeIndex(targetNodeInternal);

  return {
    sourceNodeIndex,
    sourceLayerIndex,
    targetNodeIndex,
    targetLayerIndex,
    branchTensorName: buildBranchTensorName({
      sourceLayerIndex,
      targetLayerIndex,
      sourceNodeIndex,
      targetNodeIndex,
    }),
  };
}

/**
 * Resolve a stable node export index.
 *
 * @param nodeInternal Node internals.
 * @returns Stable export index.
 */
function resolveNodeIndex(nodeInternal: NodeInternalsWithExportIndex): number {
  return nodeInternal.index ?? -1;
}

/**
 * Build the reserved branch tensor name for one cross-layer edge.
 *
 * @param context Branch-name context.
 * @returns Deterministic branch tensor name.
 */
function buildBranchTensorName(context: {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  sourceNodeIndex: number;
  targetNodeIndex: number;
}): string {
  return `Branch_l${context.sourceLayerIndex}_to_l${context.targetLayerIndex}_from_n${context.sourceNodeIndex}_to_n${context.targetNodeIndex}`;
}

/**
 * Type guard for optional cross-layer descriptors.
 *
 * @param descriptor Optional descriptor.
 * @returns Whether the descriptor exists.
 */
function isAdvancedGraphCrossLayerConnection(
  descriptor: AdvancedGraphCrossLayerConnection | undefined,
): descriptor is AdvancedGraphCrossLayerConnection {
  return descriptor !== undefined;
}

/**
 * Keep metadata emission order deterministic.
 *
 * @param left Left descriptor.
 * @param right Right descriptor.
 * @returns Sort comparison result.
 */
function compareCrossLayerConnections(
  left: AdvancedGraphCrossLayerConnection,
  right: AdvancedGraphCrossLayerConnection,
): number {
  return (
    left.sourceNodeIndex - right.sourceNodeIndex ||
    left.targetNodeIndex - right.targetNodeIndex
  );
}

/**
 * Build one metadata property with a JSON payload.
 *
 * @param key Metadata key.
 * @param value Metadata value.
 * @returns Metadata property.
 */
function buildMetadataProperty(
  key: string,
  value:
    | AdvancedGraphCrossLayerConnection[]
    | AdvancedGraphResidualAdd[]
    | AdvancedGraphConcatMerge[],
): OnnxMetadataProperty {
  return {
    key,
    value: JSON.stringify(value),
  };
}

/**
 * Append one metadata property to the ONNX model.
 *
 * @param model Target model.
 * @param metadataProperty Metadata property.
 * @returns Nothing.
 */
function appendMetadataProperty(
  model: OnnxModel,
  metadataProperty: OnnxMetadataProperty,
): void {
  model.metadata_props = [...(model.metadata_props ?? []), metadataProperty];
}