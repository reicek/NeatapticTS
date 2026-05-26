/**
 * Generic topology inference helpers for browser network visualization.
 *
 * These helpers answer: how should the network be partitioned into ordered
 * layers so layout and semantic labels stay meaningful even when metadata is missing?
 */

import Network from '../../architecture/network';
import type { VisualNetworkNode } from './network-view.layout.utils';

const TEMPORAL_MODULE_LABEL_BY_KIND = {
  lstm: 'LSTM',
  gru: 'GRU',
  'narx-memory': 'NARX Memory',
} as const;

/**
 * Semantic annotation for one layer of nodes.
 *
 * For example, recurrent networks can label hidden columns as "input gate",
 * "hidden state t-1", etc. Feed-forward networks might have generic labels.
 */
export interface NetworkLayerAnnotation {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  backgroundColor?: string;
  nodeIndices: number[];
}

/**
 * Full topology plan for layout and rendering.
 *
 * Preserves the layer-array input used by layout helpers and adds optional
 * semantic annotations for overlays.
 */
export interface NetworkVisualizationTopologyPlan {
  networkLayers: VisualNetworkNode[][];
  layerAnnotations: NetworkLayerAnnotation[];
  topologyMode: 'acyclic' | 'recurrent';
}

/**
 * Resolve ordered layered node groups from the topology plan, used by canvas layout and topology-aware rendering helpers.
 *
 * @param network - Runtime network instance (or undefined for fallback).
 * @param inputSize - Input count (used if network is undefined).
 * @param outputSize - Output count (used if network is undefined).
 * @returns Layered nodes for rendering.
 */
export function resolveNetworkVisualizationLayers(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNode[][] {
  return resolveNetworkVisualizationTopologyPlan(network, inputSize, outputSize)
    .networkLayers;
}

/**
 * Resolves the full topology plan including optional layer annotations.
 *
 * For recurrent networks, this detects temporal modules and creates annotations.
 * For feed-forward networks, this creates a simple acyclic plan.
 *
 * @param network - Runtime network instance (or undefined for fallback).
 * @param inputSize - Input count (used if network is undefined).
 * @param outputSize - Output count (used if network is undefined).
 * @returns Layered nodes plus semantic layer annotations.
 */
export function resolveNetworkVisualizationTopologyPlan(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationTopologyPlan {
  // Step 1: Build fallback layers when no runtime network is available.
  if (!network) {
    return createFallbackTopologyPlan(inputSize, outputSize);
  }

  // Step 2: Extract topology from the network.
  const topologyMode = resolveTopologyMode(network);

  // Step 3: Group nodes by type into layers.
  const { inputNodes, hiddenNodes, outputNodes } =
    groupVisualNodesByType(network);
  const networkLayers = buildNetworkLayers(
    inputNodes,
    hiddenNodes,
    outputNodes,
  );

  // Step 4: Try to infer temporal module annotations for recurrent networks.
  const layerAnnotations = resolveLayerAnnotations(
    network,
    topologyMode,
    hiddenNodes,
  );

  return {
    networkLayers,
    layerAnnotations,
    topologyMode,
  };
}

function createFallbackTopologyPlan(
  inputSize: number,
  outputSize: number,
): NetworkVisualizationTopologyPlan {
  return {
    networkLayers: [
      Array.from({ length: inputSize }, (_unusedValue, inputNodeIndex) => ({
        index: inputNodeIndex,
        type: 'input',
        bias: 0,
      })),
      Array.from({ length: outputSize }, (_unusedValue, outputNodeIndex) => ({
        index: inputSize + outputNodeIndex,
        type: 'output',
        bias: 0,
      })),
    ],
    layerAnnotations: [],
    topologyMode: 'acyclic',
  };
}

function resolveTopologyMode(
  network: Network,
): NetworkVisualizationTopologyPlan['topologyMode'] {
  return network.getTopologyIntent() === 'feed-forward'
    ? 'acyclic'
    : 'recurrent';
}

function groupVisualNodesByType(network: Network): {
  inputNodes: VisualNetworkNode[];
  hiddenNodes: VisualNetworkNode[];
  outputNodes: VisualNetworkNode[];
} {
  return {
    inputNodes: collectVisualNodesByType(network, 'input'),
    hiddenNodes: collectVisualNodesByType(network, 'hidden'),
    outputNodes: collectVisualNodesByType(network, 'output'),
  };
}

function collectVisualNodesByType(
  network: Network,
  nodeType: VisualNetworkNode['type'],
): VisualNetworkNode[] {
  return network.nodes
    .filter((node) => node.type === nodeType)
    .map((node) => ({
      index: node.index ?? 0,
      type: nodeType,
      bias: node.bias,
    }))
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
}

function buildNetworkLayers(
  inputNodes: VisualNetworkNode[],
  hiddenNodes: VisualNetworkNode[],
  outputNodes: VisualNetworkNode[],
): VisualNetworkNode[][] {
  return [inputNodes, hiddenNodes, outputNodes].filter(
    (layer): layer is VisualNetworkNode[] => layer.length > 0,
  );
}

function resolveLayerAnnotations(
  network: Network,
  topologyMode: NetworkVisualizationTopologyPlan['topologyMode'],
  hiddenNodes: VisualNetworkNode[],
): NetworkLayerAnnotation[] {
  if (topologyMode !== 'recurrent' || hiddenNodes.length === 0) {
    return [];
  }

  const nodeIndexByGeneId = createNodeIndexByGeneId(network);

  try {
    const temporalDescription = network.describeTemporalStructure();
    return temporalDescription.recurrentModules.length > 0
      ? temporalDescription.recurrentModules.map((recurrentModule) =>
          mapTemporalModuleToLayerAnnotation(
            recurrentModule,
            nodeIndexByGeneId,
          ),
        )
      : [];
  } catch {
    return [];
  }
}

function createNodeIndexByGeneId(network: Network): Map<number, number> {
  return new Map<number, number>(
    network.nodes.flatMap((node) =>
      typeof node.geneId === 'number'
        ? [[node.geneId, node.index ?? 0] as const]
        : [],
    ),
  );
}

function mapTemporalModuleToLayerAnnotation(
  recurrentModule: ReturnType<
    Network['describeTemporalStructure']
  >['recurrentModules'][number],
  nodeIndexByGeneId: Map<number, number>,
): NetworkLayerAnnotation {
  const label =
    recurrentModule.moduleLabel ??
    TEMPORAL_MODULE_LABEL_BY_KIND[recurrentModule.kind];
  const roleLabels = Object.keys(recurrentModule.nodeGeneIdsByRole);

  return {
    label,
    labelLines: [label],
    tooltipHeading: label,
    tooltipBodyParagraphs: [
      `Roles: ${roleLabels.join(', ') || 'unlabeled module'}.`,
    ],
    nodeIndices: resolveTemporalModuleNodeIndices(
      recurrentModule.nodeGeneIdsByRole,
      nodeIndexByGeneId,
    ),
  };
}

function resolveTemporalModuleNodeIndices(
  nodeGeneIdsByRole: ReturnType<
    Network['describeTemporalStructure']
  >['recurrentModules'][number]['nodeGeneIdsByRole'],
  nodeIndexByGeneId: Map<number, number>,
): number[] {
  return [
    ...new Set(
      Object.values(nodeGeneIdsByRole)
        .flatMap((nodeGeneIds) => nodeGeneIds)
        .flatMap((nodeGeneId) => {
          const nodeIndex = nodeIndexByGeneId.get(nodeGeneId);
          return typeof nodeIndex === 'number' ? [nodeIndex] : [];
        }),
    ),
  ];
}
