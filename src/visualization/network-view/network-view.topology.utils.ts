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
 * Resolves layered node groups for network layout.
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

  // Step 2: Extract topology from the network.
  const topologyMode =
    network.getTopologyIntent() === 'feed-forward'
      ? ('acyclic' as const)
      : ('recurrent' as const);

  // Step 3: Group nodes by type into layers.
  const inputNodes = network.nodes
    .filter((node) => node.type === 'input')
    .map((node) => ({
      index: node.index ?? 0,
      type: 'input' as const,
      bias: node.bias,
    }))
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const hiddenNodes = network.nodes
    .filter((node) => node.type === 'hidden')
    .map((node) => ({
      index: node.index ?? 0,
      type: 'hidden' as const,
      bias: node.bias,
    }))
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const outputNodes = network.nodes
    .filter((node) => node.type === 'output')
    .map((node) => ({
      index: node.index ?? 0,
      type: 'output' as const,
      bias: node.bias,
    }))
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);

  const networkLayers: VisualNetworkNode[][] = [];
  if (inputNodes.length > 0) {
    networkLayers.push(inputNodes);
  }
  if (hiddenNodes.length > 0) {
    networkLayers.push(hiddenNodes);
  }
  if (outputNodes.length > 0) {
    networkLayers.push(outputNodes);
  }

  // Step 4: Try to infer temporal module annotations for recurrent networks.
  let layerAnnotations: NetworkLayerAnnotation[] = [];
  if (topologyMode === 'recurrent' && hiddenNodes.length > 0) {
    const nodeIndexByGeneId = new Map<number, number>(
      network.nodes.flatMap((node) =>
        typeof node.geneId === 'number'
          ? [[node.geneId, node.index ?? 0] as const]
          : [],
      ),
    );

    // Try to derive recurrent module annotations from the public descriptor.
    try {
      const temporalDescription = network.describeTemporalStructure();
      if (temporalDescription.recurrentModules.length > 0) {
        layerAnnotations = temporalDescription.recurrentModules.map(
          (recurrentModule) => {
            const label =
              recurrentModule.moduleLabel ??
              TEMPORAL_MODULE_LABEL_BY_KIND[recurrentModule.kind];
            const roleLabels = Object.keys(recurrentModule.nodeGeneIdsByRole);
            const nodeIndices = [
              ...new Set(
                Object.values(recurrentModule.nodeGeneIdsByRole)
                  .flatMap((nodeGeneIds) => nodeGeneIds)
                  .flatMap((nodeGeneId) => {
                    const nodeIndex = nodeIndexByGeneId.get(nodeGeneId);
                    return typeof nodeIndex === 'number' ? [nodeIndex] : [];
                  }),
              ),
            ];

            return {
              label,
              labelLines: [label],
              tooltipHeading: label,
              tooltipBodyParagraphs: [
                `Roles: ${roleLabels.join(', ') || 'unlabeled module'}.`,
              ],
              nodeIndices,
            };
          },
        );
      }
    } catch {
      // Silently fall back to empty annotations if temporal description fails
      layerAnnotations = [];
    }
  }

  return {
    networkLayers,
    layerAnnotations,
    topologyMode,
  };
}
