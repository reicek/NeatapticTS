/**
 * Generic topology inference helpers for browser network visualization.
 *
 * These helpers answer: how should the network be partitioned into ordered
 * layers so layout and semantic labels stay meaningful even when metadata is missing?
 */

import Network from '../../architecture/network';
import type { VisualNetworkNode } from './network-view.layout.utils';

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
  const nodes = network.nodes.toSorted((a, b) => {
    if (a.type !== b.type) {
      const typeOrder: Record<string, number> = {
        input: 0,
        hidden: 1,
        output: 2,
      };
      return (typeOrder[a.type] ?? 99) - (typeOrder[b.type] ?? 99);
    }
    return (a.index ?? 0) - (b.index ?? 0);
  });

  // Step 3: Group nodes by type into layers.
  const inputNodes = nodes.filter((n) => n.type === 'input');
  const hiddenNodes = nodes.filter((n) => n.type === 'hidden');
  const outputNodes = nodes.filter((n) => n.type === 'output');

  const networkLayers: VisualNetworkNode[][] = [];
  if (inputNodes.length > 0) {
    networkLayers.push(
      inputNodes.map((n) => ({
        index: n.index ?? 0,
        type: 'input' as const,
        bias: n.bias,
      })),
    );
  }
  if (hiddenNodes.length > 0) {
    networkLayers.push(
      hiddenNodes.map((n) => ({
        index: n.index ?? 0,
        type: 'hidden' as const,
        bias: n.bias,
      })),
    );
  }
  if (outputNodes.length > 0) {
    networkLayers.push(
      outputNodes.map((n) => ({
        index: n.index ?? 0,
        type: 'output' as const,
        bias: n.bias,
      })),
    );
  }

  // Step 4: Try to infer temporal module annotations for recurrent networks.
  let layerAnnotations: NetworkLayerAnnotation[] = [];
  if (topologyMode === 'recurrent' && hiddenNodes.length > 0) {
    // Try to use network.describeTemporalStructure() if available
    try {
      const temporalDescription = (
        network as any
      ).describeTemporalStructure?.();
      if (temporalDescription?.recurrentModules) {
        layerAnnotations = temporalDescription.recurrentModules.map(
          (module: any) => ({
            label: module.label ?? 'Module',
            labelLines: module.labelLines ?? [module.label ?? 'Module'],
            tooltipHeading: module.tooltipHeading ?? module.label ?? 'Module',
            tooltipBodyParagraphs: module.tooltipBodyParagraphs ?? [],
            backgroundColor: module.backgroundColor,
            nodeIndices: module.nodeIndices ?? [],
          }),
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
