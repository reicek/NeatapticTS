import type Network from '../network';
import Connection from '../connection';
import type NeatapticNode from '../node';
import type {
  NodeInternals,
  OnnxExportOptions,
} from './network.onnx.types.utils';

/**
 * Rebuild the network's flat connections array from each node's outgoing list.
 *
 * @param networkLike Network-like instance to mutate.
 * @returns Nothing.
 */
export function rebuildConnectionsLocal(networkLike: Network): void {
  const uniqueConnections = new Set<Connection>();
  networkLike.nodes.forEach((node) =>
    node.connections?.out.forEach((connection) =>
      uniqueConnections.add(connection),
    ),
  );
  networkLike.connections = Array.from(uniqueConnections);
}

/**
 * Map an internal activation function (squash) to an ONNX op_type.
 *
 * @param squash Activation function reference.
 * @returns ONNX activation operator name.
 */
export function mapActivationToOnnx(
  squash: ((x: number, derivate?: boolean) => number) & { name?: string },
): string {
  const upperName = (squash?.name || '').toUpperCase();
  if (upperName.includes('TANH')) return 'Tanh';
  if (upperName.includes('LOGISTIC') || upperName.includes('SIGMOID'))
    return 'Sigmoid';
  if (upperName.includes('RELU')) return 'Relu';
  if (squash) {
    console.warn(
      `Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`,
    );
  }
  return 'Identity';
}

/**
 * Infer strictly layered ordering from a network.
 *
 * @param network Source network.
 * @returns Ordered layers: input, hidden..., output.
 */
export function inferLayerOrdering(network: Network): NeatapticNode[][] {
  const inputNodes = network.nodes.filter((node) => node.type === 'input');
  const outputNodes = network.nodes.filter((node) => node.type === 'output');
  const hiddenNodes = network.nodes.filter((node) => node.type === 'hidden');
  if (hiddenNodes.length === 0) return [inputNodes, outputNodes];

  let remainingHidden = [...hiddenNodes];
  let previousLayer = inputNodes;
  const layerAccumulator: NeatapticNode[][] = [];

  while (remainingHidden.length) {
    const currentLayer = remainingHidden.filter((hiddenNode) =>
      (hiddenNode as unknown as NodeInternals).connections.in.every(
        (connection) => previousLayer.includes(connection.from),
      ),
    );
    if (!currentLayer.length) {
      throw new Error(
        'Invalid network structure for ONNX export: cannot resolve layered ordering.',
      );
    }
    layerAccumulator.push(previousLayer);
    previousLayer = currentLayer;
    remainingHidden = remainingHidden.filter(
      (hiddenNode) => !currentLayer.includes(hiddenNode),
    );
  }

  layerAccumulator.push(previousLayer);
  layerAccumulator.push(outputNodes);
  return layerAccumulator;
}

/**
 * Validate connectivity and activation homogeneity constraints per layer.
 *
 * @param layers Layered node arrays.
 * @param network Source network (reserved for compatibility).
 * @param options Export options.
 * @returns Nothing.
 */
export function validateLayerHomogeneityAndConnectivity(
  layers: NeatapticNode[][],
  network: Network,
  options: OnnxExportOptions,
): void {
  void network;
  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    const previousLayerNodes = layers[layerIndex - 1];
    const currentLayerNodes = layers[layerIndex];
    const activationNameSet = new Set(
      currentLayerNodes.map((node) => {
        const nodeInternal = node as unknown as NodeInternals;
        return nodeInternal.squash && nodeInternal.squash.name;
      }),
    );
    if (activationNameSet.size > 1 && !options.allowMixedActivations) {
      throw new Error(
        `ONNX export error: Mixed activation functions detected in layer ${layerIndex}. (enable allowMixedActivations to decompose layer)`,
      );
    }
    if (activationNameSet.size > 1 && options.allowMixedActivations) {
      console.warn(
        `Warning: Mixed activations in layer ${layerIndex}; exporting per-neuron Gemm + Activation (+Concat) baseline.`,
      );
    }

    for (const targetNode of currentLayerNodes) {
      const targetInternal = targetNode as unknown as NodeInternals;
      for (const sourceNode of previousLayerNodes) {
        const isConnected = targetInternal.connections.in.some(
          (connection) => connection.from === sourceNode,
        );
        if (!isConnected && !options.allowPartialConnectivity) {
          throw new Error(
            `ONNX export error: Missing connection from node ${sourceNode.index} to node ${targetNode.index} in layer ${layerIndex}. (enable allowPartialConnectivity)`,
          );
        }
      }
    }
  }
}
