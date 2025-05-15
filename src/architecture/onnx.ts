type OnnxModel = {
  graph: OnnxGraph;
};

type OnnxGraph = {
  inputs: any[];
  outputs: any[];
  initializer: OnnxTensor[];
  node: OnnxNode[];
};

type OnnxTensor = {
  name: string;
  data_type: number;
  dims: number[];
  float_data: number[];
};

type OnnxNode = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
};

import Node from './node';
import Network from './network';

/**
 * Exports a given network to ONNX format (JSON object, minimal MLP support).
 * Only standard feedforward architectures and standard activations are supported.
 * Gating, custom activations, and evolutionary features are ignored or replaced with Identity.
 *
 * @param {Network} network - The network instance to export.
 * @returns {OnnxModel} ONNX model as a JSON object.
 */
export function exportToONNX(network: Network): OnnxModel {
  // Ensure the network.connections array is up-to-date with per-node connections
  Network.rebuildConnections(network);

  // Assign indices to all nodes if not already set
  network.nodes.forEach((node: Node, idx: number) => { node.index = idx; });

  // Throw if no connections (invalid MLP)
  if (!network.connections || network.connections.length === 0) {
    throw new Error('ONNX export currently only supports simple MLPs');
  }

  // Helper to map Neataptic activation to ONNX operator type
  const mapActivationToOnnx = (squash: Function): string => {
    const name = squash.name.toUpperCase();
    if (name.includes('TANH')) return 'Tanh';
    if (name.includes('LOGISTIC') || name.includes('SIGMOID')) return 'Sigmoid';
    if (name.includes('RELU')) return 'Relu';
    // Note: LeakyReLU, Softmax, and other activations are not mapped and will fall through to Identity
    console.warn(`Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`);
    return 'Identity';
  };

  // Build layers: input, hidden(s), output
  const inputNodes = network.nodes.filter((n: Node) => n.type === 'input');
  const outputNodes = network.nodes.filter((n: Node) => n.type === 'output');
  const hiddenNodes = network.nodes.filter((n: Node) => n.type === 'hidden');

  let layers: Node[][] = [];
  if (hiddenNodes.length === 0) {
    layers = [inputNodes, outputNodes];
  } else {
    let currentLayer: Node[] = [];
    let prevLayer = inputNodes;
    let remaining = [...hiddenNodes];
    while (remaining.length > 0) {
      currentLayer = remaining.filter(h => h.connections.in.every(conn => prevLayer.includes(conn.from)));
      if (currentLayer.length === 0) {
        throw new Error('Invalid network structure for ONNX export.');
      }
      layers.push(prevLayer);
      prevLayer = currentLayer;
      remaining = remaining.filter(h => !currentLayer.includes(h));
    }
    layers.push(prevLayer);
    layers.push(outputNodes);
  }

  // Enforce: All nodes in a layer must use the same activation function for ONNX compatibility
  for (const [layerIdx, layer] of layers.entries()) {
    const activations = new Set(layer.map(n => n.squash && n.squash.name));
    if (activations.size > 1) {
      throw new Error(
        `ONNX export error: Mixed activation functions detected in layer ${layerIdx}. ` +
        'All nodes in a layer must use the same activation function for ONNX compatibility.'
      );
    }
  }

  const onnx: OnnxModel = {
    graph: {
      inputs: [
        {
          name: 'input',
          type: { tensor_type: { elem_type: 'FLOAT', shape: { dim: [{ dim_value: network.input }] } } }
        }
      ],
      outputs: [
        {
          name: 'output',
          type: { tensor_type: { elem_type: 'FLOAT', shape: { dim: [{ dim_value: network.output }] } } }
        }
      ],
      initializer: [],
      node: []
    }
  };

  let lastOutput = 'input';
  for (let l = 1; l < layers.length; l++) {
    const prev = layers[l - 1];
    const curr = layers[l];

    const W = Array(curr.length).fill(0).map(() => Array(prev.length).fill(0));
    const B = Array(curr.length).fill(0);

    for (let j = 0; j < curr.length; j++) {
      const node = curr[j];
      B[j] = node.bias;
      node.connections.in.forEach(conn => {
        const i = prev.indexOf(conn.from);
        if (i >= 0) {
          W[j][i] = conn.weight;
        }
      });

      onnx.graph.node.push({
        input: [lastOutput],
        output: [`layer${l}_${j}`],
        name: `node${node.index}`,
        op_type: mapActivationToOnnx(node.squash)
      });
    }

    onnx.graph.initializer.push({
      name: `W${l - 1}`,
      data_type: 1,
      dims: [curr.length, prev.length],
      float_data: W.flat()
    });
    onnx.graph.initializer.push({
      name: `B${l - 1}`,
      data_type: 1,
      dims: [curr.length],
      float_data: B
    });

    lastOutput = `layer${l}`;
  }

  return onnx;
}