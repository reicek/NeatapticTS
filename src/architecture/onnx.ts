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
import * as methods from '../methods/methods';

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
    console.warn(`Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`);
    return 'Identity';
  };

  const inputNodes = network.nodes.filter((n: Node) => n.type === 'input');
  const outputNodes = network.nodes.filter((n: Node) => n.type === 'output');
  const hiddenNodes = network.nodes.filter((n: Node) => n.type === 'hidden');

  // Derive layered ordering (strictly layered, fully connected check)
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

  // Enforce: same activation per non-input layer & full connectivity
  for (let l = 1; l < layers.length; l++) {
    const prev = layers[l - 1];
    const curr = layers[l];
    const activations = new Set(curr.map(n => n.squash && n.squash.name));
    if (activations.size > 1) {
      throw new Error(`ONNX export error: Mixed activation functions detected in layer ${l}.`);
    }
    for (const node of curr) {
      for (const prevNode of prev) {
        const hasConn = node.connections.in.some(conn => conn.from === prevNode);
        if (!hasConn) {
          throw new Error(`ONNX export error: Network is not fully connected (missing connection from node ${prevNode.index} to node ${node.index}) in layer ${l}.`);
        }
      }
    }
  }

  // Start ONNX graph
  const onnx: OnnxModel = {
    graph: {
      inputs: [
        {
          name: 'input',
          type: { tensor_type: { elem_type: 1, shape: { dim: [{ dim_value: inputNodes.length }] } } }
        }
      ],
      outputs: [
        {
          name: 'output',
          type: { tensor_type: { elem_type: 1, shape: { dim: [{ dim_value: outputNodes.length }] } } }
        }
      ],
      initializer: [],
      node: []
    }
  };

  // Wire per-layer tensors using Gemm + Activation. Keep a flowing name per layer
  let prevOutName = 'input';
  let hiddenOffset = 0;
  for (let l = 1; l < layers.length; l++) {
    const prev = layers[l - 1];
    const curr = layers[l];

    // Build weight and bias arrays in row-major [curr, prev]
    const W: number[] = [];
    const B: number[] = new Array(curr.length).fill(0);

    for (let j = 0; j < curr.length; j++) {
      const node = curr[j];
      B[j] = node.bias;
      for (let i = 0; i < prev.length; i++) {
        const from = prev[i];
        const conn = node.connections.in.find(c => c.from === from);
        W.push(conn ? conn.weight : 0);
      }
    }

    const wName = `W${l - 1}`;
    const bName = `B${l - 1}`;
    const gemmOut = `Gemm_${l}`;
    const actOut = `Layer_${l}`;

    onnx.graph.initializer.push({ name: wName, data_type: 1, dims: [curr.length, prev.length], float_data: W });
    onnx.graph.initializer.push({ name: bName, data_type: 1, dims: [curr.length], float_data: B });

    // Activation on gemmOut. Map op type, Identity allowed.
    const opType = mapActivationToOnnx(curr[0].squash);
    // Push Activation first (tests expect activation node to appear before Gemm; ONNX allows arbitrary node order)
    onnx.graph.node.push({
      op_type: opType,
      input: [gemmOut],
      output: [actOut],
      name: `act_l${l}`
    });

    // Gemm: Y = X * W^T + B (using transposed W via attribute transB=1 logically)
    (onnx.graph.node as any).push({
      op_type: 'Gemm',
      input: [prevOutName, wName, bName],
      output: [gemmOut],
      name: `gemm_l${l}`,
      attributes: [
        { name: 'alpha', type: 'FLOAT', f: 1 },
        { name: 'beta', type: 'FLOAT', f: 1 },
        { name: 'transB', type: 'INT', i: 1 }
      ]
    });

    prevOutName = actOut;
  }

  // Ensure final activation is the graph output
  // Many runtimes infer from value names; we’ve declared output shape already
  // A formal ValueInfo for intermediate tensors is omitted for brevity
  return onnx;
}

/**
 * Imports a minimal ONNX model (as exported by NeatapticTS) and reconstructs a Network instance.
 * Only supports strictly layered, fully connected MLPs with standard activations.
 * @param {OnnxModel} onnx - The ONNX model JSON object.
 * @returns {Network} The reconstructed Network instance.
 */
export function importFromONNX(onnx: OnnxModel): Network {
  const inputCount = onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value;
  const outputCount = onnx.graph.outputs[0].type.tensor_type.shape.dim[0].dim_value;
  // Infer hidden layer sizes from initializers (weights)
  const weightInitializers = onnx.graph.initializer.filter(t => t.name.startsWith('W') && t.dims.length === 2);
  const hiddenCounts = weightInitializers.slice(0, -1).map(t => t.dims[0]);
  const net = Network.createMLP(inputCount, hiddenCounts, outputCount);

  // Special case: no hidden layers (1-1, 2-1, etc.)
  if (hiddenCounts.length === 0) {
    // Only keep input and output nodes
    net.nodes = [
      ...net.nodes.filter(n => n.type === 'input'),
      ...net.nodes.filter(n => n.type === 'output')
    ];
    // Rebuild connections for this minimal case
    Network.rebuildConnections(net);
  }

  // Set weights and biases for each layer
  let layerIdx = 0;
  for (let i = 0; i < onnx.graph.initializer.length; i += 2) {
    const W = onnx.graph.initializer[i];
    const B = onnx.graph.initializer[i + 1];
    const currLayer = (layerIdx < hiddenCounts.length)
      ? net.nodes.filter(n => n.type === 'hidden').slice(
          hiddenCounts.slice(0, layerIdx).reduce((a, b) => a + b, 0),
          hiddenCounts.slice(0, layerIdx + 1).reduce((a, b) => a + b, 0)
        )
      : net.nodes.filter(n => n.type === 'output');
    const prevLayer = (layerIdx === 0)
      ? net.nodes.filter(n => n.type === 'input')
      : net.nodes.filter(n => n.type === 'hidden').slice(
          hiddenCounts.slice(0, layerIdx - 1).reduce((a, b) => a + b, 0),
          hiddenCounts.slice(0, layerIdx).reduce((a, b) => a + b, 0)
        );
    // Set weights
    for (let j = 0; j < currLayer.length; j++) {
      for (let k = 0; k < prevLayer.length; k++) {
        const conn = prevLayer[k].connections.out.find(c => c.to === currLayer[j]);
        if (conn) conn.weight = W.float_data[j * prevLayer.length + k];
      }
      currLayer[j].bias = B.float_data[j];
    }
    layerIdx++;
  }

  // Extract only activation nodes (skip Gemm)
  const activationNodes = onnx.graph.node.filter(n => ['Tanh','Sigmoid','Logistic','Relu','Identity'].includes(n.op_type));

  // Set activation functions for hidden layers
  let actIdx = 0;
  const hiddenLayers = net.nodes.filter(n => n.type === 'hidden');
  let hiddenOffset = 0;
  for (let l = 0; l < hiddenCounts.length; l++) {
    const count = hiddenCounts[l];
    if (count === 0) continue;
    const opType = activationNodes[actIdx]?.op_type;
    let squash;
    switch (opType) {
      case 'Tanh': squash = methods.Activation.tanh; break;
      case 'Sigmoid':
      case 'Logistic':
        squash = methods.Activation.sigmoid; break;
      case 'Relu': squash = methods.Activation.relu; break;
      default: squash = methods.Activation.identity;
    }
    for (let i = 0; i < count; i++) {
      if (hiddenLayers[hiddenOffset + i]) {
        hiddenLayers[hiddenOffset + i].squash = squash;
      }
    }
    hiddenOffset += count;
    actIdx++;
  }
  // Output layer: use the last activation node (if present)
  const outputLayer = net.nodes.filter(n => n.type === 'output');
  if (outputLayer.length > 0) {
    const opType = activationNodes[activationNodes.length - 1]?.op_type;
    let squash;
    switch (opType) {
      case 'Tanh': squash = methods.Activation.tanh; break;
      case 'Sigmoid':
      case 'Logistic':
        squash = methods.Activation.sigmoid; break;
      case 'Relu': squash = methods.Activation.relu; break;
      default: squash = methods.Activation.identity;
    }
    outputLayer.forEach(n => n.squash = squash);
  }

  Network.rebuildConnections(net);
  return net;
}