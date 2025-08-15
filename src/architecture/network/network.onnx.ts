/**
 * ONNX export/import utilities for a constrained subset of networks (minimal MLPs).
 *
 * Scope & Assumptions:
 *  - Supports strictly layered, fully-connected feed‑forward topologies only.
 *  - Each non-input layer must have a homogeneous activation function (single activation per layer).
 *  - No recurrence, self/backward connections, gating, or heterogeneous per-node activations.
 *  - Unsupported activation functions degrade to Identity with a console warning.
 *
 * Design Goals:
 *  - Keep zero external ONNX dependencies (simple JSON shape) to avoid heavy protobuf runtime.
 *  - Perform structural validation early to produce informative errors.
 *  - Provide clearly documented stepwise transformation for auditability & reproducibility.
 *
 * Future Extension Ideas (not implemented):
 *  - BatchNorm / Dropout folding recognition.
 *  - Recurrent unrolling (time-limited) to pseudo-feedforward graph.
 *  - Operator fusion (Gemm + Activation ordering refinement).
 *  - Custom activation registration to ONNX via function proto definitions.
 *
 * NOTE: Import is only guaranteed to work for models produced by {@link exportToONNX};
 * arbitrary ONNX graphs are NOT universally supported.
 */

import * as methods from '../../methods/methods';
import type Network from '../network';

// --- Lightweight ONNX type aliases (minimal subset used for export/import) ---
export type OnnxModel = { graph: OnnxGraph };
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
  attributes?: any[];
};

// ---------------------------------------------------------------------------
// Internal helpers (not exported)
// ---------------------------------------------------------------------------

/** Rebuild the network's flat connections array from each node's outgoing list (avoids circular import). */
function rebuildConnectionsLocal(networkLike: any): void {
  /** Set used to deduplicate connection objects. */
  const uniqueConnections = new Set<any>();
  networkLike.nodes.forEach((node: any) =>
    node.connections?.out.forEach((conn: any) => uniqueConnections.add(conn))
  );
  networkLike.connections = Array.from(uniqueConnections);
}

/** Map an internal activation function (squash) to an ONNX op_type, defaulting to Identity. */
function mapActivationToOnnx(squash: any): string {
  const upperName = (squash?.name || '').toUpperCase();
  if (upperName.includes('TANH')) return 'Tanh';
  if (upperName.includes('LOGISTIC') || upperName.includes('SIGMOID'))
    return 'Sigmoid';
  if (upperName.includes('RELU')) return 'Relu';
  if (squash)
    console.warn(
      `Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`
    );
  return 'Identity';
}

/** Infer strictly layered ordering from a network, ensuring feed-forward fully-connected structure. */
function inferLayerOrdering(network: Network): any[][] {
  /** All input nodes (first layer). */
  const inputNodes = network.nodes.filter((n: any) => n.type === 'input');
  /** All output nodes (final layer). */
  const outputNodes = network.nodes.filter((n: any) => n.type === 'output');
  /** All hidden nodes requiring layer inference. */
  const hiddenNodes = network.nodes.filter((n: any) => n.type === 'hidden');
  if (hiddenNodes.length === 0) return [inputNodes, outputNodes];
  /** Remaining hidden nodes to allocate. */
  let remainingHidden = [...hiddenNodes];
  /** Previously accepted layer (starts at inputs). */
  let previousLayer = inputNodes;
  /** Accumulated layers (excluding final output which is appended later). */
  const layerAccumulator: any[][] = [];
  while (remainingHidden.length) {
    /** Hidden nodes whose inbound connections originate only from previousLayer. */
    const currentLayer = remainingHidden.filter((hidden) =>
      hidden.connections.in.every((conn: any) =>
        previousLayer.includes(conn.from)
      )
    );
    if (!currentLayer.length)
      throw new Error(
        'Invalid network structure for ONNX export: cannot resolve layered ordering.'
      );
    layerAccumulator.push(previousLayer);
    previousLayer = currentLayer;
    remainingHidden = remainingHidden.filter((h) => !currentLayer.includes(h));
  }
  // Append the last hidden layer and output layer.
  layerAccumulator.push(previousLayer);
  layerAccumulator.push(outputNodes);
  return layerAccumulator;
}

/** Validate that each non-input layer has homogeneous activation and is fully connected from previous layer. */
function validateLayerHomogeneityAndConnectivity(
  layers: any[][],
  network: Network
): void {
  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    /** Nodes in the source (previous) layer feeding current layer. */
    const previousLayerNodes = layers[layerIndex - 1];
    /** Nodes in the current destination layer being validated. */
    const currentLayerNodes = layers[layerIndex];
    /** Set of activation names encountered. */
    const activationNameSet = new Set(
      currentLayerNodes.map((n: any) => n.squash && n.squash.name)
    );
    if (activationNameSet.size > 1)
      throw new Error(
        `ONNX export error: Mixed activation functions detected in layer ${layerIndex}.`
      );
    for (const targetNode of currentLayerNodes) {
      for (const sourceNode of previousLayerNodes) {
        const isConnected = targetNode.connections.in.some(
          (conn: any) => conn.from === sourceNode
        );
        if (!isConnected) {
          throw new Error(
            `ONNX export error: Missing connection from node ${sourceNode.index} to node ${targetNode.index} in layer ${layerIndex}.`
          );
        }
      }
    }
  }
}

/** Construct the ONNX model graph (initializers + nodes) given validated layers. */
function buildOnnxModel(network: Network, layers: any[][]): OnnxModel {
  /** Input layer nodes (used for input tensor dimension). */
  const inputLayerNodes = layers[0];
  /** Output layer nodes (used for output tensor dimension). */
  const outputLayerNodes = layers[layers.length - 1];
  /** Mutable ONNX model under construction. */
  const model: OnnxModel = {
    graph: {
      inputs: [
        {
          name: 'input',
          type: {
            tensor_type: {
              elem_type: 1,
              shape: { dim: [{ dim_value: inputLayerNodes.length }] },
            },
          },
        },
      ],
      outputs: [
        {
          name: 'output',
          type: {
            tensor_type: {
              elem_type: 1,
              shape: { dim: [{ dim_value: outputLayerNodes.length }] },
            },
          },
        },
      ],
      initializer: [],
      node: [],
    },
  };
  /** Name of the tensor that feeds into the current Gemm. */
  let previousOutputName = 'input';
  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    const previousLayerNodes = layers[layerIndex - 1];
    const currentLayerNodes = layers[layerIndex];
    /** Flattened row-major weight matrix (rows = current layer neurons, cols = previous layer neurons). */
    const weightMatrixValues: number[] = [];
    /** Bias vector for current layer. */
    const biasVector: number[] = new Array(currentLayerNodes.length).fill(0);
    for (let neuronRow = 0; neuronRow < currentLayerNodes.length; neuronRow++) {
      const targetNode: any = currentLayerNodes[neuronRow];
      biasVector[neuronRow] = targetNode.bias;
      for (
        let neuronCol = 0;
        neuronCol < previousLayerNodes.length;
        neuronCol++
      ) {
        const sourceNode = previousLayerNodes[neuronCol];
        const inboundConn = targetNode.connections.in.find(
          (c: any) => c.from === sourceNode
        );
        weightMatrixValues.push(inboundConn ? inboundConn.weight : 0);
      }
    }
    /** Symbolic weight tensor name. */
    const weightTensorName = `W${layerIndex - 1}`;
    /** Symbolic bias tensor name. */
    const biasTensorName = `B${layerIndex - 1}`;
    /** Intermediate Gemm output name. */
    const gemmOutputName = `Gemm_${layerIndex}`;
    /** Post-activation output name (feeds next layer). */
    const activationOutputName = `Layer_${layerIndex}`;
    model.graph.initializer.push({
      name: weightTensorName,
      data_type: 1,
      dims: [currentLayerNodes.length, previousLayerNodes.length],
      float_data: weightMatrixValues,
    });
    model.graph.initializer.push({
      name: biasTensorName,
      data_type: 1,
      dims: [currentLayerNodes.length],
      float_data: biasVector,
    });
    // Activation node first (to preserve original order used historically in this project)
    model.graph.node.push({
      op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
      input: [gemmOutputName],
      output: [activationOutputName],
      name: `act_l${layerIndex}`,
    });
    (model.graph.node as any).push({
      op_type: 'Gemm',
      input: [previousOutputName, weightTensorName, biasTensorName],
      output: [gemmOutputName],
      name: `gemm_l${layerIndex}`,
      attributes: [
        { name: 'alpha', type: 'FLOAT', f: 1 },
        { name: 'beta', type: 'FLOAT', f: 1 },
        { name: 'transB', type: 'INT', i: 1 },
      ],
    });
    previousOutputName = activationOutputName;
  }
  return model;
}

/** Extract hidden layer sizes from ONNX initializers (weight tensors). */
function deriveHiddenLayerSizes(initializers: OnnxTensor[]): number[] {
  /** All 2D weight tensors (W*) representing layer transitions. */
  const weightTensors = initializers.filter(
    (t) => t.name.startsWith('W') && t.dims.length === 2
  );
  // All except the last weight tensor correspond to hidden layers (last maps into output layer)
  return weightTensors.slice(0, -1).map((t) => t.dims[0]);
}

/** Apply weights & biases from ONNX initializers onto the newly created network. */
function assignWeightsAndBiases(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[]
): void {
  /** Index of current layer transition (0..hiddenCount then output). */
  let layerOffset = 0; // which layer (0..hiddenCount then output)
  for (
    let initializerIndex = 0;
    initializerIndex < onnx.graph.initializer.length;
    initializerIndex += 2
  ) {
    /** Weight tensor for this layer transition. */
    const weightTensor = onnx.graph.initializer[initializerIndex];
    /** Bias tensor for this layer transition. */
    const biasTensor = onnx.graph.initializer[initializerIndex + 1];
    /** True if assigning into a hidden layer; false for final output layer. */
    const isHiddenLayer = layerOffset < hiddenLayerSizes.length;
    /** Destination layer nodes whose parameters are being assigned. */
    const currentLayerNodes = isHiddenLayer
      ? network.nodes
          .filter((n: any) => n.type === 'hidden')
          .slice(
            hiddenLayerSizes.slice(0, layerOffset).reduce((a, b) => a + b, 0),
            hiddenLayerSizes
              .slice(0, layerOffset + 1)
              .reduce((a, b) => a + b, 0)
          )
      : network.nodes.filter((n: any) => n.type === 'output');
    /** Source layer nodes providing incoming weights. */
    const previousLayerNodes =
      layerOffset === 0
        ? network.nodes.filter((n: any) => n.type === 'input')
        : network.nodes
            .filter((n: any) => n.type === 'hidden')
            .slice(
              hiddenLayerSizes
                .slice(0, layerOffset - 1)
                .reduce((a, b) => a + b, 0),
              hiddenLayerSizes.slice(0, layerOffset).reduce((a, b) => a + b, 0)
            );
    for (let row = 0; row < currentLayerNodes.length; row++) {
      for (let col = 0; col < previousLayerNodes.length; col++) {
        /** Existing forward connection (if present) from source to target. */
        const existingConn = previousLayerNodes[col].connections.out.find(
          (c: any) => c.to === currentLayerNodes[row]
        );
        if (existingConn)
          existingConn.weight =
            weightTensor.float_data[row * previousLayerNodes.length + col];
      }
      currentLayerNodes[row].bias = biasTensor.float_data[row];
    }
    layerOffset++;
  }
}

/** Map activation op_types from ONNX nodes back to internal activation functions. */
function assignActivationFunctions(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[]
): void {
  /** All ONNX activation nodes in traversal order. */
  const activationNodes = onnx.graph.node.filter((n) =>
    ['Tanh', 'Sigmoid', 'Logistic', 'Relu', 'Identity'].includes(n.op_type)
  );
  /** Index into activationNodes corresponding to current hidden layer. */
  let activationNodeIndex = 0;
  /** Flat list of hidden nodes (original network order). */
  const hiddenLayerNodes = network.nodes.filter(
    (n: any) => n.type === 'hidden'
  );
  /** Offset into hiddenLayerNodes tracking start of current layer segment. */
  let hiddenLayerOffset = 0;
  for (let layerIndex = 0; layerIndex < hiddenLayerSizes.length; layerIndex++) {
    /** Number of neurons in this hidden layer. */
    const size = hiddenLayerSizes[layerIndex];
    if (!size) continue;
    /** ONNX op_type representing this layer's activation. */
    const opType = activationNodes[activationNodeIndex]?.op_type;
    /** Resolved internal squash function for this layer. */
    let squashFn;
    switch (opType) {
      case 'Tanh':
        squashFn = methods.Activation.tanh;
        break;
      case 'Sigmoid':
      case 'Logistic':
        squashFn = methods.Activation.sigmoid;
        break;
      case 'Relu':
        squashFn = methods.Activation.relu;
        break;
      default:
        squashFn = methods.Activation.identity;
        break;
    }
    for (let i = 0; i < size; i++) {
      if (hiddenLayerNodes[hiddenLayerOffset + i])
        hiddenLayerNodes[hiddenLayerOffset + i].squash = squashFn;
    }
    hiddenLayerOffset += size;
    activationNodeIndex++;
  }
  /** Output nodes requiring activation assignment. */
  const outputLayerNodes = network.nodes.filter(
    (n: any) => n.type === 'output'
  );
  if (outputLayerNodes.length) {
    /** ONNX op_type of last activation node (assumed output layer). */
    const opType = activationNodes[activationNodes.length - 1]?.op_type;
    /** Resolved internal squash function for output layer. */
    let outputSquash;
    switch (opType) {
      case 'Tanh':
        outputSquash = methods.Activation.tanh;
        break;
      case 'Sigmoid':
      case 'Logistic':
        outputSquash = methods.Activation.sigmoid;
        break;
      case 'Relu':
        outputSquash = methods.Activation.relu;
        break;
      default:
        outputSquash = methods.Activation.identity;
        break;
    }
    outputLayerNodes.forEach((n: any) => (n.squash = outputSquash));
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Export a minimal multilayer perceptron Network to a lightweight ONNX JSON object.
 *
 * Steps:
 *  1. Rebuild connection cache ensuring up-to-date adjacency.
 *  2. Index nodes for error messaging.
 *  3. Infer strict layer ordering (throws if structure unsupported).
 *  4. Validate homogeneity & full connectivity layer-to-layer.
 *  5. Build initializer tensors (weights + biases) and node list (Gemm + activation pairs).
 *
 * Constraints: See module doc. Throws descriptive errors when assumptions violated.
 */
export function exportToONNX(network: Network): OnnxModel {
  rebuildConnectionsLocal(network as any);
  network.nodes.forEach((node: any, idx: number) => (node.index = idx));
  if (!network.connections || network.connections.length === 0)
    throw new Error('ONNX export currently only supports simple MLPs');
  /** Layered node arrays (input, hidden..., output) inferred for export. */
  const layers = inferLayerOrdering(network);
  validateLayerHomogeneityAndConnectivity(layers, network);
  return buildOnnxModel(network, layers);
}

/**
 * Import a model previously produced by {@link exportToONNX} into a fresh Network instance.
 *
 * Steps:
 *  1. Read input/output dimensions.
 *  2. Derive hidden layer sizes from weight tensor shapes.
 *  3. Create corresponding MLP with identical layer counts.
 *  4. Assign weights & biases.
 *  5. Map activation op_types back to internal activation functions.
 *  6. Rebuild flat connection list.
 *
 * Limitations: Only guaranteed for self-produced ONNX; inconsistent naming or ordering will break.
 */
export function importFromONNX(onnx: OnnxModel): Network {
  const { default: NetworkVal } = require('../network'); // dynamic import to avoid circular reference at module load
  /** Number of input features (dimension of input tensor). */
  const inputCount =
    onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value;
  /** Number of output neurons (dimension of output tensor). */
  const outputCount =
    onnx.graph.outputs[0].type.tensor_type.shape.dim[0].dim_value;
  /** Hidden layer sizes derived from weight tensor shapes. */
  const hiddenLayerSizes = deriveHiddenLayerSizes(onnx.graph.initializer);
  /** Newly constructed network mirroring the ONNX architecture. */
  const network: Network = NetworkVal.createMLP(
    inputCount,
    hiddenLayerSizes,
    outputCount
  );
  if (hiddenLayerSizes.length === 0) {
    // Edge case: single-layer perceptron (inputs -> outputs); prune hidden placeholders if any.
    network.nodes = [
      ...network.nodes.filter((n: any) => n.type === 'input'),
      ...network.nodes.filter((n: any) => n.type === 'output'),
    ];
    rebuildConnectionsLocal(network as any);
  }
  assignWeightsAndBiases(network, onnx, hiddenLayerSizes);
  assignActivationFunctions(network, onnx, hiddenLayerSizes);
  rebuildConnectionsLocal(network as any);
  return network;
}

export default { exportToONNX, importFromONNX };
