import type NeatapticNode from '../node';
import type {
  NodeInternals,
  OnnxExportOptions,
  OnnxModel,
} from './network.onnx.types.utils';
import {
  buildDenseWeightsAndBiases,
  emitOptionalPoolingAndFlatten,
} from './network.onnx.export-layer-common.utils';
import { mapActivationToOnnx } from './network.onnx.layer-analysis.utils';

/**
 * Emit dense layer representation.
 *
 * @param params Dense emission parameters.
 * @returns Output tensor name.
 */
export function emitDenseLayer(params: {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  legacyNodeOrdering: boolean;
  options: OnnxExportOptions;
}): string {
  const {
    model,
    layerIndex,
    previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
    legacyNodeOrdering,
    options,
  } = params;

  const tensorNames = emitDenseInitializers(
    model,
    layerIndex,
    previousLayerNodes,
    currentLayerNodes,
  );
  const graphNames = {
    gemmOutputName: `Gemm_${layerIndex}`,
    activationOutputName: `Layer_${layerIndex}`,
  };

  emitDenseActivationSequence(
    model,
    layerIndex,
    previousOutputName,
    tensorNames,
    graphNames,
    (currentLayerNodes[0] as unknown as NodeInternals).squash,
    legacyNodeOrdering,
  );

  return emitOptionalPoolingAndFlatten({
    model,
    options,
    layerIndex,
    sourceOutputName: graphNames.activationOutputName,
    poolSpec: options.pool2dMappings?.find(
      (pooling) => pooling.afterLayerIndex === layerIndex,
    ),
  });
}

/**
 * Emit per-neuron decomposition layer representation.
 *
 * @param params Per-neuron emission parameters.
 * @returns Output tensor name.
 */
export function emitPerNeuronLayer(params: {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
  options: OnnxExportOptions;
  batchDimension: boolean;
}): string {
  const {
    model,
    layerIndex,
    previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
    options,
    batchDimension,
  } = params;

  const perNeuronActivationOutputs = currentLayerNodes.map(
    (targetNode, neuronIndex) =>
      emitPerNeuronSubgraph(
        model,
        layerIndex,
        neuronIndex,
        previousOutputName,
        previousLayerNodes,
        targetNode,
      ),
  );

  const layerOutputName = `Layer_${layerIndex}`;
  model.graph.node.push({
    op_type: 'Concat',
    input: perNeuronActivationOutputs,
    output: [layerOutputName],
    name: `concat_l${layerIndex}`,
    attributes: [{ name: 'axis', type: 'INT', i: batchDimension ? 1 : 0 }],
  });

  return emitOptionalPoolingAndFlatten({
    model,
    options,
    layerIndex,
    sourceOutputName: layerOutputName,
    poolSpec: options.pool2dMappings?.find(
      (pooling) => pooling.afterLayerIndex === layerIndex,
    ),
  });
}

/**
 * Emit dense initializers and return tensor names.
 *
 * @param model Target ONNX model.
 * @param layerIndex Layer index.
 * @param previousLayerNodes Previous layer nodes.
 * @param currentLayerNodes Current layer nodes.
 * @returns Tensor names.
 */
function emitDenseInitializers(
  model: OnnxModel,
  layerIndex: number,
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): { weightTensorName: string; biasTensorName: string } {
  const { weightMatrixValues, biasVector } = buildDenseWeightsAndBiases(
    previousLayerNodes,
    currentLayerNodes,
  );
  const weightTensorName = `W${layerIndex - 1}`;
  const biasTensorName = `B${layerIndex - 1}`;
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
  return { weightTensorName, biasTensorName };
}

/**
 * Emit Gemm and activation nodes using requested ordering.
 *
 * @param model Target ONNX model.
 * @param layerIndex Layer index.
 * @param previousOutputName Previous tensor output name.
 * @param tensorNames Weight and bias tensor names.
 * @param graphNames Gemm and activation output names.
 * @param squash Activation function.
 * @param legacyNodeOrdering Whether to preserve legacy ordering.
 * @returns Nothing.
 */
function emitDenseActivationSequence(
  model: OnnxModel,
  layerIndex: number,
  previousOutputName: string,
  tensorNames: { weightTensorName: string; biasTensorName: string },
  graphNames: { gemmOutputName: string; activationOutputName: string },
  squash: ((x: number, derivate?: boolean) => number) & { name?: string },
  legacyNodeOrdering: boolean,
): void {
  const activationNode = createActivationNode(
    layerIndex,
    graphNames.gemmOutputName,
    graphNames.activationOutputName,
    squash,
  );
  const gemmNode = createGemmNode(
    layerIndex,
    previousOutputName,
    tensorNames.weightTensorName,
    tensorNames.biasTensorName,
    graphNames.gemmOutputName,
  );

  if (!legacyNodeOrdering) {
    model.graph.node.push(gemmNode);
    model.graph.node.push(activationNode);
    return;
  }

  model.graph.node.push(activationNode);
  model.graph.node.push(gemmNode);
}

/**
 * Create dense Gemm node definition.
 *
 * @param layerIndex Layer index.
 * @param previousOutputName Previous output name.
 * @param weightTensorName Weight tensor name.
 * @param biasTensorName Bias tensor name.
 * @param gemmOutputName Gemm output name.
 * @returns ONNX Gemm node payload.
 */
function createGemmNode(
  layerIndex: number,
  previousOutputName: string,
  weightTensorName: string,
  biasTensorName: string,
  gemmOutputName: string,
): {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes: { name: string; type: string; f?: number; i?: number }[];
} {
  return {
    op_type: 'Gemm',
    input: [previousOutputName, weightTensorName, biasTensorName],
    output: [gemmOutputName],
    name: `gemm_l${layerIndex}`,
    attributes: [
      { name: 'alpha', type: 'FLOAT', f: 1 },
      { name: 'beta', type: 'FLOAT', f: 1 },
      { name: 'transB', type: 'INT', i: 1 },
    ],
  };
}

/**
 * Create dense activation node definition.
 *
 * @param layerIndex Layer index.
 * @param gemmOutputName Gemm output name.
 * @param activationOutputName Activation output name.
 * @param squash Activation function.
 * @returns ONNX activation node payload.
 */
function createActivationNode(
  layerIndex: number,
  gemmOutputName: string,
  activationOutputName: string,
  squash: ((x: number, derivate?: boolean) => number) & { name?: string },
): {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
} {
  return {
    op_type: mapActivationToOnnx(squash),
    input: [gemmOutputName],
    output: [activationOutputName],
    name: `act_l${layerIndex}`,
  };
}

/**
 * Emit per-neuron Gemm + activation subgraph.
 *
 * @param model Target ONNX model.
 * @param layerIndex Layer index.
 * @param neuronIndex Neuron index.
 * @param previousOutputName Previous output tensor name.
 * @param previousLayerNodes Previous layer nodes.
 * @param targetNode Target node.
 * @returns Per-neuron activation output name.
 */
function emitPerNeuronSubgraph(
  model: OnnxModel,
  layerIndex: number,
  neuronIndex: number,
  previousOutputName: string,
  previousLayerNodes: NeatapticNode[],
  targetNode: NeatapticNode,
): string {
  const targetNodeInternal = targetNode as unknown as NodeInternals;
  const weightRow = buildSingleNeuronWeightRow(
    targetNodeInternal,
    previousLayerNodes,
  );

  const weightTensorName = `W${layerIndex - 1}_n${neuronIndex}`;
  const biasTensorName = `B${layerIndex - 1}_n${neuronIndex}`;
  const gemmOutputName = `Gemm_${layerIndex}_n${neuronIndex}`;
  const activationOutputName = `Layer_${layerIndex}_n${neuronIndex}`;

  model.graph.initializer.push({
    name: weightTensorName,
    data_type: 1,
    dims: [1, previousLayerNodes.length],
    float_data: weightRow,
  });
  model.graph.initializer.push({
    name: biasTensorName,
    data_type: 1,
    dims: [1],
    float_data: [targetNodeInternal.bias],
  });

  model.graph.node.push({
    op_type: 'Gemm',
    input: [previousOutputName, weightTensorName, biasTensorName],
    output: [gemmOutputName],
    name: `gemm_l${layerIndex}_n${neuronIndex}`,
    attributes: [
      { name: 'alpha', type: 'FLOAT', f: 1 },
      { name: 'beta', type: 'FLOAT', f: 1 },
      { name: 'transB', type: 'INT', i: 1 },
    ],
  });

  model.graph.node.push({
    op_type: mapActivationToOnnx(targetNodeInternal.squash),
    input: [gemmOutputName],
    output: [activationOutputName],
    name: `act_l${layerIndex}_n${neuronIndex}`,
  });

  return activationOutputName;
}

/**
 * Build one neuron's incoming weight row against previous layer.
 *
 * @param targetNodeInternal Target node internals.
 * @param previousLayerNodes Previous layer nodes.
 * @returns Weight row values.
 */
function buildSingleNeuronWeightRow(
  targetNodeInternal: NodeInternals,
  previousLayerNodes: NeatapticNode[],
): number[] {
  const weightRow: number[] = [];
  for (
    let sourceIndex = 0;
    sourceIndex < previousLayerNodes.length;
    sourceIndex++
  ) {
    const sourceNode = previousLayerNodes[sourceIndex];
    const inboundConnection = targetNodeInternal.connections.in.find(
      (connection) => connection.from === sourceNode,
    );
    weightRow.push(inboundConnection ? inboundConnection.weight : 0);
  }
  return weightRow;
}
