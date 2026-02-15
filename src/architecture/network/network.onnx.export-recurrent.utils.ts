import type NeatapticNode from '../node';
import type { NodeInternals, OnnxModel } from './network.onnx.types.utils';
import {
  buildDenseWeightsAndBiases,
  buildDiagonalRecurrentWeights,
} from './network.onnx.export-layer-common.utils';
import { mapActivationToOnnx } from './network.onnx.layer-analysis.utils';

/**
 * Emit recurrent single-step layer representation.
 *
 * @param params Recurrent emission parameters.
 * @returns Output tensor name.
 */
export function emitRecurrentLayer(params: {
  model: OnnxModel;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
}): string {
  const {
    model,
    layerIndex,
    previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
  } = params;

  const { weightMatrixValues, biasVector } = buildDenseWeightsAndBiases(
    previousLayerNodes,
    currentLayerNodes,
  );
  const recurrentWeights = buildDiagonalRecurrentWeights(currentLayerNodes);
  const weightTensorName = `W${layerIndex - 1}`;
  const biasTensorName = `B${layerIndex - 1}`;
  const recurrentTensorName = `R${layerIndex - 1}`;

  emitDenseAndRecurrentInitializers(
    model,
    weightTensorName,
    biasTensorName,
    recurrentTensorName,
    previousLayerNodes.length,
    currentLayerNodes.length,
    weightMatrixValues,
    biasVector,
    recurrentWeights,
  );

  emitGemmNode(
    model,
    [previousOutputName, weightTensorName, biasTensorName],
    `Gemm_in_${layerIndex}`,
    `gemm_in_l${layerIndex}`,
  );

  const previousHiddenInputName =
    layerIndex === 1 ? 'hidden_prev' : `hidden_prev_l${layerIndex}`;
  emitGemmNode(
    model,
    [previousHiddenInputName, recurrentTensorName],
    `Gemm_rec_${layerIndex}`,
    `gemm_rec_l${layerIndex}`,
  );

  model.graph.node.push({
    op_type: 'Add',
    input: [`Gemm_in_${layerIndex}`, `Gemm_rec_${layerIndex}`],
    output: [`RecurrentSum_${layerIndex}`],
    name: `add_recurrent_l${layerIndex}`,
  });

  model.graph.node.push({
    op_type: mapActivationToOnnx(
      (currentLayerNodes[0] as unknown as NodeInternals).squash,
    ),
    input: [`RecurrentSum_${layerIndex}`],
    output: [`Layer_${layerIndex}`],
    name: `act_l${layerIndex}`,
  });

  return `Layer_${layerIndex}`;
}

/**
 * Emit dense and recurrent initializer tensors.
 *
 * @param model Target ONNX model.
 * @param weightTensorName Weight tensor name.
 * @param biasTensorName Bias tensor name.
 * @param recurrentTensorName Recurrent tensor name.
 * @param previousLayerWidth Previous layer width.
 * @param currentLayerWidth Current layer width.
 * @param weightMatrixValues Flattened weight values.
 * @param biasVector Bias values.
 * @param recurrentWeights Flattened recurrent matrix values.
 * @returns Nothing.
 */
function emitDenseAndRecurrentInitializers(
  model: OnnxModel,
  weightTensorName: string,
  biasTensorName: string,
  recurrentTensorName: string,
  previousLayerWidth: number,
  currentLayerWidth: number,
  weightMatrixValues: number[],
  biasVector: number[],
  recurrentWeights: number[],
): void {
  model.graph.initializer.push({
    name: weightTensorName,
    data_type: 1,
    dims: [currentLayerWidth, previousLayerWidth],
    float_data: weightMatrixValues,
  });
  model.graph.initializer.push({
    name: biasTensorName,
    data_type: 1,
    dims: [currentLayerWidth],
    float_data: biasVector,
  });
  model.graph.initializer.push({
    name: recurrentTensorName,
    data_type: 1,
    dims: [currentLayerWidth, currentLayerWidth],
    float_data: recurrentWeights,
  });
}

/**
 * Emit a Gemm node with shared attributes.
 *
 * @param model Target ONNX model.
 * @param inputNames Gemm input names.
 * @param outputName Gemm output name.
 * @param nodeName Gemm node name.
 * @returns Nothing.
 */
function emitGemmNode(
  model: OnnxModel,
  inputNames: string[],
  outputName: string,
  nodeName: string,
): void {
  model.graph.node.push({
    op_type: 'Gemm',
    input: inputNames,
    output: [outputName],
    name: nodeName,
    attributes: [
      { name: 'alpha', type: 'FLOAT', f: 1 },
      { name: 'beta', type: 'FLOAT', f: 1 },
      { name: 'transB', type: 'INT', i: 1 },
    ],
  });
}
