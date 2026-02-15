import type NeatapticNode from '../node';
import type {
  NodeInternals,
  OnnxDimension,
  OnnxModel,
} from './network.onnx.types.utils';

/**
 * Build tensor dimensions for model input and output, optionally with symbolic batch dimension.
 *
 * @param inputWidth Number of input features.
 * @param outputWidth Number of output features.
 * @param batchDimension Whether to include symbolic batch dimension `N`.
 * @returns Input and output dimension arrays for ONNX value info.
 */
export function createGraphDimensions(
  inputWidth: number,
  outputWidth: number,
  batchDimension: boolean,
): { inputDims: OnnxDimension[]; outputDims: OnnxDimension[] } {
  const inputDims = batchDimension
    ? [{ dim_param: 'N' }, { dim_value: inputWidth }]
    : [{ dim_value: inputWidth }];
  const outputDims = batchDimension
    ? [{ dim_param: 'N' }, { dim_value: outputWidth }]
    : [{ dim_value: outputWidth }];
  return { inputDims, outputDims };
}

/**
 * Create the base ONNX model shell with graph input/output declarations.
 *
 * @param inputDims Input tensor dimensions.
 * @param outputDims Output tensor dimensions.
 * @returns Initialized ONNX model with empty initializer/node lists.
 */
export function createBaseModel(
  inputDims: OnnxDimension[],
  outputDims: OnnxDimension[],
): OnnxModel {
  return {
    graph: {
      inputs: [
        {
          name: 'input',
          type: {
            tensor_type: {
              elem_type: 1,
              shape: { dim: inputDims },
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
              shape: { dim: outputDims },
            },
          },
        },
      ],
      initializer: [],
      node: [],
    },
  };
}

/**
 * Attach producer and opset metadata to a model when metadata emission is enabled.
 *
 * @param model Target model to mutate.
 * @param includeMetadata Whether metadata emission is enabled.
 * @param opset ONNX opset version.
 * @param producerName Producer name.
 * @param producerVersion Producer version override.
 * @param docString Optional document string override.
 * @returns Nothing.
 */
export function applyModelMetadata(
  model: OnnxModel,
  includeMetadata: boolean,
  opset: number,
  producerName: string,
  producerVersion?: string,
  docString?: string,
): void {
  if (!includeMetadata) return;
  const pkgVersion = '0.0.0';
  model.ir_version = 9;
  model.opset_import = [{ version: opset, domain: '' }];
  model.producer_name = producerName;
  model.producer_version = producerVersion || pkgVersion;
  model.doc_string =
    docString ||
    'Exported from NeatapticTS ONNX exporter (phases 1-2 baseline)';
}

/**
 * Detect hidden layers with self-recurrence and add matching previous-state graph inputs.
 *
 * @param model Target ONNX model under construction.
 * @param layers Layered network node arrays.
 * @param allowRecurrent Whether recurrent export is enabled.
 * @param recurrentSingleStep Whether single-step recurrent form is enabled.
 * @param batchDimension Whether symbolic batch dimension is enabled.
 * @returns Export-layer indices with recurrent self-connections.
 */
export function collectRecurrentLayerIndices(
  model: OnnxModel,
  layers: NeatapticNode[][],
  allowRecurrent: boolean | undefined,
  recurrentSingleStep: boolean | undefined,
  batchDimension: boolean,
): number[] {
  const recurrentLayerIndices: number[] = [];
  if (!allowRecurrent || !recurrentSingleStep) return recurrentLayerIndices;
  for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
    const hiddenLayerNodes = layers[layerIndex];
    const hasSelfRecurrence = hiddenLayerNodes.some(
      (node) => (node as unknown as NodeInternals).connections.self.length > 0,
    );
    if (!hasSelfRecurrence) continue;
    recurrentLayerIndices.push(layerIndex);
    const previousStateInputName =
      layerIndex === 1 ? 'hidden_prev' : `hidden_prev_l${layerIndex}`;
    model.graph.inputs.push({
      name: previousStateInputName,
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: batchDimension
              ? [{ dim_param: 'N' }, { dim_value: hiddenLayerNodes.length }]
              : [{ dim_value: hiddenLayerNodes.length }],
          },
        },
      },
    });
  }
  return recurrentLayerIndices;
}
