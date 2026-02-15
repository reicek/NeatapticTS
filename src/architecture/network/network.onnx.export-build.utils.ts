import type Network from '../network';
import type NeatapticNode from '../node';
import type { OnnxExportOptions, OnnxModel } from './network.onnx.types.utils';
import {
  applyModelMetadata,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
} from './network.onnx.export-setup.utils';
import { emitLayerGraph } from './network.onnx.export-layer-graph.utils';
import {
  emitFusedRecurrentHeuristics,
  finalizeExportMetadata,
} from './network.onnx.export-postprocess.utils';

/**
 * Construct ONNX graph (initializers + nodes) from validated layered network structure.
 *
 * @param network Source network (retained for API compatibility).
 * @param layers Layered nodes including input and output layers.
 * @param options Export options.
 * @returns ONNX model.
 */
export function buildOnnxModel(
  network: Network,
  layers: NeatapticNode[][],
  options: OnnxExportOptions = {},
): OnnxModel {
  const {
    includeMetadata = false,
    opset = 18,
    batchDimension = false,
    legacyNodeOrdering = false,
    producerName = 'neataptic-ts',
    producerVersion,
    docString,
  } = options;
  void network;

  const inputLayerNodes = layers[0];
  const outputLayerNodes = layers[layers.length - 1];
  const { inputDims, outputDims } = createGraphDimensions(
    inputLayerNodes.length,
    outputLayerNodes.length,
    batchDimension,
  );
  const model = createBaseModel(inputDims, outputDims);
  applyModelMetadata(
    model,
    includeMetadata,
    opset,
    producerName,
    producerVersion,
    docString,
  );

  let previousOutputName = 'input';
  const recurrentLayerIndices = collectRecurrentLayerIndices(
    model,
    layers,
    options.allowRecurrent,
    options.recurrentSingleStep,
    batchDimension,
  );
  const hiddenSizesMetadata: number[] = [];

  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    const currentLayerNodes = layers[layerIndex];
    const isOutputLayer = layerIndex === layers.length - 1;
    if (!isOutputLayer) hiddenSizesMetadata.push(currentLayerNodes.length);

    previousOutputName = emitLayerGraph({
      model,
      layers,
      options,
      layerIndex,
      previousOutputName,
      recurrentLayerIndices,
      batchDimension,
      legacyNodeOrdering,
    });
  }

  emitFusedRecurrentHeuristics(
    model,
    layers,
    options.allowRecurrent,
    previousOutputName,
  );
  finalizeExportMetadata(
    model,
    layers,
    options,
    includeMetadata,
    hiddenSizesMetadata,
    recurrentLayerIndices,
  );
  return model;
}
