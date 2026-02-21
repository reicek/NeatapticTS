import type Network from '../../network';
import {
  assignActivationFunctions,
  assignWeightsAndBiases,
  buildOnnxModel,
  inferLayerOrdering,
  rebuildConnectionsLocal,
  validateLayerHomogeneityAndConnectivity,
} from './network.onnx.utils';
import {
  appendConvInferenceMetadata,
  appendLstmPatternStubMetadata,
  assignExportNodeIndices,
  collectLstmPatternStubs,
} from './network.onnx.export-orchestrators.utils';
import {
  attachOnnxPoolingMetadata,
  extractOnnxArchitecture,
  pruneSingleLayerHiddenPlaceholders,
  reconstructFusedRecurrentLayers,
  restoreRecurrentSelfConnections,
} from './network.onnx.import-orchestrators.utils';
import { loadRuntimeFactories } from './network.onnx.runtime-load.utils';
import type {
  Conv2DMapping,
  OnnxExportOptions,
  OnnxModel,
  Pool2DMapping,
} from './network.onnx.utils.types';

export type { Conv2DMapping, OnnxExportOptions, OnnxModel, Pool2DMapping };

/**
 * Export a Neataptic network to a minimal ONNX-like JSON model structure.
 *
 * @param network Source network.
 * @param options Optional export flags.
 * @returns ONNX-like model.
 */
export function exportToONNX(
  network: Network,
  options: OnnxExportOptions = {},
): OnnxModel {
  rebuildConnectionsLocal(network);
  assignExportNodeIndices(network);

  const layers = inferLayerOrdering(network);
  const lstmPatternStubs = collectLstmPatternStubs(
    layers,
    options.allowRecurrent,
  );
  validateLayerHomogeneityAndConnectivity(layers, network, options);

  const model = buildOnnxModel(network, layers, options);
  appendConvInferenceMetadata(model, layers, options);
  appendLstmPatternStubMetadata(model, lstmPatternStubs);

  return model;
}

/**
 * Import an ONNX-like JSON model into a Neataptic network instance.
 *
 * @param onnx ONNX-like model.
 * @returns Reconstructed network.
 */
export function importFromONNX(onnx: OnnxModel): Network {
  const architecture = extractOnnxArchitecture(onnx);

  const { perceptronFactory, layerModule } = loadRuntimeFactories();

  const network = perceptronFactory(
    architecture.inputCount,
    ...architecture.hiddenLayerSizes,
    architecture.outputCount,
  ) as Network;

  pruneSingleLayerHiddenPlaceholders(network, architecture.hiddenLayerSizes);
  assignWeightsAndBiases(
    network,
    onnx,
    architecture.hiddenLayerSizes,
    onnx.metadata_props,
  );
  assignActivationFunctions(network, onnx, architecture.hiddenLayerSizes);

  const metadata = onnx.metadata_props || [];
  restoreRecurrentSelfConnections(
    network,
    onnx,
    architecture.hiddenLayerSizes,
    metadata,
  );
  reconstructFusedRecurrentLayers(
    network,
    onnx,
    architecture.hiddenLayerSizes,
    layerModule as unknown as Record<string, (...args: unknown[]) => unknown>,
    metadata,
  );

  rebuildConnectionsLocal(network);
  attachOnnxPoolingMetadata(network, metadata);

  return network;
}

export default {
  exportToONNX,
  importFromONNX,
};
