import type Network from '../../network';
import { assignActivationFunctions } from './network.onnx.import-activations.utils';
import {
  attachOnnxPoolingMetadata,
  extractOnnxArchitecture,
  pruneSingleLayerHiddenPlaceholders,
  reconstructFusedRecurrentLayers,
  restoreRecurrentSelfConnections,
} from './network.onnx.import-orchestrators.utils';
import { assignWeightsAndBiases } from './network.onnx.import-weights.utils';
import { rebuildConnectionsLocal } from './network.onnx.layer-analysis.utils';
import { loadRuntimeFactories } from './network.onnx.runtime-load.utils';
import type {
  OnnxModel,
  OnnxRuntimeLayerFactoryMap,
} from './network.onnx.utils.types';

/**
 * Execute the complete ONNX import flow and reconstruct a runtime network.
 *
 * High-level behavior:
 *  1. Extract architecture dimensions and build a perceptron scaffold.
 *  2. Restore dense parameters and activation functions.
 *  3. Reconstruct recurrent/pooling metadata and rebuild connection caches.
 *
 * @param onnx ONNX-like model payload to reconstruct.
 * @returns Reconstructed network instance.
 */
export function runOnnxImportFlow(onnx: OnnxModel): Network {
  // Step 1: Parse architecture metadata and construct the base perceptron.
  const architecture = extractOnnxArchitecture(onnx);
  const { perceptronFactory, layerModule } = loadRuntimeFactories();
  const network = perceptronFactory(
    architecture.inputCount,
    ...architecture.hiddenLayerSizes,
    architecture.outputCount,
  ) as Network;

  // Step 2: Restore feed-forward parameters and activation operations.
  pruneSingleLayerHiddenPlaceholders(network, architecture.hiddenLayerSizes);
  assignWeightsAndBiases(
    network,
    onnx,
    architecture.hiddenLayerSizes,
    onnx.metadata_props,
  );
  assignActivationFunctions(network, onnx, architecture.hiddenLayerSizes);

  // Step 3: Reconstruct recurrent additions, refresh caches, and attach metadata.
  const metadata = onnx.metadata_props ?? [];
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
    layerModule as OnnxRuntimeLayerFactoryMap,
    metadata,
  );
  rebuildConnectionsLocal(network);
  attachOnnxPoolingMetadata(network, metadata);
  return network;
}
