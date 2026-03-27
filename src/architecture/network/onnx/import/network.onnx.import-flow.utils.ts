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
import { rebuildConnectionsLocal } from '../network.onnx.layer-analysis.utils';
import { loadRuntimeFactories } from './network.onnx.runtime-load.utils';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type { OnnxRuntimeLayerFactoryMap } from '../network.onnx.utils.types';

/**
 * ONNX import orchestration for rebuilding a NeatapticTS runtime network.
 *
 * This file is the chapter-level tour guide for the import folder. The import
 * path is intentionally staged so a reader can follow the same questions the
 * runtime asks while restoring a model:
 * 1. What architecture should be rebuilt?
 * 2. Which runtime factories should own the scaffold?
 * 3. How do dense weights and activations map back onto nodes?
 * 4. Which recurrent and pooling hints need a second pass?
 *
 * The neighboring files each own one of those stages. Keeping this overview on
 * the flow file makes the generated folder README read like an import pipeline
 * instead of an alphabetical pile of helper files.
 *
 * Example:
 * ```ts
 * const restored = runOnnxImportFlow(onnxModel);
 * const output = restored.activate([0.2, 0.8]);
 * ```
 */

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
