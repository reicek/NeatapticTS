import type Network from '../../network';
import { buildOnnxModel } from './network.onnx.export-build.utils';
import {
  appendConvInferenceMetadata,
  appendLstmPatternStubMetadata,
  assignExportNodeIndices,
  collectLstmPatternStubs,
} from './network.onnx.export-orchestrators.utils';
import {
  inferLayerOrdering,
  rebuildConnectionsLocal,
  validateLayerHomogeneityAndConnectivity,
} from '../network.onnx.layer-analysis.utils';
import type { OnnxExportOptions } from './network.onnx.export.types';
import type { OnnxModel } from '../schema/network.onnx.schema.types';

/**
 * Execute the complete ONNX export flow for one network instance.
 *
 * High-level behavior:
 *  1. Rebuild runtime connection caches and assign stable export indices.
 *  2. Infer layered ordering and collect recurrent-pattern stubs.
 *  3. Validate structural constraints for the requested export options.
 *  4. Build ONNX graph payload and append inference-oriented metadata.
 *
 * @param network Source network to serialize.
 * @param options Optional ONNX export controls.
 * @returns ONNX-like model payload.
 */
export function runOnnxExportFlow(
  network: Network,
  options: OnnxExportOptions = {},
): OnnxModel {
  // Step 1: Normalize runtime graph caches for deterministic export traversal.
  rebuildConnectionsLocal(network);
  assignExportNodeIndices(network);

  // Step 2: Infer ordered layers and collect heuristic recurrent metadata stubs.
  const layers = inferLayerOrdering(network);
  const lstmPatternStubs = collectLstmPatternStubs(
    layers,
    options.allowRecurrent,
  );

  // Step 3: Validate layer constraints before graph materialization.
  validateLayerHomogeneityAndConnectivity(layers, network, options);

  // Step 4: Build the ONNX-like payload and append post-build metadata.
  const model = buildOnnxModel(network, layers, options);
  appendConvInferenceMetadata(model, layers, options);
  appendLstmPatternStubMetadata(model, lstmPatternStubs);
  return model;
}
