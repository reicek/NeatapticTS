import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type { OnnxExportOptions } from './network.onnx.export.types';
import { pruneIdentityActivationNodes } from './network.onnx.export-optimization.utils';
import { validateOnnxModelShapes } from './network.onnx.export-shape-validation.utils';
import { resolveBuildOptions } from './network.onnx.export-build.options.utils';
import {
  applyStaticQuantizationCalibrationPostProcessing,
  applyStaticDenseQuantizationPostProcessing,
  applyStaticConvQuantizationPostProcessing,
  applyDynamicDenseQuantizationGuidancePostProcessing,
} from './network.onnx.export-build.quantization.utils';
import { applyStorageFp16PostProcessing } from './network.onnx.export-build.storage-fp16.utils';
import {
  applyPostProcessing,
  collectRecurrentIndices,
  createInitializedModel,
  emitNonInputLayers,
} from './network.onnx.export-build.emit.utils';

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
  options?: OnnxExportOptions,
): OnnxModel {
  // Step 1: Resolve stable export defaults and input options.
  const sourceOptions = options ?? {};
  const resolvedOptions = resolveBuildOptions(sourceOptions, layers.length);
  void network;

  // Step 2: Initialize base ONNX model with graph dimensions and metadata.
  const model = createInitializedModel(layers, resolvedOptions);

  // Step 3: Collect recurrent layer indices required during graph emission.
  const recurrentLayerIndices = collectRecurrentIndices({
    model,
    layers,
    options: sourceOptions,
    batchDimension: resolvedOptions.batchDimension,
  });

  // Step 4: Emit all non-input layers and collect hidden-size metadata.
  const layerEmissionResult = emitNonInputLayers({
    model,
    layers,
    options: sourceOptions,
    recurrentLayerIndices,
    batchDimension: resolvedOptions.batchDimension,
    legacyNodeOrdering: resolvedOptions.legacyNodeOrdering,
  });

  // Step 5: Apply post-processing and finalize export metadata.
  applyPostProcessing({
    model,
    layers,
    options: sourceOptions,
    includeMetadata: resolvedOptions.includeMetadata,
    recurrentLayerIndices,
    layerEmissionResult,
  });

  // Step 6: Emit calibration-owned quantization parameters for the narrow supported subset.
  applyStaticQuantizationCalibrationPostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 7: Lower the narrow same-family dense subset into qlinear affine nodes.
  applyStaticDenseQuantizationPostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 8: Lower the narrow same-family Conv subset into qlinear spatial nodes.
  applyStaticConvQuantizationPostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 9: Insert the narrow dense-only dynamic guidance lane when requested.
  applyDynamicDenseQuantizationGuidancePostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 10: Apply storage-fp16 rewrites only for the narrow supported subset.
  applyStorageFp16PostProcessing(model, sourceOptions, recurrentLayerIndices);

  // Step 11: Prune exact Identity activation nodes before validation.
  pruneIdentityActivationNodes(model);

  // Step 12: Validate exporter-owned tensor dimensions before returning.
  validateOnnxModelShapes(model);

  // Step 13: Return fully constructed ONNX model.
  return model;
}
