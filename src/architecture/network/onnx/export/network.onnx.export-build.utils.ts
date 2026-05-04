import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type {
  OnnxBuildResolvedOptions,
  OnnxExportOptions,
  OnnxGraphDimensions,
  OnnxLayerEmissionContext,
  OnnxLayerEmissionResult,
  OnnxPostProcessingContext,
  OnnxRecurrentCollectionContext,
} from './network.onnx.export.types';
import {
  applyModelMetadata,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
} from './network.onnx.export-setup.utils';
import { emitLayerGraph } from './layers/network.onnx.export-layer-graph.utils';
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
  options?: OnnxExportOptions,
): OnnxModel {
  // Step 1: Resolve stable export defaults and input options.
  const sourceOptions = options ?? {};
  const resolvedOptions = resolveBuildOptions(sourceOptions);
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

  // Step 6: Return fully constructed ONNX model.
  return model;

  /**
   * Resolve export options with defaults required by model construction.
   *
   * @param sourceOptions Raw export options.
   * @returns Resolved options used by this builder.
   */
  function resolveBuildOptions(
    sourceOptions: OnnxExportOptions,
  ): OnnxBuildResolvedOptions {
    // Step 1: Resolve caller-provided values with stable defaults.
    return {
      includeMetadata: sourceOptions.includeMetadata ?? false,
      opset: sourceOptions.opset ?? 18,
      batchDimension: sourceOptions.batchDimension ?? false,
      legacyNodeOrdering: sourceOptions.legacyNodeOrdering ?? false,
      producerName: sourceOptions.producerName ?? 'neataptic-ts',
      producerVersion: sourceOptions.producerVersion,
      docString: sourceOptions.docString,
    };
  }

  /**
   * Build base ONNX model and apply metadata.
   *
   * @param networkLayers Layered network topology.
   * @param currentOptions Resolved options.
   * @returns Initialized ONNX model.
   */
  function createInitializedModel(
    networkLayers: NeatapticNode[][],
    currentOptions: OnnxBuildResolvedOptions,
  ): OnnxModel {
    // Step 1: Resolve graph IO dimensions from layered topology.
    const graphDimensions = createModelGraphDimensions(
      networkLayers,
      currentOptions.batchDimension,
    );

    // Step 2: Create base ONNX model from resolved dimensions.
    const initializedModel = createBaseModel({
      inputDims: graphDimensions.inputDims,
      outputDims: graphDimensions.outputDims,
    });

    // Step 3: Attach metadata settings to initialized model.
    applyResolvedModelMetadata(initializedModel, currentOptions);
    return initializedModel;
  }

  /**
   * Resolve ONNX graph dimensions from input/output layer sizes.
   *
   * @param networkLayers Layered network topology.
   * @param batchDimension Whether batch dimension is enabled.
   * @returns Input and output dimensions.
   */
  function createModelGraphDimensions(
    networkLayers: NeatapticNode[][],
    batchDimension: boolean,
  ): OnnxGraphDimensions {
    // Step 1: Resolve first and last layers for IO sizing.
    const inputLayerNodes = networkLayers[0];
    const outputLayerNodes = networkLayers.at(-1)!;

    // Step 2: Build graph dimensions using setup utility.
    return createGraphDimensions({
      inputWidth: inputLayerNodes.length,
      outputWidth: outputLayerNodes.length,
      batchDimension,
    });
  }

  /**
   * Apply resolved build metadata to an initialized ONNX model.
   *
   * @param initializedModel Initialized ONNX model.
   * @param currentOptions Resolved build options.
   * @returns Nothing.
   */
  function applyResolvedModelMetadata(
    initializedModel: OnnxModel,
    currentOptions: OnnxBuildResolvedOptions,
  ): void {
    // Step 1: Delegate metadata assignment to setup utility.
    applyModelMetadata({
      model: initializedModel,
      includeMetadata: currentOptions.includeMetadata,
      opset: currentOptions.opset,
      producerName: currentOptions.producerName,
      producerVersion: currentOptions.producerVersion,
      docString: currentOptions.docString,
    });
  }

  /**
   * Collect recurrent layer indices needed during graph emission.
   *
   * @param context Recurrent collection context.
   * @returns Recurrent layer indices.
   */
  function collectRecurrentIndices(
    context: OnnxRecurrentCollectionContext,
  ): number[] {
    // Step 1: Delegate recurrent-layer discovery to setup utility.
    return collectRecurrentLayerIndices(context);
  }

  /**
   * Emit all non-input layers while tracking hidden-layer metadata.
   *
   * @param context Layer emission context.
   * @returns Final output name and hidden layer sizes metadata.
   */
  function emitNonInputLayers(
    context: OnnxLayerEmissionContext,
  ): OnnxLayerEmissionResult {
    // Step 1: Build deterministic layer index list for all non-input layers.
    const nonInputLayerIndices = createNonInputLayerIndices(
      context.layers.length,
    );

    // Step 2: Collect hidden layer sizes (excluding output layer).
    const hiddenSizesMetadata = collectHiddenLayerSizes(
      context.layers,
      nonInputLayerIndices,
    );

    // Step 3: Emit graph nodes for each non-input layer and fold output name.
    const previousOutputName = emitLayerGraphsForIndices(
      context,
      nonInputLayerIndices,
      'input',
    );

    return { previousOutputName, hiddenSizesMetadata };
  }

  /**
   * Apply export post-processing and metadata finalization.
   *
   * @param context Post-processing context.
   * @returns Nothing.
   */
  function applyPostProcessing(context: OnnxPostProcessingContext): void {
    // Step 1: Emit optional fused recurrent output adjustments.
    emitFusedRecurrentHeuristics(
      context.model,
      context.layers,
      context.options.allowRecurrent,
      context.layerEmissionResult.previousOutputName,
    );

    // Step 2: Finalize metadata using collected build artifacts.
    finalizeExportMetadata(
      context.model,
      context.layers,
      context.options,
      context.includeMetadata,
      context.layerEmissionResult.hiddenSizesMetadata,
      context.recurrentLayerIndices,
    );
  }

  /**
   * Build ordered indices for all non-input layers.
   *
   * @param layerCount Total number of layers.
   * @returns Layer indices from first hidden to output.
   */
  function createNonInputLayerIndices(layerCount: number): number[] {
    // Step 1: Generate ordered indices for all layers after the input layer.
    return Array.from(
      { length: Math.max(layerCount - 1, 0) },
      (_, offset) => offset + 1,
    );
  }

  /**
   * Collect metadata sizes for hidden layers only.
   *
   * @param networkLayers Layered network topology.
   * @param nonInputLayerIndices Layer indices excluding input.
   * @returns Hidden layer sizes metadata.
   */
  function collectHiddenLayerSizes(
    networkLayers: NeatapticNode[][],
    nonInputLayerIndices: number[],
  ): number[] {
    // Step 1: Resolve output layer index for exclusion.
    const outputLayerIndex = networkLayers.length - 1;

    // Step 2: Keep hidden layers only and map to their unit counts.
    return nonInputLayerIndices
      .filter((layerIndex) => layerIndex !== outputLayerIndex)
      .map((layerIndex) => networkLayers[layerIndex].length);
  }

  /**
   * Emit layer graphs in index order while folding the output tensor name.
   *
   * @param context Layer emission context.
   * @param nonInputLayerIndices Layer indices excluding input.
   * @param initialOutputName Initial input tensor name.
   * @returns Final output tensor name.
   */
  function emitLayerGraphsForIndices(
    context: OnnxLayerEmissionContext,
    nonInputLayerIndices: number[],
    initialOutputName: string,
  ): string {
    // Step 1: Fold layer emissions to the final output tensor name.
    return nonInputLayerIndices.reduce(
      (currentOutputName, layerIndex) =>
        emitSingleLayerGraph(context, layerIndex, currentOutputName),
      initialOutputName,
    );
  }

  /**
   * Emit graph nodes for a single layer index.
   *
   * @param context Layer emission context.
   * @param layerIndex Layer index to emit.
   * @param previousOutputName Previous tensor output name.
   * @returns Current layer output tensor name.
   */
  function emitSingleLayerGraph(
    context: OnnxLayerEmissionContext,
    layerIndex: number,
    previousOutputName: string,
  ): string {
    // Step 1: Emit ONNX operators for one layer and return its output tensor.
    return emitLayerGraph({
      model: context.model,
      layers: context.layers,
      options: context.options,
      layerIndex,
      previousOutputName,
      recurrentLayerIndices: context.recurrentLayerIndices,
      batchDimension: context.batchDimension,
      legacyNodeOrdering: context.legacyNodeOrdering,
    });
  }
}
