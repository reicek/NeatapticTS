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
import { emitShadowAttentionMappings } from './network.onnx.export-attention.utils';

export function createInitializedModel(
  networkLayers: NeatapticNode[][],
  currentOptions: OnnxBuildResolvedOptions,
): OnnxModel {
  const graphDimensions = createModelGraphDimensions(
    networkLayers,
    currentOptions.batchDimension,
  );

  const initializedModel = createBaseModel({
    inputDims: graphDimensions.inputDims,
    outputDims: graphDimensions.outputDims,
  });

  applyResolvedModelMetadata(initializedModel, currentOptions);
  return initializedModel;
}

export function collectRecurrentIndices(
  context: OnnxRecurrentCollectionContext,
): number[] {
  return collectRecurrentLayerIndices(context);
}

export function emitNonInputLayers(
  context: OnnxLayerEmissionContext,
): OnnxLayerEmissionResult {
  const nonInputLayerIndices = createNonInputLayerIndices(
    context.layers.length,
  );

  const hiddenSizesMetadata = collectHiddenLayerSizes(
    context.layers,
    nonInputLayerIndices,
  );

  const layerEmissionState = emitLayerGraphsForIndices(
    context,
    nonInputLayerIndices,
    'input',
  );

  return {
    previousOutputName: layerEmissionState.previousOutputName,
    hiddenSizesMetadata,
    layerOutputNamesByLayerIndex:
      layerEmissionState.layerOutputNamesByLayerIndex,
  };
}

export function applyPostProcessing(context: OnnxPostProcessingContext): void {
  emitFusedRecurrentHeuristics(
    context.model,
    context.layers,
    context.options.allowRecurrent,
    context.layerEmissionResult.previousOutputName,
  );

  emitShadowAttentionMappings(
    context.model,
    context.layers,
    context.options,
    context.layerEmissionResult.layerOutputNamesByLayerIndex,
    context.includeMetadata,
  );

  finalizeExportMetadata(
    context.model,
    context.layers,
    context.options,
    context.includeMetadata,
    context.layerEmissionResult.hiddenSizesMetadata,
    context.recurrentLayerIndices,
  );
}

function applyResolvedModelMetadata(
  initializedModel: OnnxModel,
  currentOptions: OnnxBuildResolvedOptions,
): void {
  applyModelMetadata({
    model: initializedModel,
    includeMetadata: currentOptions.includeMetadata,
    opset: currentOptions.opset,
    producerName: currentOptions.producerName,
    producerVersion: currentOptions.producerVersion,
    docString: currentOptions.docString,
  });
}

function createModelGraphDimensions(
  networkLayers: NeatapticNode[][],
  batchDimension: boolean,
): OnnxGraphDimensions {
  const inputLayerNodes = networkLayers[0];
  const outputLayerNodes = networkLayers.at(-1)!;

  return createGraphDimensions({
    inputWidth: inputLayerNodes.length,
    outputWidth: outputLayerNodes.length,
    batchDimension,
  });
}

function createNonInputLayerIndices(layerCount: number): number[] {
  return Array.from(
    { length: Math.max(layerCount - 1, 0) },
    (_unused, offset) => offset + 1,
  );
}

function collectHiddenLayerSizes(
  networkLayers: NeatapticNode[][],
  nonInputLayerIndices: number[],
): number[] {
  const outputLayerIndex = networkLayers.length - 1;

  return nonInputLayerIndices
    .filter((layerIndex) => layerIndex !== outputLayerIndex)
    .map((layerIndex) => networkLayers[layerIndex].length);
}

function emitLayerGraphsForIndices(
  context: OnnxLayerEmissionContext,
  nonInputLayerIndices: number[],
  initialOutputName: string,
): {
  previousOutputName: string;
  layerOutputNamesByLayerIndex: Map<number, string>;
} {
  const initialLayerOutputNamesByLayerIndex = new Map<number, string>([
    [0, initialOutputName],
  ]);

  return nonInputLayerIndices.reduce(
    (layerEmissionState, layerIndex) => {
      const currentOutputName = emitSingleLayerGraph(
        context,
        layerIndex,
        layerEmissionState.previousOutputName,
        layerEmissionState.layerOutputNamesByLayerIndex,
      );
      const nextLayerOutputNamesByLayerIndex = new Map(
        layerEmissionState.layerOutputNamesByLayerIndex,
      );
      nextLayerOutputNamesByLayerIndex.set(layerIndex, currentOutputName);
      return {
        previousOutputName: currentOutputName,
        layerOutputNamesByLayerIndex: nextLayerOutputNamesByLayerIndex,
      };
    },
    {
      previousOutputName: initialOutputName,
      layerOutputNamesByLayerIndex: initialLayerOutputNamesByLayerIndex,
    },
  );
}

function emitSingleLayerGraph(
  context: OnnxLayerEmissionContext,
  layerIndex: number,
  previousOutputName: string,
  layerOutputNamesByLayerIndex: Map<number, string>,
): string {
  return emitLayerGraph({
    model: context.model,
    layers: context.layers,
    options: context.options,
    layerIndex,
    previousOutputName,
    layerOutputNamesByLayerIndex,
    recurrentLayerIndices: context.recurrentLayerIndices,
    batchDimension: context.batchDimension,
    legacyNodeOrdering: context.legacyNodeOrdering,
  });
}
