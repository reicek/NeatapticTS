import type NeatapticNode from '../node';
import type {
  NodeInternals,
  OnnxExportOptions,
  OnnxModel,
} from './network.onnx.types.utils';
import {
  emitDenseLayer,
  emitPerNeuronLayer,
} from './network.onnx.export-dense.utils';
import { tryEmitConvLayer } from './network.onnx.export-conv.utils';
import { emitRecurrentLayer } from './network.onnx.export-recurrent.utils';

/**
 * Layer build context used while emitting ONNX graph nodes.
 */
export interface LayerBuildContext {
  model: OnnxModel;
  layers: NeatapticNode[][];
  options: OnnxExportOptions;
  layerIndex: number;
  previousOutputName: string;
  recurrentLayerIndices: number[];
  batchDimension: boolean;
  legacyNodeOrdering: boolean;
}

/**
 * Emit one export layer graph segment and return the produced output tensor name.
 *
 * @param context Layer build context.
 * @returns Output tensor name produced by this layer.
 */
export function emitLayerGraph(context: LayerBuildContext): string {
  const {
    model,
    layers,
    options,
    layerIndex,
    recurrentLayerIndices,
    batchDimension,
    legacyNodeOrdering,
  } = context;

  const previousLayerNodes = layers[layerIndex - 1];
  const currentLayerNodes = layers[layerIndex];
  const isOutputLayer = layerIndex === layers.length - 1;

  const convOutputName = tryEmitConvLayer({
    model,
    options,
    layerIndex,
    previousOutputName: context.previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
  });
  if (convOutputName) return convOutputName;

  const hasMixedActivations = checkMixedActivations(currentLayerNodes, options);
  if (shouldEmitRecurrent(recurrentLayerIndices, layerIndex, isOutputLayer)) {
    ensureRecurrentLayerSupportsActivations(layerIndex, hasMixedActivations);
    return emitRecurrentLayer({
      model,
      layerIndex,
      previousOutputName: context.previousOutputName,
      previousLayerNodes,
      currentLayerNodes,
    });
  }

  if (!hasMixedActivations) {
    return emitDenseLayer({
      model,
      layerIndex,
      previousOutputName: context.previousOutputName,
      previousLayerNodes,
      currentLayerNodes,
      legacyNodeOrdering,
      options,
    });
  }

  return emitPerNeuronLayer({
    model,
    layerIndex,
    previousOutputName: context.previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
    options,
    batchDimension,
  });
}

/**
 * Determine whether a layer has mixed activation functions.
 *
 * @param currentLayerNodes Current layer nodes.
 * @param options Export options.
 * @returns Whether mixed activations are present and enabled.
 */
function checkMixedActivations(
  currentLayerNodes: NeatapticNode[],
  options: OnnxExportOptions,
): boolean {
  if (!options.allowMixedActivations) return false;
  const activationNames = new Set(
    currentLayerNodes.map((node) => {
      const nodeInternal = node as unknown as NodeInternals;
      return nodeInternal.squash && nodeInternal.squash.name;
    }),
  );
  return activationNames.size > 1;
}

/**
 * Determine whether recurrent single-step emission applies to a layer.
 *
 * @param recurrentLayerIndices Recurrent hidden layer indices.
 * @param layerIndex Current layer index.
 * @param isOutputLayer Whether current layer is output layer.
 * @returns Whether recurrent emission path should be used.
 */
function shouldEmitRecurrent(
  recurrentLayerIndices: number[],
  layerIndex: number,
  isOutputLayer: boolean,
): boolean {
  return recurrentLayerIndices.includes(layerIndex) && !isOutputLayer;
}

/**
 * Ensure recurrent layers do not use unsupported mixed activations.
 *
 * @param layerIndex Layer index.
 * @param hasMixedActivations Whether layer has mixed activations.
 * @returns Nothing.
 */
function ensureRecurrentLayerSupportsActivations(
  layerIndex: number,
  hasMixedActivations: boolean,
): void {
  if (!hasMixedActivations) return;
  throw new Error(
    `Recurrent export does not yet support mixed activations in hidden layer ${layerIndex}.`,
  );
}
