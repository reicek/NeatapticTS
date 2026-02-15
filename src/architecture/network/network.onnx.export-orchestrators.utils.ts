import type Network from '../network';
import type NeatapticNode from '../node';
import type {
  Conv2DMapping,
  NodeInternals,
  OnnxExportOptions,
  OnnxModel,
} from './network.onnx.types.utils';

/**
 * Assign stable index values to nodes for export diagnostics.
 *
 * @param network Source network.
 * @returns Nothing.
 */
export function assignExportNodeIndices(network: Network): void {
  network.nodes.forEach((node, index) => {
    const nodeInternal = node as unknown as NodeInternals & { index?: number };
    nodeInternal.index = index;
  });
}

/**
 * Collect heuristic LSTM grouping stubs from hidden layers.
 *
 * @param layers Layered network nodes.
 * @param allowRecurrent Whether recurrent export heuristics are enabled.
 * @returns Candidate LSTM pattern stubs.
 */
export function collectLstmPatternStubs(
  layers: NeatapticNode[][],
  allowRecurrent: boolean | undefined,
): { layerIndex: number; unitSize: number }[] {
  const lstmPatternStubs: { layerIndex: number; unitSize: number }[] = [];
  if (!allowRecurrent) return lstmPatternStubs;
  try {
    for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
      const hiddenLayer = layers[layerIndex];
      const totalNodes = hiddenLayer.length;
      if (totalNodes >= 10 && totalNodes % 5 === 0) {
        const segmentSize = totalNodes / 5;
        const memorySlice = hiddenLayer.slice(segmentSize * 2, segmentSize * 3);
        const allSelf = memorySlice.every(
          (nodeItem) =>
            (nodeItem as unknown as NodeInternals).connections.self.length ===
            1,
        );
        if (allSelf)
          lstmPatternStubs.push({ layerIndex, unitSize: segmentSize });
      }
    }
  } catch {
    /* ignore heuristic errors */
  }
  return lstmPatternStubs;
}

/**
 * Append heuristic conv inference metadata when requested.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param options Export options.
 * @returns Nothing.
 */
export function appendConvInferenceMetadata(
  model: OnnxModel,
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
): void {
  if (!options.includeMetadata) return;
  const inferredSpecs: (Conv2DMapping & { note?: string })[] = [];
  const inferredLayers: number[] = [];
  for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
    const previousWidth = layers[layerIndex - 1].length;
    const currentWidth = layers[layerIndex].length;
    const squareSize = Math.sqrt(previousWidth);
    if (Math.abs(squareSize - Math.round(squareSize)) > 1e-9) continue;
    const squareInt = Math.round(squareSize);
    for (const kernel of [3, 2]) {
      if (kernel >= squareInt) continue;
      const outSpatial = squareInt - kernel + 1;
      if (outSpatial * outSpatial === currentWidth) {
        const alreadyDeclared = options.conv2dMappings?.some(
          (mapping) => mapping.layerIndex === layerIndex,
        );
        if (alreadyDeclared) break;
        inferredLayers.push(layerIndex);
        inferredSpecs.push({
          layerIndex,
          inHeight: squareInt,
          inWidth: squareInt,
          inChannels: 1,
          kernelHeight: kernel,
          kernelWidth: kernel,
          strideHeight: 1,
          strideWidth: 1,
          outHeight: outSpatial,
          outWidth: outSpatial,
          outChannels: 1,
          note: 'heuristic_inferred_no_export_applied',
        });
        break;
      }
    }
  }
  if (!inferredLayers.length) return;
  model.metadata_props = model.metadata_props || [];
  model.metadata_props.push({
    key: 'conv2d_inferred_layers',
    value: JSON.stringify(inferredLayers),
  });
  model.metadata_props.push({
    key: 'conv2d_inferred_specs',
    value: JSON.stringify(inferredSpecs),
  });
}

/**
 * Append LSTM pattern stub metadata.
 *
 * @param model Target ONNX model.
 * @param lstmPatternStubs Pattern stubs.
 * @returns Nothing.
 */
export function appendLstmPatternStubMetadata(
  model: OnnxModel,
  lstmPatternStubs: { layerIndex: number; unitSize: number }[],
): void {
  if (!lstmPatternStubs.length) return;
  model.metadata_props = model.metadata_props || [];
  model.metadata_props.push({
    key: 'lstm_groups_stub',
    value: JSON.stringify(lstmPatternStubs),
  });
}
