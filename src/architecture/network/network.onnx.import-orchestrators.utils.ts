import Connection from '../connection';
import type Network from '../network';
import { deriveHiddenLayerSizes } from './network.onnx.import-weights.utils';
import { reconstructFusedRecurrentLayers } from './network.onnx.import-fused-recurrent.utils';
import type {
  NodeInternals,
  OnnxModel,
  Pool2DMapping,
} from './network.onnx.types.utils';

export { reconstructFusedRecurrentLayers };

/**
 * Extract input/output counts and hidden layer sizes from ONNX model.
 *
 * @param onnx Source ONNX model.
 * @returns Parsed architecture dimensions.
 */
export function extractOnnxArchitecture(onnx: OnnxModel): {
  inputCount: number;
  outputCount: number;
  hiddenLayerSizes: number[];
} {
  const inputShapeDims = onnx.graph.inputs[0].type.tensor_type.shape.dim;
  const inputCount = (
    inputShapeDims[inputShapeDims.length - 1] as Record<string, number>
  ).dim_value;
  const outputShapeDims = onnx.graph.outputs[0].type.tensor_type.shape.dim;
  const outputCount = (
    outputShapeDims[outputShapeDims.length - 1] as Record<string, number>
  ).dim_value;
  const hiddenLayerSizes = deriveHiddenLayerSizes(
    onnx.graph.initializer,
    onnx.metadata_props,
  );
  return { inputCount, outputCount, hiddenLayerSizes };
}

/**
 * Remove placeholder hidden nodes for single-layer perceptron imports.
 *
 * @param network Target network.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @returns Nothing.
 */
export function pruneSingleLayerHiddenPlaceholders(
  network: Network,
  hiddenLayerSizes: number[],
): void {
  if (hiddenLayerSizes.length !== 0) return;
  network.nodes = [
    ...network.nodes.filter((node) => node.type === 'input'),
    ...network.nodes.filter((node) => node.type === 'output'),
  ];
}

/**
 * Restore recurrent self-connections from recurrent metadata and R tensors.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param metadata Parsed metadata properties.
 * @returns Nothing.
 */
export function restoreRecurrentSelfConnections(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: { key: string; value: string }[],
): void {
  const recurrentMeta = metadata.find(
    (property) => property.key === 'recurrent_single_step',
  );
  if (!recurrentMeta) return;

  const recurrentLayerIndices = parseRecurrentLayerIndices(recurrentMeta.value);
  let hiddenStart = 0;
  for (
    let hiddenIndex = 0;
    hiddenIndex < hiddenLayerSizes.length;
    hiddenIndex++
  ) {
    const hiddenLayerSize = hiddenLayerSizes[hiddenIndex];
    const layerNumber = hiddenIndex + 1;
    if (recurrentLayerIndices.includes(layerNumber)) {
      applyLayerSelfConnections(
        network,
        onnx,
        layerNumber,
        hiddenLayerSize,
        hiddenStart,
      );
    }
    hiddenStart += hiddenLayerSize;
  }
}

/**
 * Attach optional pooling metadata from ONNX model to network instance.
 *
 * @param network Target network.
 * @param metadata ONNX metadata.
 * @returns Nothing.
 */
export function attachOnnxPoolingMetadata(
  network: Network,
  metadata: { key: string; value: string }[],
): void {
  try {
    const poolLayersMeta = metadata.find(
      (property) => property.key === 'pool2d_layers',
    );
    const poolSpecsMeta = metadata.find(
      (property) => property.key === 'pool2d_specs',
    );
    if (poolLayersMeta) {
      const networkWithPooling = network as Network & {
        _onnxPooling?: { layers: number[]; specs: Pool2DMapping[] };
      };
      networkWithPooling._onnxPooling = {
        layers: JSON.parse(poolLayersMeta.value) as number[],
        specs: poolSpecsMeta
          ? (JSON.parse(poolSpecsMeta.value) as Pool2DMapping[])
          : [],
      };
    }
  } catch {
    /* ignore pooling attachment errors */
  }
}

/**
 * Parse recurrent layer indices metadata.
 *
 * @param rawMetadataValue Raw metadata JSON string.
 * @returns Normalized recurrent layer indices.
 */
function parseRecurrentLayerIndices(rawMetadataValue: string): number[] {
  try {
    const parsed = JSON.parse(rawMetadataValue);
    if (Array.isArray(parsed)) return parsed;
    return [0];
  } catch {
    return [0];
  }
}

/**
 * Apply one hidden layer diagonal recurrent self-weights.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param layerNumber One-based hidden layer number.
 * @param hiddenLayerSize Hidden layer size.
 * @param hiddenStart Global hidden offset.
 * @returns Nothing.
 */
function applyLayerSelfConnections(
  network: Network,
  onnx: OnnxModel,
  layerNumber: number,
  hiddenLayerSize: number,
  hiddenStart: number,
): void {
  const recurrentTensorName = `R${layerNumber - 1}`;
  const recurrentInitializer = onnx.graph.initializer.find(
    (tensor) => tensor.name === recurrentTensorName,
  );
  if (!recurrentInitializer) return;

  const hiddenNodes = network.nodes.filter(
    (nodeItem) => nodeItem.type === 'hidden',
  );
  for (let unitIndex = 0; unitIndex < hiddenLayerSize; unitIndex++) {
    const node = hiddenNodes[hiddenStart + unitIndex];
    const nodeInternal = node as unknown as NodeInternals;
    const recurrentWeight =
      recurrentInitializer.float_data[unitIndex * hiddenLayerSize + unitIndex];
    let selfConnection = nodeInternal.connections.self[0];
    if (!selfConnection) {
      selfConnection = Connection.acquire(node, node, recurrentWeight);
      nodeInternal.connections.self.push(selfConnection);
      node.connections.in.push(selfConnection);
      node.connections.out.push(selfConnection);
    } else {
      selfConnection.weight = recurrentWeight;
    }
  }
}
