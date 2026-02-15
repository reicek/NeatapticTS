import type NeatapticNode from '../node';
import type {
  Conv2DMapping,
  NodeInternals,
  OnnxExportOptions,
  OnnxModel,
  Pool2DMapping,
} from './network.onnx.types.utils';

/**
 * Build dense-layer weight matrix and bias vector.
 *
 * @param previousLayerNodes Source layer nodes.
 * @param currentLayerNodes Destination layer nodes.
 * @returns Flattened row-major weight matrix and bias vector.
 */
export function buildDenseWeightsAndBiases(
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): { weightMatrixValues: number[]; biasVector: number[] } {
  const weightMatrixValues: number[] = [];
  const biasVector: number[] = new Array(currentLayerNodes.length).fill(0);
  for (let rowIndex = 0; rowIndex < currentLayerNodes.length; rowIndex++) {
    const targetNode = currentLayerNodes[rowIndex];
    const targetNodeInternal = targetNode as unknown as NodeInternals;
    biasVector[rowIndex] = targetNodeInternal.bias;
    for (
      let sourceIndex = 0;
      sourceIndex < previousLayerNodes.length;
      sourceIndex++
    ) {
      const sourceNode = previousLayerNodes[sourceIndex];
      const inboundConnection = targetNodeInternal.connections.in.find(
        (connection) => connection.from === sourceNode,
      );
      weightMatrixValues.push(inboundConnection ? inboundConnection.weight : 0);
    }
  }
  return { weightMatrixValues, biasVector };
}

/**
 * Build a diagonal recurrent matrix from self-connections.
 *
 * @param currentLayerNodes Layer nodes.
 * @returns Flattened row-major recurrent matrix.
 */
export function buildDiagonalRecurrentWeights(
  currentLayerNodes: NeatapticNode[],
): number[] {
  const recurrentWeights: number[] = [];
  for (let rowIndex = 0; rowIndex < currentLayerNodes.length; rowIndex++) {
    for (let colIndex = 0; colIndex < currentLayerNodes.length; colIndex++) {
      if (rowIndex === colIndex) {
        const selfConnection = currentLayerNodes[rowIndex].connections.self[0];
        recurrentWeights.push(selfConnection ? selfConnection.weight : 0);
      } else {
        recurrentWeights.push(0);
      }
    }
  }
  return recurrentWeights;
}

/**
 * Emit optional pooling and flatten nodes after a layer output.
 *
 * @param params Pooling parameters.
 * @returns Final output tensor name after optional pooling/flatten.
 */
export function emitOptionalPoolingAndFlatten(params: {
  model: OnnxModel;
  options: OnnxExportOptions;
  layerIndex: number;
  sourceOutputName: string;
  poolSpec?: Pool2DMapping;
}): string {
  const { model, options, layerIndex, sourceOutputName, poolSpec } = params;
  if (!poolSpec) return sourceOutputName;

  const kernelShape = [poolSpec.kernelHeight, poolSpec.kernelWidth];
  const strides = [poolSpec.strideHeight, poolSpec.strideWidth];
  const pads = [
    poolSpec.padTop || 0,
    poolSpec.padLeft || 0,
    poolSpec.padBottom || 0,
    poolSpec.padRight || 0,
  ];
  const poolingOutputName = `Pool_${layerIndex}`;
  model.graph.node.push({
    op_type: poolSpec.type,
    input: [sourceOutputName],
    output: [poolingOutputName],
    name: `pool_after_l${layerIndex}`,
    attributes: [
      { name: 'kernel_shape', type: 'INTS', ints: kernelShape },
      { name: 'strides', type: 'INTS', ints: strides },
      { name: 'pads', type: 'INTS', ints: pads },
    ],
  });

  const maybeFlattenedOutput = maybeEmitFlatten(
    model,
    options.flattenAfterPooling,
    layerIndex,
    poolingOutputName,
  );

  appendIndexedMetadata(model, 'pool2d_layers', layerIndex);
  appendMetadataSpec(model, 'pool2d_specs', poolSpec);
  return maybeFlattenedOutput;
}

/**
 * Append an integer index to JSON-array metadata key.
 *
 * @param model Target model.
 * @param key Metadata key.
 * @param layerIndex Layer index to append.
 * @returns Nothing.
 */
export function appendIndexedMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void {
  model.metadata_props = model.metadata_props || [];
  const metadata = model.metadata_props.find(
    (property) => property.key === key,
  );
  if (metadata) {
    try {
      const parsed = JSON.parse(metadata.value);
      if (Array.isArray(parsed) && !parsed.includes(layerIndex)) {
        parsed.push(layerIndex);
        metadata.value = JSON.stringify(parsed);
      }
    } catch {
      metadata.value = JSON.stringify([layerIndex]);
    }
    return;
  }
  model.metadata_props.push({ key, value: JSON.stringify([layerIndex]) });
}

/**
 * Append a JSON object to JSON-array metadata key.
 *
 * @param model Target model.
 * @param key Metadata key.
 * @param spec Metadata object.
 * @returns Nothing.
 */
export function appendMetadataSpec(
  model: OnnxModel,
  key: string,
  spec: Conv2DMapping | Pool2DMapping,
): void {
  model.metadata_props = model.metadata_props || [];
  const metadata = model.metadata_props.find(
    (property) => property.key === key,
  );
  if (metadata) {
    try {
      const parsed = JSON.parse(metadata.value);
      if (Array.isArray(parsed)) {
        parsed.push({ ...spec });
        metadata.value = JSON.stringify(parsed);
      }
    } catch {
      metadata.value = JSON.stringify([spec]);
    }
    return;
  }
  model.metadata_props.push({ key, value: JSON.stringify([spec]) });
}

/**
 * Conditionally emit flatten node after pooling.
 *
 * @param model Target ONNX model.
 * @param flattenAfterPooling Whether flatten should be emitted.
 * @param layerIndex Layer index.
 * @param sourceOutputName Pool output name.
 * @returns Output tensor name after optional flatten.
 */
function maybeEmitFlatten(
  model: OnnxModel,
  flattenAfterPooling: boolean | undefined,
  layerIndex: number,
  sourceOutputName: string,
): string {
  if (!flattenAfterPooling) return sourceOutputName;
  const flattenOutputName = `PoolFlat_${layerIndex}`;
  model.graph.node.push({
    op_type: 'Flatten',
    input: [sourceOutputName],
    output: [flattenOutputName],
    name: `flatten_after_l${layerIndex}`,
    attributes: [{ name: 'axis', type: 'INT', i: 1 }],
  });
  appendIndexedMetadata(model, 'flatten_layers', layerIndex);
  return flattenOutputName;
}
