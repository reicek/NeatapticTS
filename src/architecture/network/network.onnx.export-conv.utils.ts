import type NeatapticNode from '../node';
import type {
  Conv2DMapping,
  NodeInternals,
  OnnxExportOptions,
  OnnxModel,
} from './network.onnx.types.utils';
import {
  appendIndexedMetadata,
  appendMetadataSpec,
  emitOptionalPoolingAndFlatten,
} from './network.onnx.export-layer-common.utils';
import { mapActivationToOnnx } from './network.onnx.layer-analysis.utils';

/**
 * Try to emit a conv-mapped layer.
 *
 * @param params Conv emission parameters.
 * @returns New output tensor name when handled, otherwise undefined.
 */
export function tryEmitConvLayer(params: {
  model: OnnxModel;
  options: OnnxExportOptions;
  layerIndex: number;
  previousOutputName: string;
  previousLayerNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
}): string | undefined {
  const {
    model,
    options,
    layerIndex,
    previousOutputName,
    previousLayerNodes,
    currentLayerNodes,
  } = params;

  const convSpec = options.conv2dMappings?.find(
    (mapping) => mapping.layerIndex === layerIndex,
  );
  if (!convSpec) return undefined;
  if (!isValidConvShape(convSpec, previousLayerNodes, currentLayerNodes)) {
    logConvShapeMismatch(
      layerIndex,
      convSpec,
      previousLayerNodes,
      currentLayerNodes,
    );
    return undefined;
  }

  const convParams = collectConvParams(
    convSpec,
    previousLayerNodes,
    currentLayerNodes,
  );
  const tensorNames = emitConvInitializers(
    model,
    layerIndex,
    convSpec,
    convParams,
  );
  const activationOutputName = emitConvAndActivationNodes(
    model,
    layerIndex,
    previousOutputName,
    tensorNames,
    convSpec,
    currentLayerNodes,
  );

  const pooledOutputName = emitOptionalPoolingAndFlatten({
    model,
    options,
    layerIndex,
    sourceOutputName: activationOutputName,
    poolSpec: options.pool2dMappings?.find(
      (pooling) => pooling.afterLayerIndex === layerIndex,
    ),
  });

  appendIndexedMetadata(model, 'conv2d_layers', layerIndex);
  appendMetadataSpec(model, 'conv2d_specs', convSpec);
  return pooledOutputName;
}

/**
 * Validate declared Conv2D dimensions against actual layer widths.
 *
 * @param convSpec Conv mapping spec.
 * @param previousLayerNodes Previous layer nodes.
 * @param currentLayerNodes Current layer nodes.
 * @returns Whether dimensions are compatible.
 */
function isValidConvShape(
  convSpec: Conv2DMapping,
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): boolean {
  const previousWidthExpected =
    convSpec.inHeight * convSpec.inWidth * convSpec.inChannels;
  const currentWidthExpected =
    convSpec.outChannels * convSpec.outHeight * convSpec.outWidth;
  return (
    previousWidthExpected === previousLayerNodes.length &&
    currentWidthExpected === currentLayerNodes.length
  );
}

/**
 * Log Conv2D dimension mismatch warning.
 *
 * @param layerIndex Layer index.
 * @param convSpec Conv mapping spec.
 * @param previousLayerNodes Previous layer nodes.
 * @param currentLayerNodes Current layer nodes.
 * @returns Nothing.
 */
function logConvShapeMismatch(
  layerIndex: number,
  convSpec: Conv2DMapping,
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): void {
  const previousWidthExpected =
    convSpec.inHeight * convSpec.inWidth * convSpec.inChannels;
  const currentWidthExpected =
    convSpec.outChannels * convSpec.outHeight * convSpec.outWidth;
  console.warn(
    `Conv2D mapping for layer ${layerIndex} skipped: dimension mismatch (expected prev=${previousWidthExpected} got ${previousLayerNodes.length}; expected this=${currentWidthExpected} got ${currentLayerNodes.length}).`,
  );
}

/**
 * Collect Conv initializer values from representative neurons.
 *
 * @param convSpec Conv mapping spec.
 * @param previousLayerNodes Previous layer nodes.
 * @param currentLayerNodes Current layer nodes.
 * @returns Flattened kernel weights and bias values.
 */
function collectConvParams(
  convSpec: Conv2DMapping,
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): { weights: number[]; biases: number[] } {
  const weights: number[] = [];
  const biases: number[] = [];

  for (let outChannel = 0; outChannel < convSpec.outChannels; outChannel++) {
    const representativeIndex =
      outChannel * convSpec.outHeight * convSpec.outWidth;
    const representativeNeuron = currentLayerNodes[representativeIndex];
    const representativeNeuronInternal =
      representativeNeuron as unknown as NodeInternals;
    biases.push(representativeNeuronInternal.bias);

    for (let inChannel = 0; inChannel < convSpec.inChannels; inChannel++) {
      for (let kernelRow = 0; kernelRow < convSpec.kernelHeight; kernelRow++) {
        for (let kernelCol = 0; kernelCol < convSpec.kernelWidth; kernelCol++) {
          const sourceNode = resolveConvSourceNode(
            convSpec,
            previousLayerNodes,
            inChannel,
            kernelRow,
            kernelCol,
          );
          const inboundConnection =
            representativeNeuronInternal.connections.in.find(
              (connection) => connection.from === sourceNode,
            );
          weights.push(inboundConnection ? inboundConnection.weight : 0);
        }
      }
    }
  }

  return { weights, biases };
}

/**
 * Resolve one source node index used by Conv kernel mapping.
 *
 * @param convSpec Conv mapping spec.
 * @param previousLayerNodes Previous layer nodes.
 * @param inChannel Input channel index.
 * @param kernelRow Kernel row index.
 * @param kernelCol Kernel column index.
 * @returns Source node.
 */
function resolveConvSourceNode(
  convSpec: Conv2DMapping,
  previousLayerNodes: NeatapticNode[],
  inChannel: number,
  kernelRow: number,
  kernelCol: number,
): NeatapticNode {
  const inputFeatureIndex =
    inChannel * (convSpec.inHeight * convSpec.inWidth) +
    kernelRow * convSpec.inWidth +
    kernelCol;
  return previousLayerNodes[inputFeatureIndex];
}

/**
 * Emit Conv weight and bias initializers.
 *
 * @param model Target ONNX model.
 * @param layerIndex Layer index.
 * @param convSpec Conv mapping spec.
 * @param params Flattened Conv parameters.
 * @returns Conv tensor names.
 */
function emitConvInitializers(
  model: OnnxModel,
  layerIndex: number,
  convSpec: Conv2DMapping,
  params: { weights: number[]; biases: number[] },
): { convWeightName: string; convBiasName: string } {
  const convWeightName = `ConvW${layerIndex - 1}`;
  const convBiasName = `ConvB${layerIndex - 1}`;

  model.graph.initializer.push({
    name: convWeightName,
    data_type: 1,
    dims: [
      convSpec.outChannels,
      convSpec.inChannels,
      convSpec.kernelHeight,
      convSpec.kernelWidth,
    ],
    float_data: params.weights,
  });
  model.graph.initializer.push({
    name: convBiasName,
    data_type: 1,
    dims: [convSpec.outChannels],
    float_data: params.biases,
  });

  return { convWeightName, convBiasName };
}

/**
 * Emit Conv and activation nodes.
 *
 * @param model Target ONNX model.
 * @param layerIndex Layer index.
 * @param previousOutputName Previous output tensor name.
 * @param tensorNames Conv tensor names.
 * @param convSpec Conv mapping spec.
 * @param currentLayerNodes Current layer nodes.
 * @returns Activation output tensor name.
 */
function emitConvAndActivationNodes(
  model: OnnxModel,
  layerIndex: number,
  previousOutputName: string,
  tensorNames: { convWeightName: string; convBiasName: string },
  convSpec: Conv2DMapping,
  currentLayerNodes: NeatapticNode[],
): string {
  const pads = [
    convSpec.padTop || 0,
    convSpec.padLeft || 0,
    convSpec.padBottom || 0,
    convSpec.padRight || 0,
  ];
  const convOutputName = `Conv_${layerIndex}`;
  const activationOutputName = `Layer_${layerIndex}`;

  model.graph.node.push({
    op_type: 'Conv',
    input: [
      previousOutputName,
      tensorNames.convWeightName,
      tensorNames.convBiasName,
    ],
    output: [convOutputName],
    name: `conv_l${layerIndex}`,
    attributes: [
      {
        name: 'kernel_shape',
        type: 'INTS',
        ints: [convSpec.kernelHeight, convSpec.kernelWidth],
      },
      {
        name: 'strides',
        type: 'INTS',
        ints: [convSpec.strideHeight, convSpec.strideWidth],
      },
      { name: 'pads', type: 'INTS', ints: pads },
    ],
  });

  const activationOperator =
    convSpec.activation || mapActivationToOnnx(currentLayerNodes[0].squash);
  model.graph.node.push({
    op_type: activationOperator,
    input: [convOutputName],
    output: [activationOutputName],
    name: `act_conv_l${layerIndex}`,
  });

  return activationOutputName;
}
