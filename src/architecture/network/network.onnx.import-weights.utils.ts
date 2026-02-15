import type Network from '../network';
import Connection from '../connection';
import type NeatapticNode from '../node';
import type {
  Conv2DMapping,
  NodeInternals,
  OnnxModel,
  OnnxTensor,
} from './network.onnx.types.utils';

/**
 * Extract hidden layer sizes from ONNX initializers (weight tensors).
 *
 * @param initializers ONNX initializer tensors.
 * @param metadataProps Optional ONNX metadata properties.
 * @returns Hidden layer sizes in order.
 */
export function deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps?: { key: string; value: string }[],
): number[] {
  const metadataLayerSizes = metadataProps?.find(
    (property) => property.key === 'layer_sizes',
  );
  if (metadataLayerSizes) {
    try {
      const parsed = JSON.parse(metadataLayerSizes.value);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      /* ignore parse error */
    }
  }

  const layerMap: Record<
    string,
    { aggregated?: OnnxTensor; perNeuron: OnnxTensor[] }
  > = {};
  initializers
    .filter((tensor) => tensor.name.startsWith('W'))
    .forEach((tensor) => {
      const match = /^W(\d+)(?:_n(\d+))?$/i.exec(tensor.name);
      if (!match) return;
      const layerIndex = match[1];
      layerMap[layerIndex] = layerMap[layerIndex] || { perNeuron: [] };
      if (match[2] !== undefined) layerMap[layerIndex].perNeuron.push(tensor);
      else layerMap[layerIndex].aggregated = tensor;
    });

  const sortedLayerIndices = Object.keys(layerMap)
    .map(Number)
    .sort((left, right) => left - right);
  if (!sortedLayerIndices.length) return [];

  const hiddenLayerSizes: number[] = [];
  for (
    let sortedIndex = 0;
    sortedIndex < sortedLayerIndices.length - 1;
    sortedIndex++
  ) {
    const layerEntry = layerMap[String(sortedLayerIndices[sortedIndex])];
    if (layerEntry.aggregated)
      hiddenLayerSizes.push(layerEntry.aggregated.dims[0]);
    else hiddenLayerSizes.push(layerEntry.perNeuron.length);
  }
  return hiddenLayerSizes;
}

/**
 * Assign weights and biases from ONNX initializers to a newly created network.
 *
 * @param network Target network to mutate.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param metadataProps Optional ONNX metadata properties.
 * @returns Nothing.
 */
export function assignWeightsAndBiases(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps?: { key: string; value: string }[],
): void {
  const initializerMap: Record<string, OnnxTensor> = {};
  onnx.graph.initializer.forEach((tensor) => {
    initializerMap[tensor.name] = tensor;
  });

  const layerIndices = new Set<number>();
  Object.keys(initializerMap).forEach((name) => {
    const match = /^W(\d+)(?:_n(\d+))?$/i.exec(name);
    if (match) layerIndices.add(Number(match[1]));
  });

  const sortedIndices = Array.from(layerIndices).sort(
    (left, right) => left - right,
  );
  sortedIndices.forEach((layerIndex, sequentialIndex) => {
    const isHiddenLayer = sequentialIndex < hiddenLayerSizes.length;
    const currentLayerNodes = isHiddenLayer
      ? network.nodes
          .filter((node) => node.type === 'hidden')
          .slice(
            hiddenLayerSizes
              .slice(0, sequentialIndex)
              .reduce((sum, value) => sum + value, 0),
            hiddenLayerSizes
              .slice(0, sequentialIndex + 1)
              .reduce((sum, value) => sum + value, 0),
          )
      : network.nodes.filter((node) => node.type === 'output');

    const previousLayerNodes =
      sequentialIndex === 0
        ? network.nodes.filter((node) => node.type === 'input')
        : network.nodes
            .filter((node) => node.type === 'hidden')
            .slice(
              hiddenLayerSizes
                .slice(0, sequentialIndex - 1)
                .reduce((sum, value) => sum + value, 0),
              hiddenLayerSizes
                .slice(0, sequentialIndex)
                .reduce((sum, value) => sum + value, 0),
            );

    const aggregatedWeights = initializerMap[`W${layerIndex}`];
    if (aggregatedWeights) {
      const biasTensor = initializerMap[`B${layerIndex}`];
      for (let rowIndex = 0; rowIndex < currentLayerNodes.length; rowIndex++) {
        const currentNodeInternal = currentLayerNodes[
          rowIndex
        ] as unknown as NodeInternals;
        for (
          let colIndex = 0;
          colIndex < previousLayerNodes.length;
          colIndex++
        ) {
          const previousNodeInternal = previousLayerNodes[
            colIndex
          ] as unknown as NodeInternals;
          const connection = previousNodeInternal.connections.out.find(
            (candidate) => candidate.to === currentLayerNodes[rowIndex],
          );
          if (connection) {
            connection.weight =
              aggregatedWeights.float_data[
                rowIndex * previousLayerNodes.length + colIndex
              ];
          }
        }
        currentNodeInternal.bias = biasTensor.float_data[rowIndex];
      }
    } else {
      currentLayerNodes.forEach((node, neuronIndex) => {
        const nodeInternal = node as unknown as NodeInternals;
        const weightTensor = initializerMap[`W${layerIndex}_n${neuronIndex}`];
        const biasTensor = initializerMap[`B${layerIndex}_n${neuronIndex}`];
        if (!weightTensor || !biasTensor) return;
        for (
          let colIndex = 0;
          colIndex < previousLayerNodes.length;
          colIndex++
        ) {
          const previousNodeInternal = previousLayerNodes[
            colIndex
          ] as unknown as NodeInternals;
          const connection = previousNodeInternal.connections.out.find(
            (candidate) => candidate.to === node,
          );
          if (connection) connection.weight = weightTensor.float_data[colIndex];
        }
        nodeInternal.bias = biasTensor.float_data[0];
      });
    }
  });

  try {
    const metadata = metadataProps || [];
    const convLayersMeta = metadata.find(
      (property) => property.key === 'conv2d_layers',
    );
    const convSpecsMeta = metadata.find(
      (property) => property.key === 'conv2d_specs',
    );
    if (!(convLayersMeta && convSpecsMeta)) return;

    const convLayers: number[] = JSON.parse(convLayersMeta.value);
    const convSpecs: Conv2DMapping[] = JSON.parse(convSpecsMeta.value);
    convLayers.forEach((layerExportIndex) => {
      const convSpec = convSpecs.find(
        (specification) => specification.layerIndex === layerExportIndex,
      );
      if (!convSpec) return;
      const hiddenIndex = layerExportIndex - 1;
      if (hiddenIndex < 0 || hiddenIndex >= hiddenLayerSizes.length) return;

      const hiddenNodes = network.nodes.filter(
        (node) => node.type === 'hidden',
      );
      const start = hiddenLayerSizes
        .slice(0, hiddenIndex)
        .reduce((sum, value) => sum + value, 0);
      const end = start + hiddenLayerSizes[hiddenIndex];
      const layerNodes = hiddenNodes.slice(start, end);

      const previousLayerNodes =
        hiddenIndex === 0
          ? network.nodes.filter((node) => node.type === 'input')
          : hiddenNodes.slice(
              hiddenLayerSizes
                .slice(0, hiddenIndex - 1)
                .reduce((sum, value) => sum + value, 0),
              hiddenLayerSizes
                .slice(0, hiddenIndex)
                .reduce((sum, value) => sum + value, 0),
            );

      const convWeightTensor = onnx.graph.initializer.find(
        (tensor) => tensor.name === `ConvW${layerExportIndex - 1}`,
      );
      const convBiasTensor = onnx.graph.initializer.find(
        (tensor) => tensor.name === `ConvB${layerExportIndex - 1}`,
      );
      if (!convWeightTensor || !convBiasTensor) return;
      const resolvedConvWeightTensor = convWeightTensor;

      const [outChannels, inChannels, kernelHeight, kernelWidth] =
        convWeightTensor.dims as [number, number, number, number];
      if (
        outChannels !== convSpec.outChannels ||
        inChannels !== convSpec.inChannels ||
        kernelHeight !== convSpec.kernelHeight ||
        kernelWidth !== convSpec.kernelWidth
      ) {
        return;
      }

      const strideHeight = convSpec.strideHeight;
      const strideWidth = convSpec.strideWidth;
      const padTop = convSpec.padTop || 0;
      const padLeft = convSpec.padLeft || 0;
      const inputHeight = convSpec.inHeight;
      const inputWidth = convSpec.inWidth;
      const outputHeight = convSpec.outHeight;
      const outputWidth = convSpec.outWidth;

      function getKernelWeight(
        outChannel: number,
        inChannel: number,
        kernelRow: number,
        kernelCol: number,
      ): number {
        const index =
          ((outChannel * inChannels + inChannel) * kernelHeight + kernelRow) *
            kernelWidth +
          kernelCol;
        return resolvedConvWeightTensor.float_data[index];
      }

      for (let outChannel = 0; outChannel < outChannels; outChannel++) {
        for (let outRow = 0; outRow < outputHeight; outRow++) {
          for (let outCol = 0; outCol < outputWidth; outCol++) {
            const neuronLinearIndex =
              outChannel * (outputHeight * outputWidth) +
              outRow * outputWidth +
              outCol;
            const neuron = layerNodes[neuronLinearIndex];
            if (!neuron) continue;
            const neuronInternal = neuron as unknown as NodeInternals;
            neuronInternal.bias = convBiasTensor.float_data[outChannel];

            const inboundConnectionMap = new Map<NeatapticNode, Connection>();
            neuronInternal.connections.in.forEach((connection) => {
              inboundConnectionMap.set(connection.from, connection);
            });

            for (let inChannel = 0; inChannel < inChannels; inChannel++) {
              const inputRowBase = outRow * strideHeight - padTop;
              const inputColBase = outCol * strideWidth - padLeft;
              for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
                for (let kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                  const inputRow = inputRowBase + kernelRow;
                  const inputCol = inputColBase + kernelCol;
                  if (
                    inputRow < 0 ||
                    inputRow >= inputHeight ||
                    inputCol < 0 ||
                    inputCol >= inputWidth
                  ) {
                    continue;
                  }
                  const inputFeatureIndex =
                    inChannel * (inputHeight * inputWidth) +
                    inputRow * inputWidth +
                    inputCol;
                  const sourceNode = previousLayerNodes[inputFeatureIndex];
                  if (!sourceNode) continue;
                  const connection = inboundConnectionMap.get(sourceNode);
                  if (connection) {
                    connection.weight = getKernelWeight(
                      outChannel,
                      inChannel,
                      kernelRow,
                      kernelCol,
                    );
                  }
                }
              }
            }
          }
        }
      }
    });
  } catch {
    /* Swallow conv reconstruction errors (experimental). */
  }
}
