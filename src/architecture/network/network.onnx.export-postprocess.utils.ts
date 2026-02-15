import type NeatapticNode from '../node';
import type {
  NodeInternals,
  OnnxModel,
  OnnxExportOptions,
} from './network.onnx.types.utils';

/**
 * Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param allowRecurrent Whether recurrent export is enabled.
 * @param previousOutputName Current graph output name (kept for backward-compatible emission semantics).
 * @returns Nothing.
 */
export function emitFusedRecurrentHeuristics(
  model: OnnxModel,
  layers: NeatapticNode[][],
  allowRecurrent: boolean | undefined,
  previousOutputName: string,
): void {
  if (!allowRecurrent) return;

  for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
    const currentLayerNodes = layers[layerIndex];
    const currentSize = currentLayerNodes.length;

    model.metadata_props = model.metadata_props || [];
    if (currentSize >= 8 && currentSize < 10) {
      model.metadata_props.push({
        key: 'rnn_pattern_fallback',
        value: JSON.stringify({
          layer: layerIndex,
          reason: 'size_between_gru_lstm_thresholds',
        }),
      });
    }

    tryEmitFusedLstm(
      model,
      layers,
      layerIndex,
      currentLayerNodes,
      currentSize,
      previousOutputName,
    );
    tryEmitFusedGru(model, layers, layerIndex, currentLayerNodes, currentSize);
  }
}

/**
 * Finalize export metadata and optional conv-sharing validation.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param options Export options.
 * @param includeMetadata Whether metadata emission is enabled.
 * @param hiddenSizesMetadata Hidden-layer sizes collected during emission.
 * @param recurrentLayerIndices Recurrent layer indices.
 * @returns Nothing.
 */
export function finalizeExportMetadata(
  model: OnnxModel,
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
  includeMetadata: boolean,
  hiddenSizesMetadata: number[],
  recurrentLayerIndices: number[],
): void {
  if (!includeMetadata) return;

  model.metadata_props = model.metadata_props || [];
  model.metadata_props.push({
    key: 'layer_sizes',
    value: JSON.stringify(hiddenSizesMetadata),
  });
  if (recurrentLayerIndices.length) {
    model.metadata_props.push({
      key: 'recurrent_single_step',
      value: JSON.stringify(recurrentLayerIndices),
    });
  }

  if (
    options.validateConvSharing &&
    options.conv2dMappings &&
    options.conv2dMappings.length
  ) {
    const verifiedLayers: number[] = [];
    const mismatchedLayers: number[] = [];

    for (const convSpec of options.conv2dMappings) {
      const layerIndex = convSpec.layerIndex;
      const previousLayerNodes = layers[layerIndex - 1];
      const currentLayerNodes = layers[layerIndex];
      if (!currentLayerNodes || !previousLayerNodes) continue;

      const representativeKernels: number[][] = [];
      let isConsistent = true;
      for (
        let outChannel = 0;
        outChannel < convSpec.outChannels;
        outChannel++
      ) {
        const representativeIndex =
          outChannel * (convSpec.outHeight * convSpec.outWidth);
        const representativeNeuron = currentLayerNodes[representativeIndex];
        const representativeInternal =
          representativeNeuron as unknown as NodeInternals;
        const kernel: number[] = [];
        for (let inChannel = 0; inChannel < convSpec.inChannels; inChannel++) {
          for (
            let kernelRow = 0;
            kernelRow < convSpec.kernelHeight;
            kernelRow++
          ) {
            for (
              let kernelCol = 0;
              kernelCol < convSpec.kernelWidth;
              kernelCol++
            ) {
              const inputFeatureIndex =
                inChannel * (convSpec.inHeight * convSpec.inWidth) +
                kernelRow * convSpec.inWidth +
                kernelCol;
              const sourceNode = previousLayerNodes[inputFeatureIndex];
              const connection = representativeInternal.connections.in.find(
                (candidate) => candidate.from === sourceNode,
              );
              kernel.push(connection ? connection.weight : 0);
            }
          }
        }
        representativeKernels.push(kernel);
      }

      const tolerance = 1e-9;
      for (
        let outChannel = 0;
        outChannel < convSpec.outChannels && isConsistent;
        outChannel++
      ) {
        for (
          let outRow = 0;
          outRow < convSpec.outHeight && isConsistent;
          outRow++
        ) {
          for (
            let outCol = 0;
            outCol < convSpec.outWidth && isConsistent;
            outCol++
          ) {
            const currentIndex =
              outChannel * (convSpec.outHeight * convSpec.outWidth) +
              outRow * convSpec.outWidth +
              outCol;
            const neuron = currentLayerNodes[currentIndex];
            if (!neuron) continue;
            const neuronInternal = neuron as unknown as NodeInternals;
            let kernelPointer = 0;
            for (
              let inChannel = 0;
              inChannel < convSpec.inChannels && isConsistent;
              inChannel++
            ) {
              const inputRowBase =
                outRow * convSpec.strideHeight - (convSpec.padTop || 0);
              const inputColBase =
                outCol * convSpec.strideWidth - (convSpec.padLeft || 0);
              for (
                let kernelRow = 0;
                kernelRow < convSpec.kernelHeight && isConsistent;
                kernelRow++
              ) {
                for (
                  let kernelCol = 0;
                  kernelCol < convSpec.kernelWidth && isConsistent;
                  kernelCol++
                ) {
                  const inputRow = inputRowBase + kernelRow;
                  const inputCol = inputColBase + kernelCol;
                  if (
                    inputRow < 0 ||
                    inputRow >= convSpec.inHeight ||
                    inputCol < 0 ||
                    inputCol >= convSpec.inWidth
                  ) {
                    kernelPointer++;
                    continue;
                  }
                  const inputFeatureIndex =
                    inChannel * (convSpec.inHeight * convSpec.inWidth) +
                    inputRow * convSpec.inWidth +
                    inputCol;
                  const sourceNode = previousLayerNodes[inputFeatureIndex];
                  const connection = neuronInternal.connections.in.find(
                    (candidate) => candidate.from === sourceNode,
                  );
                  const currentWeight = connection ? connection.weight : 0;
                  if (
                    Math.abs(
                      currentWeight -
                        representativeKernels[outChannel][kernelPointer],
                    ) > tolerance
                  ) {
                    isConsistent = false;
                  }
                  kernelPointer++;
                }
              }
            }
          }
        }
      }

      if (isConsistent) {
        verifiedLayers.push(layerIndex);
      } else {
        mismatchedLayers.push(layerIndex);
        console.warn(
          `Conv2D weight sharing mismatch detected in layer ${layerIndex}`,
        );
      }
    }

    if (verifiedLayers.length) {
      model.metadata_props.push({
        key: 'conv2d_sharing_verified',
        value: JSON.stringify(verifiedLayers),
      });
    }
    if (mismatchedLayers.length) {
      model.metadata_props.push({
        key: 'conv2d_sharing_mismatch',
        value: JSON.stringify(mismatchedLayers),
      });
    }
  }
}

/**
 * Try emitting heuristic fused LSTM node and metadata.
 */
function tryEmitFusedLstm(
  model: OnnxModel,
  layers: NeatapticNode[][],
  layerIndex: number,
  currentLayerNodes: NeatapticNode[],
  currentSize: number,
  previousOutputName: string,
): void {
  if (!(currentSize >= 10 && currentSize % 5 === 0)) return;

  const unit = currentSize / 5;
  const previousLayerNodes = layers[layerIndex - 1];
  const inputGate = currentLayerNodes.slice(0, unit);
  const forgetGate = currentLayerNodes.slice(unit, unit * 2);
  const cellGate = currentLayerNodes.slice(unit * 2, unit * 3);
  const outputGate = currentLayerNodes.slice(unit * 3, unit * 4);
  const gateOrder = [inputGate, forgetGate, cellGate, outputGate];
  const numberOfGates = gateOrder.length;
  const previousSize = previousLayerNodes.length;
  const inputWeights: number[] = [];
  const recurrentWeights: number[] = [];
  const biases: number[] = [];

  for (let gateIndex = 0; gateIndex < numberOfGates; gateIndex++) {
    const gate = gateOrder[gateIndex];
    for (let rowIndex = 0; rowIndex < unit; rowIndex++) {
      const neuron = gate[rowIndex];
      const neuronInternal = neuron as unknown as NodeInternals;
      for (let colIndex = 0; colIndex < previousSize; colIndex++) {
        const sourceNode = previousLayerNodes[colIndex];
        const connection = neuronInternal.connections.in.find(
          (candidate) => candidate.from === sourceNode,
        );
        inputWeights.push(connection ? connection.weight : 0);
      }
      for (let colIndex = 0; colIndex < unit; colIndex++) {
        if (gate === cellGate && colIndex === rowIndex) {
          const selfConnection = neuronInternal.connections.self[0];
          recurrentWeights.push(selfConnection ? selfConnection.weight : 0);
        } else {
          recurrentWeights.push(0);
        }
      }
      biases.push(neuronInternal.bias);
    }
  }

  model.graph.initializer.push({
    name: `LSTM_W${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit, previousSize],
    float_data: inputWeights,
  });
  model.graph.initializer.push({
    name: `LSTM_R${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit, unit],
    float_data: recurrentWeights,
  });
  model.graph.initializer.push({
    name: `LSTM_B${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit],
    float_data: biases,
  });

  model.graph.node.push({
    op_type: 'LSTM',
    input: [
      previousOutputName,
      `LSTM_W${layerIndex - 1}`,
      `LSTM_R${layerIndex - 1}`,
      `LSTM_B${layerIndex - 1}`,
    ],
    output: [`Layer_${layerIndex}_lstm_hidden`],
    name: `lstm_l${layerIndex}`,
    attributes: [
      { name: 'hidden_size', type: 'INT', i: unit },
      { name: 'layout', type: 'INT', i: 0 },
    ],
  });

  appendIndexMetadata(model, 'lstm_emitted_layers', layerIndex);
}

/**
 * Try emitting heuristic fused GRU node and metadata.
 */
function tryEmitFusedGru(
  model: OnnxModel,
  layers: NeatapticNode[][],
  layerIndex: number,
  currentLayerNodes: NeatapticNode[],
  currentSize: number,
): void {
  if (!(currentSize >= 8 && currentSize % 4 === 0)) return;

  const unit = currentSize / 4;
  const previousLayerNodes = layers[layerIndex - 1];
  const updateGate = currentLayerNodes.slice(0, unit);
  const resetGate = currentLayerNodes.slice(unit, unit * 2);
  const candidateGate = currentLayerNodes.slice(unit * 2, unit * 3);
  const gateOrder = [updateGate, resetGate, candidateGate];
  const numberOfGates = gateOrder.length;
  const previousSize = previousLayerNodes.length;
  const inputWeights: number[] = [];
  const recurrentWeights: number[] = [];
  const biases: number[] = [];

  for (let gateIndex = 0; gateIndex < numberOfGates; gateIndex++) {
    const gate = gateOrder[gateIndex];
    for (let rowIndex = 0; rowIndex < unit; rowIndex++) {
      const neuron = gate[rowIndex];
      const neuronInternal = neuron as unknown as NodeInternals;
      for (let colIndex = 0; colIndex < previousSize; colIndex++) {
        const sourceNode = previousLayerNodes[colIndex];
        const connection = neuronInternal.connections.in.find(
          (candidate) => candidate.from === sourceNode,
        );
        inputWeights.push(connection ? connection.weight : 0);
      }
      for (let colIndex = 0; colIndex < unit; colIndex++) {
        if (gate === candidateGate && colIndex === rowIndex) {
          const selfConnection = neuronInternal.connections.self[0];
          recurrentWeights.push(selfConnection ? selfConnection.weight : 0);
        } else {
          recurrentWeights.push(0);
        }
      }
      biases.push(neuronInternal.bias);
    }
  }

  model.graph.initializer.push({
    name: `GRU_W${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit, previousSize],
    float_data: inputWeights,
  });
  model.graph.initializer.push({
    name: `GRU_R${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit, unit],
    float_data: recurrentWeights,
  });
  model.graph.initializer.push({
    name: `GRU_B${layerIndex - 1}`,
    data_type: 1,
    dims: [numberOfGates * unit],
    float_data: biases,
  });

  const previousOutputName =
    layerIndex === 1 ? 'input' : `Layer_${layerIndex - 1}`;
  model.graph.node.push({
    op_type: 'GRU',
    input: [
      previousOutputName,
      `GRU_W${layerIndex - 1}`,
      `GRU_R${layerIndex - 1}`,
      `GRU_B${layerIndex - 1}`,
    ],
    output: [`Layer_${layerIndex}_gru_hidden`],
    name: `gru_l${layerIndex}`,
    attributes: [
      { name: 'hidden_size', type: 'INT', i: unit },
      { name: 'layout', type: 'INT', i: 0 },
    ],
  });

  appendIndexMetadata(model, 'gru_emitted_layers', layerIndex);
}

/**
 * Append a unique layer index to metadata array key.
 */
function appendIndexMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void {
  model.metadata_props = model.metadata_props || [];
  const metadataIndex = model.metadata_props.findIndex(
    (property) => property.key === key,
  );
  if (metadataIndex >= 0) {
    try {
      const parsed = JSON.parse(model.metadata_props[metadataIndex].value);
      if (Array.isArray(parsed) && !parsed.includes(layerIndex)) {
        parsed.push(layerIndex);
        model.metadata_props[metadataIndex].value = JSON.stringify(parsed);
      }
    } catch {
      model.metadata_props[metadataIndex].value = JSON.stringify([layerIndex]);
    }
    return;
  }
  model.metadata_props.push({ key, value: JSON.stringify([layerIndex]) });
}
