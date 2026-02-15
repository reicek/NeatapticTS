import * as methods from '../../methods/methods';
import type Network from '../network';
import type { NodeInternals, OnnxModel } from './network.onnx.types.utils';

/**
 * Assign node activation functions from ONNX activation nodes.
 *
 * @param network Target network to mutate.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer size list.
 * @returns Nothing.
 */
export function assignActivationFunctions(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void {
  const hiddenNodes = network.nodes.filter((node) => node.type === 'hidden');
  let hiddenOffset = 0;
  const perLayer: Record<number, string[]> = {};

  onnx.graph.node.forEach((node) => {
    if (
      !['Tanh', 'Sigmoid', 'Logistic', 'Relu', 'Identity'].includes(
        node.op_type,
      )
    )
      return;
    const match = /^act_l(\d+)(?:_n(\d+))?$/i.exec(node.name || '');
    if (!match) return;
    const layerIndex = Number(match[1]);
    perLayer[layerIndex] = perLayer[layerIndex] || [];
    perLayer[layerIndex].push(node.op_type);
  });

  for (
    let hiddenLayerIndex = 0;
    hiddenLayerIndex < hiddenLayerSizes.length;
    hiddenLayerIndex++
  ) {
    const exportIndex = hiddenLayerIndex + 1;
    const operations = perLayer[exportIndex] || [];
    for (
      let neuronIndex = 0;
      neuronIndex < hiddenLayerSizes[hiddenLayerIndex];
      neuronIndex++
    ) {
      const operation = operations[neuronIndex] || operations[0];
      let activationFunction = methods.Activation.identity;
      switch (operation) {
        case 'Tanh':
          activationFunction = methods.Activation.tanh;
          break;
        case 'Sigmoid':
        case 'Logistic':
          activationFunction = methods.Activation.sigmoid;
          break;
        case 'Relu':
          activationFunction = methods.Activation.relu;
          break;
      }
      if (hiddenNodes[hiddenOffset + neuronIndex]) {
        const hiddenNodeInternal = hiddenNodes[
          hiddenOffset + neuronIndex
        ] as unknown as NodeInternals;
        hiddenNodeInternal.squash = activationFunction;
      }
    }
    hiddenOffset += hiddenLayerSizes[hiddenLayerIndex];
  }

  const outputExportIndex = hiddenLayerSizes.length + 1;
  const outputOperations = perLayer[outputExportIndex] || [];
  const outputOperation = outputOperations[0];
  let outputActivationFunction = methods.Activation.identity;
  switch (outputOperation) {
    case 'Tanh':
      outputActivationFunction = methods.Activation.tanh;
      break;
    case 'Sigmoid':
    case 'Logistic':
      outputActivationFunction = methods.Activation.sigmoid;
      break;
    case 'Relu':
      outputActivationFunction = methods.Activation.relu;
      break;
  }

  network.nodes
    .filter((node) => node.type === 'output')
    .forEach((node) => {
      const outputNodeInternal = node as unknown as NodeInternals;
      outputNodeInternal.squash = outputActivationFunction;
    });
}
