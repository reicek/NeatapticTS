import Connection from '../connection';
import type Network from '../network';
import type NeatapticNode from '../node';
import type { NodeInternals, OnnxModel } from './network.onnx.types.utils';

type LayerFactory = Record<string, (...args: unknown[]) => unknown>;

type FusedKind = 'LSTM' | 'GRU';

interface FusedLayerRuntime {
  nodes: NeatapticNode[];
  input: (groupLike: unknown) => void;
  output: { nodes: NeatapticNode[] } | null;
}

interface LayerNeighborhood {
  hiddenNodes: NeatapticNode[];
  oldLayerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  nextLayerNodes: NeatapticNode[];
  start: number;
  end: number;
}

interface FusedTensors {
  inputWeights: number[];
  recurrentWeights: number[];
  biases: number[];
  rows: number;
  previousLayerWidth: number;
}

/**
 * Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param layerFactory Dynamic layer module.
 * @param metadata ONNX metadata properties.
 * @returns Nothing.
 */
export function reconstructFusedRecurrentLayers(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: LayerFactory,
  metadata: { key: string; value: string }[],
): void {
  try {
    reconstructFusedKind(
      network,
      onnx,
      hiddenLayerSizes,
      layerFactory,
      metadata,
      'LSTM',
      4,
      ['input', 'forget', 'cell', 'output'],
      'cell',
    );
    reconstructFusedKind(
      network,
      onnx,
      hiddenLayerSizes,
      layerFactory,
      metadata,
      'GRU',
      3,
      ['update', 'reset', 'candidate'],
      'candidate',
    );
  } catch {
    /* swallow experimental import errors */
  }
}

/**
 * Reconstruct one fused recurrent family (LSTM or GRU).
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param layerFactory Dynamic layer module.
 * @param metadata Metadata properties.
 * @param kind Fused kind.
 * @param gateCount Number of gates.
 * @param gateOrder Gate names in tensor order.
 * @param recurrentGateName Gate carrying recurrent diagonal values.
 * @returns Nothing.
 */
function reconstructFusedKind(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: LayerFactory,
  metadata: { key: string; value: string }[],
  kind: FusedKind,
  gateCount: number,
  gateOrder: string[],
  recurrentGateName: string,
): void {
  const emittedLayerIndices = parseEmittedLayerIndices(metadata, kind);
  for (const exportLayerIndex of emittedLayerIndices) {
    reconstructOneFusedLayer(
      network,
      onnx,
      hiddenLayerSizes,
      layerFactory,
      kind,
      gateCount,
      gateOrder,
      recurrentGateName,
      exportLayerIndex,
    );
  }
}

/**
 * Parse emitted layer indices for a fused recurrent family.
 *
 * @param metadata Metadata properties.
 * @param kind Fused kind.
 * @returns Layer indices.
 */
function parseEmittedLayerIndices(
  metadata: { key: string; value: string }[],
  kind: FusedKind,
): number[] {
  const metadataKey =
    kind === 'LSTM' ? 'lstm_emitted_layers' : 'gru_emitted_layers';
  const emittedMeta = metadata.find((property) => property.key === metadataKey);
  if (!emittedMeta) return [];
  const parsed = JSON.parse(emittedMeta.value);
  return Array.isArray(parsed) ? parsed : [];
}

/**
 * Reconstruct one fused recurrent layer at a specific export index.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param layerFactory Dynamic layer module.
 * @param kind Fused kind.
 * @param gateCount Number of gates.
 * @param gateOrder Gate names in tensor order.
 * @param recurrentGateName Gate carrying recurrent diagonal values.
 * @param exportLayerIndex Export layer index.
 * @returns Nothing.
 */
function reconstructOneFusedLayer(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: LayerFactory,
  kind: FusedKind,
  gateCount: number,
  gateOrder: string[],
  recurrentGateName: string,
  exportLayerIndex: number,
): void {
  const hiddenLayerIndex = exportLayerIndex - 1;
  if (hiddenLayerIndex < 0 || hiddenLayerIndex >= hiddenLayerSizes.length)
    return;

  const tensors = resolveFusedTensors(onnx, kind, hiddenLayerIndex);
  if (!tensors) return;

  const unitSize = deriveUnitSize(tensors.rows, gateCount);
  if (!unitSize) return;

  const neighborhood = deriveLayerNeighborhood(
    network,
    hiddenLayerSizes,
    hiddenLayerIndex,
  );
  detachOldLayerConnections(
    network,
    neighborhood.oldLayerNodes,
    neighborhood.previousLayerNodes,
    neighborhood.nextLayerNodes,
  );

  const fusedLayer = createFusedLayer(layerFactory, kind, unitSize);
  replaceHiddenNodes(
    network,
    neighborhood.hiddenNodes,
    neighborhood.start,
    neighborhood.end,
    fusedLayer.nodes,
  );
  wireFusedLayer(
    fusedLayer,
    neighborhood.previousLayerNodes,
    neighborhood.nextLayerNodes,
  );

  applyGateWeights({
    fusedLayer,
    gateOrder,
    recurrentGateName,
    unitSize,
    previousLayerWidth: tensors.previousLayerWidth,
    biases: tensors.biases,
    inputWeights: tensors.inputWeights,
    recurrentWeights: tensors.recurrentWeights,
    previousLayerNodes: neighborhood.previousLayerNodes,
  });
}

/**
 * Resolve ONNX fused tensors for a hidden layer.
 *
 * @param onnx Source ONNX model.
 * @param kind Fused kind.
 * @param hiddenLayerIndex Hidden layer index.
 * @returns Fused tensor payload when available.
 */
function resolveFusedTensors(
  onnx: OnnxModel,
  kind: FusedKind,
  hiddenLayerIndex: number,
): FusedTensors | null {
  const inputWeightTensor = onnx.graph.initializer.find(
    (tensor) => tensor.name === `${kind}_W${hiddenLayerIndex}`,
  );
  const recurrentWeightTensor = onnx.graph.initializer.find(
    (tensor) => tensor.name === `${kind}_R${hiddenLayerIndex}`,
  );
  const biasTensor = onnx.graph.initializer.find(
    (tensor) => tensor.name === `${kind}_B${hiddenLayerIndex}`,
  );
  if (!inputWeightTensor || !recurrentWeightTensor || !biasTensor) return null;

  return {
    inputWeights: inputWeightTensor.float_data,
    recurrentWeights: recurrentWeightTensor.float_data,
    biases: biasTensor.float_data,
    rows: inputWeightTensor.dims[0],
    previousLayerWidth: inputWeightTensor.dims[1],
  };
}

/**
 * Derive hidden unit size from rows and gate count.
 *
 * @param rows Tensor row count.
 * @param gateCount Number of gates.
 * @returns Unit size when valid.
 */
function deriveUnitSize(rows: number, gateCount: number): number | null {
  if (rows % gateCount !== 0) return null;
  return rows / gateCount;
}

/**
 * Build neighborhood slices around the layer being replaced.
 *
 * @param network Target network.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param hiddenLayerIndex Hidden index.
 * @returns Neighboring layer slices.
 */
function deriveLayerNeighborhood(
  network: Network,
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
): LayerNeighborhood {
  const hiddenNodes = network.nodes.filter(
    (nodeItem) => nodeItem.type === 'hidden',
  );
  const start = hiddenLayerSizes
    .slice(0, hiddenLayerIndex)
    .reduce((sum, value) => sum + value, 0);
  const end = start + hiddenLayerSizes[hiddenLayerIndex];
  const oldLayerNodes = hiddenNodes.slice(start, end);

  const previousLayerNodes =
    hiddenLayerIndex === 0
      ? network.nodes.filter((nodeItem) => nodeItem.type === 'input')
      : hiddenNodes.slice(
          hiddenLayerSizes
            .slice(0, hiddenLayerIndex - 1)
            .reduce((sum, value) => sum + value, 0),
          hiddenLayerSizes
            .slice(0, hiddenLayerIndex)
            .reduce((sum, value) => sum + value, 0),
        );

  const nextLayerNodes =
    hiddenLayerIndex === hiddenLayerSizes.length - 1
      ? network.nodes.filter((nodeItem) => nodeItem.type === 'output')
      : hiddenNodes.slice(end, end + hiddenLayerSizes[hiddenLayerIndex + 1]);

  return {
    hiddenNodes,
    oldLayerNodes,
    previousLayerNodes,
    nextLayerNodes,
    start,
    end,
  };
}

/**
 * Remove all edges that involve replaced layer nodes.
 *
 * @param network Target network.
 * @param oldLayerNodes Nodes being replaced.
 * @param previousLayerNodes Previous layer nodes.
 * @param nextLayerNodes Next layer nodes.
 * @returns Nothing.
 */
function detachOldLayerConnections(
  network: Network,
  oldLayerNodes: NeatapticNode[],
  previousLayerNodes: NeatapticNode[],
  nextLayerNodes: NeatapticNode[],
): void {
  network.connections = network.connections.filter(
    (connection) =>
      !oldLayerNodes.includes(connection.from) &&
      !oldLayerNodes.includes(connection.to),
  );

  previousLayerNodes.forEach((previousNode) => {
    const previousNodeInternal = previousNode as unknown as NodeInternals;
    previousNodeInternal.connections.out =
      previousNodeInternal.connections.out.filter(
        (connection) => !oldLayerNodes.includes(connection.to),
      );
  });

  nextLayerNodes.forEach((nextNode) => {
    const nextNodeInternal = nextNode as unknown as NodeInternals;
    nextNodeInternal.connections.in = nextNodeInternal.connections.in.filter(
      (connection) => !oldLayerNodes.includes(connection.from),
    );
  });

  oldLayerNodes.forEach((oldLayerNode) => {
    const oldLayerNodeInternal = oldLayerNode as unknown as NodeInternals;
    oldLayerNodeInternal.connections.in = [];
    oldLayerNodeInternal.connections.out = [];
  });
}

/**
 * Create a fused recurrent runtime layer from the layer factory.
 *
 * @param layerFactory Runtime layer module.
 * @param kind Fused kind.
 * @param unitSize Unit count.
 * @returns Runtime layer.
 */
function createFusedLayer(
  layerFactory: LayerFactory,
  kind: FusedKind,
  unitSize: number,
): FusedLayerRuntime {
  const factoryMethod = kind === 'LSTM' ? layerFactory.lstm : layerFactory.gru;
  return factoryMethod(unitSize) as FusedLayerRuntime;
}

/**
 * Replace hidden node segment with fused layer nodes.
 *
 * @param network Target network.
 * @param hiddenNodes All hidden nodes.
 * @param start Segment start.
 * @param end Segment end.
 * @param replacementNodes New nodes.
 * @returns Nothing.
 */
function replaceHiddenNodes(
  network: Network,
  hiddenNodes: NeatapticNode[],
  start: number,
  end: number,
  replacementNodes: NeatapticNode[],
): void {
  const replacementCount = end - start;
  const updatedHiddenNodes = [...hiddenNodes];
  updatedHiddenNodes.splice(start, replacementCount, ...replacementNodes);

  const inputNodes = network.nodes.filter(
    (nodeItem) => nodeItem.type === 'input',
  );
  const outputNodes = network.nodes.filter(
    (nodeItem) => nodeItem.type === 'output',
  );
  network.nodes = [...inputNodes, ...updatedHiddenNodes, ...outputNodes];
}

/**
 * Wire fused layer between previous and next slices.
 *
 * @param fusedLayer Runtime fused layer.
 * @param previousLayerNodes Previous layer nodes.
 * @param nextLayerNodes Next layer nodes.
 * @returns Nothing.
 */
function wireFusedLayer(
  fusedLayer: FusedLayerRuntime,
  previousLayerNodes: NeatapticNode[],
  nextLayerNodes: NeatapticNode[],
): void {
  fusedLayer.input({ output: { nodes: previousLayerNodes } } as unknown);
  fusedLayer.output?.nodes.forEach((outputNode) => {
    const outputNodeInternal = outputNode as unknown as NodeInternals & {
      connect: (to: NeatapticNode) => Connection;
    };
    nextLayerNodes.forEach((nextNode) => outputNodeInternal.connect(nextNode));
  });
}

/**
 * Apply gate weights and biases to fused layer neurons.
 *
 * @param params Weight mapping parameters.
 * @returns Nothing.
 */
function applyGateWeights(params: {
  fusedLayer: FusedLayerRuntime;
  gateOrder: string[];
  recurrentGateName: string;
  unitSize: number;
  previousLayerWidth: number;
  biases: number[];
  inputWeights: number[];
  recurrentWeights: number[];
  previousLayerNodes: NeatapticNode[];
}): void {
  const {
    fusedLayer,
    gateOrder,
    recurrentGateName,
    unitSize,
    previousLayerWidth,
    biases,
    inputWeights,
    recurrentWeights,
    previousLayerNodes,
  } = params;

  const gateGroups = buildGateGroups(fusedLayer.nodes, gateOrder, unitSize);

  for (let gateIndex = 0; gateIndex < gateOrder.length; gateIndex++) {
    for (let rowIndex = 0; rowIndex < unitSize; rowIndex++) {
      const rowOffset = gateIndex * unitSize + rowIndex;
      const gateNeuron = gateGroups[gateOrder[gateIndex]][rowIndex];
      const gateNeuronInternal = gateNeuron as unknown as NodeInternals;
      gateNeuronInternal.bias = biases[rowOffset];
      applyIncomingWeights(
        gateNeuronInternal,
        rowOffset,
        previousLayerWidth,
        inputWeights,
        previousLayerNodes,
      );
      if (gateOrder[gateIndex] === recurrentGateName) {
        const selfConnection = gateNeuronInternal.connections.self[0];
        if (selfConnection) {
          selfConnection.weight =
            recurrentWeights[rowOffset * unitSize + rowIndex];
        }
      }
    }
  }
}

/**
 * Build per-gate neuron groups from fused node ordering.
 *
 * @param fusedNodes Fused layer node list.
 * @param gateOrder Gate ordering.
 * @param unitSize Units per gate.
 * @returns Gate group map.
 */
function buildGateGroups(
  fusedNodes: NeatapticNode[],
  gateOrder: string[],
  unitSize: number,
): Record<string, NeatapticNode[]> {
  const gateGroups: Record<string, NeatapticNode[]> = {};
  for (let gateIndex = 0; gateIndex < gateOrder.length; gateIndex++) {
    const start = gateIndex * unitSize;
    const end = start + unitSize;
    gateGroups[gateOrder[gateIndex]] = fusedNodes.slice(start, end);
  }
  return gateGroups;
}

/**
 * Apply dense incoming weights for one gate neuron.
 *
 * @param gateNeuronInternal Target gate neuron internal.
 * @param rowOffset Row offset in flattened weight matrix.
 * @param previousLayerWidth Previous layer width.
 * @param inputWeights Flattened input weight matrix.
 * @param previousLayerNodes Previous layer nodes.
 * @returns Nothing.
 */
function applyIncomingWeights(
  gateNeuronInternal: NodeInternals,
  rowOffset: number,
  previousLayerWidth: number,
  inputWeights: number[],
  previousLayerNodes: NeatapticNode[],
): void {
  for (let columnIndex = 0; columnIndex < previousLayerWidth; columnIndex++) {
    const sourceNode = previousLayerNodes[columnIndex];
    const incomingConnection = gateNeuronInternal.connections.in.find(
      (candidate) => candidate.from === sourceNode,
    );
    if (!incomingConnection) continue;
    incomingConnection.weight =
      inputWeights[rowOffset * previousLayerWidth + columnIndex];
  }
}
