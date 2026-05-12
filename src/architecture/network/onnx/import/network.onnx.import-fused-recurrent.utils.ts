import Connection from '../../../connection';
import Group from '../../../group/group';
import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
} from '../schema/network.onnx.schema.types';
import type {
  OnnxFusedGateApplicationContext,
  OnnxFusedGateRowAssignmentContext,
  OnnxFusedLayerNeighborhood,
  OnnxFusedLayerReconstructionContext,
  OnnxFusedLayerRuntime,
  OnnxFusedRecurrentSpec,
  OnnxFusedTensorPayload,
  OnnxIncomingWeightAssignmentContext,
} from './network.onnx.import-fused-recurrent.types';
import type {
  NodeInternals,
  OnnxLayerFactory,
} from '../network.onnx.utils.types';

const FUSED_KIND_LSTM = 'LSTM';
const FUSED_KIND_GRU = 'GRU';
const LSTM_GATE_COUNT = 4;
const GRU_GATE_COUNT = 3;
const LSTM_GATE_ORDER = ['input', 'forget', 'cell', 'output'];
const GRU_GATE_ORDER = ['update', 'reset', 'candidate'];
const LSTM_RECURRENT_GATE_NAME = 'cell';
const GRU_RECURRENT_GATE_NAME = 'candidate';
const LSTM_METADATA_KEY = 'lstm_emitted_layers';
const GRU_METADATA_KEY = 'gru_emitted_layers';
const LAYER_FACTORY_LSTM_METHOD = 'lstm';
const LAYER_FACTORY_GRU_METHOD = 'gru';
const HIDDEN_NODE_TYPE = 'hidden';
const INPUT_NODE_TYPE = 'input';
const OUTPUT_NODE_TYPE = 'output';
const EXPORT_LAYER_TO_HIDDEN_LAYER_OFFSET = 1;
const TENSOR_SUFFIX_INPUT = 'W';
const TENSOR_SUFFIX_RECURRENT = 'R';
const TENSOR_SUFFIX_BIAS = 'B';
const INPUT_ROW_INDEX = 0;
const INPUT_COLUMN_INDEX = 1;
const EMPTY_EDGE_COLLECTION_LENGTH = 0;
const NATIVE_GRU_GROUP_COUNT = 6;
const GRU_UPDATE_GATE_GROUP_INDEX = 0;
const GRU_RESET_GATE_GROUP_INDEX = 2;
const GRU_MEMORY_CELL_GROUP_INDEX = 3;
const GRU_PREVIOUS_OUTPUT_GROUP_INDEX = 5;

type PreviousLayerSourceGroup = Group & {
  output: { nodes: NeatapticNode[] };
};

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
  layerFactory: OnnxLayerFactory,
  metadata: OnnxMetadataProperty[],
): void {
  const fusedRecurrentSpecs = createFusedRecurrentSpecs();

  // Step 1: Traverse each fused family and reconstruct emitted layers.
  try {
    reconstructAllFusedFamilies(fusedRecurrentSpecs);
  } catch {
    /* swallow experimental import errors */
  }
  /**
   * Build fused recurrent family specifications.
   *
   * @returns Family specifications.
   */
  function createFusedRecurrentSpecs(): OnnxFusedRecurrentSpec[] {
    return [
      {
        kind: FUSED_KIND_LSTM,
        gateCount: LSTM_GATE_COUNT,
        gateOrder: [...LSTM_GATE_ORDER],
        recurrentGateName: LSTM_RECURRENT_GATE_NAME,
        metadataKey: LSTM_METADATA_KEY,
      },
      {
        kind: FUSED_KIND_GRU,
        gateCount: GRU_GATE_COUNT,
        gateOrder: [...GRU_GATE_ORDER],
        recurrentGateName: GRU_RECURRENT_GATE_NAME,
        metadataKey: GRU_METADATA_KEY,
      },
    ];
  }

  /**
   * Reconstruct all fused recurrent families declared by metadata.
   *
   * @param specs Fused family specs.
   * @returns Nothing.
   */
  function reconstructAllFusedFamilies(specs: OnnxFusedRecurrentSpec[]): void {
    specs.forEach((specification) => reconstructOneFusedFamily(specification));
  }

  /**
   * Reconstruct one fused family across all emitted layer indices.
   *
   * @param spec Fused family spec.
   * @returns Nothing.
   */
  function reconstructOneFusedFamily(spec: OnnxFusedRecurrentSpec): void {
    const emittedLayerIndices = parseEmittedLayerIndices(spec);
    emittedLayerIndices.forEach((exportLayerIndex) =>
      reconstructOneFusedLayer({
        spec,
        exportLayerIndex,
        hiddenLayerIndex: toHiddenLayerIndex(exportLayerIndex),
      }),
    );
  }

  /**
   * Parse exported layer indices from metadata for one fused family.
   *
   * @param spec Fused family spec.
   * @returns Export-layer indices.
   */
  function parseEmittedLayerIndices(spec: OnnxFusedRecurrentSpec): number[] {
    const emittedProperty = metadata.find(
      (property) => property.key === spec.metadataKey,
    );
    if (!emittedProperty) return [];
    return parseMetadataJsonArray(emittedProperty.value);
  }

  /**
   * Parse metadata JSON payload as an array of indices.
   *
   * @param metadataValue Serialized metadata payload.
   * @returns Parsed index array.
   */
  function parseMetadataJsonArray(metadataValue: string): number[] {
    try {
      const parsed = JSON.parse(metadataValue);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item): item is number => typeof item === 'number');
    } catch {
      return [];
    }
  }

  /**
   * Reconstruct one fused layer from metadata and ONNX initializers.
   *
   * @param context Layer reconstruction context.
   * @returns Nothing.
   */
  function reconstructOneFusedLayer(
    context: OnnxFusedLayerReconstructionContext,
  ): void {
    // Step 1: Validate the hidden-layer target index.
    if (!isValidHiddenLayerIndex(context.hiddenLayerIndex)) return;

    // Step 2: Resolve ONNX initializer tensors for this fused layer.
    const fusedTensorPayload = resolveFusedTensors(context);
    if (!fusedTensorPayload) return;

    // Step 3: Derive recurrent unit size from tensor rows and gate count.
    const unitSize = deriveUnitSize(
      fusedTensorPayload.rows,
      context.spec.gateCount,
    );
    if (!unitSize) return;

    // Step 4: Rebuild the hidden-layer neighborhood around the fused slice.
    const layerNeighborhood = deriveLayerNeighborhood(context.hiddenLayerIndex);
    detachOldLayerConnections(layerNeighborhood);

    // Step 5: Construct, wire, and configure the replacement fused layer.
    const fusedLayerRuntime = createFusedLayerRuntime(context.spec, unitSize);
    replaceHiddenNodes(layerNeighborhood, fusedLayerRuntime.nodes);
    wireFusedLayer(
      fusedLayerRuntime,
      layerNeighborhood.previousLayerNodes,
      layerNeighborhood.nextLayerNodes,
    );

    // Step 6: Apply imported gate weights and biases.
    applyGateWeights({
      fusedLayer: fusedLayerRuntime,
      spec: context.spec,
      unitSize,
      previousLayerWidth: fusedTensorPayload.previousLayerWidth,
      biases: fusedTensorPayload.biases,
      inputWeights: fusedTensorPayload.inputWeights,
      recurrentWeights: fusedTensorPayload.recurrentWeights,
      previousLayerNodes: layerNeighborhood.previousLayerNodes,
    });
  }

  /**
   * Convert export-layer index to hidden-layer index.
   *
   * @param exportLayerIndex Export-layer index.
   * @returns Hidden-layer index.
   */
  function toHiddenLayerIndex(exportLayerIndex: number): number {
    return exportLayerIndex - EXPORT_LAYER_TO_HIDDEN_LAYER_OFFSET;
  }

  /**
   * Validate hidden-layer index boundaries.
   *
   * @param hiddenLayerIndex Hidden-layer index.
   * @returns True when index is valid.
   */
  function isValidHiddenLayerIndex(hiddenLayerIndex: number): boolean {
    if (hiddenLayerIndex < EMPTY_EDGE_COLLECTION_LENGTH) return false;
    return hiddenLayerIndex < hiddenLayerSizes.length;
  }

  /**
   * Resolve ONNX fused tensors for one hidden layer.
   *
   * @param context Layer reconstruction context.
   * @returns Tensor payload when fully available.
   */
  function resolveFusedTensors(
    context: OnnxFusedLayerReconstructionContext,
  ): OnnxFusedTensorPayload | null {
    const inputWeightTensor = findInitializerTensor(
      context.spec.kind,
      TENSOR_SUFFIX_INPUT,
      context.hiddenLayerIndex,
    );
    const recurrentWeightTensor = findInitializerTensor(
      context.spec.kind,
      TENSOR_SUFFIX_RECURRENT,
      context.hiddenLayerIndex,
    );
    const biasTensor = findInitializerTensor(
      context.spec.kind,
      TENSOR_SUFFIX_BIAS,
      context.hiddenLayerIndex,
    );

    if (!inputWeightTensor || !recurrentWeightTensor || !biasTensor) {
      return null;
    }

    return {
      inputWeights: inputWeightTensor.float_data,
      recurrentWeights: recurrentWeightTensor.float_data,
      biases: biasTensor.float_data,
      rows: inputWeightTensor.dims[INPUT_ROW_INDEX],
      previousLayerWidth: inputWeightTensor.dims[INPUT_COLUMN_INDEX],
    };
  }

  /**
   * Find one initializer tensor by fused family naming convention.
   *
   * @param kind Fused family kind.
   * @param suffix Tensor suffix.
   * @param hiddenLayerIndex Hidden-layer index.
   * @returns Initializer tensor when found.
   */
  function findInitializerTensor(
    kind: OnnxFusedRecurrentSpec['kind'],
    suffix: string,
    hiddenLayerIndex: number,
  ) {
    const tensorName = `${kind}_${suffix}${hiddenLayerIndex}`;
    return onnx.graph.initializer.find((tensor) => tensor.name === tensorName);
  }

  /**
   * Derive hidden unit size from recurrent row count and gate count.
   *
   * @param rows Recurrent rows.
   * @param gateCount Gate count.
   * @returns Unit size when compatible.
   */
  function deriveUnitSize(rows: number, gateCount: number): number | null {
    if (rows % gateCount !== EMPTY_EDGE_COLLECTION_LENGTH) return null;
    return rows / gateCount;
  }

  /**
   * Derive hidden-layer neighborhood slices for replacement traversal.
   *
   * @param hiddenLayerIndex Hidden-layer index.
   * @returns Layer neighborhood.
   */
  function deriveLayerNeighborhood(
    hiddenLayerIndex: number,
  ): OnnxFusedLayerNeighborhood {
    const hiddenNodes = filterNodesByType(HIDDEN_NODE_TYPE);
    const currentHiddenRange = createHiddenLayerRange(hiddenLayerIndex);
    const oldLayerNodes = hiddenNodes.slice(
      currentHiddenRange.start,
      currentHiddenRange.end,
    );
    const previousLayerNodes = derivePreviousLayerNodes(
      hiddenLayerIndex,
      hiddenNodes,
    );
    const nextLayerNodes = deriveNextLayerNodes(hiddenLayerIndex, hiddenNodes);

    return {
      hiddenNodes,
      oldLayerNodes,
      previousLayerNodes,
      nextLayerNodes,
      start: currentHiddenRange.start,
      end: currentHiddenRange.end,
    };
  }

  /**
   * Create hidden-range boundaries for one hidden-layer index.
   *
   * @param hiddenLayerIndex Hidden-layer index.
   * @returns Start and end boundaries.
   */
  function createHiddenLayerRange(hiddenLayerIndex: number): {
    start: number;
    end: number;
  } {
    const start = sumHiddenLayerSizes(
      EMPTY_EDGE_COLLECTION_LENGTH,
      hiddenLayerIndex,
    );
    const end = start + hiddenLayerSizes[hiddenLayerIndex];
    return { start, end };
  }

  /**
   * Sum hidden-layer sizes over a half-open range.
   *
   * @param startIndex Inclusive start.
   * @param endIndex Exclusive end.
   * @returns Summed size.
   */
  function sumHiddenLayerSizes(startIndex: number, endIndex: number): number {
    return hiddenLayerSizes
      .slice(startIndex, endIndex)
      .reduce(
        (sum, layerWidth) => sum + layerWidth,
        EMPTY_EDGE_COLLECTION_LENGTH,
      );
  }

  /**
   * Derive previous-layer nodes for a hidden layer.
   *
   * @param hiddenLayerIndex Hidden-layer index.
   * @param hiddenNodes All hidden nodes.
   * @returns Previous-layer nodes.
   */
  function derivePreviousLayerNodes(
    hiddenLayerIndex: number,
    hiddenNodes: NeatapticNode[],
  ): NeatapticNode[] {
    if (hiddenLayerIndex === EMPTY_EDGE_COLLECTION_LENGTH) {
      return filterNodesByType(INPUT_NODE_TYPE);
    }

    const previousRange = createHiddenLayerRange(
      hiddenLayerIndex - EXPORT_LAYER_TO_HIDDEN_LAYER_OFFSET,
    );
    return hiddenNodes.slice(previousRange.start, previousRange.end);
  }

  /**
   * Derive next-layer nodes for a hidden layer.
   *
   * @param hiddenLayerIndex Hidden-layer index.
   * @param hiddenNodes All hidden nodes.
   * @returns Next-layer nodes.
   */
  function deriveNextLayerNodes(
    hiddenLayerIndex: number,
    hiddenNodes: NeatapticNode[],
  ): NeatapticNode[] {
    const isLastHiddenLayer =
      hiddenLayerIndex ===
      hiddenLayerSizes.length - EXPORT_LAYER_TO_HIDDEN_LAYER_OFFSET;
    if (isLastHiddenLayer) return filterNodesByType(OUTPUT_NODE_TYPE);

    const nextRange = createHiddenLayerRange(
      hiddenLayerIndex + EXPORT_LAYER_TO_HIDDEN_LAYER_OFFSET,
    );
    return hiddenNodes.slice(nextRange.start, nextRange.end);
  }

  /**
   * Filter network nodes by semantic type.
   *
   * @param nodeType Node type name.
   * @returns Filtered node collection.
   */
  function filterNodesByType(nodeType: string): NeatapticNode[] {
    return network.nodes.filter((nodeItem) => nodeItem.type === nodeType);
  }

  /**
   * Detach all connections touching replaced hidden-layer nodes.
   *
   * @param neighborhood Layer neighborhood.
   * @returns Nothing.
   */
  function detachOldLayerConnections(
    neighborhood: OnnxFusedLayerNeighborhood,
  ): void {
    const { oldLayerNodes, previousLayerNodes, nextLayerNodes } = neighborhood;

    // Step 1: Remove global graph connections that touch the replaced layer.
    network.connections = network.connections.filter(
      (connection) =>
        !oldLayerNodes.includes(connection.from) &&
        !oldLayerNodes.includes(connection.to),
    );

    // Step 2: Remove per-node outgoing edges into replaced nodes.
    previousLayerNodes.forEach((previousNode) => {
      const previousNodeInternal = previousNode as unknown as NodeInternals;
      previousNodeInternal.connections.out =
        previousNodeInternal.connections.out.filter(
          (connection) => !oldLayerNodes.includes(connection.to),
        );
    });

    // Step 3: Remove per-node incoming edges from replaced nodes.
    nextLayerNodes.forEach((nextNode) => {
      const nextNodeInternal = nextNode as unknown as NodeInternals;
      nextNodeInternal.connections.in = nextNodeInternal.connections.in.filter(
        (connection) => !oldLayerNodes.includes(connection.from),
      );
    });

    // Step 4: Clear old nodes local edge arrays.
    oldLayerNodes.forEach((oldLayerNode) => {
      const oldLayerNodeInternal = oldLayerNode as unknown as NodeInternals;
      oldLayerNodeInternal.connections.in = [];
      oldLayerNodeInternal.connections.out = [];
    });
  }

  /**
   * Create one fused recurrent runtime layer instance.
   *
   * @param spec Fused family spec.
   * @param unitSize Unit count.
   * @returns Runtime fused layer.
   */
  function createFusedLayerRuntime(
    spec: OnnxFusedRecurrentSpec,
    unitSize: number,
  ): OnnxFusedLayerRuntime {
    const factoryMethodName =
      spec.kind === FUSED_KIND_LSTM
        ? LAYER_FACTORY_LSTM_METHOD
        : LAYER_FACTORY_GRU_METHOD;
    const factoryMethod = layerFactory[factoryMethodName];
    return factoryMethod(unitSize) as OnnxFusedLayerRuntime;
  }

  /**
   * Replace hidden node segment with reconstructed fused layer nodes.
   *
   * @param neighborhood Layer neighborhood.
   * @param replacementNodes Replacement nodes.
   * @returns Nothing.
   */
  function replaceHiddenNodes(
    neighborhood: OnnxFusedLayerNeighborhood,
    replacementNodes: NeatapticNode[],
  ): void {
    const replacementCount = neighborhood.end - neighborhood.start;
    const updatedHiddenNodes = createImmutableSplicedArray(
      neighborhood.hiddenNodes,
      neighborhood.start,
      replacementCount,
      ...replacementNodes,
    );
    const inputNodes = filterNodesByType(INPUT_NODE_TYPE);
    const outputNodes = filterNodesByType(OUTPUT_NODE_TYPE);
    network.nodes = [...inputNodes, ...updatedHiddenNodes, ...outputNodes];
  }

  /**
   * Create an immutable spliced copy, with a compatibility fallback when ES2023
   * `toSpliced` is typed as optional in ambient declarations.
   *
   * @param source Source array.
   * @param start Start index.
   * @param deleteCount Number of removed items.
   * @param insertItems Items to insert.
   * @returns New array containing the splice result.
   */
  function createImmutableSplicedArray<TItem>(
    source: TItem[],
    start: number,
    deleteCount: number,
    ...insertItems: TItem[]
  ): TItem[] {
    if (source.toSpliced) {
      return source.toSpliced(start, deleteCount, ...insertItems);
    }

    const prefix = source.slice(0, start);
    const suffix = source.slice(start + deleteCount);
    return [...prefix, ...insertItems, ...suffix];
  }

  /**
   * Wire fused layer between previous and next layer slices.
   *
   * @param fusedLayerRuntime Reconstructed fused layer runtime.
   * @param previousLayerNodes Previous-layer nodes.
   * @param nextLayerNodes Next-layer nodes.
   * @returns Nothing.
   */
  function wireFusedLayer(
    fusedLayerRuntime: OnnxFusedLayerRuntime,
    previousLayerNodes: NeatapticNode[],
    nextLayerNodes: NeatapticNode[],
  ): void {
    const previousLayerSourceGroup =
      createPreviousLayerSourceGroup(previousLayerNodes);

    fusedLayerRuntime.input(previousLayerSourceGroup as unknown);
    fusedLayerRuntime.output?.nodes.forEach((outputNode) => {
      const outputNodeInternal = outputNode as unknown as NodeInternals & {
        connect: (to: NeatapticNode) => Connection;
      };
      nextLayerNodes.forEach((nextNode) =>
        outputNodeInternal.connect(nextNode),
      );
    });
  }

  /**
   * Create a source group compatible with both runtime Layer input wiring and
   * the existing mock fused-layer tests.
   *
   * @param previousLayerNodes Previous-layer node slice.
   * @returns Group-like source wrapper.
   */
  function createPreviousLayerSourceGroup(
    previousLayerNodes: NeatapticNode[],
  ): PreviousLayerSourceGroup {
    const sourceGroup = new Group(
      EMPTY_EDGE_COLLECTION_LENGTH,
    ) as PreviousLayerSourceGroup;
    sourceGroup.nodes = previousLayerNodes as unknown as Group['nodes'];
    sourceGroup.output = { nodes: previousLayerNodes };
    return sourceGroup;
  }

  /**
   * Apply imported gate parameters to a reconstructed fused layer.
   *
   * @param context Gate application context.
   * @returns Nothing.
   */
  function applyGateWeights(context: OnnxFusedGateApplicationContext): void {
    const { gateGroups, recurrentSourceNodes } = resolveGateGroups(
      context.fusedLayer.nodes,
      context.spec,
      context.unitSize,
    );

    context.spec.gateOrder.forEach((gateName, gateIndex) => {
      const gateNeurons = gateGroups[gateName];
      gateNeurons.forEach((gateNeuron, rowIndex) =>
        assignGateRow({
          fusedKind: context.spec.kind,
          gateNeuronInternal: gateNeuron as unknown as NodeInternals,
          gateName,
          recurrentSourceNodes,
          recurrentGateName: context.spec.recurrentGateName,
          rowOffset: gateIndex * context.unitSize + rowIndex,
          rowIndex,
          unitSize: context.unitSize,
          previousLayerWidth: context.previousLayerWidth,
          biases: context.biases,
          inputWeights: context.inputWeights,
          recurrentWeights: context.recurrentWeights,
          previousLayerNodes: context.previousLayerNodes,
        }),
      );
    });
  }

  /**
   * Resolve gate groups and recurrent source nodes from one fused runtime layout.
   *
   * @param fusedNodes Fused node list.
   * @param spec Fused family specification.
   * @param unitSize Units per gate.
   * @returns Gate groups plus recurrent-source nodes.
   */
  function resolveGateGroups(
    fusedNodes: NeatapticNode[],
    spec: OnnxFusedRecurrentSpec,
    unitSize: number,
  ): {
    gateGroups: Record<string, NeatapticNode[]>;
    recurrentSourceNodes: NeatapticNode[];
  } {
    if (spec.kind === FUSED_KIND_GRU) {
      return resolveGruGateGroups(fusedNodes, unitSize);
    }

    return {
      gateGroups: buildContiguousGateGroups(
        fusedNodes,
        spec.gateOrder,
        unitSize,
      ),
      recurrentSourceNodes: [],
    };
  }

  /**
   * Build contiguous gate groups from one fused node list.
   *
   * @param fusedNodes Fused node list.
   * @param gateOrder Gate order.
   * @param unitSize Units per gate.
   * @returns Gate-name to neuron-list map.
   */
  function buildContiguousGateGroups(
    fusedNodes: NeatapticNode[],
    gateOrder: string[],
    unitSize: number,
  ): Record<string, NeatapticNode[]> {
    return gateOrder.reduce<Record<string, NeatapticNode[]>>(
      (groupMap, gateName, gateIndex) => {
        const gateStart = gateIndex * unitSize;
        const gateEnd = gateStart + unitSize;
        return {
          ...groupMap,
          [gateName]: fusedNodes.slice(gateStart, gateEnd),
        };
      },
      {},
    );
  }

  /**
   * Resolve GRU gate groups for either the native six-group layout or the
   * compact three-gate mock layout used by owner-local tests.
   *
   * @param fusedNodes Fused node list.
   * @param unitSize Units per gate.
   * @returns Gate-name to neuron-list map plus recurrent-source nodes.
   */
  function resolveGruGateGroups(
    fusedNodes: NeatapticNode[],
    unitSize: number,
  ): {
    gateGroups: Record<string, NeatapticNode[]>;
    recurrentSourceNodes: NeatapticNode[];
  } {
    const hasNativeGruLayout =
      fusedNodes.length >= unitSize * NATIVE_GRU_GROUP_COUNT;

    if (!hasNativeGruLayout) {
      return {
        gateGroups: buildContiguousGateGroups(
          fusedNodes,
          [...GRU_GATE_ORDER],
          unitSize,
        ),
        recurrentSourceNodes: [],
      };
    }

    return {
      gateGroups: {
        update: fusedNodes.slice(
          unitSize * GRU_UPDATE_GATE_GROUP_INDEX,
          unitSize * (GRU_UPDATE_GATE_GROUP_INDEX + 1),
        ),
        reset: fusedNodes.slice(
          unitSize * GRU_RESET_GATE_GROUP_INDEX,
          unitSize * (GRU_RESET_GATE_GROUP_INDEX + 1),
        ),
        candidate: fusedNodes.slice(
          unitSize * GRU_MEMORY_CELL_GROUP_INDEX,
          unitSize * (GRU_MEMORY_CELL_GROUP_INDEX + 1),
        ),
      },
      recurrentSourceNodes: fusedNodes.slice(
        unitSize * GRU_PREVIOUS_OUTPUT_GROUP_INDEX,
        unitSize * (GRU_PREVIOUS_OUTPUT_GROUP_INDEX + 1),
      ),
    };
  }

  /**
   * Assign one gate-neuron row parameters.
   *
   * @param context Gate-row assignment context.
   * @returns Nothing.
   */
  function assignGateRow(context: OnnxFusedGateRowAssignmentContext): void {
    context.gateNeuronInternal.bias = context.biases[context.rowOffset];

    assignIncomingWeights({
      gateNeuronInternal: context.gateNeuronInternal,
      rowOffset: context.rowOffset,
      previousLayerWidth: context.previousLayerWidth,
      inputWeights: context.inputWeights,
      previousLayerNodes: context.previousLayerNodes,
    });

    assignRecurrentWeights(context);
  }

  /**
   * Assign recurrent weights for one fused gate row.
   *
   * @param context Gate-row assignment context.
   * @returns Nothing.
   */
  function assignRecurrentWeights(
    context: OnnxFusedGateRowAssignmentContext,
  ): void {
    if (context.fusedKind !== FUSED_KIND_GRU) {
      if (context.gateName !== context.recurrentGateName) return;
      assignRecurrentDiagonalWeight(context);
      return;
    }

    if (context.recurrentSourceNodes.length === EMPTY_EDGE_COLLECTION_LENGTH) {
      if (context.gateName !== context.recurrentGateName) return;
      assignRecurrentDiagonalWeight(context);
      return;
    }

    Array.from({ length: context.unitSize }, (_, columnIndex) =>
      assignRecurrentIncomingWeightAtColumn(context, columnIndex),
    );
  }

  /**
   * Assign one recurrent diagonal self-weight.
   *
   * @param context Gate-row assignment context.
   * @returns Nothing.
   */
  function assignRecurrentDiagonalWeight(
    context: OnnxFusedGateRowAssignmentContext,
  ): void {
    const selfConnection = context.gateNeuronInternal.connections.self.at(
      EMPTY_EDGE_COLLECTION_LENGTH,
    );
    if (!selfConnection) return;
    selfConnection.weight =
      context.recurrentWeights[
        context.rowOffset * context.unitSize + context.rowIndex
      ];
  }

  /**
   * Assign one recurrent incoming weight from the native GRU previous-output
   * carrier into the current gate neuron.
   *
   * @param context Gate-row assignment context.
   * @param columnIndex Recurrent source column index.
   * @returns Nothing.
   */
  function assignRecurrentIncomingWeightAtColumn(
    context: OnnxFusedGateRowAssignmentContext,
    columnIndex: number,
  ): void {
    const sourceNode = context.recurrentSourceNodes[columnIndex];
    if (!sourceNode) return;

    const incomingConnection = context.gateNeuronInternal.connections.in.find(
      (candidateConnection) => candidateConnection.from === sourceNode,
    );
    if (!incomingConnection) return;

    incomingConnection.weight =
      context.recurrentWeights[
        context.rowOffset * context.unitSize + columnIndex
      ];
  }

  /**
   * Assign dense incoming weights for one gate neuron.
   *
   * @param context Incoming-weight assignment context.
   * @returns Nothing.
   */
  function assignIncomingWeights(
    context: OnnxIncomingWeightAssignmentContext,
  ): void {
    Array.from({ length: context.previousLayerWidth }, (_, columnIndex) =>
      assignIncomingWeightAtColumn(context, columnIndex),
    );
  }

  /**
   * Assign one incoming connection weight by source-column index.
   *
   * @param context Incoming-weight assignment context.
   * @param columnIndex Source column index.
   * @returns Nothing.
   */
  function assignIncomingWeightAtColumn(
    context: OnnxIncomingWeightAssignmentContext,
    columnIndex: number,
  ): void {
    const sourceNode = context.previousLayerNodes[columnIndex];
    if (!sourceNode) return;

    const incomingConnection = context.gateNeuronInternal.connections.in.find(
      (candidateConnection) => candidateConnection.from === sourceNode,
    );
    if (!incomingConnection) return;

    incomingConnection.weight =
      context.inputWeights[
        context.rowOffset * context.previousLayerWidth + columnIndex
      ];
  }
}
