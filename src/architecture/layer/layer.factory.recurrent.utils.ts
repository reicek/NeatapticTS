import type Connection from '../connection/connection';
import Group from '../group/group';
import * as methods from '../../methods/methods';
import Node from '../node';
import { isGroup as isGroupUtils } from './layer.guard.utils';
import {
  LayerMemoryInputBlockTypeError,
  LayerMemoryInputSizeMismatchError,
} from './layer.errors';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
  LayerLike,
} from './layer.utils.types';

const BIAS_ON = 1;
const BIAS_OFF = 0;
const UNIT_CONNECTION_WEIGHT = 1;
const EMPTY_GROUP_SIZE = 0;
const LAST_INDEX = -1;
const MEMORY_BLOCK_LINK_START_INDEX = 1;
const VARIANT_NODE_TYPE = 'variant';
const SELF_CONNECTION_WARNING_PREFIX =
  'LSTM Warning: No self-connection found for memory cell node ';
const MEMORY_INPUT_BLOCK_TYPE_ERROR =
  'Memory layer input block is not a Group.';
const MEMORY_LAYER_SIZE_ERROR_PREFIX = 'Previous layer size (';
const MEMORY_LAYER_SIZE_ERROR_MIDDLE = ') must be same as memory size (';
const MEMORY_LAYER_SIZE_ERROR_SUFFIX = ')';
const DEFAULT_GROUP_CONNECTION_METHOD = methods.groupConnection.ALL_TO_ALL;
const ONE_TO_ONE_CONNECTION_METHOD = methods.groupConnection.ONE_TO_ONE;
const OUTPUT_GATING_METHOD = methods.gating.OUTPUT;
const INPUT_GATING_METHOD = methods.gating.INPUT;

type LstmGroups = {
  inputGate: Group;
  forgetGate: Group;
  memoryCell: Group;
  outputGate: Group;
  outputBlock: Group;
};

type GruGroups = {
  updateGate: Group;
  inverseUpdateGate: Group;
  resetGate: Group;
  memoryCell: Group;
  output: Group;
  previousOutput: Group;
};

type MemoryBuildInput = {
  memorySteps: number;
  size: number;
};

type ConnectorInput = {
  from: LayerLike | Group;
  method?: unknown;
  weight?: number;
};

/**
 * Builds an LSTM layer using the provided factory context.
 *
 * Educational overview:
 * - **Gates** control information flow (input / forget / output).
 * - The **memory cell** stores recurrent state via a self-connection.
 * - The **output block** is what this layer exposes as `layer.output`.
 *
 * This builder wires the classic LSTM topology using `Group` blocks and gating.
 * It returns a layer object that is compatible with the rest of the layer
 * utilities (`layer.input(...)`, `layer.activate(...)`, `layer.output`, etc.).
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of units in each LSTM gate and cell.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const lstm = buildLstmLayer(factoryContext, 8);
 *
 * // Wire a previous layer (or Group) into the LSTM.
 * lstm.input(previousLayerLike);
 * ```
 */
export function buildLstmLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = context.createLayer();
  const lstmGroups = createLstmGroups(size);

  describeLstmBoundary(layer, lstmGroups, size);
  configureLstmBiases(lstmGroups);
  const lstmOutputConnections = connectLstmCoreTopology(lstmGroups);
  applyLstmOutputGating(lstmGroups, lstmOutputConnections);
  synchronizeLstmForgetSelfConnections(lstmGroups);

  layer.nodes = collectLstmNodes(lstmGroups);
  layer.output = lstmGroups.outputBlock;
  layer.input = createLstmInputConnector(context, lstmGroups);

  return layer;

  /**
   * Creates the grouped blocks used by the LSTM topology.
    * @param groupSize Number of units to allocate per group.
   * @returns The grouped LSTM components.
   */
  function createLstmGroups(groupSize: number): LstmGroups {
    return {
      inputGate: new Group(groupSize),
      forgetGate: new Group(groupSize),
      memoryCell: new Group(groupSize),
      outputGate: new Group(groupSize),
      outputBlock: new Group(groupSize),
    };
  }

  /**
   * Attaches lightweight descriptor metadata to the LSTM boundary.
   * @param targetLayer Layer receiving the public descriptor.
   * @param groups Internal grouped blocks used by the topology.
   * @param units Number of units in each grouped block.
   * @returns No return value.
   */
  function describeLstmBoundary(
    targetLayer: TLayer,
    groups: LstmGroups,
    units: number,
  ): void {
    targetLayer.describe?.({
      intent: 'recurrent',
      metadata: { family: 'lstm', units },
    });
    groups.inputGate.describe({
      label: 'inputGate',
      intent: 'gate',
      metadata: { family: 'lstm', units },
    });
    groups.forgetGate.describe({
      label: 'forgetGate',
      intent: 'gate',
      metadata: { family: 'lstm', units },
    });
    groups.memoryCell.describe({
      label: 'memoryCell',
      intent: 'state',
      metadata: { family: 'lstm', units },
    });
    groups.outputGate.describe({
      label: 'outputGate',
      intent: 'gate',
      metadata: { family: 'lstm', units },
    });
    groups.outputBlock.describe({
      label: 'outputBlock',
      intent: 'output',
      metadata: { family: 'lstm', units },
    });
  }

  /**
   * Applies baseline bias values for all LSTM groups.
    * @param groups LSTM groups to configure.
   * @returns No return value.
   */
  function configureLstmBiases(groups: LstmGroups): void {
    groups.inputGate.set({ bias: BIAS_ON });
    groups.forgetGate.set({ bias: BIAS_ON });
    groups.outputGate.set({ bias: BIAS_ON });
    groups.memoryCell.set({ bias: BIAS_OFF });
    groups.outputBlock.set({ bias: BIAS_OFF });
  }

  /**
   * Connects the core recurrent and gate topology for the LSTM cell.
    * @param groups LSTM groups that will be wired together.
   * @returns Connections from memory cell to output block.
   */
  function connectLstmCoreTopology(groups: LstmGroups): Connection[] {
    groups.memoryCell.connect(
      groups.inputGate,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    groups.memoryCell.connect(
      groups.forgetGate,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    groups.memoryCell.connect(
      groups.outputGate,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    groups.memoryCell.connect(groups.memoryCell, ONE_TO_ONE_CONNECTION_METHOD);
    return groups.memoryCell.connect(
      groups.outputBlock,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
  }

  /**
   * Applies output gating for the LSTM output projection.
   * @param groups LSTM groups that contain the output gate.
   * @param outputConnections Connections to gate.
   * @returns No return value.
   */
  function applyLstmOutputGating(
    groups: LstmGroups,
    outputConnections: Connection[],
  ): void {
    groups.outputGate.gate(outputConnections, OUTPUT_GATING_METHOD);
  }

  /**
   * Synchronizes each memory self-connection with the paired forget gate node.
    * @param groups LSTM groups containing memory and forget gate nodes.
   * @returns No return value.
   */
  function synchronizeLstmForgetSelfConnections(groups: LstmGroups): void {
    groups.memoryCell.nodes.forEach((memoryCellNode, nodeIndex) => {
      const selfConnection = findMemoryCellSelfConnection(memoryCellNode);
      if (!selfConnection) {
        logMissingLstmSelfConnection(nodeIndex);
        return;
      }

      const forgetGateNode = groups.forgetGate.nodes[nodeIndex];
      assignForgetGater(selfConnection, forgetGateNode);
      appendGatedConnectionIfMissing(forgetGateNode, selfConnection);
    });
  }

  /**
   * Finds the self-connection for a memory cell node.
   * @param memoryCellNode Node to inspect.
   * @returns Matching self-connection when present.
   */
  function findMemoryCellSelfConnection(
    memoryCellNode: Node,
  ): Connection | undefined {
    return memoryCellNode.connections.self.find(
      (selfConnection) =>
        selfConnection.to === memoryCellNode &&
        selfConnection.from === memoryCellNode,
    );
  }

  /**
   * Logs the missing self-connection warning for an LSTM memory node.
    * @param nodeIndex Memory node index.
   * @returns No return value.
   */
  function logMissingLstmSelfConnection(nodeIndex: number): void {
    console.warn(`${SELF_CONNECTION_WARNING_PREFIX}${nodeIndex}`);
  }

  /**
   * Assigns a forget gate node as the gater of a connection.
   * @param connection Connection to update.
   * @param forgetGateNode Gate node to assign.
   * @returns No return value.
   */
  function assignForgetGater(
    connection: Connection,
    forgetGateNode: Node,
  ): void {
    connection.gater = forgetGateNode;
  }

  /**
   * Adds a gated connection to the forget gate node when absent.
   * @param forgetGateNode Gate node that tracks gated connections.
   * @param connection Connection that may be appended.
   * @returns No return value.
   */
  function appendGatedConnectionIfMissing(
    forgetGateNode: Node,
    connection: Connection,
  ): void {
    if (forgetGateNode.connections.gated.includes(connection)) {
      return;
    }

    forgetGateNode.connections.gated.push(connection);
  }

  /**
   * Collects all LSTM nodes in canonical layer order.
    * @param groups LSTM groups to flatten.
   * @returns Flattened node list.
   */
  function collectLstmNodes(groups: LstmGroups): Node[] {
    return [
      ...groups.inputGate.nodes,
      ...groups.forgetGate.nodes,
      ...groups.memoryCell.nodes,
      ...groups.outputGate.nodes,
      ...groups.outputBlock.nodes,
    ];
  }

  /**
   * Creates the layer input connector for the LSTM topology.
   * @param factoryContext Factory context used to resolve source groups.
   * @param groups LSTM groups that receive source connections.
   * @returns Input connector callback.
   */
  function createLstmInputConnector(
    factoryContext: LayerFactoryContext<TLayer>,
    groups: LstmGroups,
  ): (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ) => Connection[] {
    return (
      from: LayerLike | Group,
      method?: unknown,
      weight?: number,
    ): Connection[] => {
      const connectorInput: ConnectorInput = { from, method, weight };
      const sourceGroup = resolveSourceGroup(
        factoryContext,
        connectorInput.from,
      );
      const resolvedMethod = resolveConnectionMethod(connectorInput.method);

      const memoryInputConnections = sourceGroup.connect(
        groups.memoryCell,
        resolvedMethod,
        connectorInput.weight,
      );
      const inputGateConnections = sourceGroup.connect(
        groups.inputGate,
        resolvedMethod,
        connectorInput.weight,
      );
      const outputGateConnections = sourceGroup.connect(
        groups.outputGate,
        resolvedMethod,
        connectorInput.weight,
      );
      const forgetGateConnections = sourceGroup.connect(
        groups.forgetGate,
        resolvedMethod,
        connectorInput.weight,
      );

      groups.inputGate.gate(memoryInputConnections, INPUT_GATING_METHOD);

      return flattenConnections([
        memoryInputConnections,
        inputGateConnections,
        outputGateConnections,
        forgetGateConnections,
      ]);
    };
  }
}

/**
 * Builds a GRU layer using the provided factory context.
 *
 * Educational overview:
 * - GRU is a gated recurrent unit with fewer gates than LSTM.
 * - This implementation wires update/reset gates and a memory cell, then
 * exposes a standard `layer.output` group.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of units in each GRU gate and cell.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const gru = buildGruLayer(factoryContext, 8);
 * gru.input(previousLayerLike);
 * ```
 */
export function buildGruLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = context.createLayer();
  const gruGroups = createGruGroups(size);

  configureGruNodes(gruGroups);
  describeGruBoundary(layer, gruGroups, size);
  connectGruCoreTopology(gruGroups);

  layer.nodes = collectGruNodes(gruGroups);
  layer.output = gruGroups.output;
  layer.input = createGruInputConnector(context, gruGroups);

  return layer;

  /**
   * Creates grouped blocks used by the GRU topology.
    * @param groupSize Number of units to allocate per group.
   * @returns The grouped GRU components.
   */
  function createGruGroups(groupSize: number): GruGroups {
    return {
      updateGate: new Group(groupSize),
      inverseUpdateGate: new Group(groupSize),
      resetGate: new Group(groupSize),
      memoryCell: new Group(groupSize),
      output: new Group(groupSize),
      previousOutput: new Group(groupSize),
    };
  }

  /**
   * Attaches lightweight descriptor metadata to the GRU boundary.
   * @param targetLayer Layer receiving the public descriptor.
   * @param groups Internal grouped blocks used by the topology.
   * @param units Number of units in each grouped block.
   * @returns No return value.
   */
  function describeGruBoundary(
    targetLayer: TLayer,
    groups: GruGroups,
    units: number,
  ): void {
    targetLayer.describe?.({
      intent: 'recurrent',
      metadata: { family: 'gru', units },
    });
    groups.updateGate.describe({
      label: 'updateGate',
      intent: 'gate',
      metadata: { family: 'gru', units },
    });
    groups.inverseUpdateGate.describe({
      label: 'inverseUpdateGate',
      intent: 'gate',
      metadata: { family: 'gru', units },
    });
    groups.resetGate.describe({
      label: 'resetGate',
      intent: 'gate',
      metadata: { family: 'gru', units },
    });
    groups.memoryCell.describe({
      label: 'memoryCell',
      intent: 'state',
      metadata: { family: 'gru', units },
    });
    groups.output.describe({
      label: 'outputBlock',
      intent: 'output',
      metadata: { family: 'gru', units },
    });
    groups.previousOutput.describe({
      label: 'previousOutput',
      intent: 'state',
      metadata: { family: 'gru', units },
    });
  }

  /**
   * Applies baseline node settings for GRU groups.
    * @param groups GRU groups to configure.
   * @returns No return value.
   */
  function configureGruNodes(groups: GruGroups): void {
    groups.previousOutput.set({
      bias: BIAS_OFF,
      squash: methods.Activation.identity,
      type: 'hidden',
    });
    groups.memoryCell.set({ squash: methods.Activation.tanh });
    groups.inverseUpdateGate.set({
      bias: BIAS_OFF,
      squash: methods.Activation.inverse,
      type: 'hidden',
    });
    groups.updateGate.set({ bias: BIAS_ON });
    groups.resetGate.set({ bias: BIAS_OFF });
  }

  /**
   * Wires the full GRU topology using focused connection helpers.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruCoreTopology(groups: GruGroups): void {
    connectGruGatePriming(groups);
    connectGruInverseGate(groups);
    connectGruResetPath(groups);
    connectGruUpdatePath(groups);
    connectGruStateCarry(groups);
  }

  /**
   * Connects previous output into update and reset gate inputs.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruGatePriming(groups: GruGroups): void {
    groups.previousOutput.connect(
      groups.updateGate,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    groups.previousOutput.connect(
      groups.resetGate,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
  }

  /**
   * Connects update gate to inverse update gate.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruInverseGate(groups: GruGroups): void {
    groups.updateGate.connect(
      groups.inverseUpdateGate,
      ONE_TO_ONE_CONNECTION_METHOD,
      UNIT_CONNECTION_WEIGHT,
    );
  }

  /**
   * Creates and gates reset-path connections.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruResetPath(groups: GruGroups): void {
    const resetConnections = groups.previousOutput.connect(
      groups.memoryCell,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    groups.resetGate.gate(resetConnections, OUTPUT_GATING_METHOD);
  }

  /**
   * Creates and gates update-path blend connections.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruUpdatePath(groups: GruGroups): void {
    const previousStateConnections = groups.previousOutput.connect(
      groups.output,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );
    const memoryStateConnections = groups.memoryCell.connect(
      groups.output,
      DEFAULT_GROUP_CONNECTION_METHOD,
    );

    groups.updateGate.gate(previousStateConnections, OUTPUT_GATING_METHOD);
    groups.inverseUpdateGate.gate(memoryStateConnections, OUTPUT_GATING_METHOD);
  }

  /**
   * Connects new output back into previous state storage.
    * @param groups GRU groups to wire.
   * @returns No return value.
   */
  function connectGruStateCarry(groups: GruGroups): void {
    groups.output.connect(
      groups.previousOutput,
      ONE_TO_ONE_CONNECTION_METHOD,
      UNIT_CONNECTION_WEIGHT,
    );
  }

  /**
   * Collects all GRU nodes in canonical layer order.
    * @param groups GRU groups to flatten.
   * @returns Flattened node list.
   */
  function collectGruNodes(groups: GruGroups): Node[] {
    return [
      ...groups.updateGate.nodes,
      ...groups.inverseUpdateGate.nodes,
      ...groups.resetGate.nodes,
      ...groups.memoryCell.nodes,
      ...groups.output.nodes,
      ...groups.previousOutput.nodes,
    ];
  }

  /**
   * Creates the layer input connector for the GRU topology.
   * @param factoryContext Factory context used to resolve source groups.
   * @param groups GRU groups that receive source connections.
   * @returns Input connector callback.
   */
  function createGruInputConnector(
    factoryContext: LayerFactoryContext<TLayer>,
    groups: GruGroups,
  ): (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ) => Connection[] {
    return (
      from: LayerLike | Group,
      method?: unknown,
      weight?: number,
    ): Connection[] => {
      const connectorInput: ConnectorInput = { from, method, weight };
      const sourceGroup = resolveSourceGroup(
        factoryContext,
        connectorInput.from,
      );
      const resolvedMethod = resolveConnectionMethod(connectorInput.method);

      const updateGateConnections = sourceGroup.connect(
        groups.updateGate,
        resolvedMethod,
        connectorInput.weight,
      );
      const resetGateConnections = sourceGroup.connect(
        groups.resetGate,
        resolvedMethod,
        connectorInput.weight,
      );
      const memoryCellConnections = sourceGroup.connect(
        groups.memoryCell,
        resolvedMethod,
        connectorInput.weight,
      );

      return flattenConnections([
        updateGateConnections,
        resetGateConnections,
        memoryCellConnections,
      ]);
    };
  }
}

/**
 * Builds a Memory layer using the provided factory context.
 *
 * A memory layer is a simple way to provide a fixed window of past values.
 * Internally it creates `memory` blocks, links them one-to-one, and exposes a
 * flattened output group containing all block nodes.
 *
 * Important: the input connector for a memory layer enforces **one-to-one**
 * wiring with unit weights to preserve the intended delay-line behavior.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of nodes in each memory block.
 * @param memory Number of time steps to remember.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * // A memory layer with 4 nodes per timestep, remembering 3 steps.
 * const memoryLayer = buildMemoryLayer(factoryContext, 4, 3);
 * memoryLayer.input(previousLayerLike);
 * ```
 */
export function buildMemoryLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
  memory: number,
): TLayer {
  const layer = context.createLayer();
  const memoryBuildInput: MemoryBuildInput = { memorySteps: memory, size };
  const orderedMemoryBlocks = createOrderedMemoryBlocks(memoryBuildInput);
  describeMemoryBoundary(layer, orderedMemoryBlocks, size, memory);

  layer.nodes = castMemoryBlocksToLayerNodes(orderedMemoryBlocks);
  layer.output = createMemoryOutputGroup(layer.nodes);
  layer.output.describe({
    label: 'memoryOutput',
    intent: 'memory',
    metadata: { family: 'memory', memorySteps: memory, size },
  });
  layer.input = createMemoryInputConnector(context, layer.nodes);

  return layer;

  /**
   * Creates memory blocks, links them, and returns traversal-ready ordering.
    * @param buildInput Memory layer shape parameters.
   * @returns Ordered memory blocks.
   */
  function createOrderedMemoryBlocks(buildInput: MemoryBuildInput): Group[] {
    const memoryBlocks = createMemoryBlocks(buildInput);
    connectMemoryBlocksInSequence(memoryBlocks);
    return memoryBlocks.toReversed?.() ?? [...memoryBlocks].reverse();
  }

  /**
   * Attaches lightweight descriptor metadata to the memory-layer boundary.
   * @param targetLayer Layer receiving the public descriptor.
   * @param memoryBlocks Ordered memory blocks exposed by the layer.
   * @param blockSize Number of nodes in each block.
   * @param memorySteps Number of remembered time steps.
   * @returns No return value.
   */
  function describeMemoryBoundary(
    targetLayer: TLayer,
    memoryBlocks: Group[],
    blockSize: number,
    memorySteps: number,
  ): void {
    targetLayer.describe?.({
      intent: 'memory',
      metadata: { family: 'memory', memorySteps, size: blockSize },
    });

    memoryBlocks.forEach((memoryBlock, blockIndex) => {
      memoryBlock.describe({
        label: `memoryBlock${blockIndex}`,
        intent: 'state',
        metadata: {
          family: 'memory',
          memoryBlockIndex: blockIndex,
          memorySteps,
          size: blockSize,
        },
      });
    });
  }

  /**
   * Allocates all memory blocks for the memory layer.
    * @param buildInput Memory layer shape parameters.
   * @returns Created memory blocks.
   */
  function createMemoryBlocks(buildInput: MemoryBuildInput): Group[] {
    return Array.from({ length: buildInput.memorySteps }, () =>
      createMemoryBlock(buildInput.size),
    );
  }

  /**
   * Creates one configured memory block.
    * @param blockSize Number of nodes in the block.
   * @returns Configured memory block.
   */
  function createMemoryBlock(blockSize: number): Group {
    const block = new Group(blockSize);
    block.set({
      squash: methods.Activation.identity,
      bias: BIAS_OFF,
      type: VARIANT_NODE_TYPE,
    });
    return block;
  }

  /**
   * Connects memory blocks in forward sequence.
    * @param memoryBlocks Blocks to connect in order.
   * @returns No return value.
   */
  function connectMemoryBlocksInSequence(memoryBlocks: Group[]): void {
    memoryBlocks
      .slice(MEMORY_BLOCK_LINK_START_INDEX)
      .forEach((currentBlock, blockIndex) => {
        const previousBlock = memoryBlocks[blockIndex];
        connectMemoryBlockPair(previousBlock, currentBlock);
      });
  }

  /**
   * Connects one memory block pair using one-to-one links.
   * @param previousBlock Source block in sequence.
   * @param currentBlock Destination block in sequence.
   * @returns No return value.
   */
  function connectMemoryBlockPair(
    previousBlock: Group,
    currentBlock: Group,
  ): void {
    previousBlock.connect(
      currentBlock,
      ONE_TO_ONE_CONNECTION_METHOD,
      UNIT_CONNECTION_WEIGHT,
    );
  }

  /**
   * Casts memory blocks into the layer node container type.
    * @param memoryBlocks Memory blocks to cast.
   * @returns Layer node list.
   */
  function castMemoryBlocksToLayerNodes(memoryBlocks: Group[]): Node[] {
    return memoryBlocks as unknown as Node[];
  }

  /**
   * Creates the memory layer output group from all internal blocks.
    * @param layerNodes Layer nodes that may contain memory groups.
   * @returns Aggregated output group.
   */
  function createMemoryOutputGroup(layerNodes: Node[]): Group {
    const outputGroup = new Group(EMPTY_GROUP_SIZE);

    layerNodes.forEach((layerNode) => {
      appendLayerNodeToOutputGroup(outputGroup, layerNode);
    });

    return outputGroup;
  }

  /**
   * Appends one layer node into the output group when it is group-like.
   * @param outputGroup Output group receiving nodes.
   * @param layerNode Candidate layer node.
   * @returns No return value.
   */
  function appendLayerNodeToOutputGroup(
    outputGroup: Group,
    layerNode: Node,
  ): void {
    outputGroup.nodes = outputGroup.nodes.concat(
      (layerNode as unknown as Group).nodes,
    );
  }

  /**
   * Creates the memory layer input connector.
   * @param factoryContext Factory context used to resolve source groups.
   * @param layerNodes Layer nodes used to resolve the input block.
   * @returns Input connector callback.
   */
  function createMemoryInputConnector(
    factoryContext: LayerFactoryContext<TLayer>,
    layerNodes: Node[],
  ): (
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ) => Connection[] {
    return (from: LayerLike | Group): Connection[] => {
      const sourceGroup = resolveSourceGroup(factoryContext, from);
      const inputBlock = resolveMemoryInputBlock(layerNodes);
      assertMemoryInputSize(sourceGroup, inputBlock);

      return sourceGroup.connect(
        inputBlock,
        ONE_TO_ONE_CONNECTION_METHOD,
        UNIT_CONNECTION_WEIGHT,
      );
    };
  }

  /**
   * Resolves the terminal memory block used as layer input target.
    * @param layerNodes Layer nodes to inspect.
   * @returns Resolved input block group.
   * @throws {Error} When the terminal node is not group-like.
   */
  function resolveMemoryInputBlock(layerNodes: Node[]): Group {
    const inputCandidate = layerNodes.at(LAST_INDEX);
    if (isGroupUtils(inputCandidate)) {
      return inputCandidate;
    }

    throw new LayerMemoryInputBlockTypeError(MEMORY_INPUT_BLOCK_TYPE_ERROR);
  }

  /**
   * Validates that source and memory input block sizes match.
    * @param sourceGroup Source group feeding the memory layer.
    * @param inputBlock Memory input target block.
   * @returns No return value.
   * @throws {Error} When source and target sizes differ.
   */
  function assertMemoryInputSize(sourceGroup: Group, inputBlock: Group): void {
    if (sourceGroup.nodes.length === inputBlock.nodes.length) {
      return;
    }

    throw new LayerMemoryInputSizeMismatchError(
      `${MEMORY_LAYER_SIZE_ERROR_PREFIX}${sourceGroup.nodes.length}${MEMORY_LAYER_SIZE_ERROR_MIDDLE}${inputBlock.nodes.length}${MEMORY_LAYER_SIZE_ERROR_SUFFIX}`,
    );
  }
}

/**
 * Resolves a source group from a layer-like or group input.
 *
 * Many wiring helpers accept either a `Group` or a "layer-like" object.
 * When a layer is provided, we treat `layer.output` as the actual source group.
 *
 * @param factoryContext Factory context providing layer guards.
 * @param from Source input candidate.
 * @returns Source group used for connections.
 *
 * Example:
 *
 * ```ts
 * const sourceGroup = resolveSourceGroup(factoryContext, previousLayerLike);
 * ```
 */
function resolveSourceGroup<TLayer extends LayerFactoryLayer>(
  factoryContext: LayerFactoryContext<TLayer>,
  from: LayerLike | Group,
): Group {
  return factoryContext.isLayer(from) ? from.output! : from;
}

/**
 * Resolves an optional connection method to a concrete method.
 *
 * When users don't specify a method, we default to a dense-style
 * `ALL_TO_ALL` group connection.
 *
 * @param method Optional user-supplied connection method.
 * @returns Connection method to apply.
 *
 * Example:
 *
 * ```ts
 * const method = resolveConnectionMethod(undefined);
 * ```
 */
function resolveConnectionMethod(method?: unknown): unknown {
  return method ?? DEFAULT_GROUP_CONNECTION_METHOD;
}

/**
 * Flattens grouped connection arrays into a single list.
 *
 * This keeps builder code declarative: build connections per gate/block,
 * then flatten once at the end.
 *
 * @param connectionLists Connection groups to flatten.
 * @returns Flattened connection list.
 *
 * Example:
 *
 * ```ts
 * const connections = flattenConnections([gateConnections, cellConnections]);
 * ```
 */
function flattenConnections(connectionLists: Connection[][]): Connection[] {
  return connectionLists.flat();
}
