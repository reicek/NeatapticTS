/**
 * Internal slab adjacency helpers extracted from network.slab.utils.ts.
 */
import type Network from '../../network';
import type {
  BuildAdjacencyContext,
  FanOutCollectionContext,
  NetworkSlabProps,
  OutgoingOrderBuildContext,
  PublishAdjacencyContext,
  StartIndicesBuildContext,
} from './network.slab.utils.types';

const RANGE_START_INDEX = 0;
const NEXT_INDEX_OFFSET = 1;
const INITIAL_RUNNING_OFFSET = 0;
const ADJACENCY_CLEAN_DIRTY_FLAG = false;

/**
 * Build or refresh CSR-style adjacency (outStart + outOrder) for fast fan-out traversal.
 *
 * @param network - Target network.
 * @returns Nothing.
 */
export function _buildAdjacency(network: Network): void {
  // Step 1: Resolve strongly-typed build context and exit early when slabs are unavailable.
  const buildContext = createBuildAdjacencyContext(network);
  if (!buildContext) {
    return;
  }

  // Step 2: Collect per-node fan-out counts from connection sources.
  const fanOutCounts = collectFanOutCounts(buildContext);

  // Step 3: Build CSR start indices from fan-out counts.
  const outgoingStartIndices = buildOutgoingStartIndices({
    buildContext,
    fanOutCounts,
  });

  // Step 4: Build connection order grouped by source node.
  const outgoingOrder = buildOutgoingOrder({
    buildContext,
    outgoingStartIndices,
  });

  // Step 5: Publish adjacency slabs and mark adjacency as clean.
  publishAdjacency({
    internalNet: buildContext.internalNet,
    outgoingStartIndices,
    outgoingOrder,
  });
}

/**
 * Build adjacency context when required slabs are available.
 *
 * @param network - Target network.
 * @returns Build context or null when adjacency cannot be built yet.
 */
function createBuildAdjacencyContext(
  network: Network,
): BuildAdjacencyContext | null {
  const internalNet = asNetworkSlabProps(network);
  if (!hasRequiredConnectionSlabs(internalNet)) {
    return null;
  }

  const connectionFromSlab = internalNet._connFrom;
  return {
    internalNet,
    nodeCount: network.nodes.length,
    connectionCount: internalNet._connCount ?? connectionFromSlab.length,
    connectionFromSlab,
  };
}

/**
 * Collect fan-out counts for each source node.
 *
 * @param buildContext - Shared adjacency build context.
 * @returns Fan-out counts per node.
 */
function collectFanOutCounts(buildContext: BuildAdjacencyContext): Uint32Array {
  const fanOutCollectionContext = createFanOutCollectionContext(buildContext);
  populateFanOutCounts(fanOutCollectionContext);
  return fanOutCollectionContext.fanOutCounts;
}

/**
 * Build CSR start offsets from fan-out counts.
 *
 * @param startIndicesBuildContext - Context holding build data and fan-out counts.
 * @returns Outgoing start indices slab.
 */
function buildOutgoingStartIndices(
  startIndicesBuildContext: StartIndicesBuildContext,
): Uint32Array {
  const outgoingStartIndices = createOutgoingStartIndicesBuffer(
    startIndicesBuildContext.buildContext.nodeCount,
  );
  const terminalRunningOffset = populateOutgoingStartIndices({
    fanOutCounts: startIndicesBuildContext.fanOutCounts,
    outgoingStartIndices,
  });
  setTerminalOutgoingStartOffset({
    nodeCount: startIndicesBuildContext.buildContext.nodeCount,
    outgoingStartIndices,
    terminalRunningOffset,
  });
  return outgoingStartIndices;
}

/**
 * Build source-grouped outgoing order using CSR start offsets.
 *
 * @param outgoingOrderBuildContext - Context holding build data and start offsets.
 * @returns Ordered outgoing connection indices.
 */
function buildOutgoingOrder(
  outgoingOrderBuildContext: OutgoingOrderBuildContext,
): Uint32Array {
  const outgoingOrder = createOutgoingOrderBuffer(
    outgoingOrderBuildContext.buildContext.connectionCount,
  );
  const insertionCursor = createInsertionCursor(
    outgoingOrderBuildContext.outgoingStartIndices,
  );
  populateOutgoingOrder({
    connectionCount: outgoingOrderBuildContext.buildContext.connectionCount,
    connectionFromSlab:
      outgoingOrderBuildContext.buildContext.connectionFromSlab,
    outgoingOrder,
    insertionCursor,
  });
  return outgoingOrder;
}

/**
 * Publish adjacency slabs and clear dirty flag.
 *
 * @param publishAdjacencyContext - Values to publish on the internal network slab state.
 * @returns Nothing.
 */
function publishAdjacency(
  publishAdjacencyContext: PublishAdjacencyContext,
): void {
  publishAdjacencyContext.internalNet._outStart =
    publishAdjacencyContext.outgoingStartIndices;
  publishAdjacencyContext.internalNet._outOrder =
    publishAdjacencyContext.outgoingOrder;
  publishAdjacencyContext.internalNet._adjDirty = ADJACENCY_CLEAN_DIRTY_FLAG;
}

/**
 * Build fan-out collection context.
 *
 * @param buildContext - Shared adjacency build context.
 * @returns Fan-out collection context.
 */
function createFanOutCollectionContext(
  buildContext: BuildAdjacencyContext,
): FanOutCollectionContext {
  return {
    buildContext,
    fanOutCounts: createFanOutCountsBuffer(buildContext.nodeCount),
  };
}

/**
 * Populate fan-out counts from the connection source slab.
 *
 * @param fanOutCollectionContext - Fan-out collection context.
 * @returns Nothing.
 */
function populateFanOutCounts(
  fanOutCollectionContext: FanOutCollectionContext,
): void {
  iterateConnectionIndices(
    fanOutCollectionContext.buildContext.connectionCount,
    (connectionIndex) => {
      incrementFanOutCountAtSource({
        fanOutCounts: fanOutCollectionContext.fanOutCounts,
        connectionFromSlab:
          fanOutCollectionContext.buildContext.connectionFromSlab,
        connectionIndex,
      });
    },
  );
}

/**
 * Increment fan-out count for one connection source index.
 *
 * @param context - Increment context.
 * @returns Nothing.
 */
function incrementFanOutCountAtSource(context: {
  fanOutCounts: Uint32Array;
  connectionFromSlab: Uint32Array;
  connectionIndex: number;
}): void {
  const fromNodeIndex = context.connectionFromSlab[context.connectionIndex];
  context.fanOutCounts[fromNodeIndex]++;
}

/**
 * Allocate fan-out counts buffer.
 *
 * @param nodeCount - Number of nodes.
 * @returns Zero-initialized fan-out counts.
 */
function createFanOutCountsBuffer(nodeCount: number): Uint32Array {
  return new Uint32Array(nodeCount);
}

/**
 * Allocate outgoing start indices buffer with terminal slot.
 *
 * @param nodeCount - Number of nodes.
 * @returns Outgoing start indices buffer.
 */
function createOutgoingStartIndicesBuffer(nodeCount: number): Uint32Array {
  return new Uint32Array(nodeCount + NEXT_INDEX_OFFSET);
}

/**
 * Populate outgoing start indices and return terminal offset.
 *
 * @param context - Population context.
 * @returns Terminal running offset after the last node.
 */
function populateOutgoingStartIndices(context: {
  fanOutCounts: Uint32Array;
  outgoingStartIndices: Uint32Array;
}): number {
  let runningOffset = INITIAL_RUNNING_OFFSET;

  iterateNodeIndices(context.fanOutCounts.length, (nodeIndex) => {
    context.outgoingStartIndices[nodeIndex] = runningOffset;
    runningOffset += context.fanOutCounts[nodeIndex];
  });

  return runningOffset;
}

/**
 * Set terminal outgoing start offset at the tail slot.
 *
 * @param context - Terminal offset context.
 * @returns Nothing.
 */
function setTerminalOutgoingStartOffset(context: {
  nodeCount: number;
  outgoingStartIndices: Uint32Array;
  terminalRunningOffset: number;
}): void {
  context.outgoingStartIndices[context.nodeCount] =
    context.terminalRunningOffset;
}

/**
 * Allocate outgoing order buffer.
 *
 * @param connectionCount - Number of active connections.
 * @returns Outgoing order buffer.
 */
function createOutgoingOrderBuffer(connectionCount: number): Uint32Array {
  return new Uint32Array(connectionCount);
}

/**
 * Create insertion cursor copy from outgoing start indices.
 *
 * @param outgoingStartIndices - Outgoing start indices slab.
 * @returns Mutable insertion cursor.
 */
function createInsertionCursor(outgoingStartIndices: Uint32Array): Uint32Array {
  return outgoingStartIndices.slice();
}

/**
 * Populate outgoing order by source-grouped insertion.
 *
 * @param context - Outgoing order population context.
 * @returns Nothing.
 */
function populateOutgoingOrder(context: {
  connectionCount: number;
  connectionFromSlab: Uint32Array;
  outgoingOrder: Uint32Array;
  insertionCursor: Uint32Array;
}): void {
  iterateConnectionIndices(context.connectionCount, (connectionIndex) => {
    insertConnectionIntoOutgoingOrder({
      connectionIndex,
      connectionFromSlab: context.connectionFromSlab,
      outgoingOrder: context.outgoingOrder,
      insertionCursor: context.insertionCursor,
    });
  });
}

/**
 * Insert one connection index into the proper source-grouped slot.
 *
 * @param context - Insertion context.
 * @returns Nothing.
 */
function insertConnectionIntoOutgoingOrder(context: {
  connectionIndex: number;
  connectionFromSlab: Uint32Array;
  outgoingOrder: Uint32Array;
  insertionCursor: Uint32Array;
}): void {
  const fromNodeIndex = context.connectionFromSlab[context.connectionIndex];
  const insertionIndex = context.insertionCursor[fromNodeIndex];
  context.outgoingOrder[insertionIndex] = context.connectionIndex;
  context.insertionCursor[fromNodeIndex] = insertionIndex + NEXT_INDEX_OFFSET;
}

/**
 * Iterate all connection indices.
 *
 * @param connectionCount - Number of active connections.
 * @param visitor - Index visitor.
 * @returns Nothing.
 */
function iterateConnectionIndices(
  connectionCount: number,
  visitor: (connectionIndex: number) => void,
): void {
  for (
    let connectionIndex = RANGE_START_INDEX;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    visitor(connectionIndex);
  }
}

/**
 * Iterate all node indices.
 *
 * @param nodeCount - Number of nodes.
 * @param visitor - Index visitor.
 * @returns Nothing.
 */
function iterateNodeIndices(
  nodeCount: number,
  visitor: (nodeIndex: number) => void,
): void {
  for (let nodeIndex = RANGE_START_INDEX; nodeIndex < nodeCount; nodeIndex++) {
    visitor(nodeIndex);
  }
}

/**
 * Check whether required connection slabs exist.
 *
 * @param internalNet - Internal slab-backed network representation.
 * @returns True when adjacency build prerequisites are present.
 */
function hasRequiredConnectionSlabs(
  internalNet: NetworkSlabProps,
): internalNet is NetworkSlabProps & {
  _connFrom: Uint32Array;
  _connTo: Uint32Array;
} {
  return Boolean(internalNet._connFrom && internalNet._connTo);
}

/**
 * Cast network instance into internal slab-backed shape.
 *
 * @param network - Target network.
 * @returns Internal slab-backed network representation.
 */
function asNetworkSlabProps(network: Network): NetworkSlabProps {
  return network as unknown as NetworkSlabProps;
}
