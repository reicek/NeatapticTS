/**
 * Internal slab rebuild helper functions extracted from network.slab.utils.ts.
 */
import type Network from '../../network';
import type Connection from '../../connection';
import { config } from '../../../config';
import { _acquireTA, _releaseTA } from './network.slab.pool.utils';
import type {
  ConnectionInternals,
  NetworkSlabProps,
  SlabBuildContext,
  SlabPopulateResult,
  SlabWriteArrays,
} from './network.slab.utils.types';

const ZERO = 0;
const ONE = 1;
const BYTE_WIDTH_U8 = 1;
const BYTE_WIDTH_F32 = 4;
const BYTE_WIDTH_U32 = 4;
const BYTE_WIDTH_F64 = 8;
const FLAG_MASK_ONE_BYTE = 0xff;
const PLASTIC_CONNECTION_FLAG = 0b1000;
const LARGE_ASYNC_GRAPH_THRESHOLD = 200_000;
const ADAPTIVE_BASE_OPS_PER_MS = 15_000;
const ADAPTIVE_MIN_OPS = 5_000;
const ADAPTIVE_MAX_OPS = 50_000;

const POOL_KIND_WEIGHTS = 'w';
const POOL_KIND_FROM = 'f';
const POOL_KIND_TO = 't';
const POOL_KIND_FLAGS = 'fl';
const POOL_KIND_GAIN = 'g';
const POOL_KIND_PLASTIC = 'p';

/**
 * Creates immutable slab build context for one rebuild pass.
 *
 * @param network - Target network.
 * @param growthFactor - Capacity growth multiplier.
 * @returns Build context.
 */
export function _createSlabBuildContext(
  network: Network,
  growthFactor: number,
): SlabBuildContext {
  // Step 1: Snapshot mutable slab runtime shape from the network instance.
  const internalNet = network as unknown as NetworkSlabProps;
  // Step 2: Build immutable context used by one rebuild pass.
  return {
    network,
    internalNet,
    connectionCount: network.connections.length,
    capacity: internalNet._connCapacity || ZERO,
    growthFactor,
    weightBytes: _weightByteWidth(internalNet._useFloat32Weights),
    weightCtor: _weightArrayCtor(internalNet._useFloat32Weights),
  };
}

/**
 * Determines whether slab rebuild can be skipped.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param force - True when rebuild must run regardless of dirty state.
 * @returns True when rebuild can be skipped.
 */
export function _shouldSkipSlabRebuild(
  internalNet: NetworkSlabProps,
  force: boolean,
): boolean {
  // Step 1: Skip only when not forced and slabs are already clean.
  return !force && !internalNet._slabDirty;
}

/**
 * Ensures sync rebuild has enough slab capacity.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
export function _ensureSlabCapacitySync(buildContext: SlabBuildContext): void {
  // Step 1: Reuse existing capacity when it already covers active connections.
  if (buildContext.capacity >= buildContext.connectionCount) {
    buildContext.capacity = buildContext.internalNet._connCapacity!;
    return;
  }

  // Step 2: Grow, recycle old slabs, and allocate new core arrays.
  buildContext.capacity = _expandSlabCapacity(
    buildContext.capacity,
    buildContext.connectionCount,
    buildContext.growthFactor,
  );
  _releaseExistingSlabArrays(buildContext);
  _allocateCoreSlabArrays(buildContext);
  _resetOptionalSlabArraysAfterSyncAllocate(buildContext.internalNet);
  // Step 3: Reset optional slabs and persist new capacity.
  buildContext.internalNet._connCapacity = buildContext.capacity;
}

/**
 * Ensures async rebuild has enough slab capacity.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
export function _ensureSlabCapacityAsync(buildContext: SlabBuildContext): void {
  // Step 1: Keep current slabs when capacity is already sufficient.
  if (buildContext.capacity >= buildContext.connectionCount) {
    return;
  }

  // Step 2: Grow and reallocate core slabs for async population.
  buildContext.capacity = _expandSlabCapacity(
    buildContext.capacity,
    buildContext.connectionCount,
    buildContext.growthFactor,
  );
  _releaseExistingSlabArrays(buildContext);
  _allocateCoreSlabArrays(buildContext);
  _allocateGainSlabForAsync(buildContext);
  // Step 3: Prepare async optional slab defaults.
  buildContext.internalNet._connPlastic = null;
  buildContext.internalNet._connCapacity = buildContext.capacity;
}

/**
 * Populates core slab arrays in synchronous single pass.
 *
 * @param buildContext - Slab build context.
 * @returns Population result flags and optional slabs.
 */
export function _populateSlabConnectionsSync(
  buildContext: SlabBuildContext,
): SlabPopulateResult {
  // Step 1: Resolve writable slabs and initialize aggregate flags.
  const writeArrays = _createSlabWriteArrays(buildContext);
  const populateResult = _createInitialSlabPopulateResult(
    buildContext.internalNet,
  );

  // Step 2: Write each connection into packed slab arrays.
  for (
    let connectionIndex = ZERO;
    connectionIndex < buildContext.connectionCount;
    connectionIndex++
  ) {
    const connection = buildContext.network.connections[
      connectionIndex
    ] as unknown as ConnectionInternals;
    _writeConnectionCoreFields(writeArrays, connection, connectionIndex);
    _writeConnectionGainField(
      buildContext,
      populateResult,
      connection,
      connectionIndex,
    );
    _updatePlasticPresence(populateResult, connection);
  }

  // Step 3: Return optional-slab usage summary.
  return populateResult;
}

/**
 * Populates core slab arrays in cooperative async chunks.
 *
 * @param buildContext - Slab build context.
 * @param chunkSize - Maximum items per chunk.
 * @returns Population result flags and optional slabs.
 */
export async function _populateSlabConnectionsAsync(
  buildContext: SlabBuildContext,
  chunkSize: number,
): Promise<SlabPopulateResult> {
  // Step 1: Resolve writable slabs and initialize async cursor state.
  const writeArrays = _createSlabWriteArrays(buildContext);
  const populateResult = _createInitialSlabPopulateResult(
    buildContext.internalNet,
  );
  let connectionIndex = ZERO;

  // Step 2: Populate connections in cooperative chunks.
  while (connectionIndex < buildContext.connectionCount) {
    const chunkEnd = Math.min(
      buildContext.connectionCount,
      connectionIndex + chunkSize,
    );
    _populateAsyncChunkRange(
      buildContext,
      writeArrays,
      populateResult,
      connectionIndex,
      chunkEnd,
    );
    connectionIndex = chunkEnd;
    // Step 3: Yield between chunks to reduce long event-loop blocking.
    if (connectionIndex < buildContext.connectionCount) {
      await Promise.resolve();
    }
  }

  // Step 4: Return optional-slab usage summary.
  return populateResult;
}

/**
 * Applies gain omission rule by releasing neutral gain slab.
 *
 * @param buildContext - Slab build context.
 * @param populateResult - Populate result.
 * @returns Nothing.
 */
export function _applyGainOmissionPolicy(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void {
  // Step 1: Keep gain slab when non-neutral entries were observed.
  if (populateResult.anyNonNeutralGain || !populateResult.gainArray) {
    return;
  }
  // Step 2: Release fully neutral gain slab to preserve omission semantics.
  _releaseTA(
    POOL_KIND_GAIN,
    buildContext.weightBytes,
    populateResult.gainArray,
  );
  buildContext.internalNet._connGain = null;
  populateResult.gainArray = null;
}

/**
 * Applies sync plastic slab allocation/release policy.
 *
 * @param buildContext - Slab build context.
 * @param populateResult - Populate result.
 * @returns Nothing.
 */
export function _applyPlasticPolicySync(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void {
  // Step 1: Allocate and fill plastic slab when plastic connections exist.
  if (populateResult.anyPlastic && !populateResult.plasticArray) {
    const plasticArray = _acquireTA(
      POOL_KIND_PLASTIC,
      buildContext.weightCtor,
      buildContext.capacity,
      buildContext.weightBytes,
    ) as Float32Array | Float64Array;
    buildContext.internalNet._connPlastic = plasticArray;
    _fillPlasticityRates(
      buildContext.network,
      plasticArray,
      buildContext.connectionCount,
    );
    populateResult.plasticArray = plasticArray;
    return;
  }

  // Step 2: Release plastic slab when no plastic connections remain.
  if (!populateResult.anyPlastic && populateResult.plasticArray) {
    _releaseTA(
      POOL_KIND_PLASTIC,
      buildContext.weightBytes,
      populateResult.plasticArray,
    );
    buildContext.internalNet._connPlastic = null;
    populateResult.plasticArray = null;
  }
}

/**
 * Applies async plastic slab allocation/release policy.
 *
 * @param buildContext - Slab build context.
 * @param populateResult - Populate result.
 * @returns Nothing.
 */
export function _applyPlasticPolicyAsync(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void {
  // Step 1: Allocate and fill plastic slab when async pass detected plastic edges.
  if (populateResult.anyPlastic && !populateResult.plasticArray) {
    const plasticArray = _acquireTA(
      POOL_KIND_PLASTIC,
      buildContext.weightCtor,
      buildContext.internalNet._connCapacity!,
      buildContext.weightBytes,
    ) as Float32Array | Float64Array;
    buildContext.internalNet._connPlastic = plasticArray;
    _fillPlasticityRates(
      buildContext.network,
      plasticArray,
      buildContext.connectionCount,
    );
    populateResult.plasticArray = plasticArray;
    return;
  }

  // Step 2: Release plastic slab when async pass found no plastic edges.
  if (!populateResult.anyPlastic && populateResult.plasticArray) {
    _releaseTA(
      POOL_KIND_PLASTIC,
      buildContext.weightBytes,
      populateResult.plasticArray,
    );
    buildContext.internalNet._connPlastic = null;
    populateResult.plasticArray = null;
  }
}

/**
 * Resolves effective async chunk size using adaptive heuristics.
 *
 * @param totalConnections - Number of active connections.
 * @param requestedChunkSize - Requested chunk size.
 * @returns Effective chunk size.
 */
export function _resolveAsyncChunkSize(
  totalConnections: number,
  requestedChunkSize: number,
): number {
  // Step 1: Keep requested chunking for moderate graph sizes.
  if (totalConnections <= LARGE_ASYNC_GRAPH_THRESHOLD) {
    return requestedChunkSize;
  }

  // Step 2: Adapt chunk size from target frame budget when configured.
  const targetMilliseconds = config.browserSlabChunkTargetMs;
  if (typeof targetMilliseconds === 'number' && targetMilliseconds > ZERO) {
    const estimatedOps = Math.max(
      ADAPTIVE_MIN_OPS,
      Math.min(
        ADAPTIVE_MAX_OPS,
        Math.floor(ADAPTIVE_BASE_OPS_PER_MS * targetMilliseconds),
      ),
    );
    return Math.min(requestedChunkSize, estimatedOps);
  }

  // Step 3: Fall back to conservative global chunk cap.
  return Math.min(requestedChunkSize, ADAPTIVE_MAX_OPS);
}

/**
 * Finalizes sync rebuild bookkeeping fields.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
export function _finalizeSyncSlabRebuild(buildContext: SlabBuildContext): void {
  // Step 1: Apply shared bookkeeping updates after sync rebuild.
  _finalizeSharedSlabState(
    buildContext.internalNet,
    buildContext.connectionCount,
  );
}

/**
 * Finalizes async rebuild bookkeeping fields.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
export function _finalizeAsyncSlabRebuild(
  buildContext: SlabBuildContext,
): void {
  // Step 1: Apply shared bookkeeping updates after async rebuild.
  _finalizeSharedSlabState(
    buildContext.internalNet,
    buildContext.connectionCount,
  );
  // Step 2: Increment async rebuild counter for diagnostics.
  buildContext.internalNet._slabAsyncBuilds =
    (buildContext.internalNet._slabAsyncBuilds || ZERO) + ONE;
}

/**
 * Resolves byte width for weight slab arrays.
 *
 * @param useFloat32Weights - True when 32-bit weights are enabled.
 * @returns Byte width for weight elements.
 */
function _weightByteWidth(useFloat32Weights?: boolean): number {
  // Step 1: Resolve byte width from configured weight precision.
  return useFloat32Weights ? BYTE_WIDTH_F32 : BYTE_WIDTH_F64;
}

/**
 * Resolves typed-array constructor for weight slabs.
 *
 * @param useFloat32Weights - True when 32-bit weights are enabled.
 * @returns Matching typed-array constructor.
 */
function _weightArrayCtor(
  useFloat32Weights?: boolean,
): Float32ArrayConstructor | Float64ArrayConstructor {
  // Step 1: Resolve typed-array constructor from weight precision flag.
  return useFloat32Weights ? Float32Array : Float64Array;
}

/**
 * Computes next capacity satisfying required size using geometric growth.
 *
 * @param currentCapacity - Existing capacity.
 * @param requiredCapacity - Required minimum capacity.
 * @param growthFactor - Capacity growth multiplier.
 * @returns Expanded capacity.
 */
function _expandSlabCapacity(
  currentCapacity: number,
  requiredCapacity: number,
  growthFactor: number,
): number {
  // Step 1: Seed growth from required size when capacity is still empty.
  let nextCapacity =
    currentCapacity === ZERO
      ? Math.ceil(requiredCapacity * growthFactor)
      : currentCapacity;
  // Step 2: Grow geometrically until required capacity is satisfied.
  while (nextCapacity < requiredCapacity) {
    nextCapacity = Math.ceil(nextCapacity * growthFactor);
  }
  return nextCapacity;
}

/**
 * Releases all currently allocated slab arrays back to pool.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
function _releaseExistingSlabArrays(buildContext: SlabBuildContext): void {
  // Step 1: Return all currently allocated slabs to the typed-array pool.
  const internalNet = buildContext.internalNet;
  if (internalNet._connWeights) {
    _releaseTA(
      POOL_KIND_WEIGHTS,
      buildContext.weightBytes,
      internalNet._connWeights,
    );
  }
  if (internalNet._connFrom) {
    _releaseTA(
      POOL_KIND_FROM,
      BYTE_WIDTH_U32,
      internalNet._connFrom as Uint32Array,
    );
  }
  if (internalNet._connTo) {
    _releaseTA(
      POOL_KIND_TO,
      BYTE_WIDTH_U32,
      internalNet._connTo as Uint32Array,
    );
  }
  if (internalNet._connFlags) {
    _releaseTA(
      POOL_KIND_FLAGS,
      BYTE_WIDTH_U8,
      internalNet._connFlags as Uint8Array,
    );
  }
  if (internalNet._connGain) {
    _releaseTA(
      POOL_KIND_GAIN,
      buildContext.weightBytes,
      internalNet._connGain as Float32Array | Float64Array,
    );
  }
  if (internalNet._connPlastic) {
    _releaseTA(
      POOL_KIND_PLASTIC,
      buildContext.weightBytes,
      internalNet._connPlastic as Float32Array | Float64Array,
    );
  }
}

/**
 * Allocates core slab arrays (weights/from/to/flags).
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
function _allocateCoreSlabArrays(buildContext: SlabBuildContext): void {
  // Step 1: Allocate/reuse core parallel slabs for packed connection fields.
  const internalNet = buildContext.internalNet;
  const capacity = buildContext.capacity;
  internalNet._connWeights = _acquireTA(
    POOL_KIND_WEIGHTS,
    buildContext.weightCtor,
    capacity,
    buildContext.weightBytes,
  ) as Float32Array | Float64Array;
  internalNet._connFrom = _acquireTA(
    POOL_KIND_FROM,
    Uint32Array,
    capacity,
    BYTE_WIDTH_U32,
  ) as Uint32Array;
  internalNet._connTo = _acquireTA(
    POOL_KIND_TO,
    Uint32Array,
    capacity,
    BYTE_WIDTH_U32,
  ) as Uint32Array;
  internalNet._connFlags = _acquireTA(
    POOL_KIND_FLAGS,
    Uint8Array,
    capacity,
    BYTE_WIDTH_U8,
  ) as Uint8Array;
}

/**
 * Allocates gain slab for async pass prefill strategy.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
function _allocateGainSlabForAsync(buildContext: SlabBuildContext): void {
  // Step 1: Prefill async rebuild with an explicit gain slab.
  buildContext.internalNet._connGain = _acquireTA(
    POOL_KIND_GAIN,
    buildContext.weightCtor,
    buildContext.capacity,
    buildContext.weightBytes,
  ) as Float32Array | Float64Array;
}

/**
 * Resets optional slabs after sync allocation to keep omission semantics.
 *
 * @param internalNet - Internal slab runtime shape.
 * @returns Nothing.
 */
function _resetOptionalSlabArraysAfterSyncAllocate(
  internalNet: NetworkSlabProps,
): void {
  // Step 1: Clear optional slabs so sync path keeps omission semantics.
  internalNet._connGain = null;
  internalNet._connPlastic = null;
}

/**
 * Creates strongly typed write-array bundle for connection population.
 *
 * @param buildContext - Slab build context.
 * @returns Write-array bundle.
 */
function _createSlabWriteArrays(
  buildContext: SlabBuildContext,
): SlabWriteArrays {
  // Step 1: Collect strongly typed writable slab references.
  return {
    weightArray: buildContext.internalNet._connWeights as
      | Float32Array
      | Float64Array,
    fromIndexArray: buildContext.internalNet._connFrom as Uint32Array,
    toIndexArray: buildContext.internalNet._connTo as Uint32Array,
    flagArray: buildContext.internalNet._connFlags as Uint8Array,
  };
}

/**
 * Creates initial populate result from current optional slab state.
 *
 * @param internalNet - Internal slab runtime shape.
 * @returns Initial populate result.
 */
function _createInitialSlabPopulateResult(
  internalNet: NetworkSlabProps,
): SlabPopulateResult {
  // Step 1: Initialize optional-slab usage tracking flags and references.
  return {
    anyNonNeutralGain: false,
    anyPlastic: false,
    gainArray: internalNet._connGain as Float32Array | Float64Array | null,
    plasticArray: internalNet._connPlastic as
      | Float32Array
      | Float64Array
      | null,
  };
}

/**
 * Populates one inclusive-exclusive chunk range for async rebuild.
 *
 * @param buildContext - Slab build context.
 * @param writeArrays - Core write arrays.
 * @param populateResult - Mutable populate result.
 * @param startIndex - Chunk start index.
 * @param endIndex - Chunk end index.
 * @returns Nothing.
 */
function _populateAsyncChunkRange(
  buildContext: SlabBuildContext,
  writeArrays: SlabWriteArrays,
  populateResult: SlabPopulateResult,
  startIndex: number,
  endIndex: number,
): void {
  // Step 1: Populate one inclusive-exclusive connection chunk.
  for (
    let connectionIndex = startIndex;
    connectionIndex < endIndex;
    connectionIndex++
  ) {
    const connection = buildContext.network.connections[
      connectionIndex
    ] as unknown as ConnectionInternals;
    _writeConnectionCoreFields(writeArrays, connection, connectionIndex);
    _writeConnectionGainField(
      buildContext,
      populateResult,
      connection,
      connectionIndex,
    );
    _updatePlasticPresence(populateResult, connection);
  }
}

/**
 * Writes core fields for one connection into slab arrays.
 *
 * @param writeArrays - Core write arrays.
 * @param connection - Connection internals.
 * @param connectionIndex - Connection index.
 * @returns Nothing.
 */
function _writeConnectionCoreFields(
  writeArrays: SlabWriteArrays,
  connection: ConnectionInternals,
  connectionIndex: number,
): void {
  // Step 1: Write core packed connection fields to slab arrays.
  writeArrays.weightArray[connectionIndex] = connection.weight;
  writeArrays.fromIndexArray[connectionIndex] = connection.from.index >>> ZERO;
  writeArrays.toIndexArray[connectionIndex] = connection.to.index >>> ZERO;
  writeArrays.flagArray[connectionIndex] =
    connection._flags & FLAG_MASK_ONE_BYTE;
}

/**
 * Writes gain field for one connection and updates gain flags.
 *
 * @param buildContext - Slab build context.
 * @param populateResult - Mutable populate result.
 * @param connection - Connection internals.
 * @param connectionIndex - Connection index.
 * @returns Nothing.
 */
function _writeConnectionGainField(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
  connection: ConnectionInternals,
  connectionIndex: number,
): void {
  // Step 1: Write non-neutral gain values and allocate gain slab on demand.
  const gainValue = (connection as unknown as Connection).gain;
  if (gainValue !== ONE) {
    _ensureGainArrayExistsForIndex(
      buildContext,
      populateResult,
      connectionIndex,
    );
    (populateResult.gainArray as Float32Array | Float64Array)[connectionIndex] =
      gainValue;
    populateResult.anyNonNeutralGain = true;
    return;
  }

  // Step 2: Preserve explicit neutral gain values when gain slab already exists.
  if (populateResult.gainArray) {
    populateResult.gainArray[connectionIndex] = ONE;
  }
}

/**
 * Ensures gain slab exists before writing non-neutral value.
 *
 * @param buildContext - Slab build context.
 * @param populateResult - Mutable populate result.
 * @param connectionIndex - Current connection index.
 * @returns Nothing.
 */
function _ensureGainArrayExistsForIndex(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
  connectionIndex: number,
): void {
  // Step 1: Return early when gain slab already exists.
  if (populateResult.gainArray) {
    return;
  }

  // Step 2: Allocate gain slab and backfill prior neutral entries.
  const gainArray = _acquireTA(
    POOL_KIND_GAIN,
    buildContext.weightCtor,
    buildContext.capacity,
    buildContext.weightBytes,
  ) as Float32Array | Float64Array;
  for (let fillIndex = ZERO; fillIndex < connectionIndex; fillIndex++) {
    gainArray[fillIndex] = ONE;
  }
  // Step 3: Publish allocated gain slab references.
  buildContext.internalNet._connGain = gainArray;
  populateResult.gainArray = gainArray;
}

/**
 * Updates plastic-presence flag from connection bitfield.
 *
 * @param populateResult - Mutable populate result.
 * @param connection - Connection internals.
 * @returns Nothing.
 */
function _updatePlasticPresence(
  populateResult: SlabPopulateResult,
  connection: ConnectionInternals,
): void {
  // Step 1: Track whether any connection in the pass is plastic.
  if (connection._flags & PLASTIC_CONNECTION_FLAG) {
    populateResult.anyPlastic = true;
  }
}

/**
 * Fills plastic slab values from connection plasticity rates.
 *
 * @param network - Target network.
 * @param plasticArray - Plastic slab array.
 * @param connectionCount - Number of active connections.
 * @returns Nothing.
 */
function _fillPlasticityRates(
  network: Network,
  plasticArray: Float32Array | Float64Array,
  connectionCount: number,
): void {
  // Step 1: Copy per-connection plasticity rates into packed slab.
  for (
    let connectionIndex = ZERO;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    const connection = network.connections[connectionIndex] as Connection & {
      plasticityRate?: number;
    };
    plasticArray[connectionIndex] = connection.plasticityRate || ZERO;
  }
}

/**
 * Finalizes shared rebuild bookkeeping fields.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param connectionCount - Number of active connections.
 * @returns Nothing.
 */
function _finalizeSharedSlabState(
  internalNet: NetworkSlabProps,
  connectionCount: number,
): void {
  // Step 1: Persist active connection count and clear dirty flags.
  internalNet._connCount = connectionCount;
  internalNet._slabDirty = false;
  internalNet._adjDirty = true;
  // Step 2: Advance slab version for downstream cache invalidation.
  internalNet._slabVersion = (internalNet._slabVersion || ZERO) + ONE;
}
