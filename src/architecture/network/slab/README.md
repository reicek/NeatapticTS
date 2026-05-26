# architecture/network/slab

Slab Packing / Structure‑of‑Arrays Backend (Educational Module)
==============================================================
Packs per‑connection data into parallel typed arrays (SoA) to accelerate
forward passes and to illustrate memory/layout optimizations.

Why SoA?
 - Locality & fewer cache misses.
 - Predictable tight numeric loops (JIT / SIMD friendly).
 - Easy instrumentation (single contiguous blocks to measure & diff).

Key Arrays (logical length = `used`): weights | from | to | flags | (optional) gain | (optional) plastic.
Adjacency (CSR style): outStart (nodeCount+1), outOrder (per‑source permutation) enabling fast fan‑out.

On‑Demand & Omission:
 - Gain/plastic slabs allocated only when a non‑neutral value appears; freed if neutrality returns.
 - `getConnectionSlab()` synthesizes a neutral gain view if omitted internally (keeps teaching tools simple).

Capacity Strategy: geometric growth (1.25x browser / 1.75x Node) amortizes realloc cost.
Pooling (config gated) reuses typed arrays (see `getSlabAllocationStats`).

Rebuild Steps (sync): reindex nodes → grow/allocate if needed → single pass populate → optional slabs → version++.
Async variant slices the population loop into microtasks to reduce long main‑thread blocks.

Example (inspection):
```ts
const slab = net.getConnectionSlab();
console.log('Edges', slab.used, 'Version', slab.version, 'Cap', slab.capacity);
console.log('First weight from->to', slab.weights[0], slab.from[0], slab.to[0]);
```

## architecture/network/slab/network.slab.utils.types.ts

### BuildAdjacencyContext

Shared immutable inputs used across the CSR adjacency build pipeline for outgoing-order construction.

### ConnectionInternals

Internal Connection properties accessed during slab build, serialization, and typed-array buffer write operations.

### ConnectionSlabView

Packed SoA view returned by getConnectionSlab exposing typed-array weight, index, and flag buffers.

### FanOutCollectionContext

Context for fan-out collection: build inputs plus the output count buffer.

### FastSlabNodeRuntime

Node shape required by fast slab activation kernels for typed-array forward pass inference.

### NetworkActivationRuntime

Runtime activation contract consumed by slab-based forward-pass execution paths for network inference.

### NetworkSlabProps

Internal Network properties used by slab orchestration for typed-array buffer management and dirty tracking.

### NetworkTopoRuntime

Runtime topology contract used to lazily rebuild topological order when the activation cache is dirty.

### OutgoingOrderBuildContext

Context for constructing the source-grouped outgoing connection order from precomputed CSR start indices.

### PoolKeyMetrics

Per-pool-key allocation and reuse counters used for educational diagnostics and memory-pool observability.

### PublishAdjacencyContext

Context for publishing fully built adjacency slabs to internal network state.

### SLAB_DEFAULT_ASYNC_CHUNK_SIZE

Default async slab rebuild chunk size when no override is provided.

### SLAB_GROWTH_FACTOR_BROWSER

Capacity growth factor for browser runtime slab allocations, tuned for tighter memory environments.

### SLAB_GROWTH_FACTOR_NODE

Capacity growth factor for Node.js runtime slab allocations, scaled conservatively to allow large networks.

### SLAB_ONE

Numeric one sentinel used for neutral gain defaults and index math.

### SLAB_ZERO

Numeric zero sentinel used across slab orchestration and helper pipelines.

### SlabBuildContext

Immutable inputs required to build or grow connection slab buffers.

### SlabPopulateResult

Result of scanning and populating optional gain and plastic typed-array slab buffers.

### SlabWriteArrays

Writable typed-array slab buffers targeted during connection serialization and buffer population.

### StartIndicesBuildContext

Context for constructing CSR start offsets from precomputed fan-out counts.

### TypedArray

Union of slab typed array element container types supported by activation buffer allocation.

### TypedArrayConstructor

Constructor type for typed arrays used in activation slab allocation and dynamic buffer growth.

## architecture/network/slab/network.slab.utils.ts

### canUseFastSlab

```ts
canUseFastSlab(
  training: boolean,
): boolean
```

Report whether current network state can use slab fast activation without fallback.
Mirrors `_canUseFastSlab` while exposing eligibility to callers and diagnostics.

Parameters:
- `training` - Whether caller is performing training (disables fast path if true).

Returns: True when slab fast path predicates hold.

### ConnectionSlabView

Packed SoA view returned by getConnectionSlab exposing typed-array weight, index, and flag buffers.

### fastSlabActivate

```ts
fastSlabActivate(
  input: number[],
): number[]
```

High‑performance forward pass using packed slabs + CSR adjacency.

Fallback Conditions (auto‑detected):
 - Missing slabs / adjacency structures.
 - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 - Gating present, when applicable (explicit guard).

Implementation Notes:
 - Reuses internal activation/state buffers to reduce per‑step allocation churn.
 - Applies gain multiplication if optional gain slab exists.
 - Assumes acyclic graph; topological order recomputed on demand if marked dirty.

Parameters:
- `input` - Input vector (length must equal `network.input`).

Returns: Output activations (detached plain array) of length `network.output`.

### getConnectionSlab

```ts
getConnectionSlab(): ConnectionSlabView
```

Obtain (and lazily rebuild if dirty) the current packed SoA view of connections.

Gain Omission: If the internal gain slab is absent (all gains neutral) a synthetic
neutral array is created and returned (NOT retained) to keep external educational
tooling branch‑free while preserving omission memory savings internally.

Returns: Read‑only style view (do not mutate) containing typed arrays + metadata.

### getSlabAllocationStats

```ts
getSlabAllocationStats(): { fresh: number; pooled: number; pool: Record<string, PoolKeyMetrics>; }
```

Allocation statistics snapshot for slab typed arrays.

Includes:
 - fresh: number of newly constructed typed arrays since process start / metrics reset.
 - pooled: number of arrays served from the pool.
 - pool: per‑key metrics (created, reused, maxRetained) for educational inspection.

NOTE: Stats are cumulative (not auto‑reset); callers may diff successive snapshots.

Returns: Plain object copy (safe to serialize) of current allocator counters.

### getSlabVersion

```ts
getSlabVersion(): number
```

Return the monotonic slab rebuild version used to detect stale packed views.

Returns: Non‑negative integer (0 if slab never built yet).

### rebuildConnectionSlab

```ts
rebuildConnectionSlab(
  force: boolean,
): void
```

Build (or refresh) the packed connection slabs for the network synchronously.

ACTIONS
-------
1. Optionally reindex nodes if structural mutations invalidated indices.
2. Grow (geometric) or reuse existing typed arrays to ensure capacity >= active connections.
3. Populate the logical slice [0, connectionCount) with weight/from/to/flag data.
4. Lazily allocate gain & plastic slabs only on first non‑neutral / plastic encounter; omit otherwise.
5. Release previously allocated optional slabs when they revert to neutral / unused (omission optimization).
6. Update internal bookkeeping: logical count, dirty flags, version counter.

PERFORMANCE
-----------
O(C) over active connections with amortized allocation cost due to geometric growth.

Parameters:
- `force` - When true forces rebuild even if network not marked dirty (useful for timing tests).

### rebuildConnectionSlabAsync

```ts
rebuildConnectionSlabAsync(
  chunkSize: number,
): Promise<void>
```

Cooperative asynchronous slab rebuild (Browser only).

Strategy:
 - Perform capacity decision + allocation up front (mirrors sync path).
 - Populate connection data in timer-backed macrotask slices so the browser can service other queued work between chunks.
 - Adaptive slice sizing for very large graphs if `config.browserSlabChunkTargetMs` set.

Metrics: Increments `_slabAsyncBuilds` for observability.
Fallback: On Node (no `window`) defers to synchronous rebuild for simplicity.

Parameters:
- `chunkSize` - Initial maximum connections per slice (may be reduced adaptively for huge graphs).

Returns: Promise resolving once rebuild completes.

## architecture/network/slab/network.slab.pool.utils.ts

Internal helpers for typed-array slab allocation, release, and pool stats.

### _acquireTA

```ts
_acquireTA(
  kind: string,
  ctor: TypedArrayConstructor,
  length: number,
  bytesPerElement: number,
): TypedArray
```

Acquire a typed-array slab for the requested key, reusing pooled capacity
when available and allocating only on pool miss.

Parameters:
- `kind` - Pool kind discriminator.
- `ctor` - Typed array constructor.
- `length` - Desired typed array length.
- `bytesPerElement` - Element byte width for keying.

Returns: Acquired typed array.

### _getSlabAllocationStatsSnapshot

```ts
_getSlabAllocationStatsSnapshot(): { fresh: number; pooled: number; pool: Record<string, PoolKeyMetrics>; }
```

Produce a serializable view of slab allocation telemetry, including global
fresh-versus-pooled counts and per-key pool depth.

Returns: Snapshot containing fresh, pooled, and per-key counters.

### _releaseTA

```ts
_releaseTA(
  kind: string,
  bytesPerElement: number,
  arr: TypedArray,
): void
```

Return a typed array to its bounded per-key slab pool so later activation
passes can reuse capacity without reallocating.

Parameters:
- `kind` - Pool kind discriminator.
- `bytesPerElement` - Element byte width for keying.
- `arr` - Typed array instance to retain when room exists.

Returns: Nothing.

## architecture/network/slab/network.slab.view.utils.ts

### _createConnectionSlabView

```ts
_createConnectionSlabView(
  network: default,
): ConnectionSlabView
```

Creates a read-oriented packed slab view from current network internals.

Parameters:
- `network` - Target network.

Returns: Packed connection slab view.

### _readSlabVersion

```ts
_readSlabVersion(
  network: default,
): number
```

Reads the current monotonic slab version counter from network internals.

Parameters:
- `network` - Target network.

Returns: Non-negative slab version counter.

### _resolveConnectionGainView

```ts
_resolveConnectionGainView(
  internalNet: NetworkSlabProps,
  capacity: number,
): Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | null
```

Resolves gain slab view, synthesizing neutral gain values when omitted.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `capacity` - Resolved slab capacity.

Returns: Gain array view.

### _resolveConnectionSlabCapacity

```ts
_resolveConnectionSlabCapacity(
  internalNet: NetworkSlabProps,
): number
```

Resolves effective slab capacity using explicit capacity first.

Parameters:
- `internalNet` - Internal slab runtime shape.

Returns: Effective capacity value.

## architecture/network/slab/network.slab.setup.utils.ts

### _prepareSlabBuildPreconditions

```ts
_prepareSlabBuildPreconditions(
  buildContext: SlabBuildContext,
): void
```

Applies all required prerequisite normalization steps before starting slab rebuild passes.

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

## architecture/network/slab/network.slab.activate.utils.ts

### _activateFastSlab

```ts
_activateFastSlab(
  network: default,
  input: number[],
): number[]
```

Executes fast slab activation once slab and adjacency prerequisites are prepared.

Parameters:
- `network` - Target network.
- `input` - Input activation vector.

Returns: Output activation array.

## architecture/network/slab/network.slab.shared.helpers.utils.ts

### _reindexNodes

```ts
_reindexNodes(
  network: default,
): void
```

Assigns sequential node indices used by slab packing and fast-path traversal.

Parameters:
- `network` - Target network.

Returns: Nothing.

## architecture/network/slab/network.slab.rebuild.helpers.utils.ts

Internal slab rebuild helper functions extracted from network.slab.utils.ts.

### _allocateCoreSlabArrays

```ts
_allocateCoreSlabArrays(
  buildContext: SlabBuildContext,
): void
```

Allocates core slab arrays (weights/from/to/flags).

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

### _allocateCoreSlabArraysAsync

```ts
_allocateCoreSlabArraysAsync(
  buildContext: SlabBuildContext,
): Promise<void>
```

Allocates async core slabs with optional timer yields between large allocations.

Parameters:
- `buildContext` - Slab build context.

Returns: Promise resolved after async core slabs are ready.

### _allocateGainSlabForAsync

```ts
_allocateGainSlabForAsync(
  buildContext: SlabBuildContext,
): Promise<void>
```

Allocates gain slab for async pass prefill strategy.

Parameters:
- `buildContext` - Slab build context.

Returns: Promise resolved after the gain slab is ready.

### _applyGainOmissionPolicy

```ts
_applyGainOmissionPolicy(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void
```

Applies gain omission rule by releasing a fully neutral gain slab so optional gain storage remains absent unless non-default values are present.

Parameters:
- `buildContext` - Slab build context.
- `populateResult` - Populate result.

Returns: Nothing.

### _applyPlasticPolicyAsync

```ts
_applyPlasticPolicyAsync(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void
```

Applies async plastic slab allocation and release policy so chunked rebuilds keep plastic-rate arrays aligned with observed plastic connection flags.

Parameters:
- `buildContext` - Slab build context.
- `populateResult` - Populate result.

Returns: Nothing.

### _applyPlasticPolicySync

```ts
_applyPlasticPolicySync(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
): void
```

Applies sync plastic slab allocation and release policy so plasticity-rate storage exists only when plastic connections appear in the packed set.

Parameters:
- `buildContext` - Slab build context.
- `populateResult` - Populate result.

Returns: Nothing.

### _createInitialSlabPopulateResult

```ts
_createInitialSlabPopulateResult(
  internalNet: NetworkSlabProps,
): SlabPopulateResult
```

Creates initial populate result from current optional slab state.

Parameters:
- `internalNet` - Internal slab runtime shape.

Returns: Initial populate result.

### _createSlabBuildContext

```ts
_createSlabBuildContext(
  network: default,
  growthFactor: number,
): SlabBuildContext
```

Creates immutable slab build context for one rebuild pass so capacity, precision, and runtime pointers stay consistent across helper calls.

Parameters:
- `network` - Target network.
- `growthFactor` - Capacity growth multiplier.

Returns: Build context.

### _createSlabWriteArrays

```ts
_createSlabWriteArrays(
  buildContext: SlabBuildContext,
): SlabWriteArrays
```

Creates strongly typed write-array bundle for connection population.

Parameters:
- `buildContext` - Slab build context.

Returns: Write-array bundle.

### _ensureGainArrayExistsForIndex

```ts
_ensureGainArrayExistsForIndex(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
  connectionIndex: number,
): void
```

Ensures gain slab exists before writing non-neutral value.

Parameters:
- `buildContext` - Slab build context.
- `populateResult` - Mutable populate result.
- `connectionIndex` - Current connection index.

Returns: Nothing.

### _ensureSlabCapacityAsync

```ts
_ensureSlabCapacityAsync(
  buildContext: SlabBuildContext,
): Promise<void>
```

Ensures async rebuild has enough slab capacity so cooperative chunked population can proceed without mid-pass reallocations or pointer invalidation hazards.

Parameters:
- `buildContext` - Slab build context.

Returns: Promise resolved after any required cooperative allocations finish.

### _ensureSlabCapacitySync

```ts
_ensureSlabCapacitySync(
  buildContext: SlabBuildContext,
): void
```

Ensures sync rebuild has enough slab capacity so packed arrays can hold every active connection before synchronous field population begins.

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

### _expandSlabCapacity

```ts
_expandSlabCapacity(
  currentCapacity: number,
  requiredCapacity: number,
  growthFactor: number,
): number
```

Computes next capacity satisfying required size using geometric growth.

Parameters:
- `currentCapacity` - Existing capacity.
- `requiredCapacity` - Required minimum capacity.
- `growthFactor` - Capacity growth multiplier.

Returns: Expanded capacity.

### _fillPlasticityRates

```ts
_fillPlasticityRates(
  network: default,
  plasticArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  connectionCount: number,
): void
```

Fills plastic slab values from connection plasticity rates.

Parameters:
- `network` - Target network.
- `plasticArray` - Plastic slab array.
- `connectionCount` - Number of active connections.

Returns: Nothing.

### _finalizeAsyncSlabRebuild

```ts
_finalizeAsyncSlabRebuild(
  buildContext: SlabBuildContext,
): void
```

Finalizes async rebuild bookkeeping fields so runtime counters and slab-version invalidation match the completed cooperative population pass in production telemetry.

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

### _finalizeSharedSlabState

```ts
_finalizeSharedSlabState(
  internalNet: NetworkSlabProps,
  connectionCount: number,
): void
```

Finalizes shared rebuild bookkeeping fields.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `connectionCount` - Number of active connections.

Returns: Nothing.

### _finalizeSyncSlabRebuild

```ts
_finalizeSyncSlabRebuild(
  buildContext: SlabBuildContext,
): void
```

Finalizes sync rebuild bookkeeping fields so connection counts, dirty flags, and slab versions remain coherent for downstream runtime caches and adapters.

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

### _populateAsyncChunkRange

```ts
_populateAsyncChunkRange(
  buildContext: SlabBuildContext,
  writeArrays: SlabWriteArrays,
  populateResult: SlabPopulateResult,
  startIndex: number,
  endIndex: number,
): void
```

Populates one inclusive-exclusive chunk range for async rebuild.

Parameters:
- `buildContext` - Slab build context.
- `writeArrays` - Core write arrays.
- `populateResult` - Mutable populate result.
- `startIndex` - Chunk start index.
- `endIndex` - Chunk end index.

Returns: Nothing.

### _populateSlabConnectionsAsync

```ts
_populateSlabConnectionsAsync(
  buildContext: SlabBuildContext,
  chunkSize: number,
): Promise<SlabPopulateResult>
```

Populates core slab arrays in cooperative async chunks so large graphs remain responsive while preserving deterministic packed ordering semantics for replayability.

Parameters:
- `buildContext` - Slab build context.
- `chunkSize` - Maximum items per chunk.

Returns: Population result flags and optional slabs.

### _populateSlabConnectionsSync

```ts
_populateSlabConnectionsSync(
  buildContext: SlabBuildContext,
): SlabPopulateResult
```

Populates core slab arrays in a synchronous single pass so all connection fields are packed deterministically for the active network snapshot.

Parameters:
- `buildContext` - Slab build context.

Returns: Population result flags and optional slabs.

### _releaseExistingSlabArrays

```ts
_releaseExistingSlabArrays(
  buildContext: SlabBuildContext,
): void
```

Releases all currently allocated slab arrays back to pool.

Parameters:
- `buildContext` - Slab build context.

Returns: Nothing.

### _resetOptionalSlabArraysAfterSyncAllocate

```ts
_resetOptionalSlabArraysAfterSyncAllocate(
  internalNet: NetworkSlabProps,
): void
```

Resets optional slabs after sync allocation to keep omission semantics.

Parameters:
- `internalNet` - Internal slab runtime shape.

Returns: Nothing.

### _resolveAsyncChunkSize

```ts
_resolveAsyncChunkSize(
  totalConnections: number,
  requestedChunkSize: number,
): number
```

Resolves effective async chunk size using adaptive heuristics so very large rebuilds honor browser frame budgets without starving throughput under load.

Parameters:
- `totalConnections` - Number of active connections.
- `requestedChunkSize` - Requested chunk size.

Returns: Effective chunk size.

### _shouldSkipSlabRebuild

```ts
_shouldSkipSlabRebuild(
  internalNet: NetworkSlabProps,
  force: boolean,
): boolean
```

Determines whether slab rebuild can be skipped so callers avoid unnecessary typed-array churn when no topology mutation invalidated packed connection buffers.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `force` - True when rebuild must run regardless of dirty state.

Returns: True when rebuild can be skipped.

### _updatePlasticPresence

```ts
_updatePlasticPresence(
  populateResult: SlabPopulateResult,
  connection: ConnectionInternals,
): void
```

Updates plastic-presence flag from connection bitfield.

Parameters:
- `populateResult` - Mutable populate result.
- `connection` - Connection internals.

Returns: Nothing.

### _weightArrayCtor

```ts
_weightArrayCtor(
  useFloat32Weights: boolean | undefined,
): Float32ArrayConstructor | Float64ArrayConstructor
```

Resolves typed-array constructor for weight slabs.

Parameters:
- `useFloat32Weights` - True when 32-bit weights are enabled.

Returns: Matching typed-array constructor.

### _weightByteWidth

```ts
_weightByteWidth(
  useFloat32Weights: boolean | undefined,
): number
```

Resolves byte width for weight slab arrays.

Parameters:
- `useFloat32Weights` - True when 32-bit weights are enabled.

Returns: Byte width for weight elements.

### _writeConnectionCoreFields

```ts
_writeConnectionCoreFields(
  writeArrays: SlabWriteArrays,
  connection: ConnectionInternals,
  connectionIndex: number,
): void
```

Writes core fields for one connection into slab arrays.

Parameters:
- `writeArrays` - Core write arrays.
- `connection` - Connection internals.
- `connectionIndex` - Connection index.

Returns: Nothing.

### _writeConnectionGainField

```ts
_writeConnectionGainField(
  buildContext: SlabBuildContext,
  populateResult: SlabPopulateResult,
  connection: ConnectionInternals,
  connectionIndex: number,
): void
```

Writes gain field for one connection and updates gain flags.

Parameters:
- `buildContext` - Slab build context.
- `populateResult` - Mutable populate result.
- `connection` - Connection internals.
- `connectionIndex` - Connection index.

Returns: Nothing.

### _yieldAsyncChunkMacrotask

```ts
_yieldAsyncChunkMacrotask(): Promise<void>
```

Yields one macrotask turn between browser async slab chunks.

Returns: Promise resolved on the next timer turn.

## architecture/network/slab/network.slab.adjacency.helpers.utils.ts

Internal slab adjacency helpers extracted from network.slab.utils.ts.

### _buildAdjacency

```ts
_buildAdjacency(
  network: default,
): void
```

Build or refresh CSR-style adjacency (outStart + outOrder) for fast fan-out traversal.

Parameters:
- `network` - Target network.

Returns: Nothing.

### asNetworkSlabProps

```ts
asNetworkSlabProps(
  network: default,
): NetworkSlabProps
```

Cast network instance into internal slab-backed shape.

Parameters:
- `network` - Target network.

Returns: Internal slab-backed network representation.

### buildOutgoingOrder

```ts
buildOutgoingOrder(
  outgoingOrderBuildContext: OutgoingOrderBuildContext,
): Uint32Array<ArrayBufferLike>
```

Build source-grouped outgoing order using CSR start offsets.

Parameters:
- `outgoingOrderBuildContext` - Context holding build data and start offsets.

Returns: Ordered outgoing connection indices.

### buildOutgoingStartIndices

```ts
buildOutgoingStartIndices(
  startIndicesBuildContext: StartIndicesBuildContext,
): Uint32Array<ArrayBufferLike>
```

Build CSR start offsets from fan-out counts.

Parameters:
- `startIndicesBuildContext` - Context holding build data and fan-out counts.

Returns: Outgoing start indices slab.

### collectFanOutCounts

```ts
collectFanOutCounts(
  buildContext: BuildAdjacencyContext,
): Uint32Array<ArrayBufferLike>
```

Collect fan-out counts for each source node.

Parameters:
- `buildContext` - Shared adjacency build context.

Returns: Fan-out counts per node.

### createBuildAdjacencyContext

```ts
createBuildAdjacencyContext(
  network: default,
): BuildAdjacencyContext | null
```

Build adjacency context when required slabs are available.

Parameters:
- `network` - Target network.

Returns: Build context or null when adjacency cannot be built yet.

### createFanOutCollectionContext

```ts
createFanOutCollectionContext(
  buildContext: BuildAdjacencyContext,
): FanOutCollectionContext
```

Build fan-out collection context.

Parameters:
- `buildContext` - Shared adjacency build context.

Returns: Fan-out collection context.

### createFanOutCountsBuffer

```ts
createFanOutCountsBuffer(
  nodeCount: number,
): Uint32Array<ArrayBufferLike>
```

Allocate fan-out counts buffer.

Parameters:
- `nodeCount` - Number of nodes.

Returns: Zero-initialized fan-out counts.

### createInsertionCursor

```ts
createInsertionCursor(
  outgoingStartIndices: Uint32Array<ArrayBufferLike>,
): Uint32Array<ArrayBufferLike>
```

Create insertion cursor copy from outgoing start indices.

Parameters:
- `outgoingStartIndices` - Outgoing start indices slab.

Returns: Mutable insertion cursor.

### createOutgoingOrderBuffer

```ts
createOutgoingOrderBuffer(
  connectionCount: number,
): Uint32Array<ArrayBufferLike>
```

Allocate outgoing order buffer.

Parameters:
- `connectionCount` - Number of active connections.

Returns: Outgoing order buffer.

### createOutgoingStartIndicesBuffer

```ts
createOutgoingStartIndicesBuffer(
  nodeCount: number,
): Uint32Array<ArrayBufferLike>
```

Allocate outgoing start indices buffer with terminal slot.

Parameters:
- `nodeCount` - Number of nodes.

Returns: Outgoing start indices buffer.

### hasRequiredConnectionSlabs

```ts
hasRequiredConnectionSlabs(
  internalNet: NetworkSlabProps,
): boolean
```

Check whether required connection slabs exist.

Parameters:
- `internalNet` - Internal slab-backed network representation.

Returns: True when adjacency build prerequisites are present.

### incrementFanOutCountAtSource

```ts
incrementFanOutCountAtSource(
  context: { fanOutCounts: Uint32Array<ArrayBufferLike>; connectionFromSlab: Uint32Array<ArrayBufferLike>; connectionIndex: number; },
): void
```

Increment fan-out count for one connection source index.

Parameters:
- `context` - Increment context.

Returns: Nothing.

### insertConnectionIntoOutgoingOrder

```ts
insertConnectionIntoOutgoingOrder(
  context: { connectionIndex: number; connectionFromSlab: Uint32Array<ArrayBufferLike>; outgoingOrder: Uint32Array<ArrayBufferLike>; insertionCursor: Uint32Array<ArrayBufferLike>; },
): void
```

Insert one connection index into the proper source-grouped slot.

Parameters:
- `context` - Insertion context.

Returns: Nothing.

### iterateConnectionIndices

```ts
iterateConnectionIndices(
  connectionCount: number,
  visitor: (connectionIndex: number) => void,
): void
```

Iterate all connection indices.

Parameters:
- `connectionCount` - Number of active connections.
- `visitor` - Index visitor.

Returns: Nothing.

### iterateNodeIndices

```ts
iterateNodeIndices(
  nodeCount: number,
  visitor: (nodeIndex: number) => void,
): void
```

Iterate all node indices.

Parameters:
- `nodeCount` - Number of nodes.
- `visitor` - Index visitor.

Returns: Nothing.

### populateFanOutCounts

```ts
populateFanOutCounts(
  fanOutCollectionContext: FanOutCollectionContext,
): void
```

Populate fan-out counts from the connection source slab.

Parameters:
- `fanOutCollectionContext` - Fan-out collection context.

Returns: Nothing.

### populateOutgoingOrder

```ts
populateOutgoingOrder(
  context: { connectionCount: number; connectionFromSlab: Uint32Array<ArrayBufferLike>; outgoingOrder: Uint32Array<ArrayBufferLike>; insertionCursor: Uint32Array<ArrayBufferLike>; },
): void
```

Populate outgoing order by source-grouped insertion.

Parameters:
- `context` - Outgoing order population context.

Returns: Nothing.

### populateOutgoingStartIndices

```ts
populateOutgoingStartIndices(
  context: { fanOutCounts: Uint32Array<ArrayBufferLike>; outgoingStartIndices: Uint32Array<ArrayBufferLike>; },
): number
```

Populate outgoing start indices and return terminal offset.

Parameters:
- `context` - Population context.

Returns: Terminal running offset after the last node.

### publishAdjacency

```ts
publishAdjacency(
  publishAdjacencyContext: PublishAdjacencyContext,
): void
```

Publish adjacency slabs and clear dirty flag.

Parameters:
- `publishAdjacencyContext` - Values to publish on the internal network slab state.

Returns: Nothing.

### setTerminalOutgoingStartOffset

```ts
setTerminalOutgoingStartOffset(
  context: { nodeCount: number; outgoingStartIndices: Uint32Array<ArrayBufferLike>; terminalRunningOffset: number; },
): void
```

Set terminal outgoing start offset at the tail slot.

Parameters:
- `context` - Terminal offset context.

Returns: Nothing.

## architecture/network/slab/network.slab.fast-path.helpers.utils.ts

Internal fast-slab activation helpers extracted from slab utilities so the hot-path orchestration remains testable, readable, and isolated from broader runtime wiring concerns.

### _activateThroughLegacyPath

```ts
_activateThroughLegacyPath(
  network: default,
  input: number[],
): number[]
```

Executes legacy network activation fallback.

Parameters:
- `network` - Target network.
- `input` - Activation input.

Returns: Legacy activation output.

### _canUseFastSlab

```ts
_canUseFastSlab(
  training: boolean,
): boolean
```

Evaluate whether the high-performance slab forward pass is currently safe to use under runtime, topology, gating, dropout, and stochastic-regularization constraints.

Parameters:
- `training` - Whether caller is in training mode.

Returns: True if fast path can be safely used.

### _collectFastSlabOutput

```ts
_collectFastSlabOutput(
  network: default,
  activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  nodeCount: number,
): number[]
```

Collect output activations from the slab working buffer into a detached plain array so callers receive stable values independent of pooled buffer reuse.

Parameters:
- `network` - Target network.
- `activationBuffer` - Activation buffer.
- `nodeCount` - Node count.

Returns: Output activation array.

### _createFastActivationBuffer

```ts
_createFastActivationBuffer(
  useFloat32Activation: boolean,
  nodeCount: number,
): Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>
```

Creates typed fast activation/state buffer.

Parameters:
- `useFloat32Activation` - True when 32-bit buffer is required.
- `nodeCount` - Node count.

Returns: New typed buffer.

### _ensureFastSlabBuffers

```ts
_ensureFastSlabBuffers(
  internalNet: NetworkSlabProps,
  nodeCount: number,
): void
```

Ensure reusable fast activation and state buffers are allocated with the correct length and numeric precision for the upcoming slab propagation pass.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `nodeCount` - Node count.

Returns: Nothing.

### _hasFastSlabPrerequisites

```ts
_hasFastSlabPrerequisites(
  internalNet: NetworkSlabProps,
): boolean
```

Checks whether core slab prerequisites are available.

Parameters:
- `internalNet` - Internal slab runtime shape.

Returns: True when all required slabs/adjacency arrays exist.

### _hasFastSlabRegularizationBlockers

```ts
_hasFastSlabRegularizationBlockers(
  network: default,
  internalNet: NetworkSlabProps,
): boolean
```

Check runtime regularization features that invalidate slab fast-path execution.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.

Returns: True when a blocker is present.

### _hasFastSlabStructuralEligibility

```ts
_hasFastSlabStructuralEligibility(
  network: default,
  internalNet: NetworkSlabProps,
): boolean
```

Check topology and structure constraints that must hold before the slab fast path can execute.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.

Returns: True when structure-level fast-path constraints hold.

### _maybeActivateNonInputNode

```ts
_maybeActivateNonInputNode(
  network: default,
  node: FastSlabNodeRuntime,
  nodeIndex: number,
  stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
): void
```

Activates one non-input node when required.

Parameters:
- `network` - Target network.
- `node` - Current node.
- `nodeIndex` - Node index.
- `stateBuffer` - State buffer.
- `activationBuffer` - Activation buffer.

Returns: Nothing.

### _needsFastBufferReplacement

```ts
_needsFastBufferReplacement(
  buffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | undefined,
  nodeCount: number,
  useFloat32Activation: boolean,
): boolean
```

Checks whether a fast buffer requires replacement.

Parameters:
- `buffer` - Existing buffer.
- `nodeCount` - Node count.
- `useFloat32Activation` - True when 32-bit buffer is required.

Returns: True when replacement is needed.

### _prepareFastSlabRuntime

```ts
_prepareFastSlabRuntime(
  network: default,
  internalNet: NetworkSlabProps,
  reindexNodes: (network: default) => void,
): void
```

Prepare topology ordering and node indexing prerequisites before a fast slab pass so activation loops can run on up-to-date structural metadata.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.
- `reindexNodes` - Callback used to reindex nodes when needed.

Returns: Nothing.

### _propagateFastSlabActivations

```ts
_propagateFastSlabActivations(
  network: default,
  internalNet: NetworkSlabProps,
  topoOrder: FastSlabNodeRuntime[],
  activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
): void
```

Propagate activations through topological order using packed slab arrays so weighted fan-out executes with contiguous memory access patterns and deterministic node-to-node accumulation behavior.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.
- `topoOrder` - Topological node order.
- `activationBuffer` - Activation buffer.
- `stateBuffer` - State buffer.

Returns: Nothing.

### _propagateNodeOutgoingEdges

```ts
_propagateNodeOutgoingEdges(
  internalNet: NetworkSlabProps,
  nodeIndex: number,
  activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  weightArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  toIndexArray: Uint32Array<ArrayBufferLike>,
  outgoingOrder: Uint32Array<ArrayBufferLike>,
  outgoingStartIndices: Uint32Array<ArrayBufferLike>,
): void
```

Propagates one node activation over all outgoing slab edges.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `nodeIndex` - Source node index.
- `activationBuffer` - Activation buffer.
- `stateBuffer` - State buffer.
- `weightArray` - Weight slab.
- `toIndexArray` - Destination-index slab.
- `outgoingOrder` - Outgoing edge order slab.
- `outgoingStartIndices` - Outgoing start-offset slab.

Returns: Nothing.

### _recomputeTopologyOrder

```ts
_recomputeTopologyOrder(
  network: default,
): void
```

Recomputes topological order on demand.

Parameters:
- `network` - Target network.

Returns: Nothing.

### _resolveFastTopoOrder

```ts
_resolveFastTopoOrder(
  network: default,
  internalNet: NetworkSlabProps,
): FastSlabNodeRuntime[]
```

Resolve the node iteration order used by fast slab propagation, preferring cached topological order and falling back to node storage order when needed.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.

Returns: Topological node order.

### _resolveWeightedConnectionValue

```ts
_resolveWeightedConnectionValue(
  internalNet: NetworkSlabProps,
  weightArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
  connectionIndex: number,
): number
```

Resolves effective connection weight including optional gain.

Parameters:
- `internalNet` - Internal slab runtime shape.
- `weightArray` - Weight slab.
- `connectionIndex` - Connection index.

Returns: Effective weighted value.

### _seedFastInputLayer

```ts
_seedFastInputLayer(
  network: default,
  input: number[],
  activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>,
): void
```

Seed input-layer activation values into the working buffer and mirror those values onto runtime node fields before hidden-layer propagation begins.

Parameters:
- `network` - Target network.
- `input` - Activation input.
- `activationBuffer` - Activation buffer.

Returns: Nothing.

### _tryFastSlabFallbackForGating

```ts
_tryFastSlabFallbackForGating(
  network: default,
  input: number[],
): number[] | null
```

Attempt an immediate fallback to legacy activation when gating structures are present because gated connections violate the slab fast-path assumptions.

Parameters:
- `network` - Target network.
- `input` - Activation input.

Returns: Legacy output or null when fast path may continue.

### _tryFastSlabFallbackForMissingPrerequisites

```ts
_tryFastSlabFallbackForMissingPrerequisites(
  network: default,
  internalNet: NetworkSlabProps,
  input: number[],
): number[] | null
```

Attempt an immediate fallback to legacy activation when required slab arrays or adjacency prerequisites are missing from runtime state so incomplete slab snapshots never execute unsafe fast-path logic.

Parameters:
- `network` - Target network.
- `internalNet` - Internal slab runtime shape.
- `input` - Activation input.

Returns: Legacy output or null when fast path may continue.

### _writeInputNodeRuntime

```ts
_writeInputNodeRuntime(
  node: default,
  inputValue: number,
): void
```

Writes runtime activation/state for one input node.

Parameters:
- `node` - Input node.
- `inputValue` - Input activation value.

Returns: Nothing.

### resolveFastSlabActivationPrecision

```ts
resolveFastSlabActivationPrecision(
  network: default,
): ActivationPrecision | undefined
```

Read the resolved runtime activation precision for slab output pooling.

Parameters:
- `network` - Target network.

Returns: Active per-network activation precision when present.

### resolveFastSlabBufferPrecision

```ts
resolveFastSlabBufferPrecision(
  internalNet: NetworkSlabProps & { _precisionConfig?: PrecisionConfig | undefined; _activationPrecision?: ActivationPrecision | undefined; },
): ActivationPrecision | undefined
```

Read the resolved runtime activation precision for fast slab working buffers.

Parameters:
- `internalNet` - Internal slab runtime shape.

Returns: Active per-network activation precision when present.
