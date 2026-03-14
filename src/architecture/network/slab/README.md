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
const slab = (net as any).getConnectionSlab();
console.log('Edges', slab.used, 'Version', slab.version, 'Cap', slab.capacity);
console.log('First weight from->to', slab.weights[0], slab.from[0], slab.to[0]);
```

## architecture/network/slab/network.slab.utils.types.ts

### BuildAdjacencyContext

Shared immutable inputs used across the adjacency build pipeline.

### ConnectionInternals

Internal Connection properties accessed during slab operations.

### ConnectionSlabView

Shape returned by getConnectionSlab describing the packed SoA view.

### FanOutCollectionContext

Context for fan-out collection: build inputs plus the output count buffer.

### FastSlabNodeRuntime

Node shape required by fast slab activation kernels.

### NetworkActivationRuntime

Runtime activation contract used by slab-based execution paths.

### NetworkSlabProps

Internal Network properties for slab operations.

### NetworkTopoRuntime

Runtime topology contract used to lazily rebuild topological order.

### OutgoingOrderBuildContext

Context for constructing source-grouped outgoing connection order.

### PoolKeyMetrics

Per-pool-key allocation & reuse counters (educational / diagnostics).

### PublishAdjacencyContext

Context for publishing fully built adjacency slabs to internal network state.

### SLAB_DEFAULT_ASYNC_CHUNK_SIZE

Default async slab rebuild chunk size when no override is provided.

### SLAB_GROWTH_FACTOR_BROWSER

Capacity growth factor for browser slab allocations.

### SLAB_GROWTH_FACTOR_NODE

Capacity growth factor for Node.js slab allocations.

### SLAB_ONE

Numeric one sentinel used for neutral gain defaults and index math.

### SLAB_ZERO

Numeric zero sentinel used across slab orchestration and helper pipelines.

### SlabBuildContext

Immutable inputs required to build or grow connection slab buffers.

### SlabPopulateResult

Result of scanning and populating optional gain/plastic slab arrays.

### SlabWriteArrays

Writable slab arrays targeted during connection serialization.

### StartIndicesBuildContext

Context for constructing CSR start offsets from precomputed fan-out counts.

### TypedArray

Union of slab typed array element container types.

### TypedArrayConstructor

Constructor type for typed arrays used in slabs.

## architecture/network/slab/network.slab.utils.ts

### canUseFastSlab

`(training: boolean) => boolean`

Public convenience wrapper exposing fast path eligibility.
Mirrors `_canUseFastSlab` internal predicate.

Parameters:
- `training` - Whether caller is performing training (disables fast path if true).

Returns: True when slab fast path predicates hold.

### ConnectionSlabView

Shape returned by getConnectionSlab describing the packed SoA view.

### fastSlabActivate

`(input: number[]) => number[]`

High‑performance forward pass using packed slabs + CSR adjacency.

Fallback Conditions (auto‑detected):
 - Missing slabs / adjacency structures.
 - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 - Any gating present (explicit guard).

Implementation Notes:
 - Reuses internal activation/state buffers to reduce per‑step allocation churn.
 - Applies gain multiplication if optional gain slab exists.
 - Assumes acyclic graph; topological order recomputed on demand if marked dirty.

Parameters:
- `input` - Input vector (length must equal `network.input`).

Returns: Output activations (detached plain array) of length `network.output`.

### getConnectionSlab

`() => import("src/architecture/network/slab/network.slab.utils.types").ConnectionSlabView`

Obtain (and lazily rebuild if dirty) the current packed SoA view of connections.

Gain Omission: If the internal gain slab is absent (all gains neutral) a synthetic
neutral array is created and returned (NOT retained) to keep external educational
tooling branch‑free while preserving omission memory savings internally.

Returns: Read‑only style view (do not mutate) containing typed arrays + metadata.

### getSlabAllocationStats

`() => { pool: { [x: string]: import("src/architecture/network/slab/network.slab.utils.types").PoolKeyMetrics; }; fresh: number; pooled: number; }`

Allocation statistics snapshot for slab typed arrays.

Includes:
 - fresh: number of newly constructed typed arrays since process start / metrics reset.
 - pooled: number of arrays served from the pool.
 - pool: per‑key metrics (created, reused, maxRetained) for educational inspection.

NOTE: Stats are cumulative (not auto‑reset); callers may diff successive snapshots.

Returns: Plain object copy (safe to serialize) of current allocator counters.

### getSlabVersion

`() => number`

Retrieve current monotonic slab version (increments on each successful rebuild).

Returns: Non‑negative integer (0 if slab never built yet).

### rebuildConnectionSlab

`(force: boolean) => void`

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

`(chunkSize: number) => Promise<void>`

Cooperative asynchronous slab rebuild (Browser only).

Strategy:
 - Perform capacity decision + allocation up front (mirrors sync path).
 - Populate connection data in microtask slices (yield via resolved Promise) to avoid long main‑thread stalls.
 - Adaptive slice sizing for very large graphs if `config.browserSlabChunkTargetMs` set.

Metrics: Increments `_slabAsyncBuilds` for observability.
Fallback: On Node (no `window`) defers to synchronous rebuild for simplicity.

Parameters:
- `chunkSize` - Initial maximum connections per slice (may be reduced adaptively for huge graphs).

Returns: Promise resolving once rebuild completes.

## architecture/network/slab/network.slab.pool.utils.ts

Internal slab pool/stat helpers extracted from network.slab.utils.ts.

### _acquireTA

`(kind: string, ctor: import("src/architecture/network/slab/network.slab.utils.types").TypedArrayConstructor, length: number, bytesPerElement: number) => import("src/architecture/network/slab/network.slab.utils.types").TypedArray`

Acquires a typed array from pool or allocates a fresh one.

Parameters:
- `kind` - - Pool kind discriminator.
- `ctor` - - Typed array constructor.
- `length` - - Desired typed array length.
- `bytesPerElement` - - Element byte width for keying.

Returns: Acquired typed array.

### _getSlabAllocationStatsSnapshot

`() => { pool: { [x: string]: import("src/architecture/network/slab/network.slab.utils.types").PoolKeyMetrics; }; fresh: number; pooled: number; }`

Returns allocation stats snapshot for slab typed arrays.

Returns: Serializable snapshot of fresh, pooled, and per-key metrics.

### _poolKey

`(kind: string, bytes: number, length: number) => string`

Creates a stable pool key from kind, element width, and length.

Parameters:
- `kind` - - Short pool kind discriminator.
- `bytes` - - Bytes per element.
- `length` - - Typed array logical length.

Returns: Stable pool key.

### _releaseTA

`(kind: string, bytesPerElement: number, arr: import("src/architecture/network/slab/network.slab.utils.types").TypedArray) => void`

Releases a typed array back to bounded per-key pool.

Parameters:
- `kind` - - Pool kind discriminator.
- `bytesPerElement` - - Element byte width for keying.
- `arr` - - Typed array instance to retain when room exists.

Returns: Nothing.

### _slabPoolCap

`() => number`

Computes retention cap per key.

Returns: Non-negative max retained arrays per key.

## architecture/network/slab/network.slab.view.utils.ts

### _createConnectionSlabView

`(network: import("src/architecture/network").default) => import("src/architecture/network/slab/network.slab.utils.types").ConnectionSlabView`

Creates a read-oriented packed slab view from current network internals.

Parameters:
- `network` - - Target network.

Returns: Packed connection slab view.

### _readSlabVersion

`(network: import("src/architecture/network").default) => number`

Reads the current monotonic slab version from network internals.

Parameters:
- `network` - - Target network.

Returns: Non-negative slab version counter.

### _resolveConnectionGainView

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, capacity: number) => Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | null`

Resolves gain slab view, synthesizing neutral gain values when omitted.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `capacity` - - Resolved slab capacity.

Returns: Gain array view.

### _resolveConnectionSlabCapacity

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => number`

Resolves effective slab capacity using explicit capacity first.

Parameters:
- `internalNet` - - Internal slab runtime shape.

Returns: Effective capacity value.

## architecture/network/slab/network.slab.setup.utils.ts

### _prepareSlabBuildPreconditions

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Applies prerequisite normalization for slab rebuild passes.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

## architecture/network/slab/network.slab.activate.utils.ts

### _activateFastSlab

`(network: import("src/architecture/network").default, input: number[]) => number[]`

Executes fast slab activation once slab and adjacency prerequisites are prepared.

Parameters:
- `network` - - Target network.
- `input` - - Input activation vector.

Returns: Output activation array.

## architecture/network/slab/network.slab.shared.helpers.utils.ts

### _reindexNodes

`(network: import("src/architecture/network").default) => void`

Assigns sequential node indices used by slab packing and fast-path traversal.

Parameters:
- `network` - - Target network.

Returns: Nothing.

## architecture/network/slab/network.slab.rebuild.helpers.utils.ts

Internal slab rebuild helper functions extracted from network.slab.utils.ts.

### _allocateCoreSlabArrays

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Allocates core slab arrays (weights/from/to/flags).

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _allocateGainSlabForAsync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Allocates gain slab for async pass prefill strategy.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _applyGainOmissionPolicy

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult) => void`

Applies gain omission rule by releasing neutral gain slab.

Parameters:
- `buildContext` - - Slab build context.
- `populateResult` - - Populate result.

Returns: Nothing.

### _applyPlasticPolicyAsync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult) => void`

Applies async plastic slab allocation/release policy.

Parameters:
- `buildContext` - - Slab build context.
- `populateResult` - - Populate result.

Returns: Nothing.

### _applyPlasticPolicySync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult) => void`

Applies sync plastic slab allocation/release policy.

Parameters:
- `buildContext` - - Slab build context.
- `populateResult` - - Populate result.

Returns: Nothing.

### _createInitialSlabPopulateResult

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult`

Creates initial populate result from current optional slab state.

Parameters:
- `internalNet` - - Internal slab runtime shape.

Returns: Initial populate result.

### _createSlabBuildContext

`(network: import("src/architecture/network").default, growthFactor: number) => import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext`

Creates immutable slab build context for one rebuild pass.

Parameters:
- `network` - - Target network.
- `growthFactor` - - Capacity growth multiplier.

Returns: Build context.

### _createSlabWriteArrays

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => import("src/architecture/network/slab/network.slab.utils.types").SlabWriteArrays`

Creates strongly typed write-array bundle for connection population.

Parameters:
- `buildContext` - - Slab build context.

Returns: Write-array bundle.

### _ensureGainArrayExistsForIndex

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult, connectionIndex: number) => void`

Ensures gain slab exists before writing non-neutral value.

Parameters:
- `buildContext` - - Slab build context.
- `populateResult` - - Mutable populate result.
- `connectionIndex` - - Current connection index.

Returns: Nothing.

### _ensureSlabCapacityAsync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Ensures async rebuild has enough slab capacity.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _ensureSlabCapacitySync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Ensures sync rebuild has enough slab capacity.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _expandSlabCapacity

`(currentCapacity: number, requiredCapacity: number, growthFactor: number) => number`

Computes next capacity satisfying required size using geometric growth.

Parameters:
- `currentCapacity` - - Existing capacity.
- `requiredCapacity` - - Required minimum capacity.
- `growthFactor` - - Capacity growth multiplier.

Returns: Expanded capacity.

### _fillPlasticityRates

`(network: import("src/architecture/network").default, plasticArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, connectionCount: number) => void`

Fills plastic slab values from connection plasticity rates.

Parameters:
- `network` - - Target network.
- `plasticArray` - - Plastic slab array.
- `connectionCount` - - Number of active connections.

Returns: Nothing.

### _finalizeAsyncSlabRebuild

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Finalizes async rebuild bookkeeping fields.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _finalizeSharedSlabState

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, connectionCount: number) => void`

Finalizes shared rebuild bookkeeping fields.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `connectionCount` - - Number of active connections.

Returns: Nothing.

### _finalizeSyncSlabRebuild

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Finalizes sync rebuild bookkeeping fields.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _populateAsyncChunkRange

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, writeArrays: import("src/architecture/network/slab/network.slab.utils.types").SlabWriteArrays, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult, startIndex: number, endIndex: number) => void`

Populates one inclusive-exclusive chunk range for async rebuild.

Parameters:
- `buildContext` - - Slab build context.
- `writeArrays` - - Core write arrays.
- `populateResult` - - Mutable populate result.
- `startIndex` - - Chunk start index.
- `endIndex` - - Chunk end index.

Returns: Nothing.

### _populateSlabConnectionsAsync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, chunkSize: number) => Promise<import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult>`

Populates core slab arrays in cooperative async chunks.

Parameters:
- `buildContext` - - Slab build context.
- `chunkSize` - - Maximum items per chunk.

Returns: Population result flags and optional slabs.

### _populateSlabConnectionsSync

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult`

Populates core slab arrays in synchronous single pass.

Parameters:
- `buildContext` - - Slab build context.

Returns: Population result flags and optional slabs.

### _releaseExistingSlabArrays

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext) => void`

Releases all currently allocated slab arrays back to pool.

Parameters:
- `buildContext` - - Slab build context.

Returns: Nothing.

### _resetOptionalSlabArraysAfterSyncAllocate

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => void`

Resets optional slabs after sync allocation to keep omission semantics.

Parameters:
- `internalNet` - - Internal slab runtime shape.

Returns: Nothing.

### _resolveAsyncChunkSize

`(totalConnections: number, requestedChunkSize: number) => number`

Resolves effective async chunk size using adaptive heuristics.

Parameters:
- `totalConnections` - - Number of active connections.
- `requestedChunkSize` - - Requested chunk size.

Returns: Effective chunk size.

### _shouldSkipSlabRebuild

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, force: boolean) => boolean`

Determines whether slab rebuild can be skipped.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `force` - - True when rebuild must run regardless of dirty state.

Returns: True when rebuild can be skipped.

### _updatePlasticPresence

`(populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult, connection: import("src/architecture/network/slab/network.slab.utils.types").ConnectionInternals) => void`

Updates plastic-presence flag from connection bitfield.

Parameters:
- `populateResult` - - Mutable populate result.
- `connection` - - Connection internals.

Returns: Nothing.

### _weightArrayCtor

`(useFloat32Weights: boolean | undefined) => Float32ArrayConstructor | Float64ArrayConstructor`

Resolves typed-array constructor for weight slabs.

Parameters:
- `useFloat32Weights` - - True when 32-bit weights are enabled.

Returns: Matching typed-array constructor.

### _weightByteWidth

`(useFloat32Weights: boolean | undefined) => number`

Resolves byte width for weight slab arrays.

Parameters:
- `useFloat32Weights` - - True when 32-bit weights are enabled.

Returns: Byte width for weight elements.

### _writeConnectionCoreFields

`(writeArrays: import("src/architecture/network/slab/network.slab.utils.types").SlabWriteArrays, connection: import("src/architecture/network/slab/network.slab.utils.types").ConnectionInternals, connectionIndex: number) => void`

Writes core fields for one connection into slab arrays.

Parameters:
- `writeArrays` - - Core write arrays.
- `connection` - - Connection internals.
- `connectionIndex` - - Connection index.

Returns: Nothing.

### _writeConnectionGainField

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").SlabBuildContext, populateResult: import("src/architecture/network/slab/network.slab.utils.types").SlabPopulateResult, connection: import("src/architecture/network/slab/network.slab.utils.types").ConnectionInternals, connectionIndex: number) => void`

Writes gain field for one connection and updates gain flags.

Parameters:
- `buildContext` - - Slab build context.
- `populateResult` - - Mutable populate result.
- `connection` - - Connection internals.
- `connectionIndex` - - Connection index.

Returns: Nothing.

## architecture/network/slab/network.slab.adjacency.helpers.utils.ts

Internal slab adjacency helpers extracted from network.slab.utils.ts.

### _buildAdjacency

`(network: import("src/architecture/network").default) => void`

Build or refresh CSR-style adjacency (outStart + outOrder) for fast fan-out traversal.

Parameters:
- `network` - - Target network.

Returns: Nothing.

### asNetworkSlabProps

`(network: import("src/architecture/network").default) => import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps`

Cast network instance into internal slab-backed shape.

Parameters:
- `network` - - Target network.

Returns: Internal slab-backed network representation.

### buildOutgoingOrder

`(outgoingOrderBuildContext: import("src/architecture/network/slab/network.slab.utils.types").OutgoingOrderBuildContext) => Uint32Array<ArrayBufferLike>`

Build source-grouped outgoing order using CSR start offsets.

Parameters:
- `outgoingOrderBuildContext` - - Context holding build data and start offsets.

Returns: Ordered outgoing connection indices.

### buildOutgoingStartIndices

`(startIndicesBuildContext: import("src/architecture/network/slab/network.slab.utils.types").StartIndicesBuildContext) => Uint32Array<ArrayBufferLike>`

Build CSR start offsets from fan-out counts.

Parameters:
- `startIndicesBuildContext` - - Context holding build data and fan-out counts.

Returns: Outgoing start indices slab.

### collectFanOutCounts

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").BuildAdjacencyContext) => Uint32Array<ArrayBufferLike>`

Collect fan-out counts for each source node.

Parameters:
- `buildContext` - - Shared adjacency build context.

Returns: Fan-out counts per node.

### createBuildAdjacencyContext

`(network: import("src/architecture/network").default) => import("src/architecture/network/slab/network.slab.utils.types").BuildAdjacencyContext | null`

Build adjacency context when required slabs are available.

Parameters:
- `network` - - Target network.

Returns: Build context or null when adjacency cannot be built yet.

### createFanOutCollectionContext

`(buildContext: import("src/architecture/network/slab/network.slab.utils.types").BuildAdjacencyContext) => import("src/architecture/network/slab/network.slab.utils.types").FanOutCollectionContext`

Build fan-out collection context.

Parameters:
- `buildContext` - - Shared adjacency build context.

Returns: Fan-out collection context.

### createFanOutCountsBuffer

`(nodeCount: number) => Uint32Array<ArrayBufferLike>`

Allocate fan-out counts buffer.

Parameters:
- `nodeCount` - - Number of nodes.

Returns: Zero-initialized fan-out counts.

### createInsertionCursor

`(outgoingStartIndices: Uint32Array<ArrayBufferLike>) => Uint32Array<ArrayBufferLike>`

Create insertion cursor copy from outgoing start indices.

Parameters:
- `outgoingStartIndices` - - Outgoing start indices slab.

Returns: Mutable insertion cursor.

### createOutgoingOrderBuffer

`(connectionCount: number) => Uint32Array<ArrayBufferLike>`

Allocate outgoing order buffer.

Parameters:
- `connectionCount` - - Number of active connections.

Returns: Outgoing order buffer.

### createOutgoingStartIndicesBuffer

`(nodeCount: number) => Uint32Array<ArrayBufferLike>`

Allocate outgoing start indices buffer with terminal slot.

Parameters:
- `nodeCount` - - Number of nodes.

Returns: Outgoing start indices buffer.

### hasRequiredConnectionSlabs

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => boolean`

Check whether required connection slabs exist.

Parameters:
- `internalNet` - - Internal slab-backed network representation.

Returns: True when adjacency build prerequisites are present.

### incrementFanOutCountAtSource

`(context: { fanOutCounts: Uint32Array<ArrayBufferLike>; connectionFromSlab: Uint32Array<ArrayBufferLike>; connectionIndex: number; }) => void`

Increment fan-out count for one connection source index.

Parameters:
- `context` - - Increment context.

Returns: Nothing.

### insertConnectionIntoOutgoingOrder

`(context: { connectionIndex: number; connectionFromSlab: Uint32Array<ArrayBufferLike>; outgoingOrder: Uint32Array<ArrayBufferLike>; insertionCursor: Uint32Array<ArrayBufferLike>; }) => void`

Insert one connection index into the proper source-grouped slot.

Parameters:
- `context` - - Insertion context.

Returns: Nothing.

### iterateConnectionIndices

`(connectionCount: number, visitor: (connectionIndex: number) => void) => void`

Iterate all connection indices.

Parameters:
- `connectionCount` - - Number of active connections.
- `visitor` - - Index visitor.

Returns: Nothing.

### iterateNodeIndices

`(nodeCount: number, visitor: (nodeIndex: number) => void) => void`

Iterate all node indices.

Parameters:
- `nodeCount` - - Number of nodes.
- `visitor` - - Index visitor.

Returns: Nothing.

### populateFanOutCounts

`(fanOutCollectionContext: import("src/architecture/network/slab/network.slab.utils.types").FanOutCollectionContext) => void`

Populate fan-out counts from the connection source slab.

Parameters:
- `fanOutCollectionContext` - - Fan-out collection context.

Returns: Nothing.

### populateOutgoingOrder

`(context: { connectionCount: number; connectionFromSlab: Uint32Array<ArrayBufferLike>; outgoingOrder: Uint32Array<ArrayBufferLike>; insertionCursor: Uint32Array<ArrayBufferLike>; }) => void`

Populate outgoing order by source-grouped insertion.

Parameters:
- `context` - - Outgoing order population context.

Returns: Nothing.

### populateOutgoingStartIndices

`(context: { fanOutCounts: Uint32Array<ArrayBufferLike>; outgoingStartIndices: Uint32Array<ArrayBufferLike>; }) => number`

Populate outgoing start indices and return terminal offset.

Parameters:
- `context` - - Population context.

Returns: Terminal running offset after the last node.

### publishAdjacency

`(publishAdjacencyContext: import("src/architecture/network/slab/network.slab.utils.types").PublishAdjacencyContext) => void`

Publish adjacency slabs and clear dirty flag.

Parameters:
- `publishAdjacencyContext` - - Values to publish on the internal network slab state.

Returns: Nothing.

### setTerminalOutgoingStartOffset

`(context: { nodeCount: number; outgoingStartIndices: Uint32Array<ArrayBufferLike>; terminalRunningOffset: number; }) => void`

Set terminal outgoing start offset at the tail slot.

Parameters:
- `context` - - Terminal offset context.

Returns: Nothing.

## architecture/network/slab/network.slab.fast-path.helpers.utils.ts

Internal fast slab activation helpers extracted from network.slab.utils.ts.

### _activateThroughLegacyPath

`(network: import("src/architecture/network").default, input: number[]) => number[]`

Executes legacy network activation fallback.

Parameters:
- `network` - - Target network.
- `input` - - Activation input.

Returns: Legacy activation output.

### _canUseFastSlab

`(training: boolean) => boolean`

Predicate gating usage of high-performance slab forward pass.

Parameters:
- `training` - - Whether caller is in training mode.

Returns: True if fast path can be safely used.

### _collectFastSlabOutput

`(network: import("src/architecture/network").default, activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, nodeCount: number) => number[]`

Collects output activations into detached number array.

Parameters:
- `network` - - Target network.
- `activationBuffer` - - Activation buffer.
- `nodeCount` - - Node count.

Returns: Output activation array.

### _createFastActivationBuffer

`(useFloat32Activation: boolean, nodeCount: number) => Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>`

Creates typed fast activation/state buffer.

Parameters:
- `useFloat32Activation` - - True when 32-bit buffer is required.
- `nodeCount` - - Node count.

Returns: New typed buffer.

### _ensureFastSlabBuffers

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, nodeCount: number) => void`

Ensures fast activation/state buffers are allocated and shape-compatible.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `nodeCount` - - Node count.

Returns: Nothing.

### _hasFastSlabPrerequisites

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => boolean`

Checks whether core slab prerequisites are available.

Parameters:
- `internalNet` - - Internal slab runtime shape.

Returns: True when all required slabs/adjacency arrays exist.

### _maybeActivateNonInputNode

`(network: import("src/architecture/network").default, node: import("src/architecture/network/slab/network.slab.utils.types").FastSlabNodeRuntime, nodeIndex: number, stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>) => void`

Activates one non-input node when required.

Parameters:
- `network` - - Target network.
- `node` - - Current node.
- `nodeIndex` - - Node index.
- `stateBuffer` - - State buffer.
- `activationBuffer` - - Activation buffer.

Returns: Nothing.

### _needsFastBufferReplacement

`(buffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | undefined, nodeCount: number, useFloat32Activation: boolean) => boolean`

Checks whether a fast buffer requires replacement.

Parameters:
- `buffer` - - Existing buffer.
- `nodeCount` - - Node count.
- `useFloat32Activation` - - True when 32-bit buffer is required.

Returns: True when replacement is needed.

### _prepareFastSlabRuntime

`(network: import("src/architecture/network").default, internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, reindexNodes: (network: import("src/architecture/network").default) => void) => void`

Prepares topology and indices for fast slab pass.

Parameters:
- `network` - - Target network.
- `internalNet` - - Internal slab runtime shape.
- `reindexNodes` - - Callback used to reindex nodes when needed.

Returns: Nothing.

### _propagateFastSlabActivations

`(network: import("src/architecture/network").default, internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, topoOrder: import("src/architecture/network/slab/network.slab.utils.types").FastSlabNodeRuntime[], activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>) => void`

Propagates activations through topology using slab arrays.

Parameters:
- `network` - - Target network.
- `internalNet` - - Internal slab runtime shape.
- `topoOrder` - - Topological node order.
- `activationBuffer` - - Activation buffer.
- `stateBuffer` - - State buffer.

Returns: Nothing.

### _propagateNodeOutgoingEdges

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, nodeIndex: number, activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, stateBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, weightArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, toIndexArray: Uint32Array<ArrayBufferLike>, outgoingOrder: Uint32Array<ArrayBufferLike>, outgoingStartIndices: Uint32Array<ArrayBufferLike>) => void`

Propagates one node activation over all outgoing slab edges.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `nodeIndex` - - Source node index.
- `activationBuffer` - - Activation buffer.
- `stateBuffer` - - State buffer.
- `weightArray` - - Weight slab.
- `toIndexArray` - - Destination-index slab.
- `outgoingOrder` - - Outgoing edge order slab.
- `outgoingStartIndices` - - Outgoing start-offset slab.

Returns: Nothing.

### _recomputeTopologyOrder

`(network: import("src/architecture/network").default) => void`

Recomputes topological order on demand.

Parameters:
- `network` - - Target network.

Returns: Nothing.

### _resolveFastTopoOrder

`(network: import("src/architecture/network").default, internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps) => import("src/architecture/network/slab/network.slab.utils.types").FastSlabNodeRuntime[]`

Resolves topological iteration order for fast slab pass.

Parameters:
- `network` - - Target network.
- `internalNet` - - Internal slab runtime shape.

Returns: Topological node order.

### _resolveWeightedConnectionValue

`(internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, weightArray: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>, connectionIndex: number) => number`

Resolves effective connection weight including optional gain.

Parameters:
- `internalNet` - - Internal slab runtime shape.
- `weightArray` - - Weight slab.
- `connectionIndex` - - Connection index.

Returns: Effective weighted value.

### _seedFastInputLayer

`(network: import("src/architecture/network").default, input: number[], activationBuffer: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>) => void`

Seeds input-layer activations for fast slab pass.

Parameters:
- `network` - - Target network.
- `input` - - Activation input.
- `activationBuffer` - - Activation buffer.

Returns: Nothing.

### _tryFastSlabFallbackForGating

`(network: import("src/architecture/network").default, input: number[]) => number[] | null`

Falls back to legacy activation when gating is present.

Parameters:
- `network` - - Target network.
- `input` - - Activation input.

Returns: Legacy output or null when fast path may continue.

### _tryFastSlabFallbackForMissingPrerequisites

`(network: import("src/architecture/network").default, internalNet: import("src/architecture/network/slab/network.slab.utils.types").NetworkSlabProps, input: number[]) => number[] | null`

Falls back to legacy activation when slab prerequisites are missing.

Parameters:
- `network` - - Target network.
- `internalNet` - - Internal slab runtime shape.
- `input` - - Activation input.

Returns: Legacy output or null when fast path may continue.

### _writeInputNodeRuntime

`(node: import("src/architecture/node").default, inputValue: number) => void`

Writes runtime activation/state for one input node.

Parameters:
- `node` - - Input node.
- `inputValue` - - Input activation value.

Returns: Nothing.
