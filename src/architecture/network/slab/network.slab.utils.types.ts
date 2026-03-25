import type Network from '../../network/network';
import type Connection from '../../connection';
import type Node from '../../node';

/**
 * Numeric zero sentinel used across slab orchestration and helper pipelines.
 */
export const SLAB_ZERO = 0;

/**
 * Numeric one sentinel used for neutral gain defaults and index math.
 */
export const SLAB_ONE = 1;

/**
 * Capacity growth factor for Node.js slab allocations.
 */
export const SLAB_GROWTH_FACTOR_NODE = 1.75;

/**
 * Capacity growth factor for browser slab allocations.
 */
export const SLAB_GROWTH_FACTOR_BROWSER = 1.25;

/**
 * Default async slab rebuild chunk size when no override is provided.
 */
export const SLAB_DEFAULT_ASYNC_CHUNK_SIZE = 50_000;

/**
 * Internal Connection properties accessed during slab operations.
 */
export interface ConnectionInternals {
  _flags: number;
  from: Node & { index: number };
  to: Node & { index: number };
  weight: number;
}

/**
 * Internal Network properties for slab operations.
 */
export interface NetworkSlabProps {
  _slabDirty?: boolean;
  _nodeIndexDirty?: boolean;
  _connCapacity?: number;
  _useFloat32Weights?: boolean;
  _connWeights?: Float32Array | Float64Array;
  _connFrom?: Uint32Array;
  _connTo?: Uint32Array;
  _connFlags?: Uint8Array;
  _connGain?: Float32Array | Float64Array | null;
  _connPlastic?: Float32Array | Float64Array | null;
  _connUsed?: number;
  _connVersion?: number;
  _outStart?: Uint32Array | null;
  _outOrder?: Uint32Array | null;
  _connCount?: number;
  _adjDirty?: boolean;
  _slabVersion?: number;
  _slabAsyncBuilds?: number;
  _activationPrecision?: 'f32' | 'f64';
  _enforceAcyclic?: boolean;
  _fastA?: Float32Array | Float64Array;
  _fastS?: Float32Array | Float64Array;
  _topoDirty?: boolean;
  _topoOrder?: unknown[];
  _weightNoiseStd?: number;
  _weightNoisePerHidden?: unknown[];
  _stochasticDepth?: unknown[];
  connections: (Connection & ConnectionInternals)[];
}

/**
 * Per-pool-key allocation & reuse counters (educational / diagnostics).
 */
export interface PoolKeyMetrics {
  created: number;
  reused: number;
  maxRetained: number;
}

/**
 * Union of slab typed array element container types.
 */
export type TypedArray = Float32Array | Float64Array | Uint32Array | Uint8Array;

/**
 * Constructor type for typed arrays used in slabs.
 */
export type TypedArrayConstructor =
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Uint32ArrayConstructor
  | Uint8ArrayConstructor;

/**
 * Runtime activation contract used by slab-based execution paths.
 */
export interface NetworkActivationRuntime {
  activate(input: number[], training: boolean): number[];
}

/**
 * Runtime topology contract used to lazily rebuild topological order.
 */
export interface NetworkTopoRuntime {
  _computeTopoOrder(): void;
}

/**
 * Node shape required by fast slab activation kernels.
 */
export interface FastSlabNodeRuntime extends Node {
  index: number;
  bias: number;
  squash(x: number): number;
  activation: number;
  state: number;
}

/**
 * Immutable inputs required to build or grow connection slab buffers.
 */
export interface SlabBuildContext {
  network: Network;
  internalNet: NetworkSlabProps;
  connectionCount: number;
  capacity: number;
  growthFactor: number;
  weightBytes: number;
  weightCtor: Float32ArrayConstructor | Float64ArrayConstructor;
}

/**
 * Result of scanning and populating optional gain/plastic slab arrays.
 */
export interface SlabPopulateResult {
  anyNonNeutralGain: boolean;
  anyPlastic: boolean;
  gainArray: Float32Array | Float64Array | null;
  plasticArray: Float32Array | Float64Array | null;
}

/**
 * Writable slab arrays targeted during connection serialization.
 */
export interface SlabWriteArrays {
  weightArray: Float32Array | Float64Array;
  fromIndexArray: Uint32Array;
  toIndexArray: Uint32Array;
  flagArray: Uint8Array;
}

/**
 * Shape returned by getConnectionSlab describing the packed SoA view.
 */
export interface ConnectionSlabView {
  weights: Float32Array | Float64Array;
  from: Uint32Array;
  to: Uint32Array;
  flags: Uint8Array;
  gain: Float32Array | Float64Array | null;
  plastic: Float32Array | Float64Array | null;
  version: number;
  used: number;
  capacity: number;
}

/**
 * Shared immutable inputs used across the adjacency build pipeline.
 */
export type BuildAdjacencyContext = {
  internalNet: NetworkSlabProps;
  nodeCount: number;
  connectionCount: number;
  connectionFromSlab: Uint32Array;
};

/**
 * Context for fan-out collection: build inputs plus the output count buffer.
 */
export type FanOutCollectionContext = {
  buildContext: BuildAdjacencyContext;
  fanOutCounts: Uint32Array;
};

/**
 * Context for constructing CSR start offsets from precomputed fan-out counts.
 */
export type StartIndicesBuildContext = {
  buildContext: BuildAdjacencyContext;
  fanOutCounts: Uint32Array;
};

/**
 * Context for constructing source-grouped outgoing connection order.
 */
export type OutgoingOrderBuildContext = {
  buildContext: BuildAdjacencyContext;
  outgoingStartIndices: Uint32Array;
};

/**
 * Context for publishing fully built adjacency slabs to internal network state.
 */
export type PublishAdjacencyContext = {
  internalNet: NetworkSlabProps;
  outgoingStartIndices: Uint32Array;
  outgoingOrder: Uint32Array;
};
